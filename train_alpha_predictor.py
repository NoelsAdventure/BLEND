"""Train the FriendlyPredictor.

Compared to the previous version:
  - Episode-level train/val split (no within-episode leakage).
  - Richer per-human features: observed pos/vel in robot frame, distance,
    approach rate, radius — in addition to pred_traj + uncertainty.
  - Padding mask: BCE loss is computed only on real human slots, so padded
    slots don't bias the network toward predicting 'not friendly'.
  - BCEWithLogitsLoss + cosine LR schedule + weight decay.

Mirror any feature change in rl/evaluation.py::_compute_pred_friendly_probs
(via the shared build_human_features / build_robot_features helpers below).
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from alpha_predictor import FriendlyPredictor


# Per-step uncertainty horizon, matches crowd_sim/envs/utils/human.py::pred_horizon_aci.
NUM_PRED_STEPS = 5
# Number of (dx, dy) entries in pred_traj = (predict_steps+1) * 2.
# CrowdSimPredRealGST emits 6 future positions per human.
PRED_TRAJ_LEN = 12

# Feature widths exposed so evaluation.py can match them exactly.
EXTRA_HUMAN_FEATURES = 7  # rel_x, rel_y, hvx, hvy, dist, approach_rate, radius
HUMAN_FEATURE_DIM = PRED_TRAJ_LEN + NUM_PRED_STEPS + EXTRA_HUMAN_FEATURES  # 12 + 5 + 7 = 24
ROBOT_FEATURE_DIM = 9


def build_robot_feature_vector(rx, ry, rvx, rvy, radius, gx, gy, v_pref, theta):
    """Matches the RL policy's robot input layout."""
    return [
        float(rvx), float(rvy),
        float(rx), float(ry),
        float(radius),
        float(gx), float(gy),
        float(v_pref),
        float(theta),
    ]


def build_human_feature_row(pred_traj, uncertainty_row,
                            hx, hy, hvx, hvy, h_radius,
                            rx, ry, rvx, rvy):
    """Concatenate pred_traj + uncertainty + observed-human features.

    Observed features are translation-invariant relative to the robot.
    pred_traj is kept in the env's native frame (offset-from-robot, world
    aligned) since the GST predictor already publishes it that way.
    """
    traj = [float(x) for x in list(pred_traj)]
    if len(traj) < PRED_TRAJ_LEN:
        traj = traj + [0.0] * (PRED_TRAJ_LEN - len(traj))
    elif len(traj) > PRED_TRAJ_LEN:
        traj = traj[:PRED_TRAJ_LEN]

    u_row = [float(x) for x in list(uncertainty_row)]
    if len(u_row) < NUM_PRED_STEPS:
        u_row = u_row + [0.0] * (NUM_PRED_STEPS - len(u_row))
    elif len(u_row) > NUM_PRED_STEPS:
        u_row = u_row[:NUM_PRED_STEPS]

    rel_x = float(hx) - float(rx)
    rel_y = float(hy) - float(ry)
    dist = math.sqrt(rel_x * rel_x + rel_y * rel_y)

    # Radial closing speed: positive = closing, negative = separating.
    dvx = float(hvx) - float(rvx)
    dvy = float(hvy) - float(rvy)
    if dist > 1e-6:
        approach_rate = -(rel_x * dvx + rel_y * dvy) / dist
    else:
        approach_rate = 0.0

    extras = [
        rel_x, rel_y,
        float(hvx), float(hvy),
        dist,
        float(approach_rate),
        float(h_radius),
    ]

    return traj + u_row + extras


def calculate_classification_metrics(targets, preds, mask=None):
    """All inputs are flat numpy arrays. mask=None means use everything."""
    if mask is not None:
        targets = targets[mask]
        preds = preds[mask]
    preds_binary = (preds > 0.5).astype(np.int8)
    targets = targets.astype(np.int8)

    tp = int(np.sum((targets == 1) & (preds_binary == 1)))
    fp = int(np.sum((targets == 0) & (preds_binary == 1)))
    tn = int(np.sum((targets == 0) & (preds_binary == 0)))
    fn = int(np.sum((targets == 1) & (preds_binary == 0)))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1, tp, fp, tn, fn


class FriendlyDataset(Dataset):
    """One sample = one simulator step.

    Each sample carries:
      - human_inputs: (max_humans, HUMAN_FEATURE_DIM)
      - robot_inputs: (ROBOT_FEATURE_DIM,)
      - targets:      (max_humans,)        binary {0, 1}
      - valid_mask:   (max_humans,) bool   True for real humans
    """

    def __init__(self, json_paths, max_humans=20, episode_filter=None):
        self.human_inputs = []
        self.robot_inputs = []
        self.targets = []
        self.valid_masks = []
        self.max_humans = max_humans
        self._warned_missing_v_pref = False

        if isinstance(json_paths, str):
            json_paths = [json_paths]

        for json_path in json_paths:
            print(f"Loading data from {json_path}...")
            if not os.path.exists(json_path):
                print(f"  --> Skip: Dataset not found: {json_path}")
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for ep_idx, ep in enumerate(data['episodes']):
                # episode_filter is a (path, ep_idx) → bool callable for
                # episode-level train/val splits.
                if episode_filter is not None and not episode_filter(json_path, ep_idx):
                    continue
                steps = ep.get('steps_data', ep.get('steps', []))
                for step in steps:
                    r = step['robot']
                    rx, ry = r['pos'][0], r['pos'][1]
                    rvx, rvy = r['vel'][0], r['vel'][1]
                    if 'v_pref' in r:
                        v_pref = float(r['v_pref'])
                    else:
                        if not self._warned_missing_v_pref:
                            print("  --> Legacy dump missing 'v_pref'; defaulting to 1.0. "
                                  "Regenerate dumps for clean training data.")
                            self._warned_missing_v_pref = True
                        v_pref = 1.0
                    r_state = build_robot_feature_vector(
                        rx, ry, rvx, rvy, r['radius'],
                        r['goal'][0], r['goal'][1], v_pref, r['theta'],
                    )

                    pred_trajs = step.get('pred_traj', [])
                    uncertainties = step.get('uncertainty', [])

                    humans = step['humans']

                    def get_dist(h, rx=rx, ry=ry):
                        return np.linalg.norm(np.array(h['pos']) - np.array([rx, ry]))

                    sorted_humans = sorted(humans, key=get_dist)

                    h_states = []
                    h_labels = []
                    h_valid = []
                    for i in range(max_humans):
                        if i < len(pred_trajs) and i < len(sorted_humans):
                            traj = pred_trajs[i]
                            u_raw = uncertainties[i] if uncertainties and i < len(uncertainties) else None
                            if u_raw is None:
                                u_row = [0.0] * NUM_PRED_STEPS
                            elif isinstance(u_raw, (list, tuple, np.ndarray)):
                                u_row = [float(x) for x in list(u_raw)]
                            else:
                                u_row = [float(u_raw)] * NUM_PRED_STEPS

                            h = sorted_humans[i]
                            h_state = build_human_feature_row(
                                traj, u_row,
                                h['pos'][0], h['pos'][1],
                                h['vel'][0], h['vel'][1],
                                h.get('radius', 0.3),
                                rx, ry, rvx, rvy,
                            )

                            if 'actual_friendly' in h:
                                h_label = float(h['actual_friendly'])
                            else:
                                h_label = float(h.get('is_friendly', False))
                            h_valid.append(True)
                        else:
                            h_state = [0.0] * HUMAN_FEATURE_DIM
                            h_label = 0.0
                            h_valid.append(False)

                        h_states.append(h_state)
                        h_labels.append(h_label)

                    self.human_inputs.append(h_states)
                    self.robot_inputs.append(r_state)
                    self.targets.append(h_labels)
                    self.valid_masks.append(h_valid)

        if len(self.targets) == 0:
            self.human_inputs = torch.empty((0, max_humans, HUMAN_FEATURE_DIM), dtype=torch.float32)
            self.robot_inputs = torch.empty((0, ROBOT_FEATURE_DIM), dtype=torch.float32)
            self.targets = torch.empty((0, max_humans), dtype=torch.float32)
            self.valid_masks = torch.empty((0, max_humans), dtype=torch.bool)
        else:
            self.human_inputs = torch.tensor(np.array(self.human_inputs), dtype=torch.float32)
            self.robot_inputs = torch.tensor(np.array(self.robot_inputs), dtype=torch.float32)
            self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
            self.valid_masks = torch.tensor(np.array(self.valid_masks), dtype=torch.bool)

        print(f"  --> Total steps loaded: {len(self.targets)}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.human_inputs[idx], self.robot_inputs[idx],
                self.targets[idx], self.valid_masks[idx])


def _build_episode_split(json_paths, val_frac=0.2, seed=42):
    """Return two episode_filter callables (train, val) that split by episode.

    Episode is keyed by (json_path, episode_index) so a step never appears in
    both splits.
    """
    rng = np.random.default_rng(seed)
    episode_ids = []
    for path in json_paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for ep_idx in range(len(data['episodes'])):
            episode_ids.append((path, ep_idx))

    rng.shuffle(episode_ids)
    n_val = int(round(val_frac * len(episode_ids)))
    val_set = set(episode_ids[:n_val])
    train_set = set(episode_ids[n_val:])

    def in_train(path, ep_idx):
        return (path, ep_idx) in train_set

    def in_val(path, ep_idx):
        return (path, ep_idx) in val_set

    return in_train, in_val, len(train_set), len(val_set)


def evaluate_model(model, dataloader, device, label="Evaluation"):
    model.eval()
    preds_chunks, targets_chunks, masks_chunks = [], [], []

    with torch.no_grad():
        for b_human, b_robot, b_target, b_valid in dataloader:
            b_human = b_human.to(device)
            b_robot = b_robot.to(device)
            key_padding_mask = ~b_valid.to(device)  # True = ignore
            logits = model(b_human, b_robot, key_padding_mask=key_padding_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds_chunks.append(probs)
            targets_chunks.append(b_target.numpy())
            masks_chunks.append(b_valid.numpy())

    preds = np.concatenate(preds_chunks, axis=0).flatten() if preds_chunks else np.array([])
    targets = np.concatenate(targets_chunks, axis=0).flatten() if targets_chunks else np.array([])
    masks = np.concatenate(masks_chunks, axis=0).flatten() if masks_chunks else np.array([], dtype=bool)

    acc, prec, rec, f1, tp, fp, tn, fn = calculate_classification_metrics(targets, preds, mask=masks)
    total = tp + fp + tn + fn

    def _pct(x):
        return (100.0 * x / total) if total > 0 else 0.0

    print(f"\n[{label}]")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(
        f"TP: {tp} ({_pct(tp):.1f}%) | "
        f"FP: {fp} ({_pct(fp):.1f}%) | "
        f"TN: {tn} ({_pct(tn):.1f}%) | "
        f"FN: {fn} ({_pct(fn):.1f}%)"
    )
    return acc, f1, prec, rec, tp, fp, tn, fn


def train():
    model_dir = 'trained_models/LoraF_invi_visi_rank_1'

    # Train on ALL four scenarios with balanced per-scenario sampling.
    # Mixed 50/50 forces real per-human discrimination; the three degenerate
    # scenarios (all-aware / all-ignorant / step-25-flip) teach the model to
    # collapse to a population-level mode via the transformer's self-attention
    # across humans. We sample so each scenario contributes roughly equally
    # per epoch, then track per-scenario val accuracy and save the checkpoint
    # whose WORST scenario is best (max-min-accuracy).
    #
    # Each scenario uses the base dump + _exp1 variant (different seed) when
    # available, to widen the per-scenario distribution.
    scenarios_spec = {
        'mixed_5050': [
            'seperate_mixed_5050_adaptive_gt.json',
            'seperate_mixed_5050_adaptive_gt_exp1.json',
        ],
        'all_aware': [
            'seperate_all_aware_adaptive_gt.json',
            'seperate_all_aware_adaptive_gt_exp1.json',
        ],
        'all_ignorant': [
            'seperate_all_ignorant_adaptive_gt.json',
            'seperate_all_ignorant_adaptive_gt_exp1.json',
        ],
        'switch_step25': [
            'seperate_ignorant_to_aware_step25_adaptive_gt.json',
            'seperate_ignorant_to_aware_step25_adaptive_gt_exp1.json',
        ],
    }

    per_scenario_train = {}
    per_scenario_val = {}
    for sc_name, files in scenarios_spec.items():
        paths = [os.path.join(model_dir, 'test', f) for f in files]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            print(f"  --> Skip scenario '{sc_name}': no JSONs found.")
            continue
        in_train, in_val, n_train_ep, n_val_ep = _build_episode_split(
            paths, val_frac=0.2, seed=42,
        )
        print(f"[{sc_name}] episode split: {n_train_ep} train / {n_val_ep} val "
              f"from {len(paths)} file(s)")
        per_scenario_train[sc_name] = FriendlyDataset(paths, episode_filter=in_train)
        per_scenario_val[sc_name] = FriendlyDataset(paths, episode_filter=in_val)

    if not per_scenario_train:
        print("Error: No training data loaded for any scenario.")
        return

    # Concatenate train sets, sampling-weighted so Mixed 50/50 dominates the
    # loss while the three degenerate scenarios still get represented enough
    # for the model to learn the population-mode shortcut. With balanced
    # sampling (each scenario at 25%) the degenerate scenarios saturate fast
    # but Mixed crawls because per-human discrimination is the hard task; we
    # give Mixed 4x the per-scenario weight to refocus gradient there.
    SCENARIO_WEIGHTS = {
        'mixed_5050': 4.0,
        'all_aware': 1.0,
        'all_ignorant': 1.0,
        'switch_step25': 1.0,
    }
    from torch.utils.data import ConcatDataset, WeightedRandomSampler
    train_concat = ConcatDataset(list(per_scenario_train.values()))
    weights = []
    for sc_name, ds in per_scenario_train.items():
        scenario_w = SCENARIO_WEIGHTS.get(sc_name, 1.0)
        per_sample = scenario_w / max(len(ds), 1)
        weights.extend([per_sample] * len(ds))
    weights_t = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights_t,
        num_samples=len(train_concat),
        replacement=True,
    )
    sw_str = ", ".join(f"{sc}={SCENARIO_WEIGHTS.get(sc, 1.0):.1f}"
                       for sc in per_scenario_train.keys())
    print(f"Train concat size: {len(train_concat)} (scenario weights: {sw_str})")

    train_loader = DataLoader(
        train_concat, batch_size=128, sampler=sampler,
        num_workers=2, drop_last=False,
    )
    per_scenario_val_loaders = {
        sc: DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
        for sc, ds in per_scenario_val.items()
    }

    sample0_human = next(iter(per_scenario_train.values())).human_inputs.shape[-1]
    sample0_robot = next(iter(per_scenario_train.values())).robot_inputs.shape[-1]
    assert sample0_human == HUMAN_FEATURE_DIM, (sample0_human, HUMAN_FEATURE_DIM)
    assert sample0_robot == ROBOT_FEATURE_DIM, (sample0_robot, ROBOT_FEATURE_DIM)
    human_dim, robot_dim = sample0_human, sample0_robot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim, num_heads, num_layers, dropout = 192, 4, 3, 0.1
    model = FriendlyPredictor(
        human_dim=human_dim, robot_dim=robot_dim,
        hidden_dim=hidden_dim, num_heads=num_heads,
        num_layers=num_layers, dropout=dropout,
    ).to(device)
    print(f"Feature dims: robot={robot_dim}, human={human_dim}. "
          f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    bce = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 60
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_min_acc = -1.0
    checkpoint_path = os.path.join(model_dir, 'friendly_predictor.pth')
    metrics_path = os.path.join(model_dir, 'friendly_predictor_metrics.json')
    history_path = os.path.join(model_dir, 'friendly_predictor_history.json')
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for b_human, b_robot, b_target, b_valid in train_loader:
            b_human = b_human.to(device)
            b_robot = b_robot.to(device)
            b_target = b_target.to(device)
            b_valid_dev = b_valid.to(device)
            key_padding_mask = ~b_valid_dev

            optimizer.zero_grad()
            logits = model(b_human, b_robot, key_padding_mask=key_padding_mask)
            loss_per_slot = bce(logits, b_target)
            mask_f = b_valid_dev.float()
            denom = mask_f.sum().clamp(min=1.0)
            loss = (loss_per_slot * mask_f).sum() / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * denom.item()
            running_count += denom.item()

        scheduler.step()
        train_loss = running_loss / max(running_count, 1.0)

        # Per-scenario validation table.
        per_sc = {}
        for sc, loader in per_scenario_val_loaders.items():
            acc, f1, prec, rec, tp, fp, tn, fn = evaluate_model(
                model, loader, device,
                label=f"Epoch {epoch+1} [{sc}] (loss={train_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e})",
            )
            per_sc[sc] = {
                'acc': float(acc), 'f1': float(f1),
                'prec': float(prec), 'rec': float(rec),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
                'size': int(len(per_scenario_val[sc])),
            }

        min_acc = min(s['acc'] for s in per_sc.values())
        mean_acc = float(np.mean([s['acc'] for s in per_sc.values()]))
        # Compact one-line table for monitor parsing.
        table = " | ".join(f"{sc}={per_sc[sc]['acc']:.3f}" for sc in per_sc)
        print(f"==> Epoch {epoch+1} per-scenario acc: {table} || min={min_acc:.3f} mean={mean_acc:.3f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'lr': float(scheduler.get_last_lr()[0]),
            'min_scenario_acc': float(min_acc),
            'mean_scenario_acc': float(mean_acc),
            'per_scenario': {sc: dict(per_sc[sc]) for sc in per_sc},
        })
        try:
            with open(history_path, 'w') as hf:
                json.dump({
                    'scenarios': list(scenarios_spec.keys()),
                    'scenario_weights': {k: float(v) for k, v in SCENARIO_WEIGHTS.items()},
                    'epochs_planned': int(epochs),
                    'history': history,
                }, hf, indent=2)
        except Exception as e:
            print(f"[WARN] Could not write history to {history_path}: {e}")

        if min_acc > best_min_acc:
            best_min_acc = min_acc
            torch.save(model.state_dict(), checkpoint_path)
            with open(metrics_path, 'w') as mf:
                json.dump({
                    'epoch': epoch + 1,
                    'min_scenario_acc': float(min_acc),
                    'mean_scenario_acc': float(mean_acc),
                    'per_scenario': per_sc,
                    'threshold': 0.5,
                    'human_dim': int(human_dim),
                    'robot_dim': int(robot_dim),
                    'max_humans': int(next(iter(per_scenario_train.values())).max_humans),
                    'feature_layout': {
                        'pred_traj_len': PRED_TRAJ_LEN,
                        'num_pred_steps': NUM_PRED_STEPS,
                        'extra_human_features': EXTRA_HUMAN_FEATURES,
                        'extra_human_feature_order': [
                            'rel_x', 'rel_y', 'hvx', 'hvy',
                            'dist', 'approach_rate', 'h_radius',
                        ],
                    },
                    'architecture': {
                        'kind': 'transformer',
                        'hidden_dim': hidden_dim,
                        'num_heads': num_heads,
                        'num_layers': num_layers,
                        'dropout': dropout,
                    },
                    # Back-compat fields for downstream code expecting these.
                    'val_accuracy': float(mean_acc),
                    'val_f1': float(np.mean([s['f1'] for s in per_sc.values()])),
                    'val_precision': float(np.mean([s['prec'] for s in per_sc.values()])),
                    'val_recall': float(np.mean([s['rec'] for s in per_sc.values()])),
                    'val_tp': int(sum(s['tp'] for s in per_sc.values())),
                    'val_fp': int(sum(s['fp'] for s in per_sc.values())),
                    'val_tn': int(sum(s['tn'] for s in per_sc.values())),
                    'val_fn': int(sum(s['fn'] for s in per_sc.values())),
                    'val_size': int(sum(s['size'] for s in per_sc.values())),
                }, mf, indent=2)
            print(f"  --> Best Model Saved (min-scenario-acc: {min_acc:.4f})")

    print(f"\nTraining Complete. Best model saved to {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("\n" + "=" * 40)
    print("SCENARIO GENERALIZATION TESTING")
    print("=" * 40)

    scenarios_to_test = {
        "Mixed 50/50": 'seperate_mixed_5050_adaptive_gt.json',
        "All Aware (Friendly)": 'seperate_all_aware_adaptive_gt.json',
        "All Ignorant": 'seperate_all_ignorant_adaptive_gt.json',
        "Switching (Step 25)": 'seperate_ignorant_to_aware_step25_adaptive_gt.json',
    }
    for name, filename in scenarios_to_test.items():
        try:
            test_json = os.path.join(model_dir, 'test', filename)
            test_dataset = FriendlyDataset(test_json)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
            evaluate_model(model, test_loader, device, label=f"Testing ({name})")
        except Exception as e:
            print(f"\n[{name}] Skip: {e}")

    print("\n" + "=" * 40)


if __name__ == '__main__':
    train()
