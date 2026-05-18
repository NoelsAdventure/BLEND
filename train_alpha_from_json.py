"""Standalone trainer for the FriendlyPredictor.

Accepts arbitrary lists of JSON dump paths per scenario, with per-scenario
sampling weights. Usable two ways:

  1. As a library function — `train_from_jsons(scenarios_spec, ...)`. This is
     the entry point dagger_loop.py calls each iteration.

  2. As a CLI — `python train_alpha_from_json.py --config cfg.json`, where
     cfg.json contains the same keys as the function args.

Reuses FriendlyDataset / build_human_feature_row / evaluate_model / the
episode-level split from train_alpha_predictor.py, so the feature layout,
architecture, and val protocol are exactly the same as the main trainer.
"""

import argparse
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from alpha_predictor import FriendlyPredictor
from train_alpha_predictor import (
    EXTRA_HUMAN_FEATURES,
    HUMAN_FEATURE_DIM,
    NUM_PRED_STEPS,
    PRED_TRAJ_LEN,
    ROBOT_FEATURE_DIM,
    FriendlyDataset,
    _build_episode_split,
    evaluate_model,
)


def train_from_jsons(
    scenarios_spec: Dict[str, List[str]],
    weights_spec: Dict[str, float],
    output_path: str,
    metrics_path: str,
    epochs: int = 60,
    init_checkpoint: Optional[str] = None,
    hidden_dim: int = 192,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    val_frac: float = 0.2,
    seed: int = 42,
    log_fn: Callable[[str], None] = print,
) -> Dict:
    """Train the FriendlyPredictor on the given JSON dumps.

    scenarios_spec: scenario_name -> list of JSON paths. All files in a list
        are treated as the same scenario (e.g. base + _exp1 + dagger iter
        dumps). Episode-level split is computed across all files in the list.
    weights_spec: scenario_name -> sampling weight. Defaults to 1.0 for any
        scenario without an entry.
    output_path: where to save the best .pth (overwritten each epoch when min-
        scenario-acc improves).
    metrics_path: where to save the metrics sidecar JSON.
    init_checkpoint: optional .pth to warm-start the weights. Useful for
        DAgger iterations so each pass refines the prior model.

    Returns the best metrics dict (also written to metrics_path).
    """

    # --- Per-scenario datasets with episode-level split ------------------
    per_scenario_train: Dict[str, FriendlyDataset] = {}
    per_scenario_val: Dict[str, FriendlyDataset] = {}
    for sc_name, files in scenarios_spec.items():
        paths = [p for p in files if os.path.exists(p)]
        if not paths:
            log_fn(f"[WARN] Scenario '{sc_name}': no JSONs found. Skipping.")
            continue
        in_train, in_val, n_train_ep, n_val_ep = _build_episode_split(
            paths, val_frac=val_frac, seed=seed,
        )
        log_fn(f"[{sc_name}] split: {n_train_ep} train / {n_val_ep} val "
               f"from {len(paths)} file(s)")
        per_scenario_train[sc_name] = FriendlyDataset(paths, episode_filter=in_train)
        per_scenario_val[sc_name] = FriendlyDataset(paths, episode_filter=in_val)

    if not per_scenario_train:
        raise RuntimeError("No training data loaded — every scenarios_spec list was empty.")

    # --- Weighted sampler ------------------------------------------------
    train_concat = ConcatDataset(list(per_scenario_train.values()))
    weights: List[float] = []
    for sc_name, ds in per_scenario_train.items():
        scenario_w = float(weights_spec.get(sc_name, 1.0))
        per_sample = scenario_w / max(len(ds), 1)
        weights.extend([per_sample] * len(ds))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_concat),
        replacement=True,
    )
    sw_str = ", ".join(f"{sc}={float(weights_spec.get(sc, 1.0)):.1f}"
                       for sc in per_scenario_train.keys())
    log_fn(f"Train concat size: {len(train_concat)} (scenario weights: {sw_str})")

    train_loader = DataLoader(
        train_concat, batch_size=batch_size, sampler=sampler,
        num_workers=2, drop_last=False,
    )
    per_scenario_val_loaders = {
        sc: DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
        for sc, ds in per_scenario_val.items()
    }

    # --- Model -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FriendlyPredictor(
        human_dim=HUMAN_FEATURE_DIM, robot_dim=ROBOT_FEATURE_DIM,
        hidden_dim=hidden_dim, num_heads=num_heads,
        num_layers=num_layers, dropout=dropout,
    ).to(device)
    if init_checkpoint and os.path.exists(init_checkpoint):
        try:
            state = torch.load(init_checkpoint, map_location=device)
            model.load_state_dict(state)
            log_fn(f"Resumed from {init_checkpoint}")
        except Exception as e:
            log_fn(f"[WARN] Could not load {init_checkpoint}: {e}. Training from scratch.")
    log_fn(f"Feature dims: robot={ROBOT_FEATURE_DIM}, human={HUMAN_FEATURE_DIM}. "
           f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    bce = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_min_acc = -1.0
    best_metrics: Dict = {}

    # Per-epoch history for plot_awareness_training.py. Saved alongside the
    # metrics sidecar as friendly_predictor_history.json (same dir / base
    # name, suffix swapped) and flushed every epoch so partial training runs
    # still produce a plottable file.
    history_path = (
        metrics_path.replace('_metrics', '_history')
        if '_metrics' in metrics_path
        else metrics_path.replace('.json', '_history.json')
    )
    history: List[Dict] = []

    for epoch in range(epochs):
        model.train()
        running_loss, running_count = 0.0, 0.0
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

        # Per-scenario val
        per_sc: Dict[str, Dict] = {}
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
        table = " | ".join(f"{sc}={per_sc[sc]['acc']:.3f}" for sc in per_sc)
        log_fn(f"==> Epoch {epoch+1} per-scenario acc: {table} || "
               f"min={min_acc:.3f} mean={mean_acc:.3f}")

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
                    'scenario_weights': {k: float(v) for k, v in weights_spec.items()},
                    'epochs_planned': int(epochs),
                    'init_checkpoint': init_checkpoint,
                    'history': history,
                }, hf, indent=2)
        except Exception as e:
            log_fn(f"[WARN] Could not write history to {history_path}: {e}")

        if min_acc > best_min_acc:
            best_min_acc = min_acc
            torch.save(model.state_dict(), output_path)
            best_metrics = {
                'epoch': epoch + 1,
                'min_scenario_acc': float(min_acc),
                'mean_scenario_acc': float(mean_acc),
                'per_scenario': per_sc,
                'threshold': 0.5,
                'human_dim': int(HUMAN_FEATURE_DIM),
                'robot_dim': int(ROBOT_FEATURE_DIM),
                'max_humans': 20,
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
                    'hidden_dim': int(hidden_dim),
                    'num_heads': int(num_heads),
                    'num_layers': int(num_layers),
                    'dropout': float(dropout),
                },
                'training': {
                    'epochs': int(epochs),
                    'lr': float(lr),
                    'weight_decay': float(weight_decay),
                    'batch_size': int(batch_size),
                    'val_frac': float(val_frac),
                    'seed': int(seed),
                    'init_checkpoint': init_checkpoint,
                    'scenarios': {sc: list(files) for sc, files in scenarios_spec.items()},
                    'weights': {sc: float(w) for sc, w in weights_spec.items()},
                },
                # Back-compat fields read by rl/evaluation.py
                'val_accuracy': float(mean_acc),
                'val_f1': float(np.mean([s['f1'] for s in per_sc.values()])),
                'val_precision': float(np.mean([s['prec'] for s in per_sc.values()])),
                'val_recall': float(np.mean([s['rec'] for s in per_sc.values()])),
                'val_tp': int(sum(s['tp'] for s in per_sc.values())),
                'val_fp': int(sum(s['fp'] for s in per_sc.values())),
                'val_tn': int(sum(s['tn'] for s in per_sc.values())),
                'val_fn': int(sum(s['fn'] for s in per_sc.values())),
                'val_size': int(sum(s['size'] for s in per_sc.values())),
            }
            with open(metrics_path, 'w') as mf:
                json.dump(best_metrics, mf, indent=2)
            log_fn(f"  --> Best Model Saved (min-scenario-acc: {min_acc:.4f})")

    return best_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=str, required=True,
                    help='JSON config file (see module docstring for expected keys).')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    train_from_jsons(
        scenarios_spec=cfg['scenarios_spec'],
        weights_spec=cfg.get('weights_spec', {}),
        output_path=cfg['output_path'],
        metrics_path=cfg['metrics_path'],
        epochs=cfg.get('epochs', 60),
        init_checkpoint=cfg.get('init_checkpoint'),
        hidden_dim=cfg.get('hidden_dim', 192),
        num_heads=cfg.get('num_heads', 4),
        num_layers=cfg.get('num_layers', 3),
        dropout=cfg.get('dropout', 0.1),
        lr=cfg.get('lr', 1e-3),
        weight_decay=cfg.get('weight_decay', 1e-4),
        batch_size=cfg.get('batch_size', 128),
        val_frac=cfg.get('val_frac', 0.2),
        seed=cfg.get('seed', 42),
    )


if __name__ == '__main__':
    main()
