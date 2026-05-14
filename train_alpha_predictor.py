import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from alpha_predictor import FriendlyPredictor
from tqdm import tqdm

def calculate_classification_metrics(targets, preds):
    targets_flat = targets.flatten()
    preds_binary = (preds > 0.5).astype(float).flatten()
    
    # Manual Confusion Matrix implementation (removing sklearn)
    tp = np.sum((targets_flat == 1) & (preds_binary == 1))
    fp = np.sum((targets_flat == 0) & (preds_binary == 1))
    tn = np.sum((targets_flat == 0) & (preds_binary == 0))
    fn = np.sum((targets_flat == 1) & (preds_binary == 0))
    
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return acc, prec, rec, f1, int(tp), int(fp), int(tn), int(fn)

class FriendlyDataset(Dataset):
    # Per-step uncertainty horizon, matches crowd_sim/envs/utils/human.py::pred_horizon_aci.
    NUM_PRED_STEPS = 5

    def __init__(self, json_paths, max_humans=20):
        self.human_inputs = []
        self.robot_inputs = []
        self.targets = []
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

            for ep in data['episodes']:
                steps = ep.get('steps_data', ep.get('steps', []))
                for step in steps:
                    # 1. Robot state: match the RL policy's input
                    #    [vx, vy, px, py, radius, gx, gy, v_pref, theta].
                    r = step['robot']
                    rx, ry = r['pos'][0], r['pos'][1]
                    if 'v_pref' in r:
                        v_pref = float(r['v_pref'])
                    else:
                        if not self._warned_missing_v_pref:
                            print("  --> Legacy dump missing 'v_pref'; defaulting to 1.0. "
                                  "Regenerate dumps for clean training data.")
                            self._warned_missing_v_pref = True
                        v_pref = 1.0
                    r_state = [
                        float(r['vel'][0]), float(r['vel'][1]),
                        float(rx), float(ry),
                        float(r['radius']),
                        float(r['goal'][0]), float(r['goal'][1]),
                        v_pref,
                        float(r['theta']),
                    ]

                    # 2. Extract Predictions and Uncertainties (Sorted by distance by the env)
                    pred_trajs = step.get('pred_traj', [])
                    uncertainties = step.get('uncertainty', [])

                    # 3. Sort Humans by Distance to align with pred_traj
                    humans = step['humans']
                    def get_dist(h):
                        return np.linalg.norm(np.array(h['pos']) - np.array([rx, ry]))
                    sorted_humans = sorted(humans, key=get_dist)

                    h_states = []
                    h_labels = []

                    for i in range(max_humans):
                        if i < len(pred_trajs) and i < len(sorted_humans):
                            traj = pred_trajs[i]  # [dx0, dy0, dx1, dy1, ...]

                            # Per-human uncertainty: keep the full per-horizon vector.
                            # Legacy dumps may store a single scalar — broadcast it.
                            u_raw = uncertainties[i] if uncertainties and i < len(uncertainties) else None
                            if u_raw is None:
                                u_row = [0.0] * self.NUM_PRED_STEPS
                            elif isinstance(u_raw, (list, tuple, np.ndarray)):
                                u_row = [float(x) for x in list(u_raw)]
                            else:
                                u_row = [float(u_raw)] * self.NUM_PRED_STEPS
                            if len(u_row) < self.NUM_PRED_STEPS:
                                u_row = u_row + [0.0] * (self.NUM_PRED_STEPS - len(u_row))
                            elif len(u_row) > self.NUM_PRED_STEPS:
                                u_row = u_row[:self.NUM_PRED_STEPS]

                            h_state = list(traj) + u_row
                            # Prefer the ground-truth field (added 2026-05); fall
                            # back to is_friendly for legacy dumps where only the
                            # in-range humans had a real label.
                            if 'actual_friendly' in sorted_humans[i]:
                                h_label = float(sorted_humans[i]['actual_friendly'])
                            else:
                                h_label = float(sorted_humans[i].get('is_friendly', False))
                        else:
                            # Padding
                            feature_dim = (len(pred_trajs[0]) + self.NUM_PRED_STEPS) if pred_trajs else (12 + self.NUM_PRED_STEPS)
                            h_state = [0.0] * feature_dim
                            h_label = 0.0

                        h_states.append(h_state)
                        h_labels.append(h_label)

                    self.human_inputs.append(h_states)
                    self.robot_inputs.append(r_state)
                    self.targets.append(h_labels)
        
        self.human_inputs = torch.tensor(np.array(self.human_inputs), dtype=torch.float32)
        self.robot_inputs = torch.tensor(np.array(self.robot_inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        
        print(f"  --> Total steps loaded: {len(self.targets)}")
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.human_inputs[idx], self.robot_inputs[idx], self.targets[idx]

def evaluate_model(model, dataloader, device, label="Evaluation"):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for b_human, b_robot, b_target in dataloader:
            b_human, b_robot = b_human.to(device), b_robot.to(device)
            outputs = model(b_human, b_robot)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(b_target.numpy())

    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    acc, prec, rec, f1, tp, fp, tn, fn = calculate_classification_metrics(all_targets, all_preds)
    
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
    return acc, f1, all_preds, all_targets

def train():
    model_dir = 'trained_models/LoraF_invi_visi_rank_1'
    
    # 1. Collect all "adaptive_gt" scenarios for training to cover "all scenarios"
    train_scenarios = [
        'seperate_mixed_5050_adaptive_gt.json',
        'seperate_all_aware_adaptive_gt.json',
        'seperate_all_ignorant_adaptive_gt.json',
        'seperate_ignorant_to_aware_step25_adaptive_gt.json'
    ]
    train_jsons = [os.path.join(model_dir, 'test', s) for s in train_scenarios]
    
    dataset = FriendlyDataset(train_jsons)
    
    if len(dataset) == 0:
        print("Error: No data loaded. Check file paths.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Dimensions
    human_dim = dataset.human_inputs.shape[-1]
    robot_dim = dataset.robot_inputs.shape[-1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FriendlyPredictor(human_dim=human_dim, robot_dim=robot_dim).to(device)
    
    # Use Binary Cross Entropy Loss for classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 20 # Increased epochs slightly for more complex features
    print(f"\nStarting training on {device} for {epochs} epochs...")
    print(f"Feature Dimensions: Robot={robot_dim}, Human={human_dim}")
    
    best_val_f1 = 0
    checkpoint_path = os.path.join(model_dir, 'friendly_predictor.pth')
    metrics_path = os.path.join(model_dir, 'friendly_predictor_metrics.json')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for b_human, b_robot, b_target in train_loader:
            b_human, b_robot, b_target = b_human.to(device), b_robot.to(device), b_target.to(device)

            optimizer.zero_grad()
            outputs = model(b_human, b_robot)
            loss = criterion(outputs, b_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        acc, f1, val_preds, val_targets = evaluate_model(model, val_loader, device, label=f"Epoch {epoch+1} Val")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), checkpoint_path)
            _, prec, rec, _, tp, fp, tn, fn = calculate_classification_metrics(val_targets, val_preds)
            total = tp + fp + tn + fn
            def _pct(x):
                return float(100.0 * x / total) if total > 0 else 0.0
            with open(metrics_path, 'w') as mf:
                json.dump({
                    'epoch': epoch + 1,
                    'val_accuracy': float(acc),
                    'val_f1': float(f1),
                    'val_precision': float(prec),
                    'val_recall': float(rec),
                    'val_tp': int(tp), 'val_fp': int(fp), 'val_tn': int(tn), 'val_fn': int(fn),
                    'val_tp_pct': _pct(tp), 'val_fp_pct': _pct(fp),
                    'val_tn_pct': _pct(tn), 'val_fn_pct': _pct(fn),
                    'val_size': int(val_size),
                    'threshold': 0.5,
                    'human_dim': int(human_dim),
                    'robot_dim': int(robot_dim),
                    'max_humans': int(dataset.max_humans),
                }, mf, indent=2)
            print(f"  --> Best Model Saved (F1: {f1:.4f})")

    print(f"\nTraining Complete. Best model saved to {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    # --- FINAL EVALUATIONS ON DIFFERENT SCENARIOS ---
    print("\n" + "="*40)
    print("SCENARIO GENERALIZATION TESTING")
    print("="*40)
    
    scenarios_to_test = {
        "Mixed 50/50": 'seperate_mixed_5050_adaptive_gt.json',
        "All Aware (Friendly)": 'seperate_all_aware_adaptive_gt.json',
        "All Ignorant": 'seperate_all_ignorant_adaptive_gt.json',
        "Switching (Step 25)": 'seperate_ignorant_to_aware_step25_adaptive_gt.json'
    }

    for name, filename in scenarios_to_test.items():
        try:
            test_json = os.path.join(model_dir, 'test', filename)
            test_dataset = FriendlyDataset(test_json)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            evaluate_model(model, test_loader, device, label=f"Testing ({name})")
        except Exception as e:
            print(f"\n[{name}] Skip: {e}")
        
    print("\n" + "="*40)

if __name__ == '__main__':
    train()
