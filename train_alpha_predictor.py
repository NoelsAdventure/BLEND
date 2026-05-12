import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from alpha_predictor import AlphaPredictor
from tqdm import tqdm

def calculate_metrics(targets, preds):
    mae = np.mean(np.abs(targets - preds))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return mae, r2

class ScaleDataset(Dataset):
    def __init__(self, json_path):
        print(f"Loading data from {json_path}...")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.human_inputs = []
        self.robot_inputs = []
        self.targets = []
        
        for ep in data['episodes']:
            # Handle different JSON structures if necessary (some have steps, some have steps_data)
            steps = ep.get('steps_data', ep.get('steps', []))
            for step in steps:
                # 1. Extract Robot State: [px, py, vx, vy, theta]
                r = step['robot']
                r_state = [r['pos'][0], r['pos'][1], r['vel'][0], r['vel'][1], r['theta']]
                
                # 2. Extract Human States: [px, py, vx, vy, uncertainty]
                h_states = []
                uncertainties = step.get('uncertainty', [])
                
                for i, h in enumerate(step['humans']):
                    # Get mean uncertainty for this human
                    u = uncertainties[i] if uncertainties and i < len(uncertainties) else 0.0
                    if isinstance(u, list): u = np.mean(u)
                    
                    h_state = [h['pos'][0], h['pos'][1], h['vel'][0], h['vel'][1], float(u)]
                    h_states.append(h_state)
                
                # Target: lora_scale
                scale = step['robot']['lora_scale']
                
                self.human_inputs.append(h_states)
                self.robot_inputs.append(r_state)
                self.targets.append(scale)
        
        self.human_inputs = torch.tensor(np.array(self.human_inputs), dtype=torch.float32)
        self.robot_inputs = torch.tensor(np.array(self.robot_inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        
        print(f"  --> Loaded {len(self.targets)} steps.")
        
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
    mae, r2 = calculate_metrics(all_targets, all_preds)
    
    print(f"\n[{label}]")
    print(f"MAE: {mae:.4f} | R2: {r2:.4f}")
    return mae, r2, all_preds, all_targets

def train():
    model_dir = 'trained_models/LoraF_invi_visi_rank_1'
    train_json = os.path.join(model_dir, 'test/seperate_mixed_5050_adaptive_gt.json')
    if not os.path.exists(train_json):
        train_json = os.path.join(model_dir, 'test/seperate_mixed_5050_adaptive_gt_exp1.json')
        
    dataset = ScaleDataset(train_json)
    
    train_size = int(0.7 * len(dataset))
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
    model = AlphaPredictor(human_dim=human_dim, robot_dim=robot_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 50
    print(f"\nStarting training on {device} for {epochs} epochs...")
    
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(model_dir, 'alpha_predictor.pth')
    
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
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_human, b_robot, b_target in val_loader:
                b_human, b_robot, b_target = b_human.to(device), b_robot.to(device), b_target.to(device)
                outputs = model(b_human, b_robot)
                loss = criterion(outputs, b_target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {train_loss/len(train_loader):.6f} | Val MSE: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nTraining Complete. Best model saved to {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    # --- FINAL EVALUATIONS ---
    print("\n" + "="*40)
    print("GENERALIZATION TESTING")
    print("="*40)
    
    # 1. Original Validation Split (Mixed 50/50)
    evaluate_model(model, val_loader, device, label="Validation (Mixed 50/50)")
    
    # 2. All Aware (Friendly) Dataset
    try:
        aware_json = os.path.join(model_dir, 'test/seperate_all_aware_adaptive_gt.json')
        aware_dataset = ScaleDataset(aware_json)
        aware_loader = DataLoader(aware_dataset, batch_size=64, shuffle=False)
        evaluate_model(model, aware_loader, device, label="Testing (All Aware)")
    except Exception as e:
        print(f"\n[All Aware] Skip: {e}")

    # 3. All Ignorant Dataset
    try:
        ignorant_json = os.path.join(model_dir, 'test/seperate_all_ignorant_adaptive_gt.json')
        ignorant_dataset = ScaleDataset(ignorant_json)
        ignorant_loader = DataLoader(ignorant_dataset, batch_size=64, shuffle=False)
        evaluate_model(model, ignorant_loader, device, label="Testing (All Ignorant)")
    except Exception as e:
        print(f"\n[All Ignorant] Skip: {e}")
        
    print("\n" + "="*40)

if __name__ == '__main__':
    train()
