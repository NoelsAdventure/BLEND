import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from alpha_predictor import AlphaPredictor
from tqdm import tqdm

class ScaleDataset(Dataset):
    def __init__(self, json_path):
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.inputs = []
        self.targets = []
        
        for ep in data['episodes']:
            # Use 'steps_data' as confirmed by previous inspection
            for step in ep['steps_data']:
                # Input: pred_traj (identically structured to policy input)
                spatial_edges = step['pred_traj']
                # Target: lora_scale from robot dict
                scale = step['robot']['lora_scale']
                
                self.inputs.append(spatial_edges)
                self.targets.append(scale)
        
        # Convert to numpy then tensor to handle list of lists
        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        print(f"Loaded {len(self.inputs)} steps. Input shape: {self.inputs.shape}")
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def train():
    # 1. Dataset Configuration
    json_path = 'trained_models/LoraF_invi_visi_rank_1/test/seperate_mixed_5050_adaptive_gt.json'
    if not os.path.exists(json_path):
        # Fallback to the _exp1 version if standard doesn't exist
        json_path = 'trained_models/LoraF_invi_visi_rank_1/test/seperate_mixed_5050_adaptive_gt_exp1.json'
        
    dataset = ScaleDataset(json_path)
    
    # 2. Split: 70% Training, 30% Validation/Testing
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 3. Model Architecture
    input_dim = dataset.inputs.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaPredictor(input_dim=input_dim).to(device)
    
    # 4. Training Pipeline
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 30
    print(f"Starting training on {device} for {epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
        
        # 5. Save best checkpoint directly into RL trained folder
        save_dir = 'trained_models/LoraF_invi_visi_rank_1'
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, 'alpha_predictor.pth')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            # print(f"  --> Saved new best model to {checkpoint_path}")

    print(f"\nTraining Complete. Final model saved to {checkpoint_path}")

if __name__ == '__main__':
    train()
