import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import argparse
from rl.networks.classifier import ScenarioClassifier

def train_classifier(model_dir, epochs=50, batch_size=64, lr=1e-3):
    test_dir = os.path.join(model_dir, 'test')
    data_path = os.path.join(test_dir, 'classifier_dataset.pt')
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Run prepare_classifier_data.py first.")
        return

    # Load data
    data = torch.load(data_path)
    full_dataset = TensorDataset(
        data['robot_seqs'], 
        data['human_seqs'], 
        data['human_masks'], 
        data['labels']
    )
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScenarioClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for r, h, m, l in train_loader:
            r, h, m, l = r.to(device), h.to(device), m.to(device), l.to(device)
            
            optimizer.zero_grad()
            prob, _ = model(r, h, m)
            loss = criterion(prob.squeeze(), l)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for r, h, m, l in val_loader:
                r, h, m, l = r.to(device), h.to(device), m.to(device), l.to(device)
                prob, _ = model(r, h, m)
                val_loss += criterion(prob.squeeze(), l).item()
                
                preds = (prob > 0.5).float()
                correct += (preds.squeeze() == l).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(test_dir, 'scenario_classifier.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='trained_models/LoraE_invi_visi_alpha_128')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_classifier(args.model_dir, args.epochs)
