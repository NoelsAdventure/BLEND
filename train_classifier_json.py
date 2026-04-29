import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from rl.networks.scenario_classifier import ScenarioClassifier

def load_json_data(file_path, label, max_humans=20):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return []
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    exp_id = list(data.keys())[0]
    episodes = data[exp_id]['episodes']
    
    dataset = []
    for ep in episodes:
        steps = ep.get('steps_data', [])
        if not steps: continue
        
        ep_data = []
        for s in steps:
            # Robot: [px, py, vx, vy]
            r = s['robot']
            r_vec = [r['pos'][0], r['pos'][1], r['vel'][0], r['vel'][1]]
            
            # Humans: [px, py, pred_traj...]
            # Padding to max_humans
            h_matrix = np.zeros((max_humans, 14))
            for i, h in enumerate(s['humans'][:max_humans]):
                h_matrix[i, 0:2] = h['pos']
                if i < len(s['pred_traj']):
                    h_matrix[i, 2:] = s['pred_traj'][i]
            
            ep_data.append({
                'robot': r_vec,
                'humans': h_matrix,
                'label': label
            })
        dataset.append(ep_data)
    return dataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    model_dir = "trained_models/LoraE_invi_visi_alpha_128/test"
    data_v = load_json_data(os.path.join(model_dir, "LoraE_Visi.json"), 1.0)
    data_i = load_json_data(os.path.join(model_dir, "LoraE_Invi.json"), 0.0)
    all_episodes = data_v + data_i
    
    # 2. Model
    model = ScenarioClassifier(human_dim=14, robot_dim=4, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # 3. Simple Training Loop
    epochs = 50
    for epoch in range(epochs):
        np.random.shuffle(all_episodes)
        total_loss = 0
        correct = 0
        total_steps = 0
        
        for ep in all_episodes:
            h_state = torch.zeros(1, 64).to(device)
            ep_loss = 0
            
            for step in ep:
                robot = torch.FloatTensor(step['robot']).unsqueeze(0).to(device)
                humans = torch.FloatTensor(step['humans']).unsqueeze(0).to(device)
                label = torch.FloatTensor([[step['label']]]).to(device)
                
                prob, h_state = model(robot, humans, h_state, torch.ones(1, 1).to(device))
                
                loss = criterion(prob, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                h_state = h_state.detach()
                total_loss += loss.item()
                
                pred = 1.0 if prob.item() > 0.5 else 0.0
                if pred == step['label']: correct += 1
                total_steps += 1
                
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total_steps:.4f} | Acc: {correct/total_steps:.4f}")

    torch.save(model.state_dict(), "scenario_classifier.pt")
    print("Training complete. Saved to scenario_classifier.pt")

if __name__ == "__main__":
    train()
