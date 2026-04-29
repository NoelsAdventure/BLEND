import json
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm

def prepare_data(model_dir, seq_len=5, max_humans=20):
    test_dir = os.path.join(model_dir, 'test')
    
    # Load Visi and Invi data
    visi_files = [f for f in os.listdir(test_dir) if f.startswith('LoraE_Visi')]
    invi_files = [f for f in os.listdir(test_dir) if f.startswith('LoraE_Invi')]
    
    all_robot_seqs = []
    all_human_seqs = []
    all_human_masks = []
    all_labels = []
    
    def process_json(file_path, label):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Iterate through experiments in the JSON
        for exp_id, exp_data in data.items():
            for episode in exp_data['episodes']:
                steps = episode['steps_data']
                if len(steps) < seq_len:
                    continue
                
                # Extract sequences using a sliding window
                for i in range(len(steps) - seq_len + 1):
                    window = steps[i : i + seq_len]
                    
                    robot_seq = []
                    human_seq = []
                    human_mask = []
                    
                    for step in window:
                        # 1. Robot features [px, py, vx, vy]
                        r = step['robot']
                        robot_seq.append([r['pos'][0], r['pos'][1], r['vel'][0], r['vel'][1]])
                        
                        # 2. Human features [px, py, pred...]
                        h_step_features = []
                        h_step_mask = []
                        
                        # Pad or truncate humans to max_humans
                        humans = step['humans'][:max_humans]
                        preds = step['pred_traj'][:max_humans]
                        
                        for h_idx in range(max_humans):
                            if h_idx < len(humans):
                                h = humans[h_idx]
                                p = preds[h_idx] # This is a list of 12 numbers (6 steps * 2 coords)
                                
                                # Combine pos and pred
                                feat = [h['pos'][0], h['pos'][1]] + p
                                h_step_features.append(feat)
                                h_step_mask.append(1)
                            else:
                                # Padding
                                h_step_features.append([0.0] * 26) # 2 + 24 if pred is 12*2
                                h_step_mask.append(0)
                        
                        human_seq.append(h_step_features)
                        human_mask.append(h_step_mask)
                    
                    all_robot_seqs.append(robot_seq)
                    all_human_seqs.append(human_seq)
                    all_human_masks.append(human_mask)
                    all_labels.append(label)

    print("Processing Visible data...")
    for f in visi_files:
        process_json(os.path.join(test_dir, f), 1.0)
        
    print("Processing Invisible data...")
    for f in invi_files:
        process_json(os.path.join(test_dir, f), 0.0)

    # Convert to tensors
    dataset = {
        'robot_seqs': torch.tensor(all_robot_seqs, dtype=torch.float32),
        'human_seqs': torch.tensor(all_human_seqs, dtype=torch.float32),
        'human_masks': torch.tensor(all_human_masks, dtype=torch.float32),
        'labels': torch.tensor(all_labels, dtype=torch.float32)
    }
    
    save_path = os.path.join(test_dir, 'classifier_dataset.pt')
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")
    print(f"Total sequences: {len(all_labels)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='trained_models/LoraE_invi_visi_alpha_128')
    parser.add_argument('--seq_len', type=int, default=5)
    args = parser.parse_args()
    
    prepare_data(args.model_dir, args.seq_len)
