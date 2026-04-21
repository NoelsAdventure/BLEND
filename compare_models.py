import torch
import os

model1_path = 'trained_models/ours_gst_visible_seed_42_curr_buffer_0.25_c_l_0.4_clip_param_0.08_considered_steps_2_alpha_0.1_noise_0_0.0/checkpoints/05207.pt'
model2_path = 'trained_models/LoraC_visi_invi_alpha_1024/checkpoints/00000.pt'
# model2_path = 'trained_models/Fulltune_visi_invi_seed_42_curr_buffer_0.25_c_l_0.4_clip_param_0.08_considered_steps_2_alpha_0.1_noise_0_0.0/checkpoints/05207.pt'

def compare_models(p1, p2):
    state_dict1 = torch.load(p1, map_location='cpu')
    state_dict2 = torch.load(p2, map_location='cpu')
    
    diffs = []
    
    for name in state_dict1:
        if name in state_dict2:
            w1 = state_dict1[name]
            w2 = state_dict2[name]
            
            # Calculate L2 norm of the difference
            diff = torch.norm(w1 - w2).item()
            # Calculate relative difference if possible
            norm1 = torch.norm(w1).item()
            rel_diff = diff / (norm1 + 1e-9)
            
            diffs.append((name, diff, rel_diff))
        else:
            print(f"Warning: {name} not in model 2")

    # Sort by absolute difference
    diffs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Layer Name':<60} | {'Abs Diff':<10} | {'Rel Diff':<10}")
    print("-" * 85)
    for name, diff, rel_diff in diffs[:30]: # Top 30
        print(f"{name:<60} | {diff:<10.4f} | {rel_diff:<10.4f}")

if __name__ == "__main__":
    if os.path.exists(model1_path) and os.path.exists(model2_path):
        compare_models(model1_path, model2_path)
    else:
        print("Model files not found.")
