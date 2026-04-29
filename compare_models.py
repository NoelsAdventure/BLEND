import torch
import os

# Update these paths to the models you want to compare
model1_path = 'trained_models/ours_gst_visible_seed_42_curr_buffer_0.25_c_l_0.4_clip_param_0.08_considered_steps_2_alpha_0.1_noise_0_0.0/checkpoints/05207.pt'
# model2_path = 'trained_models/Fulltune_random_visi_alpha_1024/checkpoints/00000.pt'
model2_path = 'trained_models/LoraD_visi_invi_alpha_1024/checkpoints/00000.pt'

# model1_path = 'trained_models/ours_gst_visible_seed_42_curr_buffer_0.25_c_l_0.4_clip_param_0.08_considered_steps_2_alpha_0.1_noise_0_0.0/checkpoints/05207.pt'
# model2_path = 'trained_models/Fulltune_visi_invi_seed_42_curr_buffer_0.25_c_l_0.4_clip_param_0.08_considered_steps_2_alpha_0.1_noise_0_0.0/checkpoints/05207.pt'

# model1_path = 'trained_models/Ours_GST/checkpoints/05207.pt'
# model2_path = 'trained_models/LoraC_visi_invi_seed_42/checkpoints/00200.pt'

def compare_models(p1, p2):
    state_dict1 = torch.load(p1, map_location='cpu')
    state_dict2 = torch.load(p2, map_location='cpu')
    
    diffs = []
    
    # We iterate through model 1 and try to find the matching weight in model 2
    for name1, w1 in state_dict1.items():
        # Check if model 2 has it directly or wrapped in LoRA
        name2 = None
        if name1 in state_dict2:
            name2 = name1
        else:
            # Try LoRA mapping (e.g., .weight -> .base_layer.weight)
            lora_name = name1.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
            if lora_name in state_dict2:
                name2 = lora_name

        if name2:
            w2 = state_dict2[name2]
            diff = torch.norm(w1 - w2).item()
            norm1 = torch.norm(w1).item()
            rel_diff = diff / (norm1 + 1e-9)
            diffs.append((name1, diff, rel_diff, name2))
        else:
            # Check for any other common prefixes or suffixes if needed
            pass

    # Sort by absolute difference
    diffs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Original Layer':<45} | {'Matched to':<45} | {'Abs Diff':<10} | {'Rel Diff':<10}")
    print("-" * 130)
    for n1, diff, rel, n2 in diffs[:40]:
        # Format names to fit nicely
        n1_disp = (n1[:42] + '..') if len(n1) > 45 else n1
        n2_disp = (n2[:42] + '..') if len(n2) > 45 else n2
        print(f"{n1_disp:<45} | {n2_disp:<45} | {diff:<10.4f} | {rel:<10.4f}")

if __name__ == "__main__":
    if os.path.exists(model1_path) and os.path.exists(model2_path):
        compare_models(model1_path, model2_path)
    else:
        print(f"Model files not found.\nPath 1: {model1_path}\nPath 2: {model2_path}")
