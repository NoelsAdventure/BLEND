
import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results():
    json_path = 'trained_models/LoraB_visi_invi_alpha_1024/test/all_evaluations.json'
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    scales = []
    success_rates = []
    path_lengths = []
    
    # Base model results (clean model without LoRA)
    base_data = None
    
    # Try to find base model results
    # 1. Look for 'base_model_invisible'
    # 2. Look for any experiment that has use_lora: false
    for exp_id, results in data.items():
        if exp_id == "base_model_invisible" or "base_model" in exp_id:
            base_data = results
            print(f"Found base model results by ID: {exp_id}")
            break
            
    if base_data is None:
        for exp_id, results in reversed(list(data.items())):
            if results['config'].get('use_lora') == False:
                base_data = results
                print(f"Found base model results (use_lora=False): {exp_id}")
                break

    gradual_results = []
    for exp_id, results in data.items():
        if "lorab_gradual" in exp_id:
            scale = results['config'].get('lora_scale')
            sr = results['summary']['success_rate']
            pl = results['summary']['avg_path_length']
            if scale is not None:
                gradual_results.append((scale, sr, pl))
    
    # Sort by scale
    gradual_results.sort()
    
    for scale, sr, pl in gradual_results:
        scales.append(scale)
        success_rates.append(sr)
        path_lengths.append(pl)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # --- Plot Success Rate ---
    ax1.plot(scales, success_rates, marker='o', linestyle='-', color='blue', label='LoRA Gradual Scale')
    if base_data is not None:
        bsr = base_data['summary']['success_rate']
        ax1.axhline(y=bsr, color='red', linestyle='--', label=f'Base Model SR: {bsr:.3f}')
        ax1.plot(0.0, bsr, 'rs', markersize=8)
    
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate vs LoRA Scale')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # --- Plot Path Length ---
    ax2.plot(scales, path_lengths, marker='s', linestyle='-', color='green', label='LoRA Gradual Scale')
    if base_data is not None:
        bpl = base_data['summary']['avg_path_length']
        ax2.axhline(y=bpl, color='red', linestyle='--', label=f'Base Model Path Length: {bpl:.1f}')
        ax2.plot(0.0, bpl, 'rs', markersize=8)
        
    ax2.set_ylabel('Avg Path Length (m)')
    ax2.set_xlabel('LoRA Scale (Alpha Factor)')
    ax2.set_title('Path Length vs LoRA Scale')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    save_path = 'lora_metrics_plot.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_results()
