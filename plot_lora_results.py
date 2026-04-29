import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(model_dir):
    json_path = os.path.join(model_dir, 'test', 'all_evaluations.json')
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    scales = []
    sr = []
    pl = []

    # Use a dictionary to keep only the LATEST experiment for each scale
    latest_results_by_scale = {}

    # Filter and collect results
    # Sort data items by timestamp if possible, or just rely on insertion order if it's chronological
    # In JSON, order is usually preserved. We want the last one seen for each scale.
    for exp_id, results in data.items():
        config = results.get('config', {})
        summary = results.get('summary', {})
        
        if 'lora_scale' in config:
            scale = config['lora_scale']
            # Only include if human_num is standard (20)
            if config.get('human_num') == 20:
                success_rate = summary.get('success_rate')
                path_length = summary.get('avg_path_length')
                
                if success_rate is not None and path_length is not None:
                    # Overwrite with newer results as we iterate
                    latest_results_by_scale[scale] = {
                        'sr': success_rate,
                        'pl': path_length,
                        'exp_id': exp_id
                    }

    if not latest_results_by_scale:
        print("No valid results found to plot.")
        return

    # Prepare data for plotting (sorted by scale)
    scales = []
    sr = []
    pl = []
    for scale in sorted(latest_results_by_scale.keys()):
        scales.append(scale)
        sr.append(latest_results_by_scale[scale]['sr'])
        pl.append(latest_results_by_scale[scale]['pl'])
        print(f"Scale {scale}: SR={latest_results_by_scale[scale]['sr']}, PL={latest_results_by_scale[scale]['pl']} (ID: {latest_results_by_scale[scale]['exp_id']})")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('LoRA Scale')
    ax1.set_ylabel('Success Rate', color=color)
    ax1.plot(scales, sr, 'o-', color=color, label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Avg Path Length', color=color)  
    ax2.plot(scales, pl, 's-', color=color, label='Avg Path Length')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Performance Metrics vs LoRA Scale\n({os.path.basename(model_dir)})')
    fig.tight_layout()  
    
    save_path = os.path.join(model_dir, 'test', 'lora_scale_performance.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    MODEL_DIR = "trained_models/LoraE_invi_visi_alpha_128"
    plot_results(MODEL_DIR)
