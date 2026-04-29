import os
import json
import matplotlib.pyplot as plt
import re

def plot_alpha_vs_success(model_prefix='LoraE', output_dir='result'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models_dir = 'trained_models'
    data_points = []

    # 1. Collect data from model directories
    for model_name in os.listdir(models_dir):
        # Filter for models matching the prefix (e.g., LoraE)
        if not model_name.startswith(model_prefix):
            continue
        
        # Extract alpha value using regex
        match = re.search(r'alpha_(\d+)', model_name)
        if not match:
            continue
        alpha = int(match.group(1))

        # Path to evaluation results
        eval_path = os.path.join(models_dir, model_name, 'test', 'all_evaluations.json')
        if not os.path.exists(eval_path):
            continue

        try:
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            
            for exp_id, results in eval_data.items():
                # Extract the base scenario name (e.g., "Adaptive_Separate" from "Adaptive_Separate_500")
                scenario = exp_id.rsplit('_', 1)[0] if '_' in exp_id else exp_id
                
                data_points.append({
                    'alpha': alpha,
                    'scenario': scenario,
                    'success_rate': results['summary']['success_rate']
                })
        except Exception as e:
            print(f"Error reading {eval_path}: {e}")

    if not data_points:
        print(f"No evaluation data found for models starting with {model_prefix}")
        return

    # 2. Organize data for plotting
    # Structure: { scenario: { alpha: success_rate } }
    series = {}
    for pt in data_points:
        scen = pt['scenario']
        if scen not in series:
            series[scen] = {}
        # Keep the latest/highest success rate if multiple entries exist for same alpha
        series[scen][pt['alpha']] = pt['success_rate']

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    for scenario, points in series.items():
        # Sort by alpha for a continuous line
        sorted_alphas = sorted(points.keys())
        sorted_success = [points[a] for a in sorted_alphas]
        
        plt.plot(sorted_alphas, sorted_success, marker='o', label=scenario)

    plt.xscale('log', base=2) # Use log scale if alphas are like 0, 1, 32, 128, 1024
    plt.xlabel('Alpha Value (Log Scale)')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate vs Alpha ({model_prefix} Models)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    save_path = os.path.join(output_dir, f'{model_prefix}_alpha_success.png')
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")

if __name__ == "__main__":
    plot_alpha_vs_success('LoraE')
