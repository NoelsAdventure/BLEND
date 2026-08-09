import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def plot_progress(model_configs, window_size=10, cost_limit=None):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

    found_reward = False
    found_cost = False
    found_success = False
    found_path = False
    
    # Pre-assign colors to ensure consistency across subplots
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for i, config in enumerate(model_configs):
        model_dir = config['path']
        model_name = config['name']
        color = colors[i % len(colors)]
        
        progress_file = os.path.join(model_dir, 'progress.csv')
        if not os.path.exists(progress_file):
            print(f"Warning: {progress_file} not found. Skipping.")
            continue
        
        try:
            df = pd.read_csv(progress_file)
        except Exception as e:
            print(f"Error reading {progress_file}: {e}")
            continue

        if 'misc/total_timesteps' not in df.columns:
            print(f"Warning: 'misc/total_timesteps' column missing in {model_name}. Skipping.")
            continue
            
        x = df['misc/total_timesteps']
        
        # Plot Reward
        reward_col = None
        for col in ['eprewmean', 'reward', 'Mean Reward']:
            if col in df.columns:
                reward_col = col
                break
        
        if reward_col:
            y = df[reward_col]
            y_smooth = y.rolling(window=window_size, min_periods=1).mean()
            ax1.plot(x, y_smooth, label=model_name, color=color)
            found_reward = True
        
        # Plot Cost (2nd panel)
        cost_col = None
        for col in ['epcostmean', 'cost', 'Mean Cost']:
            if col in df.columns:
                cost_col = col
                break

        if cost_col:
            y = df[cost_col]
            y_smooth = y.rolling(window=window_size, min_periods=1).mean()
            ax2.plot(x, y_smooth, label=model_name, color=color)
            found_cost = True

        # Plot Success Rate
        success_col = None
        for col in ['epsuccessmean', 'success_rate', 'Success Rate']:
            if col in df.columns:
                success_col = col
                break

        if success_col:
            y = df[success_col]
            y_smooth = y.rolling(window=window_size, min_periods=1).mean()
            ax3.plot(x, y_smooth, label=model_name, color=color)
            found_success = True

        # Plot Path Length
        path_col = None
        for col in ['eppathlengthmean', 'path_length', 'Path Length']:
            if col in df.columns:
                path_col = col
                break

        if path_col:
            y = df[path_col]
            y_smooth = y.rolling(window=window_size, min_periods=1).mean()
            ax4.plot(x, y_smooth, label=model_name, color=color)
            found_path = True

    if not any([found_reward, found_cost, found_success, found_path]):
        print("No valid data found to plot.")
        plt.close(fig)
        return

    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Learning Progress - Reward')
    if found_reward:
        ax1.legend()
    ax1.grid(True)

    ax2.set_ylabel('Mean Episode Cost')
    ax2.set_title('Learning Progress - Cost')
    if cost_limit is not None:
        ax2.axhline(y=cost_limit, color='red', linestyle='--', linewidth=1.2,
                    label=f'cost_limit = {cost_limit}')
    if found_cost or cost_limit is not None:
        ax2.legend()
    if not found_cost:
        ax2.text(0.5, 0.5, 'Cost data not available in older logs',
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    ax2.grid(True)

    ax3.set_ylabel('Mean Success Rate')
    ax3.set_title('Learning Progress - Success Rate')
    if found_success:
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Success rate data not available in older logs',
                 horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    ax3.grid(True)

    ax4.set_ylabel('Mean Path Length')
    ax4.set_xlabel('Total Timesteps')
    ax4.set_title('Learning Progress - Path Length')
    if found_path:
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Path length data not available in older logs',
                 horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
    ax4.grid(True)

    plt.tight_layout()
    output_plot = 'learning_progress.png'
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training progress from progress.csv')
    parser.add_argument('--dirs', nargs='+', help='List of model directories or path:label pairs')
    parser.add_argument('--window', type=int, default=10, help='Rolling window size for smoothing')
    parser.add_argument('--cost_limit', type=float, default=0.4,
                        help='Cost limit reference line on cost panel (set to negative to disable)')
    
    args = parser.parse_args()
    
    default_model_configs = [
        {'path': 'trained_models/Conservative_small', 'name': 'Conservative small'},
        {'path': 'trained_models/LoRA_small', 'name': 'LoRA small'},
        {'path': 'trained_models/Fullfinetune_small_v2', 'name': 'Fullfinetune small'},
        {'path': 'trained_models/Conservative_medium', 'name': 'Conservative medium'},
        {'path': 'trained_models/LoRA_medium', 'name': 'LoRA medium'},
        {'path': 'trained_models/Fullfinetune_medium', 'name': 'Fullfinetune medium'},
        {'path': 'trained_models/Conservative_large', 'name': 'Conservative large'},
        {'path': 'trained_models/LoRA_large', 'name': 'LoRA large'},
        {'path': 'trained_models/Fullfinetune_large', 'name': 'Fullfinetune large'},
        {'path': 'trained_models/Conservative_giant', 'name': 'Conservative giant'},
        {'path': 'trained_models/LoRA_giant', 'name': 'LoRA giant'},
        {'path': 'trained_models/Fullfinetune_giant', 'name': 'Fullfinetune giant'},
        {'path': 'trained_models/Fullfinetune_small', 'name': 'Fullfinetune small from Ours_GST'},
    ]

    model_configs = []
    if args.dirs:
        for d in args.dirs:
            if ':' in d:
                path, name = d.split(':', 1)
            else:
                path = d
                name = os.path.basename(d.rstrip('/'))
            model_configs.append({'path': path, 'name': name})
    else:
        model_configs = default_model_configs

    if not model_configs:
        # If no dirs provided, try to find all dirs in trained_models that have progress.csv
        base_dir = 'trained_models'
        if os.path.exists(base_dir):
            for d in sorted(os.listdir(base_dir)):
                full_path = os.path.join(base_dir, d)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'progress.csv')):
                    model_configs.append({'path': full_path, 'name': d})
    
    if model_configs:
        cl = args.cost_limit if args.cost_limit is not None and args.cost_limit >= 0 else None
        plot_progress(model_configs, args.window, cost_limit=cl)
    else:
        print("No valid model directories with progress.csv found.")

