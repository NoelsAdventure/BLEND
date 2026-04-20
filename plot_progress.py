import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def plot_progress(model_dirs, window_size=10):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    found_reward = False
    found_success = False
    found_path = False
    
    for model_dir in model_dirs:
        progress_file = os.path.join(model_dir, 'progress.csv')
        if not os.path.exists(progress_file):
            print(f"Warning: {progress_file} not found. Skipping.")
            continue
        
        try:
            df = pd.read_csv(progress_file)
        except Exception as e:
            print(f"Error reading {progress_file}: {e}")
            continue

        model_name = os.path.basename(model_dir)
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
            ax1.plot(x, y_smooth, label=model_name)
            found_reward = True
        
        # Plot Success Rate
        success_col = None
        for col in ['epsuccessmean', 'success_rate', 'Success Rate']:
            if col in df.columns:
                success_col = col
                break
        
        if success_col:
            y = df[success_col]
            y_smooth = y.rolling(window=window_size, min_periods=1).mean()
            ax2.plot(x, y_smooth, label=model_name)
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
            ax3.plot(x, y_smooth, label=model_name)
            found_path = True

    if not any([found_reward, found_success, found_path]):
        print("No valid data found to plot.")
        plt.close(fig)
        return

    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Learning Progress - Reward')
    if found_reward:
        ax1.legend()
    ax1.grid(True)

    ax2.set_ylabel('Mean Success Rate')
    ax2.set_title('Learning Progress - Success Rate')
    if found_success:
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Success rate data not available in older logs', 
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    ax2.grid(True)

    ax3.set_ylabel('Mean Path Length')
    ax3.set_xlabel('Total Timesteps')
    ax3.set_title('Learning Progress - Path Length')
    if found_path:
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Path length data not available in older logs', 
                 horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    ax3.grid(True)

    plt.tight_layout()
    output_plot = 'learning_progress.png'
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training progress from progress.csv')
    parser.add_argument('--dirs', nargs='+', help='List of model directories in trained_models/')
    parser.add_argument('--window', type=int, default=10, help='Rolling window size for smoothing')
    
    args = parser.parse_args()
    
    if not args.dirs:
        # If no dirs provided, try to find all dirs in trained_models that have progress.csv
        base_dir = 'trained_models'
        if os.path.exists(base_dir):
            args.dirs = []
            for d in os.listdir(base_dir):
                full_path = os.path.join(base_dir, d)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'progress.csv')):
                    args.dirs.append(full_path)
    
    if args.dirs:
        plot_progress(args.dirs, args.window)
    else:
        print("No valid model directories with progress.csv found.")
