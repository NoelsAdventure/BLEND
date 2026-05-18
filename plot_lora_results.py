import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np

def plot_results(model_dir, output_dir=None, min_episodes=100, selected_scales=None):
    test_dir = os.path.join(model_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found.")
        return

    # Find all scale-specific CSV files
    csv_files = glob.glob(os.path.join(test_dir, 'evaluation_data_scale_*.csv'))
    
    if not csv_files:
        print(f"No evaluation_data_scale_*.csv files found in {test_dir}")
        return

    results = []
    
    for csv_path in csv_files:
        # Extract scale from filename using regex
        match = re.search(r'evaluation_data_scale_([\d\.]+)\.csv', os.path.basename(csv_path))
        if not match:
            continue
        
        scale = float(match.group(1))

        # Filter by selected scales if provided
        if selected_scales is not None and scale not in selected_scales:
            continue
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Calculate counts
        successes = df['Success Times'].count()
        collisions = df['Collision Times'].count()
        timeouts = df['Timeout Times'].count()
        total = successes + collisions + timeouts
        
        if total < min_episodes:
            print(f"Skipping scale {scale}: Only {total} episodes (min_episodes={min_episodes})")
            continue

        # Calculate Average Path Length
        avg_path_length = df['Path Length'].mean()
        
        results.append({
            'lora_scale': scale,
            'success_rate': successes / total,
            'avg_path_length': avg_path_length,
            'total_episodes': total
        })

    if not results:
        print("No valid results found in CSV files.")
        return

    # Prepare data for plotting (sorted by scale).
    # Clamp to [0, 1]: scales outside that range are sweep stress-tests, not
    # the useful operating range — exclude them so the plot focuses on the
    # behaviourally meaningful interval.
    df_results = pd.DataFrame(results).sort_values(by='lora_scale')
    df_results = df_results[(df_results['lora_scale'] >= 0.0) & (df_results['lora_scale'] <= 1.0)]
    if df_results.empty:
        print("No scales in [0, 1] after clamping; nothing to plot.")
        return

    scales = df_results['lora_scale'].tolist()
    sr = df_results['success_rate'].tolist()
    pl = df_results['avg_path_length'].tolist()

    for i in range(len(scales)):
        print(f"Scale {scales[i]}: SR={sr[i]:.3f}, PL={pl[i]:.3f} (Episodes: {df_results.iloc[i]['total_episodes']})")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('LoRA Scale')
    ax1.set_ylabel('Success Rate', color=color)
    ax1.plot(scales, sr, 'o-', color=color, label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0.0, 1.0)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Avg Path Length (m)', color=color)  
    ax2.plot(scales, pl, 's-', color=color, label='Avg Path Length')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Performance Metrics vs LoRA Scale\n({os.path.basename(model_dir)})')
    fig.tight_layout()  
    
    save_dir = output_dir if output_dir else test_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'lora_scale_performance.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show() # Disabled for headless compatibility

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="trained_models/LoraE_invi_visi_alpha_128")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save the plot. Defaults to model_dir/test")
    parser.add_argument('--min_episodes', type=int, default=400,
                        help='Skip scale CSVs with fewer than this many episodes. '
                             'Default 400 filters out partial-overwrite files '
                             '(e.g. a 200-ep scale=1.0 dropped among 500-ep neighbours).')
    parser.add_argument('--scales', type=str, default=None, help="Comma-separated list of scales to include (e.g. 0.0,0.5,1.0)")
    args = parser.parse_args()
    
    selected_scales = None
    if args.scales:
        selected_scales = [float(s.strip()) for s in args.scales.split(',')]
    
    plot_results(args.model_dir, args.output_dir, args.min_episodes, selected_scales)
