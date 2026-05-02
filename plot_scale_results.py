import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re

def plot_scale_results(model_dir='trained_models/LoraE_invi_visi_alpha_128', output_dir='result', min_episodes=50, selected_scales=None):
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

        results.append({
            'lora_scale': scale,
            'success_rate': successes / total,
            'collision_rate': collisions / total,
            'timeout_rate': timeouts / total,
            'total_episodes': total
        })

    if not results:
        print("No valid results found in CSV files.")
        return

    df_results = pd.DataFrame(results).sort_values(by='lora_scale')

    for i, row in df_results.iterrows():
        print(f"Scale {row['lora_scale']}: SR={row['success_rate']:.3f}, CR={row['collision_rate']:.3f} (Episodes: {int(row['total_episodes'])})")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['lora_scale'], df_results['success_rate'], marker='o', linestyle='-', color='red', label='Success Rate')
    plt.plot(df_results['lora_scale'], df_results['collision_rate'], marker='s', linestyle='--', color='blue', label='Collision Rate')
    plt.plot(df_results['lora_scale'], df_results['timeout_rate'], marker='^', linestyle=':', color='green', label='Timeout Rate')

    plt.xlabel('LoRA Scale (0.0 = OFF, 1.0 = FULL)')
    plt.ylabel('Rate')
    plt.title(f'Performance vs LoRA Scale\n({os.path.basename(model_dir)})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'scale_vs_performance.png')
    plt.savefig(save_path)
    print(f"Scale graph saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='trained_models/LoraE_invi_visi_alpha_128')
    parser.add_argument('--output_dir', type=str, default='result')
    parser.add_argument('--min_episodes', type=int, default=50)
    parser.add_argument('--scales', type=str, default=None, help="Comma-separated list of scales to include (e.g. 0.0,0.5,1.0)")
    args = parser.parse_args()
    
    selected_scales = None
    if args.scales:
        selected_scales = [float(s.strip()) for s in args.scales.split(',')]
    
    plot_scale_results(args.model_dir, args.output_dir, args.min_episodes, selected_scales)
