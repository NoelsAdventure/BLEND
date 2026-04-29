import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(csv_path='result/master_summary.csv', output_dir='result'):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run aggregate_results.py first.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data to plot.")
        return

    # Use exp_id as the primary label
    df = df.sort_values('exp_id')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    x_labels = df['exp_id'].tolist()
    x = range(len(df))

    # Subplot 1: Success & Collision Rates
    ax1.bar([i - 0.2 for i in x], df['success_rate'], width=0.4, label='Success Rate', color='green')
    ax1.bar([i + 0.2 for i in x], df['collision_rate'], width=0.4, label='Collision Rate', color='red')
    ax1.set_ylabel('Rate (0.0 - 1.0)')
    ax1.set_title('Success vs Collision Rates by Experiment ID')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=30, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Subplot 2: Path Length
    ax2.bar(x, df['avg_path_length'], color='skyblue', edgecolor='navy')
    ax2.set_ylabel('Average Path Length (m)')
    ax2.set_title('Robot Path Length by Experiment ID')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=30, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'experiment_comparison.png')
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    plot_comparison()
