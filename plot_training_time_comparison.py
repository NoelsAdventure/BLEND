import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# Use a non-interactive backend for headless environments
plt.switch_backend('Agg')

# Increase global font sizes for better readability
plt.rcParams.update({'font.size': 14})

# Configuration: Define the models and their directory paths
# MODELS = {
#     'Fulltune 1': 'trained_models/Fulltune_uni_invi',
#     'Fulltune 2': 'trained_models/Fulltune_uni_invi'
# }
# # Configuration: Define the models and their directory paths
MODELS = {
    'LoRA rank 1': 'trained_models/LoraF_invi_visi_rank_1',
    'LoRA rank 4': 'trained_models/LoraF_invi_visi_rank_4',
    'Fulltune': 'trained_models/Fulltune2_invi_visi'
}

def plot_comparison():
    fig, axes = plt.subplots(2, 1, figsize=(13, 12), sharex=False)
    
    # Create twin axes ONCE outside the loop
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    # Use different colors for different models
    colors = {'LoRA rank 1': 'blue', 'LoRA rank 4': 'green', 'Fulltune': 'red'}
    
    x_label = 'Cumulative Training Time (seconds)' # Default fallback
    
    for label, path in MODELS.items():
        csv_path = os.path.join(path, 'progress.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping {label}.")
            continue
        
        df = pd.read_csv(csv_path)
        
        if 'train_time' in df.columns:
            x_axis = df['train_time']
            x_label = 'Cumulative Training Time (seconds)'
        else:
            x_axis = df['misc/total_timesteps']
            x_label = 'Total Timesteps'

        # Plot Group 1: Reward and Cost
        ax1.plot(x_axis, df['eprewmean'], label=f'{label}', color=colors[label], linestyle='-')
        ax1_twin.plot(x_axis, df['epcostmean'], color=colors[label], linestyle='--')
        
        # Plot Group 2: Success Rate and Path Length
        ax2.plot(x_axis, df['epsuccessmean'], label=f'{label}', color=colors[label], linestyle='-')
        ax2_twin.plot(x_axis, df['eppathlengthmean'], color=colors[label], linestyle='--')

    # Formatting axes 1
    ax1.set_ylabel('Mean Episode Reward')
    ax1_twin.set_ylabel('Mean Episode Cost')
    ax1.set_xlabel(x_label)
    ax1.set_title('Reward and Cost vs. Training Time')
    
    # Formatting axes 2
    ax2.set_ylabel('Success Rate')
    ax2_twin.set_ylabel('Path Length')
    ax2.set_xlabel(x_label)
    ax2.set_title('Success Rate and Path Length vs. Training Time')

    # Custom Legends
    style_reward = mlines.Line2D([], [], color='black', linestyle='-', label='Reward (Solid Line)')
    style_cost = mlines.Line2D([], [], color='black', linestyle='--', label='Cost (Dashed Line)')
    
    style_sr = mlines.Line2D([], [], color='black', linestyle='-', label='Success Rate (Solid Line)')
    style_pl = mlines.Line2D([], [], color='black', linestyle='--', label='Path Length (Dashed Line)')

    # Add legends to ax1
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1.extend([style_reward, style_cost])
    labels1.extend(['Reward (Solid Line)', 'Cost (Dashed Line)'])
    ax1.legend(handles=handles1, labels=labels1, loc='center right', title="Models & Metrics", framealpha=0.7)
    
    # Add legends to ax2
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2.extend([style_sr, style_pl])
    labels2.extend(['Success Rate (Solid Line)', 'Path Length (Dashed Line)'])
    ax2.legend(handles=handles2, labels=labels2, loc='center right', title="Models & Metrics", framealpha=0.7)

    plt.tight_layout()
    output_file = 'training_time_comparison.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    plot_comparison()
