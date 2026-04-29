import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_scale_vs_success(csv_path='result/master_summary.csv', output_dir='result'):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run run_scale_sweep.sh first.")
        return

    df = pd.read_csv(csv_path)
    
    # Filter for the Scale Sweep experiments
    # Some older entries might have string 'N/A' for lora_scale, we filter them out
    df = df[df['exp_id'].str.contains('Scale_Sweep', na=False)]
    
    # Ensure lora_scale is numeric
    df['lora_scale'] = pd.to_numeric(df['lora_scale'], errors='coerce')
    df = df.dropna(subset=['lora_scale'])
    
    if df.empty:
        print("No Scale_Sweep data found in CSV.")
        return

    # Sort by scale
    df = df.sort_values(by='lora_scale')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['lora_scale'], df['success_rate'], marker='o', linestyle='-', color='red', label='Success Rate')
    plt.plot(df['lora_scale'], df['collision_rate'], marker='s', linestyle='--', color='blue', label='Collision Rate')

    plt.xlabel('LoRA Scale (0.0 = OFF, 1.0 = FULL)')
    plt.ylabel('Rate')
    plt.title('Performance vs LoRA Scale (LoraE Model)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)

    save_path = os.path.join(output_dir, 'scale_vs_performance.png')
    plt.savefig(save_path)
    print(f"Scale graph saved to {save_path}")

if __name__ == "__main__":
    plot_scale_vs_success()
