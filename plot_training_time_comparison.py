import os
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

MODELS = {
    'LoRA_A': 'trained_models/LoraZ_invi_visi_rank_4',
    'LoRA_B': 'trained_models/LoraF_invi_visi_rank_1',
    'LoRA_C': 'trained_models/LoraZ_invi_visi_rank_1',
    'FullFineTune':    'trained_models/FullFineTune_invi_visi',
}

COLORS = {
    'LoRA_A': 'tab:blue',
    'LoRA_B': 'tab:green',
    'LoRA_C': 'tab:orange',
    'FullFineTune':    'tab:red',
}

MAX_TIMESTEPS = 10_000_000


def plot_comparison():
    fig, (ax_reward, ax_cost, ax_sr, ax_pl) = plt.subplots(
        4, 1, figsize=(10, 24), sharex=True
    )

    for label, path in MODELS.items():
        csv_path = os.path.join(path, 'progress.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping {label}.")
            continue

        df = pd.read_csv(csv_path)
        if 'misc/total_timesteps' not in df.columns:
            print(f"Warning: 'misc/total_timesteps' missing in {csv_path}. Skipping {label}.")
            continue

        x = df['misc/total_timesteps']
        color = COLORS[label]

        ax_reward.plot(x, df['eprewmean'],       color=color, label=label, linewidth=1.8)
        ax_cost.plot(  x, df['epcostmean'],      color=color, label=label, linewidth=1.8)
        ax_sr.plot(    x, df['epsuccessmean'],   color=color, label=label, linewidth=1.8)
        ax_pl.plot(    x, df['eppathlengthmean'],color=color, label=label, linewidth=1.8)

    for ax in (ax_reward, ax_cost, ax_sr, ax_pl):
        ax.set_xlim(0, MAX_TIMESTEPS)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')

    ax_reward.set_ylabel('Mean Episode Reward')
    ax_reward.set_title('Mean Episode Reward vs. Total Timesteps')

    ax_cost.set_ylabel('Mean Episode Cost')
    ax_cost.set_title('Mean Episode Cost vs. Total Timesteps')

    ax_sr.set_ylabel('Success Rate')
    ax_sr.set_ylim(0.55, 1.05)
    ax_sr.set_title('Success Rate vs. Total Timesteps')

    ax_pl.set_ylabel('Path Length')
    ax_pl.set_title('Path Length vs. Total Timesteps')
    ax_pl.set_xlabel('Total Timesteps')

    fig.tight_layout()
    output_file = 'training_time_comparison.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")


if __name__ == "__main__":
    plot_comparison()
