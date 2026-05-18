"""Plot per-epoch awareness predictor training history.

Reads one or more `*_history.json` files written by train_alpha_predictor.py
/ train_alpha_from_json.py. Draws per-scenario val accuracy + F1 curves
across epochs, with the min-over-scenarios line emphasized (that's the
checkpoint-selection metric).

Multiple history files are concatenated end-to-end along the epoch axis,
which is the right view for DAgger sweeps (iter1 epochs → iter2 epochs → ...).
Vertical dashed lines mark iteration boundaries.

Usage (typical):

    # Plot the canonical training history
    python plot_awareness_training.py \\
        trained_models/LoraF_invi_visi_rank_1/friendly_predictor_history.json

    # Plot a full DAgger sweep
    python plot_awareness_training.py \\
        trained_models/LoraF_invi_visi_rank_1/friendly_predictor_metrics_dagger_iter1_history.json \\
        trained_models/LoraF_invi_visi_rank_1/friendly_predictor_metrics_dagger_iter2_history.json \\
        trained_models/LoraF_invi_visi_rank_1/friendly_predictor_metrics_dagger_iter3_history.json \\
        --out awareness_dagger.png \\
        --target 0.80
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def _load(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def _collect_scenarios(histories: List[Dict]) -> List[str]:
    """Stable, ordered union of scenario names across all history files."""
    seen = []
    for h in histories:
        for sc in h.get('scenarios', []):
            if sc not in seen:
                seen.append(sc)
        # Also pull from per_scenario in case 'scenarios' is missing.
        for ep in h.get('history', []):
            for sc in ep.get('per_scenario', {}).keys():
                if sc not in seen:
                    seen.append(sc)
    return seen


def _series(histories: List[Dict], scenario: str, metric: str):
    """Per-epoch (concatenated across history files) values for one
    scenario/metric. Returns (x_epochs_global, y_values, boundary_x)."""
    xs, ys = [], []
    boundaries: List[int] = []
    epoch_offset = 0
    for h in histories:
        eps = h.get('history', [])
        for ep in eps:
            per_sc = ep.get('per_scenario', {})
            if scenario not in per_sc:
                continue
            xs.append(epoch_offset + ep['epoch'])
            ys.append(float(per_sc[scenario][metric]))
        if eps:
            epoch_offset += int(eps[-1]['epoch'])
            boundaries.append(epoch_offset)
    return xs, ys, boundaries


def _scalar_series(histories: List[Dict], key: str):
    """Per-epoch scalar field (e.g. min_scenario_acc, train_loss)."""
    xs, ys = [], []
    epoch_offset = 0
    for h in histories:
        eps = h.get('history', [])
        for ep in eps:
            if key not in ep:
                continue
            xs.append(epoch_offset + ep['epoch'])
            ys.append(float(ep[key]))
        if eps:
            epoch_offset += int(eps[-1]['epoch'])
    return xs, ys


def plot(history_paths: List[str], out_path: str, target: float):
    histories = [_load(p) for p in history_paths if os.path.exists(p)]
    if not histories:
        print("No history files loaded — nothing to plot.")
        return

    scenarios = _collect_scenarios(histories)
    if not scenarios:
        print("No scenarios found in history files.")
        return

    fig, (ax_acc, ax_f1) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # Per-scenario accuracy
    cmap = plt.get_cmap('tab10')
    for i, sc in enumerate(scenarios):
        xs, ys, boundaries = _series(histories, sc, 'acc')
        ax_acc.plot(xs, ys, color=cmap(i), label=sc, linewidth=1.5, alpha=0.9)

    # min-over-scenarios highlighted
    xs_min, ys_min = _scalar_series(histories, 'min_scenario_acc')
    ax_acc.plot(xs_min, ys_min, color='black', label='min over scenarios',
                linewidth=2.5, linestyle='--')

    ax_acc.axhline(target, color='red', linestyle=':', alpha=0.7,
                   label=f'target ({target:.2f})')
    ax_acc.set_ylabel('Validation accuracy')
    ax_acc.set_title('Awareness predictor training — per-scenario val accuracy')
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc='lower right', ncol=2)

    # Per-scenario F1
    for i, sc in enumerate(scenarios):
        xs, ys, _ = _series(histories, sc, 'f1')
        ax_f1.plot(xs, ys, color=cmap(i), label=sc, linewidth=1.5, alpha=0.9)

    ax_f1.set_ylabel('Validation F1')
    ax_f1.set_xlabel('Global epoch (concatenated across history files)')
    ax_f1.set_title('Per-scenario val F1 (F1=0 in degenerate scenarios with no positives is expected)')
    ax_f1.set_ylim(0.0, 1.0)
    ax_f1.grid(True, alpha=0.3)
    ax_f1.legend(loc='lower right')

    # Iteration boundaries (only meaningful when ≥2 history files)
    if len(histories) > 1:
        _, _, boundaries = _series(histories, scenarios[0], 'acc')
        for b in boundaries[:-1]:  # skip final boundary (end of data)
            for ax in (ax_acc, ax_f1):
                ax.axvline(b, color='gray', linestyle='--', alpha=0.5,
                           linewidth=1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"Saved plot → {out_path}")


def _autofind_history_files():
    """Walk trained_models/ and find friendly_predictor_history*.json files."""
    found = []
    base = 'trained_models'
    if not os.path.isdir(base):
        return found
    for root, _, files in os.walk(base):
        for f in files:
            if f.startswith('friendly_predictor_history') and f.endswith('.json'):
                found.append(os.path.join(root, f))
            elif f.startswith('friendly_predictor_metrics_dagger') and f.endswith('_history.json'):
                found.append(os.path.join(root, f))
    found.sort()
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('history_files', nargs='*',
                    help='Path(s) to *_history.json file(s). If omitted, auto-discovers under trained_models/.')
    ap.add_argument('--out', default='awareness_training.png',
                    help='Output image path (default: awareness_training.png).')
    ap.add_argument('--target', type=float, default=0.80,
                    help='Target accuracy horizontal line (default: 0.80).')
    args = ap.parse_args()

    paths = args.history_files or _autofind_history_files()
    if not paths:
        print("No history files supplied and none found under trained_models/.\n"
              "Run training first, or pass paths explicitly.")
        return
    print("Plotting from:")
    for p in paths:
        print(f"  {p}")
    plot(paths, args.out, args.target)


if __name__ == '__main__':
    main()
