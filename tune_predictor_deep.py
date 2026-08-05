"""Retrain the predictor on ALL 5 adaptive_gt seeds (exp42, exp1000, ..., exp4000)
with the current production architecture (V0) for 100 epochs.

Why this exists:
  The canonical 24-dim predictor was trained on seed 42 only. Its 88% val
  accuracy is in-distribution (same seed, different episodes), not cross-seed.
  Deploying on seed 1000/2000/... drops accuracy a lot because the model
  overfit to seed-42-specific human spawn patterns. Training on all 5 seeds
  gives 5× more diverse data and should fix cross-seed generalization.

Safety:
  Writes to trained_models/LoraF_invi_visi_rank_1/tune_deep/. Does NOT touch
  the canonical friendly_predictor.pth. To promote, copy manually.

Usage (inside the Docker container):
    python3 tune_predictor_deep.py
    EPOCHS=150 python3 tune_predictor_deep.py
    SEEDS="42 1000" python3 tune_predictor_deep.py   # subset (default = all 5)
"""

import json
import os
import sys
from datetime import datetime

from train_alpha_from_json import train_from_jsons


# ---------------- Config ----------------
MODEL_DIR  = "trained_models/LoraF_invi_visi_rank_1"
TEST_DIR   = os.path.join(MODEL_DIR, "test")
TUNE_DIR   = os.path.join(MODEL_DIR, "tune_deep")
os.makedirs(TUNE_DIR, exist_ok=True)

SCENARIOS = {
    "mixed_5050":    "seperate_mixed_5050",
    "all_aware":     "seperate_all_aware",
    "all_ignorant":  "seperate_all_ignorant",
    "cluster":       "cluster_aware_ignorant",
}
SCENARIO_WEIGHTS = {
    "mixed_5050":    4.0,
    "all_aware":     1.0,
    "all_ignorant":  1.0,
    "cluster":       4.0,
}

EPOCHS = int(os.environ.get('EPOCHS', 100))
SEEDS  = [s for s in os.environ.get('SEEDS', '42 1000 2000 3000 4000').split() if s.strip()]

# V0: current production architecture (matches train_from_jsons defaults).
ARCH_NAME = "V0_baseline_h192_L3_H4_d0.1"
ARCH_KWARGS = dict(
    hidden_dim=192,
    num_heads=4,
    num_layers=3,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
)


def main():
    # Build the multi-seed training spec.
    spec = {}
    print(f"Training set: {len(SEEDS)} seed(s) × {len(SCENARIOS)} scenarios "
          f"= up to {len(SEEDS) * len(SCENARIOS)} anchor files")
    for short, internal in SCENARIOS.items():
        paths = []
        for s in SEEDS:
            p = os.path.join(TEST_DIR, f"{internal}_adaptive_gt_exp{s}.json")
            if os.path.exists(p):
                paths.append(p)
            else:
                print(f"  WARN: missing seed {s} for {short}: {p}")
        if not paths:
            sys.exit(f"No anchors found for scenario {short}")
        spec[short] = paths
        print(f"  [{short}] {len(paths)} files")

    # Output paths (scoped to tune_deep/ to keep canonical safe).
    seeds_tag = "_".join(SEEDS)
    out_pth     = os.path.join(TUNE_DIR, f"{ARCH_NAME}_seeds{seeds_tag}_{EPOCHS}ep.pth")
    metrics_pth = os.path.join(TUNE_DIR, f"{ARCH_NAME}_seeds{seeds_tag}_{EPOCHS}ep_metrics.json")

    print(f"\n{'='*72}\n== {ARCH_NAME}  |  {len(SEEDS)} seeds  |  epochs={EPOCHS}  "
          f"|  [{datetime.now():%Y-%m-%d %H:%M:%S}]\n{'='*72}")
    print(f"  output  : {out_pth}")
    print(f"  metrics : {metrics_pth}")
    print(f"  arch    : {ARCH_KWARGS}")
    print()

    metrics = train_from_jsons(
        scenarios_spec=spec,
        weights_spec=SCENARIO_WEIGHTS,
        output_path=out_pth,
        metrics_path=metrics_pth,
        epochs=EPOCHS,
        init_checkpoint=None,   # fresh; not warm-starting from canonical
        **ARCH_KWARGS,
    )

    # Compare vs the canonical metrics if they exist (might be backed up post-revert).
    print(f"\n{'='*72}\n== RESULT  |  {ARCH_NAME} (5-seed, {EPOCHS}ep) vs canonical (seed-42-only)\n{'='*72}")
    canonical_metrics_path = os.path.join(MODEL_DIR, "friendly_predictor_metrics.json")
    backup_glob = canonical_metrics_path + ".17dim_backup_"
    canonical = None
    # Try the live canonical first, then any backup
    if os.path.exists(canonical_metrics_path):
        try:
            canonical = json.load(open(canonical_metrics_path))
        except Exception:
            pass
    if canonical is None:
        # try the most recent .17dim_backup as a fallback comparison reference
        import glob
        backups = sorted(glob.glob(canonical_metrics_path + ".*"))
        if backups:
            try:
                canonical = json.load(open(backups[-1]))
                print(f"  (comparing to backup: {backups[-1]})")
            except Exception:
                pass

    header = f"{'scenario':<14} {'5-seed acc':>10}   {'canonical':>10}   {'delta':>8}"
    print(header)
    print('-' * len(header))
    for sc in ['mixed_5050', 'cluster', 'all_aware', 'all_ignorant']:
        new = metrics['per_scenario'].get(sc, {}).get('acc', 0.0)
        if canonical:
            old = canonical.get('per_scenario', {}).get(sc, {}).get('acc', 0.0)
            print(f"  {sc:<12} {new:>10.4f}   {old:>10.4f}   {new-old:+8.4f}")
        else:
            print(f"  {sc:<12} {new:>10.4f}   {'n/a':>10}   {'n/a':>8}")
    new_min  = metrics['min_scenario_acc']
    new_mean = metrics['mean_scenario_acc']
    if canonical:
        old_min  = canonical.get('min_scenario_acc', 0.0)
        old_mean = canonical.get('mean_scenario_acc', 0.0)
        print(f"  {'min_scenario':<12} {new_min:>10.4f}   {old_min:>10.4f}   {new_min-old_min:+8.4f}")
        print(f"  {'mean_scenario':<12} {new_mean:>10.4f}   {old_mean:>10.4f}   {new_mean-old_mean:+8.4f}")
    else:
        print(f"  {'min_scenario':<12} {new_min:>10.4f}   {'n/a':>10}   {'n/a':>8}")
        print(f"  {'mean_scenario':<12} {new_mean:>10.4f}   {'n/a':>10}   {'n/a':>8}")

    print(f"\nNew checkpoint : {out_pth}")
    print(f"New metrics    : {metrics_pth}")
    print(f"\nTo promote to canonical (only if it improves cross-seed):")
    print(f"  cp {out_pth} {MODEL_DIR}/friendly_predictor.pth")
    print(f"  cp {metrics_pth} {MODEL_DIR}/friendly_predictor_metrics.json")
    print(f"\nThe canonical was NOT touched by this run.")


if __name__ == '__main__':
    main()
