"""Sweep FriendlyPredictor architectures against the same bootstrap data.

Trains each variant on exp42 adaptive_gt anchor JSONs (no DAgger, no on-policy
data) — exactly what dagger_loop's bootstrap step does — and reports
val accuracy per scenario for each variant.

The number we care about is `mixed_5050` val accuracy: that's the only
scenario with balanced labels, so it's the only one that meaningfully tests
discrimination ability. (all_aware / all_ignorant are trivially solvable by
guessing the majority class; cluster is intermediate.)

Goal: pick the best-bootstrapping variant. That sets the ceiling DAgger
can converge toward at deploy time.

Usage (inside the Docker container):
    python3 sweep_predictor_arch.py
    EPOCHS=50 python3 sweep_predictor_arch.py   # longer training per variant
"""

import json
import os
import sys
from datetime import datetime

from train_alpha_from_json import train_from_jsons


# ---------------- Config ----------------
MODEL_DIR = "trained_models/LoraF_invi_visi_rank_1"
TEST_DIR = os.path.join(MODEL_DIR, "test")
SWEEP_DIR = os.path.join(MODEL_DIR, "arch_sweep")
os.makedirs(SWEEP_DIR, exist_ok=True)

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

EPOCHS = int(os.environ.get('EPOCHS', 30))

# Bootstrap spec: exactly what dagger_loop._bootstrap_from_exp42() uses.
spec = {}
for short, internal in SCENARIOS.items():
    p = os.path.join(TEST_DIR, f"{internal}_adaptive_gt_exp42.json")
    if not os.path.exists(p):
        sys.exit(f"Missing anchor: {p}. Run adaptive_gt rollouts with --seed 42 first.")
    spec[short] = [p]


# ---------------- Variants ----------------
# (name, kwargs for train_from_jsons). Keep input dim fixed (matches policy);
# only architecture knobs vary.
VARIANTS = [
    # V0: current production config (matches dagger_loop's defaults).
    ("V0_baseline_h192_L3_H4_d0.1",
     dict(hidden_dim=192, num_heads=4, num_layers=3, dropout=0.1, lr=1e-3)),

    # V1: scaled-up capacity. More layers + more heads + wider.
    ("V1_big_h256_L6_H8_d0.1",
     dict(hidden_dim=256, num_heads=8, num_layers=6, dropout=0.1, lr=1e-3)),

    # V2: deeper but narrower (test whether depth alone helps).
    ("V2_deep_h128_L6_H4_d0.1",
     dict(hidden_dim=128, num_heads=4, num_layers=6, dropout=0.1, lr=1e-3)),

    # V3: heavier regularization at baseline capacity. If V0 is overfitting,
    # this should beat it on val. If V0 is underfitting, this should lose.
    ("V3_reg_h192_L3_H4_d0.3",
     dict(hidden_dim=192, num_heads=4, num_layers=3, dropout=0.3, lr=1e-3,
          weight_decay=1e-3)),

    # V4: lower LR, longer effective training (decouples optimization from arch).
    ("V4_slowlr_h192_L3_H4_d0.1_lr3e-4",
     dict(hidden_dim=192, num_heads=4, num_layers=3, dropout=0.1, lr=3e-4)),

    # V5: big + heavy reg — if V1 helps but overfits, V5 should beat both.
    ("V5_big_reg_h256_L6_H8_d0.3",
     dict(hidden_dim=256, num_heads=8, num_layers=6, dropout=0.3, lr=1e-3,
          weight_decay=1e-3)),
]


def main():
    results = []
    print(f"\n{'='*72}\n== Sweep starting  |  epochs/variant={EPOCHS}  "
          f"|  [{datetime.now():%Y-%m-%d %H:%M:%S}]\n{'='*72}\n")

    for idx, (name, kwargs) in enumerate(VARIANTS, start=1):
        print(f"\n{'#'*72}")
        print(f"# [{idx}/{len(VARIANTS)}] {name}")
        print(f"#   kwargs: {kwargs}")
        print(f"{'#'*72}")
        out_pth      = os.path.join(SWEEP_DIR, f"{name}.pth")
        metrics_pth  = os.path.join(SWEEP_DIR, f"{name}_metrics.json")

        try:
            metrics = train_from_jsons(
                scenarios_spec=spec,
                weights_spec=SCENARIO_WEIGHTS,
                output_path=out_pth,
                metrics_path=metrics_pth,
                epochs=EPOCHS,
                init_checkpoint=None,  # fresh train each variant
                **kwargs,
            )
            results.append((name, metrics, kwargs, None))
        except Exception as e:
            print(f"!! variant {name} FAILED: {e}")
            results.append((name, None, kwargs, str(e)))

    # ---------------- Comparison table ----------------
    print(f"\n\n{'='*72}\n== SWEEP RESULTS  "
          f"(val acc per scenario, mixed_5050 is the headline)\n{'='*72}")
    header = f"{'variant':<38} {'mixed':>7} {'cluster':>8} {'aware':>7} {'ignorant':>9} {'min':>7} {'mean':>7}"
    print(header)
    print('-' * len(header))

    rows = []
    for name, m, kwargs, err in results:
        if err:
            print(f"  {name:<36} ERROR: {err[:80]}")
            continue
        mix = m['per_scenario'].get('mixed_5050',   {}).get('acc', 0.0)
        clu = m['per_scenario'].get('cluster',      {}).get('acc', 0.0)
        awr = m['per_scenario'].get('all_aware',    {}).get('acc', 0.0)
        ign = m['per_scenario'].get('all_ignorant', {}).get('acc', 0.0)
        mn  = m.get('min_scenario_acc', 0.0)
        mean= m.get('mean_scenario_acc', 0.0)
        rows.append((name, mix, clu, awr, ign, mn, mean))
        print(f"  {name:<36} {mix:>7.3f} {clu:>8.3f} {awr:>7.3f} {ign:>9.3f} {mn:>7.3f} {mean:>7.3f}")

    if rows:
        best_by_mixed = max(rows, key=lambda r: r[1])
        best_by_min   = max(rows, key=lambda r: r[5])
        print(f"\nBest by mixed_5050 acc: {best_by_mixed[0]}  →  mixed={best_by_mixed[1]:.3f}")
        print(f"Best by min-scenario  : {best_by_min[0]}  →  min={best_by_min[5]:.3f}")
        print(f"\nCheckpoints + metrics saved under: {SWEEP_DIR}")

    # Dump combined comparison to JSON for downstream plotting.
    summary_path = os.path.join(SWEEP_DIR, "sweep_summary.json")
    with open(summary_path, 'w') as f:
        json.dump([
            {'name': name, 'kwargs': kwargs, 'metrics': m, 'error': err}
            for name, m, kwargs, err in results
        ], f, indent=2)
    print(f"Combined summary: {summary_path}")


if __name__ == '__main__':
    main()
