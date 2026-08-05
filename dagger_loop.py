"""DAgger-style iterative bootstrapping of the FriendlyPredictor.

Closes the loop between policy and predictor. Each iteration:

  1. Roll out `adaptive_pred` on every training scenario with the *current*
     predictor (the one at MODEL_DIR/friendly_predictor.pth). This generates
     fresh JSON dumps that reflect the predictor's deployment trajectories,
     not the original `adaptive_gt` rollouts the predictor was trained on.

  2. Append the new dumps to the training spec alongside the original
     `adaptive_gt` "anchor" files. The anchor files keep the predictor honest
     against ground-truth-driven dynamics; the new dumps teach it about its
     own deployment distribution.

  3. Retrain the predictor (warm-started from the current checkpoint) on the
     expanded spec, overwriting MODEL_DIR/friendly_predictor.pth. The next
     iteration's rollouts therefore use the improved predictor.

  4. Archive the per-iteration checkpoint + metrics for inspection.

Usage (inside the gen_safe_py10 container):

    docker exec -it great_chaum bash -lc \\
        'cd /workspace && python3 dagger_loop.py'

Stop early with Ctrl-C; the most recent best-by-min-scenario-acc checkpoint
is always saved to the canonical predictor path.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

from train_alpha_from_json import train_from_jsons


# --- Config (defaults; overridable via CLI / env so run_dagger_grid.sh works) ---

MODEL_DIR = "trained_models/LoraF_invi_visi_rank_1"
CHECKPOINT = "03400.pt"  # policy checkpoint used by test.py

# Tag baked into every output path so concurrent DAgger sweeps for
# different (HUMAN_NUM, driver) combos don't clobber each other.
# Suggested convention: f"h{HUMAN_NUM}_{ROLLOUT_BEHAVIOUR}".
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--human_num', type=int,
                 default=int(os.environ.get('DAGGER_HUMAN_NUM', 20)))
_ap.add_argument('--rollout_behaviour', type=str,
                 default=os.environ.get('DAGGER_ROLLOUT_BEHAVIOUR', 'adaptive_pred'),
                 choices=['adaptive_pred', 'switching_pred'])
_ap.add_argument('--iterations', type=int,
                 default=int(os.environ.get('DAGGER_ITERATIONS', 3)))
_ap.add_argument('--test_size', type=int,
                 default=int(os.environ.get('DAGGER_TEST_SIZE', 200)))
_ap.add_argument('--epochs_per_iter', type=int,
                 default=int(os.environ.get('DAGGER_EPOCHS_PER_ITER', 30)))
# Per-iteration seeds. Default mode = one fresh random seed per iteration
# (picked via SystemRandom, so different every invocation). For
# reproducibility, override with --seeds "S1 S2 S3" (must have exactly
# DAGGER_ITERATIONS entries).
_ap.add_argument('--seeds', type=str,
                 default=os.environ.get('DAGGER_SEEDS', ''),
                 help='Space-separated seeds, one per iteration. Empty = random.')
_ap.add_argument('--skip_bootstrap', action='store_true',
                 default=os.environ.get('DAGGER_SKIP_BOOTSTRAP', '0') == '1',
                 help='Skip the auto-train-on-exp42 step. Use the existing canonical predictor as-is.')
_args, _ = _ap.parse_known_args()

ROLLOUT_BEHAVIOUR = _args.rollout_behaviour
HUMAN_NUM = _args.human_num
TAG = f"h{HUMAN_NUM}_{ROLLOUT_BEHAVIOUR.replace('_pred', '')}"
# e.g. h20_adaptive, h30_switching

TEST_DIR = os.path.join(MODEL_DIR, "test")
PREDICTOR_PATH = os.path.join(MODEL_DIR, f"friendly_predictor_{TAG}.pth")
METRICS_PATH = os.path.join(MODEL_DIR, f"friendly_predictor_metrics_{TAG}.json")
# Where the canonical (untagged) predictor lives — used only to seed iter 1
# warm-start when no TAGGED checkpoint exists yet.
CANONICAL_PREDICTOR_PATH = os.path.join(MODEL_DIR, "friendly_predictor.pth")

# Scenarios to roll out + train on.
# Key = short name used in scenarios_spec / SCENARIO_WEIGHTS.
# Value = internal scenario name passed to test.py --adaptive_lora_scenario.
SCENARIOS = {
    "mixed_5050":    "seperate_mixed_5050",
    "all_aware":     "seperate_all_aware",
    "all_ignorant":  "seperate_all_ignorant",
    "cluster":       "cluster_aware_ignorant",
}

# Original adaptive_gt files per scenario (always included as anchors).
# 5-seed adaptive_gt anchor set per scenario. Pulled at iter retraining time
# so the on-policy DAgger dumps don't drown out off-policy ground truth.
# Override via env var if you need a subset:
#   ANCHOR_SEEDS="42 1000" python dagger_loop.py
_ANCHOR_SEEDS = [s.strip() for s in os.environ.get('ANCHOR_SEEDS', '42 1000 2000 3000 4000').split() if s.strip()]
_SC_TO_INTERNAL = {
    "mixed_5050":    "seperate_mixed_5050",
    "all_aware":     "seperate_all_aware",
    "all_ignorant":  "seperate_all_ignorant",
    "cluster":       "cluster_aware_ignorant",
}
ANCHOR_FILES = {
    short: [f"{internal}_adaptive_gt_exp{s}.json" for s in _ANCHOR_SEEDS]
    for short, internal in _SC_TO_INTERNAL.items()
}

# Sampling weights — Mixed (and Cluster when enabled) dominate so the
# per-human discrimination signal isn't drowned by easy degenerate scenarios.
SCENARIO_WEIGHTS = {
    "mixed_5050":    4.0,
    "all_aware":     1.0,
    "all_ignorant":  1.0,
    "cluster":       4.0,
}

# DAgger sweep params (all CLI/env-overridable above).
DAGGER_ITERATIONS = _args.iterations
TEST_SIZE = _args.test_size
EPOCHS_PER_ITER = _args.epochs_per_iter

# Resolve per-iteration seeds. Each iteration gets ONE seed (used for all
# scenarios in that iter). Length must match DAGGER_ITERATIONS.
_explicit_seeds = [s.strip() for s in _args.seeds.split() if s.strip()]
if _explicit_seeds:
    if len(_explicit_seeds) != DAGGER_ITERATIONS:
        sys.exit(
            f"--seeds has {len(_explicit_seeds)} entries but --iterations is "
            f"{DAGGER_ITERATIONS}. They must match."
        )
    SEEDS_PER_ITER = _explicit_seeds
    _seed_source = "explicit (--seeds)"
else:
    # Fresh random seed per iteration. SystemRandom = OS entropy, never collides
    # with previously-archived dumps.
    _rng = random.SystemRandom()
    SEEDS_PER_ITER = [str(_rng.randint(1, 1_000_000)) for _ in range(DAGGER_ITERATIONS)]
    _seed_source = "random (SystemRandom)"

print(f"[dagger_loop] seeds per iteration ({_seed_source}): {SEEDS_PER_ITER}")

# --------------------------------------------------------------------------


def _scenario_dump_path(scenario_internal: str, iter_idx: int, seed: str) -> str:
    """File rl/evaluation.py will write for the rollout + this exp_id.
    Encodes TAG + seed so concurrent sweeps and per-seed dumps don't collide."""
    exp_id_suffix = f"dagger_{TAG}_iter{iter_idx}_seed{seed}"
    return os.path.join(
        TEST_DIR, f"{scenario_internal}_{ROLLOUT_BEHAVIOUR}_exp{exp_id_suffix}.json"
    )


def _run_test_dump(scenario_internal: str, iter_idx: int, seed: str) -> str:
    """Run the configured rollout behaviour for one (scenario, seed); return dump path."""
    expected_path = _scenario_dump_path(scenario_internal, iter_idx, seed)
    cmd = [
        "python3", "test.py",
        "--model_dir", MODEL_DIR,
        "--test_model", CHECKPOINT,
        "--adaptive_lora_scenario", scenario_internal,
        "--lora_behaviour", ROLLOUT_BEHAVIOUR,
        "--exp_id", f"dagger_{TAG}_iter{iter_idx}_seed{seed}",
        "--predictor_tag", TAG,
        "--human_num", str(HUMAN_NUM),
        "--test_size", str(TEST_SIZE),
        "--awareness_eval", "always",
        "--seed", str(seed),
    ]
    print(f"\n>>> [dagger iter {iter_idx} / TAG={TAG} / seed={seed}] {scenario_internal}")
    print("    " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not os.path.exists(expected_path):
        raise FileNotFoundError(
            f"Expected dump at {expected_path} but it doesn't exist. "
            f"Check test.py / evaluation.py exp_id naming."
        )
    return expected_path


def _bootstrap_from_anchors():
    """Auto-train the initial predictor on the configured anchor seed set
    (default: exp42, exp1000, exp2000, exp3000, exp4000) for each scenario.

    Writes both the tagged predictor (used by this DAgger run) AND the
    canonical predictor (so future test.py invocations pick it up too).
    Skipped if --skip_bootstrap is set.
    """
    initial_spec = {}
    for sc_short, sc_internal in SCENARIOS.items():
        paths = []
        for s in _ANCHOR_SEEDS:
            p = os.path.join(TEST_DIR, f"{sc_internal}_adaptive_gt_exp{s}.json")
            if os.path.exists(p):
                paths.append(p)
            else:
                print(f"  WARN: missing anchor seed {s} for '{sc_short}': {p}")
        if not paths:
            sys.exit(
                f"Bootstrap aborted: no anchor files found for '{sc_short}' "
                f"in seed set {_ANCHOR_SEEDS}.\n"
                f"Generate them by running test.py with --lora_behaviour adaptive_gt "
                f"--seed <s> on the four scenarios, or pass --skip_bootstrap if you "
                f"already have a usable canonical predictor."
            )
        initial_spec[sc_short] = paths

    canonical_metrics = os.path.join(MODEL_DIR, "friendly_predictor_metrics.json")
    print(f"\n{'=' * 70}\n== Bootstrap: training initial predictor on exp42 adaptive_gt dumps ==\n{'=' * 70}")
    for sc_short, files in initial_spec.items():
        print(f"  [{sc_short}] {[os.path.basename(p) for p in files]}")
    print(f"  → output_path : {PREDICTOR_PATH}")
    print(f"  → metrics_path: {METRICS_PATH}")
    train_from_jsons(
        scenarios_spec=initial_spec,
        weights_spec=SCENARIO_WEIGHTS,
        output_path=PREDICTOR_PATH,
        metrics_path=METRICS_PATH,
        epochs=EPOCHS_PER_ITER,
        init_checkpoint=None,  # fresh train, no warm-start
    )
    # Mirror to canonical paths so downstream scripts (test.py / test_adaptive_lora_poc.sh)
    # pick up the same predictor without needing --predictor_tag.
    shutil.copy(PREDICTOR_PATH, CANONICAL_PREDICTOR_PATH)
    shutil.copy(METRICS_PATH, canonical_metrics)
    print(f"\nBootstrap complete. Predictor at:\n  {PREDICTOR_PATH}\n  {CANONICAL_PREDICTOR_PATH}")


def main():
    if not _args.skip_bootstrap:
        _bootstrap_from_anchors()
    else:
        # User wants to reuse an existing predictor. Mirror the original
        # safety: tagged checkpoint seeded from canonical if missing.
        if not os.path.exists(PREDICTOR_PATH):
            if not os.path.exists(CANONICAL_PREDICTOR_PATH):
                sys.exit(
                    f"--skip_bootstrap set but neither {PREDICTOR_PATH} nor "
                    f"{CANONICAL_PREDICTOR_PATH} exists. Remove --skip_bootstrap "
                    f"to auto-train, or run train_alpha_predictor.py manually."
                )
            print(f"Seeding tagged predictor from canonical:\n  {CANONICAL_PREDICTOR_PATH}\n  → {PREDICTOR_PATH}")
            shutil.copy(CANONICAL_PREDICTOR_PATH, PREDICTOR_PATH)
            canonical_metrics = os.path.join(MODEL_DIR, "friendly_predictor_metrics.json")
            if os.path.exists(canonical_metrics) and not os.path.exists(METRICS_PATH):
                shutil.copy(canonical_metrics, METRICS_PATH)

    # Archive the iter-0 (pre-DAgger) starting checkpoint for this tag.
    start_archive = os.path.join(MODEL_DIR, f"friendly_predictor_{TAG}_iter0_start.pth")
    if not os.path.exists(start_archive):
        shutil.copy(PREDICTOR_PATH, start_archive)
        print(f"Archived starting checkpoint to {start_archive}")

    accumulated_pred_dumps = {sc: [] for sc in SCENARIOS}

    # Sidecar log so the random seeds are recoverable for paper / reproduction.
    seed_log_path = os.path.join(MODEL_DIR, f"friendly_predictor_{TAG}_seeds.json")
    with open(seed_log_path, 'w') as f:
        json.dump({
            'tag': TAG,
            'iterations': DAGGER_ITERATIONS,
            'seeds_per_iter': SEEDS_PER_ITER,
            'seed_source': _seed_source,
            'created': datetime.now().isoformat(timespec='seconds'),
        }, f, indent=2)
    print(f"Per-iter seeds logged to {seed_log_path}")

    for it, seed_for_iter in enumerate(SEEDS_PER_ITER, start=1):
        print(f"\n{'=' * 70}\n== DAgger iteration {it}/{DAGGER_ITERATIONS}  TAG={TAG}  seed={seed_for_iter}  "
              f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ==\n{'=' * 70}")

        # 1. Roll out the configured behaviour with the current tagged predictor.
        #    All scenarios in this iter share the same per-iter seed; across
        #    iterations the seed changes (random by default).
        for sc_short, sc_internal in SCENARIOS.items():
            dump_path = _run_test_dump(sc_internal, iter_idx=it, seed=seed_for_iter)
            accumulated_pred_dumps[sc_short].append(dump_path)

        # 2. Build the training spec: anchors + accumulated dumps so far.
        scenarios_spec = {}
        for sc_short in SCENARIOS:
            anchors = [os.path.join(TEST_DIR, f) for f in ANCHOR_FILES.get(sc_short, [])]
            scenarios_spec[sc_short] = anchors + accumulated_pred_dumps[sc_short]
            files_summary = ", ".join(os.path.basename(p) for p in scenarios_spec[sc_short])
            print(f"  [{sc_short}] training files: {files_summary}")

        # 3. Retrain, warm-started from the current tagged predictor. Saves
        #    back to the tagged path so we never touch the canonical file.
        per_iter_metrics_path = os.path.join(
            MODEL_DIR, f"friendly_predictor_metrics_{TAG}_iter{it}.json"
        )
        print(f"\n>>> Retraining (epochs={EPOCHS_PER_ITER}, "
              f"init_checkpoint={PREDICTOR_PATH})")
        train_from_jsons(
            scenarios_spec=scenarios_spec,
            weights_spec=SCENARIO_WEIGHTS,
            output_path=PREDICTOR_PATH,
            metrics_path=per_iter_metrics_path,
            epochs=EPOCHS_PER_ITER,
            init_checkpoint=PREDICTOR_PATH,
        )

        # 4. Archive the iter's outputs (still TAG-scoped).
        archive_pth = os.path.join(MODEL_DIR, f"friendly_predictor_{TAG}_iter{it}.pth")
        shutil.copy(PREDICTOR_PATH, archive_pth)
        shutil.copy(per_iter_metrics_path, METRICS_PATH)
        iter_history_path = per_iter_metrics_path.replace('_metrics', '_history')
        if os.path.exists(iter_history_path):
            print(f"                 History at  {iter_history_path}")
        print(f"\nIter {it} complete. Checkpoint archived to {archive_pth}\n"
              f"                 Metrics archived to {per_iter_metrics_path}")

    print(f"\n{'=' * 70}\nDAgger loop complete. Final TAG={TAG} checkpoint at:\n  {PREDICTOR_PATH}")
    print(f"Canonical friendly_predictor.pth was NOT touched.\n"
          f"To activate this tag for `./test_adaptive_lora_poc.sh` runs, either:\n"
          f"  (a) pass --predictor_tag {TAG} (recommended; safe and per-run)\n"
          f"  (b) cp {os.path.basename(PREDICTOR_PATH)} friendly_predictor.pth "
          f"+ cp {os.path.basename(METRICS_PATH)} friendly_predictor_metrics.json")


if __name__ == "__main__":
    main()
