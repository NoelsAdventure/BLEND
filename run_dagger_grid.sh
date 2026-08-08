#!/bin/bash
#
# Run a grid of DAgger sweeps across HUMAN_NUM × ROLLOUT_BEHAVIOUR. Each
# combination produces its own tagged checkpoint at
#   trained_models/<MODEL>/friendly_predictor_h<HN>_<driver>.pth
# so nothing collides with anything else.
#
# Defaults: HUMAN_NUM ∈ {20, 30, 40} × ROLLOUT ∈ {adaptive_pred, switching_pred}
# = 6 sweeps. With 3 iterations × 200 episodes × 4 scenarios per sweep this
# is roughly 6 × 90 min ≈ 9 hours. Tighten DAGGER_TEST_SIZE or DAGGER_ITERATIONS
# below to make it faster.
#
# Usage (inside the Docker container):
#   ./run_dagger_grid.sh
# or, with a tighter sweep:
#   DAGGER_TEST_SIZE=100 DAGGER_ITERATIONS=2 ./run_dagger_grid.sh
# Override the grid axes by exporting before calling:
#   HUMAN_NUMS="20 30" ROLLOUTS="adaptive_pred" ./run_dagger_grid.sh

set -u  # no unbound vars; do NOT use -e so one failed combo doesn't abort the grid

HUMAN_NUMS="${HUMAN_NUMS:-20 30 40}"
ROLLOUTS="${ROLLOUTS:-adaptive_pred}"
#  switching_pred
# Per-sweep params (env overrides honored — see dagger_loop.py CLI defaults)
export DAGGER_ITERATIONS="${DAGGER_ITERATIONS:-3}"
export DAGGER_TEST_SIZE="${DAGGER_TEST_SIZE:-200}"
export DAGGER_EPOCHS_PER_ITER="${DAGGER_EPOCHS_PER_ITER:-30}"
# Max parallel dagger_loop.py processes. Each spawns its own test.py
# subprocesses (1 at a time inside the loop) + a training run between iters,
# so memory peaks higher than test_baselines. Default 2 is conservative on a
# 24 GB GPU. Set to 1 to force fully sequential, or higher if you have headroom.
MAX_PARALLEL="${MAX_PARALLEL:-2}"

LOG_DIR="${LOG_DIR:-/tmp/dagger_grid}"
mkdir -p "$LOG_DIR"

if [ -f "./gpu_affinity.sh" ]; then
    . ./gpu_affinity.sh
fi
BLEND_GPUS="$(blend_detect_gpus)"

START_TS=$(date +%s)
echo "=========================================================="
echo "DAgger grid starting"
echo "  HUMAN_NUMs        : $HUMAN_NUMS"
echo "  ROLLOUTS          : $ROLLOUTS"
echo "  DAGGER_ITERATIONS : $DAGGER_ITERATIONS"
echo "  DAGGER_TEST_SIZE  : $DAGGER_TEST_SIZE"
echo "  EPOCHS_PER_ITER   : $DAGGER_EPOCHS_PER_ITER"
echo "  MAX_PARALLEL      : $MAX_PARALLEL"
echo "  BLEND_GPUS        : $BLEND_GPUS"
echo "  log dir           : $LOG_DIR"
echo "=========================================================="

mkdir -p "$LOG_DIR/.counters"
: > "$LOG_DIR/.counters/ok"
: > "$LOG_DIR/.counters/fail"

launch_dagger_sweep() {
    local hn="$1" rollout="$2" gpu_id="$3"
    local driver="${rollout%_pred}"
    local tag="h${hn}_${driver}"
    local log_path="$LOG_DIR/dagger_${tag}.log"
    echo "[$(date '+%H:%M:%S')] START  TAG=$tag  GPU=$gpu_id   (log: $log_path)"
    if CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u dagger_loop.py \
        --human_num "$hn" \
        --rollout_behaviour "$rollout" \
        --iterations "$DAGGER_ITERATIONS" \
        --test_size "$DAGGER_TEST_SIZE" \
        --epochs_per_iter "$DAGGER_EPOCHS_PER_ITER" \
        > "$log_path" 2>&1; then
        echo x >> "$LOG_DIR/.counters/ok"
        echo "[$(date '+%H:%M:%S')] OK     TAG=$tag"
    else
        echo x >> "$LOG_DIR/.counters/fail"
        echo "[$(date '+%H:%M:%S')] FAIL   TAG=$tag   (see $log_path)"
    fi
}

running=0
n_total=0
job_index=0
for hn in $HUMAN_NUMS; do
    for rollout in $ROLLOUTS; do
        n_total=$((n_total + 1))
        gpu_id="$(blend_gpu_for_job "$job_index")"
        launch_dagger_sweep "$hn" "$rollout" "$gpu_id" &
        job_index=$((job_index + 1))
        running=$((running + 1))
        if (( running >= MAX_PARALLEL )); then
            wait -n
            running=$((running - 1))
        fi
    done
done
wait  # drain the rest

n_ok=$(wc -l < "$LOG_DIR/.counters/ok")
n_fail=$(wc -l < "$LOG_DIR/.counters/fail")

END_TS=$(date +%s)
elapsed=$(( END_TS - START_TS ))
echo
echo "=========================================================="
echo "DAgger grid done in $((elapsed/60))m $((elapsed%60))s"
echo "  combos tried : $n_total"
echo "  succeeded    : $n_ok"
echo "  failed       : $n_fail"
echo "=========================================================="
echo
echo "Per-combo final per-scenario val accuracies (from each metrics sidecar):"
python3 - <<EOF
import json, os, glob
mdir = "trained_models/LoraF_invi_visi_rank_1"
for path in sorted(glob.glob(os.path.join(mdir, "friendly_predictor_metrics_h*.json"))):
    name = os.path.basename(path).replace("friendly_predictor_metrics_", "").replace(".json", "")
    if "iter" in name:  # skip per-iter snapshots; only final per-tag
        continue
    try:
        m = json.load(open(path))
    except Exception as e:
        print(f"  {name:<24} <error loading: {e}>"); continue
    per_sc = m.get("per_scenario", {})
    table = " ".join(f"{sc}={per_sc[sc]['acc']:.3f}" for sc in per_sc)
    print(f"  TAG={name:<24} min={m.get('min_scenario_acc', 0):.3f} mean={m.get('mean_scenario_acc', 0):.3f}  ({table})")
EOF
