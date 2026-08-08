#!/bin/bash
#
# Fixed-kappa adaptive MPC clearance ablation.
#
# Sweeps k_t in:
#   d_min = 1 - k_t
#
# Usage:
#   ./test_mpc_ablation.sh
#   SCALES="0.0 0.5 1.0" ./test_mpc_ablation.sh
#   SCENARIOS="seperate_mixed_5050" TEST_SIZE=50 SEEDS="42" ./test_mpc_ablation.sh
#
# Output keys in ${MPC_MODEL_DIR}/test/all_evaluations.json:
#   {scenario}_mpc_fixed_expmpc_scale_{kappa}_{seed}

set -u

TEST_SIZE="${TEST_SIZE:-250}"
HUMAN_NUM="${HUMAN_NUM:-20}"
SCENARIOS="${SCENARIOS:-seperate_all_ignorant seperate_all_aware seperate_mixed_5050 cluster_aware_ignorant}"
SCALES="${SCALES:-0.0 0.2 0.4 0.6 0.8 1.0}"
SEEDS="${SEEDS:-42}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
AWARENESS_EVAL="${AWARENESS_EVAL:-off}"
SAVE_EPISODE_DUMP="${SAVE_EPISODE_DUMP:-never}"

MPC_MODEL_DIR="${MPC_MODEL_DIR:-trained_models/LoraF_invi_visi_rank_1}"
MPC_CHECKPOINT="${MPC_CHECKPOINT:-03400.pt}"
MPC_HUMAN_RADIUS="${MPC_HUMAN_RADIUS:-0.5}"
MPC_AMAX="${MPC_AMAX:-2.0}"

EXTRA_EXP_ID="${EXP_ID:-}"
_remaining_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --exp|--exp_id|--exp-id) EXTRA_EXP_ID="$2"; shift 2 ;;
        --exp=*|--exp_id=*|--exp-id=*) EXTRA_EXP_ID="${1#*=}"; shift ;;
        *) _remaining_args+=("$1"); shift ;;
    esac
done
SCRIPT_PASSTHRU_ARGS=("${_remaining_args[@]}")

LOG_DIR="${LOG_DIR:-/tmp/mpc_ablation}"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/.counters"
: > "$LOG_DIR/.counters/ok"
: > "$LOG_DIR/.counters/fail"

if [ -f "./gpu_affinity.sh" ]; then
    . ./gpu_affinity.sh
fi
BLEND_GPUS="$(blend_detect_gpus)"

START_TS=$(date +%s)
echo "=========================================================="
echo "MPC fixed-kappa clearance ablation starting"
echo "  MODEL           : $MPC_MODEL_DIR/checkpoints/$MPC_CHECKPOINT"
echo "  HUMAN_NUM       : $HUMAN_NUM"
echo "  TEST_SIZE       : $TEST_SIZE"
echo "  K_T SCALES      : $SCALES"
echo "  SEEDS           : $SEEDS"
echo "  SCENARIOS       : $SCENARIOS"
echo "  MPC_HUMAN_RADIUS: $MPC_HUMAN_RADIUS (ignored; d_min=1-k_t)"
echo "  MPC_AMAX        : $MPC_AMAX"
echo "  MAX_PARALLEL    : $MAX_PARALLEL"
echo "  BLEND_GPUS      : $BLEND_GPUS"
echo "  log dir         : $LOG_DIR"
[ -n "$EXTRA_EXP_ID" ] && echo "  EXTRA EXP_ID    : $EXTRA_EXP_ID"
echo "=========================================================="

if [ ! -f "$MPC_MODEL_DIR/checkpoints/$MPC_CHECKPOINT" ]; then
    echo "Missing MPC checkpoint holder: $MPC_MODEL_DIR/checkpoints/$MPC_CHECKPOINT" >&2
    exit 1
fi

launch_combo() {
    local scenario="$1" scale="$2" seed="$3" gpu_id="$4"
    local exp_id="mpc_scale_${scale}_${seed}"
    [ -n "$EXTRA_EXP_ID" ] && exp_id="${exp_id}_${EXTRA_EXP_ID}"
    local log_path="$LOG_DIR/${scenario}_mpc_scale_${scale}_seed${seed}.log"
    [ -n "$EXTRA_EXP_ID" ] && log_path="$LOG_DIR/${scenario}_mpc_scale_${scale}_seed${seed}_${EXTRA_EXP_ID}.log"

    echo "[$(date '+%H:%M:%S')] START  $scenario x MPC k_t=$scale seed=$seed GPU=$gpu_id  (log: $log_path)"
    if CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u test.py \
            --model_dir "$MPC_MODEL_DIR" \
            --test_model "$MPC_CHECKPOINT" \
            --adaptive_lora_scenario "$scenario" \
            --lora_behaviour mpc_fixed \
            --lora_scale "$scale" \
            --human_num "$HUMAN_NUM" \
            --test_size "$TEST_SIZE" \
            --awareness_eval "$AWARENESS_EVAL" \
            --save_episode_dump "$SAVE_EPISODE_DUMP" \
            --mpc_human_radius "$MPC_HUMAN_RADIUS" \
            --mpc_amax "$MPC_AMAX" \
            --seed "$seed" \
            --exp_id "$exp_id" \
            "${SCRIPT_PASSTHRU_ARGS[@]}" \
            > "$log_path" 2>&1; then
        echo x >> "$LOG_DIR/.counters/ok"
        echo "[$(date '+%H:%M:%S')] OK     $scenario x MPC k_t=$scale seed=$seed"
    else
        echo x >> "$LOG_DIR/.counters/fail"
        echo "[$(date '+%H:%M:%S')] FAIL   $scenario x MPC k_t=$scale seed=$seed  (see $log_path)"
        echo "--- last 40 log lines: $log_path ---"
        tail -40 "$log_path" 2>/dev/null || true
        echo "--- end log tail ---"
    fi
}

running=0
job_index=0
for seed in $SEEDS; do
    for scenario in $SCENARIOS; do
        for scale in $SCALES; do
            gpu_id="$(blend_gpu_for_job "$job_index")"
            launch_combo "$scenario" "$scale" "$seed" "$gpu_id" &
            job_index=$((job_index + 1))
            running=$((running + 1))
            if (( running >= MAX_PARALLEL )); then
                wait -n
                running=$((running - 1))
            fi
        done
    done
done
wait

n_ok=$(wc -l < "$LOG_DIR/.counters/ok")
n_fail=$(wc -l < "$LOG_DIR/.counters/fail")
n_total=$((n_ok + n_fail))
END_TS=$(date +%s)
elapsed=$((END_TS - START_TS))

echo
echo "=========================================================="
echo "MPC ablation done in $((elapsed/60))m $((elapsed%60))s"
echo "  combos tried : $n_total"
echo "  succeeded    : $n_ok"
echo "  failed       : $n_fail"
echo "=========================================================="

SCALES_STR="$SCALES"
SCENARIOS_STR="$SCENARIOS"
SEEDS_STR="$SEEDS"
EXP_SUFFIX_EXTRA=""
[ -n "$EXTRA_EXP_ID" ] && EXP_SUFFIX_EXTRA="_${EXTRA_EXP_ID}"

python3 - <<EOF
import json, os, math

def wilson_half(p, n, z=1.96):
    if n is None or n <= 0: return float('nan')
    denom = 1.0 + z*z/n
    return (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom

def se_half(std, n, z=1.96):
    if std is None or n is None or n <= 1: return float('nan')
    return z * float(std) / math.sqrt(n)

agg_path = os.path.join("$MPC_MODEL_DIR", "test", "all_evaluations.json")
if not os.path.exists(agg_path):
    print(f"No aggregate found at {agg_path}")
    raise SystemExit
with open(agg_path) as f:
    aggregate = json.load(f)

scales = "$SCALES_STR".split()
scenarios = "$SCENARIOS_STR".split()
seeds = "$SEEDS_STR".split()
extra_suffix = "$EXP_SUFFIX_EXTRA"
print()
print(f"Per-combo summary from {agg_path}:")
print(f"  {'scenario':<32s} {'k_t':>6s} {'seed':>6s}  {'SR +- CI':>14s}  {'PL +- SE':>14s}  {'MPC ms':>8s}  {'fail':>5s}    N")
print("  " + "-"*104)
for sc in scenarios:
    for scale in scales:
        for seed in seeds:
            key = f"{sc}_mpc_fixed_expmpc_scale_{scale}_{seed}{extra_suffix}"
            ent = aggregate.get(key)
            if ent is None:
                print(f"  {sc:<32s} {scale:>6s} {seed:>6s}  (missing)")
                continue
            s = ent.get("summary", {})
            n = s.get("num_episodes", 0) or 0
            sr = s.get("success_rate"); pl = s.get("avg_path_length")
            pl_std = s.get("std_path_length")
            mpc_ms = s.get("avg_mpc_solve_time_ms")
            fail = s.get("mpc_failure_count")
            sr_str = f"{sr:.3f} +- {wilson_half(sr, n):.3f}" if sr is not None else "-"
            pl_str = f"{pl:.2f} +- {se_half(pl_std, n):.2f}" if pl_std is not None else (f"{pl:.2f}" if pl is not None else "-")
            mpc_str = f"{mpc_ms:.2f}" if mpc_ms is not None else "-"
            fail_str = str(fail) if fail is not None else "-"
            print(f"  {sc:<32s} {scale:>6s} {seed:>6s}  {sr_str:>14s}  {pl_str:>14s}  {mpc_str:>8s}  {fail_str:>5s}    {n}")
        print()
EOF
