#!/bin/bash

# test have different policy -> distribution shift:
# retrain on the new policy or use CP to capture those distribution shift uncertainty

# Proof of Concept: Adaptive LoRA switching online
# Compares the adaptive model against static baselines (always_on, always_off)

ADAPTIVE_MODEL="trained_models/LoraF_invi_visi_rank_1"
# ADAPTIVE_MODEL="trained_models/LoraF_invi_visi_rank_1"
# ADAPTIVE_MODEL="trained_models/Fulltune_uni_invi"  # uncomment to target the
# fulltune baseline (no friendly_predictor.pth lives there, so _pred behaviours
# will FileNotFound).
CHECKPOINT="03400.pt"
# CHECKPOINT="03400.pt"
TEST_SIZE="${TEST_SIZE:-250}"
DISCREPANCY_THRESHOLD=0.15
DISCREPANCY_M=1
HUMAN_NUM="${HUMAN_NUM:-20}"
# Random seed(s) for the env / episode generation. SEED is kept for one-off
# backwards-compatible runs; use SEEDS for a multi-seed sweep.
#   SEED=123 ./test_adaptive_lora_poc.sh --visualize
#   SEEDS="42 1000 2000 3000 4000" ./test_adaptive_lora_poc.sh
SEEDS="${SEEDS:-${SEED:-42 1000 2000 3000 4000}}"
read -r -a SEEDS_ARR <<< "$SEEDS"
# Shadow predictor scoring against ground truth:
#   always    — every behaviour (default; lets adaptive_gt etc. report AwAcc)
#   pred_only — only when behaviour is switching_pred / adaptive_pred
#   off       — never run the shadow eval
AWARENESS_EVAL="${AWARENESS_EVAL:-off}"
# Optional predictor tag — selects friendly_predictor_${PREDICTOR_TAG}.pth
# instead of the canonical friendly_predictor.pth. Examples:
#   PREDICTOR_TAG=""                 # canonical (default)
#   PREDICTOR_TAG="h20_adaptive"     # DAgger sweep that ran at HUMAN_NUM=20 with adaptive_pred
#   PREDICTOR_TAG="h30_switching"    # DAgger sweep that ran at HUMAN_NUM=30 with switching_pred
# If the tagged .pth doesn't exist, that test run is skipped cleanly so the
# sweep can iterate multiple tags without aborting on the first miss.
PREDICTOR_TAG=""
# Max parallel test.py / visualize.py processes. ~5 GB per neural process;
# default to 4 on a 24 GB GPU. Set to 1 to force sequential (useful with
# --visualize which writes shared render output you don't want races over).
MAX_PARALLEL="${MAX_PARALLEL:-8}"
# Exp ID suffix — passed to test.py as --exp_id, which appends "_exp<EXP_ID>"
# to every output filename (per-episode JSON dump, all_evaluations.json key,
# render dir). Use this to *avoid overwriting* previous results when re-running:
#   EXP_ID=$(date +%Y%m%d_%H%M) ./test_adaptive_lora_poc.sh
#   EXP_ID=seed7  SEED=7 ./test_adaptive_lora_poc.sh
#   EXP_ID=hn30   HUMAN_NUM=30 ./test_adaptive_lora_poc.sh
# Leave empty to use the canonical (no-suffix) filenames (will overwrite).
EXP_ID="${EXP_ID:-}"

# Pre-parse script-level flags out of "$@" so they don't leak into the
# python invocation. Supports:
#   --visualize       → switch to visualize.py + force sequential
#   --exp N           → set EXP_ID=N (shorthand)
#   --exp_id N        → set EXP_ID=N (explicit form)
SCRIPT="test.py"
_remaining_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --visualize)
            SCRIPT="visualize.py"
            TEST_SIZE=5
            MAX_PARALLEL=1
            shift
            ;;
        --exp|--exp_id)
            EXP_ID="$2"
            shift 2
            ;;
        --exp=*|--exp_id=*)
            EXP_ID="${1#*=}"
            shift
            ;;
        *)
            _remaining_args+=("$1")
            shift
            ;;
    esac
done
# Save the post-parse args into a global array so launch_combo can forward
# them to python. (Inside launch_combo, "$@" is the function's own args —
# i.e. scenario, behaviour — not the script's, so we can't rely on it.)
SCRIPT_PASSTHRU_ARGS=("${_remaining_args[@]}")

LOG_DIR="${LOG_DIR:-/tmp/adaptive_lora_poc}"
mkdir -p "$LOG_DIR"

if [ -f "./gpu_affinity.sh" ]; then
    . ./gpu_affinity.sh
fi
BLEND_GPUS="${BLEND_GPUS:-$(blend_detect_gpus)}"

# SCENARIOS="seperate_mixed_5050"
SCENARIOS_DEFAULT="seperate_all_aware seperate_all_ignorant seperate_mixed_5050 cluster_aware_ignorant"
SCENARIOS="${SCENARIOS:-$SCENARIOS_DEFAULT}"
read -r -a SCENARIOS_ARR <<< "$SCENARIOS"
# Default comparison: existing weight-space adaptive_gt vs action-space
# adaptive_action_gt. Override with:
#   ADAPTIVE_BEHAVIOURS="adaptive_action_gt" ./test_adaptive_lora_poc.sh
#   ADAPTIVE_BEHAVIOURS="adaptive_action_gt adaptive_fullfinetune_gt" ./test_adaptive_lora_poc.sh
# Use adaptive_gt alone with --save_episode_dump always when regenerating the
# *_adaptive_gt.json training anchors.
ADAPTIVE_BEHAVIOURS="${ADAPTIVE_BEHAVIOURS:-adaptive_fullfinetune_gt}"
read -r -a BEHAVIOURS <<< "$ADAPTIVE_BEHAVIOURS"
# BEHAVIOURS=("adaptive_discrepancy")
# BEHAVIOURS=("always_off" "always_on" "switching_gt" "adaptive_gt" "adaptive_pred" "adaptive_action_gt" "adaptive_fullfinetune_gt")
# BEHAVIOURS=("switching_pred" "adaptive_pred")
# BEHAVIOURS=("always_off")

echo "MAX_PARALLEL = $MAX_PARALLEL  (logs in $LOG_DIR)"
echo "BLEND_GPUS = $BLEND_GPUS"
[ -n "$EXP_ID" ] && echo "EXP_ID = $EXP_ID  → output filenames will get '_exp${EXP_ID}' suffix"

# One combo as a background job. Output goes to its own log file.
launch_combo() {
    local scenario="$1" behaviour="$2" gpu_id="$3" seed="$4"
    local exp_id="$EXP_ID"
    local tag_suffix=""
    [ -n "$PREDICTOR_TAG" ] && tag_suffix="${tag_suffix}_${PREDICTOR_TAG}"
    if [ -z "$exp_id" ] && [ "${#SEEDS_ARR[@]}" -gt 1 ]; then
        exp_id="$seed"
    fi
    [ -n "$exp_id" ] && tag_suffix="${tag_suffix}_exp${exp_id}"
    local log_path="$LOG_DIR/${scenario}_${behaviour}_seed${seed}${tag_suffix}.log"
    local EXTRA_ARGS=()
    [ -n "$PREDICTOR_TAG" ] && EXTRA_ARGS+=(--predictor_tag "$PREDICTOR_TAG")
    [ -n "$exp_id" ]        && EXTRA_ARGS+=(--exp_id "$exp_id")
    echo "[$(date '+%H:%M:%S')] START  $scenario × $behaviour  seed=$seed  GPU=$gpu_id   (log: $log_path)"
    if CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u $SCRIPT --model_dir "$ADAPTIVE_MODEL" --test_model "$CHECKPOINT" \
            --adaptive_lora_scenario "$scenario" \
            --lora_behaviour "$behaviour" \
            --discrepancy_threshold $DISCREPANCY_THRESHOLD \
            --discrepancy_m $DISCREPANCY_M \
            --human_num $HUMAN_NUM \
            --awareness_eval "$AWARENESS_EVAL" \
            --seed "$seed" \
            "${EXTRA_ARGS[@]}" \
            --test_size $TEST_SIZE \
            "${SCRIPT_PASSTHRU_ARGS[@]}" > "$log_path" 2>&1; then
        echo "[$(date '+%H:%M:%S')] OK     $scenario × $behaviour"
    else
        echo "[$(date '+%H:%M:%S')] FAIL   $scenario × $behaviour   (see $log_path)"
    fi
}

running=0
job_index=0
for seed in "${SEEDS_ARR[@]}"; do
    for scenario in "${SCENARIOS_ARR[@]}"; do
        for behaviour in "${BEHAVIOURS[@]}"; do
            gpu_id="$(blend_gpu_for_job "$job_index")"
            launch_combo "$scenario" "$behaviour" "$gpu_id" "$seed" &
            job_index=$((job_index + 1))
            running=$((running + 1))
            if (( running >= MAX_PARALLEL )); then
                wait -n
                running=$((running - 1))
            fi
        done
    done
done
wait  # drain the rest

echo -e "\n\n=========================================================="
echo "ALL TESTS DONE. AGGREGATING RESULTS..."
echo "=========================================================="

# Per-combo summary with ± 95% CI for SR (Wilson) and PL (mean ± 1.96·σ/√N).
# Reads ${ADAPTIVE_MODEL}/test/all_evaluations.json (always-updated aggregate).
SCENARIOS_STR="${SCENARIOS_ARR[*]}"
BEHAVIOURS_STR="${BEHAVIOURS[*]}"
EXP_SUFFIX=""
[ -n "$EXP_ID" ] && EXP_SUFFIX="_exp${EXP_ID}"
python3 - <<EOF
import json, os, math

def wilson_half(p, n, z=1.96):
    if n <= 0: return float('nan')
    denom = 1.0 + z*z/n
    return (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom

agg_path = os.path.join("$ADAPTIVE_MODEL", "test", "all_evaluations.json")
if not os.path.exists(agg_path):
    print(f"  no aggregate found at {agg_path}")
    raise SystemExit

with open(agg_path) as f:
    aggregate = json.load(f)

scenarios = "$SCENARIOS_STR".split()
behaviours = "$BEHAVIOURS_STR".split()
exp_suffix = "$EXP_SUFFIX"
print()
print(f"Per-combo summary (SR ± 95% Wilson CI, PL ± 1.96·SE) from {agg_path}:")
if exp_suffix:
    print(f"  exp_id filter: keys ending in '{exp_suffix}'")
print()
print(f"  {'scenario':<32s} {'behaviour':<22s} {'SR ± CI':>14s} {'PL ± CI':>16s} {'CR':>6s}  N")
print("  " + "-"*92)
for sc in scenarios:
    for beh in behaviours:
        key = f"{sc}_{beh}{exp_suffix}"
        ent = aggregate.get(key)
        if ent is None:
            print(f"  {sc:<32s} {beh:<22s} {'(missing)':>14s}")
            continue
        s = ent.get("summary", {})
        n  = s.get("num_episodes", 0) or 0
        sr = s.get("success_rate")
        cr = s.get("collision_rate")
        pl = s.get("avg_path_length")
        pl_std = s.get("std_path_length")
        if sr is None:
            continue
        sr_ci = wilson_half(sr, n)
        if pl_std is not None and n > 1:
            pl_ci = 1.96 * float(pl_std) / math.sqrt(n)
            pl_str = f"{pl:.2f} ± {pl_ci:.2f}"
        else:
            pl_str = f"{pl:.2f} ± N/A"
        cr_str = f"{cr:.3f}" if isinstance(cr, (int, float)) else str(cr)
        sr_str = f"{sr:.3f} ± {sr_ci:.3f}"
        print(f"  {sc:<32s} {beh:<22s} {sr_str:>14s} {pl_str:>16s} {cr_str:>6s}  {n}")
EOF

echo
# You might need to update these scripts if they rely on the old names
python3 aggregate_results.py
python3 plot_experiment_results.py
