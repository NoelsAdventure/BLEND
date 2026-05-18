#!/bin/bash
#
# LoRA-scale ablation study: run the LoraF policy at fixed α values across
# every scenario. The ablation answers "how does SR / PL / ITR / SD change as
# the LoRA adapter is dialed from base-policy (α=0) to fully aggressive
# (α=1)?" — without the dynamics that adaptive_pred / adaptive_gt would
# introduce.
#
# 5 scales × 4 scenarios = 20 combos. Each uses --lora_behaviour fixed_scale.
# EXP_ID per scale is auto-set to "scale_X.X" so the all_evaluations.json
# keys are distinct (otherwise every scale would clobber the same
# "{scenario}_fixed_scale" key).
#
# Usage:
#   ./test_lora_ablation.sh                           # full sweep
#   SCALES="0.0 0.5 1.0" ./test_lora_ablation.sh      # fewer scales
#   SCENARIOS="seperate_mixed_5050" TEST_SIZE=200 ./test_lora_ablation.sh
#   ./test_lora_ablation.sh --exp seed7  SEED=7      # second seed
#
# Output files (per scale, per scenario):
#   ${MODEL_DIR}/test/all_evaluations.json
#     keyed by "{scenario}_fixed_scale_expscale_X.X"   (or with user EXP_ID
#     suffix if you also pass --exp)
#   /tmp/lora_ablation/{scenario}_scale_{X.X}.log

set -u  # don't abort on first failed combo

# --- Config ---------------------------------------------------------------
TEST_SIZE="${TEST_SIZE:-500}"
HUMAN_NUM="${HUMAN_NUM:-20}"
SCENARIOS="${SCENARIOS:-seperate_mixed_5050 seperate_all_aware seperate_all_ignorant cluster_aware_ignorant}"
SCALES="${SCALES:-0.2 0.4 0.6 0.8 1.0}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SEED="${SEED:-42}"
AWARENESS_EVAL="${AWARENESS_EVAL:-off}"  # static behaviour → predictor scoring is moot

# Models to ablate. Pipe-separated: "LABEL|MODEL_DIR|CHECKPOINT".
# Comment any line out to skip that model on this machine. Same pattern as
# test_baselines.sh BASELINES — copy the script to another PC, uncomment
# whichever LoraF variants exist locally, and you're done.
MODELS=(
    "LoraF rank=1|trained_models/LoraF_invi_visi_rank_1|03400.pt"
    "LoraF rank=4|trained_models/LoraF_invi_visi_rank_4|03400.pt"
    # "LoraE rank=1 alpha=128|trained_models/LoraE_visi_invi_rank1_alpha_128|03400.pt"
    # "LoraE alpha=128|trained_models/LoraE_invi_visi_alpha_128|03400.pt"
    # "LoraG rank=1|trained_models/LoraG_invi_visi_rank_1|03400.pt"
    # "LoraG rank=4|trained_models/LoraG_invi_visi_rank_4|03400.pt"
)

# Optional extra EXP_ID suffix appended on top of the auto "scale_X.X".
# Use for multi-seed ablation runs:  EXP_ID=seed7 SEED=7 ./test_lora_ablation.sh
EXTRA_EXP_ID="${EXP_ID:-}"

# Pre-parse script-level flags so they don't leak into python.
_remaining_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --exp|--exp_id)    EXTRA_EXP_ID="$2"; shift 2 ;;
        --exp=*|--exp_id=*) EXTRA_EXP_ID="${1#*=}"; shift ;;
        *)                  _remaining_args+=("$1"); shift ;;
    esac
done
SCRIPT_PASSTHRU_ARGS=("${_remaining_args[@]}")

LOG_DIR="${LOG_DIR:-/tmp/lora_ablation}"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/.counters"
: > "$LOG_DIR/.counters/ok"
: > "$LOG_DIR/.counters/fail"

START_TS=$(date +%s)
echo "=========================================================="
echo "LoRA-scale ablation starting"
echo "  MODELS        : ${#MODELS[@]}"
for entry in "${MODELS[@]}"; do
    echo "                  - ${entry%%|*}"
done
echo "  HUMAN_NUM     : $HUMAN_NUM"
echo "  TEST_SIZE     : $TEST_SIZE"
echo "  SCALES        : $SCALES"
echo "  SCENARIOS     : $SCENARIOS"
echo "  MAX_PARALLEL  : $MAX_PARALLEL"
echo "  SEED          : $SEED"
[ -n "$EXTRA_EXP_ID" ] && echo "  EXTRA EXP_ID  : $EXTRA_EXP_ID  (appended after scale_X.X)"
echo "  log dir       : $LOG_DIR"
echo "=========================================================="

# Launch one (model × scenario × scale) combo as a background job.
launch_combo() {
    local name="$1" model_dir="$2" ckpt="$3" sc="$4" scale="$5"
    local exp_id="scale_${scale}"
    [ -n "$EXTRA_EXP_ID" ] && exp_id="${exp_id}_${EXTRA_EXP_ID}"
    local model_tag="${name//[^A-Za-z0-9]/_}"
    local log_path="$LOG_DIR/${model_tag}_${sc}_scale_${scale}.log"
    [ -n "$EXTRA_EXP_ID" ] && log_path="$LOG_DIR/${model_tag}_${sc}_scale_${scale}_${EXTRA_EXP_ID}.log"
    echo "[$(date '+%H:%M:%S')] START  $name  ×  $sc  ×  α=$scale   (log: $log_path)"
    if python3 -u test.py \
            --model_dir "$model_dir" \
            --test_model "$ckpt" \
            --adaptive_lora_scenario "$sc" \
            --lora_behaviour fixed_scale \
            --lora_scale "$scale" \
            --human_num "$HUMAN_NUM" \
            --test_size "$TEST_SIZE" \
            --awareness_eval "$AWARENESS_EVAL" \
            --seed "$SEED" \
            --exp_id "$exp_id" \
            "${SCRIPT_PASSTHRU_ARGS[@]}" \
            > "$log_path" 2>&1; then
        echo x >> "$LOG_DIR/.counters/ok"
        echo "[$(date '+%H:%M:%S')] OK     $name  ×  $sc  ×  α=$scale"
    else
        echo x >> "$LOG_DIR/.counters/fail"
        echo "[$(date '+%H:%M:%S')] FAIL   $name  ×  $sc  ×  α=$scale   (see $log_path)"
    fi
}

running=0
for entry in "${MODELS[@]}"; do
    name="${entry%%|*}"
    rest="${entry#*|}"
    model_dir="${rest%%|*}"
    ckpt="${rest##*|}"

    if [ ! -d "$model_dir" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP  $name   (model_dir not found: $model_dir)"
        continue
    fi
    if [ ! -f "$model_dir/checkpoints/$ckpt" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP  $name   (checkpoint not found: $model_dir/checkpoints/$ckpt)"
        continue
    fi

    for sc in $SCENARIOS; do
        for scale in $SCALES; do
            launch_combo "$name" "$model_dir" "$ckpt" "$sc" "$scale" &
            running=$((running + 1))
            if (( running >= MAX_PARALLEL )); then
                wait -n
                running=$((running - 1))
            fi
        done
    done
done
wait  # drain the rest

n_ok=$(wc -l < "$LOG_DIR/.counters/ok")
n_fail=$(wc -l < "$LOG_DIR/.counters/fail")
n_total=$((n_ok + n_fail))

END_TS=$(date +%s)
elapsed=$(( END_TS - START_TS ))
echo
echo "=========================================================="
echo "Ablation sweep done in $((elapsed/60))m $((elapsed%60))s"
echo "  combos tried : $n_total"
echo "  succeeded    : $n_ok"
echo "  failed       : $n_fail"
echo "=========================================================="
echo
echo "Per-(model × scenario × scale) summary  (SR ± Wilson CI · PL ± SE · ITR · SD):"
SCALES_STR="$SCALES"
SCENARIOS_STR="$SCENARIOS"
EXP_SUFFIX_EXTRA=""
[ -n "$EXTRA_EXP_ID" ] && EXP_SUFFIX_EXTRA="_${EXTRA_EXP_ID}"

# Emit one "label|model_dir" line per model, then read in python.
MODELS_TSV=""
for entry in "${MODELS[@]}"; do
    name="${entry%%|*}"
    rest="${entry#*|}"
    model_dir="${rest%%|*}"
    MODELS_TSV+="${name}|${model_dir}"$'\n'
done

python3 - <<EOF
import json, os, math

def wilson_half(p, n, z=1.96):
    if n is None or n <= 0: return float('nan')
    denom = 1.0 + z*z/n
    return (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom

def se_half(std, n, z=1.96):
    if std is None or n is None or n <= 1: return float('nan')
    return z * float(std) / math.sqrt(n)

scales = "$SCALES_STR".split()
scenarios = "$SCENARIOS_STR".split()
extra_suffix = "$EXP_SUFFIX_EXTRA"

models = []
for line in """$MODELS_TSV""".strip().splitlines():
    name, mdir = line.split("|", 1)
    models.append((name, mdir))

for name, mdir in models:
    agg_path = os.path.join(mdir, "test", "all_evaluations.json")
    if not os.path.exists(agg_path):
        print(f"\n  [{name}]  no aggregate at {agg_path}")
        continue
    with open(agg_path) as f:
        aggregate = json.load(f)
    print(f"\n  [{name}]  ({mdir})")
    print(f"  {'scenario':<32s} {'α':>6s}  {'SR ± CI':>14s}  {'PL ± SE':>14s}  {'ITR%':>7s}  {'SD':>5s}    N")
    print("  " + "-"*92)
    for sc in scenarios:
        for scale in scales:
            key = f"{sc}_fixed_scale_expscale_{scale}{extra_suffix}"
            ent = aggregate.get(key)
            if ent is None:
                print(f"  {sc:<32s} {scale:>6s}  (missing)")
                continue
            s = ent.get("summary", {})
            n  = s.get("num_episodes", 0) or 0
            sr = s.get("success_rate"); pl = s.get("avg_path_length")
            pl_std = s.get("std_path_length")
            itr = s.get("avg_intrusion_ratio_pct")
            sd  = s.get("avg_min_social_distance")
            sr_ci_str = "{:.3f} ± {:.3f}".format(sr, wilson_half(sr, n)) if sr is not None else "—"
            pl_se_str = "{:.2f} ± {:.2f}".format(pl, se_half(pl_std, n)) if pl_std is not None else ("{:.2f}".format(pl) if pl is not None else "—")
            itr_str = "{:.2f}".format(itr) if itr is not None else "—"
            sd_str  = "{:.2f}".format(sd)  if sd is not None and sd == sd else "—"
            print(f"  {sc:<32s} {scale:>6s}  {sr_ci_str:>14s}  {pl_se_str:>14s}  {itr_str:>7s}  {sd_str:>5s}    {n}")
        print()
EOF
