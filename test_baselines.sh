#!/bin/bash
#
# Run the four paper baselines across all standard scenarios. Each baseline
# uses an existing checkpoint — NO new training needed. Outputs go to each
# model's own test/ subdirectory under trained_models/.
#
# Baselines tested (none use LoRA — --lora_behaviour always_off keeps the CLI
# happy; classical SF/ORCA don't even consult the LoRA flag):
#   SF              — Social Force, classical
#   ORCA            — RVO, classical
#   CrowdNav++      — base SRNN+GST policy without LoRA (Ours_GST checkpoint)
#   GenSafeNav-naive — Fulltune SRNN+GST trained on uni→invi but with LoRA disabled
#
# Adjust the BASELINES array below to retarget model_dir / checkpoint.
#
# Usage (inside the gen_safe_py10 container):
#   ./test_baselines.sh
#   HUMAN_NUM=30 ./test_baselines.sh         # at a different crowd size
#   TEST_SIZE=200 ./test_baselines.sh        # tighter sweep for a quick check
#   SCENARIOS="seperate_mixed_5050" ./test_baselines.sh   # one scenario only

set -u  # no unbound vars; do NOT use -e so one failed combo doesn't abort the rest

# --- Config ---------------------------------------------------------------
TEST_SIZE="${TEST_SIZE:-250}"
HUMAN_NUM="${HUMAN_NUM:-20}"
SCENARIOS="${SCENARIOS:-seperate_mixed_5050 seperate_all_aware seperate_all_ignorant cluster_aware_ignorant}"
# Test seeds — passed to test.py via --seed. Each value = a distinct simulator
# rollout. Without this loop, runs are bit-identical regardless of --exp_id.
# Override per-run:  SEEDS="42 1000" ./test_baselines.sh
SEEDS="${SEEDS:-4000}"
# Max parallel test.py processes. 4090 has 24 GB → ~4 neural processes fit
# comfortably (~5 GB each). Override per-run: MAX_PARALLEL=8 ./test_baselines.sh
MAX_PARALLEL="${MAX_PARALLEL:-12}"
# Exp ID suffix — passed to test.py as --exp_id, which appends "_exp<EXP_ID>"
# to every output filename (per-episode JSON, all_evaluations.json key, log).
# Use to keep multiple runs side-by-side without overwriting:
#   EXP_ID=hn20  ./test_baselines.sh
#   EXP_ID=hn30  HUMAN_NUM=30  ./test_baselines.sh
#   EXP_ID=$(date +%Y%m%d_%H%M)  ./test_baselines.sh
# Also accepted as script-level CLI:  ./test_baselines.sh --exp 10
EXP_ID="${EXP_ID:-}"

# Pre-parse script-level flags so they don't leak into the python invocation.
_remaining_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --exp|--exp_id)    EXP_ID="$2"; shift 2 ;;
        --exp=*|--exp_id=*) EXP_ID="${1#*=}"; shift ;;
        *)                  _remaining_args+=("$1"); shift ;;
    esac
done
SCRIPT_PASSTHRU_ARGS=("${_remaining_args[@]}")

# Each entry: "NAME|MODEL_DIR|CHECKPOINT". Pipe-separated so the names can
# contain spaces; only the array splitting is whitespace-sensitive.
BASELINES=(
    "SF|trained_models/SF|05207.pt"
    "ORCA|trained_models/ORCA|05207.pt"
    "CrowdNav++|trained_models/GST_predictor_rand|05207.pt"
    "GenSafeNav-naive|trained_models/FullFineTune_invi_visi|03400.pt"
)

LOG_DIR="${LOG_DIR:-/tmp/baselines}"
mkdir -p "$LOG_DIR"

START_TS=$(date +%s)
echo "=========================================================="
echo "Baseline test sweep starting"
echo "  HUMAN_NUM     : $HUMAN_NUM"
echo "  TEST_SIZE     : $TEST_SIZE"
echo "  SCENARIOS     : $SCENARIOS"
echo "  SEEDS         : $SEEDS"
echo "  baselines     : ${#BASELINES[@]}"
echo "  MAX_PARALLEL  : $MAX_PARALLEL"
[ -n "$EXP_ID" ] && echo "  EXP_ID        : $EXP_ID  → output filenames suffixed '_exp${EXP_ID}'"
echo "  log dir       : $LOG_DIR"
echo "=========================================================="

# Result-counter files (one int per file, atomic appends from background jobs)
mkdir -p "$LOG_DIR/.counters"
: > "$LOG_DIR/.counters/ok"
: > "$LOG_DIR/.counters/fail"
: > "$LOG_DIR/.counters/skip"

# Launch one combo as a background job.
launch_combo() {
    local name="$1" model_dir="$2" ckpt="$3" sc="$4" seed="$5"
    # exp_id always encodes the seed; optionally prefixed by EXP_ID for run tagging.
    local exp_id
    if [ -n "$EXP_ID" ]; then
        exp_id="${EXP_ID}_${seed}"
    else
        exp_id="$seed"
    fi
    local log_path="$LOG_DIR/${name//[^A-Za-z0-9]/_}_${sc}_exp${exp_id}.log"
    echo "[$(date '+%H:%M:%S')] START  $name  ×  $sc  × seed=$seed   (log: $log_path)"
    if python3 -u test.py \
            --model_dir "$model_dir" \
            --test_model "$ckpt" \
            --adaptive_lora_scenario "$sc" \
            --lora_behaviour always_off \
            --human_num "$HUMAN_NUM" \
            --test_size "$TEST_SIZE" \
            --awareness_eval off \
            --seed "$seed" \
            --exp_id "$exp_id" \
            "${SCRIPT_PASSTHRU_ARGS[@]}" \
            > "$log_path" 2>&1; then
        echo x >> "$LOG_DIR/.counters/ok"
        echo "[$(date '+%H:%M:%S')] OK     $name  ×  $sc  × seed=$seed"
    else
        echo x >> "$LOG_DIR/.counters/fail"
        echo "[$(date '+%H:%M:%S')] FAIL   $name  ×  $sc  × seed=$seed   (see $log_path)"
    fi
}

running=0
for entry in "${BASELINES[@]}"; do
    name="${entry%%|*}"
    rest="${entry#*|}"
    model_dir="${rest%%|*}"
    ckpt="${rest##*|}"

    if [ ! -d "$model_dir" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP  $name   (model_dir not found: $model_dir)"
        echo x >> "$LOG_DIR/.counters/skip"
        continue
    fi
    if [ ! -f "$model_dir/checkpoints/$ckpt" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP  $name   (checkpoint not found: $model_dir/checkpoints/$ckpt)"
        echo x >> "$LOG_DIR/.counters/skip"
        continue
    fi

    for sc in $SCENARIOS; do
        for SEED in $SEEDS; do
            launch_combo "$name" "$model_dir" "$ckpt" "$sc" "$SEED" &
            running=$((running + 1))
            if (( running >= MAX_PARALLEL )); then
                wait -n          # block until ANY background job finishes
                running=$((running - 1))
            fi
        done
    done
done
wait  # drain the rest

n_ok=$(wc -l < "$LOG_DIR/.counters/ok")
n_fail=$(wc -l < "$LOG_DIR/.counters/fail")
n_skip=$(wc -l < "$LOG_DIR/.counters/skip")
n_total=$((n_ok + n_fail))

END_TS=$(date +%s)
elapsed=$(( END_TS - START_TS ))
echo
echo "=========================================================="
echo "Baseline sweep done in $((elapsed/60))m $((elapsed%60))s"
echo "  combos tried : $n_total"
echo "  succeeded    : $n_ok"
echo "  failed       : $n_fail"
echo "  skipped      : $n_skip"
echo "=========================================================="
echo
echo "Per-combo summaries (SR mean ± seed-SD across $(echo $SEEDS | wc -w) seeds, PL mean ± seed-SD):"
python3 - <<EOF
import json, os, math
from statistics import mean, pstdev

exp_prefix = "${EXP_ID:+${EXP_ID}_}"   # optional EXP_ID prefix on exp_id labels
seeds = "$SEEDS".split()
# Read from each model_dir's all_evaluations.json — the small aggregate is
# always updated every run, even when the heavy per-episode dump is disabled
# for non-adaptive_gt behaviours.
for entry in [
    ("SF",                "trained_models/SF"),
    ("ORCA",              "trained_models/ORCA"),
    ("CrowdNav++",        "trained_models/GST_predictor_rand"),
    ("GenSafeNav-naive",  "trained_models/FullFineTune_invi_visi"),
]:
    name, mdir = entry
    aggregate_path = os.path.join(mdir, "test", "all_evaluations.json")
    if not os.path.exists(aggregate_path):
        continue
    try:
        with open(aggregate_path) as f:
            aggregate = json.load(f)
    except Exception as e:
        print("\n  {}  ({})  [load error: {}]".format(name, mdir, e))
        continue
    rows = []
    for sc in "$SCENARIOS".split():
        srs, crs, pls, ns = [], [], [], []
        for s_ in seeds:
            key = "{}_always_off_exp{}{}".format(sc, exp_prefix, s_)
            ent = aggregate.get(key)
            if ent is None:
                continue
            s = ent.get("summary", {})
            srs.append(s.get("success_rate"))
            crs.append(s.get("collision_rate"))
            pls.append(s.get("avg_path_length"))
            ns.append(s.get("num_episodes", 0) or 0)
        if not srs:
            continue
        sr_mu = mean(srs); sr_sd = pstdev(srs) if len(srs) > 1 else 0.0
        cr_mu = mean(crs); pl_mu = mean(pls); pl_sd = pstdev(pls) if len(pls) > 1 else 0.0
        rows.append((sc, sr_mu, sr_sd, cr_mu, pl_mu, pl_sd, sum(ns)))
    if rows:
        print("\n  {}  ({})".format(name, mdir))
        for sc, sr_mu, sr_sd, cr_mu, pl_mu, pl_sd, n_total in rows:
            sr_str = "{:.3f} ± {:.3f}".format(sr_mu, sr_sd)
            cr_str = "{:.3f}".format(cr_mu)
            pl_str = "{:.2f} ± {:.2f}".format(pl_mu, pl_sd)
            print("    {:<32s} SR={:>13s}  CR={}  PL={}  N_total={}".format(sc, sr_str, cr_str, pl_str, n_total))
EOF
