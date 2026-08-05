#!/bin/bash
#
# Master parallel runner: produces every result the notebook tables consume,
# in parallel, optimized for a powerful workstation (multi-GPU or 24+GB VRAM).
#
# Launches three sub-scripts in parallel, each potentially with multiple
# seeds:
#   1. test_baselines.sh         — SF / ORCA / CrowdNav++ / GenSafeNav-naive
#                                  × {scenarios} × {behaviour: always_off}
#   2. test_adaptive_lora_poc.sh — LoraF × {scenarios} × {behaviours: 6}
#   3. test_lora_ablation.sh     — LoraF rank=1/4 × {scenarios} × {scales: 5}
#
# Each sub-script handles its own internal parallelism via MAX_PARALLEL.
# This script controls how many sub-script INVOCATIONS run concurrently.
#
# Total concurrent test.py processes peaks at:
#   OUTER_PARALLEL × MAX_PARALLEL_PER_SCRIPT
# (defaults: 3 × 4 = 12 concurrent; tune for your GPU memory budget).
#
# Usage:
#   ./run_all_results.sh                              # full sweep, 2 seeds
#   SEEDS="10"            ./run_all_results.sh        # single seed
#   SEEDS="10 20 42"      ./run_all_results.sh        # 3 seeds
#   OUTER_PARALLEL=6      ./run_all_results.sh        # all 6 sub-invocations at once
#   MAX_PARALLEL_PER_SCRIPT=2 ./run_all_results.sh    # GPU-constrained mode
#   SKIP_ABLATION=1       ./run_all_results.sh        # baselines + adaptive only
#   SKIP_BASELINES=1 SKIP_ADAPTIVE=1 ./run_all_results.sh  # ablation only

set -u

# --- Config ---------------------------------------------------------------
SEEDS="${SEEDS:-42 1000 2000 3000 4000}"
OUTER_PARALLEL="${OUTER_PARALLEL:-3}"        # how many sub-scripts run concurrently
MAX_PARALLEL_PER_SCRIPT="${MAX_PARALLEL_PER_SCRIPT:-4}"  # per sub-script
TEST_SIZE="${TEST_SIZE:-1250}"
HUMAN_NUM="${HUMAN_NUM:-20}"
SCENARIOS="${SCENARIOS:-seperate_mixed_5050 seperate_all_aware seperate_all_ignorant cluster_aware_ignorant}"

# Sub-script toggles (1 to skip)
SKIP_BASELINES="${SKIP_BASELINES:-0}"
SKIP_ADAPTIVE="${SKIP_ADAPTIVE:-0}"
SKIP_ABLATION="${SKIP_ABLATION:-1}"

# Export the per-sub-script knobs so they reach the child invocations
export MAX_PARALLEL="$MAX_PARALLEL_PER_SCRIPT"
export TEST_SIZE
export HUMAN_NUM
export SCENARIOS

LOG_DIR="${LOG_DIR:-/tmp/run_all_results}"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/.counters"
: > "$LOG_DIR/.counters/ok"
: > "$LOG_DIR/.counters/fail"

START_TS=$(date +%s)

# --- Build the list of (script, seed, label) jobs to run ------------------
JOBS=()
for seed in $SEEDS; do
    [ "$SKIP_BASELINES" != "1" ] && JOBS+=("baselines|$seed")
    [ "$SKIP_ADAPTIVE"  != "1" ] && JOBS+=("adaptive|$seed")
    [ "$SKIP_ABLATION"  != "1" ] && JOBS+=("ablation|$seed")
done

echo "=========================================================="
echo "Master parallel runner starting"
echo "  SEEDS                   : $SEEDS"
echo "  OUTER_PARALLEL          : $OUTER_PARALLEL  (concurrent sub-scripts)"
echo "  MAX_PARALLEL_PER_SCRIPT : $MAX_PARALLEL_PER_SCRIPT  (test.py procs per sub-script)"
echo "  Peak concurrent procs   : $((OUTER_PARALLEL * MAX_PARALLEL_PER_SCRIPT))"
echo "  TEST_SIZE               : $TEST_SIZE"
echo "  HUMAN_NUM               : $HUMAN_NUM"
echo "  SCENARIOS               : $SCENARIOS"
echo "  total sub-script jobs   : ${#JOBS[@]}"
echo "  log dir                 : $LOG_DIR"
echo "=========================================================="
echo "  Progress tickers: an overall sub-script bar + an in-flight combo"
echo "  counter (polled from each sub-script log every ${PROGRESS_INTERVAL:-30}s)."
echo "=========================================================="

# Progress-bar config.
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"   # seconds between ticker updates
BAR_LEN=30
JOBS_TOTAL=${#JOBS[@]}

# Format an ASCII bar like [████████░░░░░░░] 42%
_bar() {
    local done=$1 total=$2 pct
    if [ "$total" -le 0 ]; then echo "[empty]  0%"; return; fi
    local filled=$(( done * BAR_LEN / total ))
    [ "$filled" -gt "$BAR_LEN" ] && filled=$BAR_LEN
    local bar=""
    local i
    for ((i=0; i<BAR_LEN; i++)); do
        if (( i < filled )); then bar+="█"; else bar+="░"; fi
    done
    pct=$(( done * 100 / total ))
    printf "[%s] %3d%%" "$bar" "$pct"
}

# Human-readable elapsed string
_fmt_elapsed() {
    local s=$1
    if (( s < 60 )); then printf "%ds" "$s"
    elif (( s < 3600 )); then printf "%dm %02ds" $((s/60)) $((s%60))
    else printf "%dh %02dm" $((s/3600)) $(((s%3600)/60))
    fi
}

# Count finished (OK/FAIL) lines across every sub-script log — combo-level.
_count_combos_done() {
    local d=0
    local log
    for log in "$LOG_DIR"/*_seed*.log; do
        [ -f "$log" ] || continue
        local n=$(grep -cE "^\[[0-9:]+\] (OK|FAIL) " "$log" 2>/dev/null)
        d=$((d + ${n:-0}))
    done
    echo "$d"
}

# Background ticker — prints a progress line every PROGRESS_INTERVAL.
# Exits when the trap fires from the main script's EXIT.
progress_ticker() {
    while true; do
        sleep "$PROGRESS_INTERVAL"
        local sub_done=$(wc -l < "$LOG_DIR/.counters/ok" 2>/dev/null || echo 0)
        local sub_fail=$(wc -l < "$LOG_DIR/.counters/fail" 2>/dev/null || echo 0)
        local sub_total_done=$(( sub_done + sub_fail ))
        local combos_done=$(_count_combos_done)
        local elapsed=$(( $(date +%s) - START_TS ))
        local sub_bar=$(_bar "$sub_total_done" "$JOBS_TOTAL")
        echo "[$(date '+%H:%M:%S')] $sub_bar  sub-scripts: $sub_total_done/$JOBS_TOTAL · combos done: $combos_done · elapsed: $(_fmt_elapsed $elapsed)"
    done
}

# Start ticker in background, kill it on exit.
progress_ticker &
TICKER_PID=$!
trap "kill $TICKER_PID 2>/dev/null; wait $TICKER_PID 2>/dev/null" EXIT

launch_subscript() {
    local kind="$1" seed="$2"
    local log_path="$LOG_DIR/${kind}_seed${seed}.log"
    local script
    case "$kind" in
        baselines) script="./test_baselines.sh" ;;
        adaptive)  script="./test_adaptive_lora_poc.sh" ;;
        ablation)  script="./test_lora_ablation.sh" ;;
        *)         echo "unknown kind: $kind"; return 1 ;;
    esac
    echo "[$(date '+%H:%M:%S')] START  $kind  seed=$seed   (log: $log_path)"
    # Each sub-script accepts --exp via its own CLI parser; SEED env var
    # controls the env-side random seed.
    if SEED="$seed" $script --exp "$seed" > "$log_path" 2>&1; then
        echo x >> "$LOG_DIR/.counters/ok"
        local sub_done=$(wc -l < "$LOG_DIR/.counters/ok")
        local sub_fail=$(wc -l < "$LOG_DIR/.counters/fail" 2>/dev/null || echo 0)
        local sub_total_done=$(( sub_done + sub_fail ))
        local bar=$(_bar "$sub_total_done" "$JOBS_TOTAL")
        local el=$(( $(date +%s) - START_TS ))
        echo "[$(date '+%H:%M:%S')] OK     $kind  seed=$seed   $bar  ($sub_total_done/$JOBS_TOTAL · elapsed $(_fmt_elapsed $el))"
    else
        echo x >> "$LOG_DIR/.counters/fail"
        local sub_done=$(wc -l < "$LOG_DIR/.counters/ok" 2>/dev/null || echo 0)
        local sub_fail=$(wc -l < "$LOG_DIR/.counters/fail")
        local sub_total_done=$(( sub_done + sub_fail ))
        local bar=$(_bar "$sub_total_done" "$JOBS_TOTAL")
        local el=$(( $(date +%s) - START_TS ))
        echo "[$(date '+%H:%M:%S')] FAIL   $kind  seed=$seed   $bar  ($sub_total_done/$JOBS_TOTAL · elapsed $(_fmt_elapsed $el))   (see $log_path)"
    fi
}

running=0
for job in "${JOBS[@]}"; do
    kind="${job%|*}"
    seed="${job#*|}"
    launch_subscript "$kind" "$seed" &
    running=$((running + 1))
    if (( running >= OUTER_PARALLEL )); then
        wait -n
        running=$((running - 1))
    fi
done
wait  # drain remaining
kill "$TICKER_PID" 2>/dev/null
wait "$TICKER_PID" 2>/dev/null

n_ok=$(wc -l < "$LOG_DIR/.counters/ok")
n_fail=$(wc -l < "$LOG_DIR/.counters/fail")
n_total=$((n_ok + n_fail))

END_TS=$(date +%s)
elapsed=$(( END_TS - START_TS ))
echo
echo "=========================================================="
echo "Master run done in $((elapsed/3600))h $((elapsed%3600/60))m $((elapsed%60))s"
echo "  sub-scripts launched : $n_total"
echo "  succeeded            : $n_ok"
echo "  failed               : $n_fail"
echo "=========================================================="

if [ "$n_fail" -gt 0 ]; then
    echo
    echo "Failed sub-scripts — last 5 lines of each log:"
    for f in "$LOG_DIR"/*_seed*.log; do
        # rough heuristic: if log doesn't contain "sweep done" or "ALL TESTS DONE", it likely failed
        if ! grep -qE "sweep done|ALL TESTS DONE|Ablation sweep done" "$f" 2>/dev/null; then
            echo
            echo "  --- $f ---"
            tail -5 "$f"
        fi
    done
fi

echo
echo "Aggregated tables now ready in the notebook (re-run all cells):"
echo "  Section 1 — Combined results (multi-seed if SEEDS had ≥2)"
echo "  Section 6 — Scale sweep vs adaptive (fixed-α + base + naive)"
echo
echo "Per-script logs:"
ls -lh "$LOG_DIR"/*_seed*.log 2>/dev/null
