#!/bin/bash
#
# Throwaway: sweep robot v_pref downward and find the first v_pref at which the
# naive baseline (FullFineTune_invi_visi + always_off) collides on the first
# episode of seed 42 — for the mixed and cluster scenarios.
#
# Usage:
#   ./sweep_naive_speed.sh             # default sweep
#   STEP=0.02 ./sweep_naive_speed.sh   # finer step
#
# Tunables (env vars):
#   START (1.00), MIN (0.05), STEP (0.05)
#   SEED (42), TEST_CASE (0), HUMAN_NUM (20)

set -u

START="${START:-1.00}"
MIN="${MIN:-0.05}"
STEP="${STEP:-0.05}"
SEED="${SEED:-42}"
TEST_CASE="${TEST_CASE:-0}"
HUMAN_NUM="${HUMAN_NUM:-20}"

MODEL_DIR="trained_models/FullFineTune_invi_visi"
CKPT="03400.pt"

SCENARIOS=(
    "seperate_mixed_5050|mixed"
    "cluster_aware_ignorant|cluster"
)

declare -A PREV_SAFE
declare -A COLLIDE_V

for sc_entry in "${SCENARIOS[@]}"; do
    scenario="${sc_entry%%|*}"
    tag="${sc_entry##*|}"
    echo "=========================================================="
    echo "[$tag] sweeping $scenario from v=$START down to v=$MIN step=$STEP"
    echo "=========================================================="

    prev_safe_v=""
    collide_v=""

    # Walk v from START down to MIN by -STEP using python for clean float math.
    v_values=$(python3 -c "
v = float('$START')
step = float('$STEP')
mn = float('$MIN')
out = []
while v >= mn - 1e-9:
    out.append(f'{v:.2f}')
    v -= step
print(' '.join(out))
")

    for v in $v_values; do
        # exp_id encodes the speed; '.' is illegal in some filename contexts,
        # but the project consistently uses _expX suffixes — replace . with p.
        exp_id="vsweep_${v//./p}"
        log_path="/tmp/vsweep_${tag}_${exp_id}.log"

        echo -n "[$tag] v=$v ... "

        if ! python3 -u test.py \
                --model_dir "$MODEL_DIR" \
                --test_model "$CKPT" \
                --lora_behaviour always_off \
                --adaptive_lora_scenario "$scenario" \
                --human_num "$HUMAN_NUM" \
                --seed "$SEED" \
                --test_case "$TEST_CASE" \
                --test_size 1 \
                --robot_v_pref "$v" \
                --save_episode_dump always \
                --exp_id "$exp_id" \
                --awareness_eval off \
                > "$log_path" 2>&1; then
            echo "RUN-FAIL (see $log_path)"
            continue
        fi

        dump_path="${MODEL_DIR}/test/${scenario}_always_off_exp${exp_id}.json"
        if [ ! -f "$dump_path" ]; then
            echo "NO-DUMP at $dump_path"
            continue
        fi

        result=$(python3 -c "
import json
d = json.load(open('$dump_path'))
eps = d.get('episodes', [])
print(eps[0].get('result', 'Unknown') if eps else 'NoEpisodes')
")
        echo "$result"

        if [ "$result" = "Collision" ]; then
            collide_v="$v"
            break
        fi
        prev_safe_v="$v"
    done

    PREV_SAFE[$tag]="${prev_safe_v:-NA}"
    COLLIDE_V[$tag]="${collide_v:-NA}"
done

echo "=========================================================="
echo "SUMMARY (seed=$SEED, test_case=$TEST_CASE, human_num=$HUMAN_NUM)"
echo "=========================================================="
printf "%-10s  %-18s  %-18s\n" "scenario" "prev_safe_v_pref" "colliding_v_pref"
for tag in mixed cluster; do
    printf "%-10s  %-18s  %-18s\n" "$tag" "${PREV_SAFE[$tag]}" "${COLLIDE_V[$tag]}"
done
echo "=========================================================="
