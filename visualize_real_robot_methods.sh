#!/bin/bash
#
# Render MP4 videos of the three methods compared in the paper's real-robot
# table on the two scenarios used for the bar chart in plot_real_robot.ipynb.
#
#   ours              — LoraF_invi_visi_rank_1   + --lora_behaviour adaptive_gt
#   gensafenav_cons   — LoraF_invi_visi_rank_1   + --lora_behaviour always_off
#   gensafenav_naive  — FullFineTune_invi_visi   + --lora_behaviour always_off
#
#   scenarios: seperate_mixed_5050 (→ "mixed"), cluster_aware_ignorant (→ "cluster")
#
# visualize.py writes to visualizations/<basename(model_dir)>/<scenario>/*.mp4.
# This script collects each run's MP4 and renames it into a scenario-keyed
# layout so the three methods sit next to each other per scenario:
#
#   visualizations/real_robot_methods/
#   ├── mixed/
#   │   ├── ours_ep0.mp4
#   │   ├── gensafenav_cons_ep0.mp4
#   │   ├── gensafenav_naive_ep0.mp4
#   │   └── ...
#   └── cluster/
#       └── ...
#
# Usage (inside the gen_safe_py10 container):
#   ./visualize_real_robot_methods.sh
#   SEED=1000 ./visualize_real_robot_methods.sh
#   ROBOT_V_PREF=0.90 ./visualize_real_robot_methods.sh
# The episode list per scenario is hard-coded in SCENARIOS below — edit it
# there rather than passing a count.

set -u  # no unbound vars; do NOT use -e so one failed combo doesn't abort the rest

# --- Config ---------------------------------------------------------------
HUMAN_NUM="${HUMAN_NUM:-20}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-visualizations/real_robot_methods}"
# Optional: override robot.v_pref (default = use the model's training-time value).
# Example: ROBOT_V_PREF=0.90 ./visualize_real_robot_methods.sh
ROBOT_V_PREF="${ROBOT_V_PREF:-}"

# Each entry: "NAME|MODEL_DIR|CHECKPOINT|BEHAVIOUR"
METHODS=(
    "ours|trained_models/LoraF_invi_visi_rank_1|03400.pt|adaptive_gt"
    # "gensafenav_cons|trained_models/LoraF_invi_visi_rank_1|03400.pt|always_off"
    # "gensafenav_naive|trained_models/FullFineTune_invi_visi|03400.pt|always_off"
)

# Each entry: "CANONICAL_SCENARIO_NAME|FOLDER_TAG|COMMA_SEPARATED_EPISODES"
# Episode lists are the top-10 test_case indices per scenario where:
#   (1) NAIVE (FullFineTune + always_off) COLLIDES, AND
#   (2) "ours" (LoraF + adaptive_gt) SUCCEEDS, AND
#   (3) cons (LoraF + always_off) also SUCCEEDS, AND
#   (4) ours.path_length < cons.path_length AND ours.time < cons.time
# sorted by the LARGEST combined Δ = (cons − ours) ΔPL + Δt — i.e. the eps
# where ours is most clearly more efficient than cons, on top of being safer
# than naive. Source: 250-ep naive + ours + cons dumps at seed=42 in
#   trained_models/FullFineTune_invi_visi/test/*_naive250_*_s42.json
#   trained_models/LoraF_invi_visi_rank_1/test/*_adaptive_gt_exp42.json
#   trained_models/LoraF_invi_visi_rank_1/test/*_cons250_*_s42.json
# EPISODE_RANGE controls how many test_case indices are rendered per scenario.
# Default: full 0..249 sweep — produces 250 ep × 3 methods × 2 scenarios = 1500
# MP4s + 1500 summary rows. Each visualize.py invocation is ~25s so the full
# sweep is ~10 h. Override with EPISODE_RANGE="0 19" (eps 0..19) for a quick
# pass.
EPISODE_RANGE="${EPISODE_RANGE:-251 400}"
_eps_csv=$(seq -s, $EPISODE_RANGE)
# SCENARIOS=(
    # "seperate_mixed_5050|mixed|${_eps_csv}"
    # "cluster_aware_ignorant|cluster|${_eps_csv}"
# )
SCENARIOS=(
    "seperate_mixed_5050|mixed|11"
    "cluster_aware_ignorant|cluster|10,40"
)


# Compute total renders = sum_over_scenarios(len(episode_list)) * methods
total_renders=0
for sc_entry in "${SCENARIOS[@]}"; do
    ep_csv="${sc_entry##*|}"
    IFS=',' read -ra _eps <<< "$ep_csv"
    total_renders=$((total_renders + ${#_eps[@]} * ${#METHODS[@]}))
done

SUMMARY_JSON="$OUT_ROOT/summary.json"
mkdir -p "$OUT_ROOT"
# Reset summary at start of sweep (idempotent).
echo "[]" > "$SUMMARY_JSON"

START_TS=$(date +%s)
echo "=========================================================="
echo "Real-robot methods visualization sweep starting"
echo "  HUMAN_NUM     : $HUMAN_NUM"
echo "  SEED          : $SEED"
echo "  OUT_ROOT      : $OUT_ROOT"
echo "  ROBOT_V_PREF  : ${ROBOT_V_PREF:-<config default>}"
echo "  methods       : ${#METHODS[@]}"
echo "  scenarios     : ${#SCENARIOS[@]}"
echo "  episodes      :"
for sc_entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r _sc _tag _eps <<< "$sc_entry"
    printf "    %-12s %s\n" "$_tag" "[$_eps]"
done
echo "  total renders : $total_renders"
echo "=========================================================="

ok_count=0
fail_count=0

# Run one (method, scenario, episode) — one visualize.py invocation, fast.
# `--test_case K` jumps case_counter straight to K (no auto-increment through
# 0..K-1). Note: the resulting episode is NOT bit-identical to test.py's ep K in
# a 250-episode sweep, because the RNG sequence diverges. Whether that matters
# depends on what you're verifying — see render_only_cases support in
# rl/evaluation.py for the slower bit-identical path.
run_combo() {
    local method="$1" model_dir="$2" ckpt="$3" behaviour="$4"
    local scenario="$5" folder_tag="$6" episode_idx="$7"

    local dst_dir="$OUT_ROOT/$folder_tag"
    mkdir -p "$dst_dir"

    local default_mp4_dir="visualizations/$(basename "$model_dir")/$scenario"
    local dst_path="$dst_dir/${method}_ep${episode_idx}.mp4"

    local cutoff_ts
    cutoff_ts=$(date +%s)

    echo "[$(date '+%H:%M:%S')] START  ${method}  ×  ${folder_tag}  × ep=${episode_idx}"

    local extra_args=()
    if [ -n "$ROBOT_V_PREF" ]; then
        extra_args+=(--robot_v_pref "$ROBOT_V_PREF")
    fi

    local exp_id="vis_${method}_${folder_tag}_ep${episode_idx}"

    if ! python3 -u visualize.py \
            --model_dir "$model_dir" \
            --test_model "$ckpt" \
            --lora_behaviour "$behaviour" \
            --adaptive_lora_scenario "$scenario" \
            --human_num "$HUMAN_NUM" \
            --seed "$SEED" \
            --test_case "$episode_idx" \
            --save_slides \
            --save_episode_dump always \
            --exp_id "$exp_id" \
            --awareness_eval off \
            "${extra_args[@]}" \
            > /tmp/vis_${method}_${folder_tag}_ep${episode_idx}.log 2>&1; then
        echo "[$(date '+%H:%M:%S')] FAIL   ${method}  ×  ${folder_tag}  × ep=${episode_idx}"
        echo "       (see /tmp/vis_${method}_${folder_tag}_ep${episode_idx}.log)"
        fail_count=$((fail_count + 1))
        return 1
    fi

    # Pick the newest MP4 produced in default_mp4_dir at-or-after the cutoff.
    local src_mp4
    src_mp4=$(find "$default_mp4_dir" -maxdepth 1 -type f -name '*.mp4' -newermt "@$cutoff_ts" 2>/dev/null \
              | xargs -I{} ls -t {} 2>/dev/null | head -n 1)
    if [ -z "$src_mp4" ] || [ ! -f "$src_mp4" ]; then
        src_mp4=$(ls -t "$default_mp4_dir"/*.mp4 2>/dev/null | head -n 1)
    fi
    if [ -z "$src_mp4" ] || [ ! -f "$src_mp4" ]; then
        echo "[$(date '+%H:%M:%S')] FAIL   ${method}  ×  ${folder_tag}  × ep=${episode_idx}  (no MP4 produced)"
        fail_count=$((fail_count + 1))
        return 1
    fi

    mv "$src_mp4" "$dst_path"

    # Pull metadata from the visualize.py JSON dump and append to summary.json.
    # The dump is at <model_dir>/test/<scenario>_<behaviour>_exp<exp_id>.json.
    local dump_path="${model_dir}/test/${scenario}_${behaviour}_exp${exp_id}.json"
    python3 - "$dump_path" "$method" "$folder_tag" "$episode_idx" "$dst_path" "$SUMMARY_JSON" <<'PY' || true
import json, sys, os, math
dump_path, method, folder_tag, episode_idx, dst_path, summary_path = sys.argv[1:7]
if not os.path.exists(dump_path):
    sys.exit(0)
ep = json.load(open(dump_path))['episodes'][0]
scales = [sd.get('robot', {}).get('lora_scale') for sd in ep.get('steps_data', [])]
scales = [s for s in scales if s is not None]

# On Collision, identify which human the robot hit by picking the one with the
# smallest (centre-to-centre distance − combined radii) at the last step. Then
# read its actual_friendly flag: True → aware, False → ignorant.
collision_with = None
collided_human_id = None
if ep.get('result') == 'Collision' and ep.get('steps_data'):
    last = ep['steps_data'][-1]
    r = last.get('robot', {})
    rx, ry = r.get('pos', [None, None])
    r_rad  = r.get('radius', 0.0) or 0.0
    best   = None
    if rx is not None and ry is not None:
        for h in last.get('humans', []) or []:
            hp = h.get('pos') or [None, None]
            if hp[0] is None: continue
            slack = math.hypot(hp[0]-rx, hp[1]-ry) - (r_rad + (h.get('radius', 0.0) or 0.0))
            if best is None or slack < best[0]:
                best = (slack, h)
    if best is not None:
        h = best[1]
        collision_with = 'aware' if h.get('actual_friendly') else 'ignorant'
        collided_human_id = h.get('id')

row = {
    'method':         method,
    'scenario':       folder_tag,
    'ep':             int(episode_idx),
    'outcome':        ep.get('result'),
    'collision_with': collision_with,
    'collided_human_id': collided_human_id,
    'path_length':    round(float(ep.get('path_length', 0.0)), 3),
    'nav_time':       round(float(ep.get('time',        0.0)), 3),
    'steps':          ep.get('steps'),
    'lora_min':       round(min(scales), 4) if scales else None,
    'lora_max':       round(max(scales), 4) if scales else None,
    'lora_span':      round(max(scales) - min(scales), 4) if scales else None,
    'video':          dst_path,
}
try:
    with open(summary_path) as f:
        rows = json.load(f)
except Exception:
    rows = []
rows.append(row)
with open(summary_path, 'w') as f:
    json.dump(rows, f, indent=2)
PY

    echo "[$(date '+%H:%M:%S')] OK     ${method}  ×  ${folder_tag}  × ep=${episode_idx}  → $dst_path"
    ok_count=$((ok_count + 1))
}

for sc_entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r scenario folder_tag episodes_csv <<< "$sc_entry"
    IFS=',' read -ra episodes <<< "$episodes_csv"
    for ep in "${episodes[@]}"; do
        for m_entry in "${METHODS[@]}"; do
            IFS='|' read -r m_name m_dir m_ckpt m_beh <<< "$m_entry"
            run_combo "$m_name" "$m_dir" "$m_ckpt" "$m_beh" "$scenario" "$folder_tag" "$ep"
        done
    done
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo "=========================================================="
echo "Done in ${ELAPSED}s   OK=${ok_count}  FAIL=${fail_count}"
echo "Summary written to: $SUMMARY_JSON"
echo
echo "Quick preview (jq required for nice output):"
if command -v jq >/dev/null 2>&1; then
    jq -r '.[] | [.scenario, .method, .ep, .outcome, (.collision_with // "-"), .path_length, .nav_time, .lora_span] | @tsv' "$SUMMARY_JSON" \
        | sort | column -t -s $'\t' -N scenario,method,ep,outcome,coll_with,PL,nt,LoRA_span
else
    cat "$SUMMARY_JSON"
fi
echo "=========================================================="
