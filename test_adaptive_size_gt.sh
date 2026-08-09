#!/bin/bash
# Run adaptive GT evaluations for LoRA AB, action-space, and dense full-finetune across model sizes.
# Default GPUs exclude GPU 2 so it can be reserved for Conservative_giant training.

set -u

TEST_SIZE="${TEST_SIZE:-250}"
HUMAN_NUM="${HUMAN_NUM:-20}"
SEEDS="${SEEDS:-1000 2000 3000 4000}"
SCENARIOS="${SCENARIOS:-seperate_all_ignorant seperate_all_aware seperate_mixed_5050 cluster_aware_ignorant}"
SIZES="${SIZES:-small medium large}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
LOG_DIR="${LOG_DIR:-/tmp/adaptive_size_gt}"
AWARENESS_EVAL="${AWARENESS_EVAL:-off}"
SAVE_EPISODE_DUMP="${SAVE_EPISODE_DUMP:-never}"
DISCREPANCY_THRESHOLD="${DISCREPANCY_THRESHOLD:-0.15}"
DISCREPANCY_M="${DISCREPANCY_M:-1}"
EXP_NOTE="${EXP_NOTE:-}"
REQUIRE_FINAL="${REQUIRE_FINAL:-1}"
FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-02603.pt}"

# Keep GPU 2 free by default. Override explicitly if needed:
#   BLEND_GPUS=0,1,2,3 ./test_adaptive_size_gt.sh
BLEND_GPUS="${BLEND_GPUS:-0,1,3}"

mkdir -p "$LOG_DIR"

if [ -f "./gpu_affinity.sh" ]; then
    . ./gpu_affinity.sh
fi

read -r -a SEEDS_ARR <<< "$SEEDS"
read -r -a SCENARIOS_ARR <<< "$SCENARIOS"
read -r -a SIZES_ARR <<< "$SIZES"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --exp-note|--exp_note)
            EXP_NOTE="$2"
            shift 2
            ;;
        --exp-note=*|--exp_note=*)
            EXP_NOTE="${1#*=}"
            shift
            ;;
        *)
            log "WARN ignoring unknown argument: $1"
            shift
            ;;
    esac
done

checkpoint_name() {
    local model_dir="$1"
    if [ "$REQUIRE_FINAL" = "1" ]; then
        if [ -f "$model_dir/checkpoints/$FINAL_CHECKPOINT" ]; then
            echo "$FINAL_CHECKPOINT"
            return 0
        fi
        return 1
    fi

    local ckpt
    ckpt="$(find "$model_dir/checkpoints" -maxdepth 1 -type f -name '[0-9][0-9][0-9][0-9][0-9].pt' 2>/dev/null | sort | tail -n 1)"
    if [ -z "$ckpt" ]; then
        return 1
    fi
    basename "$ckpt"
}

lora_model_dir_for_size() {
    case "$1" in
        small)  echo "trained_models/LoRA_small" ;;
        medium) echo "trained_models/LoRA_medium" ;;
        large)  echo "trained_models/LoRA_large" ;;
        giant)  echo "trained_models/LoRA_giant" ;;
        *)      echo "trained_models/LoRA_$1" ;;
    esac
}

fullft_model_dir_for_size() {
    case "$1" in
        small)  echo "trained_models/Fullfinetune_small_v2" ;;
        medium) echo "trained_models/Fullfinetune_medium" ;;
        large)  echo "trained_models/Fullfinetune_large" ;;
        giant)  echo "trained_models/Fullfinetune_giant" ;;
        *)      echo "trained_models/Fullfinetune_$1" ;;
    esac
}

set_size_args() {
    SIZE_ARGS=()
    case "$1" in
        small)
            ;;
        medium)
            SIZE_ARGS=(
                --human_node_rnn_size 960
                --human_human_edge_rnn_size 256
                --human_node_output_size 1920
                --human_node_embedding_size 480
                --human_human_edge_embedding_size 64
                --attention_size 480
            )
            ;;
        large)
            SIZE_ARGS=(
                --human_node_rnn_size 1984
                --human_human_edge_rnn_size 256
                --human_node_output_size 7168
                --human_node_embedding_size 992
                --human_human_edge_embedding_size 64
                --attention_size 992
            )
            ;;
        giant)
            SIZE_ARGS=(
                --human_node_rnn_size 2816
                --human_human_edge_rnn_size 256
                --human_node_output_size 10240
                --human_node_embedding_size 1408
                --human_human_edge_embedding_size 64
                --attention_size 1408
            )
            ;;
        *)
            log "FAIL unknown size=$1; add its architecture args in set_size_args"
            return 2
            ;;
    esac
}

run_one() {
    local size="$1" kind="$2" scenario="$3" seed="$4" gpu="$5"
    local lora_dir fullft_dir lora_ckpt fullft_ckpt behaviour label exp_id log_path

    lora_dir="$(lora_model_dir_for_size "$size")"
    fullft_dir="$(fullft_model_dir_for_size "$size")"
    if ! set_size_args "$size"; then
        return 2
    fi

    if ! lora_ckpt="$(checkpoint_name "$lora_dir")"; then
        log "SKIP  ${kind}_${size}  missing required LoRA checkpoint in $lora_dir"
        return 0
    fi

    if [ "$kind" = "lora_ab" ]; then
        behaviour="Lora_AB"
        label="Lora_AB_${size}"
        exp_id="${label}_${seed}"
        [ -n "$EXP_NOTE" ] && exp_id="${EXP_NOTE}_${exp_id}"
        log_path="$LOG_DIR/${scenario}_${label}_seed${seed}.log"
        log "START $scenario x $label seed=$seed GPU=$gpu model=$lora_dir/$lora_ckpt"
        CUDA_VISIBLE_DEVICES="$gpu" python3 -u test.py \
            --model_dir "$lora_dir" \
            --test_model "$lora_ckpt" \
            --adaptive_lora_scenario "$scenario" \
            --lora_behaviour "$behaviour" \
            --discrepancy_threshold "$DISCREPANCY_THRESHOLD" \
            --discrepancy_m "$DISCREPANCY_M" \
            --human_num "$HUMAN_NUM" \
            --awareness_eval "$AWARENESS_EVAL" \
            --save_episode_dump "$SAVE_EPISODE_DUMP" \
            --seed "$seed" \
            --exp_id "$exp_id" \
            --test_size "$TEST_SIZE" \
            "${SIZE_ARGS[@]}" \
            > "$log_path" 2>&1
    elif [ "$kind" = "action" ]; then
        behaviour="adaptive_action_gt"
        label="adaptive_action${size}_gt"
        exp_id="${label}_${seed}"
        [ -n "$EXP_NOTE" ] && exp_id="${EXP_NOTE}_${exp_id}"
        log_path="$LOG_DIR/${scenario}_${label}_seed${seed}.log"
        log "START $scenario x $label seed=$seed GPU=$gpu model=$lora_dir/$lora_ckpt"
        CUDA_VISIBLE_DEVICES="$gpu" python3 -u test.py \
            --model_dir "$lora_dir" \
            --test_model "$lora_ckpt" \
            --adaptive_lora_scenario "$scenario" \
            --lora_behaviour "$behaviour" \
            --discrepancy_threshold "$DISCREPANCY_THRESHOLD" \
            --discrepancy_m "$DISCREPANCY_M" \
            --human_num "$HUMAN_NUM" \
            --awareness_eval "$AWARENESS_EVAL" \
            --save_episode_dump "$SAVE_EPISODE_DUMP" \
            --seed "$seed" \
            --exp_id "$exp_id" \
            --test_size "$TEST_SIZE" \
            "${SIZE_ARGS[@]}" \
            > "$log_path" 2>&1
    elif [ "$kind" = "fullfinetune" ]; then
        if ! fullft_ckpt="$(checkpoint_name "$fullft_dir")"; then
            log "SKIP  ${kind}_${size}  missing required full-finetune checkpoint in $fullft_dir"
            return 0
        fi
        behaviour="adaptive_fullfinetune_gt"
        label="adaptive_fullfinetune${size}_gt"
        exp_id="${label}_${seed}"
        [ -n "$EXP_NOTE" ] && exp_id="${EXP_NOTE}_${exp_id}"
        log_path="$LOG_DIR/${scenario}_${label}_seed${seed}.log"
        log "START $scenario x $label seed=$seed GPU=$gpu base=$lora_dir/$lora_ckpt ft=$fullft_dir/$fullft_ckpt"
        CUDA_VISIBLE_DEVICES="$gpu" python3 -u test.py \
            --model_dir "$lora_dir" \
            --test_model "$lora_ckpt" \
            --fullfinetune_model_dir "$fullft_dir" \
            --fullfinetune_test_model "$fullft_ckpt" \
            --adaptive_lora_scenario "$scenario" \
            --lora_behaviour "$behaviour" \
            --discrepancy_threshold "$DISCREPANCY_THRESHOLD" \
            --discrepancy_m "$DISCREPANCY_M" \
            --human_num "$HUMAN_NUM" \
            --awareness_eval "$AWARENESS_EVAL" \
            --save_episode_dump "$SAVE_EPISODE_DUMP" \
            --seed "$seed" \
            --exp_id "$exp_id" \
            --test_size "$TEST_SIZE" \
            "${SIZE_ARGS[@]}" \
            > "$log_path" 2>&1
    else
        log "FAIL unknown kind=$kind"
        return 2
    fi

    local status=$?
    if [ "$status" -eq 0 ]; then
        log "OK    $scenario x $label seed=$seed"
    else
        log "FAIL  $scenario x $label seed=$seed  see $log_path"
        tail -40 "$log_path" 2>/dev/null || true
    fi
    return "$status"
}

log "Adaptive size GT sweep starting"
log "SIZES=$SIZES"
log "SCENARIOS=$SCENARIOS"
log "SEEDS=$SEEDS"
log "BLEND_GPUS=$BLEND_GPUS MAX_PARALLEL=$MAX_PARALLEL TEST_SIZE=$TEST_SIZE HUMAN_NUM=$HUMAN_NUM LOG_DIR=$LOG_DIR"
log "REQUIRE_FINAL=$REQUIRE_FINAL FINAL_CHECKPOINT=$FINAL_CHECKPOINT"
log "Result labels use exp_id: Lora_AB_<size>_<seed>, adaptive_action<size>_gt_<seed>, adaptive_fullfinetune<size>_gt_<seed>"

running=0
job_index=0
for seed in "${SEEDS_ARR[@]}"; do
    for scenario in "${SCENARIOS_ARR[@]}"; do
        for size in "${SIZES_ARR[@]}"; do
            for kind in lora_ab action fullfinetune; do
                gpu="$(blend_gpu_for_job "$job_index")"
                run_one "$size" "$kind" "$scenario" "$seed" "$gpu" &
                job_index=$((job_index + 1))
                running=$((running + 1))
                if (( running >= MAX_PARALLEL )); then
                    wait -n
                    running=$((running - 1))
                fi
            done
        done
    done
done
wait

log "DONE adaptive size GT sweep"
