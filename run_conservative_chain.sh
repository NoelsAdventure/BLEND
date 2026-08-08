#!/bin/bash
# Run conservative size/backbone experiments and dependent LoRA/full-finetune jobs.
# One process is assigned to each GPU at a time. Dependent jobs start only after
# their required backbone training process exits successfully.

set -u

NUM_ENV_STEPS="${NUM_ENV_STEPS:-10000000}"
NUM_STEPS="${NUM_STEPS:-30}"
NUM_PROCESSES="${NUM_PROCESSES:-128}"
LORA_RANK="${LORA_RANK:-1}"
LOG_DIR="${LOG_DIR:-/tmp/blend_training_chain}"
mkdir -p "$LOG_DIR"

if [ -f "./gpu_affinity.sh" ]; then
    . ./gpu_affinity.sh
fi
BLEND_GPUS="${BLEND_GPUS:-$(blend_detect_gpus)}"
IFS=, read -r -a GPU_LIST <<< "$BLEND_GPUS"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
    GPU_LIST=(0)
fi

declare -A GPU_PID=()
declare -A GPU_JOB=()
declare -A JOB_PID=()
declare -A JOB_STATUS=()

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

is_pid_running() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

reap_finished_gpus() {
    local gpu pid job
    for gpu in "${GPU_LIST[@]}"; do
        pid="${GPU_PID[$gpu]:-}"
        if [ -n "$pid" ] && ! is_pid_running "$pid"; then
            job="${GPU_JOB[$gpu]:-unknown}"
            wait "$pid"
            local status=$?
            if [ "$status" -eq 0 ]; then
                log "OK    GPU=$gpu job=$job"
                JOB_STATUS[$job]="ok"
            else
                log "FAIL  GPU=$gpu job=$job status=$status"
                JOB_STATUS[$job]="failed:$status"
            fi
            unset GPU_PID[$gpu]
            unset GPU_JOB[$gpu]
        fi
    done
}

free_gpu() {
    reap_finished_gpus
    local gpu
    for gpu in "${GPU_LIST[@]}"; do
        if [ -z "${GPU_PID[$gpu]:-}" ]; then
            echo "$gpu"
            return 0
        fi
    done
    return 1
}

wait_for_free_gpu() {
    local gpu
    while true; do
        gpu="$(free_gpu || true)"
        if [ -n "$gpu" ]; then
            echo "$gpu"
            return 0
        fi
        sleep 30
    done
}

launch_job() {
    local gpu="$1" note="$2"
    shift 2
    local log_path="$LOG_DIR/${note}.log"
    log "START GPU=$gpu job=$note log=$log_path"
    CUDA_VISIBLE_DEVICES="$gpu" python train.py "$@" > "$log_path" 2>&1 &
    GPU_PID[$gpu]=$!
    GPU_JOB[$gpu]="$note"
    JOB_PID[$note]=${GPU_PID[$gpu]}
    JOB_STATUS[$note]="running"
}

largest_checkpoint() {
    local note="$1"
    local dir="trained_models/${note}/checkpoints"
    find "$dir" -maxdepth 1 -type f -name '[0-9][0-9][0-9][0-9][0-9].pt' 2>/dev/null | sort | tail -n 1
}

wait_for_job_success() {
    local note="$1"
    while true; do
        reap_finished_gpus
        case "${JOB_STATUS[$note]:-missing}" in
            ok)
                return 0
                ;;
            failed:*)
                log "Cannot continue: dependency $note ${JOB_STATUS[$note]}"
                exit 1
                ;;
        esac
        sleep 60
    done
}

wait_for_backbone_checkpoint() {
    local note="$1"
    local ckpt
    if [ -n "${JOB_PID[$note]:-}" ]; then
        wait_for_job_success "$note"
    fi
    ckpt="$(largest_checkpoint "$note")"
    if [ -z "$ckpt" ]; then
        log "Cannot continue: no checkpoint found for $note"
        exit 1
    fi
    log "READY checkpoint=$ckpt" >&2
    echo "$ckpt"
}

model_finished() {
    local note="$1"
    [ -n "$(largest_checkpoint "$note")" ] && [ -f "trained_models/${note}/training_summary.txt" ]
}

maybe_launch_backbone() {
    local note="$1"
    shift
    if model_finished "$note"; then
        log "SKIP  $note already has an existing completed run"
        return 0
    fi
    local gpu
    gpu="$(wait_for_free_gpu)"
    launch_job "$gpu" "$note" --note "$note" --robot-invisible --load-path "" "$@" --num-env-steps "$NUM_ENV_STEPS"
}

maybe_launch_dependent() {
    local note="$1" base_note="$2" mode="$3"
    shift 3
    local output_note="$note"
    if model_finished "$output_note"; then
        log "SKIP  $output_note already has an existing completed run"
        return 0
    fi
    local base_ckpt
    base_ckpt="$(wait_for_backbone_checkpoint "$base_note")"
    local gpu
    gpu="$(wait_for_free_gpu)"
    if [ "$mode" = "lora" ]; then
        launch_job "$gpu" "$note" --note "$note" --robot-visible --use-lora --lora-rank "$LORA_RANK" --lora-base-checkpoint "$base_ckpt" "$@" --num-env-steps "$NUM_ENV_STEPS"
    elif [ "$mode" = "resume" ]; then
        launch_job "$gpu" "$note" --note "$note" --robot-visible --resume --load-path "$base_ckpt" "$@" --num-env-steps "$NUM_ENV_STEPS"
    else
        log "Unknown dependent mode: $mode"
        exit 2
    fi
}

log "Training chain starting"
log "GPUs=$BLEND_GPUS NUM_ENV_STEPS=$NUM_ENV_STEPS NUM_PROCESSES=$NUM_PROCESSES LORA_RANK=$LORA_RANK LOG_DIR=$LOG_DIR"

maybe_launch_backbone Conservative_small
maybe_launch_backbone Conservative_medium \
    --human_node_rnn_size 960 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 1920 \
    --human_node_embedding_size 480 \
    --human_human_edge_embedding_size 64 \
    --attention_size 480
maybe_launch_backbone Conservative_large \
    --human_node_rnn_size 1984 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 7168 \
    --human_node_embedding_size 992 \
    --human_human_edge_embedding_size 64 \
    --attention_size 992

if ! model_finished "Fullfinetune_small"; then
    gpu="$(wait_for_free_gpu)"
    launch_job "$gpu" Fullfinetune_small \
        --note Fullfinetune_small \
        --robot-visible \
        --resume \
        --load-path trained_models/Ours_GST/checkpoints/05207.pt \
        --num-env-steps "$NUM_ENV_STEPS"
else
    log "SKIP  Fullfinetune_small already has an existing completed run"
fi

maybe_launch_dependent LoRA_medium Conservative_medium lora \
    --human_node_rnn_size 960 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 1920 \
    --human_node_embedding_size 480 \
    --human_human_edge_embedding_size 64 \
    --attention_size 480
maybe_launch_dependent Fullfinetune_medium Conservative_medium resume \
    --human_node_rnn_size 960 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 1920 \
    --human_node_embedding_size 480 \
    --human_human_edge_embedding_size 64 \
    --attention_size 480
maybe_launch_dependent LoRA_large Conservative_large lora \
    --human_node_rnn_size 1984 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 7168 \
    --human_node_embedding_size 992 \
    --human_human_edge_embedding_size 64 \
    --attention_size 992
maybe_launch_dependent Fullfinetune_large Conservative_large resume \
    --human_node_rnn_size 1984 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 7168 \
    --human_node_embedding_size 992 \
    --human_human_edge_embedding_size 64 \
    --attention_size 992
maybe_launch_dependent LoRA_small Conservative_small lora
maybe_launch_dependent Fullfinetune_small_v2 Conservative_small resume

while true; do
    reap_finished_gpus
    active=0
    for gpu in "${GPU_LIST[@]}"; do
        if [ -n "${GPU_PID[$gpu]:-}" ]; then
            active=$((active + 1))
        fi
    done
    [ "$active" -eq 0 ] && break
    sleep 30
done

log "DONEEEE"
