#!/bin/bash
#
# Helpers for assigning background experiment jobs across visible CUDA devices.
# Override with BLEND_GPUS, for example: BLEND_GPUS=0,1,2,3 ./test_baselines.sh

_blend_gpu_list_is_valid() {
    [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]
}

blend_detect_gpus() {
    if [ -n "${BLEND_GPUS:-}" ]; then
        if _blend_gpu_list_is_valid "$BLEND_GPUS"; then
            printf "%s" "$BLEND_GPUS"
            return
        fi
        echo "Ignoring invalid BLEND_GPUS='$BLEND_GPUS'; expected comma-separated CUDA ids like 0,1,2,3" >&2
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local ids
        ids="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d '[:blank:]' | paste -sd, -)"
        if [ -n "$ids" ] && _blend_gpu_list_is_valid "$ids"; then
            printf "%s" "$ids"
            return
        fi
    fi

    if command -v python3 >/dev/null 2>&1; then
        local count
        count="$(python3 -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || true)"
        if [ "${count:-0}" -gt 0 ] 2>/dev/null; then
            seq -s, 0 $((count - 1))
            return
        fi
    fi

    printf "0"
}

blend_gpu_for_job() {
    local job_index="$1"
    local gpu_list="${BLEND_GPUS:-$(blend_detect_gpus)}"
    if ! _blend_gpu_list_is_valid "$gpu_list"; then
        gpu_list="$(BLEND_GPUS= blend_detect_gpus)"
    fi
    local old_ifs="$IFS"
    IFS=,
    read -r -a gpus <<< "$gpu_list"
    IFS="$old_ifs"

    local count="${#gpus[@]}"
    if [ "$count" -eq 0 ]; then
        printf "0"
    else
        printf "%s" "${gpus[$((job_index % count))]}"
    fi
}
