#!/usr/bin/env bash

# Shared helpers for continual LIBERO task training.

parse_task_spec() {
    local task_spec=${1:-}
    local task_start
    local task_end

    if [[ "${task_spec}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "${task_spec}"
        return 0
    fi

    if [[ ! "${task_spec}" =~ ^[[:space:]]*([0-9]+)[[:space:]]*,[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
        echo "Invalid task specification '${task_spec}'. Use N or 'START,END'." >&2
        return 1
    fi

    task_start=${BASH_REMATCH[1]}
    task_end=${BASH_REMATCH[2]}
    if (( task_start >= task_end )); then
        echo "Task range must satisfy START < END, got ${task_start},${task_end}." >&2
        return 1
    fi

    seq "${task_start}" "${task_end}"
}

validate_task_ids() {
    local max_tasks=$1
    shift
    local task_id
    for task_id in "$@"; do
        if (( task_id < 0 || task_id >= max_tasks )); then
            echo "Task ID ${task_id} is outside [0, $((max_tasks - 1))]." >&2
            return 1
        fi
    done
}

find_latest_full_weights() {
    local search_root=$1
    find "${search_root}" -type f \
        -path '*/actor/model_state_dict/full_weights.pt' -print 2>/dev/null \
        | sort -V \
        | tail -n 1
}

resolve_full_weights_checkpoint() {
    local checkpoint_path=${1:-}
    local resolved

    if [ -z "${checkpoint_path}" ]; then
        return 0
    fi

    if [ -f "${checkpoint_path}" ]; then
        resolved=$(realpath "${checkpoint_path}")
    elif [ -d "${checkpoint_path}" ]; then
        resolved=$(find_latest_full_weights "${checkpoint_path}")
    else
        echo "Checkpoint does not exist: ${checkpoint_path}" >&2
        return 1
    fi

    if [ -z "${resolved}" ] || [ ! -f "${resolved}" ]; then
        echo "No actor/model_state_dict/full_weights.pt found under ${checkpoint_path}." >&2
        return 1
    fi
    printf '%s\n' "${resolved}"
}

sanitize_config_name() {
    local config_name=$1
    printf '%s' "${config_name//\//_}" | tr -c '[:alnum:]_.-' '_'
}
