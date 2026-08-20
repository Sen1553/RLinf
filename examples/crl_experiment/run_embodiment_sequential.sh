#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash examples/crl_experiment/run_embodiment_sequential.sh \
    TASK_ID_OR_RANGE [INITIAL_LORA_FULL_WEIGHTS] [MAX_EPOCHS] [CONFIG_NAME] [SEED]

Examples:
  # Train only task 0 of the default LIBERO-Spatial config.
  bash examples/crl_experiment/run_embodiment_sequential.sh 0

  # Train tasks 0, 1, 2, 3, and 4 sequentially.
  bash examples/crl_experiment/run_embodiment_sequential.sh "0,4"

  # Train Goal-OOD tasks 0 through 4 with StarVLA.
  bash examples/crl_experiment/run_embodiment_sequential.sh \
    "0,4" "" 10 libero_goal_ood_grpo_starvla_qwen25 42

TASK_ID_OR_RANGE is either one integer or an inclusive START,END range.
INITIAL_LORA_FULL_WEIGHTS must be an RLinf v0.3 full_weights.pt produced with
actor.model.is_lora=true, or a directory that contains one. It initializes only
the first requested task.

Environment overrides:
  CRL_CONFIG_NAME     Default training config name.
  CRL_MAX_EPOCHS      Epochs per task (default: 10).
  CRL_LORA_RANK       LoRA rank used for every task (default: 32).
  CRL_NUM_TASKS       Number of tasks in the suite (default: 10).
  CRL_OUTPUT_ROOT     Parent directory for the sequence run.
  CRL_RUN_ID          Stable run directory name instead of a timestamp.
  CRL_EMBODIMENT_LAUNCHER  Alternate launcher path (primarily for testing).
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$(dirname "${SCRIPT_DIR}")")
source "${SCRIPT_DIR}/common_functions.sh"

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

TASK_SPEC=$1
INITIAL_CHECKPOINT=${2:-}
MAX_EPOCHS=${3:-${CRL_MAX_EPOCHS:-10}}
CONFIG_NAME=${4:-${CRL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_starvla_qwen25}}
SEED=${5:-${CRL_SEED:-42}}
LORA_RANK=${CRL_LORA_RANK:-32}
NUM_TASKS=${CRL_NUM_TASKS:-10}
EMBODIMENT_LAUNCHER=${CRL_EMBODIMENT_LAUNCHER:-"${REPO_ROOT}/examples/embodiment/run_embodiment.sh"}

if [[ ! "${MAX_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_EPOCHS must be a positive integer, got '${MAX_EPOCHS}'." >&2
    exit 2
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer, got '${SEED}'." >&2
    exit 2
fi
if [[ ! "${LORA_RANK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CRL_LORA_RANK must be a positive integer, got '${LORA_RANK}'." >&2
    exit 2
fi
if [[ ! "${NUM_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CRL_NUM_TASKS must be a positive integer, got '${NUM_TASKS}'." >&2
    exit 2
fi
if [ ! -f "${REPO_ROOT}/examples/embodiment/config/${CONFIG_NAME}.yaml" ]; then
    echo "Training config not found: examples/embodiment/config/${CONFIG_NAME}.yaml" >&2
    exit 2
fi
if [ ! -f "${EMBODIMENT_LAUNCHER}" ]; then
    echo "Embodied launcher not found: ${EMBODIMENT_LAUNCHER}" >&2
    exit 2
fi

mapfile -t TASK_IDS < <(parse_task_spec "${TASK_SPEC}")
if [ ${#TASK_IDS[@]} -eq 0 ]; then
    echo "No task IDs were parsed from '${TASK_SPEC}'." >&2
    exit 2
fi
validate_task_ids "${NUM_TASKS}" "${TASK_IDS[@]}" || exit 2

CURRENT_CHECKPOINT=""
if [ -n "${INITIAL_CHECKPOINT}" ]; then
    CURRENT_CHECKPOINT=$(resolve_full_weights_checkpoint "${INITIAL_CHECKPOINT}") \
        || exit 2
fi

CONFIG_TAG=$(sanitize_config_name "${CONFIG_NAME}")
RUN_ID=${CRL_RUN_ID:-"$(date +'%Y%m%d-%H%M%S')-tasks_${TASK_IDS[0]}_${TASK_IDS[-1]}_seed${SEED}"}
OUTPUT_ROOT=${CRL_OUTPUT_ROOT:-"${REPO_ROOT}/logs/continual/${CONFIG_TAG}/${RUN_ID}"}

if [ -e "${OUTPUT_ROOT}" ]; then
    echo "Output directory already exists: ${OUTPUT_ROOT}" >&2
    echo "Choose another CRL_RUN_ID or CRL_OUTPUT_ROOT to avoid overwriting it." >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

printf 'Sequential LoRA fine-tuning on LIBERO\n'
printf '  Config: %s\n' "${CONFIG_NAME}"
printf '  Tasks: %s\n' "${TASK_IDS[*]}"
printf '  Epochs per task: %s\n' "${MAX_EPOCHS}"
printf '  LoRA rank: %s\n' "${LORA_RANK}"
printf '  Seed: %s\n' "${SEED}"
printf '  Output: %s\n' "${OUTPUT_ROOT}"
if [ -n "${CURRENT_CHECKPOINT}" ]; then
    printf '  Initial weights: %s\n' "${CURRENT_CHECKPOINT}"
fi

for task_id in "${TASK_IDS[@]}"; do
    TASK_RUN_DIR="${OUTPUT_ROOT}/task_${task_id}"
    EXPERIMENT_NAME="continual_task_${task_id}"
    mkdir -p "${TASK_RUN_DIR}"

    OVERRIDES=(
        "++env.train.task_id_filter=[${task_id}]"
        "actor.model.is_lora=true"
        "actor.model.lora_rank=${LORA_RANK}"
        "actor.model.lora_path=null"
        "env.train.seed=${SEED}"
        "actor.seed=${SEED}"
        "runner.logger.experiment_name=${EXPERIMENT_NAME}"
        "runner.max_epochs=${MAX_EPOCHS}"
        "runner.max_steps=-1"
        "runner.val_check_interval=-1"
        "runner.save_interval=${MAX_EPOCHS}"
    )
    if [ -n "${CURRENT_CHECKPOINT}" ]; then
        OVERRIDES+=("runner.ckpt_path=${CURRENT_CHECKPOINT}")
    else
        OVERRIDES+=("runner.ckpt_path=null")
    fi

    echo
    echo "============================================================"
    echo "Training task ${task_id}"
    echo "  Log directory: ${TASK_RUN_DIR}"
    if [ -n "${CURRENT_CHECKPOINT}" ]; then
        echo "  Initial weights: ${CURRENT_CHECKPOINT}"
    else
        echo "  Initial weights: base model plus a new rank-${LORA_RANK} LoRA adapter"
    fi
    echo "============================================================"

    RUN_LOG_DIR="${TASK_RUN_DIR}" \
        bash "${EMBODIMENT_LAUNCHER}" \
        "${CONFIG_NAME}" "${OVERRIDES[@]}"
    exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        echo "Task ${task_id} failed with exit code ${exit_code}; stopping sequence." >&2
        exit ${exit_code}
    fi

    EXPECTED_CHECKPOINT="${TASK_RUN_DIR}/${EXPERIMENT_NAME}/checkpoints/global_step_${MAX_EPOCHS}/actor/model_state_dict/full_weights.pt"
    if [ -f "${EXPECTED_CHECKPOINT}" ]; then
        CURRENT_CHECKPOINT=$(realpath "${EXPECTED_CHECKPOINT}")
    else
        CURRENT_CHECKPOINT=$(find_latest_full_weights "${TASK_RUN_DIR}")
    fi
    if [ -z "${CURRENT_CHECKPOINT}" ] || [ ! -f "${CURRENT_CHECKPOINT}" ]; then
        echo "Task ${task_id} finished but no RLinf v0.3 LoRA full_weights.pt was found." >&2
        exit 3
    fi

    printf '%s\n' "${CURRENT_CHECKPOINT}" > "${OUTPUT_ROOT}/latest_checkpoint.txt"
    echo "Task ${task_id} complete: ${CURRENT_CHECKPOINT}"
done

echo
echo "Sequential training completed successfully."
echo "Final checkpoint: ${CURRENT_CHECKPOINT}"
echo "Checkpoint pointer: ${OUTPUT_ROOT}/latest_checkpoint.txt"
