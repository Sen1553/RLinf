#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: MODEL_PATH=/path/to/checkpoint $0 <config_name> [hydra_overrides...]" >&2
    exit 1
fi

EVALUATIONS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${EVALUATIONS_PATH}")"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
if [ -n "${LIBERO_OOD_ROOT:-}" ]; then
    MODIFIED_LIBERO_ROOT="${LIBERO_OOD_ROOT}"
else
    MODIFIED_LIBERO_ROOT="${REPO_PATH}/third_party/modified_libero"
fi
LIBERO_CORE="${MODIFIED_LIBERO_ROOT}/libero/libero"
CONFIG_NAME="$1"
shift

if [ ! -d "${LIBERO_CORE}/bddl_files/libero_goal_ood" ]; then
    echo "Modified LIBERO OOD tasks were not found under ${LIBERO_CORE}." >&2
    echo "Place modified LIBERO at ${REPO_PATH}/third_party/modified_libero or set LIBERO_OOD_ROOT." >&2
    exit 1
fi
if [ ! -f "${EVALUATIONS_PATH}/libero_ood/${CONFIG_NAME}.yaml" ]; then
    echo "Unknown LIBERO-OOD config: ${CONFIG_NAME}" >&2
    exit 1
fi

export EVALUATIONS_PATH REPO_PATH EMBODIED_PATH
export PYTHONPATH="${MODIFIED_LIBERO_ROOT}:${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Use a dedicated path so importing modified LIBERO never rewrites ~/.libero.
export LIBERO_CONFIG_PATH="${LIBERO_OOD_CONFIG_PATH:-${REPO_PATH}/logs/libero_ood_config}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_PATH}/logs/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_PATH}/logs/matplotlib_config}"
mkdir -p "${LIBERO_CONFIG_PATH}" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
python -c 'import json, pathlib, sys; core=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]); datasets=out/"datasets"; datasets.mkdir(exist_ok=True); data={"benchmark_root":str(core),"bddl_files":str(core/"bddl_files"),"init_states":str(core/"init_files"),"datasets":str(datasets.resolve()),"assets":str(core/"assets")}; (out/"config.yaml").write_text(json.dumps(data, indent=2), encoding="utf-8")' "${LIBERO_CORE}" "${LIBERO_CONFIG_PATH}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"

CMD=(
    python "${EVALUATIONS_PATH}/eval_libero_ood_agent.py"
    --config-path "${EVALUATIONS_PATH}/libero_ood/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
)
if [ -n "${MODEL_PATH:-}" ]; then
    CMD+=("rollout.model.model_path=${MODEL_PATH}")
fi
if [ $# -gt 0 ]; then
    CMD+=("$@")
fi

echo "${CMD[*]}" | tee "${LOG_DIR}/eval_libero_ood.log"
"${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/eval_libero_ood.log"
