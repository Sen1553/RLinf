#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

# POLARIS dataset
export POLARIS_DATA_PATH=${POLARIS_DATA_PATH:-"/path/to/dataset/PolaRiS-Hub"}

if [ -z "${1:-}" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"maniskill_ppo_openvlaoft"}
else
    CONFIG_NAME=$1
    shift
fi

# LIBERO-OOD reuses the normal embodied entrypoint, but it must import the
# benchmark's modified LIBERO package before the standard installed package.
CONFIG_BASENAME=${CONFIG_NAME##*/}
if [[ "${CONFIG_BASENAME}" == libero_goal_ood* || \
      "${CONFIG_BASENAME}" == libero_spatial_ood* || \
      "${CONFIG_BASENAME}" == libero_object_ood* ]]; then
    MODIFIED_LIBERO_ROOT=${LIBERO_OOD_ROOT:-"${REPO_PATH}/third_party/modified_libero"}
    LIBERO_CORE="${MODIFIED_LIBERO_ROOT}/libero/libero"
    if [ ! -d "${LIBERO_CORE}/bddl_files/libero_goal_ood" ]; then
        echo "Modified LIBERO OOD tasks were not found under ${LIBERO_CORE}." >&2
        echo "Place modified LIBERO at ${REPO_PATH}/third_party/modified_libero or set LIBERO_OOD_ROOT." >&2
        exit 1
    fi

    export PYTHONPATH="${MODIFIED_LIBERO_ROOT}:${PYTHONPATH}"
    export LIBERO_TYPE=standard
    export LIBERO_CONFIG_PATH=${LIBERO_OOD_CONFIG_PATH:-"${REPO_PATH}/logs/libero_ood_config"}
    export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-"${REPO_PATH}/logs/numba_cache"}
    export MPLCONFIGDIR=${MPLCONFIGDIR:-"${REPO_PATH}/logs/matplotlib_config"}
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
    mkdir -p "${LIBERO_CONFIG_PATH}" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
    python -c 'import json, pathlib, sys; core=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]); datasets=out/"datasets"; datasets.mkdir(exist_ok=True); data={"benchmark_root":str(core),"bddl_files":str(core/"bddl_files"),"init_states":str(core/"init_files"),"datasets":str(datasets.resolve()),"assets":str(core/"assets")}; (out/"config.yaml").write_text(json.dumps(data, indent=2), encoding="utf-8")' "${LIBERO_CORE}" "${LIBERO_CONFIG_PATH}"
    echo "Using modified LIBERO at ${MODIFIED_LIBERO_ROOT}"
fi

# NOTE: Set the active robot platform (required for correct action dimension and
# normalization). For backward compatibility, the first remaining positional
# argument is treated as a platform only when it has no "=". All other arguments
# are forwarded to Hydra as overrides.
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then
    ROBOT_PLATFORM=$1
    shift
else
    ROBOT_PLATFORM=${ROBOT_PLATFORM:-"LIBERO"}
fi
HYDRA_OVERRIDES=("$@")

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR=${RUN_LOG_DIR:-"${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"}
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
# Forward optional overrides exported by callers (e.g. tests/parity_tests/run_all.sh).
# Sentinel: "-2" means "do not override, use YAML default". -1 is a legitimate value
# (e.g. runner.max_steps=-1 means unlimited) and is forwarded as-is.
EXTRA_OVERRIDES=()
[ -n "${STEPS:-}" ]      && [ "$STEPS"      != "-2" ] && EXTRA_OVERRIDES+=("runner.max_steps=${STEPS}")
[ -n "${SAVE_INTER:-}" ] && [ "$SAVE_INTER" != "-2" ] && EXTRA_OVERRIDES+=("runner.save_interval=${SAVE_INTER}")
[ -n "${NODES:-}" ]      && [ "$NODES"      != "-2" ] && EXTRA_OVERRIDES+=("cluster.num_nodes=${NODES}")

CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "${EXTRA_OVERRIDES[@]}"
    "${HYDRA_OVERRIDES[@]}"
)
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
