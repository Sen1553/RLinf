# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO-OOD environment using BDDL-sampled initial placements."""

import math

import numpy as np

from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.envs.utils import to_tensor


def get_libero_ood_env_seed(
    base_seed: int,
    process_index: int,
    local_env_id: int,
    total_num_processes: int,
    group_size: int,
) -> int:
    """Return a deterministic seed for one logical parallel environment.

    This follows LIBERO's base-seed-plus-process-offset convention while also
    distinguishing vectorized environments inside one process. Members of the
    same RL group intentionally share a seed; different groups use independent
    placement-sampler streams.
    """
    if process_index < 0 or local_env_id < 0:
        raise ValueError("process_index and local_env_id must be non-negative.")
    if total_num_processes <= 0 or group_size <= 0:
        raise ValueError("total_num_processes and group_size must be positive.")
    if process_index >= total_num_processes:
        raise ValueError("process_index must be smaller than total_num_processes.")
    logical_local_env_id = local_env_id // group_size
    logical_global_env_id = (
        logical_local_env_id * total_num_processes + process_index
    )
    return int(base_seed + logical_global_env_id)


def get_libero_ood_eval_trials_per_task(
    total_num_envs: int,
    rollout_epoch: int,
    group_size: int,
    num_tasks: int,
) -> int:
    """Derive the number of unique evaluation episodes for each OOD task."""
    if total_num_envs <= 0 or rollout_epoch <= 0:
        raise ValueError("total_num_envs and rollout_epoch must be positive.")
    if group_size <= 0 or num_tasks <= 0:
        raise ValueError("group_size and num_tasks must be positive.")
    if total_num_envs % group_size != 0:
        raise ValueError("total_num_envs must be divisible by group_size.")

    total_eval_episodes = total_num_envs // group_size * rollout_epoch
    if total_eval_episodes % num_tasks != 0:
        raise ValueError(
            "LIBERO-OOD evaluation requires "
            "(total_num_envs / group_size) * rollout_epoch to be divisible "
            f"by the number of evaluated tasks ({num_tasks}); got "
            f"{total_eval_episodes} logical episodes."
        )
    return total_eval_episodes // num_tasks


class LiberoOODEnv(LiberoEnv):
    """Run modified-LIBERO OOD tasks without fixed initial-state files.

    A reset ID is a logical ``(task_id, trial_id)`` identifier only. Unlike the
    standard :class:`LiberoEnv`, it is never resolved through
    ``get_task_init_states`` or passed to ``set_init_state``. Each task process
    is seeded when constructed; consecutive resets then advance the BDDL
    placement sampler instead of loading a fixed state.

    Evaluation derives its logical trial budget from ``total_num_envs`` and
    ``rollout_epoch``. No fixed initial-state file or per-task trial-count option
    is required. Seeds follow the parallel environment layout, and consecutive
    resets advance each environment's placement-sampler RNG stream.
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    def _compute_total_num_group_envs(self):
        num_tasks = self.task_suite.get_num_tasks()
        validated_task_ids = (
            list(range(num_tasks))
            if self.task_id_filter is None
            else sorted({int(task_id) for task_id in self.task_id_filter})
        )
        invalid = [
            task_id for task_id in validated_task_ids if not 0 <= task_id < num_tasks
        ]
        if invalid:
            raise ValueError(
                f"task_id_filter contains invalid task IDs {invalid}; "
                f"expected IDs in [0, {num_tasks - 1}]."
            )
        if not validated_task_ids:
            raise ValueError("task_id_filter must select at least one task.")

        total_num_envs = int(self.cfg.total_num_envs)
        group_size = int(self.cfg.group_size)
        if total_num_envs <= 0 or group_size <= 0:
            raise ValueError("total_num_envs and group_size must be positive.")
        if total_num_envs % group_size != 0:
            raise ValueError("total_num_envs must be divisible by group_size.")
        num_logical_envs = total_num_envs // group_size
        if self.is_eval:
            trials_per_task = get_libero_ood_eval_trials_per_task(
                total_num_envs=total_num_envs,
                rollout_epoch=int(self.cfg.rollout_epoch),
                group_size=group_size,
                num_tasks=len(validated_task_ids),
            )
        else:
            # Training samples reset IDs rather than exhausting a finite pool.
            # Keep enough logical IDs to seed the currently parallel RL groups.
            trials_per_task = max(
                1, math.ceil(num_logical_envs / len(validated_task_ids))
            )

        self.trial_id_bins = [trials_per_task] * num_tasks
        self.total_num_group_envs = num_tasks * trials_per_task
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        if self.task_id_filter is None:
            self._valid_reset_state_ids = None
            return

        reset_state_ids = []
        for task_id in validated_task_ids:
            start = self.cumsum_trial_id_bins[task_id - 1] if task_id > 0 else 0
            reset_state_ids.extend(range(start, self.cumsum_trial_id_bins[task_id]))
        self._valid_reset_state_ids = np.asarray(reset_state_ids, dtype=np.int64)

    def get_env_fn_params(self, env_idx=None):
        if env_idx is None:
            selected_env_ids = np.arange(self.num_envs)
        else:
            # LiberoEnv emits parameters in ascending environment-ID order.
            selected_env_ids = np.sort(np.asarray(env_idx, dtype=np.int64))
        env_fn_params = super().get_env_fn_params(env_idx)
        for params, env_id in zip(env_fn_params, selected_env_ids, strict=True):
            params["seed"] = get_libero_ood_env_seed(
                base_seed=int(self.cfg.seed),
                process_index=int(self.seed_offset),
                local_env_id=int(env_id),
                total_num_processes=int(self.total_num_processes),
                group_size=int(self.group_size),
            )
        return env_fn_params

    def _reconfigure(self, reset_state_ids, env_idx):
        """Reset from the BDDL sampler without loading an initial-state file."""
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        reconfigure_env_idx = []
        for index, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[index]
            self.task_ids[env_id] = task_ids[index]
            self.trial_ids[env_id] = trial_ids[index]
            if task_changed:
                reconfigure_env_idx.append(env_id)

        if reconfigure_env_idx:
            env_fn_params = self.get_env_fn_params(reconfigure_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfigure_env_idx)

        # Deliberately do not call env.seed() here. Repeated reset() calls must
        # consume the placement sampler's RNG stream instead of restarting it.
        self.env.reset(id=env_idx)

    def _record_metrics(self, step_reward, terminations, infos):
        infos = super()._record_metrics(step_reward, terminations, infos)
        infos["episode"]["task_id"] = to_tensor(self.task_ids.copy())
        infos["episode"]["trial_id"] = to_tensor(self.trial_ids.copy())
        return infos
