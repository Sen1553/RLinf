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

"""LIBERO-OOD environment using the benchmark's random-reset protocol."""

import numpy as np

from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.envs.utils import to_tensor


def get_libero_ood_env_seed(
    base_seed: int,
    task_id: int,
    trial_id: int,
    num_trials_per_task: int,
    is_eval: bool,
) -> int:
    """Return the construction seed for one logical OOD task/trial.

    Evaluation intentionally gives every task process the reference seed. During
    training, logical trial IDs are folded into the seed so different GRPO groups
    do not all begin from the same placement-sampler state.
    """
    if is_eval:
        return int(base_seed)
    return int(base_seed + task_id * num_trials_per_task + trial_id)


class LiberoOODEnv(LiberoEnv):
    """Run modified-LIBERO OOD tasks without fixed initial-state files.

    A reset ID is a logical ``(task_id, trial_id)`` identifier only. Unlike the
    standard :class:`LiberoEnv`, it is never resolved through
    ``get_task_init_states`` or passed to ``set_init_state``. Each task process
    is seeded when constructed; consecutive resets then advance the BDDL
    placement sampler instead of loading a fixed state.

    Evaluation uses seed 7 (or ``cfg.seed``) for every process to match the
    pi0-text-latent evaluator. Training derives a deterministic seed from each
    logical task/trial ID, while all members of a GRPO group still receive the
    same seed and initial placement.
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.num_trials_per_task = int(cfg.get("num_trials_per_task", 10))
        if self.num_trials_per_task <= 0:
            raise ValueError("num_trials_per_task must be positive.")
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    def _compute_total_num_group_envs(self):
        num_tasks = self.task_suite.get_num_tasks()
        self.trial_id_bins = [self.num_trials_per_task] * num_tasks
        self.total_num_group_envs = num_tasks * self.num_trials_per_task
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        if self.task_id_filter is None:
            self._valid_reset_state_ids = None
            return

        validated_task_ids = sorted({int(task_id) for task_id in self.task_id_filter})
        invalid = [
            task_id for task_id in validated_task_ids if not 0 <= task_id < num_tasks
        ]
        if invalid:
            raise ValueError(
                f"task_id_filter contains invalid task IDs {invalid}; "
                f"expected IDs in [0, {num_tasks - 1}]."
            )

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
                task_id=int(self.task_ids[env_id]),
                trial_id=int(self.trial_ids[env_id]),
                num_trials_per_task=self.num_trials_per_task,
                is_eval=self.is_eval,
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
