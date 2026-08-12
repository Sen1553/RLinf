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

"""Environment worker for exact LIBERO-OOD episode horizons."""

import math

from rlinf.workers.env.env_worker import EnvWorker


def get_final_action_chunk_size(max_episode_steps: int, model_action_chunk: int) -> int:
    """Return how many actions to execute from the final model prediction."""
    if max_episode_steps <= 0 or model_action_chunk <= 0:
        raise ValueError("max_episode_steps and model_action_chunk must be positive.")
    remainder = max_episode_steps % model_action_chunk
    return remainder or model_action_chunk


class LiberoOODEvalEnvWorker(EnvWorker):
    """Trim the final action chunk so OOD episodes stop at the exact horizon."""

    def __init__(self, cfg):
        super().__init__(cfg)
        max_episode_steps = int(cfg.env.eval.max_episode_steps)
        model_action_chunk = int(self.model_cfg.num_action_chunks)
        expected_chunk_steps = math.ceil(max_episode_steps / model_action_chunk)
        if self.n_eval_chunk_steps != expected_chunk_steps:
            raise ValueError(
                "For LIBERO-OOD, env.eval.max_steps_per_rollout_epoch must equal "
                "ceil(max_episode_steps / num_action_chunks) * num_action_chunks; "
                f"got {cfg.env.eval.max_steps_per_rollout_epoch}."
            )
        self._final_action_chunk_size = get_final_action_chunk_size(
            max_episode_steps, model_action_chunk
        )
        self._eval_chunk_indices = [0] * self.stage_num

    def env_evaluate_step(self, raw_actions, stage_id):
        chunk_index = self._eval_chunk_indices[stage_id] % self.n_eval_chunk_steps
        if chunk_index == self.n_eval_chunk_steps - 1:
            raw_actions = raw_actions[:, : self._final_action_chunk_size]

        result = super().env_evaluate_step(raw_actions, stage_id)
        self._eval_chunk_indices[stage_id] += 1
        return result
