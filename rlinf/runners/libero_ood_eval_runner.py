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

"""Evaluation runner with per-task LIBERO-OOD success metrics."""

from collections import defaultdict

import numpy as np
import torch

from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.metric_utils import compute_evaluate_metrics


def compute_libero_ood_task_metrics(eval_metrics_list):
    """Group completed trajectory success values by LIBERO task ID."""
    successes_by_task = defaultdict(list)
    for metrics in eval_metrics_list:
        if "task_id" not in metrics or "success_once" not in metrics:
            continue
        task_ids = torch.as_tensor(metrics["task_id"]).reshape(-1).cpu().numpy()
        successes = (
            torch.as_tensor(metrics["success_once"]).reshape(-1).bool().cpu().numpy()
        )
        if task_ids.shape != successes.shape:
            raise ValueError(
                "LIBERO-OOD task_id and success_once metrics must have matching shapes."
            )
        for task_id, success in zip(task_ids, successes, strict=True):
            successes_by_task[int(task_id)].append(bool(success))

    task_metrics = {}
    for task_id, successes in sorted(successes_by_task.items()):
        task_metrics[f"task_{task_id:02d}/success_once"] = np.asarray(
            np.mean(successes), dtype=np.float64
        )
        task_metrics[f"task_{task_id:02d}/num_trajectories"] = np.asarray(
            len(successes), dtype=np.int64
        )
    return task_metrics


class LiberoOODEvalRunner(EmbodiedEvalRunner):
    """Run distributed evaluation and retain task-level success rates."""

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if not env_decoupled_mode:
            rollout_handle.wait()

        eval_metrics_list = [result for result in env_results if result is not None]
        task_metrics = compute_libero_ood_task_metrics(eval_metrics_list)
        aggregate_inputs = [
            {
                key: value
                for key, value in metrics.items()
                if key not in ("task_id", "trial_id")
            }
            for metrics in eval_metrics_list
        ]
        eval_metrics = compute_evaluate_metrics(aggregate_inputs)
        eval_metrics.update(task_metrics)
        return eval_metrics
