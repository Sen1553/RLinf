import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs import SupportedEnvType
from rlinf.envs.libero.libero_ood_env import (
    LiberoOODEnv,
    get_libero_ood_env_seed,
    get_libero_ood_eval_trials_per_task,
)
from rlinf.runners.libero_ood_eval_runner import compute_libero_ood_task_metrics
from rlinf.workers.env.libero_ood_eval_worker import get_final_action_chunk_size


def test_libero_ood_env_type_is_registered():
    assert SupportedEnvType("libero_ood") == SupportedEnvType.LIBERO_OOD


def test_libero_ood_eval_trials_are_derived_from_parallel_capacity():
    assert get_libero_ood_eval_trials_per_task(10, 10, 1, 10) == 10
    assert get_libero_ood_eval_trials_per_task(30, 5, 1, 10) == 15
    assert get_libero_ood_eval_trials_per_task(20, 5, 2, 10) == 5


def test_libero_ood_eval_trials_require_balanced_task_coverage():
    with pytest.raises(ValueError, match="divisible"):
        get_libero_ood_eval_trials_per_task(8, 1, 1, 10)


def test_libero_ood_env_builds_eval_pool_without_trial_count_config():
    env = LiberoOODEnv.__new__(LiberoOODEnv)
    env.task_suite = type("TaskSuite", (), {"get_num_tasks": lambda self: 10})()
    env.task_id_filter = None
    env.is_eval = True
    env.cfg = OmegaConf.create(
        {"total_num_envs": 30, "rollout_epoch": 5, "group_size": 1}
    )

    env._compute_total_num_group_envs()

    assert env.trial_id_bins == [15] * 10
    assert env.total_num_group_envs == 150
    assert env._valid_reset_state_ids is None


def test_libero_ood_seed_uses_global_logical_environment_offset():
    assert get_libero_ood_env_seed(7, 0, 0, 2, 1) == 7
    assert get_libero_ood_env_seed(7, 1, 0, 2, 1) == 8
    assert get_libero_ood_env_seed(7, 0, 1, 2, 1) == 9
    assert get_libero_ood_env_seed(7, 1, 1, 2, 1) == 10


def test_libero_ood_seed_is_shared_within_rl_group():
    group_seeds = [get_libero_ood_env_seed(42, 0, env_id, 1, 10) for env_id in range(10)]
    assert group_seeds == [42] * 10
    assert get_libero_ood_env_seed(42, 0, 10, 1, 10) == 43


@pytest.mark.parametrize(
    ("max_episode_steps", "model_action_chunk", "expected"),
    [(300, 8, 4), (280, 8, 8), (300, 5, 5)],
)
def test_get_final_action_chunk_size(max_episode_steps, model_action_chunk, expected):
    assert (
        get_final_action_chunk_size(max_episode_steps, model_action_chunk) == expected
    )


def test_get_final_action_chunk_size_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="must be positive"):
        get_final_action_chunk_size(0, 8)


def test_compute_libero_ood_task_metrics_groups_workers_and_tasks():
    metrics = compute_libero_ood_task_metrics(
        [
            {
                "task_id": torch.tensor([0, 1, 0]),
                "success_once": torch.tensor([True, False, True]),
            },
            {
                "task_id": torch.tensor([1, 0]),
                "success_once": torch.tensor([True, False]),
            },
        ]
    )

    np.testing.assert_allclose(metrics["task_00/success_once"], 2 / 3)
    np.testing.assert_equal(metrics["task_00/num_trajectories"], 3)
    np.testing.assert_allclose(metrics["task_01/success_once"], 0.5)
    np.testing.assert_equal(metrics["task_01/num_trajectories"], 2)


def test_compute_libero_ood_task_metrics_validates_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        compute_libero_ood_task_metrics(
            [
                {
                    "task_id": torch.tensor([0, 1]),
                    "success_once": torch.tensor([True]),
                }
            ]
        )
