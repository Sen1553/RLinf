import numpy as np
import pytest
import torch

from rlinf.envs import SupportedEnvType
from rlinf.runners.libero_ood_eval_runner import compute_libero_ood_task_metrics
from rlinf.workers.env.libero_ood_eval_worker import get_final_action_chunk_size


def test_libero_ood_env_type_is_registered():
    assert SupportedEnvType("libero_ood") == SupportedEnvType.LIBERO_OOD


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
