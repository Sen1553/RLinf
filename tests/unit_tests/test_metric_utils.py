import numpy as np
import pytest
import torch

from rlinf.utils.metric_utils import compute_evaluate_metrics


def test_compute_evaluate_metrics_groups_success_by_task_id():
    metrics = compute_evaluate_metrics(
        [
            {
                "task_id": torch.tensor([0, 1, 0]),
                "task_0_success": torch.tensor([True, False, True]),
                "task_1_success": torch.tensor([False, False, False]),
                "success_once": torch.tensor([True, False, True]),
                "return": torch.tensor([1.0, 0.0, 1.0]),
            },
            {
                "task_id": torch.tensor([1, 0]),
                "task_0_success": torch.tensor([False, False]),
                "task_1_success": torch.tensor([True, False]),
                "success_once": torch.tensor([True, False]),
                "return": torch.tensor([1.0, 0.0]),
            },
        ]
    )

    np.testing.assert_allclose(metrics["success_once"], 3 / 5)
    np.testing.assert_allclose(metrics["task_0_success"], 2 / 3)
    np.testing.assert_equal(metrics["task_0_success_total"], 3)
    np.testing.assert_allclose(metrics["task_1_success"], 0.5)
    np.testing.assert_equal(metrics["task_1_success_total"], 2)
    np.testing.assert_equal(metrics["num_trajectories"], 5)
    assert "task_id" not in metrics


def test_compute_evaluate_metrics_validates_task_metric_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        compute_evaluate_metrics(
            [
                {
                    "task_id": torch.tensor([0, 1]),
                    "success_once": torch.tensor([True]),
                }
            ]
        )
