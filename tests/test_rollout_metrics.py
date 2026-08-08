import numpy as np

from scripts.run_libero_rollout import _action_statistics


def test_action_statistics_records_finite_boundary_and_gripper_metrics() -> None:
    first = np.zeros(7, dtype=np.float32)
    second = np.ones(7, dtype=np.float32)
    third = np.ones(7, dtype=np.float32)
    third[6] = -1.0

    stats = _action_statistics([first, second, third], boundary_stride=2)

    assert stats["action_value_count"] == 21
    assert stats["action_finite_count"] == 21
    assert stats["action_nonfinite_count"] == 0
    assert stats["action_finite_ratio"] == 1.0
    assert stats["chunk_boundary_count"] == 1
    assert stats["chunk_boundary_jump_mean"] == 2.0
    assert stats["gripper_switch_count"] == 2


def test_action_statistics_does_not_emit_nan_for_one_action() -> None:
    stats = _action_statistics([np.zeros(7, dtype=np.float32)])

    assert stats["action_smoothness_mean"] is None
    assert stats["action_smoothness_max"] is None
    assert stats["chunk_boundary_count"] == 0
