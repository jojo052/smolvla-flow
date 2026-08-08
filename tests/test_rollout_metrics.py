import numpy as np
import pytest
import torch

from scripts.run_libero_rollout import (
    _action_statistics,
    _run_sync_episode,
    _sleep_for_remaining_period,
)


def test_action_statistics_records_finite_boundary_and_gripper_metrics() -> None:
    first = np.zeros(7, dtype=np.float32)
    second = np.ones(7, dtype=np.float32)
    third = np.ones(7, dtype=np.float32)
    third[6] = -1.0

    stats = _action_statistics([first, second, third], sequence_ids=[1, 1, 2])

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


def test_action_statistics_requires_sequence_alignment() -> None:
    try:
        _action_statistics(
            [np.zeros(7, dtype=np.float32)],
            sequence_ids=[],
        )
    except ValueError as error:
        assert "align" in str(error)
    else:
        raise AssertionError("sequence_ids length mismatch should fail")


def test_sync_episode_applies_and_records_gripper_hysteresis() -> None:
    class Policy:
        def __init__(self) -> None:
            self._actions = iter((-1.0, 0.0, 1.0))
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def select_action(self, batch) -> torch.Tensor:
            del batch
            action = torch.zeros(1, 7)
            action[0, 6] = next(self._actions)
            return action

    class Env:
        def __init__(self) -> None:
            self.actions: list[np.ndarray] = []

        def reset(self, *, seed: int):
            assert seed == 5
            return {"frame": 0}, {}

        def step(self, action: np.ndarray):
            self.actions.append(action.copy())
            done = len(self.actions) == 3
            return {"frame": len(self.actions)}, 0.0, done, False, {"is_success": False}

    env = Env()
    policy = Policy()
    result = _run_sync_episode(
        env,
        policy,
        policy_preprocessor=None,
        postprocessor=lambda action: action,
        task="test task",
        prepare_observation=lambda observation, preprocessor: (observation, preprocessor),
        seed=5,
        max_steps=3,
        gripper_polarity="positive_open",
        torch_seed=None,
    )

    assert policy.reset_count == 1
    assert [float(action[6]) for action in env.actions] == [-1.0, -1.0, 1.0]
    assert result["gripper_hysteresis"] == {
        "action_index": 6,
        "low_threshold": -0.5,
        "high_threshold": 0.5,
        "polarity": "positive_open",
    }
    assert result["gripper_switch_count"] == 1


def test_control_sleep_records_measured_duration() -> None:
    timestamps = iter((0.02, 0.02, 0.052))
    requested: list[float] = []

    actual = _sleep_for_remaining_period(
        0.05,
        0.0,
        clock=lambda: next(timestamps),
        sleeper=requested.append,
    )

    assert requested == [pytest.approx(0.03)]
    assert actual == pytest.approx(0.032)
