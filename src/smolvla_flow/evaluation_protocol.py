"""Shared evaluation protocol and statistics for LIBERO policy comparisons.

The protocol is deliberately small and serialisable.  It is used by the
rollout scripts and the offline baseline evaluator so that a result cannot be
compared without recording the task, seeds, action horizon, camera contract,
and normalization source.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EvaluationProtocol:
    """Locked settings shared by all policy variants in one matrix."""

    version: str = "libero-task34-v1"
    suite: str = "libero_spatial"
    task_id: int = 0
    dataset_task_index: int = 34
    state_dim: int = 8
    action_dim: int = 7
    camera_names: tuple[str, ...] = (
        "agentview_image",
        "robot0_eye_in_hand_image",
    )
    observation_height: int = 256
    observation_width: int = 256
    max_steps: int = 280
    action_execution_steps: int = 10
    control_frequency_hz: float = 20.0
    normalization: str = "policy_checkpoint_pre_postprocessor"
    success_key: str = "is_success"
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.task_id < 0 or self.dataset_task_index < 0:
            raise ValueError("task_id and dataset_task_index must be non-negative")
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("state_dim and action_dim must be positive")
        if not self.camera_names:
            raise ValueError("camera_names must not be empty")
        if self.observation_height < 1 or self.observation_width < 1:
            raise ValueError("observation dimensions must be positive")
        if self.max_steps < 1 or self.action_execution_steps < 1:
            raise ValueError("max_steps and action_execution_steps must be positive")
        if self.action_execution_steps > self.max_steps:
            raise ValueError("action_execution_steps cannot exceed max_steps")
        if self.control_frequency_hz <= 0 or not math.isfinite(self.control_frequency_hz):
            raise ValueError("control_frequency_hz must be positive and finite")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "suite": self.suite,
            "task_id": self.task_id,
            "dataset_task_index": self.dataset_task_index,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "camera_names": list(self.camera_names),
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
            "max_steps": self.max_steps,
            "action_execution_steps": self.action_execution_steps,
            "control_frequency_hz": self.control_frequency_hz,
            "normalization": self.normalization,
            "success_key": self.success_key,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationProtocol":
        """Build a protocol from JSON/TOML-like data."""

        fields = {
            "version": value.get("version", cls.version),
            "suite": value.get("suite", cls.suite),
            "task_id": int(value.get("task_id", cls.task_id)),
            "dataset_task_index": int(value.get("dataset_task_index", cls.dataset_task_index)),
            "state_dim": int(value.get("state_dim", cls.state_dim)),
            "action_dim": int(value.get("action_dim", cls.action_dim)),
            "camera_names": tuple(value.get("camera_names", cls.camera_names)),
            "observation_height": int(value.get("observation_height", cls.observation_height)),
            "observation_width": int(value.get("observation_width", cls.observation_width)),
            "max_steps": int(value.get("max_steps", cls.max_steps)),
            "action_execution_steps": int(value.get("action_execution_steps", cls.action_execution_steps)),
            "control_frequency_hz": float(value.get("control_frequency_hz", cls.control_frequency_hz)),
            "normalization": str(value.get("normalization", cls.normalization)),
            "success_key": str(value.get("success_key", cls.success_key)),
            "confidence_level": float(value.get("confidence_level", cls.confidence_level)),
        }
        return cls(**fields)


def validate_variant_contract(
    protocol: EvaluationProtocol,
    *,
    state_dim: int,
    action_dim: int,
    action_chunk_size: int,
    action_execution_steps: int,
    camera_names: Iterable[str] | None = None,
    normalization: str | None = None,
) -> None:
    """Raise when a policy does not satisfy the locked comparison contract."""

    if (state_dim, action_dim) != (protocol.state_dim, protocol.action_dim):
        raise ValueError(
            "state/action contract mismatch: "
            f"got ({state_dim}, {action_dim}), expected ({protocol.state_dim}, {protocol.action_dim})"
        )
    if action_chunk_size < protocol.action_execution_steps:
        raise ValueError(
            f"action chunk ({action_chunk_size}) is shorter than the execution horizon "
            f"({protocol.action_execution_steps})"
        )
    if action_execution_steps != protocol.action_execution_steps:
        raise ValueError(
            f"action execution horizon {action_execution_steps} does not match "
            f"protocol value {protocol.action_execution_steps}"
        )
    if camera_names is not None and tuple(camera_names) != protocol.camera_names:
        raise ValueError(
            f"camera contract {tuple(camera_names)!r} does not match {protocol.camera_names!r}"
        )
    if normalization is not None and normalization != protocol.normalization:
        raise ValueError(
            f"normalization contract {normalization!r} does not match {protocol.normalization!r}"
        )


def wilson_interval(
    success_count: int,
    trial_count: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson score interval for a Bernoulli success rate."""

    if trial_count < 1 or success_count < 0 or success_count > trial_count:
        raise ValueError("success_count must be in [0, trial_count] and trial_count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    # Normal quantiles needed by the project are evaluated without scipy.
    # statistics.NormalDist is part of the Python standard library.
    alpha = (1.0 + confidence_level) / 2.0
    z = statistics.NormalDist().inv_cdf(alpha)
    n = float(trial_count)
    p = float(success_count) / n
    denominator = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def binary_success_summary(
    successes: Sequence[bool | int],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Summarise success count, rate, and Wilson interval."""

    values = [bool(value) for value in successes]
    if not values:
        return {
            "trial_count": 0,
            "success_count": 0,
            "success_rate": None,
            "wilson_95": None,
        }
    success_count = sum(values)
    lower, upper = wilson_interval(
        success_count,
        len(values),
        confidence_level=confidence_level,
    )
    return {
        "trial_count": len(values),
        "success_count": success_count,
        "success_rate": success_count / len(values),
        "wilson_95": [lower, upper],
        "confidence_level": confidence_level,
    }


def action_error(
    prediction: Any,
    target: Any,
    *,
    execute_steps: int,
) -> dict[str, float | int]:
    """Compute action-space MSE/MAE over the shared executed prefix.

    Inputs may be NumPy arrays or PyTorch tensors.  The result is deliberately
    kept in the caller's action units.  Normalisation must happen before this
    function when a policy contract uses normalised actions.
    """

    try:
        import numpy as np

        predicted = np.asarray(prediction, dtype=np.float64)
        expected = np.asarray(target, dtype=np.float64)
    except Exception as error:  # pragma: no cover - defensive conversion path
        raise TypeError("prediction and target must be array-like") from error
    if predicted.ndim == 2:
        predicted = predicted[None, ...]
    if expected.ndim == 2:
        expected = expected[None, ...]
    if predicted.ndim != 3 or expected.ndim != 3:
        raise ValueError("prediction and target must have shape [B, T, D] or [T, D]")
    if predicted.shape[0] != expected.shape[0] or predicted.shape[2] != expected.shape[2]:
        raise ValueError("prediction and target batch/action dimensions must match")
    if execute_steps < 1 or predicted.shape[1] < execute_steps or expected.shape[1] < execute_steps:
        raise ValueError("prediction and target must contain the requested execution prefix")
    delta = predicted[:, :execute_steps] - expected[:, :execute_steps]
    return {
        "execute_steps": execute_steps,
        "action_mse": float(np.mean(delta * delta)),
        "action_mae": float(np.mean(np.abs(delta))),
        "action_l2_mean": float(np.linalg.norm(delta, axis=-1).mean()),
        "sample_count": int(delta.shape[0]),
    }


__all__ = [
    "EvaluationProtocol",
    "action_error",
    "binary_success_summary",
    "validate_variant_contract",
    "wilson_interval",
]
