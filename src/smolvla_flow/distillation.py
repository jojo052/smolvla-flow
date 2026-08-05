"""Progressive action-only distillation for a flow-matching action expert.

The implementation keeps the distillation target in action space.  A teacher
rollout is first produced with its native number of Euler steps.  The rollout
follows the SmolVLA convention, starting at ``t=1`` and descending to ``t=0``,
then is sampled on the student's time grid.  The 10-to-5 stage lands exactly on
every second teacher state, while the 5-to-2 stage uses linear interpolation at
the middle time point.  The student is trained to match this coarse trajectory
and its final action chunk.

The module deliberately accepts a small model protocol instead of importing
``TransformerActionExpert``.  This keeps the loss usable with the SmolVLA
action expert and with small deterministic test doubles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VelocityModel(Protocol):
    """Protocol implemented by a flow-matching velocity model."""

    def __call__(self, noisy_actions: Tensor, time: Tensor, context_tokens: Tensor) -> Tensor:
        ...


class NonFiniteTensorError(ValueError, FloatingPointError):
    """Raised when a distillation input, model output, or loss is non-finite."""


SUPPORTED_STEP_TRANSITIONS: tuple[tuple[int, int], ...] = ((10, 5), (5, 2))


def validate_step_transition(teacher_steps: int, student_steps: int) -> None:
    """Reject unsupported progressive-distillation stages.

    The first project version has two explicit stages.  In particular, a
    direct 10-to-2 jump is rejected so that every 2-step model is initialized
    from the already distilled 5-step teacher.
    """

    if not isinstance(teacher_steps, int) or not isinstance(student_steps, int):
        raise ValueError("teacher_steps and student_steps must be integers")
    if (teacher_steps, student_steps) not in SUPPORTED_STEP_TRANSITIONS:
        allowed = ", ".join(f"{teacher}->{student}" for teacher, student in SUPPORTED_STEP_TRANSITIONS)
        raise ValueError(
            f"unsupported distillation step transition {teacher_steps}->{student_steps}; "
            f"supported transitions are {allowed}"
        )


@dataclass(frozen=True)
class DistillationConfig:
    """Loss and integration settings for one progressive stage."""

    teacher_steps: int = 10
    student_steps: int = 5
    action_dim: int | None = None
    trajectory_consistency_weight: float = 1.0
    action_regression_weight: float = 1.0

    def __post_init__(self) -> None:
        validate_step_transition(self.teacher_steps, self.student_steps)
        if self.action_dim is not None and (
            not isinstance(self.action_dim, int) or self.action_dim < 1
        ):
            raise ValueError("action_dim must be a positive integer when provided")
        for name, value in (
            ("trajectory_consistency_weight", self.trajectory_consistency_weight),
            ("action_regression_weight", self.action_regression_weight),
        ):
            if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.trajectory_consistency_weight == 0 and self.action_regression_weight == 0:
            raise ValueError("at least one distillation loss weight must be positive")


# The longer name makes the intended two-stage schedule explicit while the
# shorter name remains convenient for scripts.
ProgressiveDistillationConfig = DistillationConfig


@dataclass
class DistillationLossOutput:
    """Losses and trajectories returned by :func:`compute_distillation_loss`."""

    loss: Tensor
    trajectory_consistency_loss: Tensor
    action_regression_loss: Tensor
    teacher_trajectory: Tensor
    coarse_teacher_trajectory: Tensor
    student_trajectory: Tensor
    action_target: Tensor

    @property
    def total_loss(self) -> Tensor:
        """Readable alias for the weighted objective."""

        return self.loss

    @property
    def trajectory_loss(self) -> Tensor:
        """Short alias used by metric loggers."""

        return self.trajectory_consistency_loss

    @property
    def action_loss(self) -> Tensor:
        """Short alias used by metric loggers."""

        return self.action_regression_loss

    def __getitem__(self, key: str) -> Tensor:
        """Allow metric-style access while retaining typed attributes."""

        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        return (
            "loss",
            "trajectory_consistency_loss",
            "action_regression_loss",
            "teacher_trajectory",
            "coarse_teacher_trajectory",
            "student_trajectory",
            "action_target",
        )

    def items(self) -> Iterator[tuple[str, Tensor]]:
        for key in self.keys():
            yield key, self[key]


def _check_finite(name: str, value: Tensor) -> Tensor:
    """Check one tensor and return it unchanged for convenient composition."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.isfinite(value).all().item():
        raise NonFiniteTensorError(f"{name} contains non-finite values")
    return value


def _check_rollout_inputs(initial_state: Tensor, context_tokens: Tensor) -> None:
    if initial_state.ndim != 3:
        raise ValueError("initial_state must have shape [batch, chunk_size, action_dim]")
    if context_tokens.ndim != 3:
        raise ValueError("context_tokens must have shape [batch, tokens, context_dim]")
    if initial_state.shape[0] != context_tokens.shape[0]:
        raise ValueError("initial_state and context_tokens must have the same batch size")
    _check_finite("initial_state", initial_state)
    _check_finite("context_tokens", context_tokens)


def _call_velocity(
    model: VelocityModel,
    state: Tensor,
    time: Tensor,
    context_tokens: Tensor,
    *,
    name: str,
) -> Tensor:
    velocity = model(state, time, context_tokens)
    if not isinstance(velocity, Tensor):
        raise TypeError(f"{name} must return a torch.Tensor")
    if velocity.shape != state.shape:
        raise ValueError(
            f"{name} output shape {tuple(velocity.shape)} does not match state shape {tuple(state.shape)}"
        )
    return _check_finite(f"{name} output", velocity)


def euler_rollout(
    model: VelocityModel,
    initial_state: Tensor,
    context_tokens: Tensor,
    *,
    num_steps: int,
    track_grad: bool = True,
) -> Tensor:
    """Integrate ``t=1`` to ``t=0`` and return ``[B,S+1,H,A]`` states.

    State index zero is the noisy action at ``t=1``.  The last state is the
    generated action at ``t=0``.  This descending-time index order is preserved
    when building coarse trajectory targets.
    """

    if not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    _check_rollout_inputs(initial_state, context_tokens)

    def integrate() -> Tensor:
        state = initial_state
        states = [state]
        dt = 1.0 / num_steps
        batch_size = state.shape[0]
        for step in range(num_steps):
            time = torch.full(
                (batch_size, 1),
                1.0 - step / num_steps,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = _call_velocity(
                model,
                state,
                time,
                context_tokens,
                name="velocity model",
            )
            state = _check_finite("rollout state", state - dt * velocity)
            states.append(state)
        return torch.stack(states, dim=1)

    if track_grad:
        return integrate()
    with torch.no_grad():
        return integrate()


def interpolate_trajectory(trajectory: Tensor, target_steps: int) -> Tensor:
    """Sample a trajectory on a new uniform time grid by linear interpolation.

    ``trajectory`` contains ``source_steps + 1`` states in descending-time
    order, including the initial noise state at ``t=1`` and the final action at
    ``t=0``.  The returned tensor contains ``target_steps + 1`` states in the
    same order.  With 10 source steps and 5 target steps, all samples are exact
    stride-2 states.  With 5 source steps and 2 target steps, the middle sample
    is the average of source states 2 and 3.
    """

    if trajectory.ndim != 4:
        raise ValueError("trajectory must have shape [batch, steps + 1, chunk_size, action_dim]")
    if not isinstance(target_steps, int) or target_steps < 1:
        raise ValueError("target_steps must be a positive integer")
    source_steps = trajectory.shape[1] - 1
    if source_steps < 1:
        raise ValueError("trajectory must contain at least one integration step")
    _check_finite("trajectory", trajectory)

    positions = torch.linspace(
        0.0,
        float(source_steps),
        target_steps + 1,
        device=trajectory.device,
        dtype=trajectory.dtype,
    )
    left = positions.floor().to(dtype=torch.long)
    right = torch.clamp(left + 1, max=source_steps)
    alpha = (positions - left.to(dtype=positions.dtype)).reshape(1, -1, 1, 1)
    left_states = trajectory[:, left]
    right_states = trajectory[:, right]
    sampled = (1.0 - alpha) * left_states + alpha * right_states
    return _check_finite("interpolated trajectory", sampled)


def build_coarse_teacher_targets(teacher_trajectory: Tensor, student_steps: int) -> Tensor:
    """Return teacher states aligned with a student's coarse time grid."""

    if teacher_trajectory.ndim != 4:
        raise ValueError("teacher_trajectory must have shape [batch, steps + 1, chunk_size, action_dim]")
    validate_step_transition(teacher_trajectory.shape[1] - 1, student_steps)
    return interpolate_trajectory(teacher_trajectory, student_steps)


def compute_distillation_loss(
    student: VelocityModel,
    teacher: VelocityModel,
    initial_noise: Tensor,
    context_tokens: Tensor,
    *,
    config: DistillationConfig | None = None,
    target_actions: Tensor | None = None,
) -> DistillationLossOutput:
    """Compute one action-only progressive-distillation objective.

    ``target_actions`` is an optional demonstrated action chunk.  When it is
    supplied, the final student state at ``t=0`` is regressed to that chunk.
    When it is omitted, the final state of the teacher rollout is used, which makes the
    function useful for checkpoint-to-checkpoint distillation on unlabeled
    context/noise pairs.
    """

    cfg = config or DistillationConfig()
    _check_rollout_inputs(initial_noise, context_tokens)
    if cfg.action_dim is not None and cfg.action_dim > initial_noise.shape[-1]:
        raise ValueError(
            f"action_dim={cfg.action_dim} exceeds padded action dimension {initial_noise.shape[-1]}"
        )
    if target_actions is not None:
        if target_actions.shape != initial_noise.shape:
            raise ValueError(
                f"target_actions shape {tuple(target_actions.shape)} must match initial_noise "
                f"shape {tuple(initial_noise.shape)}"
            )
        _check_finite("target_actions", target_actions)

    teacher_trajectory = euler_rollout(
        teacher,
        initial_noise,
        context_tokens,
        num_steps=cfg.teacher_steps,
        track_grad=False,
    )
    coarse_teacher_trajectory = build_coarse_teacher_targets(
        teacher_trajectory,
        cfg.student_steps,
    )
    student_trajectory = euler_rollout(
        student,
        initial_noise,
        context_tokens,
        num_steps=cfg.student_steps,
        track_grad=True,
    )

    if cfg.action_dim is None:
        student_loss_trajectory = student_trajectory
        teacher_loss_trajectory = coarse_teacher_trajectory
    else:
        student_loss_trajectory = student_trajectory[..., : cfg.action_dim]
        teacher_loss_trajectory = coarse_teacher_trajectory[..., : cfg.action_dim]
    trajectory_loss = F.mse_loss(student_loss_trajectory, teacher_loss_trajectory)
    action_target = coarse_teacher_trajectory[:, -1] if target_actions is None else target_actions
    if cfg.action_dim is None:
        student_loss_action = student_trajectory[:, -1]
        action_loss_target = action_target
    else:
        student_loss_action = student_trajectory[:, -1, :, : cfg.action_dim]
        action_loss_target = action_target[..., : cfg.action_dim]
    action_loss = F.mse_loss(student_loss_action, action_loss_target)
    loss = (
        cfg.trajectory_consistency_weight * trajectory_loss
        + cfg.action_regression_weight * action_loss
    )
    _check_finite("trajectory consistency loss", trajectory_loss)
    _check_finite("action regression loss", action_loss)
    _check_finite("distillation loss", loss)
    return DistillationLossOutput(
        loss=loss,
        trajectory_consistency_loss=trajectory_loss,
        action_regression_loss=action_loss,
        teacher_trajectory=teacher_trajectory,
        coarse_teacher_trajectory=coarse_teacher_trajectory,
        student_trajectory=student_trajectory,
        action_target=action_target,
    )


def train_distillation_step(
    student: nn.Module,
    teacher: VelocityModel,
    initial_noise: Tensor,
    context_tokens: Tensor,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    config: DistillationConfig | None = None,
    target_actions: Tensor | None = None,
) -> DistillationLossOutput:
    """Run one optimization step for a student action expert.

    Passing ``optimizer=None`` computes the same objective without updating
    parameters, which is useful for smoke tests and latency measurements.
    Gradients and updated parameters are checked for finiteness before the
    function returns.
    """

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    output = compute_distillation_loss(
        student,
        teacher,
        initial_noise,
        context_tokens,
        config=config,
        target_actions=target_actions,
    )
    if optimizer is None:
        return output

    output.loss.backward()
    for name, parameter in student.named_parameters():
        if parameter.grad is not None:
            _check_finite(f"gradient for {name}", parameter.grad)
    optimizer.step()
    for name, parameter in student.named_parameters():
        _check_finite(f"parameter {name}", parameter)
    return output


# Explicit aliases make the API easy to discover from experiment scripts.
distill_one_step = train_distillation_step
progressive_distillation_step = train_distillation_step


__all__ = [
    "DistillationConfig",
    "DistillationLossOutput",
    "NonFiniteTensorError",
    "ProgressiveDistillationConfig",
    "SUPPORTED_STEP_TRANSITIONS",
    "build_coarse_teacher_targets",
    "compute_distillation_loss",
    "distill_one_step",
    "euler_rollout",
    "interpolate_trajectory",
    "progressive_distillation_step",
    "train_distillation_step",
    "validate_step_transition",
]
