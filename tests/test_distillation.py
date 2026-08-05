import pytest
import torch
from torch import nn

from smolvla_flow.distillation import (
    DistillationConfig,
    NonFiniteTensorError,
    build_coarse_teacher_targets,
    compute_distillation_loss,
    euler_rollout,
    interpolate_trajectory,
    train_distillation_step,
    validate_step_transition,
)


class ConstantVelocity(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, noisy_actions: torch.Tensor, time: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        del time, context_tokens
        return self.value.expand_as(noisy_actions)


class NonFiniteVelocity(nn.Module):
    def forward(self, noisy_actions: torch.Tensor, time: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        del time, context_tokens
        return torch.full_like(noisy_actions, float("nan"))


class RecordingVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.times: list[torch.Tensor] = []

    def forward(self, noisy_actions: torch.Tensor, time: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        del context_tokens
        self.times.append(time.detach().clone())
        return torch.zeros_like(noisy_actions)


def tiny_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(2, 3, 2), torch.zeros(2, 4, 5)


def test_step_transition_only_accepts_progressive_stages() -> None:
    validate_step_transition(10, 5)
    validate_step_transition(5, 2)
    with pytest.raises(ValueError, match="unsupported"):
        validate_step_transition(10, 2)
    with pytest.raises(ValueError, match="unsupported"):
        DistillationConfig(teacher_steps=5, student_steps=5)


def test_config_rejects_invalid_loss_weights() -> None:
    with pytest.raises(ValueError, match="finite"):
        DistillationConfig(action_regression_weight=float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        DistillationConfig(trajectory_consistency_weight=-1.0)
    with pytest.raises(ValueError, match="at least one"):
        DistillationConfig(trajectory_consistency_weight=0.0, action_regression_weight=0.0)
    with pytest.raises(ValueError, match="action_dim"):
        DistillationConfig(action_dim=0)


def test_ten_to_five_uses_exact_stride_two_states() -> None:
    teacher_trajectory = torch.arange(11, dtype=torch.float32).reshape(1, 11, 1, 1)
    coarse = build_coarse_teacher_targets(teacher_trajectory, student_steps=5)
    expected = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]).reshape(1, 6, 1, 1)
    torch.testing.assert_close(coarse, expected)


def test_coarse_target_rejects_direct_ten_to_two_jump() -> None:
    teacher_trajectory = torch.zeros(1, 11, 1, 1)
    with pytest.raises(ValueError, match="unsupported"):
        build_coarse_teacher_targets(teacher_trajectory, student_steps=2)


def test_five_to_two_uses_linear_middle_state() -> None:
    teacher_trajectory = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1)
    coarse = interpolate_trajectory(teacher_trajectory, target_steps=2)
    expected = torch.tensor([0.0, 2.5, 5.0]).reshape(1, 3, 1, 1)
    torch.testing.assert_close(coarse, expected)


def test_euler_rollout_has_initial_state_and_end_state() -> None:
    initial_noise, context = tiny_inputs()
    model = ConstantVelocity(2.0)
    trajectory = euler_rollout(model, initial_noise, context, num_steps=5)
    assert trajectory.shape == (2, 6, 3, 2)
    torch.testing.assert_close(trajectory[:, 0], initial_noise)
    torch.testing.assert_close(trajectory[:, -1], torch.full_like(initial_noise, -2.0))


def test_euler_rollout_queries_descending_time() -> None:
    initial_noise, context = tiny_inputs()
    model = RecordingVelocity()
    euler_rollout(model, initial_noise, context, num_steps=5)
    queried_times = torch.cat(model.times, dim=1)[0]
    torch.testing.assert_close(queried_times, torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2]))


def test_distillation_loss_contains_both_action_losses() -> None:
    initial_noise, context = tiny_inputs()
    teacher = ConstantVelocity(1.0).eval()
    student = ConstantVelocity(0.0)
    output = compute_distillation_loss(
        student,
        teacher,
        initial_noise,
        context,
        config=DistillationConfig(teacher_steps=10, student_steps=5),
        target_actions=torch.ones_like(initial_noise),
    )
    assert output.loss.ndim == 0
    assert output.trajectory_consistency_loss.ndim == 0
    assert output.action_regression_loss.ndim == 0
    assert output.teacher_trajectory.shape == (2, 11, 3, 2)
    assert output.coarse_teacher_trajectory.shape == (2, 6, 3, 2)
    assert output.student_trajectory.shape == (2, 6, 3, 2)
    assert output.loss.requires_grad
    assert torch.isfinite(output.loss)


def test_action_dim_masks_padded_dimensions() -> None:
    initial_noise = torch.zeros(1, 2, 2)
    context = torch.zeros(1, 1, 1)
    teacher = ConstantVelocity(1.0).eval()
    student = ConstantVelocity(0.0)
    target = torch.zeros_like(initial_noise)
    # The second padded dimension is intentionally far from the target.  Only
    # the real first action dimension should contribute to this objective.
    target[..., 1] = 100.0
    output = compute_distillation_loss(
        student,
        teacher,
        initial_noise,
        context,
        config=DistillationConfig(teacher_steps=10, student_steps=5, action_dim=1),
        target_actions=target,
    )
    unmasked = compute_distillation_loss(
        student,
        teacher,
        initial_noise,
        context,
        config=DistillationConfig(teacher_steps=10, student_steps=5),
        target_actions=target,
    )
    assert output.loss < unmasked.loss


def test_single_training_step_updates_student() -> None:
    initial_noise, context = tiny_inputs()
    teacher = ConstantVelocity(1.0).eval()
    student = ConstantVelocity(0.0)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.25)
    before = student.value.detach().clone()
    output = train_distillation_step(
        student,
        teacher,
        initial_noise,
        context,
        optimizer=optimizer,
        config=DistillationConfig(teacher_steps=5, student_steps=2),
    )
    assert torch.isfinite(output.loss)
    assert not torch.equal(student.value.detach(), before)
    assert torch.isfinite(student.value)


def test_non_finite_model_output_is_rejected() -> None:
    initial_noise, context = tiny_inputs()
    with pytest.raises(NonFiniteTensorError, match="non-finite"):
        euler_rollout(
            NonFiniteVelocity(),
            initial_noise,
            context,
            num_steps=1,
        )


def test_non_finite_inputs_and_bad_shapes_are_rejected() -> None:
    initial_noise, context = tiny_inputs()
    bad_noise = initial_noise.clone()
    bad_noise[0, 0, 0] = float("inf")
    with pytest.raises(NonFiniteTensorError, match="initial_state"):
        euler_rollout(ConstantVelocity(1.0), bad_noise, context, num_steps=1)
    with pytest.raises(ValueError, match="shape"):
        compute_distillation_loss(
            ConstantVelocity(1.0),
            ConstantVelocity(1.0),
            initial_noise,
            context,
            target_actions=torch.zeros(1, 3, 2),
        )


def test_rollout_rejects_non_positive_steps() -> None:
    initial_noise, context = tiny_inputs()
    with pytest.raises(ValueError, match="positive"):
        euler_rollout(ConstantVelocity(1.0), initial_noise, context, num_steps=0)
