import torch
from torch import nn

from smolvla_flow.toy_flow import (
    VelocityMLP,
    euler_sample,
    make_flow_matching_batch,
    sample_conditional_data,
)


def test_flow_batch_shapes_and_identity() -> None:
    generator = torch.Generator().manual_seed(3)
    x_data, condition = sample_conditional_data(16, device="cpu", generator=generator)
    x_t, time, target_velocity = make_flow_matching_batch(x_data, generator=generator)

    assert x_data.shape == (16, 2)
    assert condition.shape == (16,)
    assert x_t.shape == target_velocity.shape == (16, 2)
    assert time.shape == (16, 1)


def test_velocity_model_output_shape() -> None:
    model = VelocityMLP(hidden_dim=32, time_dim=8)
    output = model(torch.randn(4, 2), torch.rand(4, 1), torch.tensor([0, 1, 0, 1]))
    assert output.shape == (4, 2)


def test_euler_sampler_integrates_constant_velocity() -> None:
    class ConstantVelocity(nn.Module):
        def forward(self, x_t: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            del time, condition
            return torch.ones_like(x_t) * 2.0

    initial = torch.zeros(3, 2)
    samples = euler_sample(
        ConstantVelocity(), torch.tensor([0, 1, 0]), num_steps=10, initial_noise=initial
    )
    torch.testing.assert_close(samples, torch.full_like(initial, 2.0))


def test_sampler_rejects_zero_steps() -> None:
    model = VelocityMLP(hidden_dim=32, time_dim=8)
    try:
        euler_sample(model, torch.tensor([0]), num_steps=0)
    except ValueError as error:
        assert "at least 1" in str(error)
    else:
        raise AssertionError("expected ValueError")
