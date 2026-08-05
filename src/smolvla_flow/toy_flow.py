"""A minimal conditional flow-matching problem.

The same tensor roles will later map to robot control as follows:

* ``condition`` becomes image, language, and robot-state context.
* a 2D ``x_data`` becomes a continuous action chunk with shape [horizon, action_dim].
* ``VelocityMLP`` becomes the Transformer Action Expert.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sample_conditional_data(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate two curved target distributions selected by a binary condition."""
    condition = torch.randint(0, 2, (batch_size,), device=device, generator=generator)
    x = torch.randn(batch_size, device=device, generator=generator) * 0.55
    side = condition.float().mul(2).sub(1)
    target_x = x + side * 2.0
    target_y = 0.55 * torch.sin(2.2 * x) + side * 0.8
    target_y = target_y + 0.12 * torch.randn(batch_size, device=device, generator=generator)
    return torch.stack((target_x, target_y), dim=-1), condition


def make_flow_matching_batch(
    x_data: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create x_t and its supervised conditional-flow velocity target."""
    batch_size = x_data.shape[0]
    noise = torch.randn(x_data.shape, device=x_data.device, dtype=x_data.dtype, generator=generator)
    time = torch.rand(batch_size, 1, device=x_data.device, dtype=x_data.dtype, generator=generator)
    x_t = (1.0 - time) * noise + time * x_data
    target_velocity = x_data - noise
    return x_t, time, target_velocity


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        angles = time * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class VelocityMLP(nn.Module):
    """Predict v_theta(x_t, t, condition) for the toy problem."""

    def __init__(self, hidden_dim: int = 128, time_dim: int = 32) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.condition_embedding = nn.Embedding(2, time_dim)
        self.network = nn.Sequential(
            nn.Linear(2 + time_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        context = torch.cat(
            (x_t, self.time_embedding(time), self.condition_embedding(condition)), dim=-1
        )
        return self.network(context)


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    condition: Tensor,
    *,
    num_steps: int = 10,
    initial_noise: Tensor | None = None,
) -> Tensor:
    """Integrate dx/dt = v_theta(x, t, condition) from t=0 to t=1."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if initial_noise is None:
        initial_noise = torch.randn(condition.shape[0], 2, device=condition.device)
    x_t = initial_noise
    step_size = 1.0 / num_steps
    for step in range(num_steps):
        time = torch.full(
            (condition.shape[0], 1),
            step / num_steps,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        x_t = x_t + step_size * model(x_t, time, condition)
    return x_t

