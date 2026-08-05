"""Flow-matching Transformer Action Expert for continuous action chunks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from smolvla_flow.toy_flow import SinusoidalTimeEmbedding


@dataclass(frozen=True)
class ActionExpertConfig:
    action_dim: int = 7
    chunk_size: int = 50
    context_dim: int = 256
    model_dim: int = 256
    time_dim: int = 32
    num_layers: int = 4
    num_heads: int = 8
    feedforward_dim: int = 1024
    dropout: float = 0.0


def make_action_flow_batch(
    actions: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a SmolVLA-style descending-time flow-matching training pair.

    ``t=1`` represents Gaussian noise and ``t=0`` represents the demonstrated
    action chunk.  The negative integration direction is handled by
    :func:`sample_action_chunk`.
    """
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, chunk_size, action_dim]")
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    time = torch.rand(actions.shape[0], 1, device=actions.device, dtype=actions.dtype, generator=generator)
    broadcast_time = time.unsqueeze(-1)
    noisy_actions = broadcast_time * noise + (1.0 - broadcast_time) * actions
    target_velocity = noise - actions
    return noisy_actions, time, target_velocity


class TransformerActionExpert(nn.Module):
    """Predict the velocity of every action in a chunk in parallel.

    ``context_tokens`` are placeholders for SmolVLA image, language, and robot-state
    tokens. Cross-attention lets each action position read the complete context.
    """

    def __init__(self, config: ActionExpertConfig) -> None:
        super().__init__()
        if config.model_dim % config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.config = config
        self.action_projection = nn.Linear(config.action_dim, config.model_dim)
        self.context_projection = nn.Linear(config.context_dim, config.model_dim)
        self.time_embedding = SinusoidalTimeEmbedding(config.time_dim)
        self.time_projection = nn.Linear(config.time_dim, config.model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.chunk_size, config.model_dim))

        layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.velocity_head = nn.Linear(config.model_dim, config.action_dim)

    def forward(self, noisy_actions: Tensor, time: Tensor, context_tokens: Tensor) -> Tensor:
        expected = (self.config.chunk_size, self.config.action_dim)
        if noisy_actions.shape[1:] != expected:
            raise ValueError(f"noisy_actions trailing shape must be {expected}")
        if context_tokens.ndim != 3 or context_tokens.shape[-1] != self.config.context_dim:
            raise ValueError(
                f"context_tokens must have shape [batch, tokens, {self.config.context_dim}]"
            )

        action_tokens = self.action_projection(noisy_actions)
        time_token = self.time_projection(self.time_embedding(time)).unsqueeze(1)
        action_tokens = action_tokens + self.position_embedding + time_token
        context = self.context_projection(context_tokens)
        hidden = self.decoder(tgt=action_tokens, memory=context)
        return self.velocity_head(self.output_norm(hidden))


@torch.no_grad()
def sample_action_chunk(
    model: TransformerActionExpert,
    context_tokens: Tensor,
    *,
    num_steps: int = 10,
    initial_noise: Tensor | None = None,
) -> Tensor:
    """Generate a complete action chunk by integrating from ``t=1`` to ``t=0``."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    batch_size = context_tokens.shape[0]
    config = model.config
    if initial_noise is None:
        initial_noise = torch.randn(
            batch_size,
            config.chunk_size,
            config.action_dim,
            device=context_tokens.device,
            dtype=context_tokens.dtype,
        )
    actions = initial_noise
    step_size = 1.0 / num_steps
    for step in range(num_steps):
        time = torch.full(
            (batch_size, 1),
            1.0 - step / num_steps,
            device=actions.device,
            dtype=actions.dtype,
        )
        actions = actions - step_size * model(actions, time, context_tokens)
    return actions
