"""Small ACT and Diffusion Policy baselines used by the development smoke.

The classes in this module keep the LIBERO state/action contract from the
project configuration while avoiding LeRobot and image-backbone imports.  They
are useful for checking tensor shapes, finite losses, optimizer updates, and
inference timing before the same configurations are attached to a real
LeRobotDataset.  They are not a replacement for the official LeRobot policy
implementations used by the final LIBERO comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ACTBaselineConfig:
    """Configuration matching the first project's ACT smoke contract."""

    state_dim: int = 8
    action_dim: int = 7
    chunk_size: int = 50
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.action_dim < 1 or self.chunk_size < 1:
            raise ValueError("state_dim, action_dim, and chunk_size must be positive")
        if self.model_dim < 1 or self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if self.num_heads < 1 or self.num_layers < 1 or self.feedforward_dim < 1:
            raise ValueError("num_heads, num_layers, and feedforward_dim must be positive")


class ACTBaseline(nn.Module):
    """A compact action-chunking Transformer.

    The state is encoded as one memory token and one learned query is used for
    each action position.  A Transformer decoder lets every action position
    attend to the state token and to the other positions, matching the
    action-chunking behavior needed by the project while keeping the smoke
    inexpensive.
    """

    def __init__(self, config: ACTBaselineConfig | None = None) -> None:
        super().__init__()
        self.config = config or ACTBaselineConfig()
        cfg = self.config
        self.state_projection = nn.Linear(cfg.state_dim, cfg.model_dim)
        self.action_queries = nn.Parameter(torch.zeros(1, cfg.chunk_size, cfg.model_dim))
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.num_layers)
        self.output_norm = nn.LayerNorm(cfg.model_dim)
        self.action_head = nn.Linear(cfg.model_dim, cfg.action_dim)

    def forward(self, state: Tensor) -> Tensor:
        cfg = self.config
        if state.ndim != 2 or state.shape[-1] != cfg.state_dim:
            raise ValueError(f"state must have shape [batch, {cfg.state_dim}]")
        memory = self.state_projection(state).unsqueeze(1)
        queries = self.action_queries.expand(state.shape[0], -1, -1)
        hidden = self.decoder(tgt=queries, memory=memory)
        return self.action_head(self.output_norm(hidden))


@dataclass(frozen=True)
class DiffusionBaselineConfig:
    """Configuration for a compact conditional action diffusion policy."""

    state_dim: int = 8
    action_dim: int = 7
    horizon: int = 64
    hidden_dim: int = 128
    num_train_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.action_dim < 1 or self.horizon < 1:
            raise ValueError("state_dim, action_dim, and horizon must be positive")
        if self.hidden_dim < 1 or self.num_train_timesteps < 2:
            raise ValueError("hidden_dim must be positive and train timesteps must be at least 2")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("beta_start and beta_end must satisfy 0 < start < end < 1")


class DiffusionBaseline(nn.Module):
    """A state-conditioned temporal denoiser for action chunks.

    Each action position shares the same denoising MLP.  The state and time
    embeddings are broadcast over the horizon, which preserves the conditional
    diffusion objective and makes inference-step ablations easy to measure.
    """

    def __init__(self, config: DiffusionBaselineConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiffusionBaselineConfig()
        cfg = self.config
        time_dim = 32
        self.state_projection = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_projection = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    @staticmethod
    def _time_embedding(timesteps: Tensor, dim: int = 32) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [batch]")
        half = dim // 2
        frequency = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        angles = timesteps.float().unsqueeze(1) * frequency.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=1)

    def forward(self, noisy_actions: Tensor, state: Tensor, timesteps: Tensor) -> Tensor:
        cfg = self.config
        if noisy_actions.ndim != 3 or tuple(noisy_actions.shape[1:]) != (cfg.horizon, cfg.action_dim):
            raise ValueError(
                f"noisy_actions must have shape [batch, {cfg.horizon}, {cfg.action_dim}]"
            )
        if state.ndim != 2 or state.shape[-1] != cfg.state_dim:
            raise ValueError(f"state must have shape [batch, {cfg.state_dim}]")
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_actions.shape[0]:
            raise ValueError("timesteps must have shape [batch]")
        state_tokens = self.state_projection(state).unsqueeze(1).expand(-1, cfg.horizon, -1)
        time_tokens = self.time_projection(self._time_embedding(timesteps)).unsqueeze(1)
        time_tokens = time_tokens.expand(-1, cfg.horizon, -1)
        action_tokens = self.action_projection(noisy_actions)
        return self.network(torch.cat((action_tokens, state_tokens, time_tokens), dim=-1))

    def add_noise(self, actions: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        """Apply the forward diffusion schedule to a clean action chunk."""

        if actions.shape != noise.shape:
            raise ValueError("actions and noise must have identical shapes")
        alpha_bar = self.alphas_cumprod[timesteps].to(dtype=actions.dtype)
        alpha_bar = alpha_bar.reshape(-1, 1, 1)
        return alpha_bar.sqrt() * actions + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, state: Tensor, *, num_steps: int, generator: torch.Generator | None = None) -> Tensor:
        """Generate an action chunk using a DDIM-like deterministic schedule."""

        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        cfg = self.config
        actions = torch.randn(
            state.shape[0], cfg.horizon, cfg.action_dim,
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
        schedule = torch.linspace(
            cfg.num_train_timesteps - 1,
            0,
            num_steps,
            device=state.device,
        ).round().long()
        for index, timestep in enumerate(schedule):
            timestep_batch = timestep.expand(state.shape[0])
            predicted_noise = self(actions, state, timestep_batch)
            alpha_bar = self.alphas_cumprod[timestep].to(dtype=actions.dtype, device=actions.device)
            clean = (actions - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1e-6)
            if index == len(schedule) - 1:
                actions = clean
                continue
            next_timestep = schedule[index + 1]
            next_alpha_bar = self.alphas_cumprod[next_timestep].to(dtype=actions.dtype, device=actions.device)
            actions = next_alpha_bar.sqrt() * clean + (1.0 - next_alpha_bar).sqrt() * predicted_noise
        return actions


def make_synthetic_baseline_data(
    sample_count: int,
    *,
    state_dim: int = 8,
    action_dim: int = 7,
    horizon: int = 64,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Create deterministic state-conditioned chunks for architecture smoke tests.

    This data intentionally has a simple closed-form target.  It checks the
    training and inference plumbing, not LIBERO task performance.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if action_dim > state_dim:
        raise ValueError("the synthetic data helper requires action_dim <= state_dim")
    generator = torch.Generator(device=device).manual_seed(seed)
    state = torch.randn(sample_count, state_dim, generator=generator, device=device)
    state_for_action = state[:, :action_dim]
    time = torch.linspace(0.0, 1.0, horizon, device=device).reshape(1, horizon, 1)
    phase = torch.arange(action_dim, device=device, dtype=state.dtype).reshape(1, 1, action_dim) * 0.17
    actions = torch.tanh(state_for_action.unsqueeze(1) + 0.35 * torch.sin(2.0 * math.pi * time + phase))
    return state, actions
