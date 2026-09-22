#!/usr/bin/env python
"""Run compact ACT and Diffusion Policy architecture smoke experiments.

The default input is deterministic synthetic state-conditioned action data with
the same 8D state and 7D action contract as LIBERO-Spatial task 0.  This script
checks 100 optimizer updates, finite losses, output shapes, and inference-step
latency.  It deliberately reports ``libero_episode_status=pending`` until the
same models are connected to a real LeRobotDataset and LIBERO environment.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

# Keep the script directly executable from a clean checkout, while preserving
# the normal package import behavior when callers already set PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smolvla_flow.baselines import (
    ACTBaseline,
    ACTBaselineConfig,
    DiffusionBaseline,
    DiffusionBaselineConfig,
    make_synthetic_baseline_data,
)
from smolvla_flow.evaluation_protocol import EvaluationProtocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("act", "diffusion", "all"), default="all")
    parser.add_argument("--steps", type=int, default=100, help="optimizer updates per baseline")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Optional torch artifact from build_baseline_data.py; defaults to synthetic smoke data.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/baselines/baseline_smoke.json"),
    )
    return parser.parse_args()


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} contains non-finite values")


def _timed(callable_, *, device: torch.device, warmup: int = 2, repeats: int = 8) -> dict[str, float]:
    for _ in range(warmup):
        callable_()
    _synchronize(device)
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        _synchronize(device)
        durations.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(statistics.fmean(durations)),
        "median_ms": float(statistics.median(durations)),
        "min_ms": float(min(durations)),
        "max_ms": float(max(durations)),
        "repeats": repeats,
    }


def _batch_indices(
    sample_count: int,
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    return torch.randint(sample_count, (batch_size,), generator=generator, device=device)


def _load_data_file(path: Path, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("splits"), dict):
        raise ValueError("data-file must contain a dict with a splits object")

    def split(name: str) -> tuple[Tensor, Tensor]:
        value = artifact["splits"].get(name)
        if not isinstance(value, dict):
            raise ValueError(f"data-file is missing split {name}")
        state = value.get("state")
        actions = value.get("actions")
        if not isinstance(state, Tensor) or not isinstance(actions, Tensor):
            raise ValueError(f"split {name} must contain tensor state and actions")
        if state.ndim != 2 or tuple(state.shape[1:]) != (8,):
            raise ValueError(f"split {name} state must have shape [N, 8]")
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (64, 7):
            raise ValueError(f"split {name} actions must have shape [N, 64, 7]")
        if state.shape[0] != actions.shape[0] or state.shape[0] < 1:
            raise ValueError(f"split {name} must contain matching non-empty state/action rows")
        return state.to(device=device, dtype=torch.float32), actions.to(device=device, dtype=torch.float32)

    train_state, train_actions = split("train")
    val_state, val_actions = split("validation")
    return (
        train_state,
        train_actions,
        val_state,
        val_actions,
        str(artifact.get("data_source", "torch_artifact")),
    )


def _train_act(
    train_state: Tensor,
    train_actions: Tensor,
    val_state: Tensor,
    val_actions: Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    config = ACTBaselineConfig(chunk_size=50)
    model = ACTBaseline(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 101)
    records: list[float] = []
    model.train()
    for step in range(steps):
        indices = _batch_indices(train_state.shape[0], batch_size, generator=generator, device=device)
        prediction = model(train_state[indices])
        loss = torch.nn.functional.mse_loss(prediction, train_actions[indices, :50])
        _finite(loss, "ACT training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        records.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        val_prediction = model(val_state)
        val_target = val_actions[:, :50]
        val_loss = torch.nn.functional.mse_loss(val_prediction, val_target)
        val_mae = torch.nn.functional.l1_loss(val_prediction, val_target)
        executed_mse = torch.nn.functional.mse_loss(
            val_prediction[:, :10], val_target[:, :10]
        )
        executed_mae = torch.nn.functional.l1_loss(
            val_prediction[:, :10], val_target[:, :10]
        )
    _finite(val_loss, "ACT validation loss")
    timing = _timed(lambda: model(val_state[:1]), device=device)
    return {
        "name": "act",
        "implementation": "compact_project_smoke",
        "official_contract": {
            "chunk_size": 50,
            "n_action_steps": 10,
            "n_obs_steps": 1,
        },
        "model": {
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "chunk_size": config.chunk_size,
            "model_dim": config.model_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "train": {
            "steps": steps,
            "batch_size": batch_size,
            "initial_loss": records[0],
            "final_loss": records[-1],
            "best_loss": min(records),
        },
        "validation": {
            "action_mse": float(val_loss.detach().cpu()),
            "action_mae": float(val_mae.detach().cpu()),
            "executed_action_mse": float(executed_mse.detach().cpu()),
            "executed_action_mae": float(executed_mae.detach().cpu()),
            "execute_steps": 10,
            "finite": True,
            "output_shape": list(val_prediction.shape),
        },
        "inference": timing,
    }


def _train_diffusion(
    train_state: Tensor,
    train_actions: Tensor,
    val_state: Tensor,
    val_actions: Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    config = DiffusionBaselineConfig(horizon=64, num_train_timesteps=100)
    model = DiffusionBaseline(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    generator = torch.Generator(device=device).manual_seed(seed + 202)
    records: list[float] = []
    model.train()
    for step in range(steps):
        indices = _batch_indices(train_state.shape[0], batch_size, generator=generator, device=device)
        state = train_state[indices]
        actions = train_actions[indices]
        timesteps = torch.randint(
            config.num_train_timesteps,
            (batch_size,),
            generator=generator,
            device=device,
        )
        noise = torch.randn(actions.shape, generator=generator, device=device, dtype=actions.dtype)
        noisy_actions = model.add_noise(actions, timesteps, noise)
        prediction = model(noisy_actions, state, timesteps)
        loss = torch.nn.functional.mse_loss(prediction, noise)
        _finite(loss, "Diffusion training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        records.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        val_timesteps = torch.full(
            (val_state.shape[0],),
            config.num_train_timesteps // 2,
            device=device,
            dtype=torch.long,
        )
        val_noise = torch.randn(
            val_actions.shape,
            generator=generator,
            device=device,
            dtype=val_actions.dtype,
        )
        val_noisy = model.add_noise(val_actions, val_timesteps, val_noise)
        val_prediction = model(val_noisy, val_state, val_timesteps)
        val_loss = torch.nn.functional.mse_loss(val_prediction, val_noise)
    _finite(val_loss, "Diffusion validation loss")

    inference: dict[str, Any] = {}
    sample_generator = torch.Generator(device=device).manual_seed(seed + 303)
    action_eval_generator = torch.Generator(device=device).manual_seed(seed + 404)
    with torch.no_grad():
        action_sample = model.sample(
            val_state,
            num_steps=10,
            generator=action_eval_generator,
        )
    _finite(action_sample, "Diffusion action validation sample")
    executed_target = val_actions[:, :10]
    executed_mse = torch.nn.functional.mse_loss(action_sample[:, :10], executed_target)
    executed_mae = torch.nn.functional.l1_loss(action_sample[:, :10], executed_target)
    for inference_steps in (100, 20, 10, 5):
        timing = _timed(
            lambda inference_steps=inference_steps: model.sample(
                val_state[:1], num_steps=inference_steps, generator=sample_generator
            ),
            device=device,
            warmup=1,
            repeats=3,
        )
        sample = model.sample(val_state[:1], num_steps=inference_steps, generator=sample_generator)
        _finite(sample, f"Diffusion {inference_steps}-step sample")
        inference[str(inference_steps)] = timing
    return {
        "name": "diffusion_policy",
        "implementation": "compact_project_smoke",
        "official_contract": {
            "horizon": 64,
            "n_action_steps": 10,
            "n_obs_steps": 2,
            "train_diffusion_steps": 100,
            "eval_diffusion_steps": [100, 20, 10, 5],
        },
        "model": {
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "horizon": config.horizon,
            "hidden_dim": config.hidden_dim,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "train": {
            "steps": steps,
            "batch_size": batch_size,
            "initial_loss": records[0],
            "final_loss": records[-1],
            "best_loss": min(records),
        },
        "validation": {
            "noise_mse": float(val_loss.detach().cpu()),
            "executed_action_mse": float(executed_mse.detach().cpu()),
            "executed_action_mae": float(executed_mae.detach().cpu()),
            "execute_steps": 10,
            "finite": True,
            "output_shape": list(val_prediction.shape),
        },
        "inference": inference,
    }


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.train_samples < 2 or args.validation_samples < 1:
        raise ValueError("steps, batch-size, train-samples, and validation-samples must be positive")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    device = _select_device(args.device)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.data_file is None:
        train_state, train_actions = make_synthetic_baseline_data(
            args.train_samples,
            seed=args.seed,
            device=device,
        )
        val_state, val_actions = make_synthetic_baseline_data(
            args.validation_samples,
            seed=args.seed + 1,
            device=device,
        )
        data_source = "synthetic_smoke"
    else:
        train_state, train_actions, val_state, val_actions, data_source = _load_data_file(
            args.data_file, device
        )
    outputs: list[dict[str, Any]] = []
    if args.baseline in ("act", "all"):
        outputs.append(
            _train_act(
                train_state,
                train_actions,
                val_state,
                val_actions,
                steps=args.steps,
                batch_size=args.batch_size,
                seed=args.seed,
                device=device,
            )
        )
    if args.baseline in ("diffusion", "all"):
        outputs.append(
            _train_diffusion(
                train_state,
                train_actions,
                val_state,
                val_actions,
                steps=args.steps,
                batch_size=args.batch_size,
                seed=args.seed,
                device=device,
            )
        )
    result = {
        "schema_version": 1,
        "status": "completed",
        "data_source": data_source,
        "input_modalities": ["state"],
        "evaluation_protocol": EvaluationProtocol(normalization="raw_dataset_action").to_dict(),
        "libero_episode_status": "pending",
        "device": str(device),
        "seed": args.seed,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "results": outputs,
        "interpretation": (
            "This artifact validates baseline tensor contracts and short training loops. "
            "It must not be used as a LIBERO success-rate result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
