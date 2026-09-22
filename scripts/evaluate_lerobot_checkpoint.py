#!/usr/bin/env python
"""Evaluate a visual LeRobot checkpoint on held-out task34 action windows.

This is the offline half of the unified matrix.  It uses the same two camera
keys, checkpoint preprocessor/postprocessor, 8D state, and common 10-action
execution prefix as the LIBERO rollout.  The result contains action MSE/MAE
and policy-only latency; environment ``step`` latency is reported only by the
online rollout.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.run_libero_rollout import _load_policy
from smolvla_flow.evaluation_protocol import EvaluationProtocol, action_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-type", choices=("smolvla", "act", "diffusion"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-index", type=int, default=34)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all valid held-out windows")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _episode_bounds(meta: Any, episode_index: int) -> tuple[int, int]:
    episodes = meta.episodes
    if isinstance(episodes, dict):
        return (
            int(episodes["dataset_from_index"][episode_index]),
            int(episodes["dataset_to_index"][episode_index]),
        )
    row = episodes[episode_index]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def _checkpoint_image_keys(policy: Any, camera_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the checkpoint's visual feature keys for the locked cameras.

    ``EvaluationProtocol.camera_names`` records the LIBERO environment camera
    contract.  Materialized LeRobot datasets and their checkpoints expose
    those streams through policy feature keys such as
    ``observation.images.image`` and ``observation.images.image2``.  The two
    namespaces therefore must agree by stream count, not by literal name.
    """

    config = getattr(policy, "config", None)
    input_features = getattr(config, "input_features", None)
    if not isinstance(input_features, dict):
        raise RuntimeError("checkpoint config does not expose input_features")
    image_keys = tuple(str(key) for key in input_features if str(key).startswith("observation.images."))
    if len(image_keys) != len(camera_names):
        raise RuntimeError(
            "checkpoint visual stream count does not match the locked LIBERO camera contract: "
            f"checkpoint={image_keys}, cameras={camera_names}"
        )
    return image_keys


def _sample_batch(sample: dict[str, Any], image_feature_keys: tuple[str, ...]) -> dict[str, Any]:
    """Keep only policy inputs and add a batch dimension."""

    batch: dict[str, Any] = {}
    expected_images = set(image_feature_keys)
    available_images = {key for key in sample if key.startswith("observation.images.")}
    missing = sorted(expected_images - available_images)
    if missing:
        raise RuntimeError(f"dataset sample is missing checkpoint image features: {missing}")
    for key, value in sample.items():
        if key in {"action", "episode_index", "frame_index", "timestamp", "index", "task_index"}:
            continue
        if key == "task":
            batch[key] = [value if isinstance(value, str) else str(value)]
        elif isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        elif key.startswith("observation."):
            batch[key] = value
    return batch


def _observation_delta_timestamps(
    config: Any,
    feature_keys: Any,
    fps: float,
) -> dict[str, list[float]] | None:
    """Build only the observation history requested by a policy config.

    LeRobot's training dataset also samples future actions.  The evaluator
    reads raw target actions itself so that every policy is compared on the
    same 10-step prefix; adding action deltas here would change those targets
    into nested action horizons.
    """

    delta_indices = getattr(config, "observation_delta_indices", None)
    if delta_indices is None:
        return None
    if fps <= 0:
        raise ValueError("dataset fps must be positive")
    observation_keys = [str(key) for key in feature_keys if str(key).startswith("observation.")]
    if not observation_keys:
        raise RuntimeError("dataset metadata does not expose observation features")
    offsets = [float(index) / float(fps) for index in delta_indices]
    return {key: offsets.copy() for key in observation_keys}


def _raw_action(postprocessor: Any, action: torch.Tensor) -> torch.Tensor:
    processed = postprocessor(action.unsqueeze(0))
    if not isinstance(processed, torch.Tensor):
        raise TypeError("checkpoint postprocessor must return a torch.Tensor")
    if processed.ndim == 2 and processed.shape[0] == 1:
        processed = processed[0]
    if processed.ndim != 1:
        raise ValueError("checkpoint postprocessor must return one action vector")
    return processed.detach().cpu().float()


def _latency(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_seconds": None, "median_seconds": None, "p95_seconds": None}
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999) - 1))
    return {
        "count": len(values),
        "mean_seconds": float(statistics.fmean(values)),
        "median_seconds": float(statistics.median(values)),
        "p95_seconds": float(ordered[index]),
    }


def _policy_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        policy_type=args.policy_type,
        checkpoint=args.checkpoint,
        adapter=args.adapter,
        mode="sync",
        flow_steps=10,
        device=args.device,
        disable_rtc=True,
        rtc_schedule="exp",
        rtc_max_guidance_weight=10.0,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_stride < 1 or args.max_samples < 0 or args.warmup < 0:
        raise ValueError("sample-stride, max-samples, and warmup must be non-negative; stride must be positive")
    protocol = EvaluationProtocol(dataset_task_index=args.task_index)
    policy, preprocessor, postprocessor, checkpoint_path, adapter_count, chunk_size, action_dim = _load_policy(
        _policy_args(args)
    )
    image_feature_keys = _checkpoint_image_keys(policy, protocol.camera_names)
    if action_dim != protocol.action_dim or chunk_size < protocol.action_execution_steps:
        raise RuntimeError("checkpoint does not satisfy the locked state/action execution contract")

    try:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("LeRobot is required for visual checkpoint evaluation") from error
    dataset_kwargs: dict[str, Any] = {"repo_id": args.dataset_repo_id}
    if args.dataset_root is not None:
        dataset_kwargs["root"] = args.dataset_root
    dataset_meta = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )
    observation_deltas = _observation_delta_timestamps(
        policy.config,
        dataset_meta.features,
        dataset_meta.fps,
    )
    if observation_deltas is not None:
        dataset_kwargs["delta_timestamps"] = observation_deltas
    dataset = LeRobotDataset(**dataset_kwargs)

    starts: list[int] = []
    selected_episodes = getattr(dataset, "episodes", None)
    if selected_episodes is None:
        selected_episodes = list(range(int(dataset.num_episodes)))
    for episode_index in selected_episodes:
        first, last = _episode_bounds(dataset.meta, episode_index)
        for start in range(first, max(first, last - protocol.action_execution_steps + 1), args.sample_stride):
            if start + protocol.action_execution_steps <= last:
                starts.append(start)
    if args.max_samples:
        starts = starts[: args.max_samples]
    if not starts:
        raise RuntimeError("no valid held-out action windows were found")

    policy.eval()
    policy_latency: list[float] = []
    preprocess_latency: list[float] = []
    postprocess_latency: list[float] = []
    mse_values: list[float] = []
    mae_values: list[float] = []
    l2_values: list[float] = []
    finite = True

    def run_one(start: int) -> dict[str, float]:
        nonlocal finite
        sample = dataset[start]
        task_index = sample.get("task_index")
        if isinstance(task_index, torch.Tensor):
            task_index = int(task_index.item())
        if task_index is not None and int(task_index) != args.task_index:
            raise RuntimeError(f"dataset task index mixed at frame {start}: {task_index}")
        raw_batch = _sample_batch(sample, image_feature_keys)
        preprocess_started = time.perf_counter()
        batch = preprocessor(raw_batch)
        preprocess_latency.append(time.perf_counter() - preprocess_started)
        policy_started = time.perf_counter()
        with torch.inference_mode():
            prediction = policy.predict_action_chunk(batch)
        policy_latency.append(time.perf_counter() - policy_started)
        if prediction.ndim == 2:
            prediction = prediction.unsqueeze(0)
        if prediction.ndim != 3 or prediction.shape[-1] != protocol.action_dim:
            raise RuntimeError(f"policy output must be [B, T, 7], got {tuple(prediction.shape)}")
        finite = finite and bool(torch.isfinite(prediction).all().item())
        postprocess_started = time.perf_counter()
        predicted_raw = torch.stack(
            [_raw_action(postprocessor, action) for action in prediction[0, : protocol.action_execution_steps]],
            dim=0,
        )
        postprocess_latency.append(time.perf_counter() - postprocess_started)
        target = torch.stack(
            [torch.as_tensor(dataset[start + offset]["action"]).float() for offset in range(protocol.action_execution_steps)],
            dim=0,
        )
        errors = action_error(predicted_raw, target, execute_steps=protocol.action_execution_steps)
        mse_values.append(float(errors["action_mse"]))
        mae_values.append(float(errors["action_mae"]))
        l2_values.append(float(errors["action_l2_mean"]))
        return errors

    warmup_count = min(args.warmup, len(starts))
    for start in starts[:warmup_count]:
        run_one(start)
    mse_values.clear()
    mae_values.clear()
    l2_values.clear()
    preprocess_latency.clear()
    policy_latency.clear()
    postprocess_latency.clear()
    eval_starts = starts[warmup_count:] or starts
    policy.reset()
    for start in eval_starts:
        run_one(start)

    result = {
        "schema_version": 1,
        "status": "completed",
        "policy_type": args.policy_type,
        "checkpoint_path": str(checkpoint_path),
        "adapter_parameter_count": adapter_count,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root) if args.dataset_root is not None else None,
        "input_modalities": ["images", "state"],
        "checkpoint_image_features": list(image_feature_keys),
        "observation_delta_timestamps": observation_deltas,
        "evaluation_protocol": protocol.to_dict(),
        "sample_count": len(mse_values),
        "sample_stride": args.sample_stride,
        "action_error_space": "raw_action_after_checkpoint_postprocessor",
        "action_error": {
            "execute_steps": protocol.action_execution_steps,
            "action_mse_mean": float(statistics.fmean(mse_values)),
            "action_mae_mean": float(statistics.fmean(mae_values)),
            "action_l2_mean": float(statistics.fmean(l2_values)),
        },
        "latency": {
            "preprocess": _latency(preprocess_latency),
            "policy_only": _latency(policy_latency),
            "postprocess": _latency(postprocess_latency),
            "offline_end_to_end": _latency(
                [a + b + c for a, b, c in zip(preprocess_latency, policy_latency, postprocess_latency)]
            ),
        },
        "finite": finite,
        "note": (
            "This is held-out offline action error and policy latency. "
            "LIBERO success rate and env.step latency come from run_libero_rollout.py."
        ),
    }
    return result


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate(args)
    except Exception as error:
        result = {"status": "blocked", "error": f"{type(error).__name__}: {error}"}
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
