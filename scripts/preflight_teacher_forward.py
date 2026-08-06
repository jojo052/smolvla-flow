#!/usr/bin/env python
"""Run one real LIBERO observation through a 10-step SmolVLA teacher."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=1272)
    parser.add_argument(
        "--task",
        default="pick up the black bowl between the plate and the ramekin and place it on the plate",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-flow-steps", type=int, nargs="*", default=[])
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=10)
    parser.add_argument(
        "--skip-base-vlm-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Construct the architecture from the base VLM config, then load all weights from the policy checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/preflight/teacher_forward.json"),
    )
    return parser.parse_args()


def _image_to_tensor(value):
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path") is not None:
            value = Path(value["path"])
    if isinstance(value, (bytes, bytearray, memoryview)):
        image = Image.open(io.BytesIO(bytes(value)))
    elif isinstance(value, (str, Path)):
        image = Image.open(value)
    else:
        image = value
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _shape_summary(batch: dict) -> dict:
    result = {}
    for key, value in batch.items():
        if hasattr(value, "shape"):
            result[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device) if hasattr(value, "device") else None,
            }
        else:
            result[key] = {"type": type(value).__name__}
    return result


def main() -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "preflight_teacher_forward.py requires a CUDA device; "
            "torch.cuda.is_available() is false"
        )
    from huggingface_hub import hf_hub_download
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    table = pq.read_table(args.parquet)
    table = table.filter(pc.equal(table["episode_index"], args.episode_index))
    if table.num_rows == 0:
        raise RuntimeError(f"episode {args.episode_index} is absent from {args.parquet}")
    row = table.slice(0, 1).to_pylist()[0]
    raw_observation = {
        "observation.images.image": _image_to_tensor(row["observation.images.image"]),
        "observation.images.image2": _image_to_tensor(row["observation.images.image2"]),
        "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
        "task": args.task,
    }

    config_path = Path(
        hf_hub_download(
            repo_id=args.checkpoint,
            filename="config.json",
            local_files_only=True,
        )
    )
    checkpoint_path = config_path.parent
    config = SmolVLAConfig.from_pretrained(checkpoint_path, local_files_only=True)
    config.device = args.device
    config.load_vlm_weights = not args.skip_base_vlm_weights
    if config.num_steps != 10:
        raise RuntimeError(f"expected a 10-step teacher, found num_steps={config.num_steps}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=config,
        local_files_only=True,
        strict=True,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    load_peak_bytes = torch.cuda.max_memory_allocated()

    preprocessor, _ = make_pre_post_processors(
        config,
        checkpoint_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    processed = preprocessor(raw_observation)

    policy.eval()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        actions = policy.predict_action_chunk(processed)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    latency_benchmarks = []
    for flow_steps in args.benchmark_flow_steps:
        policy.config.num_steps = flow_steps
        policy.model.config.num_steps = flow_steps
        with torch.inference_mode():
            for _ in range(args.benchmark_warmup):
                policy.predict_action_chunk(processed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        with torch.inference_mode():
            for _ in range(args.benchmark_repeats):
                started = time.perf_counter()
                benchmark_actions = policy.predict_action_chunk(processed)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
        sorted_durations = sorted(durations)
        p95_index = max(0, min(len(sorted_durations) - 1, int(0.95 * len(sorted_durations) + 0.999) - 1))
        latency_benchmarks.append(
            {
                "flow_steps": flow_steps,
                "warmup_calls": args.benchmark_warmup,
                "measured_calls": args.benchmark_repeats,
                "mean_seconds": statistics.fmean(durations),
                "median_seconds": statistics.median(durations),
                "p95_seconds": sorted_durations[p95_index],
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                "output_shape": list(benchmark_actions.shape),
                "output_finite": bool(torch.isfinite(benchmark_actions).all().item()),
                "distilled_model": False,
                "interpretation": "undistilled teacher short-step latency baseline"
                if flow_steps != 10
                else "native 10-step teacher latency baseline",
            }
        )
    policy.config.num_steps = 10
    policy.model.config.num_steps = 10

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "parquet": str(args.parquet),
        "episode_index": args.episode_index,
        "frame_index": int(row["frame_index"]),
        "task": args.task,
        "flow_steps": int(config.num_steps),
        "strict_state_dict_load": True,
        "base_vlm_weights_loaded_before_policy": bool(config.load_vlm_weights),
        "raw_inputs": _shape_summary(raw_observation),
        "processed_inputs": _shape_summary(processed),
        "output_shape": list(actions.shape),
        "output_dtype": str(actions.dtype),
        "output_device": str(actions.device),
        "output_finite": bool(torch.isfinite(actions).all().item()),
        "load_seconds": load_seconds,
        "load_peak_gpu_bytes": int(load_peak_bytes),
        "inference_seconds": inference_seconds,
        "inference_peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "latency_benchmarks": latency_benchmarks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
