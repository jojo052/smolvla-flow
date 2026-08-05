#!/usr/bin/env python
"""Benchmark a native 10-step SmolVLA and a distilled 2-step checkpoint."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from pathlib import Path


TASK0_LANGUAGE = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=1272)
    parser.add_argument("--task", default=TASK0_LANGUAGE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/distillation/task0_dev5_masked/benchmark.json"))
    return parser.parse_args()


def _image_to_tensor(value):
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        image = Image.open(io.BytesIO(bytes(value)))
    else:
        image = Image.open(value)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _load_policy(checkpoint_path: Path, flow_steps: int, device: str):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig.from_pretrained(checkpoint_path, local_files_only=True)
    config.device = device
    config.load_vlm_weights = False
    config.compile_model = False
    if not config.use_cache:
        raise ValueError("benchmark requires SmolVLA use_cache=True")
    config.num_steps = flow_steps
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.config.num_steps = flow_steps
    policy.model.config.num_steps = flow_steps
    contract = (
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
        int(policy.config.action_feature.shape[0]),
    )
    if contract != (50, 32, 7):
        raise ValueError(f"expected checkpoint action contract (50, 32, 7), got {contract}")
    preprocessor, _ = make_pre_post_processors(
        config,
        checkpoint_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor


def _load_observation(parquet_path: Path, episode_index: int, task: str):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch

    table = pq.read_table(parquet_path)
    table = table.filter(pc.equal(table["episode_index"], episode_index))
    if table.num_rows == 0:
        raise RuntimeError(f"episode {episode_index} is absent from {parquet_path}")
    row = table.slice(0, 1).to_pylist()[0]
    return {
        "observation.images.image": _image_to_tensor(row["observation.images.image"]),
        "observation.images.image2": _image_to_tensor(row["observation.images.image2"]),
        "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
        "task": task,
    }


def _load_adapter(policy, adapter_path: Path, device: str) -> dict:
    import torch

    adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
    state = policy.state_dict()
    missing_in_base = sorted(set(adapter) - set(state))
    if missing_in_base:
        raise RuntimeError(f"adapter keys are absent from policy: {missing_in_base[:3]}")
    for name, value in adapter.items():
        state[name] = value.to(device=device)
    policy.load_state_dict(state, strict=True)
    return {"adapter_parameter_count": sum(value.numel() for value in adapter.values())}


def _run_once(policy, processed):
    import torch

    with torch.inference_mode():
        return policy.predict_action_chunk(processed)


def _benchmark(policy, processed, warmup: int, repeats: int) -> dict:
    import torch

    for _ in range(warmup):
        _run_once(policy, processed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    durations = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = _run_once(policy, processed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "mean_seconds": statistics.fmean(durations),
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "p95_seconds": sorted(durations)[max(0, min(len(durations) - 1, int(0.95 * len(durations) + 0.999) - 1))],
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download

    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("warmup must be non-negative and repeats must be positive")
    config_path = Path(hf_hub_download(args.checkpoint, "config.json", local_files_only=True))
    checkpoint_path = config_path.parent
    observation = _load_observation(args.parquet, args.episode_index, args.task)
    teacher, teacher_preprocessor = _load_policy(checkpoint_path, 10, args.device)
    student, student_preprocessor = _load_policy(checkpoint_path, 2, args.device)
    adapter_info = _load_adapter(student, args.adapter, args.device)
    teacher_processed = teacher_preprocessor(observation)
    student_processed = student_preprocessor(observation)
    teacher.eval()
    student.eval()
    teacher_result = _benchmark(teacher, teacher_processed, args.warmup, args.repeats)
    student_result = _benchmark(student, student_processed, args.warmup, args.repeats)
    torch.manual_seed(args.seed)
    teacher_action = _run_once(teacher, teacher_processed)
    torch.manual_seed(args.seed)
    student_action = _run_once(student, student_processed)
    comparison = {
        "same_seed_action_mse": float((teacher_action - student_action).square().mean().cpu()),
        "same_seed_action_mae": float((teacher_action - student_action).abs().mean().cpu()),
        "teacher_action_shape": list(teacher_action.shape),
        "student_action_shape": list(student_action.shape),
    }
    result = {
        "checkpoint": args.checkpoint,
        "adapter": str(args.adapter),
        "parquet": str(args.parquet),
        "episode_index": args.episode_index,
        "task": args.task,
        "teacher_flow_steps": 10,
        "student_flow_steps": 2,
        "teacher": teacher_result,
        "student": {**student_result, **adapter_info},
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
