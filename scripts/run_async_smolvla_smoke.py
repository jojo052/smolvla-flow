#!/usr/bin/env python
"""Run the asynchronous runtime with the distilled 2-step SmolVLA policy."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import torch

from smolvla_flow.async_runtime import (
    ActionQueue,
    AsyncClosedLoopController,
    AsyncPolicyServer,
    AsyncRuntimeConfig,
    LeRobotPostprocessorAdapter,
    PreprocessedPolicyAdapter,
)


TASK0_LANGUAGE = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def _image_to_tensor(value):
    import numpy as np
    from PIL import Image

    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        image = Image.open(io.BytesIO(bytes(value)))
    else:
        image = Image.open(value)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=1272)
    parser.add_argument("--task", default=TASK0_LANGUAGE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--tick-sleep", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/async_smolvla_smoke.json"))
    return parser.parse_args()


def _load_observation(parquet_path: Path, episode_index: int, task: str) -> dict:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

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


def _load_policy(checkpoint: str, adapter_path: Path, device: str):
    from huggingface_hub import hf_hub_download
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.rtc import RTCConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint_path = Path(hf_hub_download(checkpoint, "config.json", local_files_only=True)).parent
    config = SmolVLAConfig.from_pretrained(checkpoint_path, local_files_only=True)
    config.device = device
    config.load_vlm_weights = False
    config.compile_model = False
    config.num_steps = 2
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=config,
        local_files_only=True,
        strict=True,
    )
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
    state = policy.state_dict()
    unknown = sorted(set(adapter) - set(state))
    if unknown:
        raise RuntimeError(f"adapter keys are absent from policy: {unknown[:3]}")
    for name, value in adapter.items():
        state[name] = value.to(device=device)
    policy.load_state_dict(state, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        checkpoint_path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, checkpoint_path


def main() -> None:
    args = parse_args()
    if args.ticks < 1:
        raise SystemExit("ticks must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for a real SmolVLA smoke")
    observation = _load_observation(args.parquet, args.episode_index, args.task)
    policy, preprocessor, postprocessor, checkpoint_path = _load_policy(args.checkpoint, args.adapter, args.device)
    policy.eval()
    config = AsyncRuntimeConfig(
        control_frequency_hz=20.0,
        chunk_size=50,
        execute_steps=10,
        overlap_steps=10,
        rtc_enabled=True,
        rtc_execution_horizon=10,
        gripper_polarity="positive_open",
    )
    action_queue = ActionQueue(overlap_steps=config.overlap_steps)
    warmup_started = time.monotonic()
    with torch.inference_mode():
        initial_batch = preprocessor(dict(observation))
        initial_actions = policy.predict_action_chunk(initial_batch)
    warmup_seconds = time.monotonic() - warmup_started
    server = AsyncPolicyServer(
        PreprocessedPolicyAdapter(policy, preprocessor),
        action_queue,
        config,
    )
    controller = AsyncClosedLoopController(
        server,
        action_queue,
        config,
        action_postprocessor=LeRobotPostprocessorAdapter(postprocessor),
        simulation_only=False,
    )
    controller.seed_actions(initial_actions)
    tick_sleep = config.period_seconds if args.tick_sleep is None else args.tick_sleep
    tick_results = []
    started = time.monotonic()
    try:
        for tick in range(args.ticks):
            result = controller.tick(observation)
            tick_results.append(result)
            if tick_sleep:
                time.sleep(tick_sleep)
        server.wait_for_idle(timeout=10.0)
    finally:
        controller.close()
    elapsed = time.monotonic() - started
    events = server.events()
    result = {
        "status": "completed",
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "adapter": str(args.adapter),
        "episode_index": args.episode_index,
        "runtime": {
            "control_frequency_hz": config.control_frequency_hz,
            "chunk_size": config.chunk_size,
            "execute_steps": config.execute_steps,
            "overlap_steps": config.overlap_steps,
            "prefetch_threshold": config.trigger_threshold,
            "rtc_enabled": config.rtc_enabled,
            "rtc_execution_horizon": config.rtc_execution_horizon,
            "gripper_polarity": config.gripper_polarity,
            "postprocessor_enabled": True,
        },
        "ticks": args.ticks,
        "elapsed_seconds": elapsed,
        "initial_warmup_seconds": warmup_seconds,
        "actions_emitted": sum(result.action is not None for result in tick_results),
        "waiting_ticks": sum(result.waiting_for_policy for result in tick_results),
        "triggered_inference_ticks": sum(result.triggered_inference for result in tick_results),
        "queue_depth_min": min(result.queue_depth_after for result in tick_results),
        "queue_depth_max": max(result.queue_depth_after for result in tick_results),
        "server": server.stats(),
        "inference_events": [event.__dict__ for event in events],
        "output_shape": [1, 50, 7],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
