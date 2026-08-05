#!/usr/bin/env python
"""Evaluate SmolVLA variants in a real LIBERO-Spatial environment.

The evaluator keeps the environment-side processor sequence used by LeRobot:
raw LIBERO observation -> ``preprocess_observation`` -> ``LiberoProcessorStep``
-> policy preprocessor.  In asynchronous mode the action queue keeps the
normalized action space for RTC, and the LeRobot postprocessor runs only after
the control thread pops one action for the simulator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument("--observation-height", type=int, default=256)
    parser.add_argument("--observation-width", type=int, default=256)
    parser.add_argument("--tick-sleep", type=float, default=None)
    parser.add_argument("--overlap-steps", type=int, default=10)
    parser.add_argument("--prefetch-threshold", type=int, default=10)
    parser.add_argument("--disable-rtc", action="store_true")
    parser.add_argument(
        "--disable-rtc-prefix",
        action="store_true",
        help="keep RTC enabled but omit the unexecuted action prefix from policy calls (diagnostic)",
    )
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--rtc-schedule", choices=("linear", "exp"), default="exp")
    parser.add_argument("--gripper-polarity", choices=("positive_open", "negative_open"), default="positive_open")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/rollout/libero_spatial_task0_rollout.json"),
    )
    return parser.parse_args()


def _package_root() -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("the Python package 'libero' is not installed")
    return Path(next(iter(spec.submodule_search_locations))) / "libero"


def _configure_libero(assets_dir: Path | None) -> tuple[Path, Path]:
    """Create the non-interactive LIBERO config and validate local assets."""

    package_root = _package_root()
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")).expanduser()
    config_file = config_dir / "config.yaml"
    config_dir.mkdir(parents=True, exist_ok=True)

    if not config_file.exists():
        import yaml

        config = {
            "benchmark_root": str(package_root),
            "bddl_files": str(package_root / "bddl_files"),
            "init_states": str(package_root / "init_files"),
            "datasets": str(package_root.parent / "datasets"),
            "assets": str(package_root / "assets"),
        }
        config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    selected_assets = assets_dir or Path(os.environ.get("LIBERO_ASSETS_PATH", "~/.cache/libero/assets")).expanduser()
    required = (
        "scenes",
        "articulated_objects",
        "stable_scanned_objects",
        "turbosquid_objects",
    )
    missing = [name for name in required if not (selected_assets / name).exists()]
    if missing:
        raise RuntimeError(
            "LIBERO assets are missing. Expected "
            f"{selected_assets} with subdirectories {', '.join(required)}; missing {', '.join(missing)}. "
            "Download the official lerobot/libero-assets dataset on a networked machine, "
            "copy it to this path, or pass --assets-dir."
        )
    return package_root, selected_assets


def _load_policy(args: argparse.Namespace):
    from huggingface_hub import hf_hub_download
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.rtc import RTCConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if args.flow_steps < 1:
        raise ValueError("flow-steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this rollout on the configured device")

    checkpoint = Path(args.checkpoint).expanduser()
    if checkpoint.is_dir():
        checkpoint_path = checkpoint
    else:
        config_file = Path(hf_hub_download(args.checkpoint, "config.json", local_files_only=True))
        checkpoint_path = config_file.parent

    config = SmolVLAConfig.from_pretrained(checkpoint_path, local_files_only=True)
    config.device = args.device
    config.load_vlm_weights = False
    config.compile_model = False
    config.num_steps = args.flow_steps
    if args.mode == "async" and not args.disable_rtc:
        rtc_schedule = (
            RTCAttentionSchedule.LINEAR if args.rtc_schedule == "linear" else RTCAttentionSchedule.EXP
        )
        config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=10,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=rtc_schedule,
        )
    else:
        config.rtc_config = None

    policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.config.num_steps = args.flow_steps
    policy.model.config.num_steps = args.flow_steps

    contract = (
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
        int(policy.config.action_feature.shape[0]),
    )
    if contract != (50, 32, 7):
        raise RuntimeError(f"expected SmolVLA action contract (50, 32, 7), got {contract}")

    adapter_count = 0
    if args.adapter is not None:
        adapter = torch.load(args.adapter, map_location="cpu", weights_only=True)
        state = policy.state_dict()
        unknown = sorted(set(adapter) - set(state))
        if unknown:
            raise RuntimeError(f"adapter keys are absent from policy: {unknown[:3]}")
        for name, value in adapter.items():
            state[name] = value.to(device=args.device)
            adapter_count += int(value.numel())
        policy.load_state_dict(state, strict=True)

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        checkpoint_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": args.device}},
    )
    return policy, preprocessor, postprocessor, checkpoint_path, adapter_count


def _make_observation_pipeline(task: str):
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline

    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    def batch_robot_state(value: Any) -> Any:
        """Add the batch dimension expected by ``LiberoProcessorStep``.

        ``LiberoEnv`` is a single-environment wrapper and returns each nested
        robot-state leaf without a leading batch dimension.  The generic
        ``preprocess_observation`` helper converts those leaves to tensors but
        intentionally leaves their nesting untouched, while
        ``LiberoProcessorStep`` consumes ``eef.quat`` as ``(B, 4)``.  Batch
        only this nested structure so image and already-batched top-level
        tensors keep their existing shapes.
        """

        if isinstance(value, dict):
            return {key: batch_robot_state(item) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            return value.unsqueeze(0)
        return value

    def prepare(raw_observation: dict[str, Any], policy_preprocessor):
        observation = preprocess_observation(raw_observation)
        robot_state_key = "observation.robot_state"
        if robot_state_key in observation:
            observation[robot_state_key] = batch_robot_state(observation[robot_state_key])
        observation["task"] = [task]
        observation = env_preprocessor(observation)
        return policy_preprocessor(observation)

    return prepare


def _action_statistics(actions: list[np.ndarray]) -> dict[str, float | None]:
    if not actions:
        return {
            "action_l2_mean": None,
            "action_abs_mean": None,
            "action_smoothness_mean": None,
            "action_smoothness_max": None,
        }
    values = np.stack(actions, axis=0).astype(np.float64)
    smoothness = np.abs(np.diff(values, axis=0))
    return {
        "action_l2_mean": float(np.linalg.norm(values, axis=-1).mean()),
        "action_abs_mean": float(np.abs(values).mean()),
        "action_smoothness_mean": float(smoothness.mean()) if len(smoothness) else 0.0,
        "action_smoothness_max": float(smoothness.max()) if len(smoothness) else 0.0,
    }


def _latency_statistics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_seconds": None, "median_seconds": None, "p95_seconds": None, "max_seconds": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999) - 1))
    return {
        "count": len(values),
        "mean_seconds": float(statistics.fmean(values)),
        "median_seconds": float(statistics.median(values)),
        "p95_seconds": float(ordered[p95_index]),
        "max_seconds": float(max(values)),
    }


def _make_env(suite, args: argparse.Namespace, task_id: int, episode_index: int):
    from lerobot.envs.libero import LiberoEnv

    return LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=args.suite,
        episode_length=args.max_steps,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=args.observation_width,
        observation_height=args.observation_height,
        init_states=True,
        episode_index=episode_index,
        n_envs=1,
        control_mode="relative",
    )


def _run_sync_episode(
    env,
    policy,
    policy_preprocessor,
    postprocessor,
    task: str,
    prepare_observation,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    from smolvla_flow.async_runtime import LeRobotPostprocessorAdapter

    policy.reset()
    raw_observation, _ = env.reset(seed=seed)
    postprocess_action = LeRobotPostprocessorAdapter(postprocessor)
    inference_seconds: list[float] = []
    actions: list[np.ndarray] = []
    reward_sum = 0.0
    success = False
    started = time.perf_counter()

    for step in range(max_steps):
        batch = prepare_observation(raw_observation, policy_preprocessor)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            normalized_action = policy.select_action(batch)
        inference_seconds.append(time.perf_counter() - inference_started)
        action = postprocess_action(normalized_action.squeeze(0)).numpy().astype(np.float32)
        actions.append(action.copy())
        raw_observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        success = bool(info.get("is_success", False))
        if bool(terminated or truncated):
            break

    elapsed = time.perf_counter() - started
    return {
        "mode": "sync",
        "seed": seed,
        "success": success,
        "steps": len(actions),
        "reward_sum": reward_sum,
        "elapsed_seconds": elapsed,
        "effective_control_hz": len(actions) / elapsed if elapsed > 0 else None,
        "waiting_ticks": 0,
        "initial_warmup_seconds": None,
        "policy_requests": len(inference_seconds),
        "inference": _latency_statistics(inference_seconds),
        **_action_statistics(actions),
    }


def _run_async_episode(
    env,
    policy,
    policy_preprocessor,
    postprocessor,
    task: str,
    prepare_observation,
    seed: int,
    max_steps: int,
    tick_sleep: float,
    gripper_polarity: str,
    overlap_steps: int,
    prefetch_threshold: int,
    rtc_enabled: bool,
    rtc_use_prefix: bool,
) -> dict[str, Any]:
    policy.reset()
    raw_observation, _ = env.reset(seed=seed)
    runtime_config = AsyncRuntimeConfig(
        control_frequency_hz=20.0,
        chunk_size=50,
        execute_steps=10,
        overlap_steps=overlap_steps,
        prefetch_threshold=prefetch_threshold,
        rtc_enabled=rtc_enabled,
        rtc_use_prefix=rtc_use_prefix,
        rtc_execution_horizon=10,
        gripper_polarity=gripper_polarity,
    )
    action_queue = ActionQueue(overlap_steps=runtime_config.overlap_steps)
    postprocess_action = LeRobotPostprocessorAdapter(postprocessor)

    with torch.inference_mode():
        warmup_started = time.perf_counter()
        initial_actions = policy.predict_action_chunk(prepare_observation(raw_observation, policy_preprocessor))
        warmup_seconds = time.perf_counter() - warmup_started

    def prepare_for_worker(observation: dict[str, Any]) -> dict[str, Any]:
        return prepare_observation(observation, policy_preprocessor)

    server = AsyncPolicyServer(
        PreprocessedPolicyAdapter(policy, prepare_for_worker),
        action_queue,
        runtime_config,
    )
    controller = AsyncClosedLoopController(
        server,
        action_queue,
        runtime_config,
        action_postprocessor=postprocess_action,
        simulation_only=False,
    )
    controller.seed_actions(initial_actions)

    actions: list[np.ndarray] = []
    inference_seconds: list[float] = []
    waiting_ticks = 0
    reward_sum = 0.0
    success = False
    started = time.perf_counter()
    try:
        for _step in range(max_steps):
            tick_started = time.perf_counter()
            tick = controller.tick(raw_observation)
            if tick.action is None:
                waiting_ticks += 1
                break
            action = tick.action.numpy().astype(np.float32)
            actions.append(action.copy())
            raw_observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            success = bool(info.get("is_success", False))
            if bool(terminated or truncated):
                break
            elapsed_tick = time.perf_counter() - tick_started
            if tick_sleep > elapsed_tick:
                time.sleep(tick_sleep - elapsed_tick)
    finally:
        server.wait_for_idle(timeout=30.0)
        events = [event.__dict__ for event in server.events()]
        server_stats = server.stats()
        controller.close()

    elapsed = time.perf_counter() - started
    inference_seconds = [float(event["elapsed_seconds"]) for event in events if event.get("error") is None]
    return {
        "mode": "async",
        "seed": seed,
        "success": success,
        "steps": len(actions),
        "reward_sum": reward_sum,
        "elapsed_seconds": elapsed,
        "effective_control_hz": len(actions) / elapsed if elapsed > 0 else None,
        "waiting_ticks": waiting_ticks,
        "initial_warmup_seconds": warmup_seconds,
        "policy_requests": server_stats["completed"],
        "inference": _latency_statistics(inference_seconds),
        "runtime": {
            "control_frequency_hz": runtime_config.control_frequency_hz,
            "chunk_size": runtime_config.chunk_size,
            "execute_steps": runtime_config.execute_steps,
            "overlap_steps": runtime_config.overlap_steps,
            "prefetch_threshold": runtime_config.trigger_threshold,
            "rtc_enabled": runtime_config.rtc_enabled,
            "rtc_use_prefix": runtime_config.rtc_use_prefix,
            "rtc_execution_horizon": runtime_config.rtc_execution_horizon,
            "gripper_polarity": runtime_config.gripper_polarity,
            "tick_sleep_seconds": tick_sleep,
        },
        "server": server_stats,
        "inference_events": events,
        **_action_statistics(actions),
    }


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [bool(item["success"]) for item in episodes]
    return {
        "episode_count": len(episodes),
        "success_count": sum(successes),
        "success_rate": float(sum(successes) / len(successes)) if successes else None,
        "mean_steps": float(statistics.fmean(item["steps"] for item in episodes)) if episodes else None,
        "mean_reward_sum": float(statistics.fmean(item["reward_sum"] for item in episodes)) if episodes else None,
        "mean_waiting_ticks": float(statistics.fmean(item["waiting_ticks"] for item in episodes)) if episodes else None,
        "mean_action_smoothness": float(
            statistics.fmean(item["action_smoothness_mean"] for item in episodes if item["action_smoothness_mean"] is not None)
        )
        if any(item["action_smoothness_mean"] is not None for item in episodes)
        else None,
        "mean_effective_control_hz": float(
            statistics.fmean(item["effective_control_hz"] for item in episodes if item["effective_control_hz"] is not None)
        )
        if any(item["effective_control_hz"] is not None for item in episodes)
        else None,
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "blocked",
        "checkpoint": args.checkpoint,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "mode": args.mode,
        "flow_steps": args.flow_steps,
        "suite": args.suite,
        "task_id": args.task_id,
        "episodes_requested": args.episodes,
        "start_seed": args.start_seed,
    }
    try:
        if args.episodes < 1:
            raise ValueError("episodes must be positive")
        package_root, assets_dir = _configure_libero(args.assets_dir)
        from libero.libero import benchmark
        # LeRobot's LIBERO wrapper resolves assets through LIBERO's module cache.
        # Point that cache at the validated path so --assets-dir is effective even
        # when the assets are stored outside ~/.cache/libero/assets.
        import libero.libero as libero_core

        libero_core._assets_path_cache = str(assets_dir)

        suite_map = benchmark.get_benchmark_dict()
        if args.suite not in suite_map:
            raise ValueError(f"unknown LIBERO suite {args.suite!r}; available={sorted(suite_map)}")
        suite = suite_map[args.suite]()
        task_info = suite.get_task(args.task_id)
        task = args.task or task_info.language or TASK0_LANGUAGE
        policy, policy_preprocessor, postprocessor, checkpoint_path, adapter_parameter_count = _load_policy(args)
        prepare_observation = _make_observation_pipeline(task)
        max_steps = args.max_steps or {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300}.get(args.suite, 500)
        tick_sleep = 1.0 / 20.0 if args.tick_sleep is None and args.mode == "async" else float(args.tick_sleep or 0.0)

        episode_results = []
        for offset in range(args.episodes):
            seed = args.start_seed + offset
            env = _make_env(suite, args, args.task_id, seed)
            try:
                if args.mode == "sync":
                    episode = _run_sync_episode(
                        env,
                        policy,
                        policy_preprocessor,
                        postprocessor,
                        task,
                        prepare_observation,
                        seed,
                        max_steps,
                    )
                else:
                    episode = _run_async_episode(
                        env,
                        policy,
                        policy_preprocessor,
                        postprocessor,
                        task,
                        prepare_observation,
                        seed,
                        max_steps,
                        tick_sleep,
                        args.gripper_polarity,
                        args.overlap_steps,
                        args.prefetch_threshold,
                        not args.disable_rtc,
                        not args.disable_rtc_prefix,
                    )
            finally:
                env.close()
            episode_results.append(episode)
            print(json.dumps(episode, indent=2), flush=True)

        result.update(
            {
                "status": "completed",
                "package_root": str(package_root),
                "assets_dir": str(assets_dir),
                "checkpoint_path": str(checkpoint_path),
                "task": task,
                "adapter_parameter_count": adapter_parameter_count,
                "episodes": episode_results,
                "aggregate": _aggregate(episode_results),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["next_action"] = "Provide LIBERO assets and rerun this command."
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 2

    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
