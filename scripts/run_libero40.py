#!/usr/bin/env python3
"""Frozen, resumable SmolVLA public benchmark. ACT requires a separate verified adapter."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from smolvla_flow.benchmark40 import (
    SUITES, OFFICIAL_SHA256, atomic_json, episode_key, fingerprint,
    freeze_manifest, markdown_report, sha256, summarize, validate_episode,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--assets-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/libero40_public_v1.json")
    p.add_argument("--phase", choices=["pilot", "formal", "all", "report"], default="all")
    p.add_argument("--acknowledge-errors", action="store_true", help="Resume only after an operator has inspected recorded errors")
    return p.parse_args(argv)


def save_report(output, manifest):
    run_id = fingerprint(manifest)
    rows = []
    for file in sorted((output / "formal").glob("*/*/task*/init*.json")):
        row = json.loads(file.read_text())
        validate_episode(row, run_id, str(file.relative_to(output).with_suffix("")))
        task = next(t for t in manifest["tasks"] if t["suite"] == row["suite"] and t["task_id"] == row["task_id"])
        if row["init_state_sha256"] != task["state_sha256"][row["init_index"]]:
            raise ValueError("Episode initial state hash differs from manifest")
        rows.append(row)
    summary = summarize(rows, ["smolvla_official"])
    summary["act_status"] = manifest["protocol"]["act_status"]
    summary["run_id"] = run_id
    atomic_json(output / "results.json", summary)
    text = markdown_report(summary) + "\nACT：公开权重尚未通过核验，未运行；不形成两模型优劣结论。\n"
    temporary = output / "results.md.tmp"
    temporary.write_text(text)
    temporary.replace(output / "results.md")
    return len(rows)


def prepare_manifest(args, config):
    from scripts.run_libero_rollout import _configure_libero
    import torch
    package_root, assets = _configure_libero(args.assets_dir)
    import libero.libero as core
    from libero.libero import benchmark, get_libero_path
    core._assets_path_cache = str(assets)
    suites, tasks = {}, []
    for name, limit in SUITES.items():
        suite = benchmark.get_benchmark_dict()[name]()
        suites[name] = suite
        if suite.n_tasks != 10:
            raise ValueError(f"{name}: expected 10 tasks")
        for tid in range(10):
            task = suite.get_task(tid)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            init = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            states = suite.get_task_init_states(tid)
            if len(states) < 11 or not task.language:
                raise ValueError("Missing initial states or instruction")
            state_hashes = [__import__("hashlib").sha256(x.detach().cpu().numpy().tobytes() if isinstance(x, torch.Tensor) else x.tobytes()).hexdigest() for x in states[:11]]
            tasks.append({"suite": name, "task_id": tid, "name": task.name, "language": task.language,
                          "bddl_sha256": sha256(bddl), "init_file_sha256": sha256(init),
                          "state_sha256": state_hashes, "max_steps": limit})
    if len({(x["suite"], x["name"]) for x in tasks}) != 40:
        raise ValueError("Duplicate task identity")
    files = {p.name: sha256(p) for p in sorted(args.checkpoint.iterdir()) if p.is_file()}
    if files.get("model.safetensors") != OFFICIAL_SHA256:
        raise ValueError("Official checkpoint hash mismatch; refusing substituted weights")
    sources = [Path(__file__), ROOT / "src/smolvla_flow/benchmark40.py", ROOT / "scripts/run_libero_rollout.py", ROOT / "src/smolvla_flow/async_runtime.py"]
    from lerobot.envs import libero as env_module
    from lerobot.processor import env_processor
    from lerobot.policies.smolvla import modeling_smolvla
    sources += [Path(m.__file__) for m in (env_module, env_processor, modeling_smolvla)]
    manifest = {"protocol": config, "checkpoint": str(args.checkpoint.resolve()), "checkpoint_files": files,
                "tasks": tasks, "sources": {str(p): sha256(p) for p in sources},
                "versions": {name: importlib.metadata.version(name) for name in ["torch", "lerobot", "mujoco", "robosuite", "transformers"]},
                "python": sys.version, "gpu": torch.cuda.get_device_name(0),
                "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
                "gpu_driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
                "assets_sha256": {str(p.relative_to(assets)): sha256(p) for p in sorted(assets.rglob("*")) if p.is_file() and ".git" not in p.parts and p.name != ".DS_Store"}}
    return manifest, suites


def run_episode(env, loaded, task, index, phase, run_id, video_paths):
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from scripts.run_libero_rollout import _make_observation_pipeline
    from smolvla_flow.async_runtime import seed_policy_rng, LeRobotPostprocessorAdapter

    seed_policy_rng(123)
    policy = loaded.policy
    policy.reset()
    if any(len(q) for q in policy._queues.values()):
        raise RuntimeError("Policy queues not empty after reset")
    env.init_state_id = index
    if index >= len(env._init_states) or env.num_steps_wait != 10:
        raise RuntimeError("Initial state or settling contract violated")
    torch.cuda.reset_peak_memory_stats()
    total_started = time.perf_counter()
    observation, _ = env.reset(seed=123)
    seed_policy_rng(123)
    if env.init_state_id != index + 1:
        raise RuntimeError("Initial state selection did not advance exactly once")
    reset_seconds = time.perf_counter() - total_started
    prepare = _make_observation_pipeline(task["language"])
    postprocess = LeRobotPostprocessorAdapter(loaded.postprocessor)
    select_times, prediction_times = [], []
    original = policy._get_action_chunk

    def measured_chunk(*a, **kw):
        torch.cuda.synchronize()
        start = time.perf_counter()
        chunk = original(*a, **kw)
        torch.cuda.synchronize()
        prediction_times.append(time.perf_counter() - start)
        if tuple(chunk.shape) != (1, 50, 7) or not torch.isfinite(chunk).all().item():
            raise RuntimeError("Full prediction violates finite [1,50,7] contract")
        return chunk

    policy._get_action_chunk = measured_chunk
    frames, success = [], False
    record_video = not all(p.exists() for p in video_paths.values())
    reward_sum, action_abs_max = 0., 0.
    initial_batch = None
    started = time.perf_counter()
    try:
        for step in range(task["max_steps"]):
            batch = prepare(observation, loaded.preprocessor)
            if initial_batch is None:
                state = batch["observation.state"]
                if tuple(state.shape) != (1, 8) or not torch.isfinite(state).all().item():
                    raise RuntimeError("Invalid policy state")
                images = {k: list(v.shape) for k, v in batch.items() if k.startswith("observation.images.")}
                if sorted(images.values()) != [[1, 3, 256, 256], [1, 3, 256, 256]]:
                    raise RuntimeError(f"Invalid policy images: {images}")
                initial_batch = {"state_shape": list(state.shape), "images": images,
                                 "image_transform": "LiberoProcessorStep flip H and W"}
            if record_video:
                frames.append(np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1]))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                normalized = policy.select_action(batch)
            torch.cuda.synchronize()
            select_times.append(time.perf_counter() - begin)
            action = postprocess(normalized.squeeze(0)).numpy().astype(np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError("Nonfinite or malformed environment action")
            action_abs_max = max(action_abs_max, float(np.abs(action).max()))
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            success = bool(info["is_success"])
            if success:
                break
            if terminated or truncated:
                raise RuntimeError("Unexpected early environment termination without success")
        elapsed = time.perf_counter() - started
        total_seconds = time.perf_counter() - total_started
    finally:
        policy._get_action_chunk = original
    if len(prediction_times) != len(select_times):
        raise RuntimeError("Expected a fresh prediction for every executed action")
    video = video_paths["success" if success else "failure"]
    if not video.exists() and record_video:
        frames.append(np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1]))
        temp_video = video.with_name(video.stem + ".tmp.mp4")
        imageio.mimwrite(temp_video, frames, fps=20, codec="libx264", quality=6)
        temp_video.replace(video)
    return {"status": "complete", "run_id": run_id, "phase": phase, "model": "smolvla_official",
            "key": episode_key("smolvla_official", task["suite"], task["task_id"], index, phase),
            "suite": task["suite"], "task_id": task["task_id"], "task_name": task["name"],
            "language": task["language"], "init_index": index, "init_state_sha256": task["state_sha256"][index],
            "policy_seed": 123, "environment_seed": 123, "settle_steps": 10,
            "steps": len(select_times), "success": success, "failure_type": None if success else "timeout",
            "reward_sum": reward_sum, "elapsed_seconds": elapsed, "total_seconds": total_seconds,
            "reset_seconds": reset_seconds, "prediction_seconds": prediction_times, "select_seconds": select_times,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(), "action_nonfinite_count": 0,
            "action_abs_max": action_abs_max, "input_contract": initial_batch,
            "action_processing": "checkpoint_unnormalization_only_no_extra_hysteresis",
            "video": str(video) if video.exists() else None}


def execute(args):
    config = json.loads(args.config.read_text())
    expected = json.loads((ROOT / "configs/libero40_public_v1.json").read_text())
    if config != expected or config["suites"] != SUITES:
        raise ValueError("This entrypoint only accepts the frozen v1 protocol")
    if args.phase == "report":
        manifest = json.loads((args.output / "manifest.json").read_text())
        save_report(args.output, manifest)
        return
    status_path = args.output / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "error" and not args.acknowledge_errors:
        raise RuntimeError("Inspect errors first; explicit --acknowledge-errors is required")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU evaluation fallback")
    torch.set_num_threads(4)
    manifest, suites = prepare_manifest(args, config)
    run_id = freeze_manifest(args.output / "manifest.json", manifest)
    from scripts.run_libero_rollout import parse_args as legacy_args, _load_policy
    from lerobot.envs.libero import LiberoEnv

    class NoAutoResetEnv(LiberoEnv):
        def step(self, action):
            # Keep native action/observation processing; omit wrapper's success auto-reset.
            raw, reward, done, info = self._env.step(action)
            success = bool(self._env.check_success())
            return self._format_raw_obs(raw), reward, bool(done or success), False, {**info, "is_success": success}

    load_args = legacy_args(["--checkpoint", str(args.checkpoint), "--flow-steps", "10", "--disable-rtc"])
    loaded = _load_policy(load_args, action_execution_steps=1)
    phases = ["pilot", "formal"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "formal":
            pilots = []
            for name in SUITES:
                key = episode_key("smolvla_official", name, 0, 10, "pilot")
                row = json.loads((args.output / (key + ".json")).read_text())
                validate_episode(row, run_id, key)
                pilots.append(row)
            atomic_json(args.output / "estimate.json", {"method": "pilot total_seconds per suite x remaining episodes x 1.3",
                        "estimated_full_400_seconds": sum(r["total_seconds"] * 100 * 1.3 for r in pilots),
                        "note": "task0-only estimate; other tasks and first-load overhead may differ"})
        for task in manifest["tasks"]:
            if phase == "pilot" and task["task_id"] != 0:
                continue
            env = None
            try:
                for idx in ([10] if phase == "pilot" else range(10)):
                    key = episode_key("smolvla_official", task["suite"], task["task_id"], idx, phase)
                    path = args.output / (key + ".json")
                    if path.exists():
                        validate_episode(json.loads(path.read_text()), run_id, key)
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if env is None:
                        env = NoAutoResetEnv(task_suite=suites[task["suite"]], task_id=task["task_id"],
                                            task_suite_name=task["suite"], episode_length=task["max_steps"],
                                            obs_type="pixels_agent_pos", observation_width=256, observation_height=256,
                                            num_steps_wait=10, init_states=True, n_envs=1, control_mode="relative")
                    atomic_json(args.output / "status.json", {"status": "running", "current": key, "updated_at": time.time()})
                    print(f"START {key}", flush=True)
                    try:
                        result = run_episode(env, loaded, task, idx, phase, run_id,
                                             {k: path.parent / f"first_{k}.mp4" for k in ["success", "failure"]})
                        validate_episode(result, run_id, key)
                        atomic_json(path, result)
                    except Exception:
                        atomic_json(args.output / "errors" / (key + f"-{time.time_ns()}.json"),
                                    {"key": key, "traceback": traceback.format_exc(), "run_id": run_id})
                        raise
                    n = save_report(args.output, manifest)
                    print(f"DONE {key} success={result['success']} steps={result['steps']} formal={n}/400", flush=True)
            finally:
                if env is not None:
                    env.close()
    # Verify that the entire official checkpoint (including processors) was preserved.
    if {p.name: sha256(p) for p in sorted(args.checkpoint.iterdir()) if p.is_file()} != manifest["checkpoint_files"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    n = save_report(args.output, manifest)
    atomic_json(args.output / "status.json", {"status": "complete" if n == 400 else "pilot_complete",
                "formal_complete": n, "act_status": config["act_status"], "updated_at": time.time()})


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args)
        except Exception:
            atomic_json(args.output / "status.json", {"status": "error", "traceback": traceback.format_exc(), "updated_at": time.time()})
            raise


if __name__ == "__main__":
    main()
