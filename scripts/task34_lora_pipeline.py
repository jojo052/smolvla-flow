#!/usr/bin/env python3
"""Isolated task34 LoRA experiment with validation selection and locked rollout.

Run on the existing AutoDL LeRobot environment. Existing base weights and
historical rollout artifacts are read-only inputs. No automatic shutdown.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src")]
BASE = Path("/root/autodl-tmp/checkpoints/smolvla_libero")
DATA = Path("/root/autodl-tmp/datasets/libero_task34_v2")
EXPECTED_BASE_SHA = "71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def base_manifest():
    values = {p.name: digest(p) for p in BASE.iterdir() if p.is_file()}
    assert values["model.safetensors"] == EXPECTED_BASE_SHA, "Base differs from formal baseline"
    return values


def status(root, stage, **details):
    value = {"stage": stage, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}
    write(root / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def call(root, name, args):
    print("RUN", name, args, flush=True)
    with (root / (name + ".log")).open("x") as out:
        subprocess.run(args, cwd=PROJECT, stdout=out, stderr=subprocess.STDOUT, check=True)


def merge(adapter, target):
    import torch
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    from peft import PeftModel
    from safetensors import safe_open
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from scripts.evaluate_lerobot_checkpoint import _sample_batch

    assert not target.exists(), f"Refusing existing merged directory: {target}"
    ac = read(adapter / "adapter_config.json")
    assert Path(ac["base_model_name_or_path"]).resolve() == BASE.resolve()
    config = SmolVLAConfig.from_pretrained(str(adapter), local_files_only=True)
    config.device = "cuda"
    config.load_vlm_weights = False
    config.compile_model = False
    config.use_peft = False
    config.num_steps = 10
    config.n_action_steps = 10
    policy = SmolVLAPolicy.from_pretrained(str(BASE), config=config, local_files_only=True, strict=True)
    # Prove all serialized base tensors were restored before applying LoRA.
    state = policy.state_dict()
    with safe_open(BASE / "model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            assert torch.equal(state[key].cpu(), f.get_tensor(key)), f"Base load mismatch: {key}"
    del state
    original_dtypes = {n: p.dtype for n, p in policy.named_parameters()}
    buffer_dtypes = {n: p.dtype for n, p in policy.named_buffers()}
    wrapped = PeftModel.from_pretrained(policy, str(adapter), is_trainable=False).eval()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    lora_state = [p for n, p in wrapped.named_parameters() if "lora_B" in n]
    assert lora_state and any(bool(torch.count_nonzero(p)) for p in lora_state), "LoRA has no learned updates"
    pre, _ = make_pre_post_processors(config, str(adapter), preprocessor_overrides={"device_processor": {"device": "cuda"}})
    dataset = LeRobotDataset("local/libero_task34_validation", root=DATA / "validation")
    keys = tuple(k for k in config.input_features if k.startswith("observation.images."))
    batch = pre(_sample_batch(dataset[0], keys))
    print("Parameter dtypes", sorted({str(p.dtype) for p in wrapped.parameters()}), flush=True)
    # Verify merge algebra in FP32, then export in the original layer dtypes.
    # BF16 W + delta rounding need not reproduce separately evaluated LoRA exactly.
    with torch.inference_mode():
        torch.manual_seed(123)
        wrapped.reset()
        unmerged_native = wrapped.predict_action_chunk(copy.deepcopy(batch)).clone()
        wrapped.float()
        torch.manual_seed(123)
        wrapped.reset()
        original = wrapped.predict_action_chunk(copy.deepcopy(batch)).clone()
        merged = wrapped.merge_and_unload(safe_merge=True).eval()
        torch.manual_seed(123)
        merged.reset()
        fused_fp32 = merged.predict_action_chunk(copy.deepcopy(batch)).clone()
        torch.testing.assert_close(original, fused_fp32, rtol=1e-3, atol=1e-3)
        for name, param in merged.named_parameters():
            param.data = param.data.to(dtype=original_dtypes[name])
        for name, buffer in merged.named_buffers():
            buffer.data = buffer.data.to(dtype=buffer_dtypes[name])
        torch.manual_seed(123)
        merged.reset()
        fused = merged.predict_action_chunk(copy.deepcopy(batch)).clone()
    assert original.shape == (1, 50, 7) and torch.isfinite(original).all()
    assert torch.isfinite(fused).all()
    merged.config.use_peft = False
    merged.config.pretrained_path = str(target)
    merged.save_pretrained(target)
    for p in adapter.iterdir():
        if p.name.startswith("policy_preprocessor") or p.name.startswith("policy_postprocessor"):
            shutil.copy2(p, target / p.name)
    # Reload through the SAME strict loader that the formal rollout uses.
    from scripts.run_libero_rollout import _load_policy, parse_args
    del wrapped, merged, policy
    torch.cuda.empty_cache()
    loaded = _load_policy(parse_args(["--checkpoint", str(target), "--flow-steps", "10", "--disable-rtc"]))
    loaded.policy.eval()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        torch.manual_seed(123)
        reloaded = loaded.policy.predict_action_chunk(copy.deepcopy(batch))
    torch.testing.assert_close(fused, reloaded, rtol=1e-5, atol=1e-5)
    write(target / "merge_verification.json", {
        "base_sha256": digest(BASE / "model.safetensors"),
        "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
        "merged_sha256": digest(target / "model.safetensors"),
        "base_tensors_exact": True, "strict_reload": True, "finite": True,
        "shape": list(original.shape),
        "fp32_merge_max_abs_difference": float((original - fused_fp32).abs().max()),
        "native_dtype_merge_max_abs_difference": float((unmerged_native - fused).abs().max()),
        "reload_max_abs_difference": float((fused - reloaded).abs().max()),
        "source_adapter": str(adapter), "source_training_config": read(adapter / "train_config.json"),
    })


def validate(checkpoint, output, max_samples):
    from argparse import Namespace
    from smolvla_flow.async_runtime import seed_policy_rng
    from scripts.evaluate_lerobot_checkpoint import evaluate
    seed_policy_rng(123)
    result = evaluate(Namespace(policy_type="smolvla", checkpoint=str(checkpoint), adapter=None,
        dataset_repo_id="local/libero_task34_validation", dataset_root=DATA / "validation",
        device="cuda", task_index=34, sample_stride=4, max_samples=max_samples, warmup=3))
    assert result["finite"] and math.isfinite(result["action_error"]["action_mse_mean"])
    result["torch_seed"] = 123
    write(output, result)


def report(root):
    from smolvla_flow.evaluation_protocol import binary_success_summary
    rollout = read(root / "rollout_30.json")
    episodes = rollout["episodes"]
    assert rollout["status"] == "completed" and len(episodes) == 30
    assert sorted(e["seed"] for e in episodes) == list(range(30))
    assert rollout["flow_steps"] == 10 and rollout["policy_n_action_steps"] == 10
    assert rollout["mode"] == "sync" and rollout["dataset_task_index"] == 34
    assert all(e.get("torch_seed", rollout.get("torch_seed")) == 123 for e in episodes)
    assert all(e["action_nonfinite_count"] == 0 for e in episodes)
    n = sum(bool(e["success"]) for e in episodes)
    z = 1.959963984540054
    p, den = n / 30, 1 + z*z / 30
    center = (p + z*z / 60) / den
    radius = z * math.sqrt(p*(1-p)/30 + z*z/3600) / den
    row = {"success_count": n, "episode_count": 30, "success_rate": p,
        "wilson_95": [center-radius, center+radius],
        "mean_wall_throughput_hz": statistics.fmean(e["wall_throughput_hz"] for e in episodes),
        "mixed_select_action_call_mean_seconds": sum(e["inference"]["mean_seconds"] * e["inference"]["count"] for e in episodes) / sum(e["inference"]["count"] for e in episodes),
        "failure_seeds": [e["seed"] for e in episodes if not e["success"]],
        "success_statistics": binary_success_summary([bool(e["success"]) for e in episodes])}
    old = read(PROJECT / "artifacts/rollout/formal_seed123_30_execution10/comparison_summary.json")
    rows = [("ACT 050000", old["variants"]["act_050000"]),
            ("Diffusion 080000", old["variants"]["diffusion_080000"]),
            ("原生 SmolVLA，10 steps", old["variants"]["smolvla_native_10step"]),
            ("task34 LoRA SmolVLA，10 steps", row)]
    lines = ["# 正式结果", "", "| 策略 | 成功率 | Wilson 95% 区间 | 完整 rollout 吞吐 | 混合 select_action 平均耗时 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, v in rows:
        lo, hi = v["wilson_95"]
        lines.append(f'| {name} | {v["success_count"]}/30，{v["success_rate"]:.2%} | {lo:.2%} 到 {hi:.2%} | {v["mean_wall_throughput_hz"]:.2f} tick/s | {v["mixed_select_action_call_mean_seconds"]*1000:.2f} ms |')
    lines += ["", "前三行为已存档历史实验；LoRA 为本次运行。统一环境 seeds 0..29、策略 seed 123、同步执行、前缀10、上限280步、RTC关闭。",
              "混合 select_action 包含完整动作生成与缓存弹出。验证集挑选 checkpoint，正式测试集不参与挑选。",
              "", "选模记录：", "```json", json.dumps(read(root / "selection.json"), ensure_ascii=False, indent=2), "```", ""]
    (root / "results.md").write_text("\n".join(lines))
    write(root / "results.json", {"lora": row, "rows": dict(rows), "selection": read(root / "selection.json")})


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    for key, value in {"HF_HOME": "/root/autodl-tmp/huggingface-cache", "TORCH_HOME": "/root/autodl-tmp/torch-cache",
        "TMPDIR": "/root/autodl-tmp/tmp", "PIP_CACHE_DIR": "/root/autodl-tmp/pip-cache", "HF_HUB_OFFLINE": "1",
        "WANDB_MODE": "disabled", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}.items():
        os.environ[key] = value
    os.environ["PYTHONPATH"] = str(PROJECT / "src") + ":" + str(PROJECT)
    before = base_manifest()
    train_dir = root / "train"
    command = [str(Path(sys.executable).with_name("lerobot-train")), f"--policy.path={BASE}",
        "--policy.push_to_hub=false", "--policy.load_vlm_weights=false", "--policy.device=cuda",
        "--policy.use_amp=true", "--policy.n_action_steps=10", "--policy.num_steps=10",
        "--dataset.repo_id=local/libero_task34_train", f"--dataset.root={DATA / 'train'}", f"--output_dir={train_dir}",
        "--steps=5000", "--batch_size=8", "--num_workers=8", "--seed=123", "--save_checkpoint=true",
        "--save_freq=1000", "--log_freq=100", "--wandb.enable=false",
        "--peft.method_type=LORA", "--peft.r=32", "--peft.lora_alpha=32", "--policy.optimizer_lr=0.0001",
        "--policy.scheduler_warmup_steps=200", "--policy.scheduler_decay_steps=5000",
        "--policy.scheduler_decay_lr=0.00001", "--job_name=task34_smolvla_lora_v1"]
    manifest = {"base_files_sha256": before, "command": command, "validation_selection": "lowest held-out raw action MSE, stride 4, three warmup windows excluded, fixed RNG seed 123, then earliest step",
        "training_seed": 123, "steps": 5000, "rank": 32, "alpha": 32,
        "script_sha256": digest(__file__),
        "evaluation_inputs_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in [PROJECT / "scripts/run_libero_rollout.py", PROJECT / "scripts/evaluate_lerobot_checkpoint.py", PROJECT / "configs/evaluation_protocol.toml"]},
        "dataset_metadata_sha256": {split: digest(DATA / split / "meta/info.json") for split in ("train", "validation", "test")},
        "export_precision": "FP32 merge algebra verified, then original parameter dtypes restored; BF16 rounding difference recorded",
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "lerobot", "transformers", "peft", "accelerate")},
        "protocol": read(PROJECT / "artifacts/rollout/formal_seed123_30_execution10/comparison_summary.json")["protocol"] if "protocol" in read(PROJECT / "artifacts/rollout/formal_seed123_30_execution10/comparison_summary.json") else str(PROJECT / "configs/evaluation_protocol.toml")}
    write(root / "manifest.json", manifest)
    try:
        status(root, "training", target_steps=5000)
        call(root, "training", command)
        candidates = []
        for step in (1000, 2000, 3000, 4000, 5000):
            adapter = train_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
            assert read(adapter.parent / "training_state/training_step.json")["step"] == step
            target = root / "merged" / f"{step:06d}"
            status(root, "merging_and_validating", step=step)
            call(root, f"merge_{step:06d}", [sys.executable, __file__, "merge", "--adapter", str(adapter), "--target", str(target)])
            validation = root / f"validation_{step:06d}.json"
            call(root, f"validation_{step:06d}", [sys.executable, __file__, "validate", "--checkpoint", str(target), "--output", str(validation)])
            result = read(validation)
            candidates.append({"step": step, "checkpoint": str(target), "mse": result["action_error"]["action_mse_mean"], "sample_count": result["sample_count"]})
        best = min(candidates, key=lambda x: (x["mse"], x["step"]))
        write(root / "selection.json", {"rule": manifest["validation_selection"], "candidates": candidates, "selected": best})
        status(root, "formal_rollout", selected_step=best["step"])
        call(root, "rollout_30", [sys.executable, str(PROJECT / "scripts/run_libero_rollout.py"),
            "--policy-type", "smolvla", "--checkpoint", best["checkpoint"], "--flow-steps", "10", "--mode", "sync",
            "--disable-rtc", "--suite", "libero_spatial", "--task-id", "0", "--dataset-task-index", "34",
            "--protocol", str(PROJECT / "configs/evaluation_protocol.toml"), "--episodes", "30", "--start-seed", "0",
            "--torch-seed", "123", "--max-steps", "280", "--assets-dir", "/root/autodl-tmp/libero-assets",
            "--gripper-polarity", "positive_open", "--output", str(root / "rollout_30.json")])
        report(root)
        assert base_manifest() == before, "Official checkpoint contents changed"
        status(root, "completed", selected_step=best["step"], report=str(root / "results.md"), official_checkpoint_unchanged=True)
    except BaseException as e:
        status(root, "failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(),
               official_checkpoint_unchanged=base_manifest() == before)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["run", "merge", "validate", "report"])
    p.add_argument("--run-root", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--target", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--max-samples", type=int, default=0)
    a = p.parse_args()
    if a.operation == "run":
        run(a.run_root)
    elif a.operation == "merge":
        merge(a.adapter, a.target)
    elif a.operation == "validate":
        validate(a.checkpoint, a.output, a.max_samples)
    else:
        report(a.run_root)


if __name__ == "__main__":
    main()
