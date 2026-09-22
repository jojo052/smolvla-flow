#!/usr/bin/env python3
"""Download a pinned public ACT artifact and audit it on CPU without using CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = "jamongsteak/act_libero"
REVISION = "8744e679f4038386070ff70723edab9bb6228bd3"
ENDPOINT = "https://hf-mirror.com"
FILES = {"README.md", "config.json", "train_config.json", "model.safetensors",
         "policy_preprocessor.json", "policy_postprocessor.json",
         "policy_preprocessor_step_3_normalizer_processor.safetensors",
         "policy_postprocessor_step_0_unnormalizer_processor.safetensors"}


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"repo": REPO, "revision": REVISION, "endpoint": ENDPOINT,
              "status": "downloading", "files": {}, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "hash_trust_boundary": "Digests compared to pinned mirror-reported Hub metadata; origin not independently reachable"}
    write_json(args.output / "audit.json", report)
    try:
        # Mirror's revision API is unavailable. Check the returned main commit
        # before resolving every file by the immutable revision below.
        url = f"{ENDPOINT}/api/models/{REPO}?blobs=true"
        metadata = json.loads(subprocess.check_output(
            ["curl", "-fLsS", "--connect-timeout", "10", "--max-time", "30", url], text=True))
        if metadata["sha"] != REVISION:
            raise ValueError("Metadata revision mismatch")
        write_json(args.output / "hub_metadata.json", metadata)
        siblings = {s["rfilename"]: s for s in metadata["siblings"]}
        for name in sorted(FILES):
            entry = siblings[name]
            target = args.checkpoint / name
            if not target.exists():
                temp = target.with_suffix(target.suffix + ".part")
                subprocess.run(["curl", "-fLsS", "--connect-timeout", "10", "--max-time", "1800",
                                "--speed-time", "60", "--speed-limit", "1", "--retry", "2",
                                "-o", str(temp), f"{ENDPOINT}/{REPO}/resolve/{REVISION}/{name}"], check=True)
                temp.replace(target)
            content = target.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            git_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            lfs = entry.get("lfs")
            expected = lfs.get("sha256", lfs.get("oid")) if lfs else entry.get("blobId")
            actual = sha if lfs else git_sha
            if not expected or actual != expected or len(content) != entry["size"]:
                raise ValueError(f"Hash/size mismatch for {name}: {actual} vs {expected}")
            report["files"][name] = {"sha256": sha, "bytes": len(content), "metadata_digest": expected,
                                      "digest_type": "sha256" if lfs else "git_blob_sha1", "verified": True}
            write_json(args.output / "audit.json", report)
            print(f"VERIFIED {name} bytes={len(content)} sha256={sha}", flush=True)
            del content

        # The process is isolated from CUDA; no change to the running evaluator.
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for the CPU-only audit")
        import torch
        from safetensors.torch import load_file
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies import make_pre_post_processors
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        config = ACTConfig.from_pretrained(args.checkpoint, local_files_only=True)
        report["native_config"] = json.loads((args.checkpoint / "config.json").read_text())
        config.device = "cpu"
        # Full policy contains backbone tensors. Avoid a redundant ImageNet download.
        config.pretrained_backbone_weights = None
        started = time.perf_counter()
        policy = ACTPolicy.from_pretrained(args.checkpoint, config=config, local_files_only=True, strict=True)
        tensors = load_file(str(args.checkpoint / "model.safetensors"), device="cpu")
        incompatible = policy.load_state_dict(tensors, strict=True)
        if any(not torch.isfinite(t).all().item() for t in tensors.values()):
            raise ValueError("Nonfinite model tensor")
        report["strict_load"] = {"passed": True, "seconds": time.perf_counter() - started,
                                 "missing_keys": list(incompatible.missing_keys), "unexpected_keys": list(incompatible.unexpected_keys),
                                 "tensor_count": len(tensors), "parameters": sum(p.numel() for p in policy.parameters()),
                                 "devices": sorted({str(p.device) for p in policy.parameters()}),
                                 "overrides": {"device": "cpu", "pretrained_backbone_weights": None}}
        del tensors
        processor_files = {}
        for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
            value = json.loads((args.checkpoint / filename).read_text())
            for step in value["steps"]:
                if step.get("state_file"):
                    state = load_file(str(args.checkpoint / step["state_file"]), device="cpu")
                    if any(not torch.isfinite(t).all().item() for t in state.values()):
                        raise ValueError("Nonfinite normalization stats")
                    if any((t < 0).any().item() for k, t in state.items() if k.endswith("std")):
                        raise ValueError("Negative normalization std")
                    step["verified_tensor_shapes"] = {k: list(t.shape) for k, t in state.items()}
            processor_files[filename] = value
        report["processors"] = processor_files
        pre, post = make_pre_post_processors(config, args.checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        torch.manual_seed(123)
        raw = {"observation.images.image": torch.rand(1, 3, 256, 256),
               "observation.images.image2": torch.rand(1, 3, 256, 256), "observation.state": torch.zeros(1, 8),
               "task": ["pick up the black bowl and place it on the plate"]}
        batch = pre(raw)
        policy.reset()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
            policy.reset()
            action = policy.select_action(batch)
            final = post(action)
            altered = dict(batch)
            altered["task"] = ["open the drawer"]
            other = policy.predict_action_chunk(altered)
        if tuple(chunk.shape) != (1, 100, 7) or not torch.isfinite(chunk).all() or not torch.isfinite(final).all():
            raise RuntimeError("CPU smoke action contract failed")
        report["cpu_smoke"] = {"chunk_shape": list(chunk.shape), "action_shape": list(final.shape),
                                "all_finite": True, "queue_remaining_after_select": len(policy._action_queue),
                                "language_change_max_abs_diff": float((chunk - other).abs().max()),
                                "note": "Synthetic input smoke only; not a LIBERO success evaluation"}
        training = json.loads((args.checkpoint / "train_config.json").read_text())
        report["training"] = {k: training.get(k) for k in ["dataset", "env", "steps", "batch_size", "seed", "eval", "optimizer", "scheduler", "resume"]}
        report["training_policy_matches_config"] = {k: training.get("policy", {}).get(k) == report["native_config"].get(k)
            for k in ["type", "chunk_size", "n_action_steps", "use_vae", "input_features", "output_features", "normalization_mapping"]}
        report["status"] = "audit_passed_with_scope_caveats"
        report["admission_caveats"] = ["Standard ACT architecture has no language conditioning; multitask goals can be ambiguous",
                                       "Training coverage still requires checking dataset selection against model card",
                                       "CPU smoke is not GPU rollout validation; do not start alongside GPU timing benchmark"]
        write_json(args.output / "audit.json", report)
        print(json.dumps({k: report[k] for k in ["status", "strict_load", "cpu_smoke", "training", "training_policy_matches_config"]}, indent=2), flush=True)
    except Exception:
        import traceback
        report["status"] = "error"
        report["traceback"] = traceback.format_exc()
        write_json(args.output / "audit.json", report)
        raise


if __name__ == "__main__":
    main()
