#!/usr/bin/env python
"""Progressively distill a real SmolVLA action expert from 10 to 5 to 2 steps.

This script is intentionally a small, single-GPU research runner.  It keeps the
teacher and student in the same normalized action space used by the LeRobot
checkpoint, caches the multimodal prefix once per observation, and applies the
descending-time SmolVLA Euler convention (``t=1`` to ``t=0``).

The script does not modify the original checkpoint.  Each stage writes only the
trainable action-expert parameters and a JSON metrics file under ``--output-dir``.
Run it inside the remote ``smolvla-flow`` environment, where LeRobot, the
checkpoint cache, and the LIBERO parquet file are available.
"""

from __future__ import annotations

import argparse
import io
import inspect
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn


TASK0_LANGUAGE = "pick up the black bowl between the plate and the ramekin and place it on the plate"
SUPPORTED_STAGE_PAIRS = ((10, 5), (5, 2))


@dataclass(frozen=True)
class Sample:
    """One observation and its 50-step demonstrated action chunk."""

    episode_index: int
    frame_index: int
    row: dict[str, Any]
    action_chunk: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--parquet", type=Path, action="append", required=True)
    parser.add_argument("--episode-index", type=int, action="append", default=None)
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Keep only rows whose dataset task_index matches this value",
    )
    parser.add_argument("--task", default=TASK0_LANGUAGE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps-per-stage", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=8, help="0 means all selected samples")
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/distillation/task0"))
    return parser.parse_args()


def _image_to_tensor(value: Any):
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


def _load_samples(
    parquet_paths: list[Path],
    episode_indices: list[int] | None,
    stride: int,
    task_index: int | None = None,
    filter_stats: dict[str, Any] | None = None,
) -> list[Sample]:
    import pyarrow.parquet as pq

    if stride < 1:
        raise ValueError("sample-stride must be at least 1")
    selected = set(episode_indices) if episode_indices else None
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def _normalized_task_index(value: Any) -> int | str:
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            episode_index = int(row["episode_index"])
            if selected is None or episode_index in selected:
                grouped[episode_index].append(row)

    grouped_before_filter = {episode_index: list(rows) for episode_index, rows in grouped.items()}
    visible_task_indices: set[int | str] = set()
    if task_index is not None:
        episode_task_indices: dict[int, set[int | str]] = defaultdict(set)
        for episode_index, rows in grouped.items():
            for row in rows:
                raw_task_index = row.get("task_index")
                if raw_task_index is None:
                    visible_value: int | str = "<missing>"
                else:
                    visible_value = _normalized_task_index(raw_task_index)
                visible_task_indices.add(visible_value)
                episode_task_indices[episode_index].add(visible_value)
        conflicts = {
            episode_index: sorted(values, key=str)
            for episode_index, values in episode_task_indices.items()
            if len(values) > 1
        }
        if conflicts:
            raise ValueError(f"multiple task_index values found in episode(s): {conflicts}")
        grouped = {
            episode_index: [
                row
                for row in rows
                if row.get("task_index") is not None
                and _normalized_task_index(row["task_index"]) == task_index
            ]
            for episode_index, rows in grouped.items()
        }

    def build_samples(grouped_rows: dict[int, list[dict[str, Any]]]) -> list[Sample]:
        samples: list[Sample] = []
        for episode_index in sorted(grouped_rows):
            rows = sorted(grouped_rows[episode_index], key=lambda item: int(item["frame_index"]))
            for start in range(0, len(rows), stride):
                row = rows[start]
                chunk_rows = rows[start : start + 50]
                # Do not fabricate supervision for the final 49 frames of an
                # episode.  LeRobot masks those action_is_pad positions during
                # training; this first distillation runner keeps only full chunks.
                if len(chunk_rows) != 50:
                    continue
                action_chunk = [list(map(float, item["action"])) for item in chunk_rows]
                action_dim = len(action_chunk[0])
                if action_dim != 7 or any(len(action) != action_dim for action in action_chunk):
                    raise ValueError(f"expected 7D action chunks, got episode={episode_index}, frame={row['frame_index']}")
                samples.append(
                    Sample(
                        episode_index=episode_index,
                        frame_index=int(row["frame_index"]),
                        row=row,
                        action_chunk=action_chunk,
                    )
                )
        return samples

    samples = build_samples(grouped)
    if filter_stats is not None:
        filter_stats.update(
            {
                "task_index_filter": task_index,
                "sample_count_before_task_index_filter": len(
                    build_samples(grouped_before_filter)
                )
                if task_index is not None
                else len(samples),
                "sample_count_after_task_index_filter": len(samples),
            }
        )
    if not samples:
        wanted = "all episodes" if selected is None else sorted(selected)
        if task_index is not None:
            visible_values = sorted(visible_task_indices, key=str)
            raise RuntimeError(
                f"no usable samples found for task-index={task_index} in {parquet_paths} for {wanted}; "
                f"visible task_index values={visible_values}"
            )
        raise RuntimeError(f"no usable samples found in {parquet_paths} for {wanted}")
    return samples


def _load_policy(checkpoint_path: Path, checkpoint_name: str, flow_steps: int, device: str):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig.from_pretrained(checkpoint_path, local_files_only=True)
    config.device = device
    # The policy checkpoint contains the SmolVLA weights.  Do not separately
    # fetch or initialize the base VLM from Hub during this experiment.
    config.load_vlm_weights = False
    config.compile_model = False
    if not config.use_cache:
        raise ValueError("the direct denoise-step distillation adapter requires config.use_cache=True")
    config.num_steps = flow_steps
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.config.num_steps = flow_steps
    policy.model.config.num_steps = flow_steps
    chunk_size = int(policy.config.chunk_size)
    max_action_dim = int(policy.config.max_action_dim)
    action_dim = int(policy.config.action_feature.shape[0])
    if (chunk_size, max_action_dim, action_dim) != (50, 32, 7):
        raise ValueError(
            "this first LIBERO runner expects chunk_size=50, max_action_dim=32, action_dim=7; "
            f"got {(chunk_size, max_action_dim, action_dim)}"
        )
    preprocessor, _ = make_pre_post_processors(
        config,
        checkpoint_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor


def _freeze_teacher(policy) -> None:
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)


def _configure_action_only_student(policy) -> list[str]:
    """Freeze VLM/state projection and expose only the action expert."""

    model = policy.model
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    trainable_modules = [
        model.action_in_proj,
        model.action_out_proj,
        model.action_time_mlp_in,
        model.action_time_mlp_out,
    ]
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    # SmolVLA's transformer action expert is the language-model expert branch.
    # The lm_head is deliberately excluded because it is not used for actions.
    for name, parameter in model.vlm_with_expert.lm_expert.named_parameters():
        if "lm_head" not in name:
            parameter.requires_grad_(True)

    policy.train()
    # SmolVLMWithExpertModel.train() keeps the frozen VLM in eval mode.
    model.vlm_with_expert.vlm.eval()
    return [name for name, parameter in policy.named_parameters() if parameter.requires_grad]


def _prepare_sample(sample: Sample, task: str, preprocessor, policy):
    import torch

    row = sample.row
    raw = {
        "observation.images.image": _image_to_tensor(row["observation.images.image"]),
        "observation.images.image2": _image_to_tensor(row["observation.images.image2"]),
        "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
        # Add the batch dimension explicitly.  LeRobot only adds one for a 1D
        # action, while a [50, 7] tensor would otherwise be interpreted as B=50.
        "action": torch.tensor(sample.action_chunk, dtype=torch.float32).unsqueeze(0),
        "task": task,
    }
    processed = preprocessor(raw)
    target_actions = policy.prepare_action(processed).detach()
    if target_actions.ndim != 3 or tuple(target_actions.shape[1:]) != (50, 32):
        raise RuntimeError(f"expected padded normalized target [1, 50, 32], got {tuple(target_actions.shape)}")
    return processed, target_actions


class CachedSmolVLAVelocity(nn.Module):
    """Velocity-model adapter backed by one cached multimodal prefix."""

    def __init__(self, policy, processed_batch):
        import torch

        super().__init__()
        self.model = policy.model
        images, image_masks = policy.prepare_images(processed_batch)
        state = policy.prepare_state(processed_batch)
        language_tokens = processed_batch["observation.language.tokens"]
        language_masks = processed_batch["observation.language.attention_mask"]
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images,
                image_masks,
                language_tokens,
                language_masks,
                state=state,
            )
            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = self.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                **(
                    {"fill_kv_cache": True}
                    if "fill_kv_cache"
                    in inspect.signature(self.model.vlm_with_expert.forward).parameters
                    else {}
                ),
            )
        self.prefix_pad_masks = prefix_pad_masks
        self.past_key_values = past_key_values

    def forward(self, noisy_actions, time, context_tokens):
        del context_tokens
        return self.model.denoise_step(
            x_t=noisy_actions,
            prefix_pad_masks=self.prefix_pad_masks,
            past_key_values=self.past_key_values,
            timestep=time.squeeze(-1),
        )


def _train_stage(
    teacher_policy,
    teacher_preprocessor,
    student_policy,
    student_preprocessor,
    samples: list[Sample],
    *,
    task: str,
    teacher_steps: int,
    student_steps: int,
    steps_per_stage: int,
    learning_rate: float,
    seed: int,
):
    import torch
    from smolvla_flow.distillation import DistillationConfig, train_distillation_step

    if (teacher_steps, student_steps) not in SUPPORTED_STAGE_PAIRS:
        raise ValueError(f"unsupported stage {teacher_steps}->{student_steps}")
    if steps_per_stage < 1:
        raise ValueError("steps-per-stage must be at least 1")
    trainable_names = _configure_action_only_student(student_policy)
    _freeze_teacher(teacher_policy)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student_policy.parameters() if parameter.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    config = DistillationConfig(
        teacher_steps=teacher_steps,
        student_steps=student_steps,
        action_dim=7,
        trajectory_consistency_weight=1.0,
        action_regression_weight=1.0,
    )
    device = next(student_policy.parameters()).device
    context_tokens = torch.zeros(1, 1, 1, device=device, dtype=torch.float32)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    probe_name, probe_parameter = next(
        (name, parameter)
        for name, parameter in student_policy.named_parameters()
        if parameter.requires_grad and name.endswith("action_in_proj.weight")
    )
    probe_before = probe_parameter.detach().clone()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
    started = time.perf_counter()
    for step in range(steps_per_stage):
        sample = samples[rng.randrange(len(samples))]
        processed_teacher, target_actions = _prepare_sample(sample, task, teacher_preprocessor, teacher_policy)
        # Both policies use the same normalized observation.  The preprocessor
        # state is detached before the cached prefix to keep the action-only
        # stage free of state-projection gradients.
        processed_student, student_target_actions = _prepare_sample(
            sample, task, student_preprocessor, student_policy
        )
        if not torch.allclose(target_actions, student_target_actions, atol=0.0, rtol=0.0):
            raise RuntimeError("teacher and student action normalization disagree")
        teacher_velocity = CachedSmolVLAVelocity(teacher_policy, processed_teacher)
        student_velocity = CachedSmolVLAVelocity(student_policy, processed_student)
        noise = torch.randn((1, 50, 32), device=device, dtype=torch.float32)
        output = train_distillation_step(
            student_velocity,
            teacher_velocity,
            noise,
            context_tokens,
            optimizer=optimizer,
            config=config,
            target_actions=student_target_actions,
        )
        record = {
            "step": step + 1,
            "episode_index": sample.episode_index,
            "frame_index": sample.frame_index,
            "loss": float(output.loss.detach().cpu()),
            "trajectory_consistency_loss": float(output.trajectory_consistency_loss.detach().cpu()),
            "action_regression_loss": float(output.action_regression_loss.detach().cpu()),
            "finite": bool(torch.isfinite(output.loss).item()),
        }
        if step == 0:
            record["probe_parameter_name"] = probe_name
            record["probe_parameter_delta_mean"] = float(
                (probe_parameter.detach() - probe_before).abs().mean().cpu()
            )
        records.append(record)
        print(json.dumps({"stage": f"{teacher_steps}_to_{student_steps}", **record}), flush=True)
    elapsed = time.perf_counter() - started
    return {
        "teacher_steps": teacher_steps,
        "student_steps": student_steps,
        "steps": steps_per_stage,
        "learning_rate": learning_rate,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in student_policy.parameters() if parameter.requires_grad)),
        "trainable_parameter_names": trainable_names,
        "elapsed_seconds": elapsed,
        "mean_step_seconds": elapsed / steps_per_stage,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device=device)) if torch.cuda.is_available() else None,
        "records": records,
    }


def _save_trainable_state(policy, path: Path) -> None:
    import torch

    state = {
        name: parameter.detach().cpu()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download

    args = parse_args()
    if args.steps_per_stage < 1:
        raise SystemExit("--steps-per-stage must be at least 1")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required for this real SmolVLA runner")
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    sample_filter_stats: dict[str, Any] = {}
    samples = _load_samples(
        args.parquet,
        args.episode_index,
        args.sample_stride,
        task_index=args.task_index,
        filter_stats=sample_filter_stats,
    )
    if args.max_samples:
        samples = samples[: args.max_samples]
    checkpoint_config = Path(
        hf_hub_download(
            repo_id=args.checkpoint,
            filename="config.json",
            local_files_only=True,
        )
    )
    checkpoint_path = checkpoint_config.parent
    teacher10, teacher_preprocessor = _load_policy(checkpoint_path, args.checkpoint, 10, args.device)
    student5, student5_preprocessor = _load_policy(checkpoint_path, args.checkpoint, 5, args.device)

    stage1 = _train_stage(
        teacher10,
        teacher_preprocessor,
        student5,
        student5_preprocessor,
        samples,
        task=args.task,
        teacher_steps=10,
        student_steps=5,
        steps_per_stage=args.steps_per_stage,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_trainable_state(student5, args.output_dir / "student_5_action_expert.pt")
    del teacher10, teacher_preprocessor
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    student2, student2_preprocessor = _load_policy(checkpoint_path, args.checkpoint, 2, args.device)
    student2.model.load_state_dict(student5.model.state_dict(), strict=True)
    stage2 = _train_stage(
        student5,
        student5_preprocessor,
        student2,
        student2_preprocessor,
        samples,
        task=args.task,
        teacher_steps=5,
        student_steps=2,
        steps_per_stage=args.steps_per_stage,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
    )
    _save_trainable_state(student2, args.output_dir / "student_2_action_expert.pt")

    result = {
        "status": "completed",
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "parquet": [str(path) for path in args.parquet],
        "episode_indices": sorted({sample.episode_index for sample in samples}),
        "sample_count": len(samples),
        **sample_filter_stats,
        "task": args.task,
        "action_dim": 7,
        "max_action_dim": 32,
        "distillation_loss_action_dim": 7,
        "chunk_size": 50,
        "time_direction": "descending_t_1_to_0",
        "action_space": "LeRobot normalized MEAN_STD",
        "semantic_loss_enabled": False,
        "teacher_frozen": True,
        "seed": args.seed,
        "steps_per_stage": args.steps_per_stage,
        "stages": [stage1, stage2],
    }
    output_path = args.output_dir / "distillation_metrics.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
