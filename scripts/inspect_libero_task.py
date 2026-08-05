#!/usr/bin/env python
"""Resolve a LIBERO suite task to LeRobot task indices and episode indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smolvla_flow.libero_mapping import aggregate_action_dimension, match_dataset_task_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--suite-task-id", type=int, default=0)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="Local dataset root containing meta/. Skips Hugging Face access when set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/preflight/libero_spatial_task0.json"),
    )
    return parser.parse_args()


def _read_local_metadata(metadata_root: Path) -> tuple[list[dict], list[dict]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("pyarrow is required with --metadata-root") from error

    meta_dir = metadata_root / "meta"
    tasks_table = pq.read_table(meta_dir / "tasks.parquet")
    task_text_key = next(
        (name for name in tasks_table.column_names if name != "task_index"),
        None,
    )
    if task_text_key is None:
        raise RuntimeError("tasks.parquet has no task text column")
    tasks = [
        {"task_index": int(row["task_index"]), "task": str(row[task_text_key])}
        for row in tasks_table.to_pylist()
    ]

    episode_paths = sorted(
        path
        for path in (meta_dir / "episodes").glob("chunk-*/*.parquet")
        if not path.name.startswith(".")
    )
    if not episode_paths:
        raise RuntimeError(f"no episode metadata parquet found under {meta_dir / 'episodes'}")
    episodes = []
    for path in episode_paths:
        episodes.extend(pq.read_table(path).to_pylist())
    return tasks, episodes


def main() -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from libero.libero import benchmark
    except ImportError as error:
        raise SystemExit(
            "Run this script inside the remote smolvla-flow environment with LeRobot and LIBERO installed"
        ) from error

    args = parse_args()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    suite_task = suite.get_task(args.suite_task_id)
    if args.metadata_root is None:
        metadata = LeRobotDatasetMetadata(args.repo_id)
        dataset_task_index = match_dataset_task_index(suite_task.language, metadata.tasks)
        episodes = metadata.filter_episodes(
            lambda episode: int(episode["task_index"]) == dataset_task_index
        )
        gripper_stats = None
    else:
        tasks, all_episodes = _read_local_metadata(args.metadata_root)
        dataset_task_index = match_dataset_task_index(suite_task.language, tasks)
        normalized_language = " ".join(suite_task.language.lower().replace("_", " ").split()).strip(" .")
        episodes = [
            episode
            for episode in all_episodes
            if any(
                " ".join(str(task).lower().replace("_", " ").split()).strip(" .")
                == normalized_language
                for task in episode["tasks"]
            )
        ]
        gripper_stats = aggregate_action_dimension(episodes, dimension=6)

    result = {
        "repo_id": args.repo_id,
        "suite": args.suite,
        "suite_task_id": args.suite_task_id,
        "language": suite_task.language,
        "dataset_task_index": dataset_task_index,
        "episode_count": len(episodes),
        "episode_indices": [int(episode["episode_index"]) for episode in episodes],
        "episode_lengths": [int(episode["length"]) for episode in episodes]
        if args.metadata_root is not None
        else None,
        "gripper_action_stats": gripper_stats,
        "metadata_source": str(args.metadata_root) if args.metadata_root is not None else "hub",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
