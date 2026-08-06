#!/usr/bin/env python
"""Build a fixed train/validation/test manifest for one LeRobot task index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smolvla_flow.libero_mapping import normalize_task_text
from smolvla_flow.task_split import count_full_windows, split_episode_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        required=True,
        help="Local LeRobot dataset root containing meta/tasks.parquet and meta/episodes/.",
    )
    parser.add_argument("--task-index", type=int, default=34)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/preflight/libero_spatial_task0_split.json"),
    )
    return parser.parse_args()


def _read_metadata(metadata_root: Path, task_index: int) -> tuple[str, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("pyarrow is required to build the task split manifest") from error

    meta_root = metadata_root / "meta"
    tasks_path = meta_root / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"missing task metadata: {tasks_path}")
    tasks_table = pq.read_table(tasks_path)
    text_column = next(
        (name for name in tasks_table.column_names if name != "task_index"),
        None,
    )
    if text_column is None:
        raise RuntimeError("tasks.parquet has no language column")
    task_rows = tasks_table.to_pylist()
    language_rows = [row for row in task_rows if int(row["task_index"]) == task_index]
    if len(language_rows) != 1:
        raise RuntimeError(f"expected one language row for task_index={task_index}")
    language = str(language_rows[0][text_column])
    target = normalize_task_text(language)

    episode_paths = sorted(
        path
        for path in (meta_root / "episodes").glob("chunk-*/*.parquet")
        if not path.name.startswith(".")
    )
    if not episode_paths:
        raise FileNotFoundError(f"no episode metadata under {meta_root / 'episodes'}")
    selected: list[dict[str, Any]] = []
    for path in episode_paths:
        for row in pq.read_table(path).to_pylist():
            tasks = row.get("tasks") or []
            if any(normalize_task_text(str(task)) == target for task in tasks):
                selected.append(
                    {
                        "episode_index": int(row["episode_index"]),
                        "length": int(row["length"]),
                        "data_chunk_index": int(row.get("data/chunk_index", 0)),
                        "data_file_index": int(row.get("data/file_index", 0)),
                        "dataset_from_index": int(row.get("dataset_from_index", 0)),
                        "dataset_to_index": int(row.get("dataset_to_index", 0)),
                    }
                )
    selected.sort(key=lambda row: row["episode_index"])
    if not selected:
        raise RuntimeError(f"no episodes found for task_index={task_index}")
    return language, selected


def _episode_entry(row: dict[str, Any], *, chunk_size: int, sample_stride: int) -> dict[str, Any]:
    length = int(row["length"])
    entry = dict(row)
    entry["data_shard"] = (
        f"data/chunk-{row['data_chunk_index']:03d}/file-{row['data_file_index']:03d}.parquet"
    )
    entry["complete_chunk_count"] = count_full_windows(
        length,
        chunk_size=chunk_size,
        stride=sample_stride,
    )
    entry["nonoverlap_chunk_count"] = count_full_windows(
        length,
        chunk_size=chunk_size,
        stride=chunk_size,
    )
    return entry


def _summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(entries),
        "episode_indices": [entry["episode_index"] for entry in entries],
        "frame_count": sum(entry["length"] for entry in entries),
        "complete_chunk_count": sum(entry["complete_chunk_count"] for entry in entries),
        "nonoverlap_chunk_count": sum(entry["nonoverlap_chunk_count"] for entry in entries),
        "episodes": entries,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_size < 1 or args.sample_stride < 1:
        raise ValueError("chunk-size and sample-stride must be positive")
    language, raw_entries = _read_metadata(args.metadata_root, args.task_index)
    entries = [
        _episode_entry(row, chunk_size=args.chunk_size, sample_stride=args.sample_stride)
        for row in raw_entries
    ]
    split_ids = split_episode_indices(
        [entry["episode_index"] for entry in entries],
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    by_id = {entry["episode_index"]: entry for entry in entries}
    splits = {
        name: _summarize([by_id[index] for index in ids])
        for name, ids in split_ids.items()
    }
    return {
        "schema_version": 1,
        "dataset": "HuggingFaceVLA/libero",
        "suite": "libero_spatial",
        "suite_task_id": 0,
        "task_index": args.task_index,
        "language": language,
        "metadata_source": "local LeRobot metadata",
        "split": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        },
        "chunking": {
            "chunk_size": args.chunk_size,
            "sample_stride": args.sample_stride,
            "window_definition": "start + chunk_size <= episode length",
        },
        "all": _summarize(entries),
        "splits": splits,
    }


def main() -> None:
    args = parse_args()
    result = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
