#!/usr/bin/env python
"""Export state/action windows for the ACT and Diffusion Policy smoke runner.

The exporter reads only LeRobot parquet columns.  Images and language fields
are intentionally left out of this first compact baseline artifact.  A later
official LeRobot baseline can reuse the episode split and window manifest from
the resulting metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smolvla_flow.task_split import split_episode_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, action="append", required=True)
    parser.add_argument("--task-index", type=int, default=34)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Existing task split manifest; when omitted, split episode ids deterministically.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/baselines/task34_state_action.pt"),
    )
    return parser.parse_args()


def _normalise_task_index(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"task_index must be an integer, got {value!r}") from error


def _load_rows(paths: list[Path], task_index: int) -> dict[int, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("pyarrow is required to export baseline data") from error

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        table = pq.read_table(path)
        required = {"episode_index", "frame_index", "observation.state", "action", "task_index"}
        missing = sorted(required - set(table.column_names))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in table.to_pylist():
            if _normalise_task_index(row["task_index"]) != task_index:
                continue
            grouped[int(row["episode_index"])].append(row)
    if not grouped:
        raise RuntimeError(f"no rows found for task_index={task_index}")
    return grouped


def _image_columns(paths: list[Path]) -> list[str]:
    """List visual feature columns without materialising video frames."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("pyarrow is required to inspect baseline data") from error
    columns: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        columns.update(
            name for name in pq.read_schema(path).names if name.startswith("observation.images.")
        )
    return sorted(columns)


def _manifest_episode_ids(manifest: Path) -> dict[str, list[int]]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"manifest has no splits object: {manifest}")
    result: dict[str, list[int]] = {}
    for name in ("train", "validation", "test"):
        split = splits.get(name, {})
        if "episode_indices" in split:
            values = split["episode_indices"]
        else:
            values = [entry["episode_index"] for entry in split.get("episodes", [])]
        result[name] = [int(value) for value in values]
    return result


def _window_episode(
    rows: list[dict[str, Any]],
    *,
    chunk_size: int,
    horizon: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    rows = sorted(rows, key=lambda row: int(row["frame_index"]))
    state_windows: list[list[float]] = []
    action_windows: list[list[list[float]]] = []
    starts: list[int] = []
    for start in range(0, len(rows) - horizon + 1, stride):
        window = rows[start : start + horizon]
        state = list(map(float, window[0]["observation.state"]))
        actions = [list(map(float, row["action"])) for row in window]
        if len(state) != 8 or any(len(action) != 7 for action in actions):
            raise ValueError(
                f"expected state_dim=8 and action_dim=7, got state={len(state)}, "
                f"actions={sorted({len(action) for action in actions})}"
            )
        state_windows.append(state)
        action_windows.append(actions)
        starts.append(int(window[0]["frame_index"]))
    if not state_windows:
        return torch.empty((0, 8)), torch.empty((0, horizon, 7)), []
    return torch.tensor(state_windows), torch.tensor(action_windows), starts


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_stride < 1 or args.chunk_size < 1 or args.horizon < args.chunk_size:
        raise ValueError("sample-stride and chunk-size must be positive, horizon must cover chunk-size")
    grouped = _load_rows(args.parquet, args.task_index)
    image_columns = _image_columns(args.parquet)
    episode_ids = sorted(grouped)
    split_ids = (
        _manifest_episode_ids(args.manifest)
        if args.manifest is not None
        else split_episode_indices(episode_ids, seed=args.seed)
    )
    visible_ids = set(episode_ids)
    for name, ids in split_ids.items():
        unknown = sorted(set(ids) - visible_ids)
        if unknown:
            raise ValueError(f"manifest split {name} references missing episodes: {unknown[:5]}")

    split_data: dict[str, dict[str, Any]] = {}
    for name in ("train", "validation", "test"):
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        episode_index: list[int] = []
        frame_index: list[int] = []
        for episode in split_ids[name]:
            state, action, starts = _window_episode(
                grouped[episode],
                chunk_size=args.chunk_size,
                horizon=args.horizon,
                stride=args.sample_stride,
            )
            if state.shape[0] == 0:
                continue
            states.append(state)
            actions.append(action)
            episode_index.extend([episode] * state.shape[0])
            frame_index.extend(starts)
        if states:
            split_data[name] = {
                "state": torch.cat(states, dim=0),
                "actions": torch.cat(actions, dim=0),
                "episode_index": torch.tensor(episode_index, dtype=torch.long),
                "frame_index": torch.tensor(frame_index, dtype=torch.long),
            }
        else:
            split_data[name] = {
                "state": torch.empty((0, 8)),
                "actions": torch.empty((0, args.horizon, 7)),
                "episode_index": torch.empty((0,), dtype=torch.long),
                "frame_index": torch.empty((0,), dtype=torch.long),
            }
    if split_data["train"]["state"].shape[0] < 1 or split_data["validation"]["state"].shape[0] < 1:
        raise RuntimeError("train and validation splits must each contain at least one complete window")
    train_state = split_data["train"]["state"]
    train_actions = split_data["train"]["actions"]

    def stats(value: torch.Tensor) -> dict[str, list[float]]:
        flattened = value.reshape(-1, value.shape[-1]).to(dtype=torch.float64)
        mean = flattened.mean(dim=0)
        std = flattened.std(dim=0, unbiased=False).clamp_min(1e-6)
        return {
            "mean": [float(item) for item in mean.tolist()],
            "std": [float(item) for item in std.tolist()],
        }

    return {
        "schema_version": 1,
        "data_source": "lerobot_parquet_state_action",
        "task_index": args.task_index,
        "episode_ids": episode_ids,
        "data_contract": {
            "state_dim": 8,
            "action_dim": 7,
            "act_chunk_size": args.chunk_size,
            "diffusion_horizon": args.horizon,
            "action_execution_steps": 10,
            "camera_columns": image_columns,
            "camera_columns_available": bool(image_columns),
            "normalization": "raw_dataset_action",
            "note": (
                "This compact artifact contains state/action windows only. "
                "Official visual ACT/Diffusion training must read the same camera "
                "columns through LeRobotDataset and its checkpoint processors."
            ),
        },
        "stats": {
            "state": stats(train_state),
            "action": stats(train_actions),
        },
        "split": {
            "seed": args.seed,
            "sample_stride": args.sample_stride,
            "chunk_size": args.chunk_size,
            "horizon": args.horizon,
            "manifest": str(args.manifest) if args.manifest else None,
            "episode_counts": {name: len(split_ids[name]) for name in split_ids},
        },
        "splits": split_data,
    }


def main() -> int:
    args = parse_args()
    artifact = build_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    summary = {
        key: artifact[key]
        for key in ("schema_version", "data_source", "task_index", "episode_ids", "split")
    }
    summary["sample_counts"] = {
        name: int(value["state"].shape[0]) for name, value in artifact["splits"].items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
