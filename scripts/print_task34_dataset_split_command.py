#!/usr/bin/env python
"""Print the LeRobot command that materialises the locked task34 split."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/preflight/libero_spatial_task0_split.json"))
    parser.add_argument("--repo-id", required=True, help="source LeRobot dataset repo id")
    parser.add_argument("--new-repo-id", required=True, help="destination prefix for train/validation/test repos")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("manifest has no splits object")
    episode_splits: dict[str, list[int]] = {}
    for name in ("train", "validation", "test"):
        value = splits.get(name)
        if not isinstance(value, dict) or not isinstance(value.get("episode_indices"), list):
            raise ValueError(f"manifest split {name} has no episode_indices")
        episode_splits[name] = [int(item) for item in value["episode_indices"]]
    splits_json = json.dumps(episode_splits, separators=(",", ":"))
    command = [
        "lerobot-edit-dataset",
        "--repo_id",
        args.repo_id,
        "--new_repo_id",
        args.new_repo_id,
        "--operation.type",
        "split",
        "--operation.splits",
        splits_json,
    ]
    print(" ".join(shlex.quote(item) for item in command))
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "source_repo_id": args.repo_id,
                "destination_repo_prefix": args.new_repo_id,
                "episode_counts": {name: len(items) for name, items in episode_splits.items()},
                "episode_indices": episode_splits,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
