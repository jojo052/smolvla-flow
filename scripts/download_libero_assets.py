#!/usr/bin/env python
"""Download the official LeRobot LIBERO simulation assets on a networked host."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_DIRS = (
    "scenes",
    "articulated_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="directory that will contain the asset subdirectories")
    parser.add_argument("--repo-id", default="lerobot/libero-assets")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    from huggingface_hub import snapshot_download

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output),
        max_workers=args.workers,
    )
    missing = [name for name in REQUIRED_DIRS if not (output / name).is_dir()]
    if missing:
        raise RuntimeError(f"download completed but required directories are missing: {', '.join(missing)}")
    print(f"LIBERO assets are ready at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
