#!/usr/bin/env python
"""Validate the locked experiment choices without importing LeRobot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smolvla_flow.experiment_config import load_experiment_config, unresolved_preflight_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/libero_spatial_task0.toml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    result = {
        "config": str(args.config),
        "experiment": config["experiment"]["name"],
        "valid": True,
        "unresolved_preflight_items": unresolved_preflight_items(config),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
