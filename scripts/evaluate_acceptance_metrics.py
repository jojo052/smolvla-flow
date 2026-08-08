#!/usr/bin/env python
"""Aggregate the SmolVLA acceptance metrics from JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smolvla_flow.evaluation import evaluate_acceptance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, action="append", required=True, help="rollout JSON; repeat for sync/async comparisons")
    parser.add_argument("--benchmark", type=Path, action="append", default=[], help="distillation benchmark JSON")
    parser.add_argument("--fused-rollout", type=Path, default=None, help="rollout using overlap/RTC fusion")
    parser.add_argument("--unfused-rollout", type=Path, default=None, help="rollout using the unfused queue baseline")
    parser.add_argument("--target-task-index", type=int, default=34)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/acceptance_metrics.json"))
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def main() -> int:
    args = parse_args()
    rollout_paths = list(dict.fromkeys(args.rollout))
    rollouts = [_read(path) for path in rollout_paths]
    benchmarks = [_read(path) for path in args.benchmark]
    fused = _read(args.fused_rollout) if args.fused_rollout is not None else None
    unfused = _read(args.unfused_rollout) if args.unfused_rollout is not None else None
    result = {
        "rollouts": [str(path) for path in rollout_paths],
        "benchmarks": [str(path) for path in args.benchmark],
        "fused_rollout": str(args.fused_rollout) if args.fused_rollout is not None else None,
        "unfused_rollout": str(args.unfused_rollout) if args.unfused_rollout is not None else None,
        "target_task_index": args.target_task_index,
        "metrics": evaluate_acceptance(
            rollouts,
            benchmarks,
            target_task_index=args.target_task_index,
            fused_rollout=fused,
            unfused_rollout=unfused,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
