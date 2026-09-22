#!/usr/bin/env python
"""Create one comparable table for the task34 policy matrix.

Usage repeats ``--rollout LABEL=PATH`` for LIBERO episodes and optionally
``--offline LABEL=PATH`` for validation/action-error artifacts.  The command
fails on malformed JSON, while protocol mismatches are retained as explicit
diagnostics in the output rather than silently comparing incompatible runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from smolvla_flow.evaluation_protocol import binary_success_summary


def _pair(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("LABEL and PATH must be non-empty")
    return label, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=_pair, action="append", required=True)
    parser.add_argument("--offline", type=_pair, action="append", default=[])
    parser.add_argument("--min-episodes", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/policy_matrix.json"))
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _seed_signature(payload: dict[str, Any]) -> list[tuple[int, int | None]]:
    top_torch_seed = payload.get("torch_seed")
    episodes = payload.get("episodes", [])
    return sorted(
        (
            int(item["seed"]),
            int(item.get("torch_seed", top_torch_seed)) if item.get("torch_seed", top_torch_seed) is not None else None,
        )
        for item in episodes
        if isinstance(item, dict) and "seed" in item
    )


def _latency(episodes: list[dict[str, Any]]) -> dict[str, float | None]:
    # Keep the three summaries separate.  Each episode already contains the
    # policy-only statistics measured around select_action or chunk inference.
    summaries: dict[str, list[float]] = {"mean_seconds": [], "median_seconds": [], "p95_seconds": []}
    for item in episodes:
        inference = item.get("inference")
        if not isinstance(inference, dict):
            continue
        for key in summaries:
            value = inference.get(key)
            if isinstance(value, (int, float)):
                summaries[key].append(float(value))
    return {
        key: (float(statistics.fmean(values)) if values else None)
        for key, values in summaries.items()
    }


def _rollout_summary(label: str, payload: dict[str, Any], min_episodes: int) -> dict[str, Any]:
    episodes = [item for item in payload.get("episodes", []) if isinstance(item, dict)]
    successes = [bool(item.get("success", False)) for item in episodes]
    success = binary_success_summary(successes)
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        aggregate = {}
    hz_values = [
        float(item["effective_control_hz"])
        for item in episodes
        if isinstance(item.get("effective_control_hz"), (int, float))
    ]
    waiting = [
        int(item["waiting_ticks"])
        for item in episodes
        if isinstance(item.get("waiting_ticks"), int)
    ]
    return {
        "label": label,
        "policy_type": payload.get("policy_type"),
        "mode": payload.get("mode"),
        "episode_count": len(episodes),
        "formal_sample_size_ready": len(episodes) >= min_episodes,
        "success": success,
        "policy_only_latency_seconds": _latency(episodes),
        "effective_control_hz": {
            "mean": float(statistics.fmean(hz_values)) if hz_values else None,
            "minimum": min(hz_values) if hz_values else None,
        },
        "waiting_ticks": sum(waiting) if waiting else None,
        "deadline_misses": sum(
            bool(event.get("deadline_miss"))
            for item in episodes
            for event in item.get("inference_events", [])
            if isinstance(event, dict)
        ),
        "evaluation_protocol": payload.get("evaluation_protocol"),
        "runtime_contract": payload.get("runtime_contract"),
        "seed_signature": _seed_signature(payload),
        "source": payload.get("source", "rollout"),
    }


def _offline_summary(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    results = []
    action_error = payload.get("action_error")
    latency = payload.get("latency")
    if isinstance(action_error, dict):
        policy_latency = latency.get("policy_only") if isinstance(latency, dict) else None
        results.append(
            {
                "name": payload.get("policy_type", label),
                "input_modalities": payload.get("input_modalities"),
                "action_mse": action_error.get("action_mse_mean"),
                "action_mae": action_error.get("action_mae_mean"),
                "execute_steps": action_error.get("execute_steps"),
                "policy_only_latency": policy_latency,
                "sample_count": payload.get("sample_count"),
            }
        )
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        validation = item.get("validation")
        if not isinstance(validation, dict):
            continue
        results.append(
            {
                "name": item.get("name"),
                "input_modalities": payload.get("input_modalities"),
                "action_mse": validation.get("executed_action_mse"),
                "action_mae": validation.get("executed_action_mae"),
                "execute_steps": validation.get("execute_steps"),
                "noise_mse": validation.get("noise_mse"),
            }
        )
    return {"label": label, "results": results, "evaluation_protocol": payload.get("evaluation_protocol")}


def main() -> int:
    args = parse_args()
    if args.min_episodes < 1:
        raise ValueError("--min-episodes must be positive")
    rollouts = []
    for label, path in args.rollout:
        rollouts.append(_rollout_summary(label, _read(path), args.min_episodes))
    offline = [_offline_summary(label, _read(path)) for label, path in args.offline]

    protocols = [item["evaluation_protocol"] for item in rollouts if item["evaluation_protocol"] is not None]
    protocol_consistent = bool(protocols) and all(item == protocols[0] for item in protocols[1:])
    signatures = [item["seed_signature"] for item in rollouts if item["seed_signature"]]
    seeds_consistent = bool(signatures) and all(item == signatures[0] for item in signatures[1:])
    result = {
        "schema_version": 1,
        "status": "completed",
        "protocol_consistent": protocol_consistent,
        "seed_consistent": seeds_consistent,
        "protocol": protocols[0] if protocols else None,
        "rollouts": rollouts,
        "offline_validation": offline,
        "diagnostics": {
            "protocol": "pass" if protocol_consistent else "unknown_or_mismatch",
            "seeds": "pass" if seeds_consistent else "unknown_or_mismatch",
            "action_error_scope": "offline validation artifacts only",
            "success_rate_scope": "LIBERO episode terminal is_success without intervention",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
