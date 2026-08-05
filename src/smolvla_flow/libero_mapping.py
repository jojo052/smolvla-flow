"""Pure helpers for mapping LIBERO suite tasks to LeRobot metadata."""

from __future__ import annotations

import math
from typing import Any, Iterable


def normalize_task_text(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split()).strip(" .")


def task_rows(tasks: Any) -> Iterable[tuple[int, str]]:
    if isinstance(tasks, dict):
        for index, value in tasks.items():
            if isinstance(value, dict):
                text = value.get("task") or value.get("name") or value.get("language_instruction")
            else:
                text = value
            if text is not None:
                yield int(index), str(text)
        return
    for index, value in enumerate(tasks):
        if isinstance(value, dict):
            task_index = int(value.get("task_index", value.get("index", index)))
            text = value.get("task") or value.get("name") or value.get("language_instruction")
        else:
            task_index = index
            text = value
        if text is not None:
            yield task_index, str(text)


def match_dataset_task_index(suite_language: str, tasks: Any) -> int:
    target = normalize_task_text(suite_language)
    matches = [index for index, text in task_rows(tasks) if normalize_task_text(text) == target]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one dataset task for {suite_language!r}, found indices {matches}"
        )
    return matches[0]


def aggregate_action_dimension(
    episodes: list[dict[str, Any]],
    dimension: int,
) -> dict[str, float | int | bool]:
    """Aggregate LeRobot per-episode action moments for one dimension."""
    records = []
    for episode in episodes:
        count = int(episode["stats/action/count"][0])
        records.append(
            {
                "count": count,
                "min": float(episode["stats/action/min"][dimension]),
                "max": float(episode["stats/action/max"][dimension]),
                "mean": float(episode["stats/action/mean"][dimension]),
                "std": float(episode["stats/action/std"][dimension]),
            }
        )
    total_count = sum(record["count"] for record in records)
    if total_count == 0:
        raise RuntimeError("selected episodes contain no action samples")
    mean = sum(record["count"] * record["mean"] for record in records) / total_count
    variance = sum(
        record["count"]
        * (record["std"] ** 2 + (record["mean"] - mean) ** 2)
        for record in records
    ) / total_count
    minimum = min(record["min"] for record in records)
    maximum = max(record["max"] for record in records)
    return {
        "dimension": dimension,
        "count": total_count,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "observed_both_hysteresis_sides": minimum <= -0.5 and maximum >= 0.5,
    }
