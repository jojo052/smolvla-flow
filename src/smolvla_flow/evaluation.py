"""Acceptance-metric aggregation for SmolVLA rollout artifacts.

The evaluator deliberately distinguishes ``pass``, ``fail`` and ``unknown``.
An old artifact that predates a metric field must not be treated as a passing
result just because the runtime completed successfully.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _metric(status: str, *, value: Any = None, detail: str | None = None, target: Any = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "value": value}
    if target is not None:
        result["target"] = target
    if detail is not None:
        result["detail"] = detail
    return result


def _episodes(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        return []
    return [episode for episode in episodes if isinstance(episode, Mapping)]


def _mode(payload: Mapping[str, Any]) -> str | None:
    mode = payload.get("mode")
    return str(mode) if mode in {"sync", "async"} else None


def _success_rate(payload: Mapping[str, Any]) -> float | None:
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, Mapping) and isinstance(aggregate.get("success_rate"), (int, float)):
        return float(aggregate["success_rate"])
    episodes = _episodes(payload)
    if not episodes or any("success" not in episode for episode in episodes):
        return None
    return sum(bool(episode["success"]) for episode in episodes) / len(episodes)


def _finite_metric(
    rollouts: Sequence[Mapping[str, Any]],
    benchmarks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ratios: list[float] = []
    missing = 0
    for rollout in rollouts:
        for episode in _episodes(rollout):
            ratio = episode.get("action_finite_ratio")
            if isinstance(ratio, (int, float)):
                ratios.append(float(ratio))
            else:
                missing += 1
    benchmark_outputs: list[bool] = []
    for benchmark in benchmarks:
        for role in ("teacher", "student"):
            result = benchmark.get(role)
            if isinstance(result, Mapping) and isinstance(result.get("output_finite"), bool):
                benchmark_outputs.append(bool(result["output_finite"]))
    if missing or not ratios or (benchmarks and not benchmark_outputs):
        return _metric(
            "unknown",
            value={"rollout_min_ratio": min(ratios) if ratios else None, "benchmark_outputs": benchmark_outputs},
            detail="at least one artifact lacks action_finite_ratio or benchmark output_finite",
            target=1.0,
        )
    passed = min(ratios) >= 1.0 and all(benchmark_outputs)
    return _metric(
        "pass" if passed else "fail",
        value={"rollout_min_ratio": min(ratios), "benchmark_outputs": benchmark_outputs},
        target=1.0,
    )


def _task_isolation_metric(
    rollouts: Sequence[Mapping[str, Any]],
    distillation_metrics: Sequence[Mapping[str, Any]],
    target_task_index: int,
) -> dict[str, Any]:
    observed: list[int] = []
    missing_rollout = 0
    for rollout in rollouts:
        top_level = rollout.get("dataset_task_index")
        for episode in _episodes(rollout):
            value = episode.get("dataset_task_index", top_level)
            if isinstance(value, int) and not isinstance(value, bool):
                observed.append(value)
            else:
                missing_rollout += 1

    distillation_evidence: list[dict[str, int]] = []
    missing_distillation = 0
    for metrics in distillation_metrics:
        task_index_filter = metrics.get("task_index_filter")
        sample_count_after_filter = metrics.get("sample_count_after_task_index_filter")
        sample_count = metrics.get("sample_count")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (task_index_filter, sample_count_after_filter, sample_count)
        ):
            missing_distillation += 1
            continue
        distillation_evidence.append(
            {
                "task_index_filter": task_index_filter,
                "sample_count_after_task_index_filter": sample_count_after_filter,
                "sample_count": sample_count,
            }
        )

    value = {
        "observed_task_indices": sorted(set(observed)),
        "distillation_evidence": distillation_evidence,
    }
    rollout_mismatch = any(task_index != target_task_index for task_index in observed)
    distillation_mismatch = any(
        evidence["task_index_filter"] != target_task_index
        or evidence["sample_count_after_task_index_filter"] < evidence["sample_count"]
        or evidence["sample_count"] <= 0
        for evidence in distillation_evidence
    )
    if rollout_mismatch or distillation_mismatch:
        return _metric(
            "fail",
            value=value,
            detail="rollout task index or filtered distillation sample counts do not match the acceptance target",
            target=target_task_index,
        )
    if missing_rollout or not observed or missing_distillation or not distillation_evidence:
        return _metric(
            "unknown",
            value=value,
            detail=(
                "every rollout episode requires dataset_task_index and every distillation artifact requires "
                "task_index_filter, sample_count_after_task_index_filter and sample_count"
            ),
            target=target_task_index,
        )
    return _metric("pass", value=value, target=target_task_index)


def _waiting_metric(rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    waits = 0
    steps = 0
    missing = 0
    for rollout in rollouts:
        for episode in _episodes(rollout):
            if not isinstance(episode.get("waiting_ticks"), int) or not isinstance(episode.get("steps"), int):
                missing += 1
                continue
            waits += int(episode["waiting_ticks"])
            steps += int(episode["steps"])
    if missing or not steps:
        return _metric("unknown", value={"waiting_ticks": waits, "steps": steps}, detail="waiting_ticks or steps is missing", target=0)
    return _metric("pass" if waits == 0 else "fail", value={"waiting_ticks": waits, "steps": steps}, target=0)


def _deadline_metric(rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events: list[Mapping[str, Any]] = []
    missing = 0
    for rollout in rollouts:
        if _mode(rollout) != "async":
            continue
        for episode in _episodes(rollout):
            episode_events = episode.get("inference_events")
            if not isinstance(episode_events, list):
                missing += 1
                continue
            events.extend(event for event in episode_events if isinstance(event, Mapping))
    if missing or not events or any("deadline_miss" not in event for event in events):
        return _metric("unknown", value={"misses": None, "events": len(events)}, detail="async inference events lack deadline_miss", target=0.01)
    misses = sum(bool(event["deadline_miss"]) for event in events)
    ratio = misses / len(events)
    return _metric("pass" if ratio < 0.01 else "fail", value={"misses": misses, "events": len(events), "ratio": ratio}, target=0.01)


def _frequency_metric(rollouts: Sequence[Mapping[str, Any]], minimum_hz: float) -> dict[str, Any]:
    values: list[float] = []
    missing = 0
    for rollout in rollouts:
        if _mode(rollout) != "async":
            continue
        for episode in _episodes(rollout):
            value = episode.get("effective_control_hz")
            if isinstance(value, (int, float)):
                values.append(float(value))
            else:
                missing += 1
    if missing or not values:
        return _metric("unknown", value={"minimum_hz": min(values) if values else None}, detail="effective_control_hz is missing", target=minimum_hz)
    return _metric(
        "pass" if min(values) >= minimum_hz else "fail",
        value={"minimum_hz": min(values), "mean_hz": sum(values) / len(values), "per_episode_hz": values},
        target=minimum_hz,
    )


def _speedup_metric(benchmarks: Sequence[Mapping[str, Any]], minimum_speedup: float) -> dict[str, Any]:
    speedups: list[float] = []
    for benchmark in benchmarks:
        teacher = benchmark.get("teacher")
        student = benchmark.get("student")
        if not isinstance(teacher, Mapping) or not isinstance(student, Mapping):
            continue
        teacher_time = teacher.get("mean_seconds")
        student_time = student.get("mean_seconds")
        if isinstance(teacher_time, (int, float)) and isinstance(student_time, (int, float)) and student_time > 0:
            speedups.append(float(teacher_time) / float(student_time))
    if not speedups:
        return _metric("unknown", value=None, detail="benchmark lacks teacher/student mean_seconds", target=minimum_speedup)
    return _metric("pass" if min(speedups) >= minimum_speedup else "fail", value={"speedups": speedups, "minimum": min(speedups)}, target=minimum_speedup)


def _success_drop_metric(rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sync = [payload for payload in rollouts if _mode(payload) == "sync"]
    asynchronous = [payload for payload in rollouts if _mode(payload) == "async"]
    if len(sync) != 1 or len(asynchronous) != 1:
        return _metric("unknown", value=None, detail="provide exactly one sync and one async rollout", target=5.0)
    sync_rate = _success_rate(sync[0])
    async_rate = _success_rate(asynchronous[0])
    if sync_rate is None or async_rate is None:
        return _metric("unknown", value=None, detail="success_rate is missing", target=5.0)
    drop = (sync_rate - async_rate) * 100.0
    return _metric(
        "pass" if drop <= 5.0 else "fail",
        value={"sync_success_rate": sync_rate, "async_success_rate": async_rate, "drop_percentage_points": drop},
        target=5.0,
    )


def _field_mean(payload: Mapping[str, Any], field: str) -> float | None:
    values = [episode.get(field) for episode in _episodes(payload)]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def _boundary_metric(
    fused: Mapping[str, Any] | None,
    unfused: Mapping[str, Any] | None,
    field: str,
) -> dict[str, Any]:
    if fused is None or unfused is None:
        return _metric("unknown", value=None, detail="provide fused and unfused rollout artifacts", target="fused < unfused")
    fused_value = _field_mean(fused, field)
    unfused_value = _field_mean(unfused, field)
    if fused_value is None or unfused_value is None:
        return _metric("unknown", value={"fused": fused_value, "unfused": unfused_value}, detail=f"{field} is missing", target="fused < unfused")
    return _metric(
        "pass" if fused_value < unfused_value else "fail",
        value={"fused": fused_value, "unfused": unfused_value},
        target="fused < unfused",
    )


def _signature(payload: Mapping[str, Any]) -> list[tuple[int, int | None]] | None:
    top_seed = payload.get("torch_seed")
    result: list[tuple[int, int | None]] = []
    for episode in _episodes(payload):
        seed = episode.get("seed")
        torch_seed = episode.get("torch_seed", top_seed)
        if not isinstance(seed, int) or not isinstance(torch_seed, int):
            return None
        result.append((seed, torch_seed))
    return sorted(result) if result else None


def _same_episodes_metric(rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signatures = [_signature(rollout) for rollout in rollouts]
    if len(signatures) < 2 or any(signature is None for signature in signatures):
        return _metric("unknown", value=None, detail="at least two rollouts with episode seed and torch_seed are required", target="same")
    passed = all(signature == signatures[0] for signature in signatures[1:])
    return _metric("pass" if passed else "fail", value={"signatures": signatures}, target="same")


def evaluate_acceptance(
    rollouts: Sequence[Mapping[str, Any]],
    benchmarks: Sequence[Mapping[str, Any]] = (),
    *,
    target_task_index: int = 34,
    fused_rollout: Mapping[str, Any] | None = None,
    unfused_rollout: Mapping[str, Any] | None = None,
    distillation_metrics: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Evaluate the acceptance contract without guessing missing evidence."""

    return {
        "finite_output": _finite_metric(rollouts, benchmarks),
        "task_isolation": _task_isolation_metric(rollouts, distillation_metrics, target_task_index),
        "waiting_ticks": _waiting_metric(rollouts),
        "deadline_miss_ratio": _deadline_metric(rollouts),
        "effective_control_frequency": _frequency_metric(rollouts, 15.0),
        "two_step_speedup": _speedup_metric(benchmarks, 2.5),
        "async_success_drop": _success_drop_metric(rollouts),
        "validation_action_error": _metric(
            "unknown",
            value=None,
            detail="requires held-out validation errors for distilled and undistilled 2-step models",
            target="distilled < undistilled",
        ),
        "chunk_boundary_jump": _boundary_metric(fused_rollout, unfused_rollout, "chunk_boundary_jump_mean"),
        "action_smoothness": _boundary_metric(fused_rollout, unfused_rollout, "action_smoothness_mean"),
        "same_episodes_and_seeds": _same_episodes_metric(rollouts),
    }


__all__ = ["evaluate_acceptance"]
