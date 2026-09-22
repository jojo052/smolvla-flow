"""Pure, replayable bookkeeping for the fixed 40-task public-model benchmark."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path

SUITES = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
OFFICIAL_SHA256 = "71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def freeze_manifest(path, manifest):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Frozen manifest differs; use a new output directory")
    else:
        atomic_json(path, manifest)
    return fingerprint(manifest)


def episode_key(model, suite, task_id, index, phase="formal"):
    if suite not in SUITES or task_id not in range(10):
        raise ValueError("Unknown benchmark task")
    if phase not in {"formal", "pilot"} or index not in (range(10) if phase == "formal" else [10]):
        raise ValueError("Initial state index is outside frozen protocol")
    if phase == "pilot" and task_id != 0:
        raise ValueError("Pilot uses task 0 only")
    return f"{phase}/{model}/{suite}/task{task_id:02d}/init{index:02d}"


def validate_episode(row, run_id, expected_key):
    if row.get("run_id") != run_id or row.get("key") != expected_key:
        raise ValueError("Episode identity does not match frozen run")
    if episode_key(row["model"], row["suite"], row["task_id"], row["init_index"], row["phase"]) != expected_key:
        raise ValueError("Episode fields disagree with storage identity")
    if row.get("status") != "complete":
        raise ValueError("Execution error requires inspection before resuming")
    if type(row.get("success")) is not bool or not 1 <= row["steps"] <= SUITES[row["suite"]]:
        raise ValueError("Invalid outcome or step count")
    if row.get("action_nonfinite_count") != 0:
        raise ValueError("Nonfinite action in completed episode")
    if row.get("failure_type") != (None if row["success"] else "timeout"):
        raise ValueError("Invalid failure classification")
    if not row["success"] and row["steps"] != SUITES[row["suite"]]:
        raise ValueError("Early termination cannot be a normal timeout")
    for key in ("elapsed_seconds", "total_seconds"):
        if not math.isfinite(row[key]) or row[key] <= 0:
            raise ValueError("Invalid elapsed time")
    samples = row["select_seconds"]
    chunks = row["prediction_seconds"]
    if len(samples) != row["steps"] or not chunks:
        raise ValueError("Missing latency samples")
    if any(not math.isfinite(x) or x < 0 for x in samples + chunks):
        raise ValueError("Invalid latency sample")


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0, center - radius), min(1, center + radius)]


def latency(values):
    if not values:
        return None
    values = sorted(values)
    return {"mean_ms": statistics.mean(values) * 1000,
            "p50_ms": statistics.median(values) * 1000,
            "p95_ms": values[max(0, math.ceil(.95 * len(values)) - 1)] * 1000,
            "count": len(values)}


def summarize(rows, models):
    rows = [r for r in rows if r["phase"] == "formal"]
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate episode")
    tasks, overview = [], {}
    for model in models:
        selected = [r for r in rows if r["model"] == model]
        suites = {}
        for suite in SUITES:
            for tid in range(10):
                subset = [r for r in selected if r["suite"] == suite and r["task_id"] == tid]
                n, k = len(subset), sum(r["success"] for r in subset)
                tasks.append({"model": model, "suite": suite, "task_id": tid, "trials": n,
                              "successes": k, "success_rate": k / n if n else None,
                              "wilson_95": wilson(k, n), "timeouts": n - k})
            subset = [r for r in selected if r["suite"] == suite]
            suites[suite] = {"trials": len(subset), "successes": sum(r["success"] for r in subset),
                             "success_rate": sum(r["success"] for r in subset) / 100 if len(subset) == 100 else None}
        elapsed = sum(r["elapsed_seconds"] for r in selected)
        total = sum(r["total_seconds"] for r in selected)
        overview[model] = {"trials": len(selected), "expected": 400, "complete": len(selected) == 400,
                           "suites": suites,
                           "macro_success_rate": sum(r["success"] for r in selected) / 400 if len(selected) == 400 else None,
                           "wall_tick_per_s": sum(r["steps"] for r in selected) / elapsed if elapsed else None,
                           "success_per_hour_including_reset": 3600 * sum(r["success"] for r in selected) / total if total else None,
                           "prediction_latency": latency([v for r in selected for v in r["prediction_seconds"]]),
                           "select_latency": latency([v for r in selected for v in r["select_seconds"]]),
                           "peak_vram_bytes": max((r["peak_vram_bytes"] for r in selected), default=0)}
    paired = None
    if len(models) == 2 and all(x["complete"] for x in overview.values()):
        maps = [{(r["suite"], r["task_id"], r["init_index"]): r for r in rows if r["model"] == m} for m in models]
        if maps[0].keys() != maps[1].keys():
            raise ValueError("Unpaired evaluation states")
        counts = {"both_success": 0, "first_only": 0, "second_only": 0, "both_fail": 0}
        deltas = []
        for suite in SUITES:
            for tid in range(10):
                delta = 0
                for idx in range(10):
                    a, b = [m[(suite, tid, idx)] for m in maps]
                    if a["init_state_sha256"] != b["init_state_sha256"]:
                        raise ValueError("Paired initial states differ")
                    x, y = a["success"], b["success"]
                    counts["both_success" if x and y else "first_only" if x else "second_only" if y else "both_fail"] += 1
                    delta += int(x) - int(y)
                deltas.append(delta / 10)
        rng = random.Random(123)
        samples = sorted(statistics.mean(rng.choices(deltas, k=40)) for _ in range(10000))
        paired = {"models": models, "counts": counts, "first_minus_second": statistics.mean(deltas),
                  "task_bootstrap_95": [samples[249], samples[9749]], "bootstrap_replicates": 10000}
    return {"models": overview, "tasks": tasks, "paired": paired}


def markdown_report(summary):
    lines = ["# 四套件 40 任务评估", "", "未完成套件的成功率显示为待完成，不用部分回合估计正式结果。", "",
             "| 模型 | Spatial | Object | Goal | Long | 宏平均 | rollout tick/s | 完整预测 P50/P95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "待完成" if x is None else f"{100*x:.2f}%"
    for model, row in summary["models"].items():
        rate = [fmt(row["suites"][s]["success_rate"]) for s in SUITES]
        lat = row["prediction_latency"]
        timing = f'{lat["p50_ms"]:.2f}/{lat["p95_ms"]:.2f}' if lat else "待测"
        ticks = f'{row["wall_tick_per_s"]:.2f}' if row["wall_tick_per_s"] else "待测"
        lines.append(f'| {model} ({row["trials"]}/400) | ' + " | ".join(rate + [fmt(row["macro_success_rate"]), ticks, timing]) + " |")
    lines += ["", "## 逐任务", "", "| 模型 | 套件 | task | 成功/回合 | Wilson 95% | 超时 |", "|---|---|---:|---:|---|---:|"]
    for row in summary["tasks"]:
        ci = row["wilson_95"]
        interval = f"{ci[0]*100:.2f}% 至 {ci[1]*100:.2f}%" if ci else "待测"
        lines.append(f'| {row["model"]} | {row["suite"]} | {row["task_id"]} | {row["successes"]}/{row["trials"]} | {interval} | {row["timeouts"]} |')
    return "\n".join(lines) + "\n"
