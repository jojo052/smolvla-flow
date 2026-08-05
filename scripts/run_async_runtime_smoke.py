#!/usr/bin/env python
"""Exercise the asynchronous queue, prefetch trigger, fusion, and RTC kwargs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from smolvla_flow.async_runtime import (
    ActionQueue,
    AsyncClosedLoopController,
    AsyncPolicyServer,
    AsyncRuntimeConfig,
)


class MockPolicy:
    def __init__(self, latency_seconds: float) -> None:
        self.latency_seconds = latency_seconds
        self.calls: list[dict] = []

    def __call__(self, observation, **kwargs):
        time.sleep(self.latency_seconds)
        call_id = len(self.calls) + 1
        previous = kwargs.get("prev_chunk_left_over")
        self.calls.append(
            {
                "observation": observation,
                "inference_delay": int(kwargs["inference_delay"]),
                "execution_horizon": int(kwargs["execution_horizon"]),
                "previous_prefix_shape": list(previous.shape) if previous is not None else None,
            }
        )
        actions = torch.full((1, 50, 7), float(call_id))
        actions[:, :, 6] = 1.0 if call_id % 2 else -1.0
        return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--latency-seconds", type=float, default=0.08)
    parser.add_argument("--tick-sleep", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/async_runtime_smoke.json"))
    args = parser.parse_args()
    if args.ticks < 1 or args.latency_seconds < 0:
        raise SystemExit("ticks must be positive and latency-seconds must be non-negative")

    config = AsyncRuntimeConfig(
        control_frequency_hz=20.0,
        chunk_size=50,
        execute_steps=10,
        overlap_steps=10,
        rtc_enabled=True,
        rtc_execution_horizon=10,
        gripper_polarity="positive_open",
    )
    action_queue = ActionQueue(overlap_steps=config.overlap_steps)
    policy = MockPolicy(args.latency_seconds)
    server = AsyncPolicyServer(policy, action_queue, config)
    controller = AsyncClosedLoopController(
        server,
        action_queue,
        config,
        simulation_only=False,
    )
    controller.seed_actions(torch.zeros(50, 7))
    tick_sleep = config.period_seconds if args.tick_sleep is None else args.tick_sleep
    actions = []
    results = []
    started = time.monotonic()
    try:
        for tick in range(args.ticks):
            result = controller.tick({"frame": tick})
            results.append(result)
            if result.action is not None:
                actions.append(result.action)
            if tick_sleep:
                time.sleep(tick_sleep)
        server.wait_for_idle(timeout=max(2.0, args.latency_seconds * 4.0))
    finally:
        controller.close()
    elapsed = time.monotonic() - started
    events = server.events()
    result = {
        "status": "completed",
        "runtime": {
            "control_frequency_hz": config.control_frequency_hz,
            "chunk_size": config.chunk_size,
            "execute_steps": config.execute_steps,
            "overlap_steps": config.overlap_steps,
            "prefetch_threshold": config.trigger_threshold,
            "rtc_enabled": config.rtc_enabled,
            "rtc_execution_horizon": config.rtc_execution_horizon,
            "gripper_polarity": config.gripper_polarity,
        },
        "ticks": args.ticks,
        "elapsed_seconds": elapsed,
        "actions_emitted": len(actions),
        "waiting_ticks": sum(result.waiting_for_policy for result in results),
        "triggered_inference_ticks": sum(result.triggered_inference for result in results),
        "queue_depth_min": min(result.queue_depth_after for result in results),
        "queue_depth_max": max(result.queue_depth_after for result in results),
        "server": server.stats(),
        "inference_events": [event.__dict__ for event in events],
        "policy_calls": policy.calls,
        "effective_control_hz": len(actions) / elapsed if elapsed else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
