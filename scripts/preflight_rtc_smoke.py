#!/usr/bin/env python
"""Construct and exercise the installed LeRobot RTC components offline."""

from __future__ import annotations

import argparse
import inspect
import json
import traceback
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"constructed": False, "single_call_checked": False}
    try:
        from lerobot.configs import RTCAttentionSchedule
        from lerobot.policies.rtc import ActionQueue, RTCConfig, RTCProcessor

        result["class"] = f"{RTCProcessor.__module__}.{RTCProcessor.__name__}"
        result["signature"] = str(inspect.signature(RTCProcessor))
        try:
            config = RTCConfig(
                enabled=True,
                execution_horizon=10,
                max_guidance_weight=10.0,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
            )
            processor = RTCProcessor(config)
            result["constructed"] = True
            latent = torch.randn(1, 50, 7)
            previous = torch.randn(1, 10, 7)
            guided = processor.denoise_step(
                latent,
                previous,
                inference_delay=4,
                time=torch.tensor(0.5),
                original_denoise_step_partial=lambda value: torch.zeros_like(value),
            )
            queue = ActionQueue(config)
            actions = torch.randn(50, 7)
            queue.merge(actions, actions, real_delay=4)
            first_action = queue.get()
            result["single_call_checked"] = True
            result["processor_output_shape"] = list(guided.shape)
            result["processor_output_finite"] = bool(torch.isfinite(guided).all())
            result["queue_remaining_after_one_get"] = queue.qsize()
            result["queue_first_action_shape"] = list(first_action.shape) if first_action is not None else None
            result["config"] = {
                "enabled": config.enabled,
                "execution_horizon": config.execution_horizon,
                "max_guidance_weight": config.max_guidance_weight,
                "prefix_attention_schedule": config.prefix_attention_schedule.name,
            }
        except Exception as error:
            result["construction_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        result["import_error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
