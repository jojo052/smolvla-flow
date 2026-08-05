"""Loading and validation for the locked LIBERO experiment configuration."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration violates a required invariant."""


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    teacher = config["teacher"]
    act = config["baseline"]["act"]
    diffusion = config["baseline"]["diffusion"]
    distillation = config["distillation"]
    control = config["control"]
    rtc = control["rtc"]

    if dataset["suite"] != "libero_spatial" or dataset["suite_task_id"] != 0:
        raise ExperimentConfigError("v1 must target LIBERO-Spatial task 0")
    if dataset["state_dim"] != 8 or dataset["action_dim"] != 7:
        raise ExperimentConfigError("LIBERO observations/actions must use 8D state and 7D action")
    if teacher["flow_steps"] != 10:
        raise ExperimentConfigError("the selected SmolVLA teacher must use 10 flow steps")
    if teacher["chunk_size"] != act["chunk_size"] or teacher["chunk_size"] != control["chunk_size"]:
        raise ExperimentConfigError("teacher, ACT, and controller must share chunk_size")
    if teacher["execute_steps"] != act["n_action_steps"]:
        raise ExperimentConfigError("teacher and ACT must execute the same number of actions")
    if teacher["execute_steps"] != diffusion["n_action_steps"]:
        raise ExperimentConfigError("teacher and Diffusion Policy must execute the same number of actions")
    if distillation["scope"] != "action_only" or distillation["semantic_loss_enabled"]:
        raise ExperimentConfigError("v1 is restricted to action-only distillation")
    if distillation["progressive_flow_steps"] != [5, 2]:
        raise ExperimentConfigError("v1 progressive distillation must follow 10 to 5 to 2 steps")
    if control["frequency_hz"] != 20 or control["period_ms"] != 50:
        raise ExperimentConfigError("v1 controller must run at 20 Hz with a 50 ms period")
    if control["overlap_steps"] != 10:
        raise ExperimentConfigError("v1 overlap must contain 10 actions")
    if not rtc["enabled"] or rtc["execution_horizon"] != 10:
        raise ExperimentConfigError("RTC must be enabled with a 10-step execution horizon")
    if control["gripper"]["mode"] != "hysteresis":
        raise ExperimentConfigError("the gripper dimension must use hysteresis")


def unresolved_preflight_items(config: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if config["dataset"]["dataset_task_index"] < 0:
        items.append("dataset_task_index")
    if config["control"]["gripper"]["polarity"].startswith("pending"):
        items.append("gripper_polarity")
    if config["teacher"]["checkpoint_status"].startswith("pending"):
        items.append("teacher_checkpoint_interface")
    if config["evaluation"]["formal"]["status"].startswith("pending"):
        items.append("formal_evaluation_budget")
    return items
