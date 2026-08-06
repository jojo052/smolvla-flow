from copy import deepcopy
from pathlib import Path

import pytest

from smolvla_flow.experiment_config import (
    ExperimentConfigError,
    load_experiment_config,
    unresolved_preflight_items,
    validate_experiment_config,
    validate_rtc_config,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "libero_spatial_task0.toml"


def test_locked_experiment_config_is_valid() -> None:
    config = load_experiment_config(CONFIG_PATH)
    assert config["dataset"]["dataset_task_index"] == 34
    assert config["teacher"]["checkpoint"] == "HuggingFaceVLA/smolvla_libero"
    assert config["distillation"]["progressive_flow_steps"] == [5, 2]
    assert unresolved_preflight_items(config) == [
        "gripper_polarity",
        "formal_evaluation_budget",
    ]


def test_rejects_control_frequency_drift() -> None:
    config = load_experiment_config(CONFIG_PATH)
    changed = deepcopy(config)
    changed["control"]["frequency_hz"] = 10
    with pytest.raises(ExperimentConfigError, match="20 Hz"):
        validate_experiment_config(changed)


def test_rejects_semantic_loss_in_v1() -> None:
    config = load_experiment_config(CONFIG_PATH)
    changed = deepcopy(config)
    changed["distillation"]["semantic_loss_enabled"] = True
    with pytest.raises(ExperimentConfigError, match="action-only"):
        validate_experiment_config(changed)


def test_rtc_guard_rejects_horizon_lock_drift() -> None:
    config = load_experiment_config(CONFIG_PATH)
    changed = deepcopy(config)
    changed["control"]["rtc"]["execution_horizon"] = 5
    with pytest.raises(ExperimentConfigError, match="10-step"):
        validate_rtc_config(changed)


def test_rtc_guard_rejects_horizon_execute_steps_mismatch() -> None:
    config = load_experiment_config(CONFIG_PATH)
    changed = deepcopy(config)
    changed["control"]["execute_steps"] = 5
    with pytest.raises(ExperimentConfigError, match="execute_steps"):
        validate_rtc_config(changed)


def test_rtc_guard_rejects_horizon_above_chunk_size() -> None:
    config = load_experiment_config(CONFIG_PATH)
    changed = deepcopy(config)
    changed["control"]["chunk_size"] = 5
    with pytest.raises(ExperimentConfigError, match="chunk_size"):
        validate_rtc_config(changed)


def test_rtc_guard_rejects_non_positive_guidance() -> None:
    config = load_experiment_config(CONFIG_PATH)
    changed = deepcopy(config)
    changed["control"]["rtc"]["max_guidance_weight"] = 0
    with pytest.raises(ExperimentConfigError, match="max_guidance_weight"):
        validate_rtc_config(changed)


def test_rtc_guard_rejects_unknown_prefix_schedule() -> None:
    config = load_experiment_config(CONFIG_PATH)
    changed = deepcopy(config)
    changed["control"]["rtc"]["prefix_attention_schedule"] = "CUSTOM"
    with pytest.raises(ExperimentConfigError, match="prefix_attention_schedule"):
        validate_rtc_config(changed)


def test_rtc_guard_accepts_locked_config() -> None:
    config = load_experiment_config(CONFIG_PATH)
    validate_rtc_config(config)
