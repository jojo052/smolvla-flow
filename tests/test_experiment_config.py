from copy import deepcopy
from pathlib import Path

import pytest

from smolvla_flow.experiment_config import (
    ExperimentConfigError,
    load_experiment_config,
    unresolved_preflight_items,
    validate_experiment_config,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "libero_spatial_task0.toml"


def test_locked_experiment_config_is_valid() -> None:
    config = load_experiment_config(CONFIG_PATH)
    assert config["teacher"]["checkpoint"] == "lerobot/smolvla_libero"
    assert config["distillation"]["progressive_flow_steps"] == [5, 2]
    assert unresolved_preflight_items(config) == [
        "dataset_task_index",
        "gripper_polarity",
        "teacher_checkpoint_interface",
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
