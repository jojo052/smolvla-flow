from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_lerobot_checkpoint import (
    _checkpoint_image_keys,
    _observation_delta_timestamps,
    _sample_batch,
)


def test_checkpoint_image_keys_use_policy_feature_namespace() -> None:
    policy = SimpleNamespace(
        config=SimpleNamespace(
            input_features={
                "observation.images.image": object(),
                "observation.images.image2": object(),
                "observation.state": object(),
            }
        )
    )

    keys = _checkpoint_image_keys(policy, ("agentview_image", "robot0_eye_in_hand_image"))

    assert keys == ("observation.images.image", "observation.images.image2")


def test_checkpoint_image_keys_require_one_policy_stream_per_camera() -> None:
    policy = SimpleNamespace(
        config=SimpleNamespace(input_features={"observation.images.image": object()})
    )

    with pytest.raises(RuntimeError, match="visual stream count"):
        _checkpoint_image_keys(policy, ("agentview_image", "robot0_eye_in_hand_image"))


def test_sample_batch_accepts_materialized_lerobot_image_keys() -> None:
    sample = {
        "observation.images.image": torch.zeros(3, 256, 256),
        "observation.images.image2": torch.ones(3, 256, 256),
        "observation.state": torch.zeros(8),
        "action": torch.zeros(7),
        "task_index": torch.tensor(34),
    }

    batch = _sample_batch(
        sample,
        ("observation.images.image", "observation.images.image2"),
    )

    assert batch["observation.images.image"].shape == (1, 3, 256, 256)
    assert batch["observation.images.image2"].shape == (1, 3, 256, 256)
    assert batch["observation.state"].shape == (1, 8)
    assert "action" not in batch
    assert "task_index" not in batch


def test_sample_batch_reports_missing_checkpoint_image_key() -> None:
    sample = {
        "observation.images.image": torch.zeros(3, 256, 256),
        "observation.state": torch.zeros(8),
    }

    with pytest.raises(RuntimeError, match="observation.images.image2"):
        _sample_batch(
            sample,
            ("observation.images.image", "observation.images.image2"),
        )


def test_observation_delta_timestamps_follow_policy_history_without_action_targets() -> None:
    config = SimpleNamespace(observation_delta_indices=[-1, 0])

    deltas = _observation_delta_timestamps(
        config,
        [
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
            "action",
        ],
        10.0,
    )

    assert deltas == {
        "observation.images.image": [-0.1, 0.0],
        "observation.images.image2": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
    }
    assert "action" not in deltas


def test_observation_delta_timestamps_are_omitted_for_single_observation_policy() -> None:
    config = SimpleNamespace(observation_delta_indices=None)

    assert _observation_delta_timestamps(config, ["observation.state"], 10.0) is None
