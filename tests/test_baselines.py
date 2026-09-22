import pytest
import torch

from scripts.run_baseline_smoke import _load_data_file
from smolvla_flow.baselines import (
    ACTBaseline,
    ACTBaselineConfig,
    DiffusionBaseline,
    DiffusionBaselineConfig,
    make_synthetic_baseline_data,
)


def test_act_baseline_emits_locked_action_chunk() -> None:
    model = ACTBaseline(
        ACTBaselineConfig(
            model_dim=32,
            num_heads=4,
            num_layers=1,
            feedforward_dim=64,
        )
    )
    state = torch.randn(3, 8)
    output = model(state)
    assert output.shape == (3, 50, 7)
    assert torch.isfinite(output).all()


def test_diffusion_baseline_forward_and_sample_are_finite() -> None:
    model = DiffusionBaseline(
        DiffusionBaselineConfig(
            hidden_dim=32,
            horizon=16,
            num_train_timesteps=20,
        )
    )
    state = torch.randn(2, 8)
    actions = torch.randn(2, 16, 7)
    timesteps = torch.tensor([0, 19])
    noise = torch.randn_like(actions)
    noisy = model.add_noise(actions, timesteps, noise)
    prediction = model(noisy, state, timesteps)
    sample = model.sample(state[:1], num_steps=5)
    assert prediction.shape == actions.shape
    assert sample.shape == (1, 16, 7)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(sample).all()


def test_diffusion_baseline_rejects_invalid_steps() -> None:
    model = DiffusionBaseline(DiffusionBaselineConfig(hidden_dim=16, horizon=8))
    with pytest.raises(ValueError, match="positive"):
        model.sample(torch.zeros(1, 8), num_steps=0)


def test_synthetic_baseline_data_is_reproducible() -> None:
    left = make_synthetic_baseline_data(4, seed=7)
    right = make_synthetic_baseline_data(4, seed=7)
    torch.testing.assert_close(left[0], right[0])
    torch.testing.assert_close(left[1], right[1])


def test_smoke_can_load_exported_state_action_artifact(tmp_path) -> None:
    state = torch.zeros(2, 8)
    actions = torch.zeros(2, 64, 7)
    path = tmp_path / "baseline.pt"
    torch.save(
        {
            "data_source": "lerobot_parquet_state_action",
            "splits": {
                "train": {"state": state, "actions": actions},
                "validation": {"state": state[:1], "actions": actions[:1]},
            },
        },
        path,
    )
    loaded = _load_data_file(path, torch.device("cpu"))
    assert loaded[-1] == "lerobot_parquet_state_action"
    assert loaded[0].shape == (2, 8)
    assert loaded[1].shape == (2, 64, 7)
