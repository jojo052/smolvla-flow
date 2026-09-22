from types import SimpleNamespace

import pytest

from scripts.run_libero_rollout import (
    PolicyLoadResult,
    _aggregate,
    _configure_policy_temporal_contract,
    _resolve_episode_seeds,
    parse_args,
)


def test_rollout_aggregate_contains_wilson_success_statistics() -> None:
    result = _aggregate(
        [
            {"success": True, "steps": 10, "reward_sum": 1.0, "waiting_ticks": 0, "effective_control_hz": 10.0, "action_smoothness_mean": 0.1},
            {"success": False, "steps": 11, "reward_sum": 0.0, "waiting_ticks": 0, "effective_control_hz": 9.0, "action_smoothness_mean": 0.2},
        ]
    )
    assert result["success_rate"] == 0.5
    assert result["success_statistics"]["trial_count"] == 2
    assert len(result["success_statistics"]["wilson_95"]) == 2
    assert result["mean_effective_control_hz"] == 9.5
    assert result["mean_wall_throughput_hz"] == 9.5


def test_rollout_parser_accepts_policy_type_and_protocol() -> None:
    args = parse_args(
        [
            "--policy-type",
            "diffusion",
            "--checkpoint",
            "checkpoint",
            "--protocol",
            "configs/evaluation_protocol.toml",
        ]
    )
    assert args.policy_type == "diffusion"
    assert str(args.protocol).endswith("evaluation_protocol.toml")


def test_smolvla_execution_horizon_is_overridden_before_loading() -> None:
    config = SimpleNamespace(n_action_steps=50, n_obs_steps=1)

    action_steps, obs_steps = _configure_policy_temporal_contract(
        config,
        policy_type="smolvla",
        action_execution_steps=10,
    )

    assert config.n_action_steps == 10
    assert action_steps == 10
    assert obs_steps == 1


@pytest.mark.parametrize("policy_type", ["act", "diffusion"])
def test_checkpoint_policy_rejects_mismatched_execution_horizon(policy_type: str) -> None:
    config = SimpleNamespace(n_action_steps=8, n_obs_steps=2)

    with pytest.raises(RuntimeError, match=r"n_action_steps 8 does not match protocol 10"):
        _configure_policy_temporal_contract(
            config,
            policy_type=policy_type,
            action_execution_steps=10,
        )


@pytest.mark.parametrize("policy_type", ["act", "diffusion"])
def test_checkpoint_policy_rejects_missing_execution_horizon(policy_type: str) -> None:
    with pytest.raises(RuntimeError, match="checkpoint is missing n_action_steps"):
        _configure_policy_temporal_contract(
            SimpleNamespace(n_obs_steps=1),
            policy_type=policy_type,
            action_execution_steps=10,
        )


def test_policy_load_result_keeps_legacy_seven_value_unpacking() -> None:
    loaded = PolicyLoadResult(
        policy="policy",
        preprocessor="pre",
        postprocessor="post",
        checkpoint_path="checkpoint",
        adapter_parameter_count=0,
        chunk_size=50,
        action_dim=7,
        policy_n_action_steps=10,
        policy_n_obs_steps=2,
    )

    assert tuple(loaded) == ("policy", "pre", "post", "checkpoint", 0, 50, 7)
    assert loaded.policy_n_action_steps == 10
    assert loaded.policy_n_obs_steps == 2


def test_default_request_expands_to_exact_formal_seed_list() -> None:
    assert _resolve_episode_seeds(
        episodes=1,
        start_seed=0,
        formal_episode_seeds=[3, 7, 11],
    ) == [3, 7, 11]


def test_explicit_complete_formal_seed_range_is_accepted() -> None:
    assert _resolve_episode_seeds(
        episodes=3,
        start_seed=4,
        formal_episode_seeds=[4, 5, 6],
    ) == [4, 5, 6]


@pytest.mark.parametrize(
    ("episodes", "start_seed"),
    [(2, 0), (1, 1), (3, 1)],
)
def test_formal_seed_subset_or_shift_is_rejected(episodes: int, start_seed: int) -> None:
    with pytest.raises(ValueError, match="complete protocol formal episode_seeds"):
        _resolve_episode_seeds(
            episodes=episodes,
            start_seed=start_seed,
            formal_episode_seeds=[0, 1, 2],
        )
