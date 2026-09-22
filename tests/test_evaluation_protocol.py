import numpy as np
import pytest

from smolvla_flow.evaluation_protocol import (
    EvaluationProtocol,
    action_error,
    binary_success_summary,
    validate_variant_contract,
    wilson_interval,
)


def test_protocol_round_trip_and_variant_contract() -> None:
    protocol = EvaluationProtocol()
    restored = EvaluationProtocol.from_mapping(protocol.to_dict())
    assert restored == protocol
    validate_variant_contract(
        protocol,
        state_dim=8,
        action_dim=7,
        action_chunk_size=64,
        action_execution_steps=10,
        camera_names=protocol.camera_names,
        normalization=protocol.normalization,
    )


def test_protocol_rejects_mismatched_execution_horizon() -> None:
    with pytest.raises(ValueError, match="execution horizon"):
        validate_variant_contract(
            EvaluationProtocol(),
            state_dim=8,
            action_dim=7,
            action_chunk_size=50,
            action_execution_steps=5,
        )


def test_action_error_uses_common_execution_prefix() -> None:
    predicted = np.zeros((2, 64, 7))
    target = np.ones((2, 64, 7))
    result = action_error(predicted, target, execute_steps=10)
    assert result["execute_steps"] == 10
    assert result["action_mse"] == 1.0
    assert result["action_mae"] == 1.0
    assert result["sample_count"] == 2


def test_binary_success_summary_reports_wilson_interval() -> None:
    result = binary_success_summary([True, False, True, False, False])
    assert result["success_count"] == 2
    assert result["trial_count"] == 5
    assert result["success_rate"] == 0.4
    assert len(result["wilson_95"]) == 2
    assert result["wilson_95"] == list(wilson_interval(2, 5))
