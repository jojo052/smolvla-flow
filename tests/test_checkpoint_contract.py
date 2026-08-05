from smolvla_flow.checkpoint_contract import checkpoint_contract, contract_matches_libero_v3


def make_preprocessor(state_dim: int, camera_count: int) -> dict:
    features = {
        "observation.state": {"type": "STATE", "shape": [state_dim]},
        "action": {"type": "ACTION", "shape": [7]},
    }
    for index in range(camera_count):
        features[f"observation.images.camera{index + 1}"] = {
            "type": "VISUAL",
            "shape": [3, 256, 256],
        }
    return {
        "steps": [
            {
                "registry_name": "normalizer_processor",
                "config": {"features": features},
            }
        ]
    }


def make_policy_config(state_dim: int) -> dict:
    return {
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [state_dim]},
        },
        "num_steps": 10,
        "chunk_size": 50,
    }


def test_matches_current_libero_contract() -> None:
    contract = checkpoint_contract(make_policy_config(8), make_preprocessor(8, 2))
    assert contract_matches_libero_v3(contract)


def test_rejects_six_state_three_camera_contract() -> None:
    contract = checkpoint_contract(make_policy_config(6), make_preprocessor(6, 3))
    assert not contract_matches_libero_v3(contract)
