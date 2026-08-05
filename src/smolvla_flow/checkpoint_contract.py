"""Static contract inspection for SmolVLA checkpoints."""

from __future__ import annotations

from typing import Any


def feature_shape(features: dict[str, Any], key: str) -> tuple[int, ...] | None:
    feature = features.get(key)
    if feature is None:
        return None
    return tuple(int(value) for value in feature["shape"])


def normalizer_features(preprocessor: dict[str, Any]) -> dict[str, Any]:
    for step in preprocessor["steps"]:
        if step["registry_name"] == "normalizer_processor":
            return step["config"]["features"]
    raise ValueError("checkpoint preprocessor has no normalizer_processor")


def checkpoint_contract(
    policy_config: dict[str, Any],
    preprocessor: dict[str, Any],
) -> dict[str, Any]:
    policy_features = policy_config["input_features"]
    processor_features = normalizer_features(preprocessor)
    visual_features = [
        key
        for key, value in processor_features.items()
        if value["type"].upper() == "VISUAL"
    ]
    return {
        "state_shape": feature_shape(processor_features, "observation.state"),
        "action_shape": feature_shape(processor_features, "action"),
        "camera_count": len(visual_features),
        "camera_keys": sorted(visual_features),
        "policy_state_shape": feature_shape(policy_features, "observation.state"),
        "flow_steps": int(policy_config["num_steps"]),
        "chunk_size": int(policy_config["chunk_size"]),
    }


def contract_matches_libero_v3(contract: dict[str, Any]) -> bool:
    return (
        contract["state_shape"] == (8,)
        and contract["action_shape"] == (7,)
        and contract["camera_count"] == 2
        and contract["flow_steps"] == 10
        and contract["chunk_size"] == 50
    )
