#!/usr/bin/env python
"""Download small checkpoint JSON files and compare their input contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smolvla_flow.checkpoint_contract import checkpoint_contract, contract_matches_libero_v3
from smolvla_flow.experiment_config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/libero_spatial_task0.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/preflight/teacher_contracts.json"),
    )
    return parser.parse_args()


def load_hub_json(repo_id: str, filename: str) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit("huggingface_hub is required for checkpoint inspection") from error
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    results = []
    for repo_id in config["teacher"]["candidates"]:
        policy_config = load_hub_json(repo_id, "config.json")
        preprocessor = load_hub_json(repo_id, "policy_preprocessor.json")
        contract = checkpoint_contract(policy_config, preprocessor)
        results.append(
            {
                "repo_id": repo_id,
                "contract": contract,
                "matches_libero_v3": contract_matches_libero_v3(contract),
            }
        )
    output = {"candidates": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
