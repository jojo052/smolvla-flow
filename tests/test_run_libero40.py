"""CPU-only integration checks for the new runner; no simulator/model downloads."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.run_libero40 import run_episode, save_report
from smolvla_flow.benchmark40 import atomic_json, fingerprint


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    from scripts import run_libero_rollout
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    batch = {"observation.state": torch.zeros(1, 8),
             "observation.images.image": torch.zeros(1, 3, 256, 256),
             "observation.images.image2": torch.zeros(1, 3, 256, 256)}
    monkeypatch.setattr(run_libero_rollout, "_make_observation_pipeline", lambda task: lambda raw, pre: batch)

    class Policy:
        def __init__(self):
            self._queues = {"action": deque([123])}

        def reset(self):
            self._queues["action"].clear()

        def _get_action_chunk(self, batch):
            return torch.zeros(1, 50, 7)

        def select_action(self, batch):
            return self._get_action_chunk(batch)[:, 0]

    class Env:
        num_steps_wait = 10
        _init_states = list(range(11))

        def reset(self, seed):
            self.selected = self.init_state_id
            self.init_state_id += 1
            return {"pixels": {"image": np.zeros((256, 256, 3), dtype=np.uint8)}}, {}

        def step(self, action):
            self.action = action
            return {}, 1., True, False, {"is_success": True}

    videos = {k: tmp_path / f"{k}.mp4" for k in ["success", "failure"]}
    for p in videos.values():
        p.touch()
    task = {"suite": "libero_spatial", "task_id": 0, "name": "test", "language": "instruction",
            "max_steps": 220, "state_sha256": [str(i) for i in range(11)]}
    loaded = SimpleNamespace(policy=Policy(), preprocessor=None, postprocessor=lambda x: x)
    return Env(), loaded, task, videos


def test_real_runner_resets_queue_selects_exact_state_and_counts_prediction(fake_runtime):
    env, loaded, task, videos = fake_runtime
    original = loaded.policy._get_action_chunk
    row = run_episode(env, loaded, task, 10, "pilot", "r", videos)
    assert env.selected == 10
    assert row["success"] and row["steps"] == 1
    assert len(row["select_seconds"]) == len(row["prediction_seconds"]) == 1
    assert loaded.policy._get_action_chunk == original
    assert row["init_state_sha256"] == "10"
    assert np.array_equal(env.action, np.zeros(7))


def test_full_chunk_nonfinite_detected_before_env_step(fake_runtime):
    env, loaded, task, videos = fake_runtime
    loaded.policy._get_action_chunk = lambda batch: torch.full((1, 50, 7), float("nan"))
    with pytest.raises(RuntimeError, match="finite"):
        run_episode(env, loaded, task, 10, "pilot", "r", videos)
    assert not hasattr(env, "action")


def test_report_does_not_change_existing_episode(fake_runtime, tmp_path):
    env, loaded, task, videos = fake_runtime
    manifest = {"tasks": [task], "protocol": {"act_status": "not_admitted"}}
    row = run_episode(env, loaded, task, 0, "formal", fingerprint(manifest), videos)
    path = tmp_path / (row["key"] + ".json")
    atomic_json(path, row)
    before = path.read_bytes()
    assert save_report(tmp_path, manifest) == 1
    assert save_report(tmp_path, manifest) == 1
    assert path.read_bytes() == before
