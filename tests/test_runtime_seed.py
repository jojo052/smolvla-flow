import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from smolvla_flow.async_runtime import seed_policy_rng
from smolvla_flow.async_runtime import (
    ActionQueue,
    AsyncPolicyServer,
    AsyncRuntimeConfig,
)


_ROLLOUT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_libero_rollout.py"


def _load_rollout_script():
    spec = importlib.util.spec_from_file_location("run_libero_rollout", _ROLLOUT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_seed_policy_rng_makes_policy_noise_reproducible() -> None:
    seed_policy_rng(123)
    first = torch.randn(50, 7)
    second = torch.randn(3)
    seed_policy_rng(123)
    torch.testing.assert_close(first, torch.randn(50, 7))
    torch.testing.assert_close(second, torch.randn(3))


def test_seed_policy_rng_covers_numpy_and_python_rng() -> None:
    seed_policy_rng(7)
    numpy_values = np.random.rand(4)
    python_values = [random.random() for _ in range(4)]
    seed_policy_rng(7)
    np.testing.assert_array_equal(numpy_values, np.random.rand(4))
    assert python_values == [random.random() for _ in range(4)]


def test_seed_policy_rng_rejects_non_integer() -> None:
    try:
        seed_policy_rng("123")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("seed_policy_rng accepted a non-integer seed")

    try:
        seed_policy_rng(True)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("seed_policy_rng accepted a boolean seed")


def test_rollout_script_exposes_torch_seed_option() -> None:
    rollout = _load_rollout_script()
    assert rollout.parse_args(["--mode", "sync", "--episodes", "1"]).torch_seed is None
    assert rollout.parse_args(["--torch-seed", "123"]).torch_seed == 123


def test_policy_server_seeds_worker_thread_rng() -> None:
    """CPU noise drawn inside the policy worker is controlled by the seed.

    PyTorch's default CPU generator is thread-local, so main-thread seeding
    does not reach the worker thread.  The server must seed its own thread.
    """

    def collect(seed: int) -> list:
        config = AsyncRuntimeConfig(rtc_enabled=False, gripper_polarity="positive_open")
        queue = ActionQueue(overlap_steps=config.overlap_steps)
        collected = []

        def policy(observation, **kwargs):
            del observation, kwargs
            collected.append(torch.randn(5, 7).clone())
            return torch.zeros(2, 7)

        server = AsyncPolicyServer(policy, queue, config, seed=seed)
        try:
            assert server.submit({"frame": 0})
            assert server.wait_for_idle(timeout=2.0)
            assert server.submit({"frame": 1})
            assert server.wait_for_idle(timeout=2.0)
        finally:
            server.close()
        return collected

    first = collect(123)
    second = collect(123)
    assert len(first) == len(second) == 2
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_policy_server_rejects_non_integer_seed() -> None:
    config = AsyncRuntimeConfig(rtc_enabled=False, gripper_polarity="positive_open")
    queue = ActionQueue(overlap_steps=config.overlap_steps)

    def policy(observation, **kwargs):
        del observation, kwargs
        return torch.zeros(2, 7)

    try:
        AsyncPolicyServer(policy, queue, config, seed="123")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("AsyncPolicyServer accepted a non-integer seed")

    with pytest.raises(TypeError, match="seed"):
        AsyncPolicyServer(policy, queue, config, seed=True)  # type: ignore[arg-type]
