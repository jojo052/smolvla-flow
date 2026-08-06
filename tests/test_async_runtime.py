import threading
import time

import pytest
import torch

from smolvla_flow.async_runtime import (
    ActionQueue,
    AsyncClosedLoopController,
    AsyncPolicyServer,
    AsyncRuntimeConfig,
    GripperHysteresis,
    LeRobotPostprocessorAdapter,
    RuntimeStateError,
)


def test_action_queue_blends_overlap_and_discards_stale_suffix() -> None:
    queue = ActionQueue(overlap_steps=3)
    queue.seed(torch.zeros(5, 2))
    result = queue.merge(torch.full((4, 2), 10.0))
    assert result.old_depth == 5
    assert result.blended_steps == 3
    assert result.dropped_old_steps == 2
    merged = queue.snapshot(batched=False)
    assert merged is not None
    torch.testing.assert_close(
        merged,
        torch.tensor(
            [
                [10.0 / 3.0, 10.0 / 3.0],
                [20.0 / 3.0, 20.0 / 3.0],
                [10.0, 10.0],
                [10.0, 10.0],
            ]
        ),
    )


def test_action_queue_discards_inference_delay_before_blending() -> None:
    queue = ActionQueue(overlap_steps=2)
    queue.seed(torch.zeros(3, 1))
    result = queue.merge(torch.arange(6, dtype=torch.float32).reshape(6, 1), discard_prefix_steps=2)
    assert result.dropped_new_steps == 2
    assert result.blended_steps == 2
    merged = queue.snapshot(batched=False)
    assert merged is not None
    torch.testing.assert_close(
        merged,
        torch.tensor([[1.0], [3.0], [4.0], [5.0]]),
    )


def test_rtc_action_queue_replaces_without_blending_and_snapshots_original() -> None:
    queue = ActionQueue(overlap_steps=3, rtc_enabled=True)
    queue.seed(torch.zeros(5, 2))

    result = queue.merge(
        torch.full((4, 2), 10.0),
        discard_prefix_steps=1,
    )

    assert result.old_depth == 5
    assert result.blended_steps == 0
    assert result.dropped_old_steps == 5
    assert result.dropped_new_steps == 1
    assert queue.depth() == 3
    snapshot = queue.snapshot(batched=False)
    assert snapshot is not None
    torch.testing.assert_close(snapshot, torch.full((3, 2), 10.0))

    popped = queue.pop()
    assert popped is not None
    torch.testing.assert_close(popped, torch.full((2,), 10.0))
    remaining = queue.snapshot(batched=False)
    assert remaining is not None
    torch.testing.assert_close(remaining, torch.full((2, 2), 10.0))


def test_policy_server_syncs_rtc_mode_to_existing_queue() -> None:
    config = AsyncRuntimeConfig(
        rtc_enabled=True,
        overlap_steps=2,
        rtc_execution_horizon=2,
        gripper_polarity="positive_open",
    )
    queue = ActionQueue(overlap_steps=config.overlap_steps)
    queue.seed(torch.zeros(5, 1))

    def policy(observation, **kwargs):
        del observation, kwargs
        return torch.full((4, 1), 10.0)

    server = AsyncPolicyServer(policy, queue, config)
    try:
        assert server.submit({"frame": 1})
        assert server.wait_for_idle(timeout=1.0)
        event = server.events()[0]
        merged = queue.snapshot(batched=False)
        assert merged is not None
        assert tuple(merged.shape) == (4 - event.dropped_new_steps, 1)
        torch.testing.assert_close(merged, torch.full_like(merged, 10.0))
    finally:
        server.close()


def test_gripper_hysteresis_holds_between_thresholds() -> None:
    hysteresis = GripperHysteresis(
        action_index=1,
        low_threshold=-0.5,
        high_threshold=0.5,
        polarity="positive_open",
    )
    closed = hysteresis.apply(torch.tensor([0.0, -1.0]))
    held = hysteresis.apply(torch.tensor([0.0, 0.0]))
    opened = hysteresis.apply(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(closed, torch.tensor([0.0, -1.0]))
    torch.testing.assert_close(held, torch.tensor([0.0, -1.0]))
    torch.testing.assert_close(opened, torch.tensor([0.0, 1.0]))


def test_lerobot_postprocessor_runs_after_queue_pop() -> None:
    calls: list[tuple[int, ...]] = []

    def postprocessor(action: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(action.shape))
        return action + 2.0

    adapter = LeRobotPostprocessorAdapter(postprocessor)
    output = adapter(torch.ones(7))
    torch.testing.assert_close(output, torch.full((7,), 3.0))
    assert calls == [(1, 7)]


def test_pending_gripper_polarity_blocks_hardware_controller() -> None:
    config = AsyncRuntimeConfig(gripper_polarity="pending", require_gripper_polarity=True)
    queue = ActionQueue(overlap_steps=config.overlap_steps)

    def policy(observation, **kwargs):
        del observation, kwargs
        return torch.zeros(50, 7)

    server = AsyncPolicyServer(policy, queue, config)
    try:
        with pytest.raises(RuntimeStateError, match="polarity"):
            AsyncClosedLoopController(server, queue, config, simulation_only=False)
    finally:
        server.close()


def test_policy_server_passes_rtc_context_and_measures_delay() -> None:
    config = AsyncRuntimeConfig(
        control_frequency_hz=20.0,
        overlap_steps=2,
        rtc_enabled=True,
        rtc_execution_horizon=2,
        gripper_polarity="positive_open",
    )
    queue = ActionQueue(overlap_steps=config.overlap_steps)
    queue.seed(torch.zeros(5, 7))
    calls: list[dict] = []
    called = threading.Event()

    def policy(observation, **kwargs):
        calls.append({"observation": observation, **kwargs})
        called.set()
        time.sleep(0.005)
        return torch.ones(4, 7)

    server = AsyncPolicyServer(policy, queue, config)
    try:
        assert server.submit({"frame": 1})
        assert server.wait_for_idle(timeout=1.0)
        assert called.is_set()
        assert len(calls) == 1
        assert calls[0]["inference_delay"] == 0
        assert calls[0]["execution_horizon"] == 2
        assert tuple(calls[0]["prev_chunk_left_over"].shape) == (1, 2, 7)
        assert server.stats()["completed"] == 1
        assert server.last_delay_steps >= 1
        event = server.events()[0]
        assert event.dropped_new_steps == event.resulting_delay
        assert queue.depth() == 4 - event.dropped_new_steps
    finally:
        server.close()


def test_policy_server_can_keep_rtc_without_passing_action_prefix() -> None:
    config = AsyncRuntimeConfig(
        rtc_enabled=True,
        rtc_use_prefix=False,
        rtc_execution_horizon=2,
        gripper_polarity="positive_open",
    )
    queue = ActionQueue(overlap_steps=config.overlap_steps)
    queue.seed(torch.zeros(5, 7))
    calls: list[dict] = []

    def policy(observation, **kwargs):
        del observation
        calls.append(kwargs)
        return torch.ones(4, 7)

    server = AsyncPolicyServer(policy, queue, config)
    try:
        assert server.submit({"frame": 1})
        assert server.wait_for_idle(timeout=1.0)
        assert calls[0]["prev_chunk_left_over"] is None
        assert calls[0]["inference_delay"] >= 0
    finally:
        server.close()


def test_policy_server_uses_consumed_control_ticks_when_supplied() -> None:
    config = AsyncRuntimeConfig(
        control_frequency_hz=20.0,
        overlap_steps=2,
        rtc_enabled=True,
        rtc_execution_horizon=2,
        gripper_polarity="positive_open",
    )
    queue = ActionQueue(overlap_steps=config.overlap_steps)
    queue.seed(torch.zeros(8, 7))
    started = threading.Event()
    release = threading.Event()

    def policy(observation, **kwargs):
        del observation, kwargs
        started.set()
        assert release.wait(timeout=1.0)
        return torch.ones(4, 7)

    server = AsyncPolicyServer(policy, queue, config)
    try:
        submitted_step = server.advance_control_step()
        assert server.submit({"frame": 1}, control_step=submitted_step)
        assert started.wait(timeout=1.0)
        for _ in range(3):
            server.advance_control_step()
            queue.pop()
        release.set()
        assert server.wait_for_idle(timeout=1.0)
        event = server.events()[0]
        assert event.resulting_delay == 3
        assert event.wall_delay_steps is not None
        assert event.dropped_new_steps == 3
    finally:
        release.set()
        server.close()


def test_policy_server_reuses_resulting_delay_for_next_rtc_request() -> None:
    config = AsyncRuntimeConfig(
        control_frequency_hz=20.0,
        overlap_steps=2,
        rtc_enabled=True,
        rtc_execution_horizon=2,
        gripper_polarity="positive_open",
    )
    queue = ActionQueue(overlap_steps=config.overlap_steps)
    queue.seed(torch.zeros(8, 7))
    calls: list[dict] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def policy(observation, **kwargs):
        del observation
        calls.append(kwargs)
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(timeout=1.0)
        return torch.ones(4, 7)

    server = AsyncPolicyServer(policy, queue, config)
    try:
        submitted_step = server.advance_control_step()
        assert server.submit({"frame": 1}, control_step=submitted_step)
        assert first_started.wait(timeout=1.0)
        for _ in range(3):
            server.advance_control_step()
            queue.pop()
        release_first.set()
        assert server.wait_for_idle(timeout=1.0)
        first_event = server.events()[0]
        assert first_event.resulting_delay == 3

        assert server.submit({"frame": 2}, control_step=server.advance_control_step())
        assert server.wait_for_idle(timeout=1.0)
        assert calls[1]["inference_delay"] == first_event.resulting_delay
    finally:
        release_first.set()
        server.close()


def test_controller_prefetches_before_queue_is_empty() -> None:
    config = AsyncRuntimeConfig(
        overlap_steps=2,
        execute_steps=2,
        gripper_polarity="positive_open",
    )
    queue = ActionQueue(overlap_steps=config.overlap_steps)
    call_count = 0

    def policy(observation, **kwargs):
        nonlocal call_count
        del observation, kwargs
        call_count += 1
        return torch.full((4, 7), float(call_count))

    server = AsyncPolicyServer(policy, queue, config)
    controller = AsyncClosedLoopController(server, queue, config, simulation_only=False)
    try:
        controller.seed_actions(torch.zeros(4, 7))
        results = []
        for _ in range(3):
            results.append(controller.tick({"frame": 0}))
        assert results[2].triggered_inference
        assert all(result.action is not None for result in results)
        assert server.wait_for_idle(timeout=1.0)
        assert call_count == 1
        merged_action = queue.pop()
        assert merged_action is not None
        assert float(merged_action[0]) > 0.0
    finally:
        controller.close()
