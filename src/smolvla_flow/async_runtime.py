"""Asynchronous action-chunk execution for a 20 Hz closed loop.

The control thread only consumes one action per tick.  A worker thread runs
policy inference in parallel and replaces the remaining queue prefix with a
linearly blended overlap when a fresh chunk arrives.  The worker also measures
inference time and passes its delay, the previous unexecuted prefix, and the
execution horizon to an RTC-aware policy callable.

The module does not import LeRobot.  A LeRobot policy can be adapted with
``PreprocessedPolicyAdapter`` and therefore uses the same runtime as a mock
policy, ACT, or Diffusion Policy.
"""

from __future__ import annotations

import math
import queue as thread_queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Protocol

import torch
from torch import Tensor


class RuntimeStateError(RuntimeError):
    """Raised when a runtime configuration is unsafe or internally invalid."""


def _as_action_chunk(actions: Tensor) -> Tensor:
    if not isinstance(actions, Tensor):
        raise TypeError("policy output must be a torch.Tensor")
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError("batched policy output must have batch size 1")
        actions = actions[0]
    if actions.ndim != 2:
        raise ValueError("action chunk must have shape [chunk_size, action_dim] or [1, chunk_size, action_dim]")
    if actions.shape[0] < 1 or actions.shape[1] < 1:
        raise ValueError("action chunk dimensions must be positive")
    if not torch.isfinite(actions).all().item():
        raise ValueError("policy output contains non-finite values")
    return actions.detach().clone()


@dataclass(frozen=True)
class AsyncRuntimeConfig:
    """Timing, overlap, RTC, and gripper settings for one controller."""

    control_frequency_hz: float = 20.0
    chunk_size: int = 50
    execute_steps: int = 10
    overlap_steps: int = 10
    prefetch_threshold: int | None = None
    rtc_enabled: bool = True
    rtc_use_prefix: bool = True
    rtc_execution_horizon: int = 10
    gripper_action_index: int = 6
    gripper_low_threshold: float = -0.5
    gripper_high_threshold: float = 0.5
    gripper_polarity: str = "pending"
    require_gripper_polarity: bool = True

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")
        for name, value in (
            ("chunk_size", self.chunk_size),
            ("execute_steps", self.execute_steps),
            ("rtc_execution_horizon", self.rtc_execution_horizon),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.overlap_steps, int) or self.overlap_steps < 0:
            raise ValueError("overlap_steps must be a non-negative integer")
        if self.execute_steps > self.chunk_size:
            raise ValueError("execute_steps cannot exceed chunk_size")
        if self.overlap_steps > self.chunk_size:
            raise ValueError("overlap_steps cannot exceed chunk_size")
        threshold = self.prefetch_threshold
        if threshold is not None and (threshold < 1 or threshold > self.chunk_size):
            raise ValueError("prefetch_threshold must be in [1, chunk_size]")
        if self.rtc_execution_horizon > self.chunk_size:
            raise ValueError("rtc_execution_horizon cannot exceed chunk_size")
        if self.gripper_action_index < 0:
            raise ValueError("gripper_action_index must be non-negative")
        if self.gripper_low_threshold >= self.gripper_high_threshold:
            raise ValueError("gripper_low_threshold must be below gripper_high_threshold")
        if self.gripper_polarity not in {"positive_open", "negative_open", "pending"}:
            raise ValueError("gripper_polarity must be positive_open, negative_open, or pending")

    @property
    def trigger_threshold(self) -> int:
        return self.prefetch_threshold if self.prefetch_threshold is not None else self.overlap_steps

    @property
    def period_seconds(self) -> float:
        return 1.0 / self.control_frequency_hz

    def latency_to_steps(self, elapsed_seconds: float) -> int:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative")
        return int(math.ceil(elapsed_seconds * self.control_frequency_hz))


@dataclass(frozen=True)
class QueueMergeResult:
    """Summary of one fresh chunk merge."""

    old_depth: int
    new_depth: int
    blended_steps: int
    dropped_old_steps: int
    dropped_new_steps: int
    sequence_id: int


class ActionQueue:
    """Thread-safe queue of actions for RTC and non-RTC execution.

    The queue keeps the policy output used for RTC prefix guidance separate
    from the actions consumed by the control loop.  The first version applies
    no action-space postprocessing before :meth:`pop`, so the two queues store
    the same normalized chunk when RTC is enabled.  Keeping the two queues
    separate preserves the LeRobot contract and leaves room for a distinct
    execution-space postprocessor later.

    RTC mode follows LeRobot's replacement semantics: after the inference
    delay, the new chunk replaces the unexecuted queue and no hand-written
    overlap blend is applied.  Non-RTC mode retains this project's existing
    linear overlap blend behavior.
    """

    def __init__(
        self,
        *,
        overlap_steps: int = 10,
        max_depth: int = 100,
        rtc_enabled: bool = False,
    ) -> None:
        if overlap_steps < 0:
            raise ValueError("overlap_steps must be non-negative")
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if not isinstance(rtc_enabled, bool):
            raise ValueError("rtc_enabled must be a boolean")
        self.overlap_steps = overlap_steps
        self.max_depth = max_depth
        self.rtc_enabled = rtc_enabled
        self._original_actions: Deque[Tensor] = deque()
        self._execution_actions: Deque[Tensor] = deque()
        self._lock = threading.Lock()
        self._sequence_id = 0

    def set_rtc_enabled(self, enabled: bool) -> None:
        """Synchronize queue semantics with the policy server configuration.

        Queues are normally configured before they receive their seed chunk.
        If a caller changes the mode after seeding, promote the execution
        queue to the original queue because a previously blended original
        chunk cannot be reconstructed.
        """

        if not isinstance(enabled, bool):
            raise ValueError("rtc_enabled must be a boolean")
        with self._lock:
            if enabled and not self.rtc_enabled:
                self._original_actions = deque(
                    action.clone() for action in self._execution_actions
                )
            self.rtc_enabled = enabled

    def depth(self) -> int:
        with self._lock:
            return len(self._execution_actions)

    def clear(self) -> None:
        with self._lock:
            self._original_actions.clear()
            self._execution_actions.clear()

    def pop(self) -> Tensor | None:
        with self._lock:
            if not self._execution_actions:
                return None
            action = self._execution_actions.popleft().clone()
            # Keep the RTC source queue aligned with the execution queue so
            # `snapshot()` exposes only unexecuted original actions.
            if self._original_actions:
                self._original_actions.popleft()
            return action

    def snapshot(self, length: int | None = None, *, batched: bool = True) -> Tensor | None:
        with self._lock:
            source = self._original_actions if self.rtc_enabled else self._execution_actions
            if not source:
                return None
            actions = list(source)
            if length is not None:
                if length < 1:
                    raise ValueError("snapshot length must be positive")
                actions = actions[:length]
            result = torch.stack(actions, dim=0)
        return result.unsqueeze(0) if batched else result

    def seed(self, actions: Tensor) -> QueueMergeResult:
        """Set an initial queue without blending with previous actions."""

        return self.merge(actions, overlap_steps=0)

    def merge(
        self,
        actions: Tensor,
        *,
        overlap_steps: int | None = None,
        discard_prefix_steps: int = 0,
    ) -> QueueMergeResult:
        """Merge a fresh chunk into the currently unexecuted queue prefix.

        ``discard_prefix_steps`` removes actions that correspond to the time
        spent running inference.  In RTC mode the old suffix is replaced with
        the delayed new chunk.  In non-RTC mode the first
        ``min(old_depth, new_depth, overlap_steps)`` actions are linearly
        blended, which is the hand-written overlap behavior retained for that
        path.
        """

        raw_actions = _as_action_chunk(actions)
        if not isinstance(discard_prefix_steps, int) or discard_prefix_steps < 0:
            raise ValueError("discard_prefix_steps must be a non-negative integer")
        dropped_new_steps = min(discard_prefix_steps, raw_actions.shape[0])
        new_actions = raw_actions[dropped_new_steps:]
        requested_overlap = self.overlap_steps if overlap_steps is None else overlap_steps
        if requested_overlap < 0:
            raise ValueError("overlap_steps must be non-negative")
        with self._lock:
            old_actions = list(self._execution_actions)
            old_depth = len(old_actions)

            if self.rtc_enabled:
                # LeRobot RTC replacement semantics.  `original_actions` and
                # the execution queue are equal in this first version because
                # the controller applies the postprocessor after `pop()`.
                blend_steps = 0
                merged = [action.clone() for action in new_actions]
                original = [action.clone() for action in new_actions]
            else:
                # Preserve the existing ordinary overlap fusion for policies
                # without RTC.  The original queue remains available for
                # future mode synchronization but is not consumed by
                # `snapshot()` while RTC is disabled.
                blend_steps = min(requested_overlap, old_depth, new_actions.shape[0])
                if blend_steps:
                    old_prefix = torch.stack(old_actions[:blend_steps], dim=0)
                    new_prefix = new_actions[:blend_steps]
                    blend_weight = torch.arange(
                        1,
                        blend_steps + 1,
                        dtype=new_actions.dtype,
                        device=new_actions.device,
                    ).div_(blend_steps).unsqueeze(-1)
                    blended = (1.0 - blend_weight) * old_prefix.to(new_actions.device) + blend_weight * new_prefix
                    merged = [blended[index] for index in range(blend_steps)]
                    merged.extend(new_actions[blend_steps:])
                elif new_actions.shape[0]:
                    merged = [new_actions[index] for index in range(new_actions.shape[0])]
                else:
                    merged = []
                original = [action.clone() for action in new_actions]

            self._sequence_id += 1
            self._original_actions = deque(original[: self.max_depth])
            self._execution_actions = deque(merged[: self.max_depth])
            sequence_id = self._sequence_id
            new_depth = len(self._execution_actions)
        return QueueMergeResult(
            old_depth=old_depth,
            new_depth=new_depth,
            blended_steps=blend_steps,
            dropped_old_steps=max(0, old_depth - blend_steps),
            dropped_new_steps=dropped_new_steps,
            sequence_id=sequence_id,
        )


@dataclass
class GripperHysteresis:
    """Hysteresis state machine for one gripper action dimension."""

    action_index: int = 6
    low_threshold: float = -0.5
    high_threshold: float = 0.5
    polarity: str = "pending"
    _is_open: bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.action_index < 0:
            raise ValueError("action_index must be non-negative")
        if self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold must be below high_threshold")
        if self.polarity not in {"positive_open", "negative_open", "pending"}:
            raise ValueError("polarity must be positive_open, negative_open, or pending")

    def reset(self) -> None:
        self._is_open = None

    @property
    def state(self) -> bool | None:
        return self._is_open

    def apply(self, action: Tensor) -> Tensor:
        if action.ndim != 1:
            raise ValueError("gripper hysteresis expects one action vector")
        if self.action_index >= action.shape[0]:
            raise ValueError("gripper action index exceeds action dimension")
        output = action.clone()
        if self.polarity == "pending":
            return output
        value = float(output[self.action_index].item())
        if value >= self.high_threshold:
            self._is_open = True
        elif value <= self.low_threshold:
            self._is_open = False
        if self._is_open is not None:
            if self.polarity == "positive_open":
                output[self.action_index] = 1.0 if self._is_open else -1.0
            else:
                output[self.action_index] = -1.0 if self._is_open else 1.0
        return output


class PolicyCallable(Protocol):
    def __call__(
        self,
        observation: Any,
        *,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int,
        execution_horizon: int,
    ) -> Tensor:
        ...


class PreprocessedPolicyAdapter:
    """Adapt a LeRobot-style ``predict_action_chunk`` policy to this runtime."""

    def __init__(self, policy: Any, preprocessor: Callable[[Any], dict[str, Any]]) -> None:
        self.policy = policy
        self.preprocessor = preprocessor

    def __call__(
        self,
        observation: Any,
        *,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int,
        execution_horizon: int,
    ) -> Tensor:
        # LeRobot processors may add token and mask fields in place.  Keep the
        # control thread's observation object private from the worker thread.
        worker_observation = dict(observation) if isinstance(observation, dict) else observation
        batch = self.preprocessor(worker_observation)
        return self.policy.predict_action_chunk(
            batch,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )


class LeRobotPostprocessorAdapter:
    """Apply a LeRobot action postprocessor to one normalized action."""

    def __init__(self, postprocessor: Callable[[Tensor], Tensor]) -> None:
        self.postprocessor = postprocessor

    def __call__(self, action: Tensor) -> Tensor:
        if action.ndim != 1:
            raise ValueError("LeRobot postprocessor adapter expects one action vector")
        processed = self.postprocessor(action.unsqueeze(0))
        if not isinstance(processed, Tensor):
            raise TypeError("LeRobot postprocessor must return a torch.Tensor")
        if processed.ndim == 2 and processed.shape[0] == 1:
            processed = processed[0]
        if processed.ndim != 1:
            raise ValueError("LeRobot postprocessor must return shape [1, action_dim] or [action_dim]")
        if not torch.isfinite(processed).all().item():
            raise ValueError("LeRobot postprocessor returned non-finite values")
        return processed.detach().cpu()


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    observation: Any
    prev_chunk_left_over: Tensor | None
    inference_delay: int
    submitted_at: float
    submitted_control_step: int | None = None


@dataclass(frozen=True)
class InferenceEvent:
    request_id: int
    elapsed_seconds: float
    inference_delay: int
    resulting_delay: int
    queue_depth_before_merge: int
    queue_depth_at_completion: int
    dropped_new_steps: int
    deadline_miss: bool
    error: str | None = None
    wall_delay_steps: int | None = None


class AsyncPolicyServer:
    """Single-worker policy server with stale-request replacement."""

    def __init__(
        self,
        policy: PolicyCallable,
        action_queue: ActionQueue,
        config: AsyncRuntimeConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.action_queue = action_queue
        self.config = config
        # Keep the queue's replacement/append semantics aligned with the
        # policy's RTC setting.  Existing callers can continue constructing an
        # ActionQueue with only `overlap_steps`.
        self.action_queue.set_rtc_enabled(config.rtc_enabled)
        self.clock = clock
        self._requests: thread_queue.Queue[InferenceRequest | None] = thread_queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._inflight = False
        self._next_request_id = 0
        self._last_delay_steps = 0
        self._last_rtc_delay_steps = 0
        self._control_step = 0
        self._last_event: InferenceEvent | None = None
        self._events: list[InferenceEvent] = []
        self._submitted = 0
        self._completed = 0
        self._dropped_stale = 0
        self._thread = threading.Thread(target=self._worker_loop, name="smolvla-policy-worker", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._inflight or not self._requests.empty()

    @property
    def last_error(self) -> str | None:
        with self._state_lock:
            return self._last_event.error if self._last_event and self._last_event.error else None

    @property
    def last_delay_steps(self) -> int:
        with self._state_lock:
            return self._last_delay_steps

    def advance_control_step(self) -> int:
        """Record one control tick and return its monotonic index.

        The simulator or robot loop can run below the configured nominal
        frequency.  Passing this index with an inference request lets the
        worker measure delay in actions actually consumed, rather than
        converting wall-clock latency with an inaccurate nominal frequency.
        """

        with self._state_lock:
            self._control_step += 1
            return self._control_step

    def submit(self, observation: Any, *, control_step: int | None = None) -> bool:
        """Submit the newest observation, replacing one stale pending request."""

        with self._state_lock:
            self._next_request_id += 1
            request_id = self._next_request_id
            delay = self._last_rtc_delay_steps
        previous = (
            self.action_queue.snapshot(self.config.rtc_execution_horizon, batched=True)
            if self.config.rtc_enabled and self.config.rtc_use_prefix
            else None
        )
        request = InferenceRequest(
            request_id=request_id,
            observation=observation,
            prev_chunk_left_over=previous,
            inference_delay=delay if self.config.rtc_enabled else 0,
            submitted_at=self.clock(),
            submitted_control_step=control_step,
        )
        try:
            self._requests.put_nowait(request)
        except thread_queue.Full:
            try:
                self._requests.get_nowait()
            except thread_queue.Empty:
                return False
            with self._state_lock:
                self._dropped_stale += 1
            self._requests.put_nowait(request)
        with self._state_lock:
            self._submitted += 1
        return True

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        deadline = self.clock() + timeout
        while self.busy:
            if self.clock() >= deadline:
                return False
            time.sleep(0.001)
        return True

    def events(self) -> list[InferenceEvent]:
        with self._state_lock:
            return list(self._events)

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "dropped_stale": self._dropped_stale,
                "busy": self._inflight or not self._requests.empty(),
                "last_delay_steps": self._last_delay_steps,
                "last_rtc_delay_steps": self._last_rtc_delay_steps,
                "last_error": self._last_event.error if self._last_event else None,
            }

    def _current_control_step(self) -> int:
        with self._state_lock:
            return self._control_step

    def close(self, timeout: float = 5.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._requests.put_nowait(None)
        except thread_queue.Full:
            try:
                self._requests.get_nowait()
            except thread_queue.Empty:
                pass
            self._requests.put_nowait(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise RuntimeStateError("policy worker did not stop before timeout")

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                request = self._requests.get(timeout=0.05)
            except thread_queue.Empty:
                continue
            if request is None:
                return
            with self._state_lock:
                self._inflight = True
            started = self.clock()
            error = None
            merge_result: QueueMergeResult | None = None
            try:
                result = self.policy(
                    request.observation,
                    prev_chunk_left_over=request.prev_chunk_left_over,
                    inference_delay=request.inference_delay,
                    execution_horizon=self.config.rtc_execution_horizon,
                )
                elapsed = self.clock() - started
                wall_delay_steps = self.config.latency_to_steps(elapsed)
                queue_depth_before_merge = self.action_queue.depth()
                if request.submitted_control_step is None:
                    resulting_delay = wall_delay_steps
                else:
                    resulting_delay = max(0, self._current_control_step() - request.submitted_control_step)
                merge_result = self.action_queue.merge(result, discard_prefix_steps=resulting_delay)
            except Exception as exc:  # pragma: no cover - exercised through error state tests
                elapsed = max(0.0, self.clock() - started)
                wall_delay_steps = self.config.latency_to_steps(elapsed)
                queue_depth_before_merge = self.action_queue.depth()
                if request.submitted_control_step is None:
                    resulting_delay = wall_delay_steps
                else:
                    resulting_delay = max(0, self._current_control_step() - request.submitted_control_step)
                error = f"{type(exc).__name__}: {exc}"
            depth = self.action_queue.depth()
            event = InferenceEvent(
                request_id=request.request_id,
                elapsed_seconds=elapsed,
                inference_delay=request.inference_delay,
                resulting_delay=resulting_delay,
                queue_depth_before_merge=queue_depth_before_merge,
                queue_depth_at_completion=depth,
                dropped_new_steps=merge_result.dropped_new_steps if merge_result is not None else 0,
                deadline_miss=queue_depth_before_merge == 0 and error is None,
                error=error,
                wall_delay_steps=wall_delay_steps,
            )
            with self._state_lock:
                self._inflight = False
                self._last_delay_steps = resulting_delay
                self._last_rtc_delay_steps = wall_delay_steps
                self._last_event = event
                self._events.append(event)
                self._completed += 1


@dataclass(frozen=True)
class TickResult:
    action: Tensor | None
    triggered_inference: bool
    waiting_for_policy: bool
    queue_depth_before: int
    queue_depth_after: int
    inference_delay_steps: int


class AsyncClosedLoopController:
    """20 Hz control-side facade over the queue and policy worker."""

    def __init__(
        self,
        policy_server: AsyncPolicyServer,
        action_queue: ActionQueue,
        config: AsyncRuntimeConfig,
        *,
        gripper: GripperHysteresis | None = None,
        action_postprocessor: Callable[[Tensor], Tensor] | None = None,
        simulation_only: bool = True,
    ) -> None:
        self.policy_server = policy_server
        self.action_queue = action_queue
        self.config = config
        self.action_postprocessor = action_postprocessor
        self.gripper = gripper or GripperHysteresis(
            action_index=config.gripper_action_index,
            low_threshold=config.gripper_low_threshold,
            high_threshold=config.gripper_high_threshold,
            polarity=config.gripper_polarity,
        )
        if config.require_gripper_polarity and config.gripper_polarity == "pending" and not simulation_only:
            raise RuntimeStateError("gripper polarity is pending; hardware execution is disabled")

    def seed_actions(self, actions: Tensor) -> QueueMergeResult:
        return self.action_queue.seed(actions)

    def tick(self, observation: Any) -> TickResult:
        control_step = self.policy_server.advance_control_step()
        before = self.action_queue.depth()
        triggered = False
        if before <= self.config.trigger_threshold and not self.policy_server.busy:
            triggered = self.policy_server.submit(observation, control_step=control_step)
        action = self.action_queue.pop()
        waiting = action is None
        if action is not None:
            action = self.gripper.apply(action)
            if self.action_postprocessor is not None:
                action = self.action_postprocessor(action)
        return TickResult(
            action=action,
            triggered_inference=triggered,
            waiting_for_policy=waiting,
            queue_depth_before=before,
            queue_depth_after=self.action_queue.depth(),
            inference_delay_steps=self.policy_server.last_delay_steps,
        )

    def close(self) -> None:
        self.policy_server.close()


__all__ = [
    "ActionQueue",
    "AsyncClosedLoopController",
    "AsyncPolicyServer",
    "AsyncRuntimeConfig",
    "GripperHysteresis",
    "InferenceEvent",
    "InferenceRequest",
    "LeRobotPostprocessorAdapter",
    "PreprocessedPolicyAdapter",
    "QueueMergeResult",
    "RuntimeStateError",
    "TickResult",
]
