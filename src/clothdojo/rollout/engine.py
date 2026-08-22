"""Small asynchronous RTC action-chunk execution engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from clothdojo.rollout.rtc import RTCScheduler
from clothdojo.rollout.scenario import RTCScenario


@dataclass(frozen=True, slots=True)
class RTCContext:
    previous_model_actions: np.ndarray
    executed_horizon: int
    predicted_delay_steps: int
    chunk_id: int

    @property
    def previous_remaining(self) -> np.ndarray:
        return self.previous_model_actions[:, self.executed_horizon :].copy()


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    actions: np.ndarray
    model_actions: np.ndarray | None = None


class Policy(Protocol):
    def reset(self, seed: int) -> None: ...

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        *,
        rtc: RTCContext | None = None,
    ) -> PolicyOutput: ...


class Episode(Protocol):
    @property
    def action_dim(self) -> int: ...

    def observe(self) -> dict[str, np.ndarray]: ...

    def step(self, policy_action: np.ndarray) -> None: ...


class _Future(Protocol):
    def done(self) -> bool: ...

    def result(self) -> PolicyOutput: ...


class _Executor(Protocol):
    def submit(self, function, /, *args, **kwargs) -> _Future: ...


@dataclass(frozen=True, slots=True)
class RTCRolloutReport:
    steps: int
    inference_requests: int
    resolved_replans: int
    discarded_final_replan: bool
    observed_delay_steps: tuple[int, ...]
    held_action_steps: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "inference_requests": self.inference_requests,
            "resolved_replans": self.resolved_replans,
            "discarded_final_replan": self.discarded_final_replan,
            "observed_delay_steps": list(self.observed_delay_steps),
            "held_action_steps": self.held_action_steps,
        }


def run_rtc_episode(
    episode: Episode,
    policy: Policy,
    scenario: RTCScenario,
    *,
    executor: _Executor | None = None,
) -> RTCRolloutReport:
    """Execute one fixed-length scenario while overlapping replanning."""

    policy.reset(scenario.seed)
    initial = _validate_output(
        policy.infer(episode.observe(), scenario.prompt),
        action_dim=episode.action_dim,
        expected_horizon=scenario.action_horizon,
    )
    if executor is None:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="clothdojo-rtc") as pool:
            return _run(episode, policy, scenario, initial, pool)
    return _run(episode, policy, scenario, initial, executor)


def _run(
    episode: Episode,
    policy: Policy,
    scenario: RTCScenario,
    initial: PolicyOutput,
    executor: _Executor,
) -> RTCRolloutReport:
    scheduler = RTCScheduler(
        min_exec_horizon=scenario.rtc.min_exec_horizon,
        initial_delay_steps=scenario.rtc.initial_delay_steps,
        delay_buffer_size=scenario.rtc.delay_buffer_size,
    )
    actions = initial.actions
    model_actions = initial.model_actions if initial.model_actions is not None else actions
    action_index = 0
    chunk_id = 0
    requests = 1
    resolved_replans = 0
    held_steps = 0
    delays: list[int] = []
    pending: _Future | None = None

    for step in range(scenario.steps):
        if pending is not None and pending.done():
            output = _validate_output(
                pending.result(),
                action_dim=episode.action_dim,
                expected_horizon=scenario.action_horizon,
            )
            delay = scheduler.resolve(step)
            delays.append(delay)
            resolved_replans += 1
            pending = None
            chunk_id += 1
            actions = output.actions
            model_actions = output.model_actions if output.model_actions is not None else actions
            action_index = min(delay, actions.shape[1] - 1)

        if scheduler.should_request(action_index, actions.shape[1]):
            context = RTCContext(
                previous_model_actions=model_actions.copy(),
                executed_horizon=action_index,
                predicted_delay_steps=scheduler.predicted_delay_steps,
                chunk_id=chunk_id + 1,
            )
            observation = _copy_observation(episode.observe())
            pending = executor.submit(
                policy.infer,
                observation,
                scenario.prompt,
                rtc=context,
            )
            scheduler.mark_requested(step)
            requests += 1

        selected = min(action_index, actions.shape[1] - 1)
        held_steps += int(action_index >= actions.shape[1])
        episode.step(actions[0, selected].copy())
        action_index += 1

    discarded = pending is not None
    if pending is not None:
        _validate_output(
            pending.result(),
            action_dim=episode.action_dim,
            expected_horizon=scenario.action_horizon,
        )
    return RTCRolloutReport(
        steps=scenario.steps,
        inference_requests=requests,
        resolved_replans=resolved_replans,
        discarded_final_replan=discarded,
        observed_delay_steps=tuple(delays),
        held_action_steps=held_steps,
    )


def _validate_output(
    output: PolicyOutput,
    *,
    action_dim: int,
    expected_horizon: int,
) -> PolicyOutput:
    actions = np.asarray(output.actions, dtype=np.float32)
    expected = (1, expected_horizon, action_dim)
    if actions.shape != expected or not np.isfinite(actions).all():
        raise ValueError(
            f"RTC policy actions must be finite with shape {expected}, got {actions.shape}"
        )
    model_actions = output.model_actions
    if model_actions is not None:
        model_actions = np.asarray(model_actions, dtype=np.float32)
        if (
            model_actions.ndim != 3
            or model_actions.shape[:2] != actions.shape[:2]
            or not np.isfinite(model_actions).all()
        ):
            raise ValueError("RTC model_actions must be finite and match action batch/horizon")
    return PolicyOutput(
        np.ascontiguousarray(actions),
        None if model_actions is None else np.ascontiguousarray(model_actions),
    )


def _copy_observation(value: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(value, dict) or "observation.state" not in value:
        raise ValueError("RTC observation must contain observation.state")
    return {key: np.asarray(item).copy() for key, item in value.items()}
