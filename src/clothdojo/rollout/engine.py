"""Synchronous action-chunk and guided RTC execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from clothdojo.rollout.scenario import RolloutScenario


@dataclass(frozen=True, slots=True)
class RTCContext:
    """Previous model-space chunk and delay used for RTC inpainting."""

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


@dataclass(frozen=True, slots=True)
class RolloutReport:
    mode: Literal["chunk", "rtc"]
    steps: int
    inference_requests: int
    exec_horizon: int

    def to_json_dict(self) -> dict[str, int | str]:
        return {
            "mode": self.mode,
            "steps": self.steps,
            "inference_requests": self.inference_requests,
            "exec_horizon": self.exec_horizon,
        }


def run_chunk_episode(
    episode: Episode,
    policy: Policy,
    scenario: RolloutScenario,
    *,
    exec_horizon: int,
) -> RolloutReport:
    """Synchronously replan without carrying information across chunks."""

    return _run_episode(
        episode,
        policy,
        scenario,
        exec_horizon=exec_horizon,
        guided=False,
    )


def run_rtc_episode(
    episode: Episode,
    policy: Policy,
    scenario: RolloutScenario,
    *,
    exec_horizon: int,
) -> RolloutReport:
    """Synchronously replan with guided overlap and zero inference delay."""

    return _run_episode(
        episode,
        policy,
        scenario,
        exec_horizon=exec_horizon,
        guided=True,
    )


def _run_episode(
    episode: Episode,
    policy: Policy,
    scenario: RolloutScenario,
    *,
    exec_horizon: int,
    guided: bool,
) -> RolloutReport:
    if not 1 <= exec_horizon <= scenario.action_horizon:
        raise ValueError("exec_horizon must be in [1, action_horizon]")

    policy.reset(scenario.seed)
    actions: np.ndarray | None = None
    model_actions: np.ndarray | None = None
    action_index = exec_horizon
    requests = 0
    chunk_id = 0

    for _ in range(scenario.steps):
        if actions is None or action_index >= exec_horizon:
            rtc = None
            if guided and model_actions is not None:
                rtc = RTCContext(
                    previous_model_actions=model_actions.copy(),
                    executed_horizon=action_index,
                    predicted_delay_steps=0,
                    chunk_id=chunk_id,
                )
            output = _validate_output(
                policy.infer(episode.observe(), scenario.prompt, rtc=rtc),
                action_dim=episode.action_dim,
                expected_horizon=scenario.action_horizon,
            )
            actions = output.actions
            model_actions = (
                output.model_actions if output.model_actions is not None else actions
            )
            action_index = 0
            requests += 1
            chunk_id += 1

        episode.step(actions[0, action_index].copy())
        action_index += 1

    return RolloutReport(
        mode="rtc" if guided else "chunk",
        steps=scenario.steps,
        inference_requests=requests,
        exec_horizon=exec_horizon,
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
            f"policy actions must be finite with shape {expected}, got {actions.shape}"
        )
    model_actions = output.model_actions
    if model_actions is not None:
        model_actions = np.asarray(model_actions, dtype=np.float32)
        if (
            model_actions.ndim != 3
            or model_actions.shape[:2] != actions.shape[:2]
            or not np.isfinite(model_actions).all()
        ):
            raise ValueError("model_actions must be finite and match action batch/horizon")
    return PolicyOutput(
        np.ascontiguousarray(actions),
        None if model_actions is None else np.ascontiguousarray(model_actions),
    )
