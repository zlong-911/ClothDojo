"""One synchronous batched rollout loop for chunk and guided RTC execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from clothdojo.rollout.scenario import RolloutScenario

ExecutionMode = Literal["chunk", "rtc"]


class Policy(Protocol):
    """A batched action policy that owns any mode-specific inference state."""

    def reset(
        self,
        seed: int,
        *,
        mode: ExecutionMode,
        exec_horizon: int,
    ) -> None: ...

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
    ) -> np.ndarray: ...


class Episode(Protocol):
    """A synchronous batch of one or more simulation worlds."""

    @property
    def action_dim(self) -> int: ...

    @property
    def world_count(self) -> int: ...

    def observe(self) -> dict[str, np.ndarray]: ...

    def step(self, policy_actions: np.ndarray) -> None: ...


@dataclass(frozen=True, slots=True)
class RolloutReport:
    mode: ExecutionMode
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


def run_episode(
    episode: Episode,
    policy: Policy,
    scenario: RolloutScenario,
    *,
    mode: ExecutionMode,
    exec_horizon: int,
    policy_seed: int,
) -> RolloutReport:
    """Run chunk or RTC inference over a synchronized world batch.

    ``world_count == 1`` is the serial case. RTC history is intentionally not
    represented here: it belongs to the policy session that produces chunks.
    """

    if mode not in {"chunk", "rtc"}:
        raise ValueError("mode must be 'chunk' or 'rtc'")
    if not 1 <= exec_horizon <= scenario.action_horizon:
        raise ValueError("exec_horizon must be in [1, action_horizon]")
    if mode == "rtc" and exec_horizon == scenario.action_horizon:
        raise ValueError("RTC exec_horizon must be smaller than action_horizon")
    if isinstance(policy_seed, bool) or not isinstance(policy_seed, int) or policy_seed < 0:
        raise ValueError("policy_seed must be a non-negative integer")
    world_count = int(episode.world_count)
    if world_count < 1:
        raise ValueError("episode world_count must be positive")

    policy.reset(policy_seed, mode=mode, exec_horizon=exec_horizon)
    actions: np.ndarray | None = None
    action_index = exec_horizon
    requests = 0

    for _ in range(scenario.steps):
        if actions is None or action_index >= exec_horizon:
            actions = _validate_actions(
                policy.infer(episode.observe(), scenario.prompt),
                action_dim=episode.action_dim,
                expected_horizon=scenario.action_horizon,
                expected_batch=world_count,
            )
            action_index = 0
            requests += 1

        episode.step(actions[:, action_index].copy())
        action_index += 1

    return RolloutReport(
        mode=mode,
        steps=scenario.steps,
        inference_requests=requests,
        exec_horizon=exec_horizon,
    )


def _validate_actions(
    actions: np.ndarray,
    *,
    action_dim: int,
    expected_horizon: int,
    expected_batch: int,
) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32)
    expected = (expected_batch, expected_horizon, action_dim)
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(
            f"policy actions must be finite with shape {expected}, got {result.shape}"
        )
    return np.ascontiguousarray(result)
