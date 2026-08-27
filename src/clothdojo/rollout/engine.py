"""One synchronous batched rollout loop for chunk and guided RTC execution."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Protocol

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
        *,
        inference_delay_steps: int = 0,
    ) -> np.ndarray: ...


class Episode(Protocol):
    """A synchronous batch of one or more simulation worlds."""

    @property
    def action_dim(self) -> int: ...

    @property
    def world_count(self) -> int: ...

    def observe(self) -> dict[str, np.ndarray]: ...

    def step(self, policy_actions: np.ndarray) -> None: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PhaseTiming:
    calls: int
    total_seconds: float
    mean_seconds: float
    max_seconds: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> PhaseTiming:
        total = sum(samples)
        return cls(
            calls=len(samples),
            total_seconds=total,
            mean_seconds=total / len(samples),
            max_seconds=max(samples),
        )

    def to_json_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.mean_seconds,
            "max_seconds": self.max_seconds,
        }


@dataclass(frozen=True, slots=True)
class RolloutTimings:
    """End-to-end wall timings with device completion at each execution window."""

    wall_seconds: float
    policy_session_reset: PhaseTiming
    observation: PhaseTiming
    policy_end_to_end: PhaseTiming
    execution: PhaseTiming
    phases_overlap: bool = False

    def to_json_dict(self, *, world_count: int, steps: int) -> dict[str, Any]:
        phase_total = (
            self.policy_session_reset.total_seconds
            + self.observation.total_seconds
            + self.policy_end_to_end.total_seconds
            + self.execution.total_seconds
        )
        world_steps = world_count * steps
        return {
            "schema": "clothdojo_episode_performance",
            "format_version": 1,
            "scope": "episode_core",
            "clock": "perf_counter",
            "synchronization": "after_each_execution_window",
            "phases_overlap": self.phases_overlap,
            "wall_seconds": self.wall_seconds,
            "phases": {
                "policy_session_reset": self.policy_session_reset.to_json_dict(),
                "observation": self.observation.to_json_dict(),
                "policy_end_to_end": self.policy_end_to_end.to_json_dict(),
                "execution": self.execution.to_json_dict(),
            },
            "unattributed_seconds": max(0.0, self.wall_seconds - phase_total),
            "overlapped_phase_seconds": max(0.0, phase_total - self.wall_seconds),
            "executed_world_steps_per_worker_second": world_steps / self.wall_seconds,
        }


@dataclass(frozen=True, slots=True)
class RolloutReport:
    mode: ExecutionMode
    steps: int
    inference_requests: int
    exec_horizon: int
    inference_delay_steps: int
    world_count: int
    timings: RolloutTimings

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "steps": self.steps,
            "inference_requests": self.inference_requests,
            "exec_horizon": self.exec_horizon,
            "inference_delay_steps": self.inference_delay_steps,
            "world_count": self.world_count,
            "performance": self.timings.to_json_dict(
                world_count=self.world_count,
                steps=self.steps,
            ),
        }


def run_episode(
    episode: Episode,
    policy: Policy,
    scenario: RolloutScenario,
    *,
    mode: ExecutionMode,
    exec_horizon: int,
    policy_seed: int,
    rtc_delay_steps: int = 0,
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
    if isinstance(rtc_delay_steps, bool) or not isinstance(rtc_delay_steps, int):
        raise TypeError("rtc_delay_steps must be an integer")
    if mode == "chunk" and rtc_delay_steps != 0:
        raise ValueError("chunk execution does not accept an RTC delay")
    if mode == "rtc" and not 0 <= rtc_delay_steps < exec_horizon:
        raise ValueError("RTC delay must be in [0, exec_horizon)")
    if mode == "rtc" and exec_horizon + rtc_delay_steps > scenario.action_horizon:
        raise ValueError("RTC horizon plus delay exceeds the model action horizon")
    if isinstance(policy_seed, bool) or not isinstance(policy_seed, int) or policy_seed < 0:
        raise ValueError("policy_seed must be a non-negative integer")
    world_count = int(episode.world_count)
    if world_count < 1:
        raise ValueError("episode world_count must be positive")

    # Exclude asynchronous reset/restore work submitted before the episode.
    episode.synchronize()
    wall_started = time.perf_counter()
    reset_started = time.perf_counter()
    policy.reset(policy_seed, mode=mode, exec_horizon=exec_horizon)
    reset_samples = [time.perf_counter() - reset_started]
    observation_samples: list[float] = []
    policy_samples: list[float] = []
    execution_samples: list[float] = []
    actions: np.ndarray | None = None
    action_index = 0
    requests = 0
    executed = 0

    def observe() -> dict[str, np.ndarray]:
        started = time.perf_counter()
        result = episode.observe()
        observation_samples.append(time.perf_counter() - started)
        return result

    def infer(observation: dict[str, np.ndarray], delay: int) -> np.ndarray:
        started = time.perf_counter()
        result = _validate_actions(
            policy.infer(
                observation,
                scenario.prompt,
                inference_delay_steps=delay,
            ),
            action_dim=episode.action_dim,
            expected_horizon=scenario.action_horizon,
            expected_batch=world_count,
        )
        policy_samples.append(time.perf_counter() - started)
        return result

    actions = infer(observe(), 0)
    requests += 1
    if mode == "chunk":
        while executed < scenario.steps:
            execution_started = time.perf_counter()
            count = min(exec_horizon, scenario.steps - executed)
            for _ in range(count):
                episode.step(actions[:, action_index].copy())
                action_index += 1
                executed += 1
            episode.synchronize()
            execution_samples.append(time.perf_counter() - execution_started)
            if executed < scenario.steps:
                actions = infer(observe(), 0)
                action_index = 0
                requests += 1
    else:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="clothdojo-rtc") as executor:
            execution_started = time.perf_counter()
            initial = min(exec_horizon, scenario.steps)
            for _ in range(initial):
                episode.step(actions[:, action_index].copy())
                action_index += 1
                executed += 1
            episode.synchronize()
            execution_samples.append(time.perf_counter() - execution_started)

            while executed < scenario.steps:
                execution_started = time.perf_counter()
                remaining = scenario.steps - executed
                if remaining <= rtc_delay_steps:
                    for _ in range(remaining):
                        episode.step(actions[:, action_index].copy())
                        action_index += 1
                        executed += 1
                    episode.synchronize()
                    execution_samples.append(time.perf_counter() - execution_started)
                    break

                future = executor.submit(infer, observe(), rtc_delay_steps)
                for _ in range(rtc_delay_steps):
                    episode.step(actions[:, action_index].copy())
                    action_index += 1
                    executed += 1
                actions = future.result()
                action_index = rtc_delay_steps
                requests += 1

                after_merge = min(
                    exec_horizon - rtc_delay_steps,
                    scenario.steps - executed,
                )
                for _ in range(after_merge):
                    episode.step(actions[:, action_index].copy())
                    action_index += 1
                    executed += 1
                episode.synchronize()
                execution_samples.append(time.perf_counter() - execution_started)

    wall_seconds = time.perf_counter() - wall_started
    return RolloutReport(
        mode=mode,
        steps=scenario.steps,
        inference_requests=requests,
        exec_horizon=exec_horizon,
        inference_delay_steps=rtc_delay_steps,
        world_count=world_count,
        timings=RolloutTimings(
            wall_seconds=wall_seconds,
            policy_session_reset=PhaseTiming.from_samples(reset_samples),
            observation=PhaseTiming.from_samples(observation_samples),
            policy_end_to_end=PhaseTiming.from_samples(policy_samples),
            execution=PhaseTiming.from_samples(execution_samples),
            phases_overlap=mode == "rtc" and rtc_delay_steps > 0,
        ),
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
        raise ValueError(f"policy actions must be finite with shape {expected}, got {result.shape}")
    return np.ascontiguousarray(result)
