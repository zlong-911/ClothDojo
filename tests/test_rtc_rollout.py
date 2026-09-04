from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.rollout.engine import run_episode
from clothdojo.rollout.newton import (
    NewtonBatchEnvironment,
    pack_policy_state,
    unpack_policy_actions,
)
from clothdojo.rollout.scenario import RolloutScenario, load_rollout_scenario


def _scenario(*, steps: int = 6) -> RolloutScenario:
    return RolloutScenario(
        scenario_id="test",
        task_name="shirt_flatten",
        garment_id="garment",
        seed=7,
        prompt="Flatten the shirt.",
        steps=steps,
        action_horizon=4,
    )


class _Episode:
    action_dim = 14

    def __init__(self, world_count: int = 3) -> None:
        self.world_count = world_count
        self.actions: list[np.ndarray] = []
        self.step_index = 0
        self.synchronize_calls = 0

    def observe(self) -> dict[str, np.ndarray]:
        values = np.arange(self.world_count, dtype=np.float32)[:, None]
        return {"observation.state": np.repeat(values + self.step_index, 14, axis=1)}

    def step(self, policy_actions: np.ndarray) -> None:
        self.actions.append(np.asarray(policy_actions).copy())
        self.step_index += 1

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _Policy:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, np.ndarray], str, int]] = []
        self.reset_args: tuple[int, str, int] | None = None

    def reset(self, seed: int, *, mode: str, exec_horizon: int) -> None:
        self.reset_args = (seed, mode, exec_horizon)

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        *,
        inference_delay_steps: int = 0,
    ) -> np.ndarray:
        call = len(self.calls)
        self.calls.append((observation, prompt, inference_delay_steps))
        batch = observation["observation.state"].shape[0]
        steps = np.arange(4, dtype=np.float32) + call * 100
        actions = np.broadcast_to(steps[None, :, None], (batch, 4, 14)).copy()
        actions += np.arange(batch, dtype=np.float32)[:, None, None] * 10
        return actions


@pytest.mark.parametrize("mode", ["chunk", "rtc"])
def test_one_runner_uses_the_same_synchronous_batch_cadence(mode: str) -> None:
    episode = _Episode()
    policy = _Policy()

    report = run_episode(
        episode,
        policy,
        _scenario(steps=5),
        mode=mode,
        exec_horizon=2,
        policy_seed=123,
    )

    assert policy.reset_args == (123, mode, 2)
    assert [int(call[0]["observation.state"][0, 0]) for call in policy.calls] == [0, 2, 4]
    np.testing.assert_array_equal(episode.actions[0][:, 0], [0, 10, 20])
    np.testing.assert_array_equal(episode.actions[1][:, 0], [1, 11, 21])
    np.testing.assert_array_equal(episode.actions[2][:, 0], [100, 110, 120])
    payload = report.to_json_dict()
    assert {
        key: payload[key]
        for key in ("mode", "steps", "inference_requests", "exec_horizon", "world_count")
    } == {
        "mode": mode,
        "steps": 5,
        "inference_requests": 3,
        "exec_horizon": 2,
        "world_count": 3,
    }
    assert payload["performance"]["wall_seconds"] > 0.0
    assert payload["performance"]["executed_world_steps_per_worker_second"] > 0.0
    assert payload["performance"]["phases"]["observation"]["calls"] == 3
    assert payload["performance"]["phases"]["policy_end_to_end"]["calls"] == 3
    assert payload["performance"]["phases"]["execution"]["calls"] == 3
    assert episode.synchronize_calls == 4


def test_rtc_fixed_delay_executes_old_actions_while_replanning() -> None:
    episode = _Episode(world_count=1)
    policy = _Policy()

    report = run_episode(
        episode,
        policy,
        _scenario(steps=5),
        mode="rtc",
        exec_horizon=2,
        rtc_delay_steps=1,
        policy_seed=123,
    )

    assert [call[2] for call in policy.calls] == [0, 1]
    assert [int(call[0]["observation.state"][0, 0]) for call in policy.calls] == [0, 2]
    np.testing.assert_array_equal(
        np.asarray(episode.actions)[:, 0, 0],
        [0, 1, 2, 101, 102],
    )
    assert report.inference_delay_steps == 1
    assert report.timings.phases_overlap is True


def test_rtc_fixed_delay_trajectory_is_independent_of_policy_wall_latency() -> None:
    trajectories = []
    for latency in (0.0, 0.02):
        class DelayedPolicy(_Policy):
            def __init__(self, delay: float) -> None:
                super().__init__()
                self.delay = delay

            def infer(self, observation, prompt, *, inference_delay_steps=0):
                if self.calls:
                    time.sleep(self.delay)
                return super().infer(
                    observation,
                    prompt,
                    inference_delay_steps=inference_delay_steps,
                )

        episode = _Episode(world_count=1)
        run_episode(
            episode,
            DelayedPolicy(latency),
            _scenario(steps=5),
            mode="rtc",
            exec_horizon=2,
            rtc_delay_steps=1,
            policy_seed=123,
        )
        trajectories.append(np.asarray(episode.actions))

    np.testing.assert_array_equal(trajectories[0], trajectories[1])


def test_world_count_one_is_the_serial_case_of_the_same_runner() -> None:
    episode = _Episode(world_count=1)

    report = run_episode(
        episode,
        _Policy(),
        _scenario(steps=3),
        mode="chunk",
        exec_horizon=2,
        policy_seed=7,
    )

    assert all(action.shape == (1, 14) for action in episode.actions)
    np.testing.assert_array_equal(
        np.asarray(episode.actions)[:, 0, 0],
        [0, 1, 100],
    )
    assert report.inference_requests == 2


def test_policy_layout_matches_aloha_order() -> None:
    arms = np.arange(12, dtype=np.float32).reshape(1, 12)
    grippers = np.asarray([[0.25, 0.75]], dtype=np.float32)

    state = pack_policy_state(arms, grippers)
    unpacked_arms, unpacked_grippers = unpack_policy_actions(state)

    np.testing.assert_array_equal(
        state,
        [[0, 1, 2, 3, 4, 5, 0.25, 6, 7, 8, 9, 10, 11, 0.75]],
    )
    np.testing.assert_array_equal(unpacked_arms, arms)
    np.testing.assert_array_equal(unpacked_grippers, grippers)


def test_rollout_forwards_unclipped_actions_to_runtime() -> None:
    class Buffer:
        def assign(self, values: np.ndarray) -> None:
            self.values = np.asarray(values).copy()

    class Runtime:
        model = SimpleNamespace(world_count=1)

        def step(self, arms: Buffer, grippers: Buffer) -> None:
            self.received = (arms.values.copy(), grippers.values.copy())

    runtime = Runtime()
    environment = NewtonBatchEnvironment.__new__(NewtonBatchEnvironment)
    environment.runtime = runtime
    environment._sample_index = 0
    environment._cached_observation = None
    environment._arm_device = Buffer()
    environment._gripper_device = Buffer()
    actions = np.asarray(
        [[-4, -3, -2, -1, 0, 1, -0.25, 2, 3, 4, 5, 6, 7, 1.25]],
        dtype=np.float32,
    )

    arms, grippers = environment.step(actions)

    expected_arms, expected_grippers = unpack_policy_actions(actions)
    np.testing.assert_array_equal(arms, expected_arms)
    np.testing.assert_array_equal(grippers, expected_grippers)
    np.testing.assert_array_equal(runtime.received[0], expected_arms)
    np.testing.assert_array_equal(runtime.received[1], expected_grippers)
    assert environment.sample_index == 1


def test_project_rollout_scenario_is_strict_and_fingerprinted(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    scenario = load_rollout_scenario(project / "configs/rollout/shirt_flatten.yaml")
    fold = load_rollout_scenario(project / "configs/rollout/shirt_fold.yaml")

    assert scenario.action_horizon == 50
    assert len(scenario.fingerprint) == 64
    assert fold.task_name == "shirt_fold"
    assert fold.prompt == "Fold the shirt."
    assert fold.steps == 3600

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        (project / "configs/rollout/shirt_flatten.yaml")
        .read_text(encoding="utf-8")
        .replace("action_horizon: 50", "action_horizon: 0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive integer"):
        load_rollout_scenario(invalid)

    unsupported = tmp_path / "unsupported.yaml"
    unsupported.write_text(
        (project / "configs/rollout/shirt_flatten.yaml")
        .read_text(encoding="utf-8")
        .replace("task_name: shirt_flatten", "task_name: shirt_iron"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown task profile"):
        load_rollout_scenario(unsupported)


def test_runner_rejects_wrong_policy_action_horizon() -> None:
    class ShortPolicy(_Policy):
        def infer(self, observation, prompt, *, inference_delay_steps=0):
            return super().infer(
                observation,
                prompt,
                inference_delay_steps=inference_delay_steps,
            )[:, :3]

    with pytest.raises(ValueError, match="shape"):
        run_episode(
            _Episode(),
            ShortPolicy(),
            _scenario(),
            mode="rtc",
            exec_horizon=2,
            policy_seed=7,
        )
