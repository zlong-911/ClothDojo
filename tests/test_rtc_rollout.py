from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clothdojo.rollout.engine import PolicyOutput, run_rtc_episode
from clothdojo.rollout.newton import pack_policy_state, unpack_policy_action
from clothdojo.rollout.scenario import RTCScenario, RTCSettings, load_rtc_scenario


def _scenario(*, steps: int = 6) -> RTCScenario:
    return RTCScenario(
        scenario_id="test",
        task_name="shirt_flatten",
        garment_id="garment",
        seed=7,
        prompt="Flatten the shirt.",
        steps=steps,
        action_horizon=4,
        rtc=RTCSettings(
            min_exec_horizon=2,
            initial_delay_steps=1,
            delay_buffer_size=8,
        ),
    )


class _Episode:
    action_dim = 14

    def __init__(self) -> None:
        self.actions = []
        self.step_index = 0

    def observe(self):
        return {"observation.state": np.full((1, 14), self.step_index, dtype=np.float32)}

    def step(self, policy_action):
        self.actions.append(np.asarray(policy_action).copy())
        self.step_index += 1


class _Policy:
    def __init__(self) -> None:
        self.calls = []
        self.seed = None

    def reset(self, seed):
        self.seed = seed

    def infer(self, observation, prompt, *, rtc=None):
        call = len(self.calls)
        self.calls.append((observation, prompt, rtc))
        steps = np.arange(4, dtype=np.float32) + call * 100
        actions = np.repeat(steps[None, :, None], 14, axis=2)
        model_actions = np.repeat(actions, 2, axis=2)
        return PolicyOutput(actions=actions, model_actions=model_actions)


class _ImmediateFuture:
    def __init__(self, value) -> None:
        self.value = value

    def done(self):
        return True

    def result(self):
        return self.value


class _ImmediateExecutor:
    def submit(self, function, /, *args, **kwargs):
        return _ImmediateFuture(function(*args, **kwargs))


class _SlowFuture(_ImmediateFuture):
    def done(self):
        return False


class _SlowExecutor(_ImmediateExecutor):
    def submit(self, function, /, *args, **kwargs):
        return _SlowFuture(function(*args, **kwargs))


def test_rtc_replans_with_observed_delay_and_model_overlap() -> None:
    episode = _Episode()
    policy = _Policy()

    report = run_rtc_episode(
        episode,
        policy,
        _scenario(steps=5),
        executor=_ImmediateExecutor(),
    )

    assert policy.seed == 7
    assert [action[0] for action in episode.actions] == [0, 1, 2, 101, 102]
    assert report.inference_requests == 3
    assert report.resolved_replans == 1
    assert report.observed_delay_steps == (1,)
    assert report.discarded_final_replan is True
    first_context = policy.calls[1][2]
    assert first_context.executed_horizon == 2
    assert first_context.predicted_delay_steps == 1
    assert first_context.chunk_id == 1
    assert first_context.previous_remaining.shape == (1, 2, 28)


def test_rtc_holds_last_action_while_replan_is_in_flight() -> None:
    episode = _Episode()
    report = run_rtc_episode(
        episode,
        _Policy(),
        _scenario(),
        executor=_SlowExecutor(),
    )

    assert [action[0] for action in episode.actions] == [0, 1, 2, 3, 3, 3]
    assert report.inference_requests == 2
    assert report.resolved_replans == 0
    assert report.held_action_steps == 2
    assert report.discarded_final_replan is True


def test_policy_layout_matches_aloha_order() -> None:
    arms = np.arange(12, dtype=np.float32).reshape(1, 12)
    grippers = np.asarray([[0.25, 0.75]], dtype=np.float32)

    state = pack_policy_state(arms, grippers)
    unpacked_arms, unpacked_grippers = unpack_policy_action(state[0])

    np.testing.assert_array_equal(
        state,
        [[0, 1, 2, 3, 4, 5, 0.25, 6, 7, 8, 9, 10, 11, 0.75]],
    )
    np.testing.assert_array_equal(unpacked_arms, arms)
    np.testing.assert_array_equal(unpacked_grippers, grippers)


def test_project_rtc_scenario_is_strict_and_fingerprinted(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    scenario = load_rtc_scenario(project / "configs/rollout/shirt_flatten_rtc_gate.yaml")

    assert scenario.action_horizon == 4
    assert scenario.rtc.min_exec_horizon == 2
    assert len(scenario.fingerprint) == 64

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        (project / "configs/rollout/shirt_flatten_rtc_gate.yaml")
        .read_text(encoding="utf-8")
        .replace("min_exec_horizon: 2", "min_exec_horizon: 4"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="smaller than action_horizon"):
        load_rtc_scenario(invalid)


def test_rtc_policy_rejects_wrong_action_horizon() -> None:
    class ShortPolicy(_Policy):
        def infer(self, observation, prompt, *, rtc=None):
            output = super().infer(observation, prompt, rtc=rtc)
            return replace(output, actions=output.actions[:, :3])

    with pytest.raises(ValueError, match="shape"):
        run_rtc_episode(_Episode(), ShortPolicy(), _scenario())
