from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clothdojo.rollout.engine import PolicyOutput, run_chunk_episode, run_rtc_episode
from clothdojo.rollout.newton import pack_policy_state, unpack_policy_action
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


def test_rtc_replans_synchronously_with_guidance_and_zero_delay() -> None:
    episode = _Episode()
    policy = _Policy()

    report = run_rtc_episode(
        episode,
        policy,
        _scenario(steps=5),
        exec_horizon=2,
    )

    assert policy.seed == 7
    assert [action[0] for action in episode.actions] == [0, 1, 100, 101, 200]
    assert [int(call[0]["observation.state"][0, 0]) for call in policy.calls] == [0, 2, 4]
    assert policy.calls[0][2] is None
    first_context = policy.calls[1][2]
    assert first_context.executed_horizon == 2
    assert first_context.predicted_delay_steps == 0
    assert first_context.chunk_id == 1
    assert first_context.previous_remaining.shape == (1, 2, 28)
    assert report.to_json_dict() == {
        "mode": "rtc",
        "steps": 5,
        "inference_requests": 3,
        "exec_horizon": 2,
    }


def test_chunk_baseline_uses_the_same_synchronous_cadence_without_guidance() -> None:
    episode = _Episode()
    policy = _Policy()

    report = run_chunk_episode(episode, policy, _scenario(steps=5), exec_horizon=2)

    assert policy.seed == 7
    assert [action[0] for action in episode.actions] == [0, 1, 100, 101, 200]
    assert all(call[2] is None for call in policy.calls)
    assert report.to_json_dict() == {
        "mode": "chunk",
        "steps": 5,
        "inference_requests": 3,
        "exec_horizon": 2,
    }


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


def test_project_rollout_scenario_is_strict_and_fingerprinted(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    scenario = load_rollout_scenario(project / "configs/rollout/shirt_flatten.yaml")

    assert scenario.action_horizon == 50
    assert len(scenario.fingerprint) == 64

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        (project / "configs/rollout/shirt_flatten.yaml")
        .read_text(encoding="utf-8")
        .replace("action_horizon: 50", "action_horizon: 0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive integer"):
        load_rollout_scenario(invalid)


def test_rtc_policy_rejects_wrong_action_horizon() -> None:
    class ShortPolicy(_Policy):
        def infer(self, observation, prompt, *, rtc=None):
            output = super().infer(observation, prompt, rtc=rtc)
            return replace(output, actions=output.actions[:, :3])

    with pytest.raises(ValueError, match="shape"):
        run_rtc_episode(
            _Episode(),
            ShortPolicy(),
            _scenario(),
            exec_horizon=2,
        )
