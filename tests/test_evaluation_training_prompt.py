from dataclasses import replace

import pytest

from clothdojo.evaluation.results import (
    FORMAL_BENCHMARK_INITIAL_STATES_SHA256,
    FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT,
    FORMAL_BENCHMARK_SCENARIO_FINGERPRINTS,
    validate_formal_evaluation_metadata,
)
from clothdojo.rollout.scenario import RolloutScenario


def scenario(prompt):
    return RolloutScenario("shirt_flatten_v1", "shirt_flatten", None, 42, prompt, 4500, 50)


@pytest.mark.parametrize("prompt", ["flatten the shirt", "Flatten the shirt."])
def test_formal_panel_accepts_both_recorded_training_prompts(prompt):
    actual = scenario(prompt)
    assert actual.fingerprint in FORMAL_BENCHMARK_SCENARIO_FINGERPRINTS
    validate_formal_evaluation_metadata(
        {
            "case_count": 96,
            "initial_states_sha256": FORMAL_BENCHMARK_INITIAL_STATES_SHA256,
            "protocol": {"fingerprint": FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT},
            "scenario": actual.to_json_dict(),
        }
    )
    assert actual.to_json_dict()["prompt"] == prompt


def test_training_prompt_support_preserves_scenario_identity():
    original = scenario("flatten the shirt")
    assert original.fingerprint != scenario("Flatten the shirt.").fingerprint
    for changed in (
        replace(original, steps=4499),
        replace(original, seed=43),
        replace(original, action_horizon=10),
        replace(original, prompt="fold the shirt"),
    ):
        assert changed.fingerprint not in FORMAL_BENCHMARK_SCENARIO_FINGERPRINTS
