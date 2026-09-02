from dataclasses import replace

import pytest

from clothdojo.evaluation.results import (
    FORMAL_BENCHMARK_ASSET_SPLITS_SHA256,
    FORMAL_BENCHMARK_CASE_COUNT,
    FORMAL_BENCHMARK_INITIAL_STATES_SHA256,
    FORMAL_BENCHMARK_PANEL_ID,
    FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT,
    FORMAL_BENCHMARK_SCENARIO_FINGERPRINTS,
    FORMAL_BENCHMARK_SEEN_ASSETS,
    FORMAL_BENCHMARK_SELECTION_SHA256,
    FORMAL_BENCHMARK_STATES_PER_ASSET,
    FORMAL_BENCHMARK_UNSEEN_ASSETS,
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
            "case_count": FORMAL_BENCHMARK_CASE_COUNT,
            "initial_states_sha256": FORMAL_BENCHMARK_INITIAL_STATES_SHA256,
            "protocol": {"fingerprint": FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT},
            "scenario": actual.to_json_dict(),
            "benchmark": {
                "panel_id": FORMAL_BENCHMARK_PANEL_ID,
                "selection_sha256": FORMAL_BENCHMARK_SELECTION_SHA256,
                "asset_splits_sha256": FORMAL_BENCHMARK_ASSET_SPLITS_SHA256,
                "seen_assets": FORMAL_BENCHMARK_SEEN_ASSETS,
                "unseen_assets": FORMAL_BENCHMARK_UNSEEN_ASSETS,
                "states_per_asset": FORMAL_BENCHMARK_STATES_PER_ASSET,
            },
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
