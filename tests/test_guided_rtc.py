from __future__ import annotations

import math

import pytest
import torch

from clothdojo.rollout.guided_rtc import (
    GuidedRTCConfig,
    guided_velocity,
    prefix_weights,
)


def test_exponential_prefix_weights_match_lerobot() -> None:
    result = prefix_weights(2, 6, 8, "exp")
    linear = torch.tensor([1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0])
    expected = linear * torch.expm1(linear) / (math.e - 1.0)

    torch.testing.assert_close(result, expected)


def test_guided_velocity_matches_lerobot_upstream_direct_correction() -> None:
    x_t = torch.tensor([[[2.0]]])
    previous = torch.tensor([[[5.0]]])
    config = GuidedRTCConfig(
        execution_horizon=1,
        max_guidance_weight=5.0,
        prefix_attention_schedule="ones",
    )

    result = guided_velocity(
        x_t,
        previous,
        inference_delay=0,
        time=torch.tensor([0.5]),
        denoise=lambda latent: 0.5 * latent,
        config=config,
    )

    # x1 = x - 0.5 * (0.5*x) = 1.5; direct error = 3.5;
    # the LeRobot guidance weight at t=0.5 is 2.
    assert result.item() == pytest.approx(1.0 - 2.0 * 3.5)


def test_guided_rtc_metadata_names_the_algorithm() -> None:
    assert GuidedRTCConfig().to_json_dict() == {
        "implementation": "lerobot_guided_upstream",
        "execution_horizon": 10,
        "max_guidance_weight": 10.0,
        "prefix_attention_schedule": "linear",
        "delta_rebase": True,
    }
