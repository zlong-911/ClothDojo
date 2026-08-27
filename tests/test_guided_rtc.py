from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from clothdojo.rollout.guided_rtc import (
    GuidedRTCConfig,
    guided_velocity,
    prefix_weights,
    sample_guided_cfg_actions,
    sample_guided_openpi_rlinf_actions,
)


def test_exponential_prefix_weights_match_lerobot() -> None:
    result = prefix_weights(2, 6, 8, "exp")
    linear = torch.tensor([1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0])
    expected = linear * torch.expm1(linear) / (math.e - 1.0)

    torch.testing.assert_close(result, expected)


def test_guided_velocity_uses_the_full_denoising_vjp() -> None:
    x_t = torch.tensor([[[2.0]]])
    previous = torch.tensor([[[5.0]]])
    config = GuidedRTCConfig(
        guidance_horizon=1,
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

    # x1 = x - 0.5 * (0.5*x) = 1.5, so dx1/dx = 0.75.  The
    # target error is 3.5 and the RTC guidance weight at t=0.5 is 2.
    assert result.item() == pytest.approx(1.0 - 2.0 * (0.75 * 3.5))


def test_guided_velocity_keeps_batch_vjps_independent() -> None:
    x_t = torch.tensor([[[1.0]], [[2.0]]])
    previous = torch.tensor([[[3.0]], [[7.0]]])

    result = guided_velocity(
        x_t,
        previous,
        inference_delay=0,
        time=torch.tensor([0.5, 0.5]),
        denoise=lambda latent: latent.square(),
        config=GuidedRTCConfig(
            guidance_horizon=1,
            max_guidance_weight=5.0,
            prefix_attention_schedule="ones",
        ),
    )

    # Per sample: x1=x-0.5*x^2, dx1/dx=1-x.  Summing the VJP over the
    # batch is still sample-independent because the denoiser has no batch mix.
    expected = torch.tensor([[[1.0]], [[18.0]]])
    torch.testing.assert_close(result, expected)


def test_guided_velocity_zero_pads_nonphysical_model_dimensions() -> None:
    result = guided_velocity(
        torch.ones((1, 1, 3)),
        torch.full((1, 1, 1), 2.0),
        inference_delay=0,
        time=torch.tensor([0.5]),
        denoise=torch.zeros_like,
        config=GuidedRTCConfig(
            guidance_horizon=1,
            max_guidance_weight=5.0,
            prefix_attention_schedule="ones",
        ),
    )

    # The one physical dimension follows the previous action. The two padded
    # model dimensions follow their training target, zero, rather than an old
    # unconstrained latent prediction.
    torch.testing.assert_close(result, torch.tensor([[[-2.0, 2.0, 2.0]]]))


def test_guided_rtc_metadata_names_the_algorithm() -> None:
    assert GuidedRTCConfig().to_json_dict() == {
        "implementation": "physical_intelligence_guided_vjp",
        "guidance_horizon": 10,
        "max_guidance_weight": 10.0,
        "prefix_attention_schedule": "linear",
        "delta_rebase": True,
    }


def test_openpi_rlinf_rtc_uses_one_prefix_cache_and_requested_steps(
    monkeypatch,
) -> None:
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module

    monkeypatch.setattr(
        model_module,
        "preprocess_observation",
        lambda observation, *, train: observation,
    )

    class Core:
        action_horizon = 4
        action_dim = 1

        def __init__(self) -> None:
            self.cache_calls = 0
            self.suffix_calls = 0

        def build_prefix_cache(self, observation):
            del observation
            self.cache_calls += 1
            return None, torch.ones((2, 1), dtype=torch.bool), object()

        def run_suffix(self, observation, latent, time, cache, prefix_mask):
            del observation, time, cache, prefix_mask
            self.suffix_calls += 1
            return latent * 0.25

        def velocity_from_suffix(self, suffix):
            return suffix

    core = Core()
    result = sample_guided_openpi_rlinf_actions(
        core,
        object(),
        torch.zeros((2, 3, 1)),
        inference_delay=0,
        num_steps=2,
        config=GuidedRTCConfig(
            guidance_horizon=1,
            prefix_attention_schedule="zeros",
        ),
    )

    assert result.shape == (2, 4, 1)
    assert torch.isfinite(result).all()
    assert core.cache_calls == 1
    assert core.suffix_calls == 2


class _FakePaligemma:
    def __init__(self) -> None:
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(config=SimpleNamespace())
        )

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        return None, inputs_embeds[0]


class _FakeCFGModel:
    def __init__(self, scale: float, batch_size: int = 1) -> None:
        self.batch_size = batch_size
        self.config = SimpleNamespace(
            guidance_type="positive",
            positive_only_conditional=False,
            cfgrl_guidance_scale=scale,
            num_steps=1,
            action_horizon=1,
            action_dim=1,
        )
        self.paligemma_with_expert = _FakePaligemma()

    def _preprocess_observation(self, observation, *, train):
        del observation, train
        image = [torch.zeros((self.batch_size, 1, 1, 1))]
        image_mask = [torch.ones((self.batch_size, 1), dtype=torch.bool)]
        token_mask = torch.ones((self.batch_size, 1), dtype=torch.bool)
        return (
            image,
            image_mask,
            torch.ones((self.batch_size, 1)),
            token_mask,
            torch.full((self.batch_size, 1), 3.0),
            token_mask,
            torch.full((self.batch_size, 1), -1.0),
            token_mask,
            torch.zeros((self.batch_size, 1)),
        )

    def embed_prefix(self, images, image_masks, tokens, masks):
        del images, image_masks
        return tokens[..., None], masks, torch.ones_like(masks)

    def _prepare_attention_masks_4d(self, value):
        return value

    def sample_noise(self, shape, device):
        return torch.ones(shape, device=device)

    def denoise_step(self, state, masks, cache, latent, time):
        del state, masks, time
        return cache * latent


@pytest.mark.parametrize(("scale", "expected"), [(0.0, 0.0), (1.0, -2.0)])
def test_cfg_scale_selects_unconditional_or_positive_velocity(
    scale: float,
    expected: float,
) -> None:
    result = sample_guided_cfg_actions(
        _FakeCFGModel(scale),
        object(),
        torch.full((1, 1, 1), 2.0),
        inference_delay=0,
        config=GuidedRTCConfig(
            guidance_horizon=1,
            max_guidance_weight=5.0,
            prefix_attention_schedule="zeros",
        ),
    )

    assert result.item() == pytest.approx(expected)


def test_rtc_vjp_wraps_the_composed_cfg_velocity() -> None:
    result = sample_guided_cfg_actions(
        _FakeCFGModel(0.5),
        object(),
        torch.full((1, 1, 1), 2.0),
        inference_delay=0,
        config=GuidedRTCConfig(
            guidance_horizon=1,
            max_guidance_weight=5.0,
            prefix_attention_schedule="ones",
        ),
    )

    assert result.item() == pytest.approx(-16.0)


def test_zero_prefix_rtc_equals_plain_cfg_euler_step() -> None:
    result = sample_guided_cfg_actions(
        _FakeCFGModel(0.5),
        object(),
        torch.zeros((1, 1, 1)),
        inference_delay=0,
        config=GuidedRTCConfig(
            guidance_horizon=1,
            max_guidance_weight=5.0,
            prefix_attention_schedule="zeros",
        ),
    )

    # x=1; unconditional velocity=1; positive velocity=3; CFG velocity=2.
    assert result.item() == pytest.approx(-1.0)


def test_cfg_rtc_batch_matches_independent_rows() -> None:
    previous = torch.tensor([[[2.0]], [[4.0]]])
    config = GuidedRTCConfig(
        guidance_horizon=1,
        max_guidance_weight=5.0,
        prefix_attention_schedule="ones",
    )

    batched = sample_guided_cfg_actions(
        _FakeCFGModel(0.5, batch_size=2),
        object(),
        previous,
        inference_delay=0,
        config=config,
    )
    rows = torch.cat(
        [
            sample_guided_cfg_actions(
                _FakeCFGModel(0.5),
                object(),
                previous[index : index + 1],
                inference_delay=0,
                config=config,
            )
            for index in range(2)
        ]
    )

    torch.testing.assert_close(batched, rows)
