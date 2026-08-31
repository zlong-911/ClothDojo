# Adapted from Hugging Face LeRobot's RTC implementation, Copyright 2025
# The HuggingFace Inc. team, licensed under the Apache License, Version 2.0.

"""Physical Intelligence/LeRobot guided RTC at the RLINF sampling boundary."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class GuidedRTCConfig:
    """The small, inference-only part of LeRobot's guided RTC config."""

    guidance_horizon: int = 10
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: Literal["linear", "exp", "ones", "zeros"] = "linear"

    def __post_init__(self) -> None:
        if isinstance(self.guidance_horizon, bool) or not isinstance(
            self.guidance_horizon, int
        ):
            raise TypeError("guidance_horizon must be an integer")
        if self.guidance_horizon < 1:
            raise ValueError("guidance_horizon must be positive")
        if (
            isinstance(self.max_guidance_weight, bool)
            or not isinstance(self.max_guidance_weight, (int, float))
            or not math.isfinite(self.max_guidance_weight)
            or self.max_guidance_weight <= 0
        ):
            raise ValueError("max_guidance_weight must be finite and positive")
        if self.prefix_attention_schedule not in {"linear", "exp", "ones", "zeros"}:
            raise ValueError("unsupported RTC prefix schedule")

    def to_json_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "implementation": "physical_intelligence_guided_vjp",
            "guidance_horizon": self.guidance_horizon,
            "max_guidance_weight": self.max_guidance_weight,
            "prefix_attention_schedule": self.prefix_attention_schedule,
            "delta_rebase": True,
        }


def prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: Literal["linear", "exp", "ones", "zeros"],
    *,
    device: Any = None,
    dtype: Any = None,
):
    """Return LeRobot/Physical Intelligence's soft overlap mask."""

    import torch

    if total < 1 or start < 0 or end < 0:
        raise ValueError("prefix horizons must be non-negative and total must be positive")
    start = min(start, end)
    positions = torch.arange(total, device=device, dtype=dtype or torch.float32)
    if schedule == "ones":
        weights = torch.ones(total, device=device, dtype=dtype)
    elif schedule == "zeros":
        weights = (positions < start).to(dtype=dtype or torch.float32)
    elif schedule in {"linear", "exp"}:
        denominator = end - start + 1
        weights = torch.clamp((start - 1 - positions) / denominator + 1, 0, 1)
        if schedule == "exp":
            weights = weights * torch.expm1(weights) / (math.e - 1.0)
    else:
        raise ValueError(f"unsupported RTC prefix schedule: {schedule!r}")
    return torch.where(positions >= end, 0, weights)


def guided_velocity(
    x_t: Any,
    previous_actions: Any,
    *,
    inference_delay: int,
    time: Any,
    denoise: Callable[[Any], Any],
    config: GuidedRTCConfig,
):
    """Apply the correction used by LeRobot's released guided RTC runtime."""

    import torch

    if x_t.ndim != 3 or previous_actions.ndim != 3:
        raise ValueError("RTC actions must have shape (batch, horizon, action_dim)")
    if x_t.shape[0] != previous_actions.shape[0]:
        raise ValueError("RTC action batches must match")
    if previous_actions.shape[2] > x_t.shape[2]:
        raise ValueError("the previous RTC action dimension exceeds the model dimension")

    target = torch.zeros_like(x_t)
    overlap = min(previous_actions.shape[1], x_t.shape[1])
    target[:, :overlap, : previous_actions.shape[2]] = previous_actions[:, :overlap]
    guidance_horizon = min(config.guidance_horizon, overlap)
    weights = prefix_weights(
        inference_delay,
        guidance_horizon,
        x_t.shape[1],
        config.prefix_attention_schedule,
        device=x_t.device,
        dtype=x_t.dtype,
    )[None, :, None]

    time_tensor = torch.as_tensor(time, device=x_t.device, dtype=x_t.dtype)
    if time_tensor.ndim == 0:
        time_tensor = time_tensor.reshape(1, 1, 1)
    elif time_tensor.ndim == 1:
        time_tensor = time_tensor[:, None, None]
    # RTC guides the final-action estimate through the complete denoising map.
    # Keeping the velocity's Jacobian here is the defining difference from a
    # direct output-space blend.  ``sample_guided_actions`` is called below a
    # no-grad inference boundary, so explicitly re-enable autograd for the VJP.
    with torch.enable_grad():
        latent = x_t.detach().clone().requires_grad_(True)
        velocity = denoise(latent)
        predicted_actions = latent - time_tensor * velocity
        error = ((target - predicted_actions) * weights).detach()
        correction = torch.autograd.grad(
            predicted_actions,
            latent,
            grad_outputs=error,
            retain_graph=False,
            create_graph=False,
        )[0]

    tau = 1.0 - time_tensor
    maximum = torch.as_tensor(
        config.max_guidance_weight,
        device=x_t.device,
        dtype=x_t.dtype,
    )
    inv_r2 = (time_tensor.square() + tau.square()) / time_tensor.square()
    coefficient = torch.nan_to_num(time_tensor / tau, posinf=maximum)
    guidance_weight = torch.minimum(
        torch.nan_to_num(coefficient * inv_r2, posinf=maximum),
        maximum,
    )
    return (velocity - guidance_weight * correction).detach()


def sample_guided_actions(
    model: Any,
    observation: Any,
    previous_actions: Any,
    *,
    inference_delay: int,
    config: GuidedRTCConfig,
):
    """Run RLINF's OpenPI flow ODE with LeRobot guided RTC velocity."""

    import torch

    with torch.no_grad():
        images, image_masks, language, language_masks, state = (
            model._preprocess_observation(observation, train=False)
        )
        _, prefix_masks, prefix_cache = model._build_prefix_cache(
            images,
            image_masks,
            language,
            language_masks,
        )
        x_t = model.sample_noise(
            (state.shape[0], model.config.action_horizon, model.config.action_dim),
            state.device,
        )
        timesteps = model._get_timesteps(model.config.num_steps, state.device)

    for index in range(model.config.num_steps):
        time = timesteps[index].expand(state.shape[0])

        def denoise(latent, current_time=time):
            return model.get_velocity(
                state,
                latent,
                current_time,
                prefix_masks,
                prefix_cache,
            )[0]

        velocity = guided_velocity(
            x_t,
            previous_actions,
            inference_delay=inference_delay,
            time=time,
            denoise=denoise,
            config=config,
        )
        delta = timesteps[index] - timesteps[index + 1]
        x_t = (x_t - delta * velocity).detach()
    return x_t


def sample_guided_openpi_rlinf_actions(
    model: Any,
    observation: Any,
    previous_actions: Any,
    *,
    inference_delay: int,
    num_steps: int,
    config: GuidedRTCConfig,
):
    """Run guided RTC on RLinf's current ``openpi_rlinf`` Pi0 core."""

    import torch
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module

    if num_steps < 1:
        raise ValueError("RTC num_steps must be positive")
    observation = model_module.preprocess_observation(observation, train=False)
    with torch.no_grad():
        _, prefix_mask, prefix_cache = model.build_prefix_cache(observation)
        x_t = torch.randn(
            previous_actions.shape[0],
            model.action_horizon,
            model.action_dim,
            device=previous_actions.device,
            dtype=torch.float32,
        )

    dt = -1.0 / num_steps
    time = 1.0
    while time >= -dt / 2:
        time_tensor = torch.full(
            (x_t.shape[0],),
            time,
            device=x_t.device,
            dtype=torch.float32,
        )

        def denoise(latent, current_time=time_tensor):
            suffix = model.run_suffix(
                observation,
                latent,
                current_time,
                prefix_cache,
                prefix_mask,
            )
            return model.velocity_from_suffix(suffix)

        velocity = guided_velocity(
            x_t,
            previous_actions,
            inference_delay=inference_delay,
            time=time_tensor,
            denoise=denoise,
            config=config,
        )
        x_t = (x_t + dt * velocity).detach()
        time += dt
    return x_t

def sample_guided_cfg_openpi_rlinf_actions(
    model: Any,
    unconditional_observation: Any,
    conditional_observation: Any | None,
    previous_actions: Any,
    *,
    inference_delay: int,
    num_steps: int,
    guidance_scale: float,
    config: GuidedRTCConfig,
):
    """Compose current-core CFG velocity, then apply guided RTC."""

    import torch
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module

    if num_steps < 1:
        raise ValueError("RTC num_steps must be positive")
    unconditional = model_module.preprocess_observation(
        unconditional_observation,
        train=False,
    )
    conditional = (
        None
        if conditional_observation is None
        else model_module.preprocess_observation(conditional_observation, train=False)
    )
    with torch.no_grad():
        _, unconditional_mask, unconditional_cache = model.build_prefix_cache(
            unconditional
        )
        if conditional is None:
            conditional_mask = conditional_cache = None
        else:
            _, conditional_mask, conditional_cache = model.build_prefix_cache(conditional)
        x_t = torch.randn(
            previous_actions.shape[0],
            model.action_horizon,
            model.action_dim,
            device=previous_actions.device,
            dtype=torch.float32,
        )

    dt = -1.0 / num_steps
    time = 1.0
    while time >= -dt / 2:
        time_tensor = torch.full(
            (x_t.shape[0],),
            time,
            device=x_t.device,
            dtype=torch.float32,
        )

        def denoise(latent, current_time=time_tensor):
            unconditional_suffix = model.run_suffix(
                unconditional,
                latent,
                current_time,
                unconditional_cache,
                unconditional_mask,
            )
            velocity = model.velocity_from_suffix(unconditional_suffix)
            if conditional is not None:
                conditional_suffix = model.run_suffix(
                    conditional,
                    latent,
                    current_time,
                    conditional_cache,
                    conditional_mask,
                )
                conditional_velocity = model.velocity_from_suffix(conditional_suffix)
                velocity = (
                    (1.0 - guidance_scale) * velocity
                    + guidance_scale * conditional_velocity
                )
            return velocity

        velocity = guided_velocity(
            x_t,
            previous_actions,
            inference_delay=inference_delay,
            time=time_tensor,
            denoise=denoise,
            config=config,
        )
        x_t = (x_t + dt * velocity).detach()
        time += dt
    return x_t
