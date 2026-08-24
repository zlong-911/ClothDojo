# Adapted from Hugging Face LeRobot's RTC implementation, Copyright 2025
# The HuggingFace Inc. team, licensed under the Apache License, Version 2.0.

"""LeRobot guided RTC at ClothDojo's RLINF sampling boundary."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class GuidedRTCConfig:
    """The small, inference-only part of LeRobot's guided RTC config."""

    execution_horizon: int = 10
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: Literal["linear", "exp", "ones", "zeros"] = "linear"

    def __post_init__(self) -> None:
        if self.execution_horizon < 1:
            raise ValueError("execution_horizon must be positive")
        if self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be positive")
        if self.prefix_attention_schedule not in {"linear", "exp", "ones", "zeros"}:
            raise ValueError("unsupported RTC prefix schedule")

    def to_json_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "implementation": "lerobot_guided_upstream",
            "execution_horizon": self.execution_horizon,
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
    execution_horizon = min(config.execution_horizon, overlap)
    weights = prefix_weights(
        inference_delay,
        execution_horizon,
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
    latent = x_t.detach().clone()
    velocity = denoise(latent)
    predicted_actions = latent - time_tensor * velocity
    correction = ((target - predicted_actions) * weights).detach()

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
    return velocity - guidance_weight * correction


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
