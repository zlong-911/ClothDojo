"""Training-time action conditioning for real-time chunking."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class TrainingTimeRTCSpec:
    max_delay: int

    def __post_init__(self) -> None:
        if isinstance(self.max_delay, bool) or not isinstance(self.max_delay, int):
            raise TypeError("training-time RTC max_delay must be an integer")
        if self.max_delay < 1:
            raise ValueError("training-time RTC max_delay must be positive")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "method": "training_time_action_conditioning",
            "max_delay": self.max_delay,
            "delay_support": list(range(self.max_delay)),
            "delay_distribution": "uniform",
            "conditioned_prefix": "clean_ground_truth",
            "loss": "postfix_only",
        }


def training_time_rtc_spec_from_config(cfg: Any) -> TrainingTimeRTCSpec | None:
    raw = cfg.get("clothdojo", {}).get("training_time_rtc")
    if raw is None:
        return None
    if set(raw) != {"max_delay"}:
        raise ValueError("clothdojo.training_time_rtc must contain exactly max_delay")
    spec = TrainingTimeRTCSpec(max_delay=raw["max_delay"])
    horizon = int(cfg.actor.model.num_action_chunks)
    if spec.max_delay > horizon:
        raise ValueError(
            f"training-time RTC max_delay exceeds action horizon: {spec.max_delay} > {horizon}"
        )
    return spec


def training_time_rtc_manifest_from_config(cfg: Any) -> dict[str, Any]:
    spec = training_time_rtc_spec_from_config(cfg)
    return {"enabled": False} if spec is None else spec.to_json_dict()


def install_training_time_rtc(core: Any, spec: TrainingTimeRTCSpec | None) -> None:
    """Install parameter-preserving per-action time conditioning on one Pi0.5 core."""

    if spec is None:
        return
    if not bool(getattr(core, "pi05", False)):
        raise ValueError("training-time RTC currently requires the Pi0.5 action expert")
    if spec.max_delay > int(core.action_horizon):
        raise ValueError("training-time RTC max_delay exceeds the model action horizon")
    if getattr(core.compute_loss, "_clothdojo_training_time_rtc", None) is not None:
        raise ValueError("training-time RTC is already installed")

    _install_per_action_time_embedding(core)
    original_compute_loss = core.compute_loss.__func__

    @wraps(original_compute_loss)
    def compute_loss(
        model: Any,
        observation: Any,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        delay: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module
        from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import make_attn_mask

        batch = actions.shape[0]
        device = actions.device
        observation = model_module.preprocess_observation(observation, train=train, rng=rng)
        observation = model_module._observation_to_dtype(observation, model.embed_dtype)
        actions = actions.to(dtype=model.embed_dtype)
        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=actions.dtype, generator=rng)
        else:
            noise = noise.to(device=device, dtype=actions.dtype)
        if time is None:
            time = (
                torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
                .sample((batch,))
                .to(device=device, dtype=actions.dtype)
            )
            time = time * 0.999 + 0.001
        else:
            time = time.to(device=device, dtype=actions.dtype)
        if delay is None:
            delay = torch.randint(spec.max_delay, (batch,), device=device, generator=rng)
        else:
            delay = delay.to(device=device)
        if delay.shape != (batch,) or delay.dtype == torch.bool:
            raise ValueError("training-time RTC delay must have shape [batch]")
        if torch.any(delay < 0) or torch.any(delay >= spec.max_delay):
            raise ValueError("training-time RTC delay is outside configured support")

        prefix = torch.arange(model.action_horizon, device=device)[None, :] < delay[:, None]
        per_action_time = torch.where(prefix, torch.zeros_like(time[:, None]), time[:, None])
        x_t = per_action_time[:, :, None] * noise + (1 - per_action_time[:, :, None]) * actions
        target_velocity = noise - actions

        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(
            observation, x_t, per_action_time
        )
        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar, suffix_ar], dim=0)
        positions = torch.cumsum(input_mask.int(), dim=1) - 1
        _, suffix_out = model.llm(
            [prefix_tokens, suffix_tokens],
            positions=positions,
            mask=make_attn_mask(input_mask, ar_mask),
            adarms_cond=[None, adarms],
        )[0]
        velocity = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        per_offset = torch.mean(torch.square(velocity - target_velocity), dim=-1)
        postfix = ~prefix
        # The surrounding SFT wrapper calls mean(); rescale so it equals the
        # paper's sum(loss * postfix) / sum(postfix).
        scale = per_offset.numel() / postfix.sum().clamp_min(1)
        return per_offset * postfix.to(per_offset.dtype) * scale

    compute_loss._clothdojo_training_time_rtc = spec
    core.compute_loss = MethodType(compute_loss, core)


def _install_per_action_time_embedding(core: Any) -> None:
    original_embed_suffix = core.embed_suffix.__func__

    @wraps(original_embed_suffix)
    def embed_suffix(
        model: Any,
        observation: Any,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ):
        if timestep.ndim == 1:
            return original_embed_suffix(model, observation, noisy_actions, timestep)
        if timestep.shape != noisy_actions.shape[:2]:
            raise ValueError("per-action timestep must have shape [batch, action_horizon]")
        noisy_actions = noisy_actions.to(model.embed_dtype)
        timestep = timestep.to(model.embed_dtype)
        action_tokens = model.action_in_proj(noisy_actions)
        width = model.action_in_proj.out_features
        fraction = torch.linspace(0.0, 1.0, width // 2, device=timestep.device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        phase = timestep.float()[..., None] / period * 2 * torch.pi
        time_emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).to(timestep.dtype)
        time_emb = torch.nn.functional.silu(model.time_mlp_in(time_emb))
        time_emb = torch.nn.functional.silu(model.time_mlp_out(time_emb))
        mask = torch.ones(action_tokens.shape[:2], dtype=torch.bool, device=action_tokens.device)
        ar_mask = torch.zeros(action_tokens.shape[1], dtype=torch.bool, device=action_tokens.device)
        ar_mask[0] = True
        return action_tokens, mask, ar_mask, time_emb

    core.embed_suffix = MethodType(embed_suffix, core)

    for module in core.llm.modules():
        if not bool(getattr(module, "adaptive", False)):
            continue
        module.forward = MethodType(_per_action_rms_forward(module.forward.__func__), module)


def _per_action_rms_forward(original_forward: Any):
    @wraps(original_forward)
    def forward(norm: Any, x: torch.Tensor, cond: torch.Tensor | None = None):
        if cond is None or cond.ndim != 3:
            return original_forward(norm, x, cond)
        dtype = x.dtype
        variance = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(variance + 1e-6)
        modulation = norm.ada_modulation(cond.to(norm.ada_modulation.weight.dtype))
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    return forward
