"""Project-owned CFG training and inference on RLINF's current OpenPI core."""

from __future__ import annotations

import dataclasses
from typing import Any

from clothdojo.training.augmentation import install_training_augmentation


@dataclasses.dataclass(frozen=True, slots=True)
class CfgRoutingConfig:
    positive_only_conditional: bool
    unconditional_prob: float
    guidance_type: str
    guidance_scale: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.unconditional_prob <= 1.0:
            raise ValueError("CFG unconditional_prob must be in [0, 1]")
        if self.guidance_type not in {"positive", "negative", "no_guide"}:
            raise ValueError(f"unsupported CFG guidance type: {self.guidance_type!r}")
        if self.positive_only_conditional and self.guidance_type == "negative":
            raise ValueError("negative CFG guidance requires negative-conditioned training")


def compute_cfg_routing_masks(
    advantage: Any,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    random_values: Any | None = None,
) -> dict[str, Any]:
    """Route binary STEAM labels to conditional and unconditional branches."""

    import torch

    labels = advantage.to(dtype=torch.bool)
    if random_values is None:
        random_values = torch.rand(labels.shape[0], device=labels.device)
    else:
        random_values = random_values.to(device=labels.device)
    positive = labels
    negative = ~positive
    if positive_only_conditional:
        positive_conditional = positive & (random_values > unconditional_prob)
        negative_conditional = torch.zeros_like(positive)
    else:
        conditional = random_values > unconditional_prob
        positive_conditional = positive & conditional
        negative_conditional = negative & conditional
    conditional = positive_conditional | negative_conditional
    return {
        "positive_mask": positive,
        "negative_mask": negative,
        "conditional_mask": conditional,
        "positive_conditional_mask": positive_conditional,
        "positive_unconditional_mask": positive & ~positive_conditional,
        "negative_conditional_mask": negative_conditional,
        "negative_unconditional_mask": negative & ~negative_conditional,
    }


def build_current_openpi_cfg_model(cfg: Any, torch_dtype: Any = None) -> Any:
    """Build CFG on the same ``openpi_rlinf`` wrapper used by project SFT."""

    import torch
    from omegaconf import OmegaConf
    from rlinf.config import torch_dtype_from_precision
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_rlinf.utils.model_builders import (
        _build_eval_model,
    )

    if bool(cfg.openpi.train_expert_only):
        raise ValueError("ClothDojo CFG requires full-model training")
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("resolved CFG model config must be a mapping")
    openpi = raw.setdefault("openpi", {})
    openpi["task"] = "eval"
    openpi.setdefault("model_action_dim", 32)
    openpi.setdefault("paligemma_variant", "gemma_2b")
    openpi.setdefault("action_expert_variant", "gemma_300m")
    openpi.setdefault("discrete_state_input", True)
    openpi.setdefault("max_token_len", 200)
    openpi.setdefault("use_rlt", False)
    raw.setdefault("pi05", True)
    raw["num_steps"] = int(openpi["num_steps"])
    resolved = OmegaConf.create(raw)
    pi0_config = Pi0Config(
        pi05=bool(raw["pi05"]),
        action_horizon=int(raw["num_action_chunks"]),
        action_dim=int(openpi["model_action_dim"]),
        paligemma_variant=str(openpi["paligemma_variant"]),
        action_expert_variant=str(openpi["action_expert_variant"]),
        dtype="bfloat16",
        pcd=False,
        discrete_state_input=bool(openpi["discrete_state_input"]),
        max_token_len=int(openpi["max_token_len"]),
    )
    target_dtype = (
        torch_dtype if torch_dtype is not None else torch_dtype_from_precision(resolved.precision)
    )
    # The outer project loader replaces every parameter from the strictly
    # validated checkpoint. Constructing 3.35B random CPU values on every rank
    # only adds several minutes of startup and heavy memory traffic.
    with torch.device("meta"):
        core = pi0_config.create()
        if target_dtype is not None:
            core = core.to(target_dtype)
        base = _build_eval_model(
            resolved,
            resolved.openpi,
            core,
            num_steps=int(raw["num_steps"]),
            action_chunk=int(raw["num_action_chunks"]),
            action_env_dim=int(raw["action_dim"]),
        )

    routing = CfgRoutingConfig(
        positive_only_conditional=bool(openpi["positive_only_conditional"]),
        unconditional_prob=float(openpi["unconditional_prob"]),
        guidance_type=str(openpi["guidance_type"]),
        guidance_scale=float(openpi["cfgrl_guidance_scale"]),
    )
    model = ClothDojoOpenPiCfgModel(
        base.model,
        num_steps=int(raw["num_steps"]),
        action_env_dim=int(raw["action_dim"]),
        action_chunk=int(raw["num_action_chunks"]),
        config_name=str(openpi["config_name"]),
        state_indices=base.state_indices,
        routing=routing,
    )
    model._input_transform_fn = base._input_transform_fn
    model._output_transform_fn = base._output_transform_fn
    # Allocate uninitialized CPU storage while preserving the module's tied
    # parameter contract. ``to_empty`` itself creates a distinct Parameter for
    # every registered name, so explicitly restore aliases afterwards.
    tied_parameters = _tied_parameter_names(model)
    model.to_empty(device="cpu")
    _restore_tied_parameters(model, tied_parameters)
    return model


def _tied_parameter_names(model: Any) -> tuple[tuple[str, ...], ...]:
    groups: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    return tuple(tuple(names) for names in groups.values() if len(names) > 1)


def _restore_tied_parameters(
    model: Any,
    groups: tuple[tuple[str, ...], ...],
) -> None:
    for group in groups:
        source_parent, _, source_name = group[0].rpartition(".")
        source_module = model.get_submodule(source_parent) if source_parent else model
        parameter = getattr(source_module, source_name)
        for target in group[1:]:
            target_parent, _, target_name = target.rpartition(".")
            target_module = model.get_submodule(target_parent) if target_parent else model
            setattr(target_module, target_name, parameter)


def _move_cfg_observation(observation: Any, tokens: Any, masks: Any, device: Any) -> Any:
    import torch
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation

    def move(value: Any, *, state: bool = False) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        if state:
            return value.to(device=device, dtype=torch.float32)
        return value.to(device=device)

    return Observation(
        images={key: move(value) for key, value in observation.images.items()},
        image_masks={key: move(value) for key, value in observation.image_masks.items()},
        state=move(observation.state, state=True),
        tokenized_prompt=move(tokens),
        tokenized_prompt_mask=move(masks),
        token_ar_mask=move(getattr(observation, "token_ar_mask", None)),
        token_loss_mask=move(getattr(observation, "token_loss_mask", None)),
        pcd_xyz=move(getattr(observation, "pcd_xyz", None)),
    )


def _masked_loss_sum(loss: Any, mask: Any) -> float:
    if mask.numel() == 0 or not mask.any():
        return 0.0
    return float((loss * mask.to(loss.dtype)).sum().item())


def _cfg_model_type() -> type:
    import torch
    from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
        OpenPiPytorchEvalActionModel,
    )
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module

    class _ClothDojoOpenPiCfgModel(OpenPiPytorchEvalActionModel):
        def __init__(self, *args: Any, routing: CfgRoutingConfig, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Match SFT's deterministic image policy; routing and flow sampling stay random.
            install_training_augmentation(self.model)
            self.routing = routing
            self._keep_gradient_checkpointing = True

        def gradient_checkpointing_disable(self, **kwargs: Any) -> None:
            # FSDPCfgWorker disables this unconditionally inside every step even
            # when its documented config enables it. The current OpenPI core
            # supports checkpointing, so preserve the configured state.
            if not self._keep_gradient_checkpointing:
                super().gradient_checkpointing_disable(**kwargs)

        def forward(self, data: dict[str, Any], **kwargs: Any):
            del kwargs
            observation = data["observation"]
            actions = data["actions"].to(device=self.device, dtype=torch.float32)
            advantage = data["advantage"].to(device=self.device, dtype=torch.bool)
            routing = compute_cfg_routing_masks(
                advantage,
                positive_only_conditional=self.routing.positive_only_conditional,
                unconditional_prob=self.routing.unconditional_prob,
            )

            positive_tokens = observation.tokenized_positive_guidance_prompt
            positive_masks = observation.tokenized_positive_guidance_prompt_mask
            negative_tokens = observation.tokenized_negative_guidance_prompt
            negative_masks = observation.tokenized_negative_guidance_prompt_mask
            if self.routing.positive_only_conditional:
                route_tokens = positive_tokens
                route_masks = positive_masks
            else:
                route_tokens = torch.where(advantage[:, None], positive_tokens, negative_tokens)
                route_masks = torch.where(advantage[:, None], positive_masks, negative_masks)
            conditional = routing["conditional_mask"][:, None]
            tokens = torch.where(conditional, route_tokens, observation.tokenized_prompt)
            masks = torch.where(conditional, route_masks, observation.tokenized_prompt_mask)
            routed_observation = _move_cfg_observation(observation, tokens, masks, self.device)
            per_offset_loss = self.model.compute_loss(routed_observation, actions, train=True)
            per_sample_loss = per_offset_loss.detach().mean(dim=-1)
            metrics = {
                "conditional_count": routing["conditional_mask"].sum().item(),
                "unconditional_count": (~routing["conditional_mask"]).sum().item(),
                "positive_label_count": routing["positive_mask"].sum().item(),
                "negative_label_count": routing["negative_mask"].sum().item(),
            }
            for name in (
                "positive_conditional",
                "positive_unconditional",
                "negative_conditional",
                "negative_unconditional",
            ):
                mask = routing[f"{name}_mask"]
                metrics[f"{name}_count"] = mask.sum().item()
                metrics[f"{name}_loss_sum"] = _masked_loss_sum(per_sample_loss, mask)
            metrics["conditional_loss_sum"] = _masked_loss_sum(
                per_sample_loss, routing["conditional_mask"]
            )
            metrics["unconditional_loss_sum"] = _masked_loss_sum(
                per_sample_loss, ~routing["conditional_mask"]
            )
            return per_offset_loss.mean(), metrics

        def prepare_cfg_observations(self, env_obs: dict[str, Any]):
            unconditional = self._prepare_prompt_observation(env_obs, suffix=None)
            if self.routing.guidance_type == "no_guide":
                return unconditional, None
            suffix = f"Advantage: {self.routing.guidance_type}"
            conditional = self._prepare_prompt_observation(env_obs, suffix=suffix)
            return unconditional, conditional

        def _prepare_prompt_observation(self, env_obs: dict[str, Any], *, suffix: str | None):
            routed = dict(env_obs)
            if suffix is not None:
                routed["task_descriptions"] = [
                    f"{prompt}\n{suffix}" for prompt in env_obs["task_descriptions"]
                ]
            processed = self.input_transform(self._repack_env_obs(routed), transpose=False)
            return self._observation_dict_to_device(processed)

        @torch.no_grad()
        def predict_action_batch(
            self,
            env_obs: dict[str, Any],
            mode: str = "eval",
            compute_values: bool = False,
            *,
            noise: Any | None = None,
            rng: Any | None = None,
            **kwargs: Any,
        ):
            del compute_values, kwargs
            if mode != "eval":
                raise ValueError("ClothDojo CFG serving only supports eval mode")
            unconditional, conditional = self.prepare_cfg_observations(env_obs)
            unconditional = model_module.preprocess_observation(unconditional, train=False)
            if conditional is not None:
                conditional = model_module.preprocess_observation(conditional, train=False)
            batch_size = unconditional.state.shape[0]
            if noise is None:
                noise = torch.randn(
                    batch_size,
                    self.model.action_horizon,
                    self.model.action_dim,
                    device=self.device,
                    dtype=torch.float32,
                    generator=rng,
                )
            model_actions = _sample_cfg_actions(
                self.model,
                unconditional,
                conditional,
                noise,
                num_steps=self.num_steps,
                guidance_scale=self.routing.guidance_scale,
            )
            actions = self.output_transform(
                {"actions": model_actions, "state": unconditional.state}
            )["actions"].to(device=self.device, dtype=torch.float32)
            return actions, {
                "prev_logprobs": None,
                "prev_values": None,
                "forward_inputs": {"raw_actions": model_actions},
            }

    return _ClothDojoOpenPiCfgModel


ClothDojoOpenPiCfgModel = _cfg_model_type()


def _sample_cfg_actions(
    model: Any,
    unconditional: Any,
    conditional: Any | None,
    noise: Any,
    *,
    num_steps: int,
    guidance_scale: float,
) -> Any:
    import torch

    _, unconditional_mask, unconditional_cache = model.build_prefix_cache(unconditional)
    if conditional is None:
        conditional_mask = conditional_cache = None
    else:
        _, conditional_mask, conditional_cache = model.build_prefix_cache(conditional)
    actions = noise
    step = -1.0 / num_steps
    for index in range(num_steps):
        time = torch.full(
            (actions.shape[0],),
            1.0 - index / num_steps,
            device=actions.device,
            dtype=torch.float32,
        )
        unconditional_suffix = model.run_suffix(
            unconditional, actions, time, unconditional_cache, unconditional_mask
        )
        velocity = model.velocity_from_suffix(unconditional_suffix)
        if conditional is not None:
            conditional_suffix = model.run_suffix(
                conditional, actions, time, conditional_cache, conditional_mask
            )
            conditional_velocity = model.velocity_from_suffix(conditional_suffix)
            velocity = (1.0 - guidance_scale) * velocity + (guidance_scale * conditional_velocity)
        actions = actions + step * velocity
    return actions
