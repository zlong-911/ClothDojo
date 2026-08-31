# Portions derived from RLINF, Copyright 2026 The RLINF Authors,
# licensed under the Apache License, Version 2.0.

"""State-aware STEAM training/inference extension for ClothDojo's pinned RLINF.

The three-member evaluator was trained with a small state-fusion extension
that is not present in the pinned upstream package.  This module owns that
exact process-local extension so inference never depends on the old training
checkout or on edits inside site-packages.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_ENABLED = False


def enable_state_aware_steam() -> None:
    """Install the checkpoint's state/config/data contract into pinned RLINF."""

    global _ENABLED
    if _ENABLED:
        return

    from clothdojo.training.steam_dataloader import install_parquet_archive_guard

    install_parquet_archive_guard()

    import rlinf.models.embodiment.value_model.steam as steam_factory
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from rlinf.data.datasets.steam import pair_dataset
    from rlinf.models.embodiment.value_model.steam.configuration import SteamConfig
    from rlinf.models.embodiment.value_model.steam.modeling_critic import (
        CriticOutput,
        SteamCriticModel,
    )
    from rlinf.models.embodiment.value_model.steam.modeling_steam import (
        SteamBackbone,
        _module_parameter_dtype,
    )

    required_defaults = {
        "include_state": False,
        "state_dim": 0,
        "state_mean": None,
        "state_std": None,
        "state_keys": ["observation.state"],
        "geometry_contract": None,
    }
    overlap = set(required_defaults) & set(steam_factory._STEAM_CONFIG_DEFAULTS)
    if overlap:
        raise RuntimeError(
            "pinned RLINF now contains a state-aware STEAM implementation; "
            f"remove or re-audit ClothDojo's extension first: {sorted(overlap)}"
        )
    steam_factory._STEAM_CONFIG_DEFAULTS.update(required_defaults)

    original_config_init = SteamConfig.__init__

    def config_init(
        self,
        *args,
        include_state: bool = False,
        state_dim: int = 0,
        state_mean: list[float] | None = None,
        state_std: list[float] | None = None,
        state_keys=None,
        geometry_contract=None,
        **kwargs,
    ) -> None:
        original_config_init(self, *args, **kwargs)
        self.include_state = bool(include_state)
        self.state_dim = int(state_dim)
        self.state_keys = list(state_keys or ["observation.state"])
        self.geometry_contract = geometry_contract
        self.state_mean = None if state_mean is None else [float(value) for value in state_mean]
        self.state_std = None if state_std is None else [float(value) for value in state_std]
        if self.include_state:
            if self.state_dim < 1:
                raise ValueError("state_dim must be positive when include_state=true")
            if self.state_mean is None or len(self.state_mean) != self.state_dim:
                raise ValueError("state_mean must contain state_dim values")
            if self.state_std is None or len(self.state_std) != self.state_dim:
                raise ValueError("state_std must contain state_dim values")
            if any(value <= 0.0 for value in self.state_std):
                raise ValueError("state_std values must be positive")

    SteamConfig.__init__ = config_init

    original_backbone_init = SteamBackbone.__init__
    original_compute_features = SteamBackbone._compute_projected_features

    def backbone_init(self, config) -> None:
        original_backbone_init(self, config)
        if not config.include_state:
            return
        self.register_buffer(
            "state_mean",
            torch.tensor(config.state_mean, dtype=torch.float32).view(1, 1, -1),
            persistent=True,
        )
        self.register_buffer(
            "state_std",
            torch.tensor(config.state_std, dtype=torch.float32).view(1, 1, -1),
            persistent=True,
        )
        self.state_projector = nn.Sequential(
            nn.Linear(config.state_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.frame_projector = nn.Sequential(
            nn.Linear(2 * config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def compute_features(
        self,
        input_ids,
        attention_mask,
        images,
        image_attention_mask,
        state=None,
    ):
        _unused_fused, per_frame, language = original_compute_features(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_attention_mask,
        )
        if not self.cfg.include_state:
            return self._fuse(per_frame, language), per_frame, language
        expected = (
            int(images.shape[0]),
            int(self.cfg.num_frames_per_pair),
            int(self.cfg.state_dim),
        )
        if state is None or tuple(state.shape) != expected:
            actual = None if state is None else tuple(state.shape)
            raise ValueError(
                f"state-conditioned STEAM expects state shape {expected}, got {actual}"
            )
        state_float = state.to(device=images.device, dtype=torch.float32)
        normalized = (state_float - self.state_mean) / self.state_std
        state_dtype = _module_parameter_dtype(self.state_projector, normalized.dtype)
        state_features = self.state_projector(normalized.to(dtype=state_dtype))
        frame_dtype = _module_parameter_dtype(self.frame_projector, per_frame.dtype)
        per_frame = self.frame_projector(
            torch.cat(
                (
                    per_frame.to(dtype=frame_dtype),
                    state_features.to(dtype=frame_dtype),
                ),
                dim=-1,
            )
        )
        return self._fuse(per_frame, language), per_frame, language

    SteamBackbone.__init__ = backbone_init
    SteamBackbone._compute_projected_features = compute_features

    original_stack_observation = SteamCriticModel._stack_observation

    def stack_observation(self, observation):
        return (*original_stack_observation(self, observation), observation.get("state"))

    def run_head(self, observation, labels=None):
        input_ids, attention_mask, images, image_mask, state = self._stack_observation(observation)
        hidden, _per_frame, _language = self.model._compute_projected_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_mask,
            state=state,
        )
        head_dtype = _module_parameter_dtype(self.model.value_head, hidden.dtype)
        logits = self.model.value_head(hidden.to(dtype=head_dtype))
        probabilities = functional.softmax(logits, dim=-1)
        values = self._predicted_signed_value(probabilities)
        loss = metrics = None
        if labels is not None:
            per_sample_loss, metrics = self._compute_loss(logits, labels)
            loss = per_sample_loss.mean()
        return CriticOutput(
            loss=loss,
            predicted_values=values,
            logits=logits,
            probs=probabilities,
            atoms=None,
            expert_loss=loss,
            hidden_states=hidden,
            cat_acc_best=metrics["acc_best"] if metrics else None,
            cat_acc_neighbor=metrics["acc_neighbor"] if metrics else None,
            mae=metrics["mae"] if metrics else None,
            progress_values=values,
        )

    def forward(self, observation, labels=None, **_kwargs):
        return run_head(self, observation, labels)

    @torch.no_grad()
    def predict(self, observation):
        return run_head(self, observation)

    SteamCriticModel._stack_observation = stack_observation
    SteamCriticModel.forward = forward
    SteamCriticModel.predict = predict

    original_dataset_init = pair_dataset.BinaryPairInferenceDataset.__init__
    original_build_sample = pair_dataset.BinaryPairInferenceDataset._build_sample_from_pair
    original_collate = pair_dataset.BinaryPairDataCollator.__call__

    def dataset_init(
        self,
        *args,
        include_state: bool = False,
        state_key: str = "observation.state",
        state_keys=None,
        **kwargs,
    ) -> None:
        original_dataset_init(self, *args, **kwargs)
        self.include_state = bool(include_state)
        self.state_key = str(state_key)
        self.state_keys = list(state_keys or [self.state_key])

    def build_sample(self, *, raw_t=None, raw_tk=None, **kwargs):
        sample = original_build_sample(
            self,
            raw_t=raw_t,
            raw_tk=raw_tk,
            **kwargs,
        )
        if self.include_state:
            if raw_t is None or raw_tk is None:
                raise RuntimeError("state-conditioned STEAM requires raw frame pairs")
            sample["state_t"] = _state_from_keys(raw_t, self.state_keys)
            sample["state_tk"] = _state_from_keys(raw_tk, self.state_keys)
        return sample

    original_train_sample = pair_dataset.PairDataset._build_sample

    def build_train_sample(self, *, raw_t=None, raw_tk=None, **kwargs):
        sample = original_train_sample(self, raw_t=raw_t, raw_tk=raw_tk, **kwargs)
        keys = getattr(self, "state_keys", None)
        if keys is not None:
            if raw_t is None or raw_tk is None:
                raise RuntimeError("state-conditioned STEAM requires raw training frames")
            sample["state_t"] = _state_from_keys(raw_t, keys)
            sample["state_tk"] = _state_from_keys(raw_tk, keys)
        return sample

    def collate(self, examples):
        batch = original_collate(self, examples)
        present = ["state_t" in example and "state_tk" in example for example in examples]
        if any(present) and not all(present):
            raise ValueError("STEAM batch mixes state-aware and image-only samples")
        if all(present):
            state_t = torch.from_numpy(
                np.stack([example["state_t"] for example in examples]).astype(
                    np.float32, copy=False
                )
            )
            state_tk = torch.from_numpy(
                np.stack([example["state_tk"] for example in examples]).astype(
                    np.float32, copy=False
                )
            )
            batch["observation"]["state"] = torch.stack((state_t, state_tk), dim=1)
        return batch

    pair_dataset.BinaryPairInferenceDataset.__init__ = dataset_init
    pair_dataset.BinaryPairInferenceDataset._build_sample_from_pair = build_sample
    pair_dataset.BinaryPairDataCollator.__call__ = collate
    pair_dataset.PairDataset._build_sample = build_train_sample

    def strict_checkpoint_load(model, state_dict, model_path) -> None:
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"STEAM checkpoint does not exactly match the state-aware model: {model_path}"
            ) from exc

    steam_factory._load_checkpoint_into_model = strict_checkpoint_load
    _ENABLED = True


def _state_from_sample(sample: dict[str, Any], key: str) -> np.ndarray:
    import torch

    if key not in sample:
        raise KeyError(f"STEAM state key {key!r} is absent from a frame")
    value = sample[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"STEAM state {key!r} must be finite and non-empty")
    return result


def _state_from_keys(sample: dict[str, Any], keys) -> np.ndarray:
    return np.concatenate([_state_from_sample(sample, key) for key in keys])


def require_state_aware_evaluator(model: Any, *, state_dim: int | None = None) -> None:
    """Fail unless all loaded ensemble members own the trained state branch."""

    config = model.config
    keys = list(getattr(config, "state_keys", ["observation.state"]))
    supported = {("observation.state",): 14, ("observation.state", "observation.cloth_metrics"): 16}
    expected_dim = supported.get(tuple(keys))
    if expected_dim is None:
        raise RuntimeError(f"unsupported STEAM state keys: {keys}")
    if state_dim is None:
        state_dim = expected_dim
    if expected_dim != state_dim:
        raise RuntimeError("STEAM state keys and dimensions disagree")
    if getattr(config, "include_state", False) is not True:
        raise RuntimeError("STEAM evaluator config does not enable state")
    if int(getattr(config, "state_dim", 0)) != state_dim:
        raise RuntimeError(f"STEAM evaluator state_dim is not {state_dim}")
    members = list(getattr(model, "members", (model,)))
    for index, member in enumerate(members):
        backbone = member.model
        for name in ("state_mean", "state_std", "state_projector", "frame_projector"):
            if not hasattr(backbone, name):
                raise RuntimeError(f"STEAM member {index} is missing {name}")
