"""Worker boundary for RLINF's current ``openpi_rlinf`` SFT model."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from torch.utils.data.distributed import DistributedSampler

from clothdojo.training.augmentation import install_training_augmentation
from clothdojo.training.openpi import TrainingPrompt
from clothdojo.training.worker_environment import apply_worker_environment


class _EpochAdvancingSampler:
    """Advance the wrapped OpenPI sampler whenever its infinite loader restarts."""

    def __init__(self, sampler: DistributedSampler[Any]) -> None:
        self._sampler = sampler
        self._next_epoch = 0

    def __iter__(self) -> Iterator[int]:
        self._sampler.set_epoch(self._next_epoch)
        self._next_epoch += 1
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._sampler)

    def set_epoch(self, epoch: int) -> None:
        self._next_epoch = int(epoch)


def _select_pyav_backend() -> None:
    """Select the supported decoder before RLINF constructs the LeRobot dataset."""

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    lerobot_dataset.get_safe_default_codec = lambda: "pyav"


class OpenPiRlinfSftWorker(FSDPVlaSftWorker):
    """Current OpenPI_RLinf worker with deterministic official data iteration."""

    def __init__(self, cfg: Any) -> None:
        apply_worker_environment(cfg)
        _select_pyav_backend()
        _seed_rank(int(cfg.actor.seed), int(os.environ.get("RANK", 0)))
        self._action_offset_weights = _action_offset_weights_from_config(cfg)
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        data_loader, data_config = super().build_dataloader(
            data_paths,
            eval_dataset=eval_dataset,
        )
        _set_training_prompt(data_loader, self.cfg.actor.model.openpi_data.default_prompt)
        if not eval_dataset:
            _adapt_official_openpi_loader(
                data_loader,
                seed=int(self.cfg.actor.seed),
                rank=int(self._rank),
            )
        return data_loader, data_config

    def init_worker(self) -> None:
        super().init_worker()
        module = getattr(self.model, "module", self.model)
        install_training_augmentation(
            module.model,
            enabled=self.cfg.get("clothdojo", {}).get("image_augmentation", False),
        )
        _install_action_offset_reduction(self.model, self._action_offset_weights)
        _validate_fp32_training_state(self.model, self.optimizer)
        self._ema_decay = float(self.cfg.actor.optim.ema_decay)
        if not 0.0 < self._ema_decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {self._ema_decay}")
        self._ema_state = _initialize_ema_state(self.model)

    def optimizer_step(self):
        grad_norm, lr_list = super().optimizer_step()
        if torch.isfinite(torch.as_tensor(grad_norm)).item():
            _update_ema_state(self.model, self._ema_state, self._ema_decay)
        return grad_norm, lr_list

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """Save online DCP state and JAX-compatible EMA inference weights."""

        if self.cfg.actor.fsdp_config.get("save_full_model_weights", True):
            raise ValueError(
                "OpenPI/RLinf EMA checkpoints require "
                "actor.fsdp_config.save_full_model_weights=false"
            )
        super().save_checkpoint(save_path, step)
        _swap_parameter_data_with_ema(self.model, self._ema_state)
        try:
            state = self.get_model_state_dict(cpu_offload=True, full_state_dict=True)
            if torch.distributed.get_rank() == 0:
                weights_dir = Path(save_path) / "model_state_dict"
                weights_dir.mkdir(parents=True, exist_ok=True)
                torch.save(state, weights_dir / "full_weights.pt")
                contract = {
                    "schema": "clothdojo_openpi_rlinf_checkpoint",
                    "format_version": 1,
                    "step": int(step),
                    "ema": {
                        "decay": self._ema_decay,
                        "path": "model_state_dict/full_weights.pt",
                        "purpose": "inference and rollout",
                    },
                    "online": {
                        "path": "dcp_checkpoint",
                        "purpose": "online weights and optimizer-update analysis; resume disabled",
                    },
                }
                (Path(save_path) / "clothdojo_weights.json").write_text(
                    json.dumps(contract, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            torch.distributed.barrier()
        finally:
            _swap_parameter_data_with_ema(self.model, self._ema_state)

    def load_checkpoint(self, load_path: str) -> None:
        """Reject upstream resume calls; SFT runs always start from step zero."""

        raise ValueError("SFT resume is disabled; restart training in a new output directory")


def _action_offset_weights_from_config(cfg: Any) -> tuple[float, ...]:
    """Resolve the shared Pi0.5 action-offset loss contract for RLINF."""

    from omegaconf import OmegaConf

    from clothdojo.training.jax_sft import (
        _action_offset_weights,
        _compose_action_offset_loss,
    )

    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("resolved OpenPI/RLinf config must be a mapping")
    spec = _compose_action_offset_loss(raw)
    return _action_offset_weights(
        spec,
        action_horizon=int(cfg.actor.model.num_action_chunks),
    )


def _action_offset_loss_manifest_from_config(cfg: Any) -> dict[str, Any]:
    """Describe the exact reduction that will run inside the RLINF worker."""

    from omegaconf import OmegaConf

    from clothdojo.training.jax_sft import (
        _action_offset_loss_manifest,
        _compose_action_offset_loss,
    )

    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("resolved OpenPI/RLinf config must be a mapping")
    return _action_offset_loss_manifest(
        _compose_action_offset_loss(raw),
        action_horizon=int(cfg.actor.model.num_action_chunks),
    )


def _reduce_action_offset_loss(
    per_offset_loss: torch.Tensor,
    weights: tuple[float, ...],
) -> torch.Tensor:
    """Apply the same normalized action-offset reduction as OpenPI/JAX."""

    if per_offset_loss.ndim != 2:
        raise ValueError(
            "OpenPI/RLinf per-offset loss must have shape [batch, action_horizon], "
            f"got {tuple(per_offset_loss.shape)}"
        )
    if per_offset_loss.shape[-1] != len(weights):
        raise ValueError(
            "OpenPI/RLinf action-offset weight count differs from model horizon: "
            f"{len(weights)} != {per_offset_loss.shape[-1]}"
        )
    weight_tensor = per_offset_loss.new_tensor(weights)
    return ((per_offset_loss * weight_tensor).sum(dim=-1) / weight_tensor.sum()).mean()


def _install_action_offset_reduction(
    fsdp_model: Any,
    weights: tuple[float, ...],
) -> None:
    """Replace only OpenPI_RLinf's scalar SFT reduction on one model instance."""

    if all(weight == 1.0 for weight in weights):
        return
    module = getattr(fsdp_model, "module", fsdp_model)
    from rlinf.models.embodiment.openpi_rlinf.sft_action_model import (
        OpenPiPytorchSFTActionModel,
    )

    if not isinstance(module, OpenPiPytorchSFTActionModel):
        raise TypeError(
            f"action-offset loss requires OpenPiPytorchSFTActionModel, got {type(module).__name__}"
        )
    if bool(module.rlt_cfg.use_rlt):
        raise ValueError("action-offset loss is not defined for the legacy RLT objective")

    def weighted_sft_forward(model: Any, data: Any) -> torch.Tensor:
        observation, actions = model._unpack_sft_batch(data)
        observation = model._observation_to_device(observation)
        actions = model._actions_to_device(actions)
        per_offset_loss = model.model.compute_loss(observation, actions, train=True)
        return _reduce_action_offset_loss(per_offset_loss, weights)

    module.sft_forward = MethodType(weighted_sft_forward, module)


def _seed_rank(seed: int, rank: int) -> int:
    from rlinf.utils.utils import seed_everything

    rank_seed = int(seed) + int(rank)
    seed_everything(rank_seed)
    return rank_seed


def _openpi_torch_loader(data_loader: Any) -> Any:
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader,
        "torch_loader",
        None,
    )
    if torch_loader is None:
        raise TypeError("OpenPI loader layout changed: inner torch DataLoader is missing")
    return torch_loader


def _set_training_prompt(data_loader: Any, prompt: str) -> None:
    """Override dataset task text before the official repack/tokenize pipeline."""

    from openpi import transforms
    from openpi.training.data_loader import TransformedDataset

    dataset = _openpi_torch_loader(data_loader).dataset
    if not isinstance(dataset, TransformedDataset):
        raise TypeError("OpenPI loader layout changed: expected TransformedDataset")
    # The official loader has not started its workers yet. Keep its dataset,
    # batching and transforms intact; only supply the configured task text first.
    dataset._transform = transforms.compose([TrainingPrompt(prompt), dataset._transform])


def _adapt_official_openpi_loader(data_loader: Any, *, seed: int, rank: int) -> None:
    """Bind actor seed and epoch advancement at OpenPI's guarded torch boundary."""

    torch_loader = _openpi_torch_loader(data_loader)
    batch_sampler = getattr(torch_loader, "batch_sampler", None)
    sampler = getattr(batch_sampler, "sampler", None)
    if not isinstance(sampler, DistributedSampler):
        raise TypeError(
            "OpenPI/RLinf SFT requires a DistributedSampler after process-group "
            f"initialization, got {type(sampler).__name__}"
        )
    sampler.seed = int(seed)
    batch_sampler.sampler = _EpochAdvancingSampler(sampler)
    generator = getattr(torch_loader, "generator", None)
    if generator is not None:
        generator.manual_seed(int(seed) + int(rank))


def _validate_fp32_training_state(model: Any, optimizer: Any) -> None:
    """Fail before step one if FSDP or AdamW lost FP32 master state."""

    parameter_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError(
            f"OpenPI/RLinf trainable parameters must be FP32 masters, got {parameter_dtypes}"
        )
    state_dtypes = {
        value.dtype
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    }
    if state_dtypes and state_dtypes != {torch.float32}:
        raise RuntimeError(
            f"OpenPI/RLinf optimizer state must be FP32, got {state_dtypes}"
        )


def _trainable_parameters(model: Any) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.is_floating_point()
    }


@torch.no_grad()
def _initialize_ema_state(model: Any) -> dict[str, torch.Tensor]:
    parameters = _trainable_parameters(model)
    if not parameters:
        raise RuntimeError("OpenPI/RLinf EMA requires trainable floating parameters")
    return {name: parameter.detach().clone() for name, parameter in parameters.items()}


@torch.no_grad()
def _update_ema_state(
    model: Any,
    ema_state: dict[str, torch.Tensor],
    decay: float,
) -> None:
    parameters = _trainable_parameters(model)
    _validate_ema_layout(parameters, ema_state)
    ema_values = list(ema_state.values())
    parameter_values = [parameter.detach() for parameter in parameters.values()]
    torch._foreach_mul_(ema_values, decay)
    torch._foreach_add_(ema_values, parameter_values, alpha=1.0 - decay)


@torch.no_grad()
def _swap_parameter_data_with_ema(
    model: Any,
    ema_state: dict[str, torch.Tensor],
    *,
    chunk_elements: int = 4 * 1024 * 1024,
) -> None:
    """Swap online and EMA values without allocating another model copy."""

    parameters = _trainable_parameters(model)
    _validate_ema_layout(parameters, ema_state)
    for name, parameter in parameters.items():
        online = parameter.view(-1)
        ema = ema_state[name].view(-1)
        for start in range(0, online.numel(), chunk_elements):
            stop = min(start + chunk_elements, online.numel())
            temporary = online[start:stop].clone()
            online[start:stop].copy_(ema[start:stop])
            ema[start:stop].copy_(temporary)


def _validate_ema_layout(
    parameters: dict[str, torch.nn.Parameter],
    ema_state: dict[str, torch.Tensor],
) -> None:
    if parameters.keys() != ema_state.keys():
        raise RuntimeError("OpenPI/RLinf EMA parameter names changed after initialization")
    for name, parameter in parameters.items():
        shadow = ema_state[name]
        if parameter.shape != shadow.shape or parameter.dtype != shadow.dtype:
            raise RuntimeError(
                f"OpenPI/RLinf EMA layout changed for {name}: "
                f"{parameter.shape}@{parameter.dtype} != {shadow.shape}@{shadow.dtype}"
            )
