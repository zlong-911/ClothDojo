"""Narrow RLINF adapter for the audited ClothDojo STEAM training path."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from rlinf.utils.utils import seed_everything
from rlinf.workers.sft.fsdp_steam_sft_worker import FSDPSteamSftWorker

from clothdojo.training.worker_environment import apply_worker_environment


def prepare_steam_driver_runtime() -> None:
    """Publish the runtime selected by the validated TorchCodec gate."""

    _prepend_ffmpeg_library_path()
    os.environ["RLINF_STEAM_VIDEO_BACKEND"] = "torchcodec"
    os.environ["RLINF_ISOLATE_HF_DATASETS_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"


def configure_steam_cluster(cluster: Any, cfg: Any) -> None:
    """Propagate decoder and member settings into every RLINF node runtime."""

    prepare_steam_driver_runtime()
    names = (
        "RLINF_STEAM_VIDEO_BACKEND",
        "RLINF_ISOLATE_HF_DATASETS_CACHE",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "CLOTHDOJO_STEAM_FFMPEG_DIR",
        "CLOTHDOJO_STEAM_MEMBER_INDEX",
        "CLOTHDOJO_STEAM_HEAD_SEED",
        "CLOTHDOJO_STEAM_RUN_SEED",
    )
    for node_rank in range(int(cfg.cluster.num_nodes)):
        node = cluster.get_node_info(node_rank)
        for name in names:
            node.env_vars[name] = os.environ[name]
        node.env_vars["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]


class ClothDojoSteamSftWorker(FSDPSteamSftWorker):
    """Add deterministic decoding, state, full-model, and FP32-master gates."""

    def __init__(self, cfg: Any):
        apply_worker_environment(cfg)
        prepare_steam_driver_runtime()
        from clothdojo.training.rlinf_steam_state import enable_state_aware_steam

        enable_state_aware_steam()
        multiprocessing.set_forkserver_preload(
            ["torch", "numpy", "clothdojo.training.steam_dataloader"]
        )
        multiprocessing.set_start_method("forkserver", force=True)
        seed_everything(_run_seed())
        self._state_input_audit: dict[str, Any] | None = None
        self._precision_audited = False
        super().__init__(cfg)

    def build_dataloader(self):
        data_loader, eval_loaders = super().build_dataloader()
        dataset = data_loader._data_loader.dataset
        if not hasattr(dataset, "_rng"):
            raise RuntimeError("STEAM requires one direct PairDataset with a seedable RNG")
        rank_seed = _run_seed() + int(self._rank) * 1_000
        dataset._rng = np.random.default_rng(rank_seed)
        dataset.state_keys = list(self.cfg.actor.model.get("state_keys", ["observation.state"]))
        sampler = data_loader._data_loader.sampler
        if isinstance(sampler, torch.utils.data.DistributedSampler):
            sampler.seed = _run_seed()
        if int(self.cfg.data.train_num_workers) > 0:
            from clothdojo.training.steam_dataloader import seed_pair_dataset_worker

            data_loader._data_loader.worker_init_fn = partial(
                seed_pair_dataset_worker,
                rank_seed=rank_seed,
            )
        return data_loader, eval_loaders

    def init_worker(self):
        result = super().init_worker()
        seed_everything(_run_seed() + int(self._rank))
        return result

    def model_provider_func(self):
        module = super().model_provider_func()
        from rlinf.models.embodiment.value_model.steam import (
            reinitialize_member_value_heads,
        )

        reinitialize_member_value_heads([module], _head_seed())
        state_parameters = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if "state_projector" in name or "frame_projector" in name
        ]
        if not state_parameters:
            raise RuntimeError(f"STEAM state branch is absent on rank {self._rank}")
        frozen = [name for name, parameter in state_parameters if not parameter.requires_grad]
        if frozen:
            raise RuntimeError(f"STEAM state branch is frozen on rank {self._rank}: {frozen}")
        self._prewrap_state_parameters = state_parameters
        self._prewrap_state_elements = sum(parameter.numel() for _, parameter in state_parameters)
        if self._prewrap_state_elements == 0:
            raise RuntimeError(f"STEAM state branch is empty on rank {self._rank}")
        return module

    def _prepare_input(self, batch: dict):
        observation, labels = super()._prepare_input(batch)
        state = observation.get("state")
        if not isinstance(state, torch.Tensor):
            raise RuntimeError(f"STEAM state input is absent on rank {self._rank}")
        state_dim = int(self.cfg.actor.model.state_dim)
        expected_shape = (int(labels.shape[0]), 2, state_dim)
        if tuple(state.shape) != expected_shape:
            raise RuntimeError(
                f"STEAM state shape changed on rank {self._rank}: "
                f"{tuple(state.shape)} != {expected_shape}"
            )
        finite = bool(torch.isfinite(state).all())
        if state.dtype != torch.float32 or not finite:
            raise RuntimeError(
                f"STEAM state input is invalid on rank {self._rank}: "
                f"dtype={state.dtype}, finite={finite}"
            )
        if self._state_input_audit is None:
            flat = state.detach().float().reshape(-1, state_dim)
            self._state_input_audit = {
                "shape": list(state.shape),
                "dtype": str(state.dtype),
                "finite": True,
                "mean": flat.mean(dim=0).cpu().tolist(),
                "std": flat.std(dim=0, unbiased=False).cpu().tolist(),
                "min": flat.min(dim=0).values.cpu().tolist(),
                "max": flat.max(dim=0).values.cpu().tolist(),
            }
        return observation, labels

    def run_training(self):
        metrics = super().run_training()
        if self._precision_audited:
            return metrics
        if self._state_input_audit is None:
            raise RuntimeError(f"STEAM state gate saw no batch on rank {self._rank}")
        audit = self._dtype_audit()
        audit["first_batch_state"] = self._state_input_audit
        for key in ("trainable_master_fp32_fraction", "optimizer_state_fp32_fraction"):
            if float(audit[key]) < 0.999999:
                raise RuntimeError(
                    f"STEAM FP32-master gate failed on rank {self._rank}: {key}={audit[key]}"
                )
            metrics[f"diagnostic/{key}"] = float(audit[key])
        self._write_runtime_audit(audit)
        self._precision_audited = True
        return metrics

    def _write_runtime_audit(self, audit: dict[str, Any]) -> None:
        root = (
            Path(str(self.cfg.runner.logger.log_path))
            / str(self.cfg.runner.logger.experiment_name)
            / "precision_state_audit"
        )
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"rank{int(self._rank):03d}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _dtype_audit(self) -> dict[str, Any]:
        trainable: dict[str, int] = {}
        frozen: dict[str, int] = {}
        for _, parameter in self.model.named_parameters():
            if not parameter.is_floating_point():
                continue
            target = trainable if parameter.requires_grad else frozen
            dtype = str(parameter.dtype)
            target[dtype] = target.get(dtype, 0) + parameter.numel()
        if sum(frozen.values()) != 0:
            raise RuntimeError(f"STEAM full-model gate found frozen weights: {frozen}")

        optimizer_parameter_ids = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        state_names = [name for name, _ in self._prewrap_state_parameters]
        optimizer_state_names = [
            name
            for name, parameter in self._prewrap_state_parameters
            if id(parameter) in optimizer_parameter_ids
        ]
        if set(optimizer_state_names) != set(state_names):
            raise RuntimeError(
                "STEAM optimizer does not own the full state branch: "
                f"model={state_names}, optimizer={optimizer_state_names}"
            )

        optimizer: dict[str, int] = {}
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and value.is_floating_point():
                    dtype = str(value.dtype)
                    optimizer[dtype] = optimizer.get(dtype, 0) + value.numel()
        trainable_total = sum(trainable.values())
        optimizer_total = sum(optimizer.values())
        if trainable_total == 0 or optimizer_total == 0:
            raise RuntimeError("STEAM FP32-master gate saw empty model or optimizer state")
        return {
            "rank": int(self._rank),
            "world_size": int(self._world_size),
            "hostname": os.uname().nodename,
            "communication": {
                name: os.environ.get(name)
                for name in (
                    "NCCL_SOCKET_IFNAME",
                    "GLOO_SOCKET_IFNAME",
                    "NCCL_IB_HCA",
                    "NCCL_IB_GID_INDEX",
                )
            },
            "configured_master_precision": str(self.cfg.actor.model.precision),
            "configured_compute_param_dtype": str(
                self.cfg.actor.fsdp_config.mixed_precision.param_dtype
            ),
            "trainable_elements": trainable,
            "frozen_elements": frozen,
            "state_branch_parameter_names": state_names,
            "state_branch_optimizer_parameter_names": optimizer_state_names,
            "state_branch_prewrap_elements": self._prewrap_state_elements,
            "optimizer_state_elements": optimizer,
            "trainable_master_fp32_fraction": trainable.get("torch.float32", 0) / trainable_total,
            "optimizer_state_fp32_fraction": optimizer.get("torch.float32", 0) / optimizer_total,
        }


def _prepend_ffmpeg_library_path() -> Path:
    value = os.environ.get("CLOTHDOJO_STEAM_FFMPEG_DIR", "")
    directory = Path(value).expanduser().resolve()
    if not value or not directory.is_dir():
        raise FileNotFoundError(
            "CLOTHDOJO_STEAM_FFMPEG_DIR must name the FFmpeg 7 library directory"
        )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [str(directory)] + [
        item for item in current.split(os.pathsep) if item and item != str(directory)
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
    return directory


def _run_seed() -> int:
    return int(os.environ["CLOTHDOJO_STEAM_RUN_SEED"])


def _head_seed() -> int:
    return int(os.environ["CLOTHDOJO_STEAM_HEAD_SEED"])
