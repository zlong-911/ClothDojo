"""Thin entry points over RLINF's SFT runner."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TrainingKind = Literal["openpi-rlinf", "steam", "cfg"]


@dataclass(frozen=True)
class _Runtime:
    validate_cfg: Callable[[Any], Any]
    cluster_type: type
    placement_type: type
    runner_type: type
    worker_type: type
    multiprocessing: Any
    omega_conf: Any


def run(
    *,
    kind: TrainingKind,
    config_path: str | Path,
    overrides: Sequence[str] = (),
    steam_member_index: int | None = None,
) -> None:
    """Compose one ClothDojo config and run it through the selected RLINF worker."""

    cfg = compose_config(config_path, overrides)
    cfg_manifest = None
    if kind == "cfg":
        from clothdojo.training.cfg import inspect_cfg_training

        cfg_manifest = inspect_cfg_training(config_path, list(overrides))
    run_config(
        kind=kind,
        cfg=cfg,
        cfg_manifest=cfg_manifest,
        steam_member_index=steam_member_index,
        config_path=config_path,
    )


def check(
    *,
    kind: TrainingKind,
    config_path: str | Path,
    overrides: Sequence[str] = (),
    steam_member_index: int | None = None,
) -> dict[str, Any]:
    """Compose and validate a config without launching training workers."""

    cfg = compose_config(config_path, overrides)
    if kind == "openpi-rlinf":
        return inspect_openpi_rlinf_sft(cfg, config_path=config_path)
    if kind == "cfg":
        from clothdojo.training.cfg import inspect_cfg_training

        return inspect_cfg_training(config_path, list(overrides))
    if kind == "steam":
        if steam_member_index is None:
            raise ValueError("STEAM training requires a member index")
        from clothdojo.training.steam import inspect_steam_training

        return inspect_steam_training(cfg, steam_member_index)
    runtime = _load_runtime(kind)
    cfg = runtime.validate_cfg(cfg)
    resolved = runtime.omega_conf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved RLINF config must be a mapping")
    return resolved


def compose_config(config_path: str | Path, overrides: Sequence[str] = ()) -> Any:
    """Compose a Hydra config rooted at the directory containing ``config_path``."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    if path.suffix != ".yaml":
        raise ValueError(f"Training config must be a .yaml file: {path}")

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(path.parent), version_base="1.1"):
        return compose(config_name=path.stem, overrides=list(overrides))


def inspect_openpi_rlinf_sft(
    cfg: Any,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the current RLINF OpenPI data and numerical contract."""

    from omegaconf import OmegaConf

    from clothdojo.training.openpi_artifacts import (
        inspect_openpi_dataset,
        inspect_openpi_norm_stats,
    )

    model_type = str(cfg.actor.model.model_type)
    if model_type != "openpi_rlinf":
        raise ValueError(
            "OpenPI/RLinf training requires "
            f"actor.model.model_type=openpi_rlinf, got {model_type!r}"
        )
    numerical = {
        "model_precision": str(cfg.actor.model.precision),
        "parameter_compute_dtype": str(cfg.actor.fsdp_config.mixed_precision.param_dtype),
        "gradient_reduce_dtype": str(cfg.actor.fsdp_config.mixed_precision.reduce_dtype),
        "buffer_dtype": str(cfg.actor.fsdp_config.mixed_precision.buffer_dtype),
    }
    expected_numerical = {
        "model_precision": "fp32",
        "parameter_compute_dtype": "bf16",
        "gradient_reduce_dtype": "fp32",
        "buffer_dtype": "fp32",
    }
    if numerical != expected_numerical:
        raise ValueError(
            f"OpenPI/RLinf numerical contract differs: {numerical} != {expected_numerical}"
        )
    openpi_data_home = Path(str(cfg.data.openpi_data_home)).expanduser().resolve()
    tokenizer = openpi_data_home / "big_vision" / "paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"cached PaliGemma tokenizer is missing: {tokenizer}")
    training_contract = {
        "seed": int(cfg.actor.seed),
        "global_batch_size": int(cfg.actor.global_batch_size),
        "steps": int(cfg.runner.max_steps),
        "peak_lr": float(cfg.actor.optim.lr),
        "warmup_steps": int(cfg.actor.optim.lr_warmup_steps),
        "decay_steps": int(cfg.actor.optim.total_training_steps),
        "floor_lr": float(cfg.actor.optim.min_lr),
        "adam_b1": float(cfg.actor.optim.adam_beta1),
        "adam_b2": float(cfg.actor.optim.adam_beta2),
        "adam_eps": float(cfg.actor.optim.adam_eps),
        "weight_decay": float(cfg.actor.optim.weight_decay),
        "clip_gradient_norm": float(cfg.actor.optim.clip_grad),
        "ema_decay": float(cfg.actor.optim.ema_decay),
    }
    _validate_openpi_rlinf_runtime_contract(cfg, training_contract)
    jax_reference = _inspect_jax_reference(cfg, config_path, training_contract)
    dataset = inspect_openpi_dataset(cfg.data.train_data_paths)
    openpi_data = OmegaConf.to_container(cfg.actor.model.openpi_data, resolve=True)
    if not isinstance(openpi_data, dict):
        raise TypeError("OpenPI/RLinf openpi_data config must be a mapping")
    norm_path = openpi_data.pop("norm_stats_path", None)
    if norm_path is None:
        raise ValueError("OpenPI/RLinf requires openpi_data.norm_stats_path")
    norm = inspect_openpi_norm_stats(
        norm_path,
        dataset=dataset,
        config_name=str(cfg.actor.model.openpi.config_name),
        data_kwargs=openpi_data,
        action_horizon=int(cfg.actor.model.num_action_chunks),
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved OpenPI/RLinf config must be a mapping")
    return {
        "schema": "clothdojo_openpi_rlinf_sft_preflight",
        "compatible": True,
        "model_type": model_type,
        "numerical_contract": numerical,
        "training_contract": training_contract,
        "jax_reference": jax_reference,
        "openpi_data_home": str(openpi_data_home),
        "tokenizer": str(tokenizer),
        "dataset": dataset,
        "norm_stats": norm,
        "resolved_config": resolved,
    }


def run_config(
    *,
    kind: TrainingKind,
    cfg: Any,
    cfg_manifest: dict[str, Any] | None = None,
    steam_member_index: int | None = None,
    config_path: str | Path | None = None,
) -> None:
    """Run an already composed config without modifying RLINF source code."""

    data_root = cfg.get("data", {}).get("data_root")
    if data_root:
        os.environ["HF_LEROBOT_HOME"] = str(data_root)

    if kind == "openpi-rlinf":
        manifest = inspect_openpi_rlinf_sft(cfg, config_path=config_path)
        os.environ["OPENPI_DATA_HOME"] = manifest["openpi_data_home"]
        logging.info("OpenPI/RLinf preflight:\n%s", json.dumps(manifest, indent=2))
    elif kind == "cfg":
        if cfg_manifest is None:
            raise ValueError("CFG training requires its checked launch manifest")
        from clothdojo.training.cfg import write_cfg_launch_manifest

        manifest_path = write_cfg_launch_manifest(cfg, cfg_manifest)
        from omegaconf import OmegaConf

        OmegaConf.update(
            cfg,
            "actor.model.expected_weights_sha256",
            cfg_manifest["init_checkpoint_weights_sha256"],
            force_add=True,
        )
        logging.info("CFG launch manifest: %s", manifest_path)
    elif kind == "steam":
        if steam_member_index is None:
            raise ValueError("STEAM training requires a member index")
        from clothdojo.training.steam import (
            inspect_steam_training,
            write_steam_launch_manifest,
        )

        manifest = inspect_steam_training(cfg, steam_member_index)
        manifest_path = write_steam_launch_manifest(cfg, manifest)
        from clothdojo.training.rlinf_steam import prepare_steam_driver_runtime

        prepare_steam_driver_runtime()
        logging.info("STEAM launch manifest: %s", manifest_path)

    # Importing the scheduler initializes Ray on this RLINF revision, so keep
    # the runtime import strictly after the launch gate and manifest write.
    runtime = _load_runtime(kind)
    cfg = runtime.validate_cfg(cfg)
    if kind != "steam":
        runtime.multiprocessing.set_start_method("spawn", force=True)
    logging.info(
        "RLINF %s config:\n%s",
        kind,
        json.dumps(runtime.omega_conf.to_container(cfg, resolve=True), indent=2),
    )

    cluster = runtime.cluster_type(cluster_cfg=cfg.cluster)
    if kind == "steam":
        from clothdojo.training.rlinf_steam import configure_steam_cluster

        configure_steam_cluster(cluster, cfg)
    placement = runtime.placement_type(cfg, cluster)
    actor_group = runtime.worker_type.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )

    runner = runtime.runner_type(cfg=cfg, actor=actor_group)
    runner.init_workers()
    runner.run()


def _validate_openpi_rlinf_runtime_contract(
    cfg: Any,
    contract: dict[str, int | float],
) -> None:
    if str(cfg.actor.optim.lr_scheduler) != "openpi_cosine":
        raise ValueError("OpenPI/RLinf comparison requires the OpenPI cosine schedule")
    if float(cfg.actor.optim.num_cycles) != 0.5:
        raise ValueError("OpenPI/RLinf comparison requires a half-cosine decay")
    if contract["steps"] != contract["decay_steps"]:
        raise ValueError("OpenPI/RLinf steps and cosine decay horizon must match")
    if int(cfg.runner.save_interval) < 1:
        raise ValueError("OpenPI/RLinf save_interval must be positive")
    if bool(cfg.actor.fsdp_config.get("save_full_model_weights", True)):
        raise ValueError("OpenPI/RLinf EMA owns the inference full-weights checkpoint")
    if int(cfg.actor.micro_batch_size) < 1:
        raise ValueError("OpenPI/RLinf micro_batch_size must be positive")


def _inspect_jax_reference(
    cfg: Any,
    config_path: str | Path | None,
    actual: dict[str, int | float],
) -> dict[str, Any] | None:
    project = cfg.get("clothdojo")
    reference = None if project is None else project.get("jax_reference")
    if reference is None:
        return None
    if config_path is None:
        raise ValueError("OpenPI/RLinf JAX comparison requires its config path")

    from clothdojo.training.jax_sft import compose_jax_sft_spec
    from clothdojo.training.matrix import compose_jax_sft_matrix

    root = Path(config_path).expanduser().resolve().parent
    matrix_path = Path(str(reference.get("matrix"))).expanduser()
    if not matrix_path.is_absolute():
        matrix_path = root / matrix_path
    matrix = compose_jax_sft_matrix(matrix_path)
    trial_name = str(reference.get("trial", ""))
    matches = [trial for trial in matrix.trials if trial.name == trial_name]
    if len(matches) != 1:
        raise ValueError(f"JAX comparison trial is not unique: {trial_name!r}")
    spec = compose_jax_sft_spec(matrix.base_config, matches[0].overrides)
    expected = {
        "seed": spec.seed,
        "global_batch_size": spec.batch_size,
        "steps": spec.num_train_steps,
        "peak_lr": spec.peak_lr,
        "warmup_steps": spec.warmup_steps,
        "decay_steps": spec.decay_steps,
        "floor_lr": spec.decay_lr,
        "adam_b1": spec.adam_b1,
        "adam_b2": spec.adam_b2,
        "adam_eps": spec.adam_eps,
        "weight_decay": spec.weight_decay,
        "clip_gradient_norm": spec.clip_gradient_norm,
        "ema_decay": spec.ema_decay,
    }
    if actual != expected:
        mismatches = {
            key: {"jax": expected[key], "rlinf": actual.get(key)}
            for key in expected
            if actual.get(key) != expected[key]
        }
        raise ValueError(f"OpenPI/RLinf differs from its JAX reference: {mismatches}")

    actor_devices = _placement_size(str(cfg.cluster.component_placement.actor))
    if int(cfg.cluster.num_nodes) * actor_devices != spec.fsdp_devices:
        raise ValueError("OpenPI/RLinf and JAX use different accelerator counts")
    if int(cfg.actor.micro_batch_size) * spec.fsdp_devices != spec.batch_size:
        raise ValueError("OpenPI/RLinf microbatching differs from the JAX per-device batch")
    if int(cfg.data.num_workers) * spec.fsdp_devices != spec.num_workers:
        raise ValueError("OpenPI/RLinf and JAX use different total decoder worker counts")
    if int(cfg.runner.save_interval) != spec.save_interval:
        raise ValueError("OpenPI/RLinf and JAX use different checkpoint intervals")

    data_settings = cfg.actor.model.openpi_data
    aligned = {
        "openpi_config": str(cfg.actor.model.openpi.config_name) == spec.openpi_config,
        "default_prompt": str(data_settings.default_prompt) == spec.default_prompt,
        "adapt_to_pi": bool(data_settings.adapt_to_pi) is False,
        "extra_delta_transform": bool(data_settings.extra_delta_transform) is True,
        "dataset": Path(str(cfg.data.train_data_paths)).expanduser().resolve()
        == spec.dataset,
        "norm_stats": Path(str(data_settings.norm_stats_path)).expanduser().resolve()
        == spec.norm_stats,
        "pytorch_base": Path(str(cfg.actor.model.model_path)).expanduser().resolve()
        == spec.pytorch_base,
        "openpi_cache": Path(str(cfg.data.openpi_data_home)).expanduser().resolve()
        == spec.openpi_data_home,
    }
    mismatches = [name for name, is_aligned in aligned.items() if not is_aligned]
    if mismatches:
        raise ValueError(
            f"OpenPI/RLinf data or initialization differs from JAX: {mismatches}"
        )
    return {
        "matrix": str(matrix.config_path),
        "trial": trial_name,
        "base_config": str(matrix.base_config),
        "run_name": spec.run_name,
        "training_contract": expected,
    }


def _placement_size(value: str) -> int:
    devices: set[int] = set()
    for item in value.split(","):
        bounds = item.strip().split("-", 1)
        try:
            start = int(bounds[0])
            stop = int(bounds[-1])
        except ValueError as exc:
            raise ValueError(f"invalid actor placement: {value!r}") from exc
        if start < 0 or stop < start:
            raise ValueError(f"invalid actor placement: {value!r}")
        devices.update(range(start, stop + 1))
    if not devices:
        raise ValueError("actor placement must contain at least one device")
    return len(devices)


def _load_runtime(kind: TrainingKind) -> _Runtime:
    import torch.multiprocessing as multiprocessing
    from omegaconf import OmegaConf
    from rlinf.config import validate_cfg
    from rlinf.runners.sft_runner import SFTRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    if kind == "openpi-rlinf":
        from clothdojo.training.openpi_rlinf_sft import OpenPiRlinfSftWorker

        worker_type = OpenPiRlinfSftWorker
    elif kind == "steam":
        from clothdojo.training.rlinf_steam import ClothDojoSteamSftWorker

        worker_type = ClothDojoSteamSftWorker
    elif kind == "cfg":
        from clothdojo.training.rlinf_cfg import ClothDojoCfgWorker, enable

        enable()
        worker_type = ClothDojoCfgWorker
    else:
        raise ValueError(f"Unknown RLINF training kind: {kind}")

    return _Runtime(
        validate_cfg=validate_cfg,
        cluster_type=Cluster,
        placement_type=HybridComponentPlacement,
        runner_type=SFTRunner,
        worker_type=worker_type,
        multiprocessing=multiprocessing,
        omega_conf=OmegaConf,
    )
