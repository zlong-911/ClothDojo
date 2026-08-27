"""Guarded OpenPI/JAX Pi0.5 SFT workflow.

RLINF owns the later CFG/rollout boundary.  This module deliberately reuses the
OpenPI libraries installed by the pinned ``rlinf-openpi`` dependency for SFT,
while making weight initialization and the EMA/online checkpoint distinction
explicit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Gate = Literal["initialize", "overfit", "pilot", "parity", "full"]


@dataclass(frozen=True, slots=True)
class JaxSftSpec:
    config_path: Path
    tuning_profile: str
    openpi_config: str
    video_backend: str
    dataset: Path
    norm_stats: Path
    base_params: Path
    pytorch_base: Path
    openpi_data_home: Path
    checkpoint_base_dir: Path
    gate_dir: Path
    project_name: str
    run_name: str
    wandb_group: str
    wandb_enabled: bool
    seed: int
    batch_size: int
    num_workers: int
    fsdp_devices: int
    num_train_steps: int
    log_interval: int
    save_interval: int
    keep_period: int
    warmup_steps: int
    peak_lr: float
    decay_steps: int
    decay_lr: float
    adam_b1: float
    adam_b2: float
    adam_eps: float
    weight_decay: float
    clip_gradient_norm: float
    ema_decay: float
    overfit_steps: int
    pilot_steps: int
    overfit_max_loss_ratio: float

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_base_dir / self.openpi_config / self.run_name


_CONTRACT: dict[str, Any] = {
    "schema": "clothdojo_pi05_jax_sft",
    "format_version": 1,
    "openpi.config_name": "pi05_aloha_robotwin",
    "openpi.data.adapt_to_pi": False,
    "openpi.data.extra_delta_transform": True,
    "openpi.data.default_prompt": "Flatten the shirt.",
    "openpi.data.video_backend": "pyav",
    "training.fsdp_devices": 8,
    "optimizer.adam_b1": 0.9,
    "optimizer.adam_b2": 0.95,
    "optimizer.adam_eps": 1.0e-8,
    "optimizer.weight_decay": 1.0e-10,
    "optimizer.clip_gradient_norm": 1.0,
    "optimizer.ema_decay": 0.99,
}

_TUNING_PROFILES: dict[str, dict[str, float | int]] = {
    "openpi_reference": {
        "training.seed": 42,
        "training.batch_size": 64,
        "training.num_train_steps": 40_000,
        "training.log_interval": 100,
        "training.save_interval": 10_000,
        "optimizer.warmup_steps": 1_000,
        "optimizer.peak_lr": 2.5e-5,
        "optimizer.decay_steps": 30_000,
        "optimizer.decay_lr": 2.5e-6,
    },
    "decay_40k_lr_0p5x": {
        "training.seed": 42,
        "training.batch_size": 64,
        "training.num_train_steps": 40_000,
        "training.log_interval": 100,
        "training.save_interval": 10_000,
        "optimizer.warmup_steps": 1_000,
        "optimizer.peak_lr": 1.25e-5,
        "optimizer.decay_steps": 40_000,
        "optimizer.decay_lr": 1.25e-6,
    },
    "decay_40k_lr_1x": {
        "training.seed": 42,
        "training.batch_size": 64,
        "training.num_train_steps": 40_000,
        "training.log_interval": 100,
        "training.save_interval": 10_000,
        "optimizer.warmup_steps": 1_000,
        "optimizer.peak_lr": 2.5e-5,
        "optimizer.decay_steps": 40_000,
        "optimizer.decay_lr": 2.5e-6,
    },
    "decay_40k_lr_2x": {
        "training.seed": 42,
        "training.batch_size": 64,
        "training.num_train_steps": 40_000,
        "training.log_interval": 100,
        "training.save_interval": 10_000,
        "optimizer.warmup_steps": 1_000,
        "optimizer.peak_lr": 5.0e-5,
        "optimizer.decay_steps": 40_000,
        "optimizer.decay_lr": 5.0e-6,
    },
    "batch_128_sample_equivalent": {
        "training.seed": 42,
        "training.batch_size": 128,
        "training.num_train_steps": 20_000,
        "training.log_interval": 50,
        "training.save_interval": 5_000,
        "optimizer.warmup_steps": 500,
        "optimizer.peak_lr": 2.5e-5,
        "optimizer.decay_steps": 20_000,
        "optimizer.decay_lr": 2.5e-6,
    },
    "decay_40k_lr_1x_seed43": {
        "training.seed": 43,
        "training.batch_size": 64,
        "training.num_train_steps": 40_000,
        "training.log_interval": 100,
        "training.save_interval": 10_000,
        "optimizer.warmup_steps": 1_000,
        "optimizer.peak_lr": 2.5e-5,
        "optimizer.decay_steps": 40_000,
        "optimizer.decay_lr": 2.5e-6,
    },
}


def compose_jax_sft_spec(
    config_path: str | Path,
    overrides: Sequence[str] = (),
) -> JaxSftSpec:
    """Load the small JAX SFT YAML and apply optional OmegaConf dot-list overrides."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JAX SFT config does not exist: {path}")
    if path.suffix != ".yaml":
        raise ValueError(f"JAX SFT config must be a .yaml file: {path}")

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("JAX SFT config must resolve to a mapping")

    _validate_contract(raw)
    experiment = _mapping(raw, "experiment")
    tuning_profile = str(experiment["profile"])
    _validate_tuning_profile(raw, tuning_profile)
    openpi = _mapping(raw, "openpi")
    paths = _mapping(raw, "paths")
    training = _mapping(raw, "training")
    optimizer = _mapping(raw, "optimizer")
    gates = _mapping(raw, "gates")
    logging = _mapping(raw, "logging")

    return JaxSftSpec(
        config_path=path,
        tuning_profile=tuning_profile,
        openpi_config=str(openpi["config_name"]),
        video_backend=str(_mapping(openpi, "data")["video_backend"]),
        dataset=_path(paths["dataset"]),
        norm_stats=_path(paths["norm_stats"]),
        base_params=_path(paths["base_params"]),
        pytorch_base=_path(paths["pytorch_base"]),
        openpi_data_home=_path(paths["openpi_data_home"]),
        checkpoint_base_dir=_path(paths["checkpoint_base_dir"]),
        gate_dir=_path(paths["gate_dir"]),
        project_name=str(logging["project_name"]),
        run_name=str(logging["run_name"]),
        wandb_group=str(logging["wandb_group"]),
        wandb_enabled=bool(logging["wandb_enabled"]),
        seed=int(training["seed"]),
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        fsdp_devices=int(training["fsdp_devices"]),
        num_train_steps=int(training["num_train_steps"]),
        log_interval=int(training["log_interval"]),
        save_interval=int(training["save_interval"]),
        keep_period=int(training["keep_period"]),
        warmup_steps=int(optimizer["warmup_steps"]),
        peak_lr=float(optimizer["peak_lr"]),
        decay_steps=int(optimizer["decay_steps"]),
        decay_lr=float(optimizer["decay_lr"]),
        adam_b1=float(optimizer["adam_b1"]),
        adam_b2=float(optimizer["adam_b2"]),
        adam_eps=float(optimizer["adam_eps"]),
        weight_decay=float(optimizer["weight_decay"]),
        clip_gradient_norm=float(optimizer["clip_gradient_norm"]),
        ema_decay=float(optimizer["ema_decay"]),
        overfit_steps=int(gates["overfit_steps"]),
        pilot_steps=int(gates["pilot_steps"]),
        overfit_max_loss_ratio=float(gates["overfit_max_loss_ratio"]),
    )


def inspect_jax_sft(spec: JaxSftSpec) -> dict[str, Any]:
    """Validate data, normalization, base checkpoint and numerical contract."""

    from clothdojo.training.openpi_artifacts import (
        inspect_openpi_dataset,
        inspect_openpi_norm_stats,
    )

    dataset = inspect_openpi_dataset(spec.dataset)
    data_kwargs = {
        "adapt_to_pi": False,
        "extra_delta_transform": True,
        "default_prompt": "Flatten the shirt.",
    }
    norm = inspect_openpi_norm_stats(
        spec.norm_stats,
        dataset=dataset,
        config_name=spec.openpi_config,
        data_kwargs=data_kwargs,
        action_horizon=50,
    )
    base = _inspect_orbax_params(spec.base_params)
    pytorch_base = _inspect_pytorch_base(spec.pytorch_base)
    tokenizer = spec.openpi_data_home / "big_vision" / "paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"cached PaliGemma tokenizer is missing: {tokenizer}")
    os.environ["OPENPI_DATA_HOME"] = str(spec.openpi_data_home)
    train_config = build_openpi_train_config(spec)
    _validate_built_config(train_config, spec)

    manifest = {
        "schema": "clothdojo_pi05_jax_preflight",
        "format_version": 1,
        "compatible": True,
        "host": platform.node(),
        "config_path": str(spec.config_path),
        "contract": dict(_CONTRACT),
        "training": {
            "backend": "OpenPI/JAX",
            "tuning_profile": spec.tuning_profile,
            "base_initialization": "required Orbax CheckpointWeightLoader",
            "parameter_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "ema_dtype": "float32",
            "compute_dtype": "bfloat16",
            "batch_size": spec.batch_size,
            "fsdp_devices": spec.fsdp_devices,
            "steps": spec.num_train_steps,
            "lr_schedule": {
                "warmup_steps": spec.warmup_steps,
                "peak_lr": spec.peak_lr,
                "decay_steps": spec.decay_steps,
                "floor_lr": spec.decay_lr,
            },
            "checkpoint_contract": {
                "params": "EMA inference weights",
                "train_state.params": "online training weights",
                "train_state.opt_state": "AdamW state",
            },
        },
        "dataset": dataset,
        "norm_stats": norm,
        "base_checkpoint": base,
        "pytorch_base_checkpoint": pytorch_base,
        "gates": {
            "overfit_steps": spec.overfit_steps,
            "pilot_steps": spec.pilot_steps,
            "overfit_max_loss_ratio": spec.overfit_max_loss_ratio,
        },
        "tokenizer": {"path": str(tokenizer), "sha256": _sha256(tokenizer)},
        "outputs": {
            "checkpoints": str(spec.checkpoint_dir),
            "gates": str(spec.gate_dir / spec.run_name),
        },
    }
    manifest["artifact_sha256"] = _manifest_identity(manifest)
    return manifest


def build_openpi_train_config(spec: JaxSftSpec) -> Any:
    """Build the exact OpenPI config with an explicit Orbax weight loader."""

    os.environ["OPENPI_DATA_HOME"] = str(spec.openpi_data_home)

    import flax.nnx as nnx
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.training import optimizer as openpi_optimizer
    from openpi.training import weight_loaders
    from openpi.training.config import (
        AssetsConfig,
        DataConfig,
        LeRobotAlohaDataConfig,
        TrainConfig,
    )

    data = LeRobotAlohaDataConfig(
        repo_id=str(spec.dataset),
        assets=AssetsConfig(
            assets_dir=str(spec.norm_stats.parent),
            asset_id=spec.norm_stats.name,
        ),
        base_config=DataConfig(prompt_from_task=True),
        repack_transforms=transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
        default_prompt="Flatten the shirt.",
        adapt_to_pi=False,
        use_delta_joint_actions=True,
        action_sequence_keys=("action",),
    )
    return TrainConfig(
        name=spec.openpi_config,
        project_name=spec.project_name,
        exp_name=spec.run_name,
        model=pi0_config.Pi0Config(pi05=True, discrete_state_input=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(spec.base_params)),
        lr_schedule=openpi_optimizer.CosineDecaySchedule(
            warmup_steps=spec.warmup_steps,
            peak_lr=spec.peak_lr,
            decay_steps=spec.decay_steps,
            decay_lr=spec.decay_lr,
        ),
        optimizer=openpi_optimizer.AdamW(
            b1=spec.adam_b1,
            b2=spec.adam_b2,
            eps=spec.adam_eps,
            weight_decay=spec.weight_decay,
            clip_gradient_norm=spec.clip_gradient_norm,
        ),
        ema_decay=spec.ema_decay,
        freeze_filter=nnx.Nothing,
        data=data,
        assets_base_dir=str(spec.norm_stats.parent),
        checkpoint_base_dir=str(spec.checkpoint_base_dir),
        seed=spec.seed,
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        num_train_steps=spec.num_train_steps,
        log_interval=spec.log_interval,
        save_interval=spec.save_interval,
        keep_period=spec.keep_period,
        overwrite=False,
        resume=False,
        wandb_enabled=spec.wandb_enabled,
        fsdp_devices=spec.fsdp_devices,
    )


def run_jax_sft(spec: JaxSftSpec, gate: Gate) -> dict[str, Any]:
    """Run one guarded stage; ``full`` is unavailable until all short gates pass."""

    preflight = inspect_jax_sft(spec)
    if gate == "full":
        _require_gate(spec, "initialize", preflight["artifact_sha256"])
        _require_gate(spec, "overfit", preflight["artifact_sha256"])
        _require_gate(spec, "pilot", preflight["artifact_sha256"])
        _require_gate(spec, "parity", preflight["artifact_sha256"])
    if gate == "parity":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return _train(spec, gate, preflight)


def gate_manifest_path(spec: JaxSftSpec, gate: Gate) -> Path:
    return spec.gate_dir / spec.run_name / f"{gate}.json"


def write_checkpoint_contract(step_dir: str | Path, *, step: int) -> Path:
    """Describe OpenPI's two weight views without duplicating tensor payloads."""

    root = Path(step_dir).resolve()
    params = root / "params"
    train_state = root / "train_state"
    for item in (params, train_state):
        if not item.is_dir():
            raise FileNotFoundError(f"incomplete OpenPI checkpoint item: {item}")
    payload = {
        "schema": "clothdojo_openpi_jax_checkpoint",
        "format_version": 1,
        "step": step,
        "ema": {
            "path": "params",
            "purpose": "inference and JAX-to-PyTorch conversion",
        },
        "online": {
            "path": "train_state",
            "field": "params",
            "purpose": "exact resume and optimizer-update analysis",
        },
        "optimizer": {"path": "train_state", "field": "opt_state"},
    }
    path = root / "clothdojo_weights.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _train(spec: JaxSftSpec, gate: Gate, preflight: dict[str, Any]) -> dict[str, Any]:
    # Heavy imports remain behind the explicit launch boundary.
    import functools

    import flax.nnx as nnx
    import flax.traverse_util as traverse_util
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import common_utils
    from openpi.shared import array_typing as at
    from openpi.shared import nnx_utils
    from openpi.training import checkpoints, data_loader, optimizer, sharding, utils

    config = build_openpi_train_config(spec)
    # The initialization and repeated-batch gates consume one batch only.
    # Spawning the production loader pool there adds minutes and tests nothing.
    if gate in {"initialize", "overfit", "parity"}:
        config = dataclasses.replace(config, num_workers=0)
    if spec.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"batch size {spec.batch_size} is not divisible by {jax.device_count()} devices"
        )
    if jax.device_count() != spec.fsdp_devices:
        raise RuntimeError(
            f"{gate} requires exactly {spec.fsdp_devices} visible devices, got {jax.device_count()}"
        )

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    loader = _create_openpi_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        video_backend=spec.video_backend,
        data_loader_module=data_loader,
    )
    iterator = iter(loader)
    fixed_batch = next(iterator)

    tx = optimizer.create_optimizer(config.optimizer, config.lr_schedule)

    def initialize(key: Any, partial_params: Any = None) -> utils.TrainState:
        key, model_rng = jax.random.split(key)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16)),
        )
        return utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=params,
        )

    state_shape = jax.eval_shape(initialize, init_rng)
    state_sharding = sharding.fsdp_sharding(state_shape, mesh)
    loaded = config.weight_loader.load(state_shape.params.to_pure_dict())
    at.check_pytree_equality(
        expected=state_shape.params.to_pure_dict(),
        got=loaded,
        check_shapes=True,
        check_dtypes=True,
    )
    loaded = traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(loaded).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )
    state = jax.jit(
        initialize,
        donate_argnums=(1,),
        in_shardings=replicated,
        out_shardings=state_sharding,
    )(init_rng, loaded)
    jax.block_until_ready(state)
    precision = _inspect_state_precision(state, jax)
    if gate == "initialize":
        result = {
            **_gate_base(gate, preflight),
            "compatible": True,
            "devices": [str(device) for device in jax.devices()],
            "precision": precision,
            "batch": _tree_shapes(fixed_batch, jax),
        }
        _write_gate(spec, gate, result)
        return result
    if gate == "parity":
        pytorch_base = _inspect_pytorch_base(spec.pytorch_base)
        result = {
            **_gate_base(gate, preflight),
            **_compare_jax_pytorch(
                config,
                state,
                state_sharding,
                replicated,
                fixed_batch,
                spec.pytorch_base,
                nnx=nnx,
                jax=jax,
                jnp=jnp,
            ),
            "pytorch_base_checkpoint": pytorch_base,
            "precision": precision,
        }
        _write_gate(spec, gate, result)
        if not result["compatible"]:
            raise RuntimeError("JAX/PyTorch base inference parity gate failed")
        return result

    def train_step(key: Any, current: Any, batch: Any) -> tuple[Any, dict[str, Any]]:
        model = nnx.merge(current.model_def, current.params)
        model.train()

        def loss_fn(model: Any, loss_rng: Any, observation: Any, actions: Any) -> Any:
            return jnp.mean(model.compute_loss(loss_rng, observation, actions, train=True))

        step_rng = jax.random.fold_in(key, current.step)
        diff_state = nnx.DiffState(0, config.trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            model, step_rng, *batch
        )
        params = current.params.filter(config.trainable_filter)
        updates, new_opt_state = current.tx.update(grads, current.opt_state, params)
        new_trainable = optax.apply_updates(params, updates)
        nnx.update(model, new_trainable)
        new_params = nnx.state(model)
        updated = dataclasses.replace(
            current,
            step=current.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            ema_params=jax.tree.map(
                lambda old, new: current.ema_decay * old
                + (1 - current.ema_decay) * new,
                current.ema_params,
                new_params,
            ),
        )
        return updated, {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "learning_rate": config.lr_schedule.create()(current.step),
        }

    ptrain_step = jax.jit(
        functools.partial(train_step),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    steps = {
        "overfit": spec.overfit_steps,
        "pilot": spec.pilot_steps,
        "full": spec.num_train_steps,
    }[gate]
    manager = None
    if gate == "full":
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=False,
            resume=False,
        )
        if resuming:
            raise AssertionError("a fresh full reference run cannot resume")
        _write_launch_manifest(config.checkpoint_dir, preflight)

    wandb_run = _init_wandb(spec, gate, preflight)
    started = time.monotonic()
    metrics: list[dict[str, float]] = []
    window: list[dict[str, Any]] = []
    log_interval = min(spec.log_interval, 10) if gate != "full" else spec.log_interval
    try:
        for _ in range(steps):
            batch = fixed_batch if gate == "overfit" else next(iterator)
            with sharding.set_mesh(mesh):
                state, info = ptrain_step(train_rng, state, batch)
            window.append(info)
            completed = int(state.step)
            if completed % log_interval == 0 or completed == steps:
                stacked = common_utils.stack_forest(window)
                reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
                record = {name: float(value) for name, value in reduced.items()}
                record["step"] = completed
                record["samples_seen"] = completed * spec.batch_size
                record["samples_per_second"] = (
                    completed * spec.batch_size / max(time.monotonic() - started, 1e-9)
                )
                metrics.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(record, step=completed)
                window = []
            if manager is not None and (
                completed % spec.save_interval == 0 or completed == steps
            ):
                checkpoints.save_state(manager, state, loader, completed)
                manager.wait_until_finished()
                write_checkpoint_contract(config.checkpoint_dir / str(completed), step=completed)
    finally:
        if manager is not None:
            manager.wait_until_finished()
        if wandb_run is not None:
            wandb_run.finish()

    losses = [record["loss"] for record in metrics]
    compatible = bool(losses) and all(math.isfinite(value) for value in losses)
    if gate == "overfit":
        compatible = compatible and losses[-1] <= losses[0] * spec.overfit_max_loss_ratio
    elif gate == "pilot":
        compatible = compatible and losses[-1] < losses[0]
    result = {
        **_gate_base(gate, preflight),
        "compatible": compatible,
        "steps": steps,
        "elapsed_seconds": time.monotonic() - started,
        "precision": precision,
        "metrics": metrics,
    }
    _write_gate(spec, gate, result)
    if not compatible:
        raise RuntimeError(f"{gate} gate did not meet its loss acceptance criterion")
    return result


def _init_wandb(spec: JaxSftSpec, gate: Gate, preflight: dict[str, Any]) -> Any:
    if not spec.wandb_enabled:
        return None
    import wandb

    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    wandb_dir = spec.gate_dir / spec.run_name / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        dir=str(wandb_dir),
        project=spec.project_name,
        group=spec.wandb_group,
        name=f"{spec.run_name}-{gate}",
        config={"preflight": preflight, "gate": gate},
        save_code=False,
    )
    run.define_metric("samples_seen")
    for name in ("loss", "grad_norm", "learning_rate", "samples_per_second"):
        run.define_metric(name, step_metric="samples_seen")
    return run


def _compare_jax_pytorch(
    config: Any,
    state: Any,
    state_sharding: Any,
    replicated: Any,
    fixed_batch: Any,
    pytorch_base: Path,
    *,
    nnx: Any,
    jax: Any,
    jnp: Any,
) -> dict[str, Any]:
    """Compare one deterministic Euler vector-field step across both backends."""

    import numpy as np
    import safetensors.torch
    import torch
    from openpi.models import model as model_module
    from openpi.models import pi0_config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    observation = jax.device_get(jax.tree.map(lambda value: value[:1], fixed_batch[0]))
    noise = np.random.default_rng(260825).standard_normal((1, 50, 32)).astype(np.float32)
    jax_observation = jax.tree.map(jnp.asarray, observation)
    jax_noise = jnp.asarray(noise)

    def jax_infer(params: Any, obs: Any, fixed_noise: Any) -> Any:
        model = nnx.merge(state.model_def, params)
        model.eval()
        return model.sample_actions(
            jax.random.key(0),
            obs,
            num_steps=1,
            noise=fixed_noise,
        )

    jax_action = jax.jit(
        jax_infer,
        in_shardings=(state_sharding.params, replicated, replicated),
        out_shardings=replicated,
    )(state.params, jax_observation, jax_noise)
    jax_action = np.asarray(jax.device_get(jax_action), dtype=np.float32)

    device = torch.device("cuda:0")
    torch_model = PI0Pytorch(
        pi0_config.Pi0Config(pi05=True, discrete_state_input=True, dtype="float32")
    )
    missing, unexpected = safetensors.torch.load_model(
        torch_model,
        str(pytorch_base / "model.safetensors"),
        strict=False,
    )
    torch_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    torch_model = torch_model.to(device).eval()

    def tensor(value: Any) -> Any:
        # JAX batches expose read-only NumPy views.  Copy them so that Torch
        # never receives a tensor whose write semantics are undefined.
        return torch.as_tensor(np.array(value, copy=True), device=device)

    def image_tensor(value: Any) -> Any:
        # The JAX model consumes NHWC, while PI0Pytorch's vision tower consumes
        # NCHW after its preprocessing helper.  Supplying float NHWC here would
        # bypass the uint8 conversion in Observation.from_dict and reach the
        # convolution with the channel dimension in the wrong position.
        return tensor(value).permute(0, 3, 1, 2).contiguous()

    torch_observation = model_module.Observation(
        images={name: image_tensor(value) for name, value in observation.images.items()},
        image_masks={name: tensor(value) for name, value in observation.image_masks.items()},
        state=tensor(observation.state),
        tokenized_prompt=tensor(observation.tokenized_prompt),
        tokenized_prompt_mask=tensor(observation.tokenized_prompt_mask),
    )
    with torch.no_grad():
        torch_action = PI0Pytorch.sample_actions(
            torch_model,
            device,
            torch_observation,
            noise=tensor(noise),
            num_steps=1,
        )
    torch_action = torch_action.float().cpu().numpy()
    difference = np.abs(jax_action - torch_action)
    mean_abs = float(difference.mean())
    max_abs = float(difference.max())
    return {
        "compatible": mean_abs <= 0.03 and max_abs <= 0.25,
        "comparison": "one deterministic inference Euler step",
        "batch_size": 1,
        "mean_abs_error": mean_abs,
        "max_abs_error": max_abs,
        "jax_action_l2": float(np.linalg.norm(jax_action)),
        "pytorch_action_l2": float(np.linalg.norm(torch_action)),
        "pytorch_missing_keys": sorted(missing),
        "pytorch_unexpected_keys": sorted(unexpected),
    }


def _create_openpi_data_loader(
    config: Any,
    *,
    sharding: Any,
    shuffle: bool,
    video_backend: str,
    data_loader_module: Any,
) -> Any:
    """Create OpenPI's loader while selecting LeRobot's decoder explicitly."""

    from openpi import transforms

    data_config = config.data.create(config.assets_dirs, config.model)
    lerobot = data_loader_module._import_lerobot_dataset()
    metadata = lerobot.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [index / metadata.fps for index in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        video_backend=video_backend,
    )
    if data_config.prompt_from_task:
        dataset = data_loader_module.TransformedDataset(
            dataset,
            [transforms.PromptFromLeRobotTask(metadata.tasks)],
        )
    dataset = data_loader_module.transform_dataset(dataset, data_config)
    loader = data_loader_module.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_workers=config.num_workers,
        seed=config.seed,
        framework="jax",
    )
    return data_loader_module.DataLoaderImpl(data_config, loader)


def _inspect_state_precision(state: Any, jax: Any) -> dict[str, Any]:
    def floating_dtypes(tree: Any) -> list[str]:
        return sorted(
            {
                str(leaf.dtype)
                for leaf in jax.tree.leaves(tree)
                if hasattr(leaf, "dtype") and leaf.dtype.kind == "f"
            }
        )

    result = {
        "params": floating_dtypes(state.params),
        "ema_params": floating_dtypes(state.ema_params),
        "optimizer_state": floating_dtypes(state.opt_state),
    }
    if result != {
        "params": ["float32"],
        "ema_params": ["float32"],
        "optimizer_state": ["float32"],
    }:
        raise TypeError(f"unexpected JAX update precision: {result}")
    return result


def _tree_shapes(tree: Any, jax: Any) -> list[dict[str, Any]]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [
        {
            "path": jax.tree_util.keystr(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for path, value in leaves
        if hasattr(value, "shape")
    ]


def _write_launch_manifest(root: Path, preflight: dict[str, Any]) -> None:
    path = root / "clothdojo_launch.json"
    path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate_base(gate: Gate, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "clothdojo_pi05_jax_gate",
        "format_version": 1,
        "gate": gate,
        "preflight_sha256": preflight["artifact_sha256"],
    }


def _write_gate(spec: JaxSftSpec, gate: Gate, result: dict[str, Any]) -> Path:
    path = gate_manifest_path(spec, gate)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _require_gate(spec: JaxSftSpec, gate: Gate, identity: str) -> None:
    path = gate_manifest_path(spec, gate)
    payload = _read_object(path, f"{gate} gate")
    if payload.get("compatible") is not True:
        raise RuntimeError(f"required {gate} gate did not pass: {path}")
    if payload.get("preflight_sha256") != identity:
        raise RuntimeError(f"required {gate} gate belongs to a different artifact set: {path}")


def _inspect_orbax_params(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Pi0.5 JAX base params do not exist: {root}")
    metadata = root / "_METADATA"
    checkpoint_metadata = root / "_CHECKPOINT_METADATA"
    for path in (metadata, checkpoint_metadata):
        if not path.is_file():
            raise FileNotFoundError(f"Pi0.5 JAX base checkpoint is incomplete: {path}")
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    symlinks = [path for path in entries if path.is_symlink()]
    if symlinks:
        raise ValueError(f"Pi0.5 JAX base checkpoint may not contain symlinks: {symlinks[0]}")
    files = [path for path in entries if path.is_file()]
    if not files:
        raise ValueError(f"Pi0.5 JAX base checkpoint has no payload: {root}")
    payload = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        payload.update(relative.encode("utf-8"))
        payload.update(b"\0")
        payload.update(_sha256(path).encode("ascii"))
        payload.update(b"\n")
    return {
        "params_path": str(root),
        "metadata_sha256": _sha256(metadata),
        "checkpoint_metadata_sha256": _sha256(checkpoint_metadata),
        "payload_sha256": payload.hexdigest(),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "loader": "openpi.training.weight_loaders.CheckpointWeightLoader",
    }


def _inspect_pytorch_base(root: Path) -> dict[str, Any]:
    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    config = _read_object(config_path, "Pi0.5 PyTorch base config")
    expected = {
        "type": "pi05",
        "dtype": "float32",
        "chunk_size": 50,
        "max_action_dim": 32,
        "max_state_dim": 32,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Pi0.5 PyTorch base mismatch for {key}: {config.get(key)!r} != {value!r}"
            )
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pi0.5 PyTorch base weights are missing: {weights_path}")
    return {
        "root": str(root),
        "config_sha256": _sha256(config_path),
        "weights_sha256": _sha256(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "source_revision": root.name if len(root.name) == 40 else None,
    }


def _validate_built_config(config: Any, spec: JaxSftSpec) -> None:
    values = {
        "weight_loader": getattr(config.weight_loader, "params_path", None),
        "repo_id": config.data.repo_id,
        "adapt_to_pi": config.data.adapt_to_pi,
        "extra_delta_transform": config.data.use_delta_joint_actions,
        "batch_size": config.batch_size,
        "num_train_steps": config.num_train_steps,
        "log_interval": config.log_interval,
        "save_interval": config.save_interval,
        "num_workers": config.num_workers,
        "fsdp_devices": config.fsdp_devices,
        "ema_decay": config.ema_decay,
        "checkpoint_dir": str(config.checkpoint_dir),
        "model_dtype": str(config.model.dtype),
        "action_dim": config.model.action_dim,
        "action_horizon": config.model.action_horizon,
        "pi05": config.model.pi05,
        "warmup_steps": config.lr_schedule.warmup_steps,
        "peak_lr": config.lr_schedule.peak_lr,
        "decay_steps": config.lr_schedule.decay_steps,
        "decay_lr": config.lr_schedule.decay_lr,
    }
    expected = {
        "weight_loader": str(spec.base_params),
        "repo_id": str(spec.dataset),
        "adapt_to_pi": False,
        "extra_delta_transform": True,
        "batch_size": spec.batch_size,
        "num_train_steps": spec.num_train_steps,
        "log_interval": spec.log_interval,
        "save_interval": spec.save_interval,
        "num_workers": 16,
        "fsdp_devices": 8,
        "ema_decay": 0.99,
        "checkpoint_dir": str(spec.checkpoint_dir),
        "model_dtype": "bfloat16",
        "action_dim": 32,
        "action_horizon": 50,
        "pi05": True,
        "warmup_steps": spec.warmup_steps,
        "peak_lr": spec.peak_lr,
        "decay_steps": spec.decay_steps,
        "decay_lr": spec.decay_lr,
    }
    if values != expected:
        raise ValueError(f"built OpenPI config differs from the JAX SFT contract: {values}")


def _validate_contract(raw: Mapping[str, Any]) -> None:
    for key, expected in _CONTRACT.items():
        actual = _nested(raw, key)
        if actual != expected:
            raise ValueError(f"JAX SFT contract mismatch for {key}: {actual!r} != {expected!r}")


def _validate_tuning_profile(raw: Mapping[str, Any], profile: str) -> None:
    try:
        expected = _TUNING_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(_TUNING_PROFILES))
        raise ValueError(
            f"unknown JAX SFT tuning profile {profile!r}; choose one of: {choices}"
        ) from exc
    actual = {key: _nested(raw, key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"optimizer values do not match tuning profile {profile!r}: "
            f"{actual!r} != {expected!r}"
        )


def _nested(mapping: Mapping[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"missing JAX SFT config value: {dotted}")
        value = value[part]
    return value


def _mapping(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"JAX SFT config {name} must be a mapping")
    return value


def _path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _manifest_identity(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
