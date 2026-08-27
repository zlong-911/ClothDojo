"""Numerical acceptance gate for one OpenPI JAX-to-PyTorch conversion."""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.rollout.checkpoint import rlinf_openpi_weights_sha256
from clothdojo.rollout.newton import POLICY_ACTION_DIM


@dataclass(frozen=True, slots=True)
class ConversionParityLimits:
    """Errors small enough to treat conversion as irrelevant to control."""

    model_mean_abs: float = 0.005
    model_p99_abs: float = 0.02
    model_max_abs: float = 0.05
    executed_mean_abs: float = 0.003
    executed_p99_abs: float = 0.012
    executed_max_abs: float = 0.02
    minimum_cosine: float = 0.9998


def check_jax_pytorch_conversion(
    jax_checkpoint: str | Path,
    pytorch_checkpoint: str | Path,
    norm_stats: str | Path,
    training_config: str | Path,
    *,
    batch_size: int = 4,
    execution_horizon: int = 10,
    seed: int = 260826,
    limits: ConversionParityLimits | None = None,
) -> dict[str, Any]:
    """Compare both backends on the same real, normalized observations and noise."""

    if batch_size < 1:
        raise ValueError("parity batch_size must be positive")
    if execution_horizon < 1:
        raise ValueError("parity execution_horizon must be positive")
    limits = limits or ConversionParityLimits()

    # Heavy framework imports stay behind the explicit one-time conversion gate.
    import dataclasses as dc

    import jax
    import jax.numpy as jnp
    import safetensors.torch
    import torch
    from openpi import transforms
    from openpi.models import model as model_module
    from openpi.training import checkpoints, data_loader
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import (
        Observation as TorchObservation,
    )
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config

    from clothdojo.training.jax_sft import (
        _create_openpi_data_loader,
        build_openpi_train_config,
        compose_jax_sft_spec,
    )

    source = Path(jax_checkpoint).expanduser().resolve()
    target = Path(pytorch_checkpoint).expanduser().resolve()
    norm_path = Path(norm_stats).expanduser().resolve()
    spec = compose_jax_sft_spec(training_config)
    if not spec.norm_stats.samefile(norm_path):
        raise ValueError(
            "parity config norm_stats differs from conversion norm_stats: "
            f"{spec.norm_stats} != {norm_path}"
        )
    config = dc.replace(
        build_openpi_train_config(spec),
        batch_size=batch_size,
        num_workers=0,
    )
    loader = _create_openpi_data_loader(
        config,
        sharding=None,
        shuffle=False,
        video_backend=spec.video_backend,
        data_loader_module=data_loader,
    )
    observation, _ = next(iter(loader))
    if execution_horizon > config.model.action_horizon:
        raise ValueError("parity execution_horizon exceeds model action_horizon")
    if 2 * execution_horizon > config.model.action_horizon:
        raise ValueError("parity execution and RTC guidance horizons exceed the model horizon")

    jax_model = config.model.load(
        # Match the pinned OpenPI policy loader and ``policy serve-jax`` exactly.
        model_module.restore_params(source / "params", dtype=jnp.bfloat16)
    )
    jax_model.eval()
    weights_sha256 = rlinf_openpi_weights_sha256(target)
    torch_model = Pi0Config(
        pi05=True,
        action_dim=config.model.action_dim,
        action_horizon=config.model.action_horizon,
        max_token_len=config.model.max_token_len,
        paligemma_variant=config.model.paligemma_variant,
        action_expert_variant=config.model.action_expert_variant,
        dtype="bfloat16",
        pcd=False,
    ).create()
    state_dict = safetensors.torch.load_file(
        str(target / "model.safetensors"), device="cpu"
    )
    torch_model.load_state_dict(state_dict, strict=True)
    del state_dict
    device = torch.device("cuda:0")
    torch_model = (
        torch_model.to(device=device, dtype=torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )

    def tensor(value: Any) -> Any:
        return torch.as_tensor(np.array(value, copy=True), device=device)

    torch_observation = TorchObservation(
        images={name: tensor(value) for name, value in observation.images.items()},
        image_masks={
            name: tensor(value) for name, value in observation.image_masks.items()
        },
        state=tensor(observation.state),
        tokenized_prompt=tensor(observation.tokenized_prompt),
        tokenized_prompt_mask=tensor(observation.tokenized_prompt_mask),
        token_ar_mask=None,
        token_loss_mask=None,
        pcd_xyz=None,
    )
    noise = np.random.default_rng(seed).standard_normal(
        (batch_size, config.model.action_horizon, config.model.action_dim)
    ).astype(np.float32)

    metrics: dict[str, dict[str, Any]] = {}
    actions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for num_steps in (1, 10):
        with jax.default_matmul_precision("highest"):
            jax_actions = np.asarray(
                jax.device_get(
                    jax_model.sample_actions(
                        jax.random.key(0),
                        observation,
                        num_steps=num_steps,
                        noise=jnp.asarray(noise),
                    )
                ),
                dtype=np.float32,
            )
        with torch.no_grad():
            torch_actions = (
                torch_model.sample_actions(
                    torch_observation,
                    noise=tensor(noise),
                    num_steps=num_steps,
                )
                .float()
                .cpu()
                .numpy()
            )
        actions[num_steps] = (jax_actions, torch_actions)
        metrics[f"model_{num_steps}_step_full"] = _error_metrics(
            jax_actions, torch_actions
        )
        metrics[f"model_{num_steps}_step_executed"] = _error_metrics(
            jax_actions[:, :execution_horizon, :POLICY_ACTION_DIM],
            torch_actions[:, :execution_horizon, :POLICY_ACTION_DIM],
        )
        if num_steps == 10:
            metrics["model_10_step_rtc_guidance"] = _error_metrics(
                jax_actions[:, execution_horizon : 2 * execution_horizon],
                torch_actions[:, execution_horizon : 2 * execution_horizon],
            )

    jax_actions, torch_actions = actions[10]
    norm_dir = norm_path if norm_path.is_dir() else norm_path.parent
    norm_values = checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    data_config = config.data.create(config.assets_dirs, config.model)
    jax_output_transform = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_values,
                use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.data_transforms.outputs,
        ]
    )
    jax_final = _jax_final_actions(
        jax_actions,
        np.asarray(observation.state),
        jax_output_transform,
    )
    torch_final = _jax_final_actions(
        torch_actions,
        np.asarray(observation.state),
        jax_output_transform,
    )
    metrics["final_action_full"] = _error_metrics(jax_final, torch_final)
    metrics["final_action_executed"] = _error_metrics(
        jax_final[:, :execution_horizon],
        torch_final[:, :execution_horizon],
    )
    violations = _parity_violations(metrics, limits)
    report = {
        "schema": "clothdojo_jax_openpi_rlinf_parity",
        "format_version": 1,
        "compatible": not violations,
        "source_jax_checkpoint": str(source),
        "pytorch_checkpoint": str(target),
        "weights_sha256": weights_sha256,
        "norm_stats_sha256": _sha256(
            norm_path / "norm_stats.json" if norm_path.is_dir() else norm_path
        ),
        "training_config": str(Path(training_config).expanduser().resolve()),
        "batch_size": batch_size,
        "seed": seed,
        "execution_horizon": execution_horizon,
        "num_steps": [1, 10],
        "precision": {
            "jax_checkpoint": "float32",
            "jax_compute": "bfloat16",
            "pytorch_checkpoint": "float32",
            "pytorch_compute": "native_bfloat16",
        },
        "limits": dataclasses.asdict(limits),
        "metrics": metrics,
        "violations": violations,
    }
    if violations:
        raise RuntimeError(f"JAX/PyTorch conversion parity failed: {violations}")
    return report


def _jax_final_actions(
    actions: np.ndarray,
    states: np.ndarray,
    output_transform: Any,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(
                output_transform({"actions": action, "state": state})["actions"],
                dtype=np.float32,
            )
            for action, state in zip(actions, states, strict=True)
        ]
    )


def _error_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        raise ValueError(f"parity shapes differ: {reference.shape} != {actual.shape}")
    difference = np.abs(reference - actual)
    max_flat_index = int(difference.argmax())
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    cosine = float(np.sum(reference * actual) / denominator) if denominator else 1.0
    return {
        "shape": list(reference.shape),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(np.quantile(difference, 0.99)),
        "max_abs": float(difference.max()),
        "max_abs_index": [
            int(index) for index in np.unravel_index(max_flat_index, difference.shape)
        ],
        "rmse": float(np.sqrt(np.mean(np.square(reference - actual)))),
        "cosine": cosine,
    }


def _parity_violations(
    metrics: dict[str, dict[str, Any]],
    limits: ConversionParityLimits,
) -> list[str]:
    violations: list[str] = []
    for name in ("model_1_step_full", "model_10_step_full"):
        value = metrics[name]
        for field, limit in (
            ("mean_abs", limits.model_mean_abs),
            ("p99_abs", limits.model_p99_abs),
        ):
            if value[field] > limit:
                violations.append(f"{name}.{field}={value[field]:.6g}>{limit:.6g}")
        if value["cosine"] < limits.minimum_cosine:
            violations.append(
                f"{name}.cosine={value['cosine']:.6g}<{limits.minimum_cosine:.6g}"
            )
    for name in (
        "model_10_step_executed",
        "model_10_step_rtc_guidance",
        "final_action_executed",
    ):
        value = metrics[name]
        for field, limit in (
            ("mean_abs", limits.executed_mean_abs),
            ("p99_abs", limits.executed_p99_abs),
            ("max_abs", limits.executed_max_abs),
        ):
            if value[field] > limit:
                violations.append(f"{name}.{field}={value[field]:.6g}>{limit:.6g}")
        if value["cosine"] < limits.minimum_cosine:
            violations.append(
                f"{name}.cosine={value['cosine']:.6g}<{limits.minimum_cosine:.6g}"
            )
    return violations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
