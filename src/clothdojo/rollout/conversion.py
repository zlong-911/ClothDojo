"""One-time OpenPI/JAX to checked rollout-checkpoint conversion."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clothdojo.rollout.artifacts import (
    find_training_manifest,
    validate_sft_norm_binding,
)
from clothdojo.rollout.checkpoint import inspect_rlinf_openpi_checkpoint
from clothdojo.rollout.newton import POLICY_ACTION_DIM


def convert_jax_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    norm_stats: str | Path,
    parity_config: str | Path,
    *,
    openpi_config: str = "pi05_aloha_robotwin",
    action_horizon: int = 50,
    num_steps: int = 10,
    parity_batch_size: int = 4,
    parity_execution_horizon: int = 10,
    _converter: Callable[..., Any] | None = None,
    _parity_checker: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert EMA Orbax params and publish only after numerical parity passes."""

    source = Path(checkpoint).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    norm_path = Path(norm_stats).expanduser().resolve()
    parity_config_path = Path(parity_config).expanduser().resolve()
    if not source.is_dir() or not (source / "params").is_dir():
        raise FileNotFoundError(f"JAX checkpoint has no EMA params directory: {source}")
    if destination.exists():
        raise FileExistsError(f"converted checkpoint output already exists: {destination}")
    if action_horizon < 1 or num_steps < 1:
        raise ValueError("action_horizon and num_steps must be positive")
    if parity_batch_size < 1 or not 1 <= parity_execution_horizon <= action_horizon:
        raise ValueError("invalid conversion parity batch or execution horizon")
    if not parity_config_path.is_file():
        raise FileNotFoundError(f"parity training config does not exist: {parity_config_path}")
    if not openpi_config.strip():
        raise ValueError("OpenPI rollout config must not be empty")
    launch_manifest = find_training_manifest(source)
    staging = destination.parent / f".{destination.name}.tmp"
    if staging.exists():
        raise FileExistsError(f"unfinished checkpoint conversion exists: {staging}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _converter is None:
        _converter = _convert_to_openpi_rlinf
    if _parity_checker is None:
        from clothdojo.rollout.parity import check_jax_pytorch_conversion

        _parity_checker = check_jax_pytorch_conversion

    try:
        mapping = _converter(
            checkpoint_dir=str(source),
            output_path=str(staging),
            norm_stats=str(norm_path),
            openpi_config=openpi_config,
            action_horizon=action_horizon,
        )
        if (
            not isinstance(mapping, dict)
            or mapping.get("schema") != "clothdojo_jax_openpi_rlinf_mapping"
            or mapping.get("compatible") is not True
        ):
            raise RuntimeError("converter did not prove complete JAX/PyTorch weight mapping")
        if not staging.is_dir():
            raise RuntimeError("the pinned RLINF converter did not create its output")
        shutil.copy2(launch_manifest, staging / "clothdojo_launch.json")
        checkpoint_manifest = inspect_rlinf_openpi_checkpoint(
            staging,
            openpi_config=openpi_config,
            action_horizon=action_horizon,
            action_env_dim=POLICY_ACTION_DIM,
            num_steps=num_steps,
        )
        norm_identity = validate_sft_norm_binding(
            staging,
            norm_path,
            openpi_config=openpi_config,
            action_horizon=action_horizon,
        )
        parity = _parity_checker(
            source,
            staging,
            norm_path,
            parity_config_path,
            batch_size=parity_batch_size,
            execution_horizon=parity_execution_horizon,
        )
        if (
            parity.get("schema") != "clothdojo_jax_openpi_rlinf_parity"
            or parity.get("format_version") != 1
            or parity.get("compatible") is not True
            or parity.get("weights_sha256") != checkpoint_manifest["weights_sha256"]
            or parity.get("norm_stats_sha256") != norm_identity["norm_stats_sha256"]
        ):
            raise RuntimeError("conversion parity report does not bind the staged checkpoint")
        parity_path = staging / "clothdojo_parity.json"
        parity_path.write_text(
            json.dumps(parity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Exercise the same fail-closed boundary used by the policy service before
        # making the staged directory visible to rollout clients.
        norm_identity = validate_sft_norm_binding(
            staging,
            norm_path,
            openpi_config=openpi_config,
            action_horizon=action_horizon,
            weights_sha256=checkpoint_manifest["weights_sha256"],
        )
        conversion = {
            "schema": "clothdojo_openpi_rollout_checkpoint",
            "format_version": 1,
            "source_jax_checkpoint": str(source),
            "source_launch_manifest_sha256": _sha256(launch_manifest),
            "openpi_config": openpi_config,
            "checkpoint_precision": "float32",
            "runtime_precision": "bfloat16",
            "weights_sha256": checkpoint_manifest["weights_sha256"],
            "weight_mapping": mapping,
            "strict_layout": True,
            "norm_stats_sha256": norm_identity["norm_stats_sha256"],
            "parity_report_sha256": _sha256(parity_path),
        }
        (staging / "clothdojo_conversion.json").write_text(
            json.dumps(conversion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return {
        "compatible": True,
        "checkpoint": str(destination),
        **conversion,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _convert_to_openpi_rlinf(
    *,
    checkpoint_dir: str,
    output_path: str,
    norm_stats: str,
    openpi_config: str,
    action_horizon: int,
) -> dict[str, Any]:
    """Map OpenPI's EMA params directly into RLinf's current Pi0 core."""

    import numpy as np
    import torch
    from openpi.models import model as openpi_model
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config
    from rlinf.utils.ckpt_convertor.openpi import jax_to_openpi_rlinf as mapper
    from rlinf.utils.ckpt_convertor.openpi._core import (
        copy_norm_stats,
        save_safetensors,
        write_config_json,
    )

    params = openpi_model.restore_params(
        Path(checkpoint_dir) / "params",
        restore_type=np.ndarray,
        dtype=np.float32,
    )
    state_dict: dict[str, torch.Tensor] = {}
    for converted in (
        mapper.convert_siglip(params),
        mapper.convert_llm(params, pi05=True),
        mapper.convert_projections(params, pi05=True),
    ):
        overlap = set(state_dict).intersection(converted)
        if overlap:
            raise RuntimeError(f"duplicate converted keys: {sorted(overlap)}")
        state_dict.update(
            {key: value.contiguous() for key, value in converted.items()}
        )
    del params
    config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=action_horizon,
        max_token_len=200,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="float32",
        pcd=False,
    )
    model = config.create()
    model.load_state_dict(state_dict, strict=True)
    if {str(value.dtype) for value in state_dict.values()} != {"torch.float32"}:
        raise TypeError("converted checkpoint must preserve FP32 source weights")
    destination = Path(output_path)
    destination.mkdir(parents=True)
    save_safetensors(state_dict, destination / "model.safetensors")
    write_config_json(dataclasses.asdict(config), destination)
    norm_file = Path(norm_stats)
    if norm_file.is_dir():
        norm_file = norm_file / "norm_stats.json"
    copy_norm_stats(
        norm_file,
        destination / openpi_config / "norm_stats.json",
    )
    mapped = len(state_dict)
    del model
    del state_dict
    return {
        "schema": "clothdojo_jax_openpi_rlinf_mapping",
        "compatible": True,
        "mapped_tensors": mapped,
        "missing": [],
        "unexpected": [],
        "strict_load": True,
    }
