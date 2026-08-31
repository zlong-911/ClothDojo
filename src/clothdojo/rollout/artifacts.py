"""Artifact identity checks shared by frozen policy serving boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def validate_sft_norm_binding(
    checkpoint: str | Path,
    norm_stats: str | Path,
    *,
    openpi_config: str,
    action_horizon: int,
    weights_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind one served SFT checkpoint to the exact norm artifacts used at launch."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    norm_path = Path(norm_stats).expanduser().resolve()
    norm_file = norm_path / "norm_stats.json" if norm_path.is_dir() else norm_path
    if not norm_file.is_file():
        raise FileNotFoundError(f"norm_stats.json does not exist: {norm_path}")
    provenance_file = norm_file.parent / "clothdojo.json"
    if not provenance_file.is_file():
        raise FileNotFoundError(
            f"norm-stat provenance clothdojo.json does not exist: {provenance_file}"
        )

    manifest_file = find_training_manifest(checkpoint_path)
    manifest = _read_json_object(manifest_file, "Pi0.5 launch manifest")
    if (
        manifest.get("schema")
        not in {"clothdojo_pi05_launch_preflight", "clothdojo_pi05_jax_preflight"}
        or manifest.get("format_version") != 1
        or manifest.get("compatible") is not True
    ):
        raise ValueError(f"unsupported Pi0.5 launch manifest: {manifest_file}")
    expected = manifest.get("norm_stats")
    if not isinstance(expected, dict):
        raise ValueError(f"Pi0.5 launch manifest is missing norm_stats: {manifest_file}")

    stats_sha256 = _file_sha256(norm_file)
    provenance_sha256 = _file_sha256(provenance_file)
    for name, actual in (
        ("stats_sha256", stats_sha256),
        ("provenance_sha256", provenance_sha256),
    ):
        expected_digest = expected.get(name)
        _require_sha256(expected_digest, f"launch manifest norm {name}")
        if actual != expected_digest:
            label = "norm_stats.json" if name == "stats_sha256" else "norm provenance"
            raise ValueError(
                f"{label} SHA-256 differs from the checkpoint launch manifest: "
                f"{actual} != {expected_digest}"
            )

    provenance = _read_json_object(provenance_file, "OpenPI norm-stat provenance")
    if (
        provenance.get("schema") != "clothdojo_openpi_norm_stats"
        or provenance.get("format_version") != 1
    ):
        raise ValueError(f"unsupported OpenPI norm-stat provenance: {provenance_file}")
    semantic_fields = (
        "dataset_info_sha256",
        "dataset_sidecar_sha256",
        "config_name",
        "data_kwargs",
        "action_horizon",
        "frames",
        "episodes",
    )
    mismatches = {
        name: {"expected": expected.get(name), "actual": provenance.get(name)}
        for name in semantic_fields
        if provenance.get(name) != expected.get(name)
    }
    if mismatches:
        raise ValueError(f"norm-stat provenance differs from launch manifest: {mismatches}")
    if provenance.get("config_name") != openpi_config:
        raise ValueError(
            "served OpenPI config differs from norm-stat provenance: "
            f"{openpi_config!r} != {provenance.get('config_name')!r}"
        )
    if provenance.get("action_horizon") != action_horizon:
        raise ValueError(
            "served action horizon differs from norm-stat provenance: "
            f"{action_horizon} != {provenance.get('action_horizon')!r}"
        )
    for name in ("dataset_info_sha256", "dataset_sidecar_sha256"):
        _require_sha256(provenance.get(name), f"norm provenance {name}")
    for name in ("frames", "episodes"):
        value = provenance.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"norm provenance {name} must be a positive integer")
    data_kwargs = provenance.get("data_kwargs")
    if (
        not isinstance(data_kwargs, dict)
        or set(data_kwargs) != {
            "adapt_to_pi",
            "extra_delta_transform",
            "default_prompt",
        }
        or data_kwargs.get("adapt_to_pi") is not False
        or data_kwargs.get("extra_delta_transform") is not True
        or not isinstance(data_kwargs.get("default_prompt"), str)
        or not data_kwargs["default_prompt"].strip()
    ):
        raise ValueError(
            "norm-stat transform contract differs from ClothDojo Pi0.5 ALOHA: "
            f"{data_kwargs!r}"
        )

    result = {
        "source_framework": (
            "jax" if manifest.get("schema") == "clothdojo_pi05_jax_preflight" else "pytorch"
        ),
        "norm_stats_sha256": stats_sha256,
        "norm_provenance_sha256": provenance_sha256,
        "training_manifest_sha256": _file_sha256(manifest_file),
        "dataset_info_sha256": provenance["dataset_info_sha256"],
        "dataset_sidecar_sha256": provenance["dataset_sidecar_sha256"],
    }
    if manifest.get("schema") == "clothdojo_pi05_jax_preflight" and weights_sha256:
        result.update(
            _validate_jax_conversion_parity(
                checkpoint_path,
                weights_sha256=weights_sha256,
                norm_stats_sha256=stats_sha256,
            )
        )
    return result


def validate_jax_reference_binding(
    checkpoint: str | Path,
    norm_stats: str | Path,
    training_config: str | Path,
    *,
    action_horizon: int,
) -> dict[str, Any]:
    """Bind raw JAX inference to its training config and normalization artifacts."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config_path = Path(training_config).expanduser().resolve()
    identity = validate_sft_norm_binding(
        checkpoint_path,
        norm_stats,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=action_horizon,
    )
    if identity["source_framework"] != "jax":
        raise ValueError("raw JAX serving requires a JAX launch manifest")
    manifest_file = find_training_manifest(checkpoint_path)
    manifest = _read_json_object(manifest_file, "Pi0.5 JAX launch manifest")
    configured_path = manifest.get("config_path")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError(f"JAX launch manifest has no training config: {manifest_file}")
    if Path(configured_path).expanduser().resolve() != config_path:
        raise ValueError(
            "JAX serving config differs from the checkpoint launch manifest: "
            f"{config_path} != {configured_path}"
        )
    from clothdojo.training.jax_sft import compose_jax_sft_spec

    spec = compose_jax_sft_spec(config_path)
    if not spec.norm_stats.samefile(Path(norm_stats).expanduser().resolve()):
        raise ValueError("JAX serving config norm_stats differs from the checkpoint")
    contract = manifest.get("contract")
    expected_contract = {
        "openpi.config_name": spec.openpi_config,
        "openpi.data.task": spec.task,
        "openpi.data.default_prompt": spec.default_prompt,
    }
    if not isinstance(contract, dict) or any(
        contract.get(key) != value for key, value in expected_contract.items()
    ):
        raise ValueError("JAX serving config semantics differ from the checkpoint")
    dataset = manifest.get("dataset")
    if (
        not isinstance(dataset, dict)
        or Path(str(dataset.get("root", ""))).resolve() != spec.dataset
    ):
        raise ValueError("JAX serving config dataset differs from the checkpoint")

    contract_file = checkpoint_path / "clothdojo_weights.json"
    contract = _read_json_object(contract_file, "OpenPI JAX checkpoint contract")
    if (
        contract.get("schema") != "clothdojo_openpi_jax_checkpoint"
        or contract.get("format_version") != 1
        or contract.get("ema")
        != {"path": "params", "purpose": "inference and JAX-to-PyTorch conversion"}
    ):
        raise ValueError(f"unsupported OpenPI JAX checkpoint contract: {contract_file}")
    step = contract.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"OpenPI JAX checkpoint step is invalid: {contract_file}")
    return {
        **identity,
        "checkpoint_contract_sha256": _file_sha256(contract_file),
        "checkpoint_step": step,
    }


def validate_cfg_checkpoint_binding(
    checkpoint: str | Path,
    norm_stats: str | Path,
    *,
    openpi_config: str,
    action_horizon: int,
    action_dim: int,
    positive_only_conditional: bool,
) -> dict[str, Any]:
    """Bind CFG serving semantics to the immutable training launch manifest."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    norm_path = Path(norm_stats).expanduser().resolve()
    norm_file = norm_path / "norm_stats.json" if norm_path.is_dir() else norm_path
    provenance_file = norm_file.parent / "clothdojo.json"
    for path, context in (
        (norm_file, "CFG norm_stats.json"),
        (provenance_file, "CFG norm provenance"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{context} does not exist: {path}")
    manifest_file = find_cfg_training_manifest(checkpoint_path)
    manifest = _read_json_object(manifest_file, "CFG launch manifest")
    if (
        manifest.get("schema") != "clothdojo_cfg_launch"
        or manifest.get("format_version") != 1
        or manifest.get("compatible") is not True
    ):
        raise ValueError(f"unsupported CFG launch manifest: {manifest_file}")
    expected_contract = {
        "openpi_config": openpi_config,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "adapt_to_pi": False,
        "extra_delta_transform": True,
        "train_expert_only": False,
        "positive_only_conditional": positive_only_conditional,
    }
    contract = manifest.get("inference_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"CFG launch manifest is missing inference_contract: {manifest_file}")
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"served CFG inference contract differs from training: {mismatches}")
    norm_identity = manifest.get("norm_stats")
    if not isinstance(norm_identity, dict):
        raise ValueError(f"CFG launch manifest is missing norm_stats: {manifest_file}")
    actual_stats = _file_sha256(norm_file)
    actual_provenance = _file_sha256(provenance_file)
    expected_norm = {
        "stats_sha256": actual_stats,
        "provenance_sha256": actual_provenance,
    }
    if any(norm_identity.get(key) != value for key, value in expected_norm.items()):
        raise ValueError("served CFG norm artifacts differ from the training launch manifest")
    return {
        "norm_stats_sha256": actual_stats,
        "norm_provenance_sha256": actual_provenance,
        "training_manifest_sha256": _file_sha256(manifest_file),
    }


def find_training_manifest(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    for root in (checkpoint, *checkpoint.parents):
        for relative in (
            "pi05_launch_manifest.json",
            "clothdojo_launch.json",
            "metadata/pi05_launch_manifest.json",
            "metadata/clothdojo_launch.json",
        ):
            candidate = root / relative
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "checkpoint is not below a Pi0.5 run containing "
        f"pi05_launch_manifest.json or clothdojo_launch.json: {checkpoint}"
    )


def find_cfg_training_manifest(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    for root in (checkpoint, *checkpoint.parents):
        candidate = root / "cfg_launch_manifest.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "CFG checkpoint is not below a run containing cfg_launch_manifest.json: "
        f"{checkpoint}"
    )


def _validate_jax_conversion_parity(
    checkpoint: Path,
    *,
    weights_sha256: str,
    norm_stats_sha256: str,
) -> dict[str, str]:
    from clothdojo.rollout.parity import (
        ConversionParityLimits,
        _parity_violations,
    )

    _require_sha256(weights_sha256, "served checkpoint")
    for root in (checkpoint, *checkpoint.parents):
        candidate = root / "clothdojo_parity.json"
        if candidate.is_file():
            parity_file = candidate
            break
    else:
        raise FileNotFoundError(
            f"converted JAX checkpoint has no clothdojo_parity.json: {checkpoint}"
        )
    parity = _read_json_object(parity_file, "JAX/PyTorch parity report")
    expected_limits = asdict(ConversionParityLimits())
    if (
        parity.get("schema") != "clothdojo_jax_openpi_rlinf_parity"
        or parity.get("format_version") != 1
        or parity.get("compatible") is not True
        or parity.get("weights_sha256") != weights_sha256
        or parity.get("norm_stats_sha256") != norm_stats_sha256
        or parity.get("precision")
        != {
            "jax_checkpoint": "float32",
            "jax_compute": "bfloat16",
            "pytorch_checkpoint": "float32",
            "pytorch_compute": "native_bfloat16",
        }
        or parity.get("limits") != expected_limits
        or parity.get("violations") != []
    ):
        raise ValueError(f"JAX/PyTorch parity report is not bound or accepted: {parity_file}")
    metrics = parity.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"JAX/PyTorch parity report has no metrics: {parity_file}")
    try:
        violations = _parity_violations(metrics, ConversionParityLimits())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"JAX/PyTorch parity metrics are malformed: {parity_file}"
        ) from exc
    if violations:
        raise ValueError(
            f"JAX/PyTorch parity metrics exceed the serving limits: {violations}"
        )
    return {"parity_report_sha256": _file_sha256(parity_file)}


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {context} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} SHA-256 must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} SHA-256 must contain 64 hexadecimal characters") from exc
