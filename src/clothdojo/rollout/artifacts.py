"""Artifact identity checks shared by frozen policy serving boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def validate_sft_norm_binding(
    checkpoint: str | Path,
    norm_stats: str | Path,
    *,
    openpi_config: str,
    action_horizon: int,
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

    manifest_file = _find_training_manifest(checkpoint_path)
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
    required_data_kwargs = {
        "adapt_to_pi": False,
        "extra_delta_transform": True,
        "default_prompt": "Flatten the shirt.",
    }
    if provenance.get("data_kwargs") != required_data_kwargs:
        raise ValueError(
            "norm-stat transform contract differs from ClothDojo Pi0.5 ALOHA: "
            f"{provenance.get('data_kwargs')!r} != {required_data_kwargs!r}"
        )

    return {
        "norm_stats_sha256": stats_sha256,
        "norm_provenance_sha256": provenance_sha256,
        "training_manifest_sha256": _file_sha256(manifest_file),
        "dataset_info_sha256": provenance["dataset_info_sha256"],
        "dataset_sidecar_sha256": provenance["dataset_sidecar_sha256"],
    }


def _find_training_manifest(checkpoint: Path) -> Path:
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
