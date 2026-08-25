from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clothdojo.rollout.artifacts import validate_sft_norm_binding


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "global_step_10000"
    checkpoint.mkdir(parents=True)
    norm = tmp_path / "norm"
    norm.mkdir()
    stats = norm / "norm_stats.json"
    stats.write_text('{"norm_stats": {}}\n', encoding="utf-8")
    provenance = {
        "schema": "clothdojo_openpi_norm_stats",
        "format_version": 1,
        "dataset_root": "/external/good-v2",
        "dataset_info_sha256": "11" * 32,
        "dataset_sidecar_sha256": "22" * 32,
        "config_name": "pi05_aloha_robotwin",
        "data_kwargs": {
            "adapt_to_pi": False,
            "extra_delta_transform": True,
            "default_prompt": "Flatten the shirt.",
        },
        "action_horizon": 50,
        "frames": 1_287_887,
        "episodes": 719,
    }
    provenance_file = norm / "clothdojo.json"
    provenance_file.write_text(json.dumps(provenance), encoding="utf-8")
    manifest = {
        "schema": "clothdojo_pi05_launch_preflight",
        "format_version": 1,
        "compatible": True,
        "norm_stats": {
            **{
                key: provenance[key]
                for key in (
                    "dataset_info_sha256",
                    "dataset_sidecar_sha256",
                    "config_name",
                    "data_kwargs",
                    "action_horizon",
                    "frames",
                    "episodes",
                )
            },
            "stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
            "provenance_sha256": hashlib.sha256(provenance_file.read_bytes()).hexdigest(),
        },
    }
    manifest_file = run / "pi05_launch_manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, norm, manifest_file


def test_accepts_exact_launch_artifacts(tmp_path: Path) -> None:
    checkpoint, norm, manifest = _fixture(tmp_path)

    identity = validate_sft_norm_binding(
        checkpoint,
        norm,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
    )

    assert identity == {
        "norm_stats_sha256": hashlib.sha256((norm / "norm_stats.json").read_bytes()).hexdigest(),
        "norm_provenance_sha256": hashlib.sha256(
            (norm / "clothdojo.json").read_bytes()
        ).hexdigest(),
        "training_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "dataset_info_sha256": "11" * 32,
        "dataset_sidecar_sha256": "22" * 32,
    }


def test_accepts_converted_jax_launch_artifacts(tmp_path: Path) -> None:
    checkpoint, norm, manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = "clothdojo_pi05_jax_preflight"
    jax_manifest = manifest.with_name("clothdojo_launch.json")
    jax_manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.unlink()

    identity = validate_sft_norm_binding(
        checkpoint,
        norm,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
    )

    assert identity["training_manifest_sha256"] == hashlib.sha256(
        jax_manifest.read_bytes()
    ).hexdigest()


def test_rejects_stats_from_another_dataset(tmp_path: Path) -> None:
    checkpoint, norm, _ = _fixture(tmp_path)
    (norm / "norm_stats.json").write_text('{"norm_stats": {"v1": true}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"norm_stats\.json SHA-256 differs"):
        validate_sft_norm_binding(
            checkpoint,
            norm,
            openpi_config="pi05_aloha_robotwin",
            action_horizon=50,
        )


def test_rejects_different_provenance_file(tmp_path: Path) -> None:
    checkpoint, norm, _ = _fixture(tmp_path)
    provenance = json.loads((norm / "clothdojo.json").read_text(encoding="utf-8"))
    provenance["dataset_root"] = "/external/good-v1"
    (norm / "clothdojo.json").write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="norm provenance SHA-256 differs"):
        validate_sft_norm_binding(
            checkpoint,
            norm,
            openpi_config="pi05_aloha_robotwin",
            action_horizon=50,
        )


def test_rejects_semantic_drift_even_with_matching_file_hash(tmp_path: Path) -> None:
    checkpoint, norm, manifest_file = _fixture(tmp_path)
    provenance_file = norm / "clothdojo.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    provenance["data_kwargs"]["extra_delta_transform"] = False
    provenance_file.write_text(json.dumps(provenance), encoding="utf-8")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["norm_stats"]["data_kwargs"] = provenance["data_kwargs"]
    manifest["norm_stats"]["provenance_sha256"] = hashlib.sha256(
        provenance_file.read_bytes()
    ).hexdigest()
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="transform contract differs"):
        validate_sft_norm_binding(
            checkpoint,
            norm,
            openpi_config="pi05_aloha_robotwin",
            action_horizon=50,
        )


def test_requires_checkpoint_launch_manifest(tmp_path: Path) -> None:
    checkpoint, norm, manifest = _fixture(tmp_path)
    manifest.unlink()

    with pytest.raises(
        FileNotFoundError,
        match=r"pi05_launch_manifest\.json or clothdojo_launch\.json",
    ):
        validate_sft_norm_binding(
            checkpoint,
            norm,
            openpi_config="pi05_aloha_robotwin",
            action_horizon=50,
        )


def test_accepts_checkpoint_bundle_with_metadata_sibling(tmp_path: Path) -> None:
    _checkpoint, norm, manifest = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    bundled_checkpoint = bundle / "global_step_40000"
    bundled_checkpoint.mkdir(parents=True)
    metadata = bundle / "metadata"
    metadata.mkdir()
    manifest.replace(metadata / "pi05_launch_manifest.json")

    identity = validate_sft_norm_binding(
        bundled_checkpoint,
        norm,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
    )

    assert identity["training_manifest_sha256"]
