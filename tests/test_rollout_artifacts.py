from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from clothdojo.rollout.artifacts import (
    validate_cfg_checkpoint_binding,
    validate_jax_reference_binding,
    validate_sft_norm_binding,
)
from clothdojo.rollout.parity import ConversionParityLimits


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


def _bind_passing_jax_parity(
    checkpoint: Path,
    norm: Path,
    manifest: Path,
    *,
    weights_sha256: str,
) -> Path:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = "clothdojo_pi05_jax_preflight"
    jax_manifest = manifest.with_name("clothdojo_launch.json")
    jax_manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.unlink()
    metric = {
        "shape": [2, 10, 14],
        "mean_abs": 0.001,
        "p99_abs": 0.005,
        "max_abs": 0.01,
        "rmse": 0.002,
        "cosine": 0.99999,
    }
    report = {
        "schema": "clothdojo_jax_openpi_rlinf_parity",
        "format_version": 1,
        "compatible": True,
        "weights_sha256": weights_sha256,
        "norm_stats_sha256": hashlib.sha256(
            (norm / "norm_stats.json").read_bytes()
        ).hexdigest(),
        "precision": {
            "jax_checkpoint": "float32",
            "jax_compute": "bfloat16",
            "pytorch_checkpoint": "float32",
            "pytorch_compute": "native_bfloat16",
        },
        "limits": asdict(ConversionParityLimits()),
        "metrics": {
            "model_1_step_full": metric,
            "model_1_step_executed": metric,
            "model_10_step_full": metric,
            "model_10_step_executed": metric,
            "model_10_step_rtc_guidance": metric,
            "final_action_executed": metric,
        },
        "violations": [],
    }
    parity = checkpoint / "clothdojo_parity.json"
    parity.write_text(json.dumps(report), encoding="utf-8")
    return parity


def test_accepts_exact_launch_artifacts(tmp_path: Path) -> None:
    checkpoint, norm, manifest = _fixture(tmp_path)

    identity = validate_sft_norm_binding(
        checkpoint,
        norm,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
    )

    assert identity == {
        "source_framework": "pytorch",
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


def test_raw_jax_reference_binds_config_checkpoint_and_norm_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, norm, manifest = _fixture(tmp_path)
    config = tmp_path / "jax.yaml"
    config.write_text("schema: clothdojo_pi05_jax_sft\n", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update(
        schema="clothdojo_pi05_jax_preflight",
        config_path=str(config.resolve()),
        contract={
            "openpi.config_name": "pi05_aloha_robotwin",
            "openpi.data.task": "flatten the shirt",
            "openpi.data.default_prompt": "Flatten the shirt.",
        },
        dataset={"root": str((tmp_path / "dataset").resolve())},
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    contract = checkpoint / "clothdojo_weights.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "clothdojo_openpi_jax_checkpoint",
                "format_version": 1,
                "step": 10_000,
                "ema": {
                    "path": "params",
                    "purpose": "inference and JAX-to-PyTorch conversion",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft.compose_jax_sft_spec",
        lambda path: SimpleNamespace(
            config_path=Path(path).resolve(),
            norm_stats=norm.resolve(),
            openpi_config="pi05_aloha_robotwin",
            task="flatten the shirt",
            default_prompt="Flatten the shirt.",
            dataset=(tmp_path / "dataset").resolve(),
        ),
    )

    identity = validate_jax_reference_binding(
        checkpoint,
        norm,
        config,
        action_horizon=50,
    )

    assert identity["checkpoint_step"] == 10_000
    assert identity["checkpoint_contract_sha256"] == hashlib.sha256(
        contract.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="serving config differs"):
        validate_jax_reference_binding(
            checkpoint,
            norm,
            tmp_path / "other.yaml",
            action_horizon=50,
        )


def test_converted_jax_serving_requires_bound_passing_parity(tmp_path: Path) -> None:
    checkpoint, norm, manifest = _fixture(tmp_path)
    weights_sha256 = "aa" * 32
    parity = _bind_passing_jax_parity(
        checkpoint,
        norm,
        manifest,
        weights_sha256=weights_sha256,
    )

    identity = validate_sft_norm_binding(
        checkpoint,
        norm,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
        weights_sha256=weights_sha256,
    )

    assert identity["parity_report_sha256"] == hashlib.sha256(
        parity.read_bytes()
    ).hexdigest()


def test_converted_jax_serving_rejects_missing_parity(tmp_path: Path) -> None:
    checkpoint, norm, manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = "clothdojo_pi05_jax_preflight"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"no clothdojo_parity\.json"):
        validate_sft_norm_binding(
            checkpoint,
            norm,
            openpi_config="pi05_aloha_robotwin",
            action_horizon=50,
            weights_sha256="aa" * 32,
        )


def test_converted_jax_serving_rejects_tampered_parity_metrics(tmp_path: Path) -> None:
    checkpoint, norm, manifest = _fixture(tmp_path)
    weights_sha256 = "aa" * 32
    parity = _bind_passing_jax_parity(
        checkpoint,
        norm,
        manifest,
        weights_sha256=weights_sha256,
    )
    report = json.loads(parity.read_text(encoding="utf-8"))
    report["metrics"]["final_action_executed"]["max_abs"] = 1.0
    parity.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics exceed the serving limits"):
        validate_sft_norm_binding(
            checkpoint,
            norm,
            openpi_config="pi05_aloha_robotwin",
            action_horizon=50,
            weights_sha256=weights_sha256,
        )


def test_accepts_fold_prompt_bound_by_launch_artifacts(tmp_path: Path) -> None:
    checkpoint, norm, manifest_file = _fixture(tmp_path)
    provenance_file = norm / "clothdojo.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    provenance["data_kwargs"]["default_prompt"] = "Fold the shirt."
    provenance_file.write_text(json.dumps(provenance), encoding="utf-8")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["norm_stats"]["data_kwargs"] = provenance["data_kwargs"]
    manifest["norm_stats"]["provenance_sha256"] = hashlib.sha256(
        provenance_file.read_bytes()
    ).hexdigest()
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    identity = validate_sft_norm_binding(
        checkpoint,
        norm,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
    )

    assert identity["training_manifest_sha256"] == hashlib.sha256(
        manifest_file.read_bytes()
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


def test_cfg_binding_rejects_runtime_semantic_drift(tmp_path: Path) -> None:
    run = tmp_path / "cfg_run"
    checkpoint = run / "global_step_100"
    checkpoint.mkdir(parents=True)
    norm = tmp_path / "norm"
    norm.mkdir()
    stats = norm / "norm_stats.json"
    provenance = norm / "clothdojo.json"
    stats.write_text("{}\n", encoding="utf-8")
    provenance.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema": "clothdojo_cfg_launch",
        "format_version": 1,
        "compatible": True,
        "norm_stats": {
            "stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
            "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
        },
        "inference_contract": {
            "openpi_config": "pi05_aloha_robotwin",
            "action_dim": 14,
            "action_horizon": 50,
            "num_steps": 5,
            "adapt_to_pi": False,
            "extra_delta_transform": True,
            "train_expert_only": False,
            "positive_only_conditional": False,
        },
    }
    manifest_file = run / "cfg_launch_manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    identity = validate_cfg_checkpoint_binding(
        checkpoint,
        norm,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
        action_dim=14,
        positive_only_conditional=False,
    )

    assert identity["training_manifest_sha256"] == hashlib.sha256(
        manifest_file.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="inference contract differs"):
        validate_cfg_checkpoint_binding(
            checkpoint,
            norm,
            openpi_config="pi05_aloha_robotwin",
            action_horizon=40,
            action_dim=14,
            positive_only_conditional=False,
        )
    manifest["inference_contract"]["train_expert_only"] = True
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="train_expert_only"):
        validate_cfg_checkpoint_binding(
            checkpoint,
            norm,
            openpi_config="pi05_aloha_robotwin",
            action_horizon=50,
            action_dim=14,
            positive_only_conditional=False,
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
