from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clothdojo.training.jax_sft import (
    _inspect_orbax_params,
    _inspect_pytorch_base,
    compose_jax_sft_spec,
    gate_manifest_path,
    inspect_jax_sft,
    write_checkpoint_contract,
)


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "jax.yaml"
    path.write_text(
        f"""
schema: clothdojo_pi05_jax_sft
format_version: 1
experiment:
  profile: openpi_reference
openpi:
  config_name: pi05_aloha_robotwin
  data:
    adapt_to_pi: false
    extra_delta_transform: true
    default_prompt: Flatten the shirt.
    video_backend: pyav
paths:
  dataset: {tmp_path}/dataset
  norm_stats: {tmp_path}/norm
  base_params: {tmp_path}/base/params
  pytorch_base: {tmp_path}/torch-base
  openpi_data_home: {tmp_path}/openpi-cache
  checkpoint_base_dir: {tmp_path}/checkpoints
  gate_dir: {tmp_path}/gates
training:
  seed: 42
  batch_size: 64
  num_workers: 16
  fsdp_devices: 8
  num_train_steps: 40000
  log_interval: 100
  save_interval: 10000
  keep_period: 10000
optimizer:
  warmup_steps: 1000
  peak_lr: 2.5e-5
  decay_steps: 30000
  decay_lr: 2.5e-6
  adam_b1: 0.9
  adam_b2: 0.95
  adam_eps: 1.0e-8
  weight_decay: 1.0e-10
  clip_gradient_norm: 1.0
  ema_decay: 0.99
gates:
  overfit_steps: 200
  pilot_steps: 1000
  overfit_max_loss_ratio: 0.8
logging:
  project_name: clothdojo
  run_name: reference
  wandb_group: comparison
  wandb_enabled: true
""",
        encoding="utf-8",
    )
    return path


def test_compose_jax_sft_spec_keeps_reference_contract(tmp_path: Path) -> None:
    spec = compose_jax_sft_spec(_config(tmp_path), ["training.num_workers=8"])

    assert spec.batch_size == 64
    assert spec.tuning_profile == "openpi_reference"
    assert spec.num_workers == 8
    assert spec.base_params == tmp_path / "base/params"
    assert spec.checkpoint_dir == tmp_path / "checkpoints/pi05_aloha_robotwin/reference"
    assert gate_manifest_path(spec, "pilot") == tmp_path / "gates/reference/pilot.json"


def test_compose_jax_sft_spec_rejects_baseline_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"training\.batch_size"):
        compose_jax_sft_spec(_config(tmp_path), ["training.batch_size=32"])


def test_compose_jax_sft_spec_accepts_reviewed_batch_profile(tmp_path: Path) -> None:
    spec = compose_jax_sft_spec(
        _config(tmp_path),
        [
            "experiment.profile=batch_128_sample_equivalent",
            "training.batch_size=128",
            "training.num_train_steps=20000",
            "training.log_interval=50",
            "training.save_interval=5000",
            "optimizer.warmup_steps=500",
            "optimizer.decay_steps=20000",
        ],
    )

    assert spec.tuning_profile == "batch_128_sample_equivalent"
    assert spec.batch_size * spec.num_train_steps == 2_560_000
    assert spec.batch_size * spec.log_interval == 6_400


def test_checkpoint_contract_distinguishes_ema_and_online(tmp_path: Path) -> None:
    step = tmp_path / "10000"
    (step / "params").mkdir(parents=True)
    (step / "train_state").mkdir()

    path = write_checkpoint_contract(step, step=10_000)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ema"]["path"] == "params"
    assert payload["online"]["path"] == "train_state"
    assert payload["online"]["field"] == "params"


def test_checkpoint_contract_rejects_partial_save(tmp_path: Path) -> None:
    step = tmp_path / "10000"
    (step / "params").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="train_state"):
        write_checkpoint_contract(step, step=10_000)


def test_orbax_identity_covers_payload_content(tmp_path: Path) -> None:
    root = tmp_path / "params"
    root.mkdir()
    (root / "_METADATA").write_bytes(b"metadata")
    (root / "_CHECKPOINT_METADATA").write_bytes(b"checkpoint metadata")
    payload = root / "array_metadatas" / "process_0"
    payload.parent.mkdir()
    payload.write_bytes(b"first")

    first = _inspect_orbax_params(root)
    payload.write_bytes(b"other")
    second = _inspect_orbax_params(root)

    assert first["bytes"] == second["bytes"]
    assert first["payload_sha256"] != second["payload_sha256"]


def test_orbax_identity_rejects_symlinked_payload(tmp_path: Path) -> None:
    root = tmp_path / "params"
    root.mkdir()
    (root / "_METADATA").write_bytes(b"metadata")
    (root / "_CHECKPOINT_METADATA").write_bytes(b"checkpoint metadata")
    external = tmp_path / "external"
    external.write_bytes(b"payload")
    (root / "payload").symlink_to(external)

    with pytest.raises(ValueError, match="may not contain symlinks"):
        _inspect_orbax_params(root)


def test_pytorch_parity_base_requires_released_layout(tmp_path: Path) -> None:
    root = tmp_path / ("a" * 40)
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "dtype": "float32",
                "chunk_size": 50,
                "max_action_dim": 32,
                "max_state_dim": 32,
            }
        ),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"weights")

    report = _inspect_pytorch_base(root)

    assert report["source_revision"] == "a" * 40
    assert report["weights_bytes"] == 7
    assert report["weights_sha256"] == (
        "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
    )


def test_preflight_identity_covers_gate_settings_and_pytorch_weights(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = _config(tmp_path)
    tokenizer = tmp_path / "openpi-cache/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer")
    pytorch_base = tmp_path / "torch-base"
    pytorch_base.mkdir()
    (pytorch_base / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "dtype": "float32",
                "chunk_size": 50,
                "max_action_dim": 32,
                "max_state_dim": 32,
            }
        ),
        encoding="utf-8",
    )
    weights = pytorch_base / "model.safetensors"
    weights.write_bytes(b"first weights")

    monkeypatch.setattr(
        "clothdojo.training.openpi_artifacts.inspect_openpi_dataset",
        lambda _path, **_kwargs: {"dataset": "checked"},
    )
    monkeypatch.setattr(
        "clothdojo.training.openpi_artifacts.inspect_openpi_norm_stats",
        lambda *_args, **_kwargs: {"norm": "checked"},
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft._inspect_orbax_params",
        lambda _path: {"base": "checked"},
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft.build_openpi_train_config",
        lambda _spec: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft._validate_built_config",
        lambda _config, _spec: None,
    )

    original = inspect_jax_sft(compose_jax_sft_spec(config_path))
    changed_gates = [
        inspect_jax_sft(compose_jax_sft_spec(config_path, [override]))
        for override in (
            "gates.overfit_steps=201",
            "gates.pilot_steps=1001",
            "gates.overfit_max_loss_ratio=0.7",
        )
    ]
    pytorch_config = pytorch_base / "config.json"
    original_config = pytorch_config.read_bytes()
    pytorch_config.write_bytes(original_config + b"\n")
    changed_config = inspect_jax_sft(compose_jax_sft_spec(config_path))
    pytorch_config.write_bytes(original_config)
    weights.write_bytes(b"second weights")
    changed_weights = inspect_jax_sft(compose_jax_sft_spec(config_path))

    assert original["gates"] == {
        "overfit_steps": 200,
        "pilot_steps": 1000,
        "overfit_max_loss_ratio": 0.8,
    }
    assert all(
        original["artifact_sha256"] != changed["artifact_sha256"]
        for changed in changed_gates
    )
    assert original["artifact_sha256"] != changed_config["artifact_sha256"]
    assert original["artifact_sha256"] != changed_weights["artifact_sha256"]
