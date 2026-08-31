from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.training.jax_sft import (
    ActionOffsetLossSpec,
    DataSamplingSpec,
    _action_offset_weights,
    _create_openpi_data_loader,
    _data_sampling_manifest,
    _inspect_orbax_params,
    _inspect_pytorch_base,
    _make_data_sampler,
    _reduce_action_offset_loss,
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
    assert spec.action_offset_loss == ActionOffsetLossSpec(
        profile="uniform",
        focus=0.0,
        exec_horizon=50,
        delay_steps=0,
    )
    assert spec.data_sampling == DataSamplingSpec(mode="frame", alpha=None, index=None)
    assert spec.num_workers == 8
    assert spec.task == "flatten the shirt"
    assert spec.default_prompt == "Flatten the shirt."
    assert spec.base_params == tmp_path / "base/params"
    assert spec.checkpoint_dir == tmp_path / "checkpoints/pi05_aloha_robotwin/reference"
    assert gate_manifest_path(spec, "pilot") == tmp_path / "gates/reference/pilot.json"


def test_compose_jax_sft_spec_accepts_only_reviewed_asset_tempering(tmp_path: Path) -> None:
    index = tmp_path / "dataset/meta/asset_sampling_index.json"
    spec = compose_jax_sft_spec(
        _config(tmp_path),
        [
            "data.sampling.mode=asset_tempered",
            "data.sampling.alpha=0.5",
            f"data.sampling.index={index}",
        ],
    )

    assert spec.data_sampling == DataSamplingSpec(
        mode="asset_tempered",
        alpha=0.5,
        index=index,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["data.sampling.mode=unknown"], "mode"),
        (
            ["data.sampling.mode=frame", "data.sampling.alpha=0.5"],
            "exactly mode",
        ),
        (
            [
                "data.sampling.mode=asset_tempered",
                "data.sampling.alpha=0.4",
                "data.sampling.index=/tmp/index.json",
            ],
            "alpha",
        ),
        (
            ["data.sampling.mode=asset_tempered", "data.sampling.alpha=0.5"],
            "alpha and index",
        ),
    ],
)
def test_compose_jax_sft_spec_rejects_ambiguous_data_sampling(
    tmp_path: Path,
    overrides: list[str],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        compose_jax_sft_spec(_config(tmp_path), overrides)


def test_compose_jax_sft_spec_accepts_soft_rtc_action_offset_focus(
    tmp_path: Path,
) -> None:
    spec = compose_jax_sft_spec(
        _config(tmp_path),
        [
            "loss.action_offset.profile=fixed_delay_rtc",
            "loss.action_offset.focus=0.5",
            "loss.action_offset.exec_horizon=10",
            "loss.action_offset.delay_steps=0",
        ],
    )

    assert spec.action_offset_loss == ActionOffsetLossSpec(
        profile="fixed_delay_rtc",
        focus=0.5,
        exec_horizon=10,
        delay_steps=0,
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("loss.action_offset.profile=other", "profile"),
        ("loss.action_offset.focus=-0.1", "focus"),
        ("loss.action_offset.focus=1.1", "focus"),
        ("loss.action_offset.exec_horizon=0", "exec_horizon"),
        ("loss.action_offset.delay_steps=-1", "delay_steps"),
        ("loss.action_offset.delay_steps=10", "delay_steps"),
    ],
)
def test_compose_jax_sft_spec_rejects_invalid_action_offset_loss(
    tmp_path: Path,
    override: str,
    message: str,
) -> None:
    overrides = [
        "loss.action_offset.profile=fixed_delay_rtc",
        "loss.action_offset.focus=0.5",
        "loss.action_offset.exec_horizon=10",
        "loss.action_offset.delay_steps=0",
        override,
    ]

    with pytest.raises((TypeError, ValueError), match=message):
        compose_jax_sft_spec(_config(tmp_path), overrides)


def test_action_offset_weights_describe_rtc_execution_window() -> None:
    weights = _action_offset_weights(
        ActionOffsetLossSpec(
            profile="fixed_delay_rtc",
            focus=0.5,
            exec_horizon=4,
            delay_steps=2,
        ),
        action_horizon=10,
    )

    assert weights == (0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5)


@pytest.mark.parametrize(
    "spec",
    [
        ActionOffsetLossSpec(
            profile="uniform",
            focus=0.5,
            exec_horizon=10,
            delay_steps=0,
        ),
        ActionOffsetLossSpec(
            profile="fixed_delay_rtc",
            focus=float("nan"),
            exec_horizon=4,
            delay_steps=0,
        ),
        ActionOffsetLossSpec(
            profile="fixed_delay_rtc",
            focus=0.5,
            exec_horizon=8,
            delay_steps=3,
        ),
    ],
)
def test_action_offset_weights_reject_invalid_profiles(
    spec: ActionOffsetLossSpec,
) -> None:
    with pytest.raises(ValueError):
        _action_offset_weights(spec, action_horizon=10)


def test_uniform_action_offset_reduction_exactly_matches_reference_mean() -> None:
    per_offset = np.linspace(-0.5, 3.0, num=100, dtype=np.float32).reshape(2, 50)

    reduced = _reduce_action_offset_loss(
        per_offset,
        None,
        action_horizon=50,
    )

    assert reduced == per_offset.mean()


def test_weighted_action_offset_reduction_normalizes_each_batch_item() -> None:
    per_offset = np.arange(12, dtype=np.float32).reshape(2, 6)
    weights = np.asarray([0.5, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=np.float32)

    reduced = _reduce_action_offset_loss(
        per_offset,
        weights,
        action_horizon=6,
    )

    expected = np.mean(np.sum(per_offset * weights, axis=-1) / np.sum(weights))
    assert reduced == expected


@pytest.mark.parametrize(
    ("loss_shape", "weight_shape", "message"),
    [
        ((2,), None, "compute_loss"),
        ((2, 49), None, "compute_loss"),
        ((2, 50), (49,), "weights"),
        ((2, 50), (1, 50), "weights"),
    ],
)
def test_action_offset_reduction_rejects_shape_drift(
    loss_shape: tuple[int, ...],
    weight_shape: tuple[int, ...] | None,
    message: str,
) -> None:
    per_offset = np.zeros(loss_shape, dtype=np.float32)
    weights = None if weight_shape is None else np.ones(weight_shape, dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        _reduce_action_offset_loss(per_offset, weights, action_horizon=50)


def test_action_offset_reduction_is_jittable_and_has_weighted_gradient() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    per_offset = jnp.arange(12, dtype=jnp.float32).reshape(2, 6)
    weights = jnp.asarray([0.5, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=jnp.float32)

    reduce = jax.jit(
        lambda value: _reduce_action_offset_loss(
            value,
            weights,
            action_horizon=6,
        )
    )
    gradient = jax.grad(reduce)(per_offset)

    expected = np.broadcast_to(
        np.asarray(weights) / (2 * float(weights.sum())),
        per_offset.shape,
    )
    np.testing.assert_allclose(np.asarray(gradient), expected, rtol=0, atol=0)


def test_asset_sampling_manifest_and_sampler_use_training_seed(
    tmp_path: Path, monkeypatch
) -> None:
    spec = dataclasses.replace(
        compose_jax_sft_spec(_config(tmp_path)),
        data_sampling=DataSamplingSpec(
            mode="asset_tempered",
            alpha=0.5,
            index=tmp_path / "index.json",
        ),
    )
    index = SimpleNamespace(
        episodes=(
            SimpleNamespace(asset_id="asset-a"),
            SimpleNamespace(asset_id="asset-a"),
            SimpleNamespace(asset_id="asset-b"),
        ),
        asset_frame_counts={"asset-a": 80, "asset-b": 20},
        total_frames=100,
        sha256="logical-index-sha",
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft._load_asset_sampling_index",
        lambda _spec: (index, "file-sha"),
    )
    calls = []
    sampler = object()

    def make_sampler(received, *, alpha, seed):
        calls.append((received, alpha, seed))
        return sampler

    monkeypatch.setattr(
        "clothdojo.data.asset_sampling.make_asset_tempered_sampler",
        make_sampler,
    )

    manifest = _data_sampling_manifest(spec)
    actual = _make_data_sampler(spec, {"training": {"data_sampling": manifest}})

    assert manifest == {
        "mode": "asset_tempered",
        "alpha": 0.5,
        "seed": 42,
        "replacement": True,
        "num_samples": 100,
        "index": {
            "path": str(tmp_path / "index.json"),
            "logical_sha256": "logical-index-sha",
            "file_sha256": "file-sha",
        },
        "asset_count": 2,
        "asset_episode_counts": {"asset-a": 2, "asset-b": 1},
        "asset_frame_counts": {"asset-a": 80, "asset-b": 20},
    }
    assert actual is sampler
    assert calls == [(index, 0.5, 42)]


def test_asset_sampler_rejects_index_drift_after_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    spec = dataclasses.replace(
        compose_jax_sft_spec(_config(tmp_path)),
        data_sampling=DataSamplingSpec(
            mode="asset_tempered",
            alpha=0.5,
            index=tmp_path / "index.json",
        ),
    )
    index = SimpleNamespace(
        episodes=(SimpleNamespace(asset_id="asset-a"),),
        asset_frame_counts={"asset-a": 10},
        total_frames=10,
        sha256="logical-index-sha",
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft._load_asset_sampling_index",
        lambda _spec: (index, "new-file-sha"),
    )
    expected = {
        "training": {
            "data_sampling": {
                **_data_sampling_manifest(
                    dataclasses.replace(spec, data_sampling=DataSamplingSpec("frame", None, None))
                ),
                "mode": "asset_tempered",
            }
        }
    }

    with pytest.raises(RuntimeError, match="changed after preflight"):
        _make_data_sampler(spec, expected)


def test_openpi_loader_preserves_frame_shuffle_and_only_passes_explicit_sampler() -> None:
    calls = []

    class Dataset:
        def __len__(self) -> int:
            return 5

    class TorchLoader:
        def __init__(self, dataset, local_batch_size, **kwargs) -> None:
            calls.append((dataset, local_batch_size, kwargs))

    lerobot = SimpleNamespace(
        LeRobotDatasetMetadata=lambda _repo: SimpleNamespace(fps=50, tasks=[]),
        LeRobotDataset=lambda *_args, **_kwargs: Dataset(),
    )
    data_config = SimpleNamespace(
        repo_id="dataset",
        action_sequence_keys=("actions",),
        prompt_from_task=False,
    )
    config = SimpleNamespace(
        assets_dirs=(),
        model=SimpleNamespace(action_horizon=50),
        data=SimpleNamespace(create=lambda *_args: data_config),
        batch_size=2,
        num_workers=0,
        seed=42,
    )
    module = SimpleNamespace(
        _import_lerobot_dataset=lambda: lerobot,
        transform_dataset=lambda dataset, _config: dataset,
        TorchDataLoader=TorchLoader,
        DataLoaderImpl=lambda _config, loader: loader,
    )

    _create_openpi_data_loader(
        config,
        sharding="sharding",
        shuffle=True,
        video_backend="pyav",
        data_loader_module=module,
    )
    sampler = SimpleNamespace(weights=[1.0] * 5)
    _create_openpi_data_loader(
        config,
        sharding="sharding",
        shuffle=False,
        sampler=sampler,
        video_backend="pyav",
        data_loader_module=module,
    )

    assert calls[0][2]["shuffle"] is True
    assert "sampler" not in calls[0][2]
    assert calls[1][2]["shuffle"] is False
    assert calls[1][2]["sampler"] is sampler


def test_compose_jax_sft_spec_accepts_explicit_fold_task(tmp_path: Path) -> None:
    spec = compose_jax_sft_spec(
        _config(tmp_path),
        [
            "openpi.data.task=fold the shirt",
            "openpi.data.default_prompt=Fold the shirt.",
        ],
    )

    assert spec.task == "fold the shirt"
    assert spec.default_prompt == "Fold the shirt."


def test_compose_jax_sft_spec_rejects_task_prompt_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must express"):
        compose_jax_sft_spec(
            _config(tmp_path),
            [
                "openpi.data.task=fold the shirt",
                "openpi.data.default_prompt=Flatten the shirt.",
            ],
        )


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


@pytest.mark.parametrize(
    ("profile", "seed", "peak_lr", "decay_lr", "steps", "save_interval"),
    [
        ("decay_50k_lr_2x_seed42", 42, 5.0e-5, 5.0e-6, 50_000, 5_000),
        ("decay_60k_lr_2x_seed42", 42, 5.0e-5, 5.0e-6, 60_000, 10_000),
        ("decay_60k_lr_2x_seed43", 43, 5.0e-5, 5.0e-6, 60_000, 10_000),
        ("decay_60k_lr_3x", 42, 7.5e-5, 7.5e-6, 60_000, 10_000),
        ("decay_60k_lr_3x_seed43", 43, 7.5e-5, 7.5e-6, 60_000, 10_000),
    ],
)
def test_compose_jax_sft_spec_accepts_reviewed_decay_profiles(
    tmp_path: Path,
    profile: str,
    seed: int,
    peak_lr: float,
    decay_lr: float,
    steps: int,
    save_interval: int,
) -> None:
    spec = compose_jax_sft_spec(
        _config(tmp_path),
        [
            f"experiment.profile={profile}",
            f"training.seed={seed}",
            f"training.num_train_steps={steps}",
            f"training.save_interval={save_interval}",
            f"optimizer.peak_lr={peak_lr}",
            f"optimizer.decay_steps={steps}",
            f"optimizer.decay_lr={decay_lr}",
        ],
    )

    assert spec.tuning_profile == profile
    assert spec.seed == seed
    assert spec.num_train_steps == steps
    assert spec.save_interval == save_interval
    assert spec.peak_lr == peak_lr
    assert spec.decay_steps == steps
    assert spec.decay_lr == decay_lr


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
        lambda _spec: SimpleNamespace(model=SimpleNamespace(action_horizon=50)),
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
    changed_action_offset = inspect_jax_sft(
        compose_jax_sft_spec(
            config_path,
            [
                "loss.action_offset.profile=fixed_delay_rtc",
                "loss.action_offset.focus=0.5",
                "loss.action_offset.exec_horizon=10",
                "loss.action_offset.delay_steps=0",
            ],
        )
    )
    sampling_index = SimpleNamespace(
        episodes=(SimpleNamespace(asset_id="asset-a"),),
        asset_frame_counts={"asset-a": 100},
        total_frames=100,
        sha256="logical-index-sha",
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft._load_asset_sampling_index",
        lambda _spec: (sampling_index, "index-file-sha"),
    )
    changed_sampling = inspect_jax_sft(
        compose_jax_sft_spec(
            config_path,
            [
                "data.sampling.mode=asset_tempered",
                "data.sampling.alpha=0.5",
                f"data.sampling.index={tmp_path}/index.json",
            ],
        )
    )
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
    assert original["training"]["action_offset_loss"] == {
        "profile": "uniform",
        "focus": 0.0,
        "exec_horizon": 50,
        "delay_steps": 0,
        "weights": [1.0] * 50,
        "reduction": "batch_mean_of_weight_normalized_action_offset_mean",
    }
    assert original["training"]["data_sampling"] == {
        "mode": "frame",
        "alpha": None,
        "seed": 42,
        "replacement": False,
        "num_samples": None,
        "index": None,
        "asset_count": None,
        "asset_episode_counts": None,
        "asset_frame_counts": None,
    }
    assert changed_sampling["training"]["data_sampling"]["index"] == {
        "path": str(tmp_path / "index.json"),
        "logical_sha256": "logical-index-sha",
        "file_sha256": "index-file-sha",
    }
    assert all(
        original["artifact_sha256"] != changed["artifact_sha256"]
        for changed in changed_gates
    )
    assert original["artifact_sha256"] != changed_config["artifact_sha256"]
    assert original["artifact_sha256"] != changed_weights["artifact_sha256"]
    assert original["artifact_sha256"] != changed_action_offset["artifact_sha256"]
    assert original["artifact_sha256"] != changed_sampling["artifact_sha256"]
