from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clothdojo.training import cfg_pipeline, rlinf_cfg
from clothdojo.training.cfg import inspect_cfg_training, write_cfg_launch_manifest
from clothdojo.training.cfg_model import compute_cfg_routing_masks
from clothdojo.training.rlinf import compose_config
from clothdojo.training.rlinf_cfg import (
    MODEL_TYPE,
    _load_aligned_advantages,
    _openpi_config_kwargs,
)

ADVANTAGE_TAG = "test_step30000"


def _dataset(root: Path, labels: list[bool]) -> None:
    (root / "meta").mkdir(parents=True)
    names = [f"joint_{index}" for index in range(14)]
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": names},
        "action": {"dtype": "float32", "shape": [14], "names": names},
    }
    for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": [8, 8, 3],
            "names": ["height", "width", "channels"],
        }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "total_episodes": 2,
        "total_frames": 4,
        "fps": 50,
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 0, "length": 2}),
                json.dumps({"episode_index": 1, "length": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Flatten the shirt."}) + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "clothdojo.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_lerobot_export",
                "format_version": 1,
                "export_status": "complete",
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 0, 1, 1], type=pa.int64()),
                "frame_index": pa.array([0, 1, 0, 1], type=pa.int64()),
                "advantage": pa.array(labels, type=pa.bool_()),
            }
        ),
        root / "meta" / f"advantages_{ADVANTAGE_TAG}.parquet",
    )
    info_digest = hashlib.sha256((root / "meta" / "info.json").read_bytes()).hexdigest()
    positive = sum(labels)
    (root / "meta" / f"advantages_{ADVANTAGE_TAG}.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_steam_advantages",
                "format_version": 1,
                "role": "sft" if all(labels) else "rollout",
                "prompt": "Flatten the shirt.",
                "state_key": "observation.state",
                "camera_keys": [
                    "observation.images.cam_high",
                    "observation.images.cam_low",
                ],
                "checkpoint_config_sha256": "a" * 64,
                "checkpoint_weights_sha256": "b" * 64,
                "dataset_info_sha256": info_digest,
                "episodes": [0, 1],
                "stride": 64,
                "threshold": 0.0,
                "positive_fraction_requested": 0.25,
                "rows": len(labels),
                "positive": positive,
                "negative": len(labels) - positive,
            }
        ),
        encoding="utf-8",
    )


def _configure(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    expert = tmp_path / "expert"
    rollout = tmp_path / "rollout"
    _dataset(expert, [True, True, True, True])
    _dataset(rollout, [True, False, False, False])
    checkpoint = tmp_path / "checkpoint" / "actor" / "model_state_dict"
    checkpoint.mkdir(parents=True)
    (checkpoint / "full_weights.pt").write_bytes(b"weights")
    norm = tmp_path / "norm"
    norm.mkdir()
    (norm / "norm_stats.json").write_text("{}", encoding="utf-8")
    info_digest = hashlib.sha256((expert / "meta" / "info.json").read_bytes()).hexdigest()
    (norm / "clothdojo.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_openpi_norm_stats",
                "config_name": "pi05_aloha_robotwin",
                "dataset_info_sha256": info_digest,
            }
        ),
        encoding="utf-8",
    )
    values = {
        "CLOTHDOJO_CFG_EXPERT_DATASET": expert,
        "CLOTHDOJO_CFG_ROLLOUT_DATASET": rollout,
        "CLOTHDOJO_CFG_INIT_CHECKPOINT": checkpoint.parents[1],
        "CLOTHDOJO_PI05_NORM_STATS": norm,
        "CLOTHDOJO_TRAIN_OUTPUT": tmp_path / "output",
        "CLOTHDOJO_STEAM_ADVANTAGE_TAG": ADVANTAGE_TAG,
    }
    selection = tmp_path / "rollout_episodes.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "clothdojo_episode_selection",
                "format_version": 1,
                "episodes": [0, 1],
            }
        ),
        encoding="utf-8",
    )
    values["CLOTHDOJO_CFG_ROLLOUT_EPISODES"] = selection
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))
    return expert, rollout


def test_cfg_preflight_validates_exact_mixture(monkeypatch, tmp_path: Path) -> None:
    expert, rollout = _configure(monkeypatch, tmp_path)
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"

    report = inspect_cfg_training(config)

    assert report["compatible"] is True
    assert report["model_type"] == MODEL_TYPE
    assert report["inference_contract"]["positive_only_conditional"] is False
    assert report["training_prompt"] == "flatten the shirt"
    assert report["image_augmentation"] is False
    assert report["sampling_probabilities"] == {"sft": 0.75, "rollout": 0.25}
    assert report["datasets"] == [
        {
            "role": "sft",
            "dataset_path": str(expert),
            "advantage_tag": ADVANTAGE_TAG,
            "prompt": "Flatten the shirt.",
            "weight": 0.75,
            "episodes": 2,
            "frames": 4,
            "fps": 50,
            "advantage_path": str(expert / f"meta/advantages_{ADVANTAGE_TAG}.parquet"),
            "advantage_manifest_path": str(expert / f"meta/advantages_{ADVANTAGE_TAG}.json"),
            "positive_frames": 4,
            "negative_frames": 0,
            "steam_checkpoint_config_sha256": "a" * 64,
            "steam_checkpoint_weights_sha256": "b" * 64,
            "steam_stride": 64,
        },
        {
            "role": "rollout",
            "dataset_path": str(rollout),
            "advantage_tag": ADVANTAGE_TAG,
            "prompt": "Flatten the shirt.",
            "weight": 0.25,
            "episodes": 2,
            "frames": 4,
            "fps": 50,
            "advantage_path": str(rollout / f"meta/advantages_{ADVANTAGE_TAG}.parquet"),
            "advantage_manifest_path": str(rollout / f"meta/advantages_{ADVANTAGE_TAG}.json"),
            "positive_frames": 1,
            "negative_frames": 3,
            "steam_checkpoint_config_sha256": "a" * 64,
            "steam_checkpoint_weights_sha256": "b" * 64,
            "steam_stride": 64,
        },
    ]


def test_cfg_preflight_aggregates_logical_rollout_shards(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    rollout_2 = tmp_path / "rollout_2"
    _dataset(rollout_2, [False, True, False, False])
    monkeypatch.setenv("CLOTHDOJO_CFG_ROLLOUT_DATASET_2", str(rollout_2))
    source = (
        Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"
    ).read_text(encoding="utf-8")
    original = """    - dataset_path: ${oc.env:CLOTHDOJO_CFG_ROLLOUT_DATASET}
      type: rollout
      advantage_tag: ${oc.env:CLOTHDOJO_STEAM_ADVANTAGE_TAG}
      weight: 0.25
"""
    replacement = """    - dataset_path: ${oc.env:CLOTHDOJO_CFG_ROLLOUT_DATASET}
      type: rollout
      advantage_tag: ${oc.env:CLOTHDOJO_STEAM_ADVANTAGE_TAG}
      weight: 0.125
    - dataset_path: ${oc.env:CLOTHDOJO_CFG_ROLLOUT_DATASET_2}
      type: rollout
      advantage_tag: ${oc.env:CLOTHDOJO_STEAM_ADVANTAGE_TAG}
      weight: 0.125
"""
    assert source.count(original) == 1
    config = tmp_path / "cfg_shards.yaml"
    config.write_text(source.replace(original, replacement), encoding="utf-8")

    report = inspect_cfg_training(config)

    assert report["sampling_probabilities"] == {"sft": 0.75, "rollout": 0.25}
    assert [entry["probability"] for entry in report["dataset_sampling_probabilities"]] == [
        0.75,
        0.125,
        0.125,
    ]
    assert [dataset["role"] for dataset in report["datasets"]] == [
        "sft",
        "rollout",
        "rollout",
    ]
    manifest_path = rollout_2 / f"meta/advantages_{ADVANTAGE_TAG}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["threshold"] = 0.5
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different STEAM label thresholds"):
        inspect_cfg_training(config)


def test_cfg_preflight_matches_rlinf_size_balanced_sampling(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    source = (
        Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"
    ).read_text(encoding="utf-8")
    source = source.replace("balance_dataset_weights: false", "balance_dataset_weights: true")
    source = source.replace("weight: 0.75", "weight: 1.0")
    source = source.replace("weight: 0.25", "weight: 1.0")
    config = tmp_path / "cfg_size_balanced.yaml"
    config.write_text(source, encoding="utf-8")

    report = inspect_cfg_training(config)

    assert report["balance_dataset_weights"] is True
    assert report["sampling_probabilities"] == {"sft": 0.5, "rollout": 0.5}
    assert [entry["probability"] for entry in report["dataset_sampling_probabilities"]] == [
        0.5,
        0.5,
    ]


@pytest.mark.parametrize("expert_only", [False, True])
def test_cfg_pipeline_matches_entries_by_role_and_path(
    monkeypatch, tmp_path: Path, expert_only: bool
) -> None:
    from omegaconf import OmegaConf

    project_data = tmp_path / "data/datasets"
    checkpoint = tmp_path / "outputs/training/steam"
    model = tmp_path / "outputs/training/policy"
    work_dir = tmp_path / "outputs/labels"
    for path in (checkpoint, model, work_dir):
        path.mkdir(parents=True)
    project_data.mkdir(parents=True)
    datasets = []
    cfg_entries = []
    for role, name, tag in (
        ("sft", "expert", "expert_tag"),
        ("rollout", "rollout_0", "rollout_tag"),
        ("rollout", "rollout_1", "rollout_tag"),
    ):
        if expert_only and role == "rollout":
            continue
        path = project_data / name
        path.mkdir()
        datasets.append(
            SimpleNamespace(
                role=role,
                dataset_path=path,
                advantage_tag=tag,
                episodes=(0, 1),
                total_episodes=2,
                output_path=path / "meta" / f"advantages_{tag}.parquet",
            )
        )
        cfg_entries.append(
            {
                "type": role,
                "dataset_path": str(path),
                "advantage_tag": tag,
            }
        )
    spec = SimpleNamespace(
        checkpoint=checkpoint,
        work_dir=work_dir,
        datasets=tuple(datasets),
    )
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_path": str(model)}},
            "data": {"train_data_paths": cfg_entries},
        }
    )
    monkeypatch.setattr(cfg_pipeline, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(cfg_pipeline, "compose_steam_advantage_spec", lambda _path: spec)
    monkeypatch.setattr(
        cfg_pipeline,
        "inspect_steam_advantages",
        lambda _spec: {"compatible": True},
    )
    import clothdojo.training.rlinf as rlinf

    monkeypatch.setattr(rlinf, "compose_config", lambda _path, _overrides: cfg)

    report = cfg_pipeline.inspect_cfg_pipeline(
        advantage_config=tmp_path / "advantages.yaml",
        cfg_config=tmp_path / "cfg.yaml",
    )

    assert report["compatible"] is True
    assert report["labels_complete"] is False


def test_cfg_expert_only_preflight(monkeypatch, tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    expert, _ = _configure(monkeypatch, tmp_path)
    # A Good trajectory can contain both high- and low-ranked action chunks.
    _dataset_root = expert / "meta"
    table = pq.read_table(_dataset_root / f"advantages_{ADVANTAGE_TAG}.parquet")
    table = table.set_column(2, "advantage", pa.array([True, True, True, False]))
    pq.write_table(table, _dataset_root / f"advantages_{ADVANTAGE_TAG}.parquet")
    path = _dataset_root / f"advantages_{ADVANTAGE_TAG}.json"
    manifest = json.loads(path.read_text())
    manifest.update(positive=3, negative=1)
    path.write_text(json.dumps(manifest))
    source = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"
    cfg = compose_config(source)
    cfg.data.train_data_paths = [cfg.data.train_data_paths[0]]
    config = tmp_path / "expert_only.yaml"
    OmegaConf.save(cfg, config, resolve=True)

    report = inspect_cfg_training(config)
    assert report["sampling_probabilities"] == {"sft": 1.0, "rollout": 0.0}
    assert report["datasets"][0]["negative_frames"] == 1

    cfg.data.train_data_paths = []
    OmegaConf.save(cfg, config, resolve=True)
    with pytest.raises(ValueError, match="exactly one sft"):
        inspect_cfg_training(config)


def test_cfg_continuation_binds_sft_norm_for_changed_dataset(tmp_path: Path) -> None:
    from test_rollout_artifacts import _fixture

    from clothdojo.training.cfg import _validate_norm_binding

    checkpoint, norm, _ = _fixture(tmp_path)
    info = tmp_path / "expert_info.json"
    info.write_text('{"total_episodes": 1644}')
    model = SimpleNamespace(
        openpi=SimpleNamespace(config_name="pi05_aloha_robotwin"), num_action_chunks=50
    )
    provenance = json.loads((norm / "clothdojo.json").read_text())

    binding = _validate_norm_binding(checkpoint, norm, provenance, info, model)
    assert binding["source"] == "sft_initialization"
    (norm / "norm_stats.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="SHA-256 differs"):
        _validate_norm_binding(checkpoint, norm, provenance, info, model)


def test_cfg_rejects_unbound_norm_for_changed_dataset(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    path = tmp_path / "norm/clothdojo.json"
    provenance = json.loads(path.read_text())
    provenance["dataset_info_sha256"] = "0" * 64
    path.write_text(json.dumps(provenance))
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"
    with pytest.raises(ValueError, match="match neither"):
        inspect_cfg_training(config)


def test_cfg_expert_only_continuation_budget(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    config = (
        Path(__file__).resolve().parents[1]
        / "configs/training/cfg_pi05_aloha_expert822_mirrored.yaml"
    )
    cfg = compose_config(config)
    assert [entry.type for entry in cfg.data.train_data_paths] == ["sft"]
    assert cfg.actor.global_batch_size == 256
    assert cfg.actor.micro_batch_size == 16
    assert cfg.actor.seed == 43
    assert cfg.runner.max_steps == cfg.actor.optim.total_training_steps == 12000
    assert cfg.actor.optim.lr_warmup_steps == 2000
    assert cfg.actor.optim.lr == pytest.approx(1e-5)
    assert cfg.actor.model.openpi.unconditional_prob == 0.1
    assert cfg.actor.model.openpi.positive_only_conditional is True


def test_cfg_launch_manifest_binds_serving_contract(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"
    report = inspect_cfg_training(config)
    cfg = compose_config(config)

    path = write_cfg_launch_manifest(cfg, report)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "clothdojo_cfg_launch"
    assert payload["inference_contract"] == report["inference_contract"]
    assert payload["norm_stats"] == report["norm_stats"]


def test_cfg_config_is_a_guarded_acceptance_run(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"

    cfg = compose_config(config)

    assert cfg.runner.max_steps == 400
    assert cfg.runner.save_interval == 50
    assert cfg.actor.optim.total_training_steps == 400
    assert cfg.actor.global_batch_size == 512
    assert cfg.actor.micro_batch_size == 32
    assert cfg.actor.model.model_type == MODEL_TYPE
    assert cfg.actor.model.openpi.config_name == "pi05_aloha_robotwin"
    assert cfg.actor.model.openpi.unconditional_prob == 0.5
    assert cfg.actor.model.openpi.positive_only_conditional is False
    assert cfg.actor.model.openpi_data.adapt_to_pi is False
    assert cfg.data.balance_dataset_weights is False
    assert cfg.data.train_data_paths[1].get("episodes_path") is None


def test_cfg_one_expected_epoch_run(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    config = (
        Path(__file__).resolve().parents[1]
        / "configs/training/cfg_pi05_aloha_one_expected_epoch.yaml"
    )

    cfg = compose_config(config)

    assert cfg.runner.max_steps == 5000
    assert cfg.runner.save_interval == 500
    assert cfg.actor.micro_batch_size == 32
    assert cfg.actor.global_batch_size == 256
    assert cfg.actor.model.openpi.unconditional_prob == 0.1
    assert cfg.actor.model.openpi.positive_only_conditional is False
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.optim.lr == pytest.approx(2.0e-6)
    assert cfg.actor.optim.lr_scheduler == "openpi_cosine"
    assert cfg.actor.optim.lr_warmup_steps == 200
    assert cfg.actor.optim.total_training_steps == 5000
    assert cfg.actor.optim.min_lr == pytest.approx(2.0e-7)
    assert cfg.data.balance_dataset_weights is False


def test_cfg_rollout400_uses_documented_steam_routing(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha_rollout400.yaml"

    cfg = compose_config(config)

    assert cfg.cluster.num_nodes == 2
    assert cfg.cluster.component_placement.actor == "all"
    assert cfg.runner.max_steps == 30000
    assert cfg.runner.save_interval == 3000
    assert cfg.actor.micro_batch_size == 32
    assert cfg.actor.global_batch_size == 512
    assert cfg.actor.seed == 0
    assert cfg.actor.model.openpi.unconditional_prob == pytest.approx(0.1)
    assert cfg.actor.model.openpi.positive_only_conditional is True
    assert cfg.actor.model.openpi.train_expert_only is False
    assert cfg.actor.model.openpi.cfgrl_guidance_scale == pytest.approx(2.5)
    assert cfg.actor.optim.lr == pytest.approx(1.0e-5)
    assert cfg.actor.optim.lr_scheduler == "cosine"
    assert cfg.actor.optim.lr_warmup_steps == 5000
    assert cfg.actor.optim.total_training_steps == 30000
    assert cfg.actor.optim.min_lr == pytest.approx(0.0)
    assert cfg.data.balance_dataset_weights is True
    assert len(cfg.data.train_data_paths) == 17
    assert all(float(entry.weight) == pytest.approx(1.0) for entry in cfg.data.train_data_paths)


def test_cfg_preflight_rejects_incomplete_advantages(monkeypatch, tmp_path: Path) -> None:
    _, rollout = _configure(monkeypatch, tmp_path)
    path = rollout / f"meta/advantages_{ADVANTAGE_TAG}.parquet"
    table = pq.read_table(path).slice(0, 3)
    pq.write_table(table, path)
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"

    with pytest.raises(ValueError, match="coverage differs"):
        inspect_cfg_training(config)


def test_cfg_preflight_rejects_acceptance_only_rollout(monkeypatch, tmp_path: Path) -> None:
    _, rollout = _configure(monkeypatch, tmp_path)
    sidecar = rollout / "meta/clothdojo.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["training_eligible"] = False
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"

    with pytest.raises(ValueError, match="ineligible for training"):
        inspect_cfg_training(config)


def test_cfg_preflight_rejects_invalid_checkpoint_weights_digest(
    monkeypatch, tmp_path: Path
) -> None:
    expert, _ = _configure(monkeypatch, tmp_path)
    manifest = expert / f"meta/advantages_{ADVANTAGE_TAG}.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint_weights_sha256"] = "not-a-sha256"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"

    with pytest.raises(ValueError, match="checkpoint weights SHA-256"):
        inspect_cfg_training(config)


def test_cfg_preflight_rejects_different_checkpoint_weights(monkeypatch, tmp_path: Path) -> None:
    _, rollout = _configure(monkeypatch, tmp_path)
    manifest = rollout / f"meta/advantages_{ADVANTAGE_TAG}.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint_weights_sha256"] = "c" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"

    with pytest.raises(ValueError, match="different STEAM steam_checkpoint_weights_sha256"):
        inspect_cfg_training(config)


def test_cfg_openpi_arguments_are_shared_by_model_and_worker() -> None:
    cfg = SimpleNamespace(
        openpi=SimpleNamespace(config_name="pi05_aloha_robotwin"),
        model_path="/checkpoint",
        get=lambda key: (
            {
                "adapt_to_pi": False,
                "norm_stats_path": "/stats",
            }
            if key == "openpi_data"
            else None
        ),
    )

    kwargs = _openpi_config_kwargs(cfg, batch_size=64, repo_id="/dataset")

    assert kwargs == {
        "config_name": "pi05_aloha_robotwin",
        "model_path": "/checkpoint",
        "data_kwargs": {
            "adapt_to_pi": False,
            "norm_stats_path": "/stats",
        },
        "batch_size": 64,
        "repo_id": "/dataset",
    }

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    resolved = get_openpi_config(**kwargs)
    assert resolved.batch_size == 64
    assert resolved.data.repo_id == "/dataset"
    assert resolved.data.adapt_to_pi is False
    assert resolved.data.assets.assets_dir == "/"
    assert resolved.data.assets.asset_id == "stats"


def test_project_cfg_routing_matches_positive_only_contract() -> None:
    import torch

    routing = compute_cfg_routing_masks(
        torch.tensor([True, True, False, False]),
        positive_only_conditional=True,
        unconditional_prob=0.1,
        random_values=torch.tensor([0.2, 0.05, 0.9, 0.0]),
    )

    assert routing["conditional_mask"].tolist() == [True, False, False, False]
    assert routing["positive_unconditional_mask"].tolist() == [False, True, False, False]
    assert routing["negative_unconditional_mask"].tolist() == [False, False, True, True]


def test_cfg_advantages_are_vectorized_and_aligned_by_frame_key(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([1, 0, 0], type=pa.int64()),
                "frame_index": pa.array([0, 1, 0], type=pa.int64()),
                "advantage": pa.array([False, True, False], type=pa.bool_()),
            }
        ),
        root / f"meta/advantages_{ADVANTAGE_TAG}.parquet",
    )
    hf_dataset = {
        "episode_index": [0, 1, 0],
        "frame_index": [0, 0, 1],
    }

    labels = _load_aligned_advantages(
        root,
        advantage_tag=ADVANTAGE_TAG,
        hf_dataset=hf_dataset,
    )

    assert labels.tolist() == [False, False, True]


@pytest.mark.parametrize(
    "column,values,error",
    [
        ("advantage", pa.array([True, None, False, False], type=pa.bool_()), ValueError),
        ("episode_index", pa.array([0.1, 0.2, 1.1, 1.2]), TypeError),
        ("frame_index", pa.array([0, None, 0, 1], type=pa.int64()), ValueError),
    ],
)
def test_cfg_rejects_missing_labels_and_noninteger_keys(
    monkeypatch, tmp_path, column, values, error
):
    _, root = _configure(monkeypatch, tmp_path)
    path = root / f"meta/advantages_{ADVANTAGE_TAG}.parquet"
    table = pq.read_table(path)
    pq.write_table(table.set_column(table.schema.get_field_index(column), column, values), path)
    with pytest.raises(error, match="CFG advantage"):
        inspect_cfg_training(
            Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"
        )
    with pytest.raises(error, match="CFG advantage"):
        _load_aligned_advantages(
            root,
            advantage_tag=ADVANTAGE_TAG,
            hf_dataset={"episode_index": [0, 0, 1, 1], "frame_index": [0, 1, 0, 1]},
        )


def test_cfg_uses_project_owned_openpi_cache(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "data/caches/openpi"
    tokenizer = cache / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer")
    monkeypatch.setattr(rlinf_cfg, "_PROJECT_OPENPI_CACHE", cache)
    monkeypatch.delenv("OPENPI_DATA_HOME", raising=False)

    assert rlinf_cfg._configure_project_openpi_cache() == cache
    assert rlinf_cfg.os.environ["OPENPI_DATA_HOME"] == str(cache)


def test_cfg_preserves_explicit_openpi_cache(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "serving_cache"
    tokenizer = cache / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer")
    monkeypatch.setenv("OPENPI_DATA_HOME", str(cache))
    monkeypatch.setattr(rlinf_cfg, "_PROJECT_OPENPI_CACHE", tmp_path / "missing")

    assert rlinf_cfg._configure_project_openpi_cache() == cache
    assert rlinf_cfg.os.environ["OPENPI_DATA_HOME"] == str(cache)

    tokenizer.unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        rlinf_cfg._configure_project_openpi_cache()


@pytest.mark.parametrize("scale", [0.0, 1.0, 2.5])
def test_cfg_sampling_scale_interpolates_velocities(scale: float) -> None:
    import torch

    from clothdojo.training.cfg_model import _sample_cfg_actions

    class ConstantVelocityModel:
        def build_prefix_cache(self, observation):
            return None, None, None

        def run_suffix(self, observation, actions, time, cache, mask):
            return torch.full_like(actions, observation)

        def velocity_from_suffix(self, suffix):
            return suffix

    noise = torch.ones((4, 50, 32))
    actual = _sample_cfg_actions(
        ConstantVelocityModel(),
        2.0,
        5.0,
        noise,
        num_steps=10,
        guidance_scale=scale,
    )
    expected = noise - ((1.0 - scale) * 2.0 + scale * 5.0)
    torch.testing.assert_close(actual, expected)
