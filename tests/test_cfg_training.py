from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clothdojo.training import rlinf_cfg
from clothdojo.training.cfg import inspect_cfg_training
from clothdojo.training.rlinf import compose_config
from clothdojo.training.rlinf_cfg import MODEL_TYPE, _openpi_config_kwargs


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
        root / "meta" / "advantages_steam_v1.parquet",
    )
    info_digest = hashlib.sha256((root / "meta" / "info.json").read_bytes()).hexdigest()
    positive = sum(labels)
    (root / "meta" / "advantages_steam_v1.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_steam_advantages",
                "format_version": 1,
                "tag": "steam_v1",
                "checkpoint_step": 20000,
                "checkpoint_config_sha256": "a" * 64,
                "checkpoint_weights_sha256": "b" * 64,
                "target_info_sha256": info_digest,
                "episodes": [0, 1],
                "stride": 64,
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
        "CLOTHDOJO_CFG_ADVANTAGE_TAG": "steam_v1",
        "CLOTHDOJO_CFG_INIT_CHECKPOINT": checkpoint.parents[1],
        "CLOTHDOJO_PI05_NORM_STATS": norm,
        "CLOTHDOJO_TRAIN_OUTPUT": tmp_path / "output",
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
    assert report["sampling_probabilities"] == {"sft": 0.75, "rollout": 0.25}
    assert report["datasets"] == [
        {
            "role": "sft",
            "dataset_path": str(expert),
            "weight": 0.75,
            "episodes": 2,
            "frames": 4,
            "fps": 50,
            "advantage_path": str(expert / "meta/advantages_steam_v1.parquet"),
            "advantage_manifest_path": str(
                expert / "meta/advantages_steam_v1.json"
            ),
            "positive_frames": 4,
            "negative_frames": 0,
            "steam_checkpoint_step": 20000,
            "steam_checkpoint_config_sha256": "a" * 64,
            "steam_checkpoint_weights_sha256": "b" * 64,
            "steam_stride": 64,
        },
        {
            "role": "rollout",
            "dataset_path": str(rollout),
            "weight": 0.25,
            "episodes": 2,
            "frames": 4,
            "fps": 50,
            "advantage_path": str(rollout / "meta/advantages_steam_v1.parquet"),
            "advantage_manifest_path": str(
                rollout / "meta/advantages_steam_v1.json"
            ),
            "positive_frames": 1,
            "negative_frames": 3,
            "steam_checkpoint_step": 20000,
            "steam_checkpoint_config_sha256": "a" * 64,
            "steam_checkpoint_weights_sha256": "b" * 64,
            "steam_stride": 64,
        },
    ]


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
    assert cfg.data.train_data_paths[1].episodes_path.endswith("rollout_episodes.json")


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


def test_cfg_preflight_rejects_incomplete_advantages(monkeypatch, tmp_path: Path) -> None:
    _, rollout = _configure(monkeypatch, tmp_path)
    path = rollout / "meta/advantages_steam_v1.parquet"
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
    manifest = expert / "meta/advantages_steam_v1.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint_weights_sha256"] = "not-a-sha256"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    config = Path(__file__).resolve().parents[1] / "configs/training/cfg_pi05_aloha.yaml"

    with pytest.raises(ValueError, match="checkpoint weights SHA-256"):
        inspect_cfg_training(config)


def test_cfg_preflight_rejects_different_checkpoint_weights(
    monkeypatch, tmp_path: Path
) -> None:
    _, rollout = _configure(monkeypatch, tmp_path)
    manifest = rollout / "meta/advantages_steam_v1.json"
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


def test_cfg_uses_project_owned_openpi_cache(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "data/caches/openpi"
    tokenizer = cache / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer")
    monkeypatch.setattr(rlinf_cfg, "_PROJECT_OPENPI_CACHE", cache)
    monkeypatch.delenv("OPENPI_DATA_HOME", raising=False)

    assert rlinf_cfg._configure_project_openpi_cache() == cache
    assert rlinf_cfg.os.environ["OPENPI_DATA_HOME"] == str(cache)
