from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clothdojo.training.steam import inspect_steam_dataset


def test_steam_base_explicitly_disables_lora_for_rlinf_model_factory():
    from omegaconf import OmegaConf

    config = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "configs/training/steam_value_shirt_flatten.yaml"
    )
    assert config.actor.model.is_lora is False


def test_geometry_recipe_preserves_sample_and_warmup_budgets():
    from omegaconf import OmegaConf

    configs = Path(__file__).resolve().parents[1] / "configs/training"
    base = OmegaConf.load(configs / "steam_value_shirt_flatten.yaml")
    cfg = OmegaConf.merge(
        base, OmegaConf.load(configs / "steam_value_shirt_flatten_geometry_bs2048.yaml")
    )
    assert cfg.runner.max_steps == 7500
    assert cfg.actor.optim.lr_warmup_steps == 125
    assert cfg.actor.optim.total_training_steps == cfg.runner.max_steps
    assert cfg.actor.global_batch_size * cfg.runner.max_steps == (
        base.actor.global_batch_size * base.runner.max_steps
    )
    assert cfg.actor.global_batch_size * cfg.actor.optim.lr_warmup_steps == (
        base.actor.global_batch_size * base.actor.optim.lr_warmup_steps
    )
    assert cfg.actor.optim.lr == cfg.actor.optim.value_lr == 1e-4
    assert cfg.runner.max_steps % cfg.runner.save_interval == 0


@pytest.mark.parametrize("geometry", [False, True])
def test_steam_dataset_identity_is_derived_from_standard_v21_metadata(tmp_path, geometry) -> None:
    root = tmp_path / "dataset"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    states = np.arange(5 * 14, dtype=np.float32).reshape(5, 14) / 10.0
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 5,
        "fps": 50,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "observation.images.cam_high": {"dtype": "video", "shape": [8, 8, 3]},
            "observation.images.cam_low": {"dtype": "video", "shape": [8, 8, 3]},
        },
    }
    provenance = {
        "schema": "clothdojo_lerobot_export",
        "format_version": 1,
        "export_status": "complete",
        "selection": {
            "rule": "explicit_whole_trajectory_good",
            "required_review_status": "complete",
        },
        "counts": {
            "whole_trajectory_good_only": True,
            "output_episodes": 2,
            "output_frames": 5,
        },
        "episodes": [
            {
                "episode_index": 0,
                "length": 2,
                "quality": "good",
                "review_complete": True,
                "full_trajectory": True,
            },
            {
                "episode_index": 1,
                "length": 3,
                "quality": "good",
                "review_complete": True,
                "full_trajectory": True,
            },
        ],
    }
    if geometry:
        from clothdojo.training.steam_geometry import GEOMETRY_KEY, GEOMETRY_NAMES

        info["features"][GEOMETRY_KEY] = {"dtype": "float32", "shape": [2], "names": GEOMETRY_NAMES}
        provenance["geometry"] = {"contract": {"version": 1}}
        provenance["selection"] = {
            "rule": "approved_whole_trajectory_good",
            "approved_originals": {"a": "user confirmation", "b": "annotation"},
        }
        for row, name in zip(provenance["episodes"], ("a", "b"), strict=True):
            row.update(
                source_trajectory=name,
                quality_basis=provenance["selection"]["approved_originals"][name],
                review_complete=False,
            )
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "clothdojo.json").write_text(json.dumps(provenance), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Fold the shirt."}) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 0, 1, 1, 1], type=pa.int64()),
                "frame_index": pa.array([0, 1, 0, 1, 2], type=pa.int64()),
                "task_index": pa.array([0, 0, 0, 0, 0], type=pa.int64()),
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 14)),
            }
        ),
        data / "file-000.parquet",
    )
    cfg = SimpleNamespace(
        actor=SimpleNamespace(
            model=SimpleNamespace(
                state_mean=states.astype(np.float64).mean(axis=0).tolist(),
                state_std=states.astype(np.float64).std(axis=0).tolist(),
            ),
            global_batch_size=4,
        ),
        runner=SimpleNamespace(max_steps=3),
    )

    if geometry:
        path = data / "file-000.parquet"
        values = np.arange(10, dtype=np.float32).reshape(5, 2) / 10
        table = pq.read_table(path).append_column(
            GEOMETRY_KEY, pa.array(values.tolist(), type=pa.list_(pa.float32(), 2))
        )
        pq.write_table(table, path)
        joined = np.concatenate((states, values), axis=1).astype(np.float64)
        cfg.actor.model.state_dim = 16
        cfg.actor.model.state_keys = ["observation.state", GEOMETRY_KEY]
        cfg.actor.model.geometry_contract = json.dumps({"version": 1})
        cfg.actor.model.state_mean = joined.mean(axis=0).tolist()
        cfg.actor.model.state_std = joined.std(axis=0).tolist()

    report = inspect_steam_dataset(root, cfg=cfg)

    assert report["episodes"] == 2
    assert report["frames"] == 5
    assert report["pair_samples"] == 6
    assert report["prompt"] == "Fold the shirt."
    assert report["per_member_dataset_passes"] == 2.0
    assert report["state_dim"] == (16 if geometry else 14)
