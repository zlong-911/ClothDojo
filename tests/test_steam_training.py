from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from clothdojo.training.steam import inspect_steam_dataset


def test_steam_dataset_identity_is_derived_from_standard_v21_metadata(tmp_path) -> None:
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
                "observation.state": pa.array(
                    states.tolist(), type=pa.list_(pa.float32(), 14)
                ),
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

    report = inspect_steam_dataset(root, cfg=cfg)

    assert report["episodes"] == 2
    assert report["frames"] == 5
    assert report["pair_samples"] == 6
    assert report["prompt"] == "Fold the shirt."
    assert report["per_member_dataset_passes"] == 2.0
