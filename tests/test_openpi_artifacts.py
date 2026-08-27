from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clothdojo.training.openpi_artifacts import (
    inspect_openpi_dataset,
    inspect_openpi_norm_stats,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path, *, frames: int = 65) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "total_frames": frames,
        "total_tasks": 1,
        "fps": 50,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            "observation.images.cam_high": {"dtype": "video", "shape": [8, 8, 3]},
            "observation.images.cam_left_wrist": {
                "dtype": "video",
                "shape": [8, 8, 3],
            },
            "observation.images.cam_right_wrist": {
                "dtype": "video",
                "shape": [8, 8, 3],
            },
        },
    }
    sidecar = {
        "schema": "clothdojo_lerobot_export",
        "format_version": 1,
        "export_status": "complete",
        "selection": {"min_segment_frames": frames},
        "counts": {
            "output_episodes": 1,
            "output_frames": frames,
            "min_segment_frames": frames,
        },
        "episodes": [{"episode_index": 0, "length": frames}],
    }
    _write_json(dataset / "meta/info.json", info)
    _write_json(dataset / "meta/clothdojo.json", sidecar)
    (dataset / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "flatten the shirt"}) + "\n",
        encoding="utf-8",
    )
    (dataset / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": frames}) + "\n",
        encoding="utf-8",
    )

    norm = tmp_path / "norm"
    vector = [0.0] * 14
    _write_json(
        norm / "norm_stats.json",
        {
            "norm_stats": {
                "state": dict.fromkeys(("mean", "std", "q01", "q99"), vector),
                "actions": dict.fromkeys(("mean", "std", "q01", "q99"), vector),
            }
        },
    )
    _write_json(
        norm / "clothdojo.json",
        {
            "schema": "clothdojo_openpi_norm_stats",
            "format_version": 1,
            "dataset_info_sha256": _sha256(dataset / "meta/info.json"),
            "dataset_sidecar_sha256": _sha256(dataset / "meta/clothdojo.json"),
            "config_name": "pi05_aloha_robotwin",
            "data_kwargs": {"default_prompt": "Flatten the shirt."},
            "action_horizon": 50,
            "frames": frames,
            "episodes": 1,
        },
    )
    return dataset, norm


def test_openpi_artifacts_validate_current_dataset_and_norm_stats(tmp_path: Path) -> None:
    dataset_path, norm_path = _artifacts(tmp_path)

    dataset = inspect_openpi_dataset(dataset_path)
    norm = inspect_openpi_norm_stats(
        norm_path,
        dataset=dataset,
        config_name="pi05_aloha_robotwin",
        data_kwargs={"default_prompt": "Flatten the shirt."},
        action_horizon=50,
    )

    assert dataset["frames"] == 65
    assert norm["dataset_info_sha256"] == dataset["info_sha256"]


def test_openpi_artifacts_reject_short_dataset(tmp_path: Path) -> None:
    dataset_path, _ = _artifacts(tmp_path, frames=64)

    with pytest.raises(ValueError, match="min_segment_frames must be >= 65"):
        inspect_openpi_dataset(dataset_path)


def test_openpi_artifacts_reject_norm_stats_from_another_dataset(
    tmp_path: Path,
) -> None:
    dataset_path, norm_path = _artifacts(tmp_path)
    provenance_path = norm_path / "clothdojo.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["dataset_info_sha256"] = "0" * 64
    _write_json(provenance_path, provenance)

    with pytest.raises(ValueError, match="dataset_info_sha256"):
        inspect_openpi_norm_stats(
            norm_path,
            dataset=inspect_openpi_dataset(dataset_path),
            config_name="pi05_aloha_robotwin",
            data_kwargs={"default_prompt": "Flatten the shirt."},
            action_horizon=50,
        )
