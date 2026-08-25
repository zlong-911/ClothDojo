from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clothdojo.training.cfg_pipeline import _validate_existing_label_manifests
from clothdojo.training.episode_selection import validate_episode_selection
from clothdojo.training.steam_advantages import (
    _checkpoint_weights,
    _publish_dataset_labels,
    compose_steam_advantage_spec,
    inspect_steam_advantages,
)


def _dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    features = {
        "observation.images.primary": {
            "dtype": "video",
            "shape": [8, 8, 3],
        },
        "observation.images.third_person": {
            "dtype": "video",
            "shape": [8, 8, 3],
        },
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "total_episodes": 2,
                "total_frames": 4,
                "fps": 50,
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 2})
        + "\n"
        + json.dumps({"episode_index": 1, "length": 2})
        + "\n",
        encoding="utf-8",
    )


def _spec(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "model_state_dict").mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model_state_dict" / "full_weights.pt").write_bytes(b"weights")
    expert = tmp_path / "expert"
    rollout = tmp_path / "rollout"
    _dataset(expert)
    _dataset(rollout)
    selection = tmp_path / "selection.json"
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
    config = tmp_path / "steam.yaml"
    config.write_text(
        "\n".join(
            [
                "schema: clothdojo_steam_advantages",
                "format_version: 1",
                f"checkpoint: {checkpoint}",
                "checkpoint_step: 20000",
                "tag: steam_test",
                "prompt: flatten the shirt",
                "stride: 64",
                "camera_keys:",
                "  - observation.images.primary",
                "  - observation.images.third_person",
                "batch_size: 16",
                "num_workers: 1",
                "num_processes: 2",
                "master_port: 29531",
                f"work_dir: {tmp_path / 'work'}",
                "datasets:",
                "  - role: sft",
                f"    source_path: {expert}",
                f"    target_path: {expert}",
                "    positive_fraction: 0.5",
                "  - role: rollout",
                f"    source_path: {rollout}",
                f"    target_path: {rollout}",
                f"    episodes_path: {selection}",
                "    positive_fraction: 0.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return compose_steam_advantage_spec(config)


def test_steam_advantage_spec_is_owned_and_explicit(tmp_path: Path) -> None:
    spec = _spec(tmp_path)

    report = inspect_steam_advantages(spec)

    assert report["compatible"] is True
    assert report["checkpoint_step"] == 20000
    assert report["prompt"] == "flatten the shirt"
    assert report["tag"] == "steam_test"
    assert report["camera_keys"] == [
        "observation.images.primary",
        "observation.images.third_person",
    ]
    assert report["datasets"][0]["episodes"] == [0, 1]
    assert report["datasets"][1]["positive_fraction"] == 0.25


def test_publish_steam_labels_is_atomic_and_exact(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    entry = spec.datasets[0]
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "frame_index": pa.array([0], type=pa.int64()),
                "advantage_continuous": pa.array([0.9], type=pa.float32()),
            }
        ),
        shard_dir / "rank_000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([1], type=pa.int64()),
                "frame_index": pa.array([0], type=pa.int64()),
                "advantage_continuous": pa.array([-0.2], type=pa.float32()),
            }
        ),
        shard_dir / "rank_001.parquet",
    )

    _publish_dataset_labels(
        spec=spec,
        entry=entry,
        shard_dir=shard_dir,
        weights_path=_checkpoint_weights(spec.checkpoint),
    )

    table = pq.read_table(entry.output_path)
    assert table.column_names == [
        "episode_index",
        "frame_index",
        "advantage",
        "advantage_continuous",
    ]
    assert table["advantage"].to_pylist() == [True, False, True, False]
    manifest = json.loads(entry.output_path.with_suffix(".json").read_text())
    assert manifest["rows"] == 4
    assert manifest["positive"] == 2
    assert manifest["negative"] == 2
    assert manifest["checkpoint_step"] == 20000
    assert manifest["prompt"] == "flatten the shirt"


def test_existing_labels_are_bound_to_current_advantage_config(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    report = inspect_steam_advantages(spec)
    common = {
        "schema": "clothdojo_steam_advantages",
        "format_version": 1,
        "tag": spec.tag,
        "prompt": spec.prompt,
        "checkpoint_step": spec.checkpoint_step,
        "checkpoint_config_sha256": report["checkpoint_config_sha256"],
        "checkpoint_weights_sha256": report["checkpoint_weights_sha256"],
        "stride": spec.stride,
        "camera_keys": list(spec.camera_keys),
    }
    for dataset in spec.datasets:
        dataset.output_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    **common,
                    "role": dataset.role,
                    "positive_fraction_requested": dataset.positive_fraction,
                }
            ),
            encoding="utf-8",
        )

    _validate_existing_label_manifests(spec, report)
    (spec.checkpoint / "model_state_dict" / "full_weights.pt").write_bytes(
        b"different weights"
    )
    changed = inspect_steam_advantages(spec)

    with pytest.raises(ValueError, match="checkpoint_weights_sha256"):
        _validate_existing_label_manifests(spec, changed)


@pytest.mark.parametrize("episodes", [[True], [1.5], ["1"]])
def test_episode_selection_rejects_non_integer_entries(episodes: list[object]) -> None:
    with pytest.raises(TypeError, match="must be integers"):
        validate_episode_selection(episodes, total_episodes=3)


@pytest.mark.parametrize("episodes", [[1, 0], [0, 0]])
def test_episode_selection_requires_strictly_increasing_entries(
    episodes: list[int],
) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_episode_selection(episodes, total_episodes=3)
