from __future__ import annotations

import json
from pathlib import Path

import pytest

from clothdojo.data.lerobot import (
    _encode_episode_videos,
    _writer_features,
    load_episode_indices,
    open_lerobot_dataset,
)


def test_regular_v21_reader_uses_validated_video_backend(monkeypatch, tmp_path: Path) -> None:
    call = {}

    class FakeLeRobotDataset:
        def __init__(self, repo_id, *, root, **kwargs):
            call.update(repo_id=repo_id, root=root, kwargs=kwargs)

    monkeypatch.setattr(
        "lerobot.common.datasets.lerobot_dataset.LeRobotDataset",
        FakeLeRobotDataset,
    )

    dataset = open_lerobot_dataset("local/test", root=tmp_path, download_videos=False)

    assert isinstance(dataset, FakeLeRobotDataset)
    assert call == {
        "repo_id": "local/test",
        "root": tmp_path.resolve(),
        "kwargs": {"download_videos": False, "video_backend": "pyav"},
    }


def test_regular_v21_reader_rejects_indexed_media_extension(tmp_path: Path) -> None:
    index = tmp_path / "meta" / "clothdojo_video_index.jsonl"
    index.parent.mkdir()
    index.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="converted externally"):
        open_lerobot_dataset("local/test", root=tmp_path)


def test_writer_features_convert_json_shapes_to_tuples() -> None:
    source = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": None},
        "observation.images.main": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
    }

    normalized = _writer_features(source)

    assert normalized["observation.state"]["shape"] == (14,)
    assert normalized["observation.images.main"]["shape"] == (480, 640, 3)
    assert source["observation.state"]["shape"] == [14]


def test_video_export_uses_supported_h264_for_each_camera(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    validations = []

    def encode(image_dir, video_path, fps, **kwargs):
        calls.append((image_dir, video_path, fps, kwargs))

    monkeypatch.setattr("lerobot.common.datasets.video_utils.encode_video_frames", encode)
    monkeypatch.setattr(
        "clothdojo.data.lerobot._validate_video",
        lambda path, frames: validations.append((path, frames)),
    )

    class Meta:
        video_keys = ("observation.images.high", "observation.images.wrist")

        @staticmethod
        def get_video_file_path(episode_index, key):
            return Path("videos") / key / f"episode_{episode_index:06d}.mp4"

    class Writer:
        root = tmp_path
        meta = Meta()
        fps = 50

        def _get_image_file_path(self, episode_index, image_key, frame_index):
            return (
                self.root
                / "images"
                / image_key
                / f"episode_{episode_index:06d}"
                / f"frame_{frame_index:06d}.png"
            )

    paths = _encode_episode_videos(Writer(), 3)

    assert set(paths) == set(Meta.video_keys)
    assert len(calls) == 2
    assert all(call[2] == 50 for call in calls)
    assert all(call[3]["vcodec"] == "h264" for call in calls)
    assert all(call[3]["crf"] == 23 for call in calls)
    assert all(call[3]["overwrite"] is True for call in calls)
    assert len(validations) == 2


def test_load_episode_indices_reads_selected_trajectory_sets(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 3}), encoding="utf-8")
    (meta / "clothdojo.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_lerobot_export",
                "format_version": 1,
                "export_status": "complete",
                "counts": {"output_episodes": 3},
                "episode_sets": {
                    "good": [0, 1, 2],
                    "full_trajectory": [2],
                    "partial_trajectory": [0, 1],
                    "annotation_present": [0, 1],
                    "review_complete": [],
                    "default_good": [2],
                },
                "episodes": [{}, {}, {}],
            }
        ),
        encoding="utf-8",
    )

    assert load_episode_indices(tmp_path) == [0, 1, 2]
    assert load_episode_indices(tmp_path, trajectory_extent="full") == [2]
    assert load_episode_indices(tmp_path, trajectory_extent="partial") == [0, 1]
