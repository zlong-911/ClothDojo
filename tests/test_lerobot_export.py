from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clothdojo.data.lerobot import (
    _encode_episode_videos,
    _writer_features,
    export_review_dataset,
    load_episode_indices,
)


def _source_trajectory(root: Path, source_id: str, action_count: int) -> None:
    path = root / "trajectories" / source_id
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "state_count": action_count + 1,
                "action_count": action_count,
            }
        ),
        encoding="utf-8",
    )


def _bad_annotation(root: Path, source_id: str, start: int, end: int) -> None:
    path = root / "trajectories" / "annotations"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{source_id}.segments.json").write_text(
        json.dumps(
            {
                "schema": "sim1_segments_v1",
                "review_status": "in_progress",
                "segments": [{"start_frame": start, "end_frame": end, "label": "bad"}],
            }
        ),
        encoding="utf-8",
    )


def _review_source(root: Path, source_id: str, frame_count: int) -> None:
    review = root / "reviews" / f"review-{source_id}"
    (review / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "total_episodes": 1,
        "total_frames": frame_count,
        "fps": 50,
        "robot_type": "piper",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": ("videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"),
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["joint", "gripper"],
            },
            "observation.images.main": {
                "dtype": "video",
                "shape": [2, 2, 3],
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["joint", "gripper"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (review / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    provenance = {
        "source": {"episodes": [{"episode_id": source_id}]},
        "annotation": {
            "applied": False,
            "segments": [
                {
                    "source_episode": source_id,
                    "action_start": 0,
                    "action_end_exclusive": frame_count,
                    "length": frame_count,
                    "output_episode_index": 0,
                }
            ],
        },
    }
    (review / "meta" / "newton_cloth_rl.json").write_text(json.dumps(provenance), encoding="utf-8")
    for relative in (
        "data/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/tasks.parquet",
        "videos/observation.images.main/chunk-000/file-000.mp4",
    ):
        path = review / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


class _FakeReader:
    def __init__(self, source):
        self.source = source

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __iter__(self):
        for index in range(self.source.frame_count):
            yield (
                index,
                {
                    "observation.state": np.asarray([index, 0], dtype=np.float32),
                    "observation.images.main": np.zeros((2, 2, 3), dtype=np.uint8),
                    "action": np.asarray([index, 1], dtype=np.float32),
                    "task": "flatten the shirt",
                },
            )


class _FakeWriter:
    def __init__(self):
        self.buffer = []
        self.episodes = []

    def add_frame(self, frame):
        self.buffer.append(frame)

    def save_episode(self):
        self.episodes.append(self.buffer)
        self.buffer = []


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

    monkeypatch.setattr(
        "lerobot.common.datasets.video_utils.encode_video_frames", encode
    )
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


def test_review_export_removes_bad_and_records_explicit_episode_extent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_trajectory(source, "a", 6)
    _bad_annotation(source, "a", 2, 3)
    _review_source(source, "a", 6)
    _source_trajectory(source, "b", 3)
    _review_source(source, "b", 3)
    output = tmp_path / "output"
    writer = _FakeWriter()

    result = export_review_dataset(
        source,
        output,
        _writer_factory=lambda **kwargs: writer,
        _reader_factory=_FakeReader,
    )

    assert [len(episode) for episode in writer.episodes] == [2, 2, 3]
    assert result.output_episodes == 3
    assert result.output_frames == 7
    assert result.dropped_bad_frames == 2
    assert result.full_trajectory_episodes == 1
    metadata = json.loads((output / "meta" / "clothdojo.json").read_text())
    assert metadata["export_status"] == "complete"
    assert metadata["episode_sets"] == {
        "good": [0, 1, 2],
        "full_trajectory": [2],
        "partial_trajectory": [0, 1],
        "annotation_present": [0, 1],
        "review_complete": [],
        "default_good": [2],
    }
    assert metadata["episodes"][0]["source_episode_complete"] is True
    assert metadata["episodes"][0]["full_trajectory"] is False
    assert metadata["episodes"][0]["contains_source_terminal"] is False
    assert metadata["episodes"][1]["contains_source_terminal"] is True
    assert metadata["episodes"][2]["full_trajectory"] is True
    (output / "meta" / "info.json").write_text(json.dumps({"total_episodes": 3}), encoding="utf-8")
    assert load_episode_indices(output) == [0, 1, 2]
    assert load_episode_indices(output, trajectory_extent="full") == [2]
    assert load_episode_indices(output, trajectory_extent="partial") == [0, 1]
