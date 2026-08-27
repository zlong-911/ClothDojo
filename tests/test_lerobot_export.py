from __future__ import annotations

import errno
import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.data.annotations import build_good_segment_plan
from clothdojo.data.lerobot import (
    _encode_episode_videos,
    _match_reusable_episodes,
    _publish_immutable_video,
    _reindex_episode_stats,
    _writer_features,
    export_review_dataset,
    inspect_review_export,
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
        min_segment_frames=1,
        _writer_factory=lambda **kwargs: writer,
        _reader_factory=_FakeReader,
    )

    assert [len(episode) for episode in writer.episodes] == [2, 2, 3]
    assert result.output_episodes == 3
    assert result.output_frames == 7
    assert result.dropped_bad_frames == 2
    assert result.min_segment_frames == 1
    assert result.dropped_short_segments == 0
    assert result.dropped_short_frames == 0
    assert result.full_trajectory_episodes == 1
    metadata = json.loads((output / "meta" / "clothdojo.json").read_text())
    assert metadata["export_status"] == "complete"
    assert metadata["selection"]["min_segment_frames"] == 1
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


def test_review_export_records_dropped_short_range_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_trajectory(source, "native_short", 64)
    _review_source(source, "native_short", 64)
    _source_trajectory(source, "reviewed", 130)
    _bad_annotation(source, "reviewed", 64, 64)
    _review_source(source, "reviewed", 130)
    output = tmp_path / "output"
    writer = _FakeWriter()

    result = export_review_dataset(
        source,
        output,
        _writer_factory=lambda **kwargs: writer,
        _reader_factory=_FakeReader,
    )

    assert [len(episode) for episode in writer.episodes] == [65]
    assert result.source_trajectories == 2
    assert result.represented_source_trajectories == 1
    assert result.dropped_short_segments == 2
    assert result.dropped_short_frames == 128
    assert result.dropped_short_full_trajectory_segments == 1
    assert result.dropped_short_partial_segments == 1
    assert result.dropped_short_source_trajectories == 1

    metadata = json.loads((output / "meta" / "clothdojo.json").read_text())
    assert metadata["counts"]["represented_source_trajectories"] == 1
    assert metadata["counts"]["dropped_short_full_trajectory_segments"] == 1
    assert metadata["counts"]["dropped_short_partial_segments"] == 1
    assert metadata["counts"]["dropped_short_source_trajectories"] == 1
    assert metadata["dropped_short"]["source_trajectories_fully_removed"] == ["native_short"]
    assert [
        (
            item["source_trajectory"],
            item["action_start"],
            item["action_end_exclusive"],
            item["length"],
            item["full_trajectory"],
        )
        for item in metadata["dropped_short"]["ranges"]
    ] == [
        ("native_short", 0, 64, 64, True),
        ("reviewed", 0, 64, 64, False),
    ]


def test_review_inspection_can_validate_a_metadata_only_republish_index(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _source_trajectory(source, "episode", 65)
    _review_source(source, "episode", 65)
    review = source / "reviews" / "review-episode"
    for path in (
        review / "data/chunk-000/file-000.parquet",
        review / "meta/episodes/chunk-000/file-000.parquet",
        review / "meta/tasks.parquet",
        review / "videos/observation.images.main/chunk-000/file-000.mp4",
    ):
        path.unlink()

    plan, sources = inspect_review_export(
        source,
        require_review_payload=False,
    )

    assert plan.output_episodes == 1
    assert set(sources) == {"episode"}
    with pytest.raises(RuntimeError, match="files are missing"):
        inspect_review_export(source)


def test_republish_matches_only_exact_retained_segments(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_trajectory(source, "a", 4)
    _bad_annotation(source, "a", 1, 1)
    loose = build_good_segment_plan(source, min_segment_frames=1)
    strict = build_good_segment_plan(source, min_segment_frames=2)
    reusable = [
        segment.to_json_dict(episode_index=index)
        for index, segment in enumerate(loose.segments)
    ]

    assert [segment.length for segment in loose.segments] == [1, 2]
    assert [segment.length for segment in strict.segments] == [2]
    assert _match_reusable_episodes(strict, reusable) == (1,)


def test_republish_rejects_changed_recorded_review_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_trajectory(source, "a", 65)
    _review_source(source, "a", 65)
    plan, sources = inspect_review_export(source)
    reusable = [plan.segments[0].to_json_dict(episode_index=0)]
    reusable[0].update(
        {
            "review_export": sources["a"].root.name,
            "review_info_sha256": "0" * 64,
            "review_provenance_sha256": sources["a"].provenance_sha256,
        }
    )

    with pytest.raises(RuntimeError, match="review provenance differs"):
        _match_reusable_episodes(plan, reusable, sources=sources)


def test_republish_copies_video_when_filesystem_rejects_hardlinks(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    source.write_bytes(b"validated-video")

    def reject_link(_source, _target):
        raise OSError(errno.EPERM, "hard links disabled")

    monkeypatch.setattr("clothdojo.data.lerobot.os.link", reject_link)

    assert _publish_immutable_video(source, target) == "immutable_copy_reuse"
    assert target.read_bytes() == source.read_bytes()


def test_republish_rebuilds_generated_episode_statistics() -> None:
    raw = {
        "episode_index": 9,
        "stats": {
            "action": {"mean": [1.5]},
            "episode_index": {"mean": [9.0]},
            "index": {"min": [100], "max": [102]},
        },
    }

    result = _reindex_episode_stats(
        raw,
        episode_index=2,
        global_start=7,
        length=3,
    )

    assert raw["episode_index"] == 9
    assert result["episode_index"] == 2
    assert result["stats"]["action"] == {"mean": [1.5]}
    assert result["stats"]["episode_index"] == {
        "min": [2],
        "max": [2],
        "mean": [2.0],
        "std": [0.0],
        "count": [3],
    }
    assert result["stats"]["index"] == {
        "min": [7],
        "max": [9],
        "mean": [8.0],
        "std": [(2 / 3) ** 0.5],
        "count": [3],
    }
