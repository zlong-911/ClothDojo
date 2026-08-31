from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import ClassVar

import pytest

from clothdojo.data.lerobot import (
    _encode_episode_videos,
    _shift_episode_stats,
    _write_temporary_png,
    _writer_features,
    load_episode_indices,
    load_single_task_prompt,
    merge_lerobot_shards,
    open_lerobot_dataset,
)


@pytest.fixture
def lerobot_shards(tmp_path):
    import numpy as np
    from test_mirror import _collection

    from clothdojo.data.annotations import build_good_segment_plan
    from clothdojo.data.rollout import CAMERA_NAMES, RolloutLeRobotWriter

    source, original_id = _collection(tmp_path)
    second_id = "episode_001"
    shutil.copytree(
        source / "trajectories" / original_id,
        source / "trajectories" / second_id,
    )
    roots = []
    for index, source_id in enumerate((second_id, original_id)):
        plan = build_good_segment_plan(source, min_segment_frames=1, source_ids=[source_id])
        output = tmp_path / f"shard_{index}"
        sink = RolloutLeRobotWriter(
            output,
            fps=50,
            prompt="Fold the shirt.",
            scenario={"task_name": "shirt_fold", "prompt": "Fold the shirt."},
            trajectory=str(source),
            policy={},
            training_eligible=True,
            source_kind="trajectory_collection",
            rendering={"backend": "newton", "trajectory_shard": {"index": index, "count": 2}},
            image_height=4,
            image_width=4,
            image_writer_threads=1,
            selection_plan=plan,
        )
        try:
            for segment in plan.segments:
                for frame in range(segment.length):
                    observation = {
                        "observation.state": np.full((1, 14), index, dtype=np.float32),
                        **{
                            f"observation.images.{camera}": np.full(
                                (1, 4, 4, 3), frame * 20, dtype=np.uint8
                            )
                            for camera in CAMERA_NAMES
                        },
                    }
                    sink.add_frame(observation, np.full(14, index, dtype=np.float32))
                sink.save_episode()
            sink.close()
        finally:
            sink.writer.stop_image_writer()
        roots.append(output)
    return source, roots


def test_merge_shards_reorders_data_stats_and_preserves_videos(lerobot_shards, tmp_path):
    import pyarrow.parquet as pq

    _source, shards = lerobot_shards
    output = tmp_path / "merged"
    result = merge_lerobot_shards(shards, output)

    assert result["episodes"] == 2
    assert result["frames"] == 3
    assert result["videos"] == 8
    tables = [pq.read_table(path) for path in sorted((output / "data").rglob("*.parquet"))]
    assert [table["episode_index"].to_pylist() for table in tables] == [[0], [1, 1]]
    assert [table["index"].to_pylist() for table in tables] == [[0], [1, 2]]
    assert [table["action"].to_pylist()[0][0] for table in tables] == [1.0, 0.0]
    sidecar = json.loads((output / "meta/clothdojo.json").read_text())
    assert [row["source_trajectory"] for row in sidecar["episodes"]] == [
        "episode_000", "episode_001"
    ]
    assert sidecar["rendering"] == {"backend": "newton", "parallel_export": {"shards": 2}}
    stats = [
        json.loads(line)
        for line in (output / "meta/episodes_stats.jsonl").read_text().splitlines()
    ]
    assert [row["stats"]["index"]["mean"] for row in stats] == [[0.0], [1.5]]
    for index, original in enumerate(reversed(shards)):
        for video in (original / "videos").rglob("*.mp4"):
            relative = video.relative_to(original).with_name(f"episode_{index:06d}.mp4")
            assert (output / relative).read_bytes() == video.read_bytes()


@pytest.mark.parametrize("damage", ["annotation", "episode_length", "unfinished"])
def test_merge_rejects_stale_or_inconsistent_shards(lerobot_shards, tmp_path, damage):
    source, shards = lerobot_shards
    if damage == "annotation":
        annotation = source / "trajectories/annotations/episode_000.segments.json"
        payload = json.loads(annotation.read_text())
        payload["updated_at"] = "2026-08-31T00:00:00Z"
        annotation.write_text(json.dumps(payload))
    elif damage == "episode_length":
        path = shards[0] / "meta/episodes.jsonl"
        row = json.loads(path.read_text())
        row["length"] += 1
        path.write_text(json.dumps(row) + "\n")
    else:
        path = shards[0] / "meta/clothdojo.json"
        sidecar = json.loads(path.read_text())
        sidecar["export_status"] = "in_progress"
        path.write_text(json.dumps(sidecar))

    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=r"provenance differs|counts differ|completed"):
        merge_lerobot_shards(shards, output)
    assert not output.exists()
    assert not output.with_name(".rejected.tmp").exists()


def test_merge_does_not_clean_staging_created_during_preflight(
    lerobot_shards, tmp_path, monkeypatch
):
    from clothdojo.data.annotations import build_good_segment_plan

    _source, shards = lerobot_shards
    output = tmp_path / "concurrent"
    staging = output.with_name(".concurrent.tmp")
    sentinel = staging / "other-merge-data"

    def plan_with_concurrent_merge(*args, **kwargs):
        (staging / "meta").mkdir(parents=True)
        sentinel.write_bytes(b"owned by another merge")
        return build_good_segment_plan(*args, **kwargs)

    monkeypatch.setattr(
        "clothdojo.data.annotations.build_good_segment_plan", plan_with_concurrent_merge
    )

    with pytest.raises(FileExistsError):
        merge_lerobot_shards(shards, output)

    assert sentinel.read_bytes() == b"owned by another merge"
    assert not output.exists()


def test_merged_episode_stats_use_global_episode_and_frame_indices() -> None:
    record = {
        "episode_index": 1,
        "stats": {
            "episode_index": {
                "min": [1], "max": [1], "mean": [1.0], "std": [0.0], "count": [4]
            },
            "index": {
                "min": [7], "max": [10], "mean": [8.5], "std": [1.2], "count": [4]
            },
        },
    }

    _shift_episode_stats(record, episode_index=9, frame_offset=100)

    assert record["stats"]["episode_index"] == {
        "min": [9], "max": [9], "mean": [9.0], "std": [0.0], "count": [4]
    }
    assert record["stats"]["index"] == {
        "min": [100], "max": [103], "mean": [101.5], "std": [1.2], "count": [4]
    }


def test_regular_v21_reader_uses_validated_video_backend(monkeypatch, tmp_path: Path) -> None:
    call = {}
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(
        '{"codebase_version":"v2.1"}', encoding="utf-8"
    )

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
    (index.parent / "info.json").write_text(
        '{"codebase_version":"v2.1"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="converted externally"):
        open_lerobot_dataset("local/test", root=tmp_path)


def test_regular_v21_reader_rejects_other_versions(tmp_path: Path) -> None:
    info = tmp_path / "meta" / "info.json"
    info.parent.mkdir()
    info.write_text('{"codebase_version":"v3.0"}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"standard LeRobot v2\.1"):
        open_lerobot_dataset("local/test", root=tmp_path)


def test_single_task_prompt_uses_standard_lerobot_metadata(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Flatten the shirt."}) + "\n",
        encoding="utf-8",
    )

    assert load_single_task_prompt(tmp_path) == "Flatten the shirt."


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
        episode_buffer: ClassVar = {"frame_index": [0, 1]}

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
    assert all(frames == 2 for _path, frames in validations)


def test_video_export_can_use_hardware_h264(tmp_path: Path, monkeypatch) -> None:
    calls = []
    validations = []
    monkeypatch.setattr("clothdojo.data.lerobot.VIDEO_ENCODER", "h264_nvenc")
    monkeypatch.setattr(
        "clothdojo.data.lerobot._encode_nvenc_video_frames",
        lambda image_dir, video_path, fps: calls.append((image_dir, video_path, fps)),
    )
    monkeypatch.setattr(
        "clothdojo.data.lerobot._validate_video",
        lambda path, frames: validations.append((path, frames)),
    )

    class Meta:
        video_keys = ("observation.images.high",)

        @staticmethod
        def get_video_file_path(episode_index, key):
            return Path("videos") / key / f"episode_{episode_index:06d}.mp4"

    class Writer:
        root = tmp_path
        meta = Meta()
        fps = 50
        episode_buffer: ClassVar = {"frame_index": [0]}

        def _get_image_file_path(self, episode_index, image_key, frame_index):
            image_dir = self.root / "images" / image_key / f"episode_{episode_index:06d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / "frame_000000.png").touch()
            return image_dir / f"frame_{frame_index:06d}.png"

    _encode_episode_videos(Writer(), 4)

    assert len(calls) == 1
    assert calls[0][2] == 50
    assert validations == [(calls[0][1], 1)]


def test_video_export_rejects_missing_png_against_recorded_episode(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    monkeypatch.setattr("clothdojo.data.lerobot.VIDEO_ENCODER", "h264")
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4)).save(images / "frame_000000.png")
    writer = SimpleNamespace(
        root=tmp_path,
        fps=50,
        episode_buffer={"frame_index": [0, 1]},
        meta=SimpleNamespace(
            video_keys=["observation.images.high"],
            get_video_file_path=lambda *_args: Path("videos/episode_000000.mp4"),
        ),
        _get_image_file_path=lambda **_kwargs: images / "frame_000000.png",
    )

    with pytest.raises(RuntimeError, match=r"frame count differs.*1 != 2"):
        _encode_episode_videos(writer, 0)


def test_png_callback_returns_after_error_so_worker_can_complete_queue(tmp_path, caplog):
    from PIL import Image

    _write_temporary_png(Image.new("RGB", (4, 4)), tmp_path / "missing/frame.png")

    assert "cannot write temporary video frame" in caplog.text


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
