from __future__ import annotations

import json
from pathlib import Path

import pytest

from clothdojo.data.indexed_lerobot import open_lerobot_dataset


class _FakeLeRobotDataset:
    def __init__(self, repo_id, *, root, **kwargs):
        self.repo_id = repo_id
        self.root = Path(root)
        self.kwargs = kwargs

    def _query_videos(self, query_timestamps, ep_idx):
        return {"episode": ep_idx, "timestamps": query_timestamps}


def test_indexed_reader_applies_each_camera_clock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "lerobot.common.datasets.lerobot_dataset.LeRobotDataset",
        _FakeLeRobotDataset,
    )
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "clothdojo.json").write_text(
        json.dumps(
            {
                "migration": {
                    "kind": "immutable_chunked_media_index",
                    "video_index": "meta/clothdojo_video_index.jsonl",
                }
            }
        ),
        encoding="utf-8",
    )
    (meta / "clothdojo_video_index.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "timestamp_offsets": {"cam_a": 12.0, "cam_b": 41.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (meta / "info.json").write_text(json.dumps({"fps": 50}), encoding="utf-8")

    dataset = open_lerobot_dataset("local/test", root=tmp_path, download_videos=False)
    result = dataset._query_videos({"cam_a": [0.0, 0.5], "cam_b": [1.0]}, 0)

    assert result == {
        "episode": 0,
        "timestamps": {"cam_a": [12.0, 12.5], "cam_b": [42.5]},
    }
    assert dataset.kwargs["video_backend"] == "pyav"
    assert dataset.kwargs["tolerance_s"] == pytest.approx(0.010001)


def test_regular_reader_stays_on_pinned_lerobot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "lerobot.common.datasets.lerobot_dataset.LeRobotDataset",
        _FakeLeRobotDataset,
    )
    dataset = open_lerobot_dataset("local/test", root=tmp_path, download_videos=False)

    assert type(dataset) is _FakeLeRobotDataset
    assert dataset.kwargs["video_backend"] == "pyav"
