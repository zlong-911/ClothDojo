from __future__ import annotations

import json
from types import SimpleNamespace

import av
import numpy as np
import pytest

from clothdojo.data.rollout_review import (
    _RawVideoEncoder,
    _review_stride,
)


def test_review_stride_preserves_rollout_timeline() -> None:
    assert _review_stride(50.0, 10) == 5
    assert _review_stride(50.0, 25) == 2


@pytest.mark.parametrize("fps", [0, 20, 60])
def test_review_stride_rejects_invalid_or_inexact_rates(fps: int) -> None:
    with pytest.raises(ValueError):
        _review_stride(50.0, fps)


def test_review_rejects_evaluation_collection(tmp_path) -> None:
    (tmp_path / "meta.json").write_text(
        '{"schema":"clothdojo_evaluation_condition"}',
        encoding="utf-8",
    )

    from clothdojo.data.rollout_review import render_review

    with pytest.raises(ValueError, match="unsupported review source schema"):
        render_review(tmp_path)
    assert not (tmp_path / "review").exists()


def test_raw_video_encoder_streams_frames_without_image_files(tmp_path) -> None:
    output = tmp_path / "episode.mp4"
    encoder = _RawVideoEncoder(output, width=64, height=48, fps=10)
    for value in (0, 80, 160):
        encoder.write(np.full((48, 64, 3), value, dtype=np.uint8))
    encoder.close()

    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == ["episode.mp4"]


def test_review_incrementally_renders_id_addressed_trajectory_videos(
    tmp_path, monkeypatch
) -> None:
    def add_trajectory(episode_id: str) -> None:
        path = tmp_path / "trajectories" / episode_id
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "action_count": 3,
                    "metadata": {"asset_id": "shirt_a"},
                }
            ),
            encoding="utf-8",
        )

    class Reader:
        def __init__(self, path) -> None:
            value = json.loads((path / "manifest.json").read_text())
            self.manifest = SimpleNamespace(
                control_fps=50.0,
                action_count=value["action_count"],
                scene=SimpleNamespace(robot_id="piper"),
                metadata=value.get("metadata", {}),
            )

        @staticmethod
        def state_at(index):
            return index

    class Camera:
        width = 64
        height = 48

        def __init__(self, *_args, **_kwargs) -> None:
            self.surface_winding_reversed = False

        def set_cloth_surface_winding_reversed(self, reversed_: bool) -> None:
            self.surface_winding_reversed = reversed_

        @staticmethod
        def __call__(_runtime):
            return {
                "observation.images.cam_high": np.zeros(
                    (1, 48, 64, 3), dtype=np.uint8
                )
            }

        def close(self) -> None:
            pass

    built = []
    add_trajectory("teleop_a")
    monkeypatch.setattr("clothdojo.trajectory.TrajectoryReader", Reader)
    monkeypatch.setattr(
        "clothdojo.sim.scene.build_scene",
        lambda *_a, **kwargs: built.append(kwargs) or object(),
    )
    monkeypatch.setattr("clothdojo.rollout.render.ShirtFlattenCameraObservation", Camera)
    monkeypatch.setattr("clothdojo.data.rollout._TrajectoryRenderState", lambda _scene: object())
    monkeypatch.setattr("clothdojo.data.rollout._restore_render_state", lambda *_a: None)

    from clothdojo.data.rollout_review import render_review

    first = render_review(tmp_path)
    second = render_review(tmp_path)
    add_trajectory("teleop_b")
    third = render_review(tmp_path, trajectory_ids=("teleop_b",))
    fourth = render_review(tmp_path)

    assert (tmp_path / "review/videos/cam_high/teleop_a.mp4").is_file()
    assert (tmp_path / "review/videos/cam_high/teleop_b.mp4").is_file()
    assert (first["rendered"], first["reused"]) == (1, 0)
    assert (second["rendered"], second["reused"]) == (0, 1)
    assert (third["rendered"], third["reused"]) == (1, 0)
    assert third["episodes"] == 2
    assert (fourth["rendered"], fourth["reused"]) == (0, 2)
    assert {item["robot_id"] for item in built} == {"piper"}

    with pytest.raises(ValueError, match="unknown review trajectory ids"):
        render_review(tmp_path, trajectory_ids=("missing",))
