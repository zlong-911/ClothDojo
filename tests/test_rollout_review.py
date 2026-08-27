from __future__ import annotations

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


def test_review_rejects_non_rollout_collection(tmp_path) -> None:
    (tmp_path / "meta.json").write_text(
        '{"schema":"clothdojo_evaluation_condition"}',
        encoding="utf-8",
    )

    from clothdojo.data.rollout_review import render_rollout_review

    with pytest.raises(ValueError, match="unsupported rollout schema"):
        render_rollout_review(tmp_path)
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
