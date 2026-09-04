from __future__ import annotations

import numpy as np
import pytest

from clothdojo.evaluation.folding import aligned_fold_distance, fold_state_diagnostics


def _grid() -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 5), np.linspace(-0.5, 0.5, 4))
    return np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1).astype(np.float32)


def test_aligned_fold_distance_ignores_planar_pose_but_not_height() -> None:
    target = _grid()
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    current = target.copy()
    current[:, :2] = current[:, :2] @ rotation.T + np.asarray([3.0, -2.0])
    assert aligned_fold_distance(current, target, normalization_scale_m=1.0) == pytest.approx(
        0.0, abs=1e-7
    )

    current[:, 2] += 0.2
    assert aligned_fold_distance(current, target, normalization_scale_m=1.0) == pytest.approx(
        0.2
    )


def test_fold_state_diagnostics_capture_compaction_and_motion() -> None:
    initial = _grid()
    terminal = initial.copy()
    terminal[:, :2] *= 0.5
    velocity = np.zeros_like(terminal)
    velocity[:, 0] = 0.3
    result = fold_state_diagnostics(initial, terminal, velocity)
    assert result.convex_footprint_ratio == pytest.approx(0.25)
    assert result.convex_rectangularity == pytest.approx(1.0)
    assert result.footprint_aspect_ratio == pytest.approx(0.5)
    assert result.normalized_thickness_p90 == pytest.approx(0.0)
    assert result.speed_rms_m_s == pytest.approx(0.3)
