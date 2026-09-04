"""Target-aware geometric diagnostics for folded-garment states.

The flattening benchmark owns a settled, asset-specific canonical target.  Folding
does not have such a target yet, so this module deliberately contains only the
small mathematical core needed to compare a state with an explicitly supplied
folded target.  Target construction and calibration remain analysis concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True, slots=True)
class FoldStateDiagnostics:
    """Rotation/translation-invariant diagnostics for one terminal state."""

    initial_footprint_area_m2: float
    terminal_convex_footprint_area_m2: float
    convex_footprint_ratio: float
    convex_rectangularity: float
    footprint_aspect_ratio: float
    normalized_thickness_p90: float
    speed_rms_m_s: float


def aligned_fold_distance(
    particle_q: np.ndarray,
    target_q: np.ndarray,
    *,
    normalization_scale_m: float,
) -> float:
    """Mean corresponding-particle 3D distance after planar rigid alignment.

    The target is translated and continuously rotated in the table plane to fit
    the state.  Height is not translated: both states must use the same table
    frame, so lifted, wrinkled, or incorrectly stacked layers remain penalized.
    Reflection and correspondence permutations are never allowed.
    """

    current = _particle_positions(particle_q, "particle_q").astype(np.float64)
    target = _particle_positions(target_q, "target_q").astype(np.float64)
    if current.shape != target.shape:
        raise ValueError("particle_q and target_q must have identical shape")
    scale = float(normalization_scale_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization_scale_m must be finite and positive")

    current_mean = current[:, :2].mean(axis=0)
    target_mean = target[:, :2].mean(axis=0)
    current_xy = current[:, :2] - current_mean
    target_xy = target[:, :2] - target_mean

    # arg(sum(conj(target) * current)) is the least-squares planar rotation.
    real = np.einsum("ni,ni->", target_xy, current_xy)
    imag = np.einsum("n,n->", target_xy[:, 0], current_xy[:, 1])
    imag -= np.einsum("n,n->", target_xy[:, 1], current_xy[:, 0])
    angle = np.arctan2(imag, real)
    cosine, sine = np.cos(angle), np.sin(angle)
    aligned_xy = np.empty_like(target_xy)
    aligned_xy[:, 0] = target_xy[:, 0] * cosine - target_xy[:, 1] * sine
    aligned_xy[:, 1] = target_xy[:, 0] * sine + target_xy[:, 1] * cosine
    aligned_xy += current_mean

    residual_xy = current[:, :2] - aligned_xy
    residual_z = current[:, 2] - target[:, 2]
    distance = np.sqrt(np.square(residual_xy).sum(axis=1) + np.square(residual_z))
    return float(distance.mean() / scale)


def fold_state_diagnostics(
    initial_particle_q: np.ndarray,
    terminal_particle_q: np.ndarray,
    terminal_particle_qd: np.ndarray,
) -> FoldStateDiagnostics:
    """Compute target-free fold diagnostics using the initial flat state as scale."""

    initial = _particle_positions(initial_particle_q, "initial_particle_q")
    terminal = _particle_positions(terminal_particle_q, "terminal_particle_q")
    velocity = _particle_positions(terminal_particle_qd, "terminal_particle_qd")
    if initial.shape != terminal.shape or velocity.shape != terminal.shape:
        raise ValueError("initial, terminal, and velocity particle arrays must match")

    initial_area, _initial_box_area, _initial_aspect = _planar_hull(initial[:, :2])
    terminal_area, terminal_box_area, terminal_aspect = _planar_hull(terminal[:, :2])
    scale = float(np.sqrt(initial_area))
    z05, z95 = np.percentile(terminal[:, 2], (5.0, 95.0))
    speed = np.sqrt(np.square(velocity.astype(np.float64)).sum(axis=1).mean())
    return FoldStateDiagnostics(
        initial_footprint_area_m2=initial_area,
        terminal_convex_footprint_area_m2=terminal_area,
        convex_footprint_ratio=terminal_area / initial_area,
        convex_rectangularity=terminal_area / terminal_box_area,
        footprint_aspect_ratio=terminal_aspect,
        normalized_thickness_p90=float((z95 - z05) / scale),
        speed_rms_m_s=float(speed),
    )


def _particle_positions(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"{name} must have shape (particles, 3)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite")
    return points


def _planar_hull(points: np.ndarray) -> tuple[float, float, float]:
    try:
        hull = ConvexHull(np.asarray(points, dtype=np.float64))
    except Exception as exc:
        raise ValueError("particle projection has no non-degenerate convex hull") from exc
    hull_points = np.asarray(points[hull.vertices], dtype=np.float32)
    _center, (width, height), _angle = cv2.minAreaRect(hull_points)
    box_area = float(width * height)
    longest = max(float(width), float(height))
    area = float(hull.volume)
    if area <= 0.0 or box_area <= 0.0 or longest <= 0.0:
        raise ValueError("particle projection has a degenerate footprint")
    return area, box_area, min(float(width), float(height)) / longest
