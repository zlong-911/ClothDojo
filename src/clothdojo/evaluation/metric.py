"""Factorized deformable/rigid distance for offline trajectory scoring."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CANONICAL_TEMPLATE_SCHEMA = "clothdojo_canonical_template"
CANONICAL_TEMPLATE_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class CanonicalTemplate:
    asset_id: str
    asset_sha256: str
    topology_sha256: str
    scene_physics_sha256: str
    source_kind: str
    source_sha256: str
    normalization_scale_m: float
    particle_q: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "asset_id",
            "asset_sha256",
            "topology_sha256",
            "scene_physics_sha256",
            "source_kind",
            "source_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must not be empty")
        for name in (
            "asset_sha256",
            "topology_sha256",
            "scene_physics_sha256",
            "source_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not np.isfinite(self.normalization_scale_m) or self.normalization_scale_m <= 0.0:
            raise ValueError("normalization_scale_m must be finite and positive")
        points = np.asarray(self.particle_q)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("particle_q must have shape (particles, 3)")
        if points.dtype != np.float32 or not np.isfinite(points).all():
            raise ValueError("particle_q must be finite float32")

    @property
    def particle_count(self) -> int:
        return int(self.particle_q.shape[0])


def load_canonical_template(path: str | Path) -> CanonicalTemplate:
    source = Path(path).expanduser().resolve()
    required = {
        "schema",
        "format_version",
        "asset_id",
        "asset_sha256",
        "topology_sha256",
        "scene_physics_sha256",
        "source_kind",
        "source_sha256",
        "normalization_scale_m",
        "particle_q",
    }
    with np.load(source, allow_pickle=False) as data:
        actual = set(data.files)
        if actual != required:
            raise ValueError(
                f"invalid canonical template fields: missing={sorted(required - actual)} "
                f"unknown={sorted(actual - required)}"
            )
        if _scalar_string(data["schema"], "schema") != CANONICAL_TEMPLATE_SCHEMA:
            raise ValueError("unsupported canonical template schema")
        version = _scalar_integer(data["format_version"], "format_version")
        if version != CANONICAL_TEMPLATE_FORMAT_VERSION:
            raise ValueError("unsupported canonical template version")
        return CanonicalTemplate(
            asset_id=_scalar_string(data["asset_id"], "asset_id"),
            asset_sha256=_scalar_string(data["asset_sha256"], "asset_sha256"),
            topology_sha256=_scalar_string(data["topology_sha256"], "topology_sha256"),
            scene_physics_sha256=_scalar_string(
                data["scene_physics_sha256"], "scene_physics_sha256"
            ),
            source_kind=_scalar_string(data["source_kind"], "source_kind"),
            source_sha256=_scalar_string(data["source_sha256"], "source_sha256"),
            normalization_scale_m=_scalar_float(
                data["normalization_scale_m"], "normalization_scale_m"
            ),
            particle_q=np.asarray(data["particle_q"], dtype=np.float32).copy(),
        )


def save_canonical_template(path: str | Path, template: CanonicalTemplate) -> None:
    """Atomically publish an already-reviewed, asset-specific target."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            schema=np.asarray(CANONICAL_TEMPLATE_SCHEMA),
            format_version=np.asarray(CANONICAL_TEMPLATE_FORMAT_VERSION, dtype=np.int32),
            asset_id=np.asarray(template.asset_id),
            asset_sha256=np.asarray(template.asset_sha256),
            topology_sha256=np.asarray(template.topology_sha256),
            scene_physics_sha256=np.asarray(template.scene_physics_sha256),
            source_kind=np.asarray(template.source_kind),
            source_sha256=np.asarray(template.source_sha256),
            normalization_scale_m=np.asarray(template.normalization_scale_m, dtype=np.float64),
            particle_q=np.asarray(template.particle_q, dtype=np.float32),
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class FactorizedMetricValues:
    deformable_distance: Any
    rigid_distance: Any
    combined_distance: Any
    speed_rms_m_s: Any
    symmetry_index: Any

    @property
    def reward(self) -> Any:
        return -self.combined_distance


def factorized_cloth_distance(
    particle_q: np.ndarray,
    particle_qd: np.ndarray,
    template: CanonicalTemplate,
    *,
    deformable_weight: float = 0.65,
    symmetry: str = "planar_180",
    symmetry_center_xy: Any | None = None,
) -> FactorizedMetricValues:
    """NumPy reference for translation-invariant planar Procrustes scoring.

    ``planar_180`` constructs its second target at runtime around the supplied
    world-space center; the canonical template always remains the reviewed
    zero-degree target.  The deformable term removes planar translation and
    rotation, while the rigid term retains rotation but deliberately ignores
    planar translation.  Vertex correspondence and height remain intact, so
    flipped or layered cloth is not made equivalent to the target.
    """

    positions, velocities = _validate_particle_batch(particle_q, particle_qd, template)
    if not 0.0 <= deformable_weight <= 1.0:
        raise ValueError("deformable_weight must be in [0, 1]")
    goals = _goal_variants(
        template.particle_q,
        symmetry,
        symmetry_center_xy=symmetry_center_xy,
    ).astype(np.float64)
    current = positions.astype(np.float64)
    current_xy_mean = current[:, :, :2].mean(axis=1)
    current_xy = current[:, :, :2] - current_xy_mean[:, None, :]
    goal_xy_mean = goals[:, :, :2].mean(axis=1)
    goal_xy = goals[:, :, :2] - goal_xy_mean[:, None, :]
    a = np.einsum("vni,bni->bv", goal_xy, current_xy)
    b = np.einsum("vn,bn->bv", goal_xy[:, :, 0], current_xy[:, :, 1])
    b -= np.einsum("vn,bn->bv", goal_xy[:, :, 1], current_xy[:, :, 0])
    angle = np.arctan2(b, a)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    gx = goal_xy[None, :, :, 0]
    gy = goal_xy[None, :, :, 1]
    aligned_centered_xy = np.stack(
        (
            gx * cosine[:, :, None] - gy * sine[:, :, None],
            gx * sine[:, :, None] + gy * cosine[:, :, None],
        ),
        axis=-1,
    )
    aligned_xy = aligned_centered_xy + current_xy_mean[:, None, None, :]
    aligned_z = np.broadcast_to(goals[None, :, :, 2:3], (*aligned_xy.shape[:-1], 1))
    aligned = np.concatenate((aligned_xy, aligned_z), axis=-1)
    deformable = np.linalg.norm(current[:, None] - aligned, axis=-1).mean(axis=-1)
    rigid = np.linalg.norm(aligned_centered_xy - goal_xy[None], axis=-1).mean(axis=-1)
    scale = template.normalization_scale_m
    deformable /= scale
    rigid /= scale
    combined = deformable_weight * deformable + (1.0 - deformable_weight) * rigid
    selected = combined.argmin(axis=1)
    batch = np.arange(len(current))
    selected_deformable = deformable[batch, selected]
    selected_rigid = rigid[batch, selected]
    selected_combined = combined[batch, selected]
    speed = np.sqrt(np.square(velocities.astype(np.float64)).sum(axis=2).mean(axis=1))
    return FactorizedMetricValues(
        deformable_distance=selected_deformable.astype(np.float32),
        rigid_distance=selected_rigid.astype(np.float32),
        combined_distance=selected_combined.astype(np.float32),
        speed_rms_m_s=speed.astype(np.float32),
        symmetry_index=selected.astype(np.int32),
    )


def _goal_variants(
    goal: np.ndarray,
    symmetry: str,
    *,
    symmetry_center_xy: Any | None,
) -> np.ndarray:
    points = np.asarray(goal, dtype=np.float32)
    if symmetry == "none":
        return points[None].copy()
    if symmetry != "planar_180":
        raise ValueError("symmetry must be 'none' or 'planar_180'")
    center = np.asarray(symmetry_center_xy, dtype=np.float32)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("planar_180 symmetry_center_xy must be a finite planar point")
    rotated = points.copy()
    rotated[:, :2] = 2.0 * center - rotated[:, :2]
    return np.stack((points, rotated))


def _validate_particle_batch(particle_q, particle_qd, template):
    positions = np.asarray(particle_q)
    velocities = np.asarray(particle_qd)
    expected = (template.particle_count, 3)
    if positions.ndim != 3 or positions.shape[1:] != expected:
        raise ValueError(f"particle_q must have shape (batch, {expected[0]}, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("particle_qd shape must match particle_q")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("particle state must be finite")
    return positions, velocities


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US" or not str(array.item()):
        raise ValueError(f"{name} must be a non-empty scalar string")
    return str(array.item())


def _scalar_integer(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a scalar integer")
    return int(array.item())


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a scalar number")
    return float(array.item())
