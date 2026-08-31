"""Deterministic mounting-interface registration and fit checks.

The solver operates on explicit mounting datums (hole axes and a mating-plane
normal).  Geometry extraction is deliberately separate: a result derived from
an STL must remain labelled as mesh-derived, while a drawing or B-Rep can be
declared with its own measurement uncertainty.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

EvidenceKind = Literal["drawing", "brep", "mesh", "physical"]


@dataclass(frozen=True, slots=True)
class Evidence:
    kind: EvidenceKind
    reference: str
    uncertainty_mm: float | None


@dataclass(frozen=True, slots=True)
class MountInterface:
    name: str
    holes_mm: np.ndarray
    outward_normal: np.ndarray
    evidence: Evidence
    datum_direction: np.ndarray | None = None
    clearance_hole_diameter_mm: float | None = None


@dataclass(frozen=True, slots=True)
class AlignmentSpec:
    fixed: MountInterface
    moving: MountInterface
    pattern_tolerance_mm: float
    fastener_diameter_mm: float | None = None
    assembly_margin_mm: float = 0.0


def load_spec(path: str | Path) -> AlignmentSpec:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read mount-alignment spec {source}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != "clothdojo_mount_alignment_v1":
        raise ValueError("mount-alignment spec must use schema clothdojo_mount_alignment_v1")
    return AlignmentSpec(
        fixed=_interface(value.get("fixed"), "fixed"),
        moving=_interface(value.get("moving"), "moving"),
        pattern_tolerance_mm=_positive_float(
            value.get("pattern_tolerance_mm"), "pattern_tolerance_mm"
        ),
        fastener_diameter_mm=_optional_positive_float(
            value.get("fastener_diameter_mm"), "fastener_diameter_mm"
        ),
        assembly_margin_mm=_nonnegative_float(
            value.get("assembly_margin_mm", 0.0), "assembly_margin_mm"
        ),
    )


def align_mounts(spec: AlignmentSpec) -> dict[str, Any]:
    """Map ``moving`` into ``fixed`` coordinates and report every residual.

    Hole identities are not assumed.  All correspondences are enumerated for
    the small bolt patterns used by robot bases.  The mating-plane normals are
    constrained to oppose one another.  A datum direction is required on both
    interfaces to resolve rotationally symmetric hole patterns.
    """

    fixed = spec.fixed
    moving = spec.moving
    if len(fixed.holes_mm) != len(moving.holes_mm):
        return _incompatible_count_report(spec)
    if len(fixed.holes_mm) > 8:
        raise ValueError("automatic correspondence is limited to at most 8 holes")

    fixed_center = fixed.holes_mm.mean(axis=0)
    moving_center = moving.holes_mm.mean(axis=0)
    fixed_centered = fixed.holes_mm - fixed_center
    moving_centered = moving.holes_mm - moving_center
    pattern_span = max(
        float(np.linalg.norm(fixed_centered, axis=1).max()),
        float(np.linalg.norm(moving_centered, axis=1).max()),
        1.0,
    )
    candidates = []
    for permutation in itertools.permutations(range(len(fixed.holes_mm))):
        target = fixed.holes_mm[np.asarray(permutation)]
        rotation = _constrained_rotation(
            moving_centered,
            target - fixed_center,
            moving.outward_normal,
            -fixed.outward_normal,
            moving.datum_direction,
            fixed.datum_direction,
            pattern_span,
        )
        translation = fixed_center - rotation @ moving_center
        transformed = (rotation @ moving.holes_mm.T).T + translation
        residual_vectors = transformed - target
        residuals = np.linalg.norm(residual_vectors, axis=1)
        normal_angle = _angle_degrees(rotation @ moving.outward_normal, -fixed.outward_normal)
        datum_angle = None
        if moving.datum_direction is not None and fixed.datum_direction is not None:
            datum_angle = _angle_degrees(rotation @ moving.datum_direction, fixed.datum_direction)
        score = float(np.mean(residuals**2) + (pattern_span * math.radians(normal_angle)) ** 2)
        if datum_angle is not None:
            score += float((pattern_span * math.radians(datum_angle)) ** 2)
        candidates.append(
            {
                "score": score,
                "permutation": permutation,
                "rotation": rotation,
                "translation": translation,
                "transformed": transformed,
                "residual_vectors": residual_vectors,
                "residuals": residuals,
                "normal_angle": normal_angle,
                "datum_angle": datum_angle,
            }
        )

    candidates.sort(key=lambda item: (item["score"], item["permutation"]))
    best = candidates[0]
    score_epsilon = max(1e-12, best["score"] * 1e-9)
    equivalent = [item for item in candidates if item["score"] - best["score"] <= score_epsilon]
    max_residual = float(best["residuals"].max())
    pattern_status = (
        "within_tolerance" if max_residual <= spec.pattern_tolerance_mm else "incompatible"
    )
    pose_status = (
        "not_applicable"
        if pattern_status == "incompatible"
        else "ambiguous"
        if len(equivalent) > 1
        else "resolved"
    )

    transform = np.eye(4)
    transform[:3, :3] = best["rotation"]
    transform[:3, 3] = best["translation"]
    mechanical_fit = _mechanical_fit(spec, max_residual)
    return {
        "schema": "clothdojo_mount_alignment_report_v1",
        "fixed": _interface_report(fixed),
        "moving": _interface_report(moving),
        "pattern": {
            "status": pattern_status,
            "tolerance_mm": spec.pattern_tolerance_mm,
            "rms_residual_mm": float(np.sqrt(np.mean(best["residuals"] ** 2))),
            "max_residual_mm": max_residual,
            "pairwise_distances_fixed_mm": _pairwise_distances(fixed.holes_mm),
            "pairwise_distances_moving_mm": _pairwise_distances(moving.holes_mm),
        },
        "pose": {
            "status": pose_status,
            "equivalent_solution_count": len(equivalent),
            "moving_to_fixed_matrix": transform.tolist(),
            "moving_hole_to_fixed_hole": list(best["permutation"]),
            "mating_normal_error_degrees": best["normal_angle"],
            "datum_direction_error_degrees": best["datum_angle"],
            "per_hole": [
                {
                    "moving_hole": moving_index,
                    "fixed_hole": fixed_index,
                    "center_error_mm": float(best["residuals"][moving_index]),
                    "error_vector_fixed_mm": best["residual_vectors"][moving_index].tolist(),
                }
                for moving_index, fixed_index in enumerate(best["permutation"])
            ],
        },
        "mechanical_fit": mechanical_fit,
        "limitations": _limitations(spec, pose_status),
    }


def _constrained_rotation(
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    moving_normal: np.ndarray,
    fixed_mating_normal: np.ndarray,
    moving_datum: np.ndarray | None,
    fixed_datum: np.ndarray | None,
    span: float,
) -> np.ndarray:
    cross_covariance = moving_points.T @ fixed_points
    normal_weight = len(moving_points) * span * span
    cross_covariance += normal_weight * np.outer(moving_normal, fixed_mating_normal)
    if moving_datum is not None and fixed_datum is not None:
        cross_covariance += normal_weight * np.outer(moving_datum, fixed_datum)
    left, _, right_transpose = np.linalg.svd(cross_covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1, :] *= -1
        rotation = right_transpose.T @ left.T
    return rotation


def _mechanical_fit(spec: AlignmentSpec, max_residual_mm: float) -> dict[str, Any]:
    diameter = spec.fixed.clearance_hole_diameter_mm
    fastener = spec.fastener_diameter_mm
    uncertainties = (
        spec.fixed.evidence.uncertainty_mm,
        spec.moving.evidence.uncertainty_mm,
    )
    missing = []
    if diameter is None:
        missing.append("fixed.clearance_hole_diameter_mm")
    if fastener is None:
        missing.append("fastener_diameter_mm")
    if any(value is None for value in uncertainties):
        missing.append("both evidence.uncertainty_mm values")
    if missing:
        return {
            "status": "not_evaluated",
            "reason": "missing " + ", ".join(missing),
        }
    assert diameter is not None
    assert fastener is not None
    uncertainty = float(sum(value for value in uncertainties if value is not None))
    radial_margin = (
        (diameter - fastener) / 2
        - max_residual_mm
        - uncertainty
        - spec.assembly_margin_mm
    )
    return {
        "status": "pass" if radial_margin >= 0 else "fail",
        "clearance_hole_diameter_mm": diameter,
        "fastener_diameter_mm": fastener,
        "conservative_measurement_uncertainty_mm": uncertainty,
        "assembly_margin_mm": spec.assembly_margin_mm,
        "remaining_radial_margin_mm": radial_margin,
        "criterion": (
            "(clearance_diameter-fastener_diameter)/2 - max_center_error "
            "- summed_uncertainty - assembly_margin >= 0"
        ),
    }


def _limitations(spec: AlignmentSpec, pose_status: str) -> list[str]:
    limitations = []
    if "mesh" in {spec.fixed.evidence.kind, spec.moving.evidence.kind}:
        limitations.append(
            "At least one interface is mesh-derived; use a drawing, STEP B-Rep, or physical "
            "measurement before fabrication."
        )
    if pose_status == "ambiguous":
        limitations.append(
            "The hole pattern has multiple equivalent poses; add datum_direction to both "
            "interfaces using a front, cable-exit, or base-frame direction."
        )
    if spec.fastener_diameter_mm is None:
        limitations.append("Fastener clearance has not been verified.")
    return limitations


def _incompatible_count_report(spec: AlignmentSpec) -> dict[str, Any]:
    return {
        "schema": "clothdojo_mount_alignment_report_v1",
        "fixed": _interface_report(spec.fixed),
        "moving": _interface_report(spec.moving),
        "pattern": {
            "status": "incompatible",
            "reason": "hole_count_mismatch",
            "fixed_hole_count": len(spec.fixed.holes_mm),
            "moving_hole_count": len(spec.moving.holes_mm),
        },
        "pose": {"status": "not_applicable"},
        "mechanical_fit": {"status": "fail", "reason": "hole_count_mismatch"},
        "limitations": _limitations(spec, "not_applicable"),
    }


def _interface(value: object, label: str) -> MountInterface:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}.name must be a non-empty string")
    holes = _points(value.get("holes_mm"), f"{label}.holes_mm")
    if len(holes) < 3 or np.linalg.matrix_rank(holes - holes.mean(axis=0), tol=1e-8) < 2:
        raise ValueError(f"{label}.holes_mm must contain at least 3 non-collinear points")
    normal = _unit_vector(value.get("outward_normal"), f"{label}.outward_normal")
    centered = holes - holes.mean(axis=0)
    plane_offsets = np.abs(centered @ normal)
    if float(plane_offsets.max()) > 1e-5:
        raise ValueError(f"{label}.holes_mm are not coplanar with outward_normal")
    datum = value.get("datum_direction")
    datum_direction = None if datum is None else _unit_vector(datum, f"{label}.datum_direction")
    if datum_direction is not None and abs(float(datum_direction @ normal)) > 1e-6:
        raise ValueError(f"{label}.datum_direction must lie in the mating plane")
    evidence_value = value.get("evidence")
    if not isinstance(evidence_value, dict):
        raise ValueError(f"{label}.evidence must be an object")
    kind = evidence_value.get("kind")
    if kind not in {"drawing", "brep", "mesh", "physical"}:
        raise ValueError(f"{label}.evidence.kind is invalid")
    reference = evidence_value.get("reference")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{label}.evidence.reference must be a non-empty string")
    uncertainty = _optional_nonnegative_float(
        evidence_value.get("uncertainty_mm"), f"{label}.evidence.uncertainty_mm"
    )
    return MountInterface(
        name=name,
        holes_mm=holes,
        outward_normal=normal,
        evidence=Evidence(kind=kind, reference=reference, uncertainty_mm=uncertainty),
        datum_direction=datum_direction,
        clearance_hole_diameter_mm=_optional_positive_float(
            value.get("clearance_hole_diameter_mm"),
            f"{label}.clearance_hole_diameter_mm",
        ),
    )


def _points(value: object, label: str) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an N x 3 numeric array") from exc
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{label} must be an N x 3 finite numeric array")
    return points


def _unit_vector(value: object, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a 3-vector") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{label} must be non-zero")
    return vector / length


def _interface_report(interface: MountInterface) -> dict[str, Any]:
    return {
        "name": interface.name,
        "hole_count": len(interface.holes_mm),
        "holes_mm": interface.holes_mm.tolist(),
        "outward_normal": interface.outward_normal.tolist(),
        "datum_direction": (
            None if interface.datum_direction is None else interface.datum_direction.tolist()
        ),
        "clearance_hole_diameter_mm": interface.clearance_hole_diameter_mm,
        "evidence": {
            "kind": interface.evidence.kind,
            "reference": interface.evidence.reference,
            "uncertainty_mm": interface.evidence.uncertainty_mm,
        },
    }


def _pairwise_distances(points: np.ndarray) -> list[float]:
    return sorted(
        float(np.linalg.norm(points[left] - points[right]))
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(left @ right, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _positive_float(value: object, label: str) -> float:
    result = _nonnegative_float(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative finite number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return result


def _optional_positive_float(value: object, label: str) -> float | None:
    return None if value is None else _positive_float(value, label)


def _optional_nonnegative_float(value: object, label: str) -> float | None:
    return None if value is None else _nonnegative_float(value, label)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve and audit a robot mounting-interface registration."
    )
    parser.add_argument("spec", help="clothdojo_mount_alignment_v1 JSON file")
    parser.add_argument("--output", type=Path, help="optional report JSON path")
    parser.add_argument(
        "--require-compatible",
        action="store_true",
        help="exit non-zero unless the geometric pattern is within tolerance",
    )
    parser.add_argument(
        "--require-mechanical-fit",
        action="store_true",
        help="exit non-zero unless fastener clearance is fully specified and passes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = align_mounts(load_spec(args.spec))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.require_compatible and report["pattern"]["status"] != "within_tolerance":
        raise SystemExit(1)
    if args.require_mechanical_fit and report["mechanical_fit"]["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
