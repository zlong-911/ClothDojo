"""Extract circular mounting-feature candidates from a mesh section.

This is a measurement aid for STL/OBJ assets, not a replacement for a vendor
drawing or STEP B-Rep.  Every output is explicitly labelled ``mesh``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def extract_circular_section_features(
    mesh_path: str | Path,
    *,
    plane_origin: Sequence[float],
    plane_normal: Sequence[float],
    unit_scale_mm: float,
    diameter_range_mm: tuple[float, float],
    max_radial_error_mm: float,
    bounds_mm: tuple[float, float, float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Fit circles to closed loops cut by one section plane.

    ``unit_scale_mm`` is explicit because STL does not encode units.  Bounds use
    mesh world coordinates after conversion to millimetres.
    """

    import trimesh

    source = Path(mesh_path).expanduser().resolve()
    if source.suffix.lower() in {".step", ".stp", ".sldprt"}:
        raise ValueError("B-Rep/native CAD must not be processed by the mesh extractor")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not math.isfinite(unit_scale_mm) or unit_scale_mm <= 0:
        raise ValueError("unit_scale_mm must be positive and finite")
    minimum_diameter, maximum_diameter = diameter_range_mm
    if not (0 < minimum_diameter <= maximum_diameter):
        raise ValueError("diameter_range_mm must be positive and ordered")
    if not math.isfinite(max_radial_error_mm) or max_radial_error_mm <= 0:
        raise ValueError("max_radial_error_mm must be positive and finite")

    origin = _vector(plane_origin, "plane_origin")
    normal = _unit_vector(plane_normal, "plane_normal")
    loaded = trimesh.load(source, force="scene", process=False)
    mesh = loaded.to_mesh()
    if not len(mesh.faces):
        raise ValueError(f"mesh contains no triangle geometry: {source}")
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        raise ValueError("section plane does not intersect the mesh")

    basis_u, basis_v = _plane_basis(normal)
    candidates = []
    for loop_index, polyline in enumerate(section.discrete):
        points = np.asarray(polyline, dtype=float)
        if len(points) < 7:
            continue
        closure_error_mm = float(np.linalg.norm(points[0] - points[-1]) * unit_scale_mm)
        if closure_error_mm > max_radial_error_mm:
            continue
        points_mm = points[:-1] * unit_scale_mm
        center_mm, radius_mm, rms_error_mm, max_error_mm = _fit_circle(
            points_mm, basis_u, basis_v
        )
        diameter_mm = 2 * radius_mm
        if not (minimum_diameter <= diameter_mm <= maximum_diameter):
            continue
        if max_error_mm > max_radial_error_mm:
            continue
        if bounds_mm is not None and not _inside_bounds(center_mm, bounds_mm):
            continue
        candidates.append(
            {
                "section_loop": loop_index,
                "center_mm": center_mm.tolist(),
                "axis": normal.tolist(),
                "diameter_mm": diameter_mm,
                "rms_radial_error_mm": rms_error_mm,
                "max_radial_error_mm": max_error_mm,
                "closure_error_mm": closure_error_mm,
                "sample_count": len(points_mm),
            }
        )
    candidates = _deduplicate(candidates, max_radial_error_mm)
    candidates.sort(key=lambda item: tuple(item["center_mm"]))
    return {
        "schema": "clothdojo_mesh_mount_features_v1",
        "evidence": {
            "kind": "mesh",
            "reference": str(source),
            "unit_scale_mm": unit_scale_mm,
            "warning": (
                "STL/OBJ is tessellated and unitless; confirm final dimensions with a drawing, "
                "STEP B-Rep, or physical measurement."
            ),
        },
        "section": {
            "plane_origin_mesh_units": origin.tolist(),
            "plane_normal": normal.tolist(),
            "diameter_range_mm": [minimum_diameter, maximum_diameter],
            "max_radial_error_mm": max_radial_error_mm,
            "bounds_mm": None if bounds_mm is None else list(bounds_mm),
        },
        "feature_count": len(candidates),
        "features": candidates,
    }


def _fit_circle(
    points_mm: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    reference = points_mm.mean(axis=0)
    relative = points_mm - reference
    coordinates = np.column_stack((relative @ basis_u, relative @ basis_v))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    system = np.column_stack((2 * x, 2 * y, np.ones(len(coordinates))))
    right_hand_side = x * x + y * y
    solution, _, rank, _ = np.linalg.lstsq(system, right_hand_side, rcond=None)
    if rank < 3:
        raise ValueError("section loop is degenerate")
    center_2d = solution[:2]
    radius_squared = solution[2] + float(center_2d @ center_2d)
    if radius_squared <= 0:
        raise ValueError("section loop has no real fitted circle")
    radius = math.sqrt(radius_squared)
    radial_errors = np.abs(np.linalg.norm(coordinates - center_2d, axis=1) - radius)
    center_3d = reference + center_2d[0] * basis_u + center_2d[1] * basis_v
    return (
        center_3d,
        radius,
        float(np.sqrt(np.mean(radial_errors**2))),
        float(radial_errors.max()),
    )


def _deduplicate(candidates: list[dict[str, Any]], tolerance_mm: float) -> list[dict[str, Any]]:
    unique = []
    for candidate in sorted(candidates, key=lambda item: item["max_radial_error_mm"]):
        center = np.asarray(candidate["center_mm"])
        diameter = float(candidate["diameter_mm"])
        duplicate = any(
            np.linalg.norm(center - np.asarray(other["center_mm"])) <= tolerance_mm
            and abs(diameter - float(other["diameter_mm"])) <= 2 * tolerance_mm
            for other in unique
        )
        if not duplicate:
            unique.append(candidate)
    return unique


def _inside_bounds(
    center: np.ndarray, bounds: tuple[float, float, float, float, float, float]
) -> bool:
    return all(
        lower <= coordinate <= upper
        for coordinate, lower, upper in zip(center, bounds[::2], bounds[1::2], strict=True)
    )


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros(3)
    reference[int(np.argmin(np.abs(normal)))] = 1.0
    basis_u = np.cross(normal, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    return basis_u, basis_v


def _vector(value: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    return vector


def _unit_vector(value: Sequence[float], label: str) -> np.ndarray:
    vector = _vector(value, label)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{label} must be non-zero")
    return vector / length


def _three_floats(value: Sequence[str]) -> tuple[float, float, float]:
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _six_floats(value: Sequence[str]) -> tuple[float, float, float, float, float, float]:
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit circular features in a mesh section.")
    parser.add_argument("mesh")
    parser.add_argument("--plane-origin", nargs=3, required=True)
    parser.add_argument("--plane-normal", nargs=3, required=True)
    parser.add_argument("--unit-scale-mm", type=float, required=True)
    parser.add_argument("--diameter-mm", nargs=2, type=float, required=True)
    parser.add_argument("--max-radial-error-mm", type=float, default=0.1)
    parser.add_argument("--bounds-mm", nargs=6)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = extract_circular_section_features(
        args.mesh,
        plane_origin=_three_floats(args.plane_origin),
        plane_normal=_three_floats(args.plane_normal),
        unit_scale_mm=args.unit_scale_mm,
        diameter_range_mm=tuple(args.diameter_mm),
        max_radial_error_mm=args.max_radial_error_mm,
        bounds_mm=None if args.bounds_mm is None else _six_floats(args.bounds_mm),
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
