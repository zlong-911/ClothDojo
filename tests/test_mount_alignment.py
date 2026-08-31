from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clothdojo.assets.mount_alignment import (
    AlignmentSpec,
    Evidence,
    MountInterface,
    align_mounts,
    load_spec,
)
from clothdojo.assets.mount_features import extract_circular_section_features

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "config_name",
    ["kai0_x5_mount.json", "kai0_x5_right_mount.json"],
)
def test_kai0_x5_pattern_is_close_but_pose_and_fastener_fit_are_unresolved(
    config_name: str,
) -> None:
    report = align_mounts(load_spec(ROOT / "configs/hardware" / config_name))

    assert report["pattern"]["status"] == "within_tolerance"
    assert report["pattern"]["max_residual_mm"] == pytest.approx(0.6798, abs=0.002)
    assert report["pose"]["status"] == "ambiguous"
    assert report["pose"]["equivalent_solution_count"] == 4
    assert report["mechanical_fit"]["status"] == "not_evaluated"


def test_kai0_piper_pattern_is_rejected() -> None:
    report = align_mounts(load_spec(ROOT / "configs/hardware/kai0_piper_rejection.json"))

    assert report["pattern"]["status"] == "incompatible"
    assert report["pattern"]["max_residual_mm"] == pytest.approx(17.677, abs=0.002)
    assert report["pose"]["status"] == "not_applicable"


def test_asymmetric_pattern_and_datum_resolve_an_exact_pose() -> None:
    holes = np.asarray([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    fixed = MountInterface(
        name="fixed",
        holes_mm=holes,
        outward_normal=np.asarray([0.0, 0.0, 1.0]),
        datum_direction=np.asarray([1.0, 0.0, 0.0]),
        evidence=Evidence("physical", "fixture", 0.02),
        clearance_hole_diameter_mm=6.0,
    )
    moving = MountInterface(
        name="moving",
        holes_mm=holes.copy(),
        outward_normal=np.asarray([0.0, 0.0, -1.0]),
        datum_direction=np.asarray([1.0, 0.0, 0.0]),
        evidence=Evidence("physical", "robot", 0.02),
    )
    report = align_mounts(
        AlignmentSpec(
            fixed=fixed,
            moving=moving,
            pattern_tolerance_mm=0.05,
            fastener_diameter_mm=5.0,
            assembly_margin_mm=0.1,
        )
    )

    assert report["pose"]["status"] == "resolved"
    np.testing.assert_allclose(report["pose"]["moving_to_fixed_matrix"], np.eye(4), atol=1e-12)
    assert report["mechanical_fit"]["status"] == "pass"
    assert report["mechanical_fit"]["remaining_radial_margin_mm"] == pytest.approx(0.36)


def test_mesh_section_extractor_fits_four_circular_features(tmp_path: Path) -> None:
    import trimesh

    rings = []
    expected = [(-10.0, -8.0), (-10.0, 8.0), (10.0, -8.0), (10.0, 8.0)]
    for x, y in expected:
        ring = trimesh.creation.annulus(r_min=2.0, r_max=3.0, height=2.0, sections=48)
        ring.apply_translation([x, y, 0.0])
        rings.append(ring)
    source = tmp_path / "four-rings.stl"
    trimesh.util.concatenate(rings).export(source)

    report = extract_circular_section_features(
        source,
        plane_origin=[0.0, 0.0, 0.0],
        plane_normal=[0.0, 0.0, 1.0],
        unit_scale_mm=1.0,
        diameter_range_mm=(3.8, 4.2),
        max_radial_error_mm=0.03,
    )

    assert report["feature_count"] == 4
    centers = sorted(tuple(feature["center_mm"][:2]) for feature in report["features"])
    np.testing.assert_allclose(centers, sorted(expected), atol=1e-5)
    assert all(
        feature["diameter_mm"] == pytest.approx(4.0, abs=0.01)
        for feature in report["features"]
    )


def test_mesh_extractor_refuses_native_cad(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be processed"):
        extract_circular_section_features(
            tmp_path / "base.step",
            plane_origin=[0.0, 0.0, 0.0],
            plane_normal=[0.0, 0.0, 1.0],
            unit_scale_mm=1.0,
            diameter_range_mm=(1.0, 2.0),
            max_radial_error_mm=0.1,
        )


def test_mesh_section_preserves_scene_instances_and_transforms(tmp_path: Path) -> None:
    import trimesh

    ring = trimesh.creation.annulus(r_min=2.0, r_max=3.0, height=2.0, sections=48)
    scene = trimesh.Scene()
    scene.add_geometry(ring, geom_name="ring", node_name="left",
                       transform=trimesh.transformations.translation_matrix([-10, 0, 5]))
    scene.graph.update(frame_to="right", geometry="ring",
                       matrix=trimesh.transformations.translation_matrix([10, 0, 5]))
    source = tmp_path / "instances.glb"
    scene.export(source)

    report = extract_circular_section_features(
        source, plane_origin=[0, 0, 5], plane_normal=[0, 0, 1],
        unit_scale_mm=1.0, diameter_range_mm=(3.8, 4.2), max_radial_error_mm=0.03,
    )

    assert report["feature_count"] == 2
    np.testing.assert_allclose(
        [feature["center_mm"] for feature in report["features"]],
        [[-10, 0, 5], [10, 0, 5]], atol=1e-5,
    )
