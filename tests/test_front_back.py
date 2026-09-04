from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo import cli
from clothdojo.data.front_back import (
    SpatialTransform,
    _cloth_relative_attach_positions,
    _coordinated_free_motion_schedule,
    _enforce_motion_envelope,
    _geometric_surface_separation,
    _ik_branch_initializations,
    _ik_naturalness_score,
    _initial_xy_bbox_center,
    _interpolate_detached_joint_motion,
    _lift_transfer_descend,
    _local_refinement_transforms,
    _maximum_arm_joint_step,
    _placement_transforms,
    _plan_free_attach_positions,
    _retreat_terminal_pose,
    _retreat_wait_approach,
    _select_smoothed_motion,
    _should_refine,
    _should_refine_kinematics,
    _subsample_joint_path,
    _trajectory_seed,
    _transform_gripper_rotations,
    _validate_front_back_dynamics,
    _validate_initial_base_clearance,
    _validate_motion_limits,
    _validate_table_bounds,
    _write_variant_overlay,
)
from clothdojo.data.retarget import _Motion
from clothdojo.robots import PIPER, robot_profile
from clothdojo.trajectory import (
    SceneMetadata,
    TrajectoryAction,
    TrajectoryReader,
    TrajectoryState,
    TrajectoryWriter,
)
from clothdojo.trajectory.reader import _DERIVED_OVERRIDE_FIELDS


def _state(index: int) -> TrajectoryState:
    particle_q = np.asarray(
        ((0.4 + 0.1 * index, -0.2, 0.8), (0.8 + 0.1 * index, 0.3, 0.8)),
        dtype=np.float32,
    )
    return TrajectoryState(
        sample_index=index,
        joint_q=np.zeros(16, dtype=np.float32),
        joint_qd=np.zeros(16, dtype=np.float32),
        body_q=np.tile(np.asarray((0, 0, 0, 0, 0, 0, 1), dtype=np.float32), (21, 1)),
        body_qd=np.zeros((21, 6), dtype=np.float32),
        particle_q=particle_q,
        particle_qd=np.asarray(((1, 2, 3), (-1, -2, -3)), dtype=np.float32),
        gripper_open_state=np.ones(2, dtype=np.float32),
        grasp_attached=np.zeros(2, dtype=np.int32),
        grasp_previous_open=np.ones(2, dtype=np.float32),
        grasp_pin_count=np.zeros(2, dtype=np.int32),
        grasp_pins=np.full((2, 2), -1, dtype=np.int32),
        grasp_pin_local=np.zeros((2, 2, 3), dtype=np.float32),
    )


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    with TrajectoryWriter(
        source,
        control_fps=50,
        sim_substeps=6,
        max_actions=2,
        scene=SceneMetadata(task_name="shirt_fold", robot_id="piper"),
        metadata={"asset_id": "shirt"},
    ) as writer:
        writer.write_initial_state(_state(0))
        writer.append_transition(
            TrajectoryAction(
                sample_index=0,
                joint_target=np.zeros(12, dtype=np.float32),
                gripper_open_command=np.ones(2, dtype=np.float32),
            ),
            _state(1),
        )
    return source


def _overrides(reader: TrajectoryReader) -> dict[str, np.ndarray]:
    return {
        "sample_index": np.arange(reader.manifest.state_count, dtype=np.int64),
        **reader.state_arrays(_DERIVED_OVERRIDE_FIELDS),
    }


def test_spatial_overlay_composes_reflections_without_copying_particles(tmp_path: Path) -> None:
    source = _source(tmp_path)
    reader = TrajectoryReader(source)
    reflect_x = SpatialTransform(
        np.diag((-1, 1, 1)).astype(np.float32),
        np.asarray((1.4, 0.1, 0), dtype=np.float32),
    )
    first = tmp_path / "front_back"
    first_manifest = _write_variant_overlay(
        reader.manifest,
        first,
        base=source,
        states=_overrides(reader),
        actions=reader.action_arrays(),
        transform=reflect_x,
        augmentation={"kind": "front_back", "surface_winding_reversed": True},
    )
    reflect_y = SpatialTransform(
        np.diag((1, -1, 1)).astype(np.float32),
        np.zeros(3, dtype=np.float32),
    )
    second = tmp_path / "composed"
    _write_variant_overlay(
        first_manifest,
        second,
        base=first,
        states=_overrides(TrajectoryReader(first)),
        actions=TrajectoryReader(first).action_arrays(),
        transform=reflect_y,
        augmentation={"kind": "composed", "surface_winding_reversed": False},
    )

    assert set(np.load(first / "states.npz").files) == {
        "sample_index",
        *_DERIVED_OVERRIDE_FIELDS,
    }
    source_arrays = reader.state_arrays(("particle_q", "particle_qd"))
    first_arrays = TrajectoryReader(first).state_arrays(("particle_q", "particle_qd"))
    second_arrays = TrajectoryReader(second).state_arrays(("particle_q", "particle_qd"))
    np.testing.assert_allclose(
        first_arrays["particle_q"], reflect_x.points(source_arrays["particle_q"])
    )
    np.testing.assert_allclose(
        first_arrays["particle_qd"], reflect_x.vectors(source_arrays["particle_qd"])
    )
    np.testing.assert_allclose(
        second_arrays["particle_q"], reflect_y.points(first_arrays["particle_q"])
    )
    np.testing.assert_allclose(
        second_arrays["particle_qd"], reflect_y.vectors(first_arrays["particle_qd"])
    )


def test_composed_overlay_survives_atomic_parent_rename(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source_reader = TrajectoryReader(source)
    staging = tmp_path / ".collection.tmp"
    first = staging / "front_back"
    first_manifest = _write_variant_overlay(
        source_reader.manifest,
        first,
        base=source,
        states=_overrides(source_reader),
        actions=source_reader.action_arrays(),
        transform=SpatialTransform(
            np.diag((-1, 1, 1)).astype(np.float32),
            np.asarray((1.4, 0.0, 0.0), dtype=np.float32),
        ),
        augmentation={"kind": "front_back", "surface_winding_reversed": True},
    )
    composed = staging / "composed"
    _write_variant_overlay(
        first_manifest,
        composed,
        base=first,
        states=_overrides(TrajectoryReader(first)),
        actions=TrajectoryReader(first).action_arrays(),
        transform=SpatialTransform(
            np.diag((1, -1, 1)).astype(np.float32),
            np.zeros(3, dtype=np.float32),
        ),
        augmentation={"kind": "composed", "surface_winding_reversed": False},
    )
    storage = json.loads((composed / "manifest.json").read_text(encoding="utf-8"))["metadata"][
        "storage"
    ]
    assert storage["base_trajectory"] == os.path.join("..", "front_back")

    published = tmp_path / "collection"
    os.replace(staging, published)

    reader = TrajectoryReader(published / "composed")
    assert reader.manifest.state_count == source_reader.manifest.state_count
    assert reader.state_arrays(("particle_q",))["particle_q"].shape == (2, 2, 3)


def test_reader_rejects_non_orthogonal_overlay_transform(tmp_path: Path) -> None:
    source = _source(tmp_path)
    reader = TrajectoryReader(source)
    output = tmp_path / "bad"
    _write_variant_overlay(
        reader.manifest,
        output,
        base=source,
        states=_overrides(reader),
        actions=reader.action_arrays(),
        transform=None,
        augmentation={"kind": "identity"},
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["storage"]["base_spatial_transform"] = {
        "linear": [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        "translation": [0, 0, 0],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not orthogonal"):
        TrajectoryReader(output)


def test_reader_accepts_exact_legacy_episode_v1_without_rewriting_source(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    manifest_path = source / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    original.pop("format_version")
    original["schema_version"] = "newton_cloth_episode"
    original["state_profile"] = "full"
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    before = manifest_path.read_bytes()

    reader = TrajectoryReader(source)

    assert reader.manifest.format_version == 1
    assert manifest_path.read_bytes() == before


def test_reader_still_rejects_unversioned_non_episode_manifest(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manifest_path = source / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("format_version")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="format version"):
        TrajectoryReader(source)


def test_seeded_stratified_placements_are_stable_and_inside_table() -> None:
    particles = np.asarray(
        [
            [[0.4, -0.2, 0.8], [0.8, 0.3, 0.8]],
            [[0.45, -0.15, 0.8], [0.75, 0.25, 0.9]],
        ],
        dtype=np.float32,
    )
    center = _initial_xy_bbox_center(particles[0])
    table = {"center_xy": [0.65, 0.0], "half_extents_xy": [0.63, 0.7]}
    first, contract = _placement_transforms(
        particles,
        center,
        table,
        seed=123,
        count=9,
        margin_m=0.01,
    )
    repeated, _ = _placement_transforms(
        particles,
        center,
        table,
        seed=123,
        count=9,
        margin_m=0.01,
    )

    assert contract["candidate_count"] == len(first)
    assert len(first) == 9
    assert contract["max_placement_shift_xy_m"] == pytest.approx([0.15, 0.05])
    base_translation = np.asarray((2.0 * center[0], 0.0), dtype=np.float32)
    np.testing.assert_allclose(first[0].translation[:2], base_translation, atol=1.0e-7)
    for left, right in zip(first, repeated, strict=True):
        np.testing.assert_array_equal(left.linear, right.linear)
        np.testing.assert_array_equal(left.translation, right.translation)
        delta = left.translation[:2] - base_translation
        assert abs(delta[0]) <= 0.15 + 1.0e-7
        assert abs(delta[1]) <= 0.05 + 1.0e-7
        assert _validate_table_bounds(left.points(particles), table, margin_m=0.01)["valid"]


def test_single_requested_placement_does_not_add_fallback() -> None:
    particles = np.asarray(
        [[[0.4, -0.2, 0.8], [0.8, 0.3, 0.8]]],
        dtype=np.float32,
    )
    transforms, contract = _placement_transforms(
        particles,
        _initial_xy_bbox_center(particles[0]),
        {"center_xy": [0.65, 0.0], "half_extents_xy": [0.63, 0.7]},
        seed=1,
        count=1,
        margin_m=0.01,
    )

    assert len(transforms) == contract["candidate_count"] == 1


def test_initial_base_clearance_rejects_a_side_specific_near_miss() -> None:
    particles = np.asarray(
        ((0.25, -0.291, 0.8), (0.7, 0.0, 0.9)),
        dtype=np.float32,
    )
    bases = np.asarray(((0.25, 0.3, 0.8), (0.25, -0.3, 0.8)), dtype=np.float32)

    validation = _validate_initial_base_clearance(
        particles,
        bases,
        min_clearance_m=0.10,
    )

    assert not validation["valid"]
    assert validation["reason"] == "cloth_too_close_to_base"
    assert validation["worst_base"] == "right/base_link"
    assert validation["minimum_clearance_m"] == pytest.approx(0.009)


def test_free_motion_planner_overwrites_only_detached_targets() -> None:
    targets = np.zeros((9, 2, 3), dtype=np.float32)
    targets[:, 0, 0] = np.linspace(0.4, 0.8, 9)
    targets[:, 0, 2] = 0.8
    targets[:, 1] = np.asarray((0.6, -0.2, 0.85), dtype=np.float32)
    attached = np.ones((9, 2), dtype=np.int32)
    attached[1:8, 0] = 0

    planned, contract = _plan_free_attach_positions(
        targets,
        attached,
        table_top_z=0.8,
        clearance_m=0.08,
    )

    np.testing.assert_array_equal(planned[attached != 0], targets[attached != 0])
    np.testing.assert_array_equal(planned[:, 1], targets[:, 1])
    np.testing.assert_allclose(planned[4, 0], (0.6, 0.0, 0.88), atol=1.0e-7)
    assert np.all(planned[1:8, 0, 2] >= 0.8)
    assert not np.array_equal(planned[1:8, 0], targets[1:8, 0])
    assert contract["fixed_timeline"]
    assert contract["attached_targets_preserved"]
    assert contract["overwritten_state_count"] == [7, 0]
    assert len(contract["segments"]) == 1
    segment = contract["segments"][0]
    assert segment == {
        "side": "left",
        "free_start_frame": 1,
        "free_end_frame": 7,
        "anchor_start_frame": 0,
        "anchor_end_frame": 8,
        "available_state_frames": 9,
        "terminal_hold": False,
        "near_simultaneous_terminal_release": False,
        "terminal_after_other_release": False,
        "crosses_center": False,
        "near_other_attached_path": False,
        "nearest_other_attached_path_m": pytest.approx(0.2828427),
        "route": "local_transfer",
        "retreat_end_fraction": pytest.approx(0.32),
        "hold_end_fraction": pytest.approx(0.68),
        "yield_end_fraction": None,
        "yield_resume_fraction": None,
        "clearance_z_m": pytest.approx(0.88),
        "standby_planar_adjustment_m": 0.0,
        "path_length_m": pytest.approx(0.55759, abs=1.0e-4),
    }
    assert contract["schema"] == "bounded_local_free_attach_planner_v4"


def test_local_transfer_matches_nonzero_boundary_motion() -> None:
    start = np.asarray((0.70, 0.10, 0.82), dtype=np.float32)
    end = np.asarray((0.76, -0.08, 0.84), dtype=np.float32)
    start_step = np.asarray((0.002, -0.001, 0.001), dtype=np.float32)
    end_step = np.asarray((-0.001, 0.002, -0.001), dtype=np.float32)

    route = _lift_transfer_descend(
        start,
        end,
        101,
        table_top_z=0.8,
        clearance_m=0.08,
        start_step=start_step,
        end_step=end_step,
    )

    np.testing.assert_allclose(route[1] - route[0], start_step, atol=2.0e-4)
    np.testing.assert_allclose(route[-1] - route[-2], end_step, atol=2.0e-4)


def test_overlapping_release_staggers_retreat_before_approach() -> None:
    schedules = _coordinated_free_motion_schedule(
        [[(516, 701)], [(643, 872)]],
        933,
    )

    assert schedules[(0, 516, 701)]["hold_end_fraction"] == pytest.approx(142 / 187)
    assert schedules[(1, 643, 872)]["retreat_end_fraction"] == pytest.approx(0.32)
    assert schedules[(1, 643, 872)]["yield_end_fraction"] == pytest.approx(33 / 231)
    assert schedules[(1, 643, 872)]["yield_resume_fraction"] == pytest.approx(38 / 231)


def test_terminal_release_delays_overlapping_regrasp_until_retreat_begins() -> None:
    schedules = _coordinated_free_motion_schedule(
        [[(556, 914)], [(859, 1244)]],
        1245,
    )

    assert schedules[(0, 556, 914)]["hold_end_fraction"] == pytest.approx(310 / 360)
    assert schedules[(1, 859, 1244)]["yield_end_fraction"] == pytest.approx(0.1)


def test_free_motion_staging_lanes_separate_left_and_right_holds() -> None:
    start = np.asarray((0.65, 0.0, 0.82), dtype=np.float32)
    end = np.asarray((0.7, 0.0, 0.82), dtype=np.float32)
    left = _retreat_wait_approach(
        start,
        end,
        101,
        side=0,
        table_top_z=0.8,
        clearance_m=0.08,
        standby_position=np.asarray((0.55, 0.2, 0.9), dtype=np.float32),
    )
    right = _retreat_wait_approach(
        start,
        end,
        101,
        side=1,
        table_top_z=0.8,
        clearance_m=0.08,
        standby_position=np.asarray((0.55, -0.2, 0.9), dtype=np.float32),
    )

    np.testing.assert_allclose(
        left[32:69], np.tile((0.55, 0.2, 0.88), (37, 1)), atol=1.0e-7
    )
    np.testing.assert_allclose(
        right[32:69], np.tile((0.55, -0.2, 0.88), (37, 1)), atol=1.0e-7
    )
    np.testing.assert_array_equal(left[[0, -1]], np.stack((start, end)))
    np.testing.assert_array_equal(right[[0, -1]], np.stack((start, end)))


def test_yield_phase_preserves_release_x_before_staging_transfer() -> None:
    start = np.asarray((0.81, 0.18, 0.91), dtype=np.float32)
    end = np.asarray((0.72, 0.12, 0.84), dtype=np.float32)

    route = _retreat_wait_approach(
        start,
        end,
        101,
        side=1,
        table_top_z=0.8,
        clearance_m=0.08,
        retreat_end_fraction=0.32,
        hold_end_fraction=0.76,
        yield_end_fraction=0.15,
        yield_resume_fraction=0.30,
        planar_adjustment_m=0.05,
    )

    np.testing.assert_allclose(route[15:31, 0], start[0], atol=1.0e-7)
    np.testing.assert_allclose(route[15:31, 1], 0.13, atol=1.0e-7)


def test_reflected_gripper_frame_preserves_approach_and_up_axes() -> None:
    rotations = np.zeros((3, 2, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    reflection = np.diag((-1.0, 1.0, 1.0)).astype(np.float32)

    transformed = _transform_gripper_rotations(rotations, reflection)

    expected = np.diag((1.0, 1.0, 1.0))
    for quaternion in transformed.reshape(-1, 4):
        x, y, z, w = quaternion
        matrix = np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            )
        )
        np.testing.assert_allclose(matrix, expected, atol=1.0e-7)


def test_cloth_relative_attach_correction_is_smoothed_and_bounded() -> None:
    frames = 7
    targets = np.zeros((frames, 2, 3), dtype=np.float32)
    particles = np.zeros((frames, 2, 3), dtype=np.float32)
    particles[:, 0, 0] = np.linspace(0.0, 0.02, frames)
    attached = np.zeros((frames, 2), dtype=np.int32)
    attached[:, 0] = 1
    counts = np.zeros((frames, 2), dtype=np.int32)
    counts[:, 0] = 1
    pins = np.full((frames, 2, 1), -1, dtype=np.int32)
    pins[:, 0, 0] = 0
    local = np.zeros((frames, 2, 1, 3), dtype=np.float32)
    rotations = np.zeros((frames, 2, 4), dtype=np.float32)
    rotations[..., 3] = 1.0

    result, contract = _cloth_relative_attach_positions(
        targets,
        particles,
        attached,
        counts,
        pins,
        local,
        rotations,
        max_correction_m=0.005,
        smoothing_frames=3,
    )

    assert np.max(np.linalg.norm(result[:, 0], axis=-1)) <= 0.005 + 1.0e-8
    np.testing.assert_array_equal(result[[0, -1], 0], 0.0)
    np.testing.assert_array_equal(result[:, 1], targets[:, 1])
    assert contract["corrected_state_count"] == frames
    assert contract["max_applied_correction_m"] <= 0.005 + 1.0e-8
    assert contract["boundary_taper_frames"] == 3


def test_terminal_free_motion_retreats_to_own_bounded_standby_lane() -> None:
    start = np.asarray((0.8, 0.1, 0.82), dtype=np.float32)
    home = np.asarray((0.45, 0.35, 0.9), dtype=np.float32)

    route = _retreat_terminal_pose(
        start,
        21,
        side=0,
        table_top_z=0.8,
        clearance_m=0.08,
        standby_position=home,
        planar_adjustment_m=0.25,
    )

    direction = home[:2] - start[:2]
    expected_xy = start[:2] + direction * (0.25 / np.linalg.norm(direction))
    np.testing.assert_array_equal(route[0], start)
    np.testing.assert_allclose(route[-1, :2], expected_xy, atol=1.0e-7)
    assert route[-1, 1] > start[1]
    assert route[-1, 2] == pytest.approx(0.88)
    np.testing.assert_allclose(route[9:], np.tile(route[-1], (12, 1)), atol=1.0e-7)


def test_long_terminal_retreat_skips_zero_lift_and_has_fixed_horizon() -> None:
    start = np.asarray((0.8, 0.2, 0.92), dtype=np.float32)

    route = _retreat_terminal_pose(
        start,
        201,
        side=1,
        table_top_z=0.8,
        clearance_m=0.08,
        standby_position=np.asarray((0.45, -0.3, 0.9), dtype=np.float32),
    )

    assert route[1, 1] < route[0, 1]
    np.testing.assert_allclose(route[90:], np.tile(route[-1], (111, 1)), atol=1.0e-7)


def test_crossed_terminal_retreat_lifts_and_doglegs_away_from_robot_bases() -> None:
    start = np.asarray((0.72, 0.23, 0.92), dtype=np.float32)

    route = _retreat_terminal_pose(
        start,
        201,
        side=1,
        crosses_center=True,
        table_top_z=0.8,
        clearance_m=0.08,
        standby_position=np.asarray((0.49, -0.25, 1.04), dtype=np.float32),
    )

    assert np.max(route[:, 2]) == pytest.approx(1.0)
    assert route[-1, 0] > start[0]
    assert route[-1, 1] < start[1]
    assert np.linalg.norm(route[-1, :2] - start[:2]) == pytest.approx(0.25)
    np.testing.assert_allclose(route[90:], np.tile(route[-1], (111, 1)), atol=1.0e-7)


def test_free_motion_planner_uses_terminal_retreat_without_changing_release_frame() -> None:
    targets = np.zeros((21, 2, 3), dtype=np.float32)
    targets[..., 2] = 0.82
    targets[:, 0, :2] = (0.8, 0.1)
    targets[:, 1, :2] = (0.65, -0.2)
    attached = np.ones((21, 2), dtype=np.int32)
    attached[5:, 0] = 0
    home = np.asarray(((0.45, 0.35, 0.9), (0.45, -0.35, 0.9)), dtype=np.float32)

    planned, contract = _plan_free_attach_positions(
        targets,
        attached,
        table_top_z=0.8,
        clearance_m=0.08,
        home_positions=home,
    )

    np.testing.assert_array_equal(planned[4, 0], targets[4, 0])
    np.testing.assert_array_equal(planned[:, 1], targets[:, 1])
    assert planned[-1, 0, 1] > targets[-1, 0, 1]
    assert planned[-1, 0, 2] == pytest.approx(0.88)
    segment = contract["segments"][0]
    assert segment["route"] == "terminal_retreat"
    assert segment["terminal_hold"]
    assert segment["standby_planar_adjustment_m"] == pytest.approx(0.25)


def test_last_terminal_release_holds_after_earlier_arm_has_retreated() -> None:
    targets = np.zeros((21, 2, 3), dtype=np.float32)
    targets[..., 2] = 0.82
    targets[:, 0, :2] = (0.8, 0.1)
    targets[:, 1, :2] = (0.8, -0.1)
    attached = np.ones((21, 2), dtype=np.int32)
    attached[5:, 0] = 0
    attached[10:, 1] = 0
    home = np.asarray(((0.45, 0.35, 0.9), (0.45, -0.35, 0.9)), dtype=np.float32)

    planned, contract = _plan_free_attach_positions(
        targets,
        attached,
        table_top_z=0.8,
        clearance_m=0.08,
        home_positions=home,
    )

    segments = {segment["side"]: segment for segment in contract["segments"]}
    assert segments["left"]["route"] == "terminal_retreat"
    assert not segments["left"]["terminal_after_other_release"]
    assert segments["right"]["route"] == "terminal_hold_after_other_release"
    assert segments["right"]["terminal_after_other_release"]
    np.testing.assert_allclose(
        planned[10:, 1],
        np.tile(targets[9, 1], (11, 1)),
        atol=1.0e-7,
    )


def test_near_simultaneous_terminal_releases_both_hold() -> None:
    targets = np.zeros((21, 2, 3), dtype=np.float32)
    targets[..., 2] = 0.82
    targets[:, 0, :2] = (0.8, 0.1)
    targets[:, 1, :2] = (0.8, -0.1)
    attached = np.ones((21, 2), dtype=np.int32)
    attached[5:, 0] = 0
    attached[6:, 1] = 0

    planned, contract = _plan_free_attach_positions(
        targets,
        attached,
        table_top_z=0.8,
        clearance_m=0.08,
    )

    segments = contract["segments"]
    assert all(segment["near_simultaneous_terminal_release"] for segment in segments)
    assert all(
        segment["route"] == "terminal_hold_near_simultaneous_release"
        for segment in segments
    )
    np.testing.assert_array_equal(planned, targets)


def test_joint_path_subsampling_includes_midpoints_and_endpoints() -> None:
    q = np.asarray(((0.0, 2.0), (2.0, 4.0), (4.0, 8.0)), dtype=np.float32)

    sampled = _subsample_joint_path(q, 2)

    np.testing.assert_array_equal(
        sampled,
        np.asarray(((0, 2), (1, 3), (2, 4), (3, 6), (4, 8)), dtype=np.float32),
    )


def test_front_back_dynamics_uses_attached_error_and_replay_hard_limits() -> None:
    validation = {
        "dynamics_valid": False,
        "max_attach_position_error_m": 0.05,
        "max_attached_position_error_m": 0.008,
        "gripper_event_validation": {"valid": True},
        "action_replay_position_error_rad": 1.0e-6,
        "action_replay_velocity_error_rad_s": 1.0e-6,
        "action_replay_gripper_openness_error": 1.0e-6,
        "action_replay_gripper_openness_velocity_error_s": 1.0e-6,
        "limits": {
            "attach_position_error_m": 0.01,
            "action_replay_position_error_rad": 1.0e-4,
            "action_replay_velocity_error_rad_s": 1.0e-3,
            "action_replay_gripper_openness_error": 1.0e-4,
            "action_replay_gripper_openness_velocity_error_s": 1.0e-3,
        },
    }

    result = _validate_front_back_dynamics(validation)

    assert result["valid"]
    assert result["attached_targets_enforced"]
    assert result["detached_targets_are_collision_avoidance_guidance"]
    assert result["detached_motion_verified_on_realized_joint_path"]

    validation["max_attached_position_error_m"] = 0.011
    assert not _validate_front_back_dynamics(validation)["valid"]


def test_free_motion_planner_preserves_event_boundary_frames() -> None:
    targets = np.zeros((6, 2, 3), dtype=np.float32)
    targets[..., 2] = 0.8
    targets[:, 0, 0] = np.linspace(0.3, 0.8, 6)
    attached = np.zeros((6, 2), dtype=np.int32)
    attached[[0, 5], 0] = 1

    planned, _contract = _plan_free_attach_positions(
        targets,
        attached,
        table_top_z=0.8,
    )

    np.testing.assert_array_equal(planned[[0, 5], 0], targets[[0, 5], 0])
    assert np.max(planned[1:5, 0, 2]) > 0.8


def test_planner_velocity_limits_are_diagnostic_but_joint_jump_is_strict() -> None:
    profile = robot_profile(PIPER)
    arm = np.asarray(profile.arm_coord_indices, dtype=np.int64)
    joint_q = np.zeros((2, len(profile.home_joint_q)), dtype=np.float32)
    joint_qd = np.zeros((2, profile.joint_dof_count), dtype=np.float32)
    joint_qd[0] = 10.0
    joint_qd[1] = -10.0
    manifest = SimpleNamespace(
        control_fps=50.0,
        metadata={
            "scene_config": {
                "robot": {
                    "arm_joint_velocity_limit": np.ones(len(arm)).tolist(),
                    "arm_joint_acceleration_limit": np.ones(len(arm)).tolist(),
                }
            }
        },
    )
    kinematics = SimpleNamespace(
        _arm_lower=np.full(len(arm), -1.0, dtype=np.float32),
        _arm_upper=np.full(len(arm), 1.0, dtype=np.float32),
    )
    diagnostic_only = _validate_motion_limits(
        joint_q,
        joint_qd,
        manifest,
        kinematics,
        position_errors=np.zeros((2, 2), dtype=np.float32),
        position_tolerance_m=0.005,
        source_max_joint_step_rad=0.1,
    )

    assert diagnostic_only["valid"]
    assert diagnostic_only["max_velocity_ratio"] > 1.0
    assert diagnostic_only["max_acceleration_ratio"] > 1.0
    assert diagnostic_only["planner_velocity_acceleration_limits_are_diagnostic"]

    joint_q[1, arm[0]] = 0.3
    assert _maximum_arm_joint_step(joint_q) == pytest.approx(0.3)
    jumped = _validate_motion_limits(
        joint_q,
        joint_qd,
        manifest,
        kinematics,
        position_errors=np.zeros((2, 2), dtype=np.float32),
        position_tolerance_m=0.005,
        source_max_joint_step_rad=0.1,
    )

    assert not jumped["valid"]
    assert jumped["max_frame_joint_step_rad"] > jumped["joint_jump_limit_rad"]


def test_trajectory_seed_depends_on_global_seed_and_id() -> None:
    assert _trajectory_seed(7, "a") == _trajectory_seed(7, "a")
    assert _trajectory_seed(7, "a") != _trajectory_seed(8, "a")
    assert _trajectory_seed(7, "a") != _trajectory_seed(7, "b")


def test_generated_motion_cannot_exceed_source_speed_envelope() -> None:
    validation = {
        "valid": True,
        "reason": "accepted",
        "max_velocity_ratio": 2.1,
        "max_acceleration_ratio": 12.0,
    }
    source = {"max_velocity_ratio": 1.8, "max_acceleration_ratio": 18.0}

    rejected = _enforce_motion_envelope(validation, source)

    assert not rejected["valid"]
    assert rejected["reason"] == "motion_envelope"
    assert not rejected["velocity_envelope_valid"]
    assert rejected["acceleration_envelope_valid"]
    assert rejected["relative_tolerance"] == pytest.approx(0.06)
    assert rejected["allowed_max_velocity_ratio"] == pytest.approx(1.8 * 1.06)

    accepted = _enforce_motion_envelope(
        {**validation, "max_velocity_ratio": 1.5},
        source,
    )
    assert accepted["valid"]


def test_near_collision_gets_bounded_deterministic_xy_refinements() -> None:
    coarse = SpatialTransform(
        np.diag((-1, 1, 1)).astype(np.float32),
        np.asarray((1.15, -0.32, 0.0), dtype=np.float32),
    )
    contract = {
        "translation_lower_xy": [-0.35, -0.39],
        "translation_upper_xy": [0.25, 0.39],
    }
    refinements = _local_refinement_transforms(
        coarse,
        np.asarray((0.7, 0.0), dtype=np.float32),
        contract,
    )

    assert len(refinements) == 4
    np.testing.assert_allclose(
        refinements[0].translation - coarse.translation,
        np.asarray((0.0, -0.005, 0.0), dtype=np.float32),
        atol=2.0e-8,
    )
    assert _should_refine({"worst_collision": {"penetration_m": 0.002}})
    assert not _should_refine({"worst_collision": {"penetration_m": 0.00201}})


def test_near_ik_error_or_jump_gets_bounded_refinement() -> None:
    near = {
        "max_attach_position_error_m": 0.0064,
        "max_frame_joint_step_rad": 0.14,
        "joint_jump_limit_rad": 0.25,
        "minimum_joint_limit_margin_rad": 0.0,
    }

    assert _should_refine_kinematics(near, position_tolerance_m=0.005)
    assert not _should_refine_kinematics(
        {**near, "max_attach_position_error_m": 0.0101},
        position_tolerance_m=0.005,
    )
    assert _should_refine_kinematics(
        {**near, "max_frame_joint_step_rad": 0.26},
        position_tolerance_m=0.005,
    )
    assert _should_refine_kinematics(
        {**near, "max_frame_joint_step_rad": 0.50},
        position_tolerance_m=0.005,
    )
    assert _should_refine_kinematics(
        {
            **near,
            "max_frame_joint_step_rad": 0.50,
            "source_envelope_enforced": False,
            "velocity_envelope_valid": False,
            "max_velocity_ratio": 20.0,
            "allowed_max_velocity_ratio": 1.0,
        },
        position_tolerance_m=0.005,
    )
    assert _should_refine_kinematics(
        {
            **near,
            "max_attach_position_error_m": 0.001,
            "velocity_envelope_valid": False,
            "max_velocity_ratio": 1.10,
            "allowed_max_velocity_ratio": 1.0,
        },
        position_tolerance_m=0.005,
    )
    assert not _should_refine_kinematics(
        {
            **near,
            "max_attach_position_error_m": 0.001,
            "velocity_envelope_valid": False,
            "max_velocity_ratio": 1.16,
            "allowed_max_velocity_ratio": 1.0,
        },
        position_tolerance_m=0.005,
    )


def test_reference_motion_envelope_is_diagnostic() -> None:
    profile = robot_profile(PIPER)
    arm = np.asarray(profile.arm_coord_indices, dtype=np.int64)

    class CartesianJointKinematics:
        _arm_lower = np.full(12, -10.0, dtype=np.float32)
        _arm_upper = np.full(12, 10.0, dtype=np.float32)

        def attach_poses_batch(self, joint_q):
            positions = np.zeros((len(joint_q), 2, 3), dtype=np.float32)
            positions[:, 0, 0] = joint_q[:, arm[0]]
            positions[:, 1, 0] = joint_q[:, arm[6]]
            rotations = np.zeros((len(joint_q), 2, 4), dtype=np.float32)
            rotations[..., 3] = 1.0
            return np.concatenate((positions, rotations), axis=-1)

    joint_q = np.zeros((9, 16), dtype=np.float32)
    joint_q[:, arm] = np.linspace(0.0, 0.008, 9, dtype=np.float32)[:, None]
    joint_q[1:-1:2, arm] += 0.001
    kinematics = CartesianJointKinematics()
    attach_poses = kinematics.attach_poses_batch(joint_q)
    motion = _Motion(
        joint_q=joint_q,
        target_gripper_open=np.ones((9, 2), dtype=np.float32),
        target_attach_positions=attach_poses[..., :3],
        target_approach_directions=np.zeros((9, 2, 3), dtype=np.float32),
        pregrasp_mask=np.zeros((9, 2), dtype=bool),
        approach_validation_mask=np.zeros((9, 2), dtype=bool),
        attach_poses=attach_poses,
        attach_errors=np.zeros((9, 2), dtype=np.float32),
    )
    manifest = SimpleNamespace(
        control_fps=50,
        metadata={
            "scene_config": {
                "robot": {
                    "arm_joint_velocity_limit": [1.0] * 12,
                    "arm_joint_acceleration_limit": [1.0] * 12,
                }
            }
        },
    )

    selected, validation = _select_smoothed_motion(
        motion,
        attached=np.ones((9, 2), dtype=bool),
        manifest=manifest,
        kinematics=kinematics,
        profile=profile,
        position_tolerance_m=0.005,
        source_max_joint_step_rad=0.002,
        source_motion_envelope={"max_velocity_ratio": 0.0, "max_acceleration_ratio": 0.0},
    )

    assert validation["valid"]
    assert validation["joint_smoothing_attempts"] == [
        {
            "passes": 0,
            "valid": True,
            "reason": "accepted",
            "max_attach_position_error_m": 0.0,
            "max_velocity_ratio": pytest.approx(validation["max_velocity_ratio"]),
            "max_acceleration_ratio": pytest.approx(validation["max_acceleration_ratio"]),
            "max_frame_joint_step_rad": pytest.approx(validation["max_frame_joint_step_rad"]),
        }
    ]
    assert not validation["acceleration_envelope_valid"]
    assert not validation["source_envelope_enforced"]
    assert validation["max_attach_position_error_m"] <= 0.005
    np.testing.assert_array_equal(selected.joint_q, joint_q)


def test_reference_smoothing_treats_detached_targets_and_envelope_as_diagnostics() -> None:
    profile = robot_profile(PIPER)
    arm = np.asarray(profile.arm_coord_indices, dtype=np.int64)

    class DetachedGuidanceKinematics:
        _arm_lower = np.full(12, -10.0, dtype=np.float32)
        _arm_upper = np.full(12, 10.0, dtype=np.float32)

        def attach_poses_batch(self, joint_q):
            positions = np.zeros((len(joint_q), 2, 3), dtype=np.float32)
            rotations = np.zeros((len(joint_q), 2, 4), dtype=np.float32)
            rotations[..., 3] = 1.0
            return np.concatenate((positions, rotations), axis=-1)

    joint_q = np.zeros((3, 16), dtype=np.float32)
    joint_q[1, arm] = 0.1
    attach_poses = DetachedGuidanceKinematics().attach_poses_batch(joint_q)
    motion = _Motion(
        joint_q=joint_q,
        target_gripper_open=np.ones((3, 2), dtype=np.float32),
        target_attach_positions=attach_poses[..., :3],
        target_approach_directions=np.zeros((3, 2, 3), dtype=np.float32),
        pregrasp_mask=np.zeros((3, 2), dtype=bool),
        approach_validation_mask=np.zeros((3, 2), dtype=bool),
        attach_poses=attach_poses,
        attach_errors=np.full((3, 2), 0.1, dtype=np.float32),
    )
    manifest = SimpleNamespace(
        control_fps=50,
        metadata={
            "scene_config": {
                "robot": {
                    "arm_joint_velocity_limit": [1.0] * 12,
                    "arm_joint_acceleration_limit": [1.0] * 12,
                }
            }
        },
    )

    _, validation = _select_smoothed_motion(
        motion,
        attached=np.zeros((3, 2), dtype=bool),
        manifest=manifest,
        kinematics=DetachedGuidanceKinematics(),
        profile=profile,
        position_tolerance_m=0.005,
        source_max_joint_step_rad=0.1,
        source_motion_envelope={"max_velocity_ratio": 0.0, "max_acceleration_ratio": 0.0},
    )

    assert validation["valid"]
    assert validation["max_attach_position_error_m"] == 0.0
    assert not validation["velocity_envelope_valid"]
    assert not validation["acceleration_envelope_valid"]
    assert not validation["source_envelope_enforced"]
    assert validation["source_envelope_diagnostic"]


def test_detached_joint_interpolation_preserves_attached_anchors() -> None:
    profile = robot_profile(PIPER)
    arms = np.asarray(profile.arm_coord_indices, dtype=np.int64).reshape(2, -1)

    class CartesianJointKinematics:
        _arm_lower = np.full(12, -10.0, dtype=np.float32)
        _arm_upper = np.full(12, 10.0, dtype=np.float32)

        def attach_poses_batch(self, joint_q):
            positions = np.zeros((len(joint_q), 2, 3), dtype=np.float32)
            positions[:, 0, 0] = joint_q[:, arms[0, 0]]
            positions[:, 1, 0] = joint_q[:, arms[1, 0]]
            rotations = np.zeros((len(joint_q), 2, 4), dtype=np.float32)
            rotations[..., 3] = 1.0
            return np.concatenate((positions, rotations), axis=-1)

    joint_q = np.zeros((10, len(profile.home_joint_q)), dtype=np.float32)
    joint_q[2:8, arms[0]] = np.asarray((1.0, -1.0, 1.0, -1.0, 1.0, 0.6))[:, None]
    joint_q[8:, arms[0]] = 0.6
    kinematics = CartesianJointKinematics()
    poses = kinematics.attach_poses_batch(joint_q)
    attached = np.ones((10, 2), dtype=bool)
    attached[2:8, 0] = False
    motion = _Motion(
        joint_q=joint_q,
        target_gripper_open=np.ones((10, 2), dtype=np.float32),
        target_attach_positions=poses[..., :3],
        target_approach_directions=np.zeros((10, 2, 3), dtype=np.float32),
        pregrasp_mask=np.zeros((10, 2), dtype=bool),
        approach_validation_mask=np.zeros((10, 2), dtype=bool),
        attach_poses=poses,
        attach_errors=np.zeros((10, 2), dtype=np.float32),
    )

    result = _interpolate_detached_joint_motion(
        motion,
        attached,
        kinematics=kinematics,
        profile=profile,
    )

    np.testing.assert_array_equal(
        result.joint_q[attached[:, 0]][:, arms[0]],
        joint_q[attached[:, 0]][:, arms[0]],
    )
    np.testing.assert_array_equal(result.joint_q[:, arms[1]], joint_q[:, arms[1]])
    assert _maximum_arm_joint_step(result.joint_q) < _maximum_arm_joint_step(joint_q)

    selected, validation = _select_smoothed_motion(
        motion,
        attached=attached,
        manifest=SimpleNamespace(
            control_fps=50,
            metadata={
                "scene_config": {
                    "robot": {
                        "arm_joint_velocity_limit": [1.0] * 12,
                        "arm_joint_acceleration_limit": [1.0] * 12,
                    }
                }
            },
        ),
        kinematics=kinematics,
        profile=profile,
        position_tolerance_m=0.005,
        source_max_joint_step_rad=0.1,
        source_motion_envelope={"max_velocity_ratio": 0.0, "max_acceleration_ratio": 0.0},
    )

    assert validation["valid"]
    assert validation["detached_joint_smoothing_passes"] > 0
    np.testing.assert_array_equal(
        selected.joint_q[attached[:, 0]][:, arms[0]],
        joint_q[attached[:, 0]][:, arms[0]],
    )


def test_geometric_penetration_excludes_only_artificial_shape_margin() -> None:
    physical, solver_effective = _geometric_surface_separation(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, -0.0001)),
        np.asarray((0.0, 0.0, 1.0)),
        0.0004,
        0.0004,
        0.0004,
        0.0004,
    )

    assert physical == pytest.approx(-0.0001)
    assert solver_effective == pytest.approx(-0.0009)

    physical_with_radius, _ = _geometric_surface_separation(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, 0.015)),
        np.asarray((0.0, 0.0, 1.0)),
        0.0104,
        0.0104,
        0.0004,
        0.0004,
    )
    assert physical_with_radius == pytest.approx(-0.005)


def test_ik_branch_seeds_only_adjust_colliding_side_wrist() -> None:
    profile = robot_profile(PIPER)
    branches = list(
        _ik_branch_initializations(
            profile,
            {"worst_collision": {"body_pair": ["static", "right/link7"]}},
        )
    )

    assert [branch[0]["label"] for branch in branches] == [
        "right_wrist_+1.5",
        "right_wrist_-1.5",
    ]
    home = np.asarray(profile.home_joint_q, dtype=np.float32)
    wrist = np.asarray(profile.arm_coord_indices).reshape(2, -1)[1, 3]
    for branch, seed in branches:
        expected = home.copy()
        expected[wrist] += branch["wrist_offset_rad"]
        np.testing.assert_array_equal(seed, expected)


def test_ik_naturalness_prefers_smoother_near_home_path() -> None:
    profile = robot_profile(PIPER)
    home = np.asarray(profile.home_joint_q, dtype=np.float32)
    smooth = np.tile(home, (4, 1))
    rough = smooth.copy()
    rough[1, np.asarray(profile.arm_coord_indices)[0]] += 0.2

    smooth_score = _ik_naturalness_score(smooth, profile)
    rough_score = _ik_naturalness_score(rough, profile)

    assert smooth_score["cost"] < rough_score["cost"]


def test_front_back_cli_forwards_search_contract(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.data.front_back.augment_front_back",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs) or {"ok": True},
    )

    cli.main(
        [
            "data",
            "augment-front-back",
            "--source",
            "source",
            "--output",
            "output",
            "--seed",
            "17",
            "--device",
            "cuda:2",
            "--placement-candidates",
            "25",
            "--table-margin-m",
            "0.02",
        ]
    )

    assert call == {
        "args": ("source", "output"),
        "kwargs": {
            "seed": 17,
            "assets": "assets",
            "device": "cuda:2",
            "placement_candidates": 25,
            "table_margin_m": 0.02,
            "max_placement_shift_x_m": 0.15,
            "max_placement_shift_y_m": 0.05,
            "min_base_clearance_m": 0.10,
            "min_inter_arm_clearance_m": 0.03,
        },
    }
    assert json.loads(capsys.readouterr().out) == {"ok": True}
