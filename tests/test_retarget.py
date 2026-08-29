from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.data.retarget import (
    _ALOHA_ARM_INDICES,
    _ALOHA_GRIPPER_INDICES,
    _pregrasp_mask,
    _retarget_motion,
    _retarget_trajectory,
    _robot_velocity,
    piper_attach_poses,
    piper_attach_positions,
)
from clothdojo.data.retarget_dynamics import (
    DynamicsRetargetResult,
    _attach_approach_directions,
    _canonicalize_saturated_target,
    _direction_angle,
    _inverse_position_drive_target,
    _scale_effort_limits,
    _select_repair_window,
    _task_space_residual,
    _window_candidate_improves,
)
from clothdojo.robots import ALOHA2, PIPER, UR5E_2F85, robot_profile
from clothdojo.trajectory import (
    SceneMetadata,
    TrajectoryAction,
    TrajectoryReader,
    TrajectoryState,
    TrajectoryWriter,
)


def _attach_poses(positions: np.ndarray, rotations: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float32)
    if rotations is None:
        rotations = np.broadcast_to(
            np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
            (*values.shape[:-1], 4),
        )
    return np.concatenate((values, rotations), axis=-1).astype(np.float32)


class _CartesianKinematics:
    home = np.zeros(16, dtype=np.float32)
    arm_indices = _ALOHA_ARM_INDICES
    gripper_indices = _ALOHA_GRIPPER_INDICES
    gripper_closed_positions = np.asarray((0.0079, 0.0079), dtype=np.float32)
    gripper_open_positions = np.asarray((0.037, 0.037), dtype=np.float32)

    def solve(
        self,
        seed: np.ndarray,
        target_positions: np.ndarray,
        target_rotations: np.ndarray | None = None,
    ) -> np.ndarray:
        result = np.asarray(seed, dtype=np.float32).copy()
        result[0:3] = target_positions[0]
        result[8:11] = target_positions[1]
        if target_rotations is not None:
            result[3:6] = target_rotations[0, :3]
            result[11:14] = target_rotations[1, :3]
        return result

    def attach_positions_batch(self, joint_q: np.ndarray) -> np.ndarray:
        return np.stack((joint_q[:, 0:3], joint_q[:, 8:11]), axis=1)

    def attach_poses_batch(self, joint_q: np.ndarray) -> np.ndarray:
        positions = self.attach_positions_batch(joint_q)
        xyz = np.stack((joint_q[:, 3:6], joint_q[:, 11:14]), axis=1)
        scalar = np.sqrt(np.maximum(0.0, 1.0 - np.sum(xyz * xyz, axis=-1, keepdims=True)))
        rotations = np.concatenate((xyz, scalar), axis=-1)
        return np.concatenate((positions, rotations), axis=-1).astype(np.float32)

    def body_state_batch(self, joint_q, joint_qd):
        body_q = np.zeros((len(joint_q), 20, 7), dtype=np.float32)
        body_q[..., 6] = 1.0
        body_q[:, 6, :3] = joint_q[:, 0:3]
        body_q[:, 16, :3] = joint_q[:, 8:11]
        body_q[:, [6, 16], 3:7] = (-0.5, -0.5, -0.5, 0.5)
        return body_q, np.zeros((len(joint_q), 20, 6), dtype=np.float32)


class _OffsetCartesianKinematics(_CartesianKinematics):
    def attach_positions_batch(self, joint_q: np.ndarray) -> np.ndarray:
        positions = super().attach_positions_batch(joint_q)
        positions[..., 0] += 0.01
        return positions


class _ExactDynamics:
    control_fps = 50.0
    sim_substeps = 6

    def reconstruct(
        self,
        reference_joint_q,
        reference_joint_qd,
        reference_gripper_open,
        target_attach_positions,
        target_approach_directions,
        pregrasp_mask,
        approach_validation_mask,
        attached,
    ):
        assert np.asarray(target_approach_directions).shape == (
            len(reference_joint_q),
            2,
            3,
        )
        assert np.asarray(pregrasp_mask).shape == (len(reference_joint_q), 2)
        assert np.asarray(approach_validation_mask).shape == (len(reference_joint_q), 2)
        body_q, body_qd = _CartesianKinematics().body_state_batch(
            reference_joint_q, reference_joint_qd
        )
        errors = np.zeros((len(reference_joint_q), 2), dtype=np.float32)
        return DynamicsRetargetResult(
            joint_q=np.asarray(reference_joint_q, dtype=np.float32),
            joint_qd=np.asarray(reference_joint_qd, dtype=np.float32),
            body_q=body_q,
            body_qd=body_qd,
            gripper_open_state=np.asarray(reference_gripper_open, dtype=np.float32),
            joint_target=np.asarray(reference_joint_q[1:, _ALOHA_ARM_INDICES]),
            gripper_open_command=np.asarray(reference_gripper_open[1:]),
            attach_positions=np.asarray(target_attach_positions, dtype=np.float32),
            attach_errors=errors,
            validation={"dynamics_valid": True},
        )


_PIPER_PROFILE = robot_profile(PIPER)
_PIPER_ATTACH_BODY_INDICES = tuple(
    _PIPER_PROFILE.body_names.index(frame.parent_body)
    for frame in _PIPER_PROFILE.attach_frames
)


def _state(index: int, targets: np.ndarray) -> TrajectoryState:
    body_q = np.zeros((len(_PIPER_PROFILE.body_names), 7), dtype=np.float32)
    body_q[:, 6] = 1.0
    body_q[list(_PIPER_ATTACH_BODY_INDICES), :3] = targets - np.asarray(
        (0.0, 0.0, 0.13503)
    )
    return TrajectoryState(
        sample_index=index,
        joint_q=np.zeros(16, dtype=np.float32),
        joint_qd=np.zeros(16, dtype=np.float32),
        body_q=body_q,
        body_qd=np.zeros((len(_PIPER_PROFILE.body_names), 6), dtype=np.float32),
        particle_q=np.zeros((2, 3), dtype=np.float32),
        particle_qd=np.zeros((2, 3), dtype=np.float32),
        gripper_open_state=np.asarray((0.25 * index, 1.0 - 0.25 * index)),
        grasp_attached=np.asarray((index > 0, index > 1), dtype=np.int32),
        grasp_previous_open=np.ones(2, dtype=np.float32),
        grasp_pin_count=np.zeros(2, dtype=np.int32),
        grasp_pins=np.full((2, 2), -1, dtype=np.int32),
        grasp_pin_local=np.zeros((2, 2, 3), dtype=np.float32),
    )


def _write_source(path: Path, *, task_name: str = "shirt_flatten") -> Path:
    writer = TrajectoryWriter(
        path,
        control_fps=50.0,
        sim_substeps=6,
        scene=SceneMetadata(task_name=task_name, robot_id="piper"),
        metadata={"asset_id": "test_shirt"},
    )
    targets = np.asarray(((0.4, 0.3, 1.0), (0.4, -0.3, 1.0)), dtype=np.float32)
    writer.write_initial_state(_state(0, targets))
    for index in range(2):
        next_targets = targets + np.asarray((0.01 * (index + 1), 0.0, 0.0))
        writer.append_transition(
            TrajectoryAction(
                sample_index=index,
                joint_target=np.zeros(12, dtype=np.float32),
                gripper_open_command=np.asarray((0.2, 0.8), dtype=np.float32),
            ),
            _state(index + 1, next_targets),
        )
    writer.close()
    return path


def test_piper_attach_positions_applies_local_frame_rotation() -> None:
    body_q = np.zeros((1, len(_PIPER_PROFILE.body_names), 7), dtype=np.float32)
    body_q[..., 6] = 1.0
    half = np.sqrt(0.5)
    body_q[:, list(_PIPER_ATTACH_BODY_INDICES), 3:7] = np.asarray(
        (0.0, half, 0.0, half)
    )

    positions = piper_attach_positions(body_q)
    poses = piper_attach_poses(body_q)

    np.testing.assert_allclose(positions[..., 0], 0.13503, atol=1.0e-6)
    np.testing.assert_allclose(positions[..., 1:], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(poses[..., :3], positions)
    np.testing.assert_allclose(
        poses[..., 3:7], body_q[:, list(_PIPER_ATTACH_BODY_INDICES), 3:7]
    )


def test_robot_velocity_maps_ur5e_coordinates_around_closed_chain_dofs() -> None:
    profile = robot_profile(UR5E_2F85)
    q = np.repeat(np.asarray(profile.home_joint_q, dtype=np.float32)[None], 2, axis=0)
    q[1, list(profile.arm_coord_indices)] += 0.02
    q[1, list(profile.gripper_coord_indices)] += 0.01

    qd = _robot_velocity(q, profile, 50.0)

    assert qd.shape == (2, 40)
    np.testing.assert_allclose(qd[:, list(profile.arm_dof_indices)], 1.0, atol=2.0e-6)
    np.testing.assert_allclose(qd[:, list(profile.gripper_dof_indices)], 0.5, atol=2.0e-6)
    untouched = np.ones(40, dtype=bool)
    untouched[list(profile.arm_dof_indices)] = False
    untouched[list(profile.gripper_dof_indices)] = False
    np.testing.assert_array_equal(qd[:, untouched], 0.0)


def test_saturated_target_preserves_effort_over_complete_control_frame() -> None:
    current = np.asarray((0.1, -0.2, 0.3))
    velocity = np.asarray((0.4, -0.3, 0.2))
    following = np.asarray((0.14, -0.24, 0.31))
    following_velocity = np.asarray((0.6, -0.5, 0.1))
    stiffness = np.asarray((100.0, 100.0, 100.0))
    damping = np.asarray((10.0, 10.0, 10.0))
    effort_limit = np.asarray((5.0, 5.0, 5.0))
    original = np.asarray((0.4, -0.5, 0.35))
    required = np.asarray((8.0, -7.0, 2.0))
    dt = 0.02

    target = _canonicalize_saturated_target(
        current,
        velocity,
        following,
        following_velocity,
        original,
        required,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        lower=np.full(3, -2.0),
        upper=np.full(3, 2.0),
        dt=dt,
    )

    u = np.linspace(0.0, 1.0, 1001)[:, None]
    a0 = current
    a1 = dt * velocity
    a2 = -3.0 * current - 2.0 * a1 + 3.0 * following - dt * following_velocity
    a3 = 2.0 * current + a1 - 2.0 * following + dt * following_velocity
    position = a0 + u * (a1 + u * (a2 + u * a3))
    frame_velocity = (a1 + u * (2.0 * a2 + u * 3.0 * a3)) / dt
    effort = stiffness * (target - position) - damping * frame_velocity

    assert np.min(effort[:, 0]) >= effort_limit[0] - 1.0e-10
    assert np.max(effort[:, 1]) <= -effort_limit[1] + 1.0e-10
    assert target[2] == original[2]


def test_effort_scale_applies_to_selected_joints_in_every_world() -> None:
    import warp as wp

    model = type(
        "Model",
        (),
        {
            "world_count": 2,
            "joint_dof_count": 8,
            "joint_effort_limit": wp.array(
                [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                dtype=wp.float32,
                device="cpu",
            ),
        },
    )()

    _scale_effort_limits(model, np.asarray((0, 2)), 1.5)

    np.testing.assert_allclose(
        model.joint_effort_limit.numpy(),
        (1.5, 2.0, 4.5, 4.0, 15.0, 20.0, 45.0, 40.0),
    )


def test_inverse_position_drive_target_respects_effort_and_position_limits() -> None:
    target, required = _inverse_position_drive_target(
        np.asarray((0.1, 0.2)),
        np.asarray((0.5, -0.5)),
        np.asarray((0.4, -0.2)),
        np.asarray((1.0, -1.0)),
        np.asarray((2.0, -3.0)),
        stiffness=np.asarray((100.0, 100.0)),
        damping=np.asarray((10.0, 10.0)),
        effort_limit=np.asarray((5.0, 8.0)),
        lower=np.asarray((0.0, 0.0)),
        upper=np.asarray((0.2, 0.3)),
    )

    np.testing.assert_allclose(required, (37.0, -48.0))
    np.testing.assert_allclose(target, (0.2, 0.07))


def test_sparse_window_selects_worst_task_state_with_recovery_step() -> None:
    errors = np.zeros((8, 2), dtype=np.float32)
    errors[0, 0] = 1.0
    errors[3, 0] = 0.012
    errors[5, 1] = 0.018
    selected = _select_repair_window(
        errors,
        np.zeros_like(errors),
        np.zeros_like(errors, dtype=bool),
        action_count=7,
        horizon=3,
        attempted=set(),
    )

    assert selected == (3, 6, 5)


def test_sparse_window_uses_terminal_actions_and_skips_attempted_window() -> None:
    errors = np.zeros((7, 2), dtype=np.float32)
    errors[[4, 6], 0] = (0.012, 0.02)
    selected = _select_repair_window(
        errors,
        np.zeros_like(errors),
        np.zeros_like(errors, dtype=bool),
        action_count=6,
        horizon=3,
        attempted={(3, 6)},
    )

    assert selected == (2, 5, 4)


def test_sparse_window_selects_worst_normalized_arm_gate() -> None:
    errors = np.zeros((5, 2), dtype=np.float32)
    errors[2, 0] = 0.011
    approach_errors = np.zeros((5, 2), dtype=np.float32)
    approach_errors[3, 1] = 0.08
    pregrasp = np.zeros((5, 2), dtype=bool)
    pregrasp[3, 1] = True
    selected = _select_repair_window(
        errors,
        approach_errors,
        pregrasp,
        action_count=4,
        horizon=3,
        attempted=set(),
    )

    assert selected == (1, 4, 3)


def test_sparse_window_acceptance_reduces_worst_gate_without_breaking_passed_gate() -> None:
    assert _window_candidate_improves((0.012, 0.04), (0.009, 0.05))
    assert _window_candidate_improves((0.009, 0.06), (0.009, 0.04))
    assert not _window_candidate_improves((0.012, 0.04), (0.011, 0.051))
    assert not _window_candidate_improves((0.016, 0.10), (0.05, 0.04))


def test_sparse_window_uses_only_active_pregrasp_approach_gate() -> None:
    approach_errors = np.zeros((6, 2), dtype=np.float32)
    approach_errors[3, 0] = 0.08
    approach_errors[4, 1] = 1.0
    pregrasp = np.zeros((6, 2), dtype=bool)
    pregrasp[3, 0] = True

    selected = _select_repair_window(
        np.zeros_like(approach_errors),
        approach_errors,
        pregrasp,
        action_count=5,
        horizon=3,
        attempted=set(),
    )

    assert selected == (1, 4, 3)


def test_task_residual_tracks_attach_approach_and_gripper_only() -> None:
    positions = np.asarray(((0.01, 0.0, 0.0), (0.0, 0.0, 0.0)))
    openness = np.asarray((0.1, 0.0))
    approach = np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    desired_approach = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))

    residual = _task_space_residual(
        positions,
        openness,
        approach,
        np.zeros((2, 3)),
        np.zeros(2),
        desired_approach,
        np.asarray((True, False)),
    )

    np.testing.assert_allclose(residual[:6], (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert np.linalg.norm(residual[6:9]) > 0.0
    np.testing.assert_array_equal(residual[9:12], 0.0)
    np.testing.assert_allclose(residual[12:14], (1.0, 0.0))
    np.testing.assert_allclose(
        _direction_angle(
            np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))),
            np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ),
        (0.0, np.pi),
    )


def test_aloha_attach_approach_uses_local_attach_rotation() -> None:
    profile = robot_profile(ALOHA2)
    body_q = np.zeros((1, len(profile.body_names), 7), dtype=np.float32)
    body_q[..., 6] = 1.0
    body_indices = tuple(
        profile.body_names.index(frame.parent_body) for frame in profile.attach_frames
    )

    directions = _attach_approach_directions(body_q, profile.attach_frames, body_indices)

    np.testing.assert_allclose(
        directions,
        np.asarray(((1.0, 0.0, 0.0),) * 2)[None],
        atol=1.0e-7,
    )


def test_motion_solves_each_frame_and_maps_normalized_gripper_opening() -> None:
    targets = np.asarray(
        [
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.5, 0.0, 0.0), (0.5, 1.0, 0.0)),
            ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        ],
        dtype=np.float32,
    )
    grippers = np.asarray(((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)), dtype=np.float32)

    motion = _retarget_motion(
        _attach_poses(targets),
        grippers,
        np.zeros((3, 2), dtype=np.int32),
        kinematics=_CartesianKinematics(),
        position_tolerance_m=1.0e-6,
    )

    np.testing.assert_allclose(motion.target_attach_positions, targets)
    np.testing.assert_allclose(motion.attach_errors, 0.0)
    np.testing.assert_allclose(motion.target_gripper_open, grippers)
    expected = 0.0079 + grippers[:, :, None] * (0.037 - 0.0079)
    np.testing.assert_allclose(
        motion.joint_q[:, (6, 7, 14, 15)],
        expected.repeat(2, axis=2).reshape(3, 4),
    )


def test_pregrasp_motion_matches_source_approach_axis_without_full_pose_mode() -> None:
    kinematics = _CartesianKinematics()
    targets = np.asarray(
        [
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.1, 0.0, 0.0), (0.1, 1.0, 0.0)),
            ((0.2, 0.0, 0.0), (0.2, 1.0, 0.0)),
        ],
        dtype=np.float32,
    )

    angle = np.deg2rad(30.0)
    rotations = np.broadcast_to(
        np.asarray((0.0, np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0)), dtype=np.float32),
        (3, 2, 4),
    ).copy()
    motion = _retarget_motion(
        _attach_poses(targets, rotations),
        np.asarray(((1.0, 1.0), (0.5, 1.0), (0.0, 1.0)), dtype=np.float32),
        np.asarray(((0, 1), (0, 1), (1, 1)), dtype=np.int32),
        kinematics=kinematics,
        position_tolerance_m=1.0e-6,
    )

    np.testing.assert_array_equal(
        motion.pregrasp_mask,
        ((True, False), (True, False), (False, False)),
    )
    np.testing.assert_allclose(motion.approach_errors[1, 0], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(motion.attach_errors, 0.0, atol=1.0e-6)


def test_pregrasp_mask_adds_event_derived_lead_before_closing() -> None:
    openness = np.asarray(((1.0, 1.0), (0.8, 0.7), (0.6, 0.5)), dtype=np.float32)
    attached = np.asarray(((0, 0), (0, 1), (1, 1)), dtype=np.int32)

    np.testing.assert_array_equal(
        _pregrasp_mask(openness, attached),
        ((True, False), (True, False), (False, False)),
    )

    released_without_future_attach = np.asarray(((1, 1), (0, 1), (0, 1)), dtype=np.int32)
    np.testing.assert_array_equal(
        _pregrasp_mask(openness, released_without_future_attach),
        np.zeros((3, 2), dtype=bool),
    )


def test_derived_canonical_trajectory_references_source_cloth(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    output = tmp_path / "retarget"

    manifest = _retarget_trajectory(
        source,
        output,
        kinematics=_CartesianKinematics(),
        dynamics=_ExactDynamics(),
        position_tolerance_m=1.0e-6,
    )

    retarget = manifest["metadata"]["retarget"]
    assert retarget["kind"] == "dynamics_compensated_counterfactual_replay"
    assert retarget["physical_rollout"] is False
    assert retarget["robot_dynamics_rollout"] is True
    assert retarget["validation"]["kinematic_valid"] is True
    assert retarget["validation"]["dynamics_valid"] is True
    assert retarget["validation"]["training_eligible"] is True
    assert retarget["validation"]["source_timeline_preserved"] is True
    assert retarget["validation"]["source_state_count"] == 3
    assert retarget["validation"]["output_state_count"] == 3
    assert manifest["scene"]["robot_id"] == "aloha2"
    assert json.loads((output / "manifest.json").read_text())["metadata"]["storage"][
        "base_trajectory"
    ] == str(source)
    with np.load(output / "states.npz", allow_pickle=False) as arrays:
        assert "particle_q" not in arrays.files
        assert arrays["grasp_attached"].dtype == np.int32
    with np.load(output / "retarget_metrics.npz", allow_pickle=False) as metrics:
        np.testing.assert_array_equal(
            metrics["reference_joint_q"],
            np.stack(
                [reader_state.joint_q for reader_state in TrajectoryReader(output).iter_states()]
            ),
        )
        assert metrics["reference_joint_qd"].shape == (3, 16)
    assert not (output / "base_sample_indices.npy").exists()
    reader = TrajectoryReader(output)
    source_reader = TrajectoryReader(source)
    np.testing.assert_array_equal(
        reader.state_arrays(("particle_q",))["particle_q"],
        source_reader.state_arrays(("particle_q",))["particle_q"],
    )
    assert reader.state_at(0).joint_q.shape == (16,)
    assert reader.state_at(0).body_q.shape == (20, 7)
    assert next(reader.iter_actions()).joint_target.shape == (12,)


def test_retarget_rejects_task_without_a_physics_scene(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source", task_name="shirt_fold")

    with pytest.raises(RuntimeError, match="source task"):
        _retarget_trajectory(
            source,
            tmp_path / "retarget",
            kinematics=_CartesianKinematics(),
            dynamics=_ExactDynamics(),
            position_tolerance_m=1.0e-6,
        )


def test_validation_compares_target_fk_with_source_piper_targets(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    output = tmp_path / "retarget"

    manifest = _retarget_trajectory(
        source,
        output,
        kinematics=_OffsetCartesianKinematics(),
        dynamics=_ExactDynamics(),
        position_tolerance_m=0.005,
    )

    validation = manifest["metadata"]["retarget"]["validation"]
    assert validation["kinematic_valid"] is False
    np.testing.assert_allclose(validation["max_attach_position_error_m"], 0.01, atol=1.0e-7)
    with np.load(output / "retarget_metrics.npz", allow_pickle=False) as metrics:
        source_positions = piper_attach_positions(
            TrajectoryReader(source).state_arrays(("body_q",))["body_q"]
        )
        np.testing.assert_allclose(metrics["source_attach_position"], source_positions)
