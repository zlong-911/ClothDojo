from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.data.retarget import (
    _prefer_upright_roll,
    _quat_rotate,
    _retarget_motion,
    _retarget_trajectory,
    _robot_velocity,
    _upright_rotations,
    piper_attach_poses,
    piper_attach_positions,
)
from clothdojo.data.retarget_dynamics import (
    DynamicsRetargetResult,
    RobotDynamicsRetargeter,
    _advance_gripper_curve,
    _attach_approach_directions,
    _canonicalize_saturated_target,
    _direction_angle,
    _gripper_event_margin,
    _gripper_event_openness,
    _gripper_event_validation,
    _inverse_position_drive_target,
    _position_drive_search_bounds,
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
    arm_indices = np.asarray(robot_profile(ALOHA2).arm_coord_indices)
    gripper_indices = np.asarray(robot_profile(ALOHA2).gripper_coord_indices)
    gripper_closed_positions = np.asarray((0.0079, 0.0079), dtype=np.float32)
    gripper_open_positions = np.asarray((0.037, 0.037), dtype=np.float32)

    def solve(
        self,
        seed: np.ndarray,
        target_positions: np.ndarray,
        target_rotations: np.ndarray | None = None,
        *,
        align_roll: bool = False,
        orientation_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        result = np.asarray(seed, dtype=np.float32).copy()
        result[0:3] = target_positions[0]
        result[8:11] = target_positions[1]
        if target_rotations is not None:
            active = (True, True) if orientation_mask is None else orientation_mask
            for side, start in enumerate((3, 11)):
                if active[side]:
                    result[start : start + 3] = target_rotations[side, :3]
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
        *,
        target_up_directions=None,
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
            joint_target=np.asarray(reference_joint_q[1:, _CartesianKinematics.arm_indices]),
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


@pytest.mark.parametrize("robot", ("aloha2", "x5", "ur5e_2f85", "fr3_duo"))
@pytest.mark.parametrize("substeps", (3, 6))
def test_retarget_preserves_asset_drives_in_all_runtimes(monkeypatch, robot, substeps) -> None:
    from types import SimpleNamespace

    import warp as wp

    profile = robot_profile(robot)
    models = []
    originals = []
    runtime_options = []

    def build_fake_scene(*args, spec, **kwargs):
        assert kwargs["include_cloth"] is False
        count = spec.world_count
        limits = np.tile(np.arange(1, profile.joint_dof_count + 1, dtype=np.float32), count)
        fields = {
            "joint_effort_limit": limits,
            "joint_limit_lower": np.full_like(limits, -3.),
            "joint_limit_upper": np.full_like(limits, 3.),
            "joint_target_ke": limits * 10,
            "joint_target_kd": limits * 2,
        }
        model = SimpleNamespace(
            world_count=count, device=wp.get_device("cpu"),
            **{key: wp.array(value, dtype=wp.float32, device="cpu")
               for key, value in fields.items()},
        )
        models.append(model)
        originals.append(fields)
        return SimpleNamespace(model=model, world_offsets=np.zeros((count, 3)))

    monkeypatch.setattr("clothdojo.data.retarget_dynamics.build_scene", build_fake_scene)
    def fake_runtime(scene, **kwargs):
        runtime_options.append(kwargs)
        return SimpleNamespace(scene=scene, model=scene.model, control_dt=.02)

    monkeypatch.setattr("clothdojo.data.retarget_dynamics.ShirtFlattenRuntime", fake_runtime)
    retargeter = RobotDynamicsRetargeter(
        Path("unused"), garment_id="unused", control_fps=50, sim_substeps=substeps,
        device="cpu", target_robot=robot,
    )
    assert len(models) == 3
    assert len({id(model) for model in models}) == 3
    assert [model.world_count for model in models] == [1, 1, retargeter.arm_count + 3]
    assert all(options["rigid_cuda_graph"] == (substeps % 2 == 0) for options in runtime_options)
    assert all(options["use_cuda_graph"] is False for options in runtime_options)
    assert all(options["rigid_deterministic"] is False for options in runtime_options)
    for model, original in zip(models, originals, strict=True):
        for key, value in original.items():
            np.testing.assert_array_equal(getattr(model, key).numpy(), value)
    np.testing.assert_array_equal(
        retargeter.effort_limit, originals[0]["joint_effort_limit"][retargeter.arm_dof],
    )
    np.testing.assert_array_equal(
        retargeter.gripper_effort_limit,
        originals[0]["joint_effort_limit"][retargeter.gripper_dof_leaders],
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
    np.testing.assert_array_equal(residual[12:18], 0.0)
    np.testing.assert_allclose(residual[18:20], (1.0, 0.0))
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


def test_gripper_event_validation_uses_runtime_leader_and_soft_endpoints() -> None:
    attached = np.zeros((8, 2), dtype=bool)
    attached[2:5, 0] = True
    fingers = np.ones((8, 2, 2))
    # .11 already attaches and .695 already releases under ParticleGrasp's
    # margin, unlike the unadjusted .1/.7 thresholds previously used here.
    fingers[:, 0] = np.asarray([1, .2, .11, .3, .68, .695, .9, 1])[:, None]
    result = _gripper_event_validation(fingers, attached)
    assert result["valid"]
    assert result["max_frame_error"] == 0
    assert result["mismatched_frames"] == 0
    assert result["openness_source"] == "runtime_leader_with_joint_margin"
    assert result["joint_margin"] == .025
    assert result["sampling"] == "control_frame"
    assert result["sides"][0]["runtime_frames"] == [2, 5]
    # Follower motion is diagnostic, not ParticleGrasp's event signal.
    fingers[:, 0, 1] = 1
    result = _gripper_event_validation(fingers, attached)
    assert result["valid"]
    assert result["event_counts_match"]
    assert np.max(np.ptp(fingers, axis=-1)) > .8


def test_gripper_event_validation_brackets_runtime_thresholds() -> None:
    attached = np.zeros((6, 2), dtype=bool)
    attached[2:5, 0] = True
    fingers = np.ones((6, 2, 2))
    fingers[:, 0, 0] = [1, .120001, .119999, .3, .689999, .690001]
    assert _gripper_event_validation(fingers, attached)["valid"]


def test_gripper_event_openness_clips_only_the_event_signal() -> None:
    fingers = np.asarray([[-.1, 1], [.5, 0], [1.1, 0]])
    original = fingers.copy()
    np.testing.assert_allclose(_gripper_event_openness(fingers), [0, .5, 1])
    np.testing.assert_array_equal(fingers, original)


@pytest.mark.parametrize("actual_frame", [1, 3])
def test_gripper_event_validation_rejects_early_and_late_closure(actual_frame) -> None:
    attached = np.zeros((5, 2), dtype=bool)
    attached[2:, 0] = True
    fingers = np.ones((5, 2, 2))
    fingers[actual_frame:, 0] = .05
    result = _gripper_event_validation(fingers, attached)
    assert not result["valid"]
    assert result["event_counts_match"]
    assert result["max_frame_error"] == 1


def test_gripper_event_validation_rejects_extra_events_and_early_release() -> None:
    attached = np.zeros((8, 2), dtype=bool)
    attached[2:6, 0] = True
    fingers = np.ones((8, 2, 2))
    fingers[2:5, 0] = .05
    assert not _gripper_event_validation(fingers, attached)["valid"]
    fingers[5, 0] = .05
    fingers[7, 0] = .05  # Spurious re-attach after the intended release.
    result = _gripper_event_validation(fingers, attached)
    assert not result["valid"]
    assert not result["event_counts_match"]


def test_gripper_event_margin_checks_runtime_leader_before_and_after() -> None:
    fingers = np.asarray([
        [[.3, .3], [.14, .13], [.11, .09]],
        [[.3, .3], [.11, .09], [.04, .05]],  # Already closed before the event.
        [[.3, 1], [.14, 1], [.11, 1]],  # Stationary follower does not affect trigger.
        [[.3, .3], [.14, .13], [.13, .01]],  # Leader has not closed yet.
    ])
    margin = _gripper_event_margin(fingers, closing=True)
    assert margin[0] > 0
    assert margin[1] < 0
    assert margin[2] == margin[0]
    assert margin[3] < 0
    # Raw thresholds sum to .12 + .69 = .81 after the shared margin.
    opened = 0.81 - fingers
    np.testing.assert_allclose(_gripper_event_margin(opened, closing=False), margin)


def test_gripper_preview_interpolates_commands_without_extending_the_timeline() -> None:
    curve = np.asarray([1, 1, .8, .2, 0], dtype=np.float32)
    advanced = _advance_gripper_curve(curve, 1.5)
    np.testing.assert_allclose(advanced, [.9, .5, .1, 0, 0])
    np.testing.assert_allclose(_advance_gripper_curve(curve, -1.5), [1, 1, 1, .9, .5])
    np.testing.assert_array_equal(curve, np.asarray([1, 1, .8, .2, 0], dtype=np.float32))
    assert len(advanced) == len(curve)
    assert np.all((advanced >= 0) & (advanced <= 1))


@pytest.mark.parametrize("initial_lead", [0.0, 4.0])
def test_gripper_event_preview_replays_overlapping_hand_events_without_retiming(
    initial_lead,
) -> None:
    retargeter = object.__new__(RobotDynamicsRetargeter)
    retargeter.shooting_world_count = 15
    replay_commands = []

    # Independent first-order fingers expose command lag without requiring GPU
    # physics in this planner test. The actual ALOHA rollout is checked separately.
    def simulate(initial, commands):
        q = np.broadcast_to(initial.reshape(-1, 2, 2), (len(commands), 2, 2)).copy()
        frames = [q.copy()]
        for command in commands.transpose(1, 0, 2):
            q += np.asarray([.5, .55]) * (command[..., None] - q)
            frames.append(q.copy())
        return np.stack(frames, axis=1)

    def replay(q, qd, arm, grip):
        assert len(arm) == len(grip)
        replay_commands.append((arm.copy(), grip.copy()))
        fingers = simulate(q, grip[None])[0].reshape(len(grip) + 1, 4)
        return {"joint_q": fingers, "joint_qd": np.zeros_like(fingers)}

    retargeter._replay = replay
    retargeter._preview_gripper_commands = lambda state, arm, grip: simulate(state[0], grip)
    reference = np.ones((42, 2))
    attached = np.zeros((42, 2), dtype=bool)
    for side, offset in enumerate((0, 2)):
        reference[4 + offset : 13 + offset, side] = np.linspace(1, 0, 9)
        reference[13 + offset : 24 + offset, side] = 0
        reference[24 + offset : 33 + offset, side] = np.linspace(0, 1, 9)
        attached[12 + offset : 30 + offset, side] = True
    arm = np.arange(41 * 12).reshape(41, 12).astype(np.float32)
    commands = reference[1:].copy()
    for side in range(2):
        commands[:, side] = _advance_gripper_curve(commands[:, side], initial_lead)
    initial = replay(np.ones(4), np.zeros(4), arm, commands)
    assert not _gripper_event_validation(initial["joint_q"].reshape(-1, 2, 2), attached)["valid"]
    original_arm, original_commands = arm.copy(), commands.copy()
    original_states = {name: values.copy() for name, values in initial.items()}

    aligned, states, stats = retargeter._align_gripper_events(
        arm, commands, reference, attached, initial,
    )

    assert _gripper_event_validation(states["joint_q"].reshape(-1, 2, 2), attached)["valid"]
    assert len(stats["events"]) == 4
    if initial_lead == 0:
        assert all(event["command_lead_frames"] > 0 for event in stats["events"])
    else:
        assert any(event["command_lead_frames"] < 0 for event in stats["events"])
    # Finish with every action, not just the locally repaired event windows.
    np.testing.assert_array_equal(replay_commands[-1][0], arm)
    np.testing.assert_array_equal(replay_commands[-1][1], aligned)
    assert aligned.shape == commands.shape
    assert np.all((aligned >= 0) & (aligned <= 1))
    np.testing.assert_array_equal(arm, original_arm)
    np.testing.assert_array_equal(commands, original_commands)
    for name in initial:
        np.testing.assert_array_equal(initial[name], original_states[name])


def test_position_drive_search_bounds_remove_flat_saturated_tails():
    lower, upper = _position_drive_search_bounds(
        np.asarray([0.0]), np.asarray([2.0]), np.asarray([0.04]), np.asarray([2.0]),
        stiffness=np.asarray([1000.0]), damping=np.asarray([100.0]),
        effort_limit=np.asarray([10.0]), lower=np.asarray([-3.0]), upper=np.asarray([3.0]),
        dt=.02,
    )
    np.testing.assert_allclose(lower, [.19])
    np.testing.assert_allclose(upper, [.25])


@pytest.mark.parametrize("velocity", [-100.0, 100.0])
def test_position_drive_search_bounds_remain_strict_and_inside_physical_limits(velocity):
    lower, upper = _position_drive_search_bounds(
        np.asarray([0.0]), np.asarray([velocity]),
        np.asarray([.02 * velocity]), np.asarray([velocity]),
        stiffness=np.asarray([1000.0]), damping=np.asarray([100.0]),
        effort_limit=np.asarray([10.0]), lower=np.asarray([-3.0]), upper=np.asarray([3.0]),
        dt=.02,
    )
    assert -3 <= lower[0] < upper[0] <= 3


def test_gripper_position_error_triggers_shooting_without_arm_error() -> None:
    retargeter = object.__new__(RobotDynamicsRetargeter)
    residual = np.zeros(20)
    assert not retargeter._needs_arm_refinement(residual)
    assert not retargeter._needs_gripper_refinement(residual[18:])
    residual[18] = 0.2  # 0.02 normalized finger-position error.
    assert not retargeter._needs_arm_refinement(residual)
    assert retargeter._needs_gripper_refinement(residual[18:])


def test_upright_rotations_preserve_z_and_maximize_y_height() -> None:
    angles = np.asarray([.4, .8, 1.2, 1.8])
    values = np.zeros((4, 2, 4))
    values[..., 1] = np.sin(angles[:, None] / 2)
    values[..., 3] = np.cos(angles[:, None] / 2)
    original = values.copy()
    upright = _upright_rotations(values)
    z = _quat_rotate(values, np.asarray((0., 0., 1.)))
    np.testing.assert_allclose(_quat_rotate(upright, np.asarray((0., 0., 1.))), z, atol=1e-6)
    expected_y = np.asarray((0., 0., 1.)) - z[..., 2, None] * z
    expected_y /= np.linalg.norm(expected_y, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        _quat_rotate(upright, np.asarray((0., 1., 0.))), expected_y, atol=1e-6,
    )
    np.testing.assert_array_equal(values, original)


def test_upright_rotations_do_not_flip_when_z_crosses_vertical() -> None:
    angles = np.linspace(-.08, .08, 9)
    values = np.zeros((9, 2, 4))
    values[..., 1] = np.sin(angles[:, None] / 2)
    values[..., 3] = np.cos(angles[:, None] / 2)
    upright = _upright_rotations(values)
    assert np.isfinite(upright).all()
    np.testing.assert_allclose(
        _quat_rotate(upright, np.asarray((0., 1., 0.))),
        np.broadcast_to((0., 1., 0.), (9, 2, 3)), atol=1e-6,
    )
    np.testing.assert_allclose(
        _quat_rotate(upright, np.asarray((0., 0., 1.))),
        _quat_rotate(values, np.asarray((0., 0., 1.))), atol=1e-6,
    )


def test_up_axis_error_triggers_arm_refinement_without_position_or_z_error() -> None:
    retargeter = object.__new__(RobotDynamicsRetargeter)
    z = np.asarray(((1., 0., 0.), (1., 0., 0.)))
    up = np.asarray(((0., 0., 1.), (0., 0., 1.)))
    residual = _task_space_residual(
        np.zeros((2, 3)), np.ones(2), z,
        np.zeros((2, 3)), np.ones(2), z, np.ones(2, dtype=bool),
        up_directions=-up, desired_up_directions=up,
    )
    np.testing.assert_array_equal(residual[:12], 0)
    assert retargeter._needs_arm_refinement(residual)
    assert not retargeter._needs_gripper_refinement(residual[18:])


@pytest.mark.parametrize("active", [(False, False), (True, False), (False, True)])
def test_orientation_residual_only_tracks_pregrasp_hands(active) -> None:
    z = np.asarray(((1., 0., 0.), (1., 0., 0.)))
    up = np.asarray(((0., 0., 1.), (0., 0., 1.)))
    residual = _task_space_residual(
        np.zeros((2, 3)), np.ones(2), -z,
        np.zeros((2, 3)), np.ones(2), z, np.asarray(active),
        up_directions=-up, desired_up_directions=up,
    )
    for start in (6, 12):
        rows = residual[start : start + 6].reshape(2, 3)
        for side, constrained in enumerate(active):
            assert bool(np.any(rows[side])) == constrained
    np.testing.assert_array_equal(residual[:6], 0)
    np.testing.assert_array_equal(residual[18:], 0)


def test_soft_roll_preserves_position_z_and_gripper_timeline() -> None:
    kinematics = _CartesianKinematics()
    kinematics.home = kinematics.home.copy()
    kinematics.home[[4, 12]] = np.sin(.4)
    positions = np.zeros((4, 2, 3), dtype=np.float32)
    positions[:, :, 0] = np.arange(4)[:, None] * .01
    grippers = np.asarray(((1, 1), (.8, .9), (.4, .6), (0, .1)), dtype=np.float32)
    attached = np.zeros((4, 2), dtype=np.int32)
    baseline_q = np.repeat(kinematics.home[None], len(positions), axis=0)
    result = _retarget_motion(
        _attach_poses(positions), grippers, attached,
        kinematics=kinematics, position_tolerance_m=.005,
    )
    original_z = _quat_rotate(
        kinematics.attach_poses_batch(baseline_q)[..., 3:], np.asarray((0., 0., 1.))
    )
    actual = kinematics.attach_poses_batch(result.joint_q)
    np.testing.assert_allclose(actual[..., :3], positions)
    np.testing.assert_allclose(result.target_approach_directions, original_z, atol=1e-6)
    np.testing.assert_allclose(
        _quat_rotate(actual[..., 3:], np.asarray((0., 0., 1.))), original_z, atol=1e-6,
    )
    y_height = _quat_rotate(actual[..., 3:], np.asarray((0., 1., 0.)))[..., 2]
    assert np.all(y_height > 0)
    assert np.all(np.diff(y_height, axis=0) > 0)
    assert np.max(np.abs(np.diff(result.joint_q[:, kinematics.arm_indices], axis=0))) <= .02
    assert not result.pregrasp_mask.any()
    assert not result.approach_validation_mask.any()
    np.testing.assert_array_equal(result.target_gripper_open, grippers)
    expected_fingers = kinematics.gripper_closed_positions + grippers[..., None] * (
        kinematics.gripper_open_positions - kinematics.gripper_closed_positions
    )
    np.testing.assert_allclose(
        result.joint_q[:, kinematics.gripper_indices],
        expected_fingers.reshape(len(grippers), -1),
    )


def test_motion_ignores_source_rotation_and_grasp_phase() -> None:
    kinematics = _CartesianKinematics()
    kinematics.home = kinematics.home.copy()
    kinematics.home[[4, 12]] = np.sin(.4)
    positions = np.zeros((13, 2, 3), dtype=np.float32)
    positions[..., 0] = np.arange(13)[:, None] / 100
    rotations = np.broadcast_to((0., np.sin(.4), 0., np.cos(.4)), (13, 2, 4))
    grippers = np.ones((13, 2), dtype=np.float32)
    grippers[3:8, 0] = np.linspace(1, 0, 5)
    grippers[8:, 0] = 0
    grippers[6:11, 1] = np.linspace(1, 0, 5)
    grippers[11:, 1] = 0
    attached = np.zeros((13, 2), dtype=bool)
    attached[7:, 0] = True
    attached[10:, 1] = True
    result = _retarget_motion(
        _attach_poses(positions, rotations), grippers, attached,
        kinematics=kinematics, position_tolerance_m=.005,
    )
    alternative = _retarget_motion(
        _attach_poses(positions), grippers, ~attached,
        kinematics=kinematics, position_tolerance_m=.005,
    )
    np.testing.assert_array_equal(result.joint_q, alternative.joint_q)
    assert not result.pregrasp_mask.any()
    assert not result.approach_validation_mask.any()
    actual = kinematics.attach_poses_batch(result.joint_q)
    np.testing.assert_allclose(actual[..., :3], positions)


@pytest.mark.parametrize("fault", ["position", "joint_jump", "y_down", "z_drift"])
def test_roll_preference_rejects_bad_step_per_hand(fault) -> None:
    class CorruptProjection(_CartesianKinematics):
        correcting = False

        def solve(self, seed, positions, rotations=None, **kwargs):
            result = super().solve(seed, positions, rotations, **kwargs)
            if rotations is not None:
                self.correcting = True
            elif self.correcting:
                if fault == "position":
                    result[0] += .01
                elif fault == "joint_jump":
                    result[3] += .05
                elif fault == "y_down":
                    result[[3, 5]] *= -1
                elif fault == "z_drift":
                    result[4] += .005
            return result

    kinematics = CorruptProjection()
    positioned = kinematics.home.copy()
    positioned[[4, 12]] = np.sin(.4)
    achieved = kinematics.attach_poses_batch(positioned[None])[0]
    result = _prefer_upright_roll(
        kinematics, positioned, achieved[:, :3], achieved,
        _upright_rotations(achieved[None, :, 3:])[0],
        max_step_rad=.02, position_tolerance_m=.005,
    )
    arms = kinematics.arm_indices.reshape(2, -1)
    np.testing.assert_array_equal(result[arms[0]], positioned[arms[0]])
    assert np.any(result[arms[1]] != positioned[arms[1]])
    np.testing.assert_array_equal(result[kinematics.gripper_indices], 0)


def test_soft_roll_rate_uses_actual_control_fps() -> None:
    kinematics = _CartesianKinematics()
    kinematics.home = kinematics.home.copy()
    kinematics.home[[4, 12]] = np.sin(.4)
    rotations = []
    for fps in (50, 100):
        motion = _retarget_motion(
            _attach_poses(np.zeros((1, 2, 3))), np.ones((1, 2)), np.zeros((1, 2)),
            kinematics=kinematics, position_tolerance_m=.005, control_fps=fps,
        )
        initial = kinematics.attach_poses_batch(kinematics.home[None])[0, 0, 3:]
        final = kinematics.attach_poses_batch(motion.joint_q)[0, 0, 3:]
        # Use float64 here: small quaternion angles amplify float32 dot error.
        initial, final = initial.astype(float), final.astype(float)
        initial /= np.linalg.norm(initial)
        final /= np.linalg.norm(final)
        rotations.append(2 * np.arccos(np.clip(np.dot(initial, final), -1, 1)))
    np.testing.assert_allclose(rotations, [.02, .01], atol=1e-5)


@pytest.mark.parametrize("field", ["pregrasp", "validation"])
def test_dynamics_rejects_orientation_constraints(field) -> None:
    from types import SimpleNamespace

    retargeter = object.__new__(RobotDynamicsRetargeter)
    retargeter.profile = SimpleNamespace(home_joint_q=np.zeros(16), joint_dof_count=16)
    pregrasp = np.zeros((2, 2), dtype=bool)
    validation = np.zeros_like(pregrasp)
    (pregrasp if field == "pregrasp" else validation)[0, 0] = True
    with pytest.raises(ValueError, match="position-only"):
        retargeter.reconstruct(
            np.zeros((2, 16)), np.zeros((2, 16)), np.ones((2, 2)),
            np.zeros((2, 2, 3)), np.broadcast_to((0, 0, 1), (2, 2, 3)),
            pregrasp, validation, np.zeros((2, 2)),
        )


def test_window_validation_ignores_orientation_outside_pregrasp() -> None:
    from types import SimpleNamespace

    retargeter = object.__new__(RobotDynamicsRetargeter)
    frame = SimpleNamespace(position=(0, 0, 0), rotation_xyzw=(0, 0, 0, 1))
    retargeter.profile = SimpleNamespace(attach_frames=(frame, frame))
    retargeter.attach_body_indices = (0, 1)
    bodies = np.zeros((4, 2, 7), dtype=np.float32)
    bodies[..., 6] = 1
    mask = np.asarray([[True, False], [False, True], [False, False], [False, False]])
    # Rotate only the free/attached hands by pi around X: both Y/Z reverse.
    bodies[~mask, 3] = 1
    bodies[~mask, 6] = 0
    positions = np.zeros((4, 2, 3))
    z = np.broadcast_to((0, 0, 1), positions.shape)
    up = np.broadcast_to((0, 1, 0), positions.shape)
    assert retargeter._window_validation_values(
        {'body_q': bodies}, positions, z, mask, up,
    ) == (0., 0.)
    both = np.ones_like(mask)
    assert retargeter._window_validation_values(
        {'body_q': bodies}, positions, z, both, up,
    )[1] == pytest.approx(np.pi)


def test_newton_ik_applies_orientation_mask_to_residual_and_jacobian() -> None:
    from clothdojo.data.retarget import _RobotKinematics

    assets = Path(__file__).resolve().parents[1] / 'assets'
    profile = robot_profile(ALOHA2, assets_root=assets)
    ik = _RobotKinematics(profile, assets / profile.asset, device='cpu', iterations=2)
    pose = ik.attach_poses_batch(ik.home[None])[0]
    # Inactive orientation objectives must exert no influence, even if their
    # target axes are far from home. Exercise the pinned Newton implementation.
    alternative = _upright_rotations(pose[None, :, 3:])[0]
    position_only = ik.solve(ik.home, pose[:, :3])
    for align_roll in (False, True):
        for mask in ((True, False), (False, True), (False, False), (True, True)):
            q = ik.solve(
                ik.home, pose[:, :3], alternative,
                align_roll=align_roll, orientation_mask=np.asarray(mask),
            )
            for side, objective in ik._orientation_objectives[align_roll]:
                assert (objective.weight > 0) == mask[side]
            if not any(mask):
                np.testing.assert_allclose(q, position_only, atol=1e-6)


@pytest.mark.parametrize("robot", ["aloha2", "x5", "ur5e_2f85", "fr3_duo"])
def test_every_target_uses_the_same_task_motion_rule(tmp_path, monkeypatch, robot) -> None:
    source = _write_source(tmp_path / "source")

    class ReachedMotion(Exception):
        pass

    def check_motion(poses, grippers, attached, **kwargs):
        # No robot identity or per-robot roll switch enters the task planner.
        assert set(kwargs) == {"kinematics", "position_tolerance_m", "control_fps"}
        assert kwargs["control_fps"] == 50
        assert poses.shape == (3, 2, 7)
        assert grippers.shape == attached.shape == (3, 2)
        raise ReachedMotion

    monkeypatch.setattr("clothdojo.data.retarget._retarget_motion", check_motion)
    with pytest.raises(ReachedMotion):
        _retarget_trajectory(
            source, tmp_path / "output", kinematics=_CartesianKinematics(),
            dynamics=_ExactDynamics(), position_tolerance_m=.005, target_robot=robot,
        )


def test_gripper_openness_reads_both_fingers_without_hiding_limits() -> None:
    retargeter = object.__new__(RobotDynamicsRetargeter)
    profile = robot_profile(ALOHA2)
    retargeter.profile = profile
    retargeter.gripper = np.asarray(profile.gripper_coord_indices)
    retargeter.closed = np.asarray(profile.gripper_closed_positions)
    retargeter.opened = np.asarray(profile.gripper_open_positions)
    q = np.asarray(profile.home_joint_q)
    q[retargeter.gripper] = [0.0079, 0.037, 0.0079, 0.0079]
    np.testing.assert_allclose(retargeter._finger_openness(q), [[0, 1], [0, 0]])
    np.testing.assert_allclose(retargeter._openness(q), [0.5, 0])
    q[retargeter.gripper] = [0.041, 0.041, 0, 0]
    assert retargeter._openness(q)[0] > 1.0
    assert retargeter._openness(q)[1] < 0.0
    assert retargeter._finger_openness(np.stack([q, q])).shape == (2, 2, 2)


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


def test_vertical_z_does_not_chase_source_approach_even_before_grasp() -> None:
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

    assert not motion.pregrasp_mask.any()
    assert not motion.approach_validation_mask.any()
    np.testing.assert_allclose(
        kinematics.attach_poses_batch(motion.joint_q)[..., 3:],
        np.broadcast_to((0., 0., 0., 1.), (3, 2, 4)), atol=1e-6,
    )
    np.testing.assert_allclose(motion.attach_errors, 0.0, atol=1.0e-6)


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
    assert retarget["attach_alignment"] == "position_with_soft_roll"
    assert retarget["validation"]["orientation_constraint_scope"] == "none"
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


def test_trajectory_reuses_one_final_fk_pass_for_metrics(tmp_path, monkeypatch):
    source = _write_source(tmp_path / "source")
    kinematics = _CartesianKinematics()
    original = kinematics.attach_poses_batch
    full_passes = []

    def counted(joint_q):
        if len(joint_q) == 3:
            full_passes.append(joint_q.copy())
        return original(joint_q)

    monkeypatch.setattr(kinematics, "attach_poses_batch", counted)
    output = tmp_path / "retarget"
    result = _retarget_trajectory(
        source, output, kinematics=kinematics, dynamics=_ExactDynamics(),
        position_tolerance_m=.005,
    )
    assert len(full_passes) == 1
    assert result["metadata"]["retarget"]["validation"]["kinematic_valid"]
    with np.load(output / "retarget_metrics.npz") as metrics:
        np.testing.assert_array_equal(
            metrics["kinematic_attach_position"], original(full_passes[0])[..., :3],
        )
