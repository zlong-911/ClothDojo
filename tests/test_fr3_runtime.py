"""Rigid-only CPU coverage of single- and dual-articulation bimanual controls."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.robots import ALOHA2, FR3_DUO, robot_profile
from clothdojo.sim.runtime import ShirtFlattenRuntime
from clothdojo.sim.scene import _add_robot, _register_solver_attributes


@pytest.fixture(scope="module", params=[(FR3_DUO, 1), (FR3_DUO, 2), (ALOHA2, 1), (ALOHA2, 2)])
def rigid_robot(request):
    import newton
    import warp as wp

    robot_id, world_count = request.param
    profile = robot_profile(robot_id)
    newton.use_coord_layout_targets = True
    template = newton.ModelBuilder()
    _register_solver_attributes(newton, template)
    topology = _add_robot(
        newton,
        wp,
        template,
        Path(__file__).resolve().parents[1] / "assets" / profile.asset,
        profile,
        load_visual_shapes=False,
    )
    body_count, arm_joints, gripper_joints, arm_coords, end_effectors = topology
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    builder.replicate(template, world_count)
    model = builder.finalize(device="cpu")
    roots = profile.body_parents.count(-1)
    arms = newton.selection.ArticulationView(model, "*", include_joints=arm_joints)
    grippers = newton.selection.ArticulationView(model, "*", include_joints=gripper_joints)
    end_effector_view = newton.selection.ArticulationView(
        model, "*", include_links=end_effectors if roots == 1 else [end_effectors[0]]
    )
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return SimpleNamespace(
        profile=profile, model=model, arms=arms, grippers=grippers,
        end_effectors=end_effector_view, state=state, worlds=world_count,
        roots=roots, body_count=body_count, arm_coords=arm_coords,
        end_effector_indices=end_effectors,
    )


def test_runtime_articulation_views_cover_both_arms_and_grippers(rigid_robot):
    robot = rigid_robot
    profile = robot.profile
    model = robot.model
    assert model.world_count == robot.worlds
    assert model.body_count == robot.body_count * robot.worlds
    assert robot.arms.count == robot.worlds * robot.roots
    assert robot.grippers.count == robot.worlds * robot.roots
    assert tuple(robot.arm_coords) == profile.arm_coord_indices
    arm_q = robot.arms.get_attribute("joint_q", robot.state).numpy()
    gripper_q = robot.grippers.get_attribute("joint_q", robot.state).numpy()
    assert arm_q.shape == (robot.worlds, robot.roots, len(profile.arm_coord_indices) // robot.roots)
    assert gripper_q.shape == (
        robot.worlds, robot.roots, len(profile.gripper_coord_indices) // robot.roots
    )
    all_q = robot.state.joint_q.numpy().reshape(robot.worlds, -1)
    np.testing.assert_allclose(all_q, np.tile(profile.home_joint_q, (robot.worlds, 1)), atol=2e-6)
    np.testing.assert_array_equal(
        arm_q.reshape(robot.worlds, -1), all_q[:, profile.arm_coord_indices]
    )
    np.testing.assert_array_equal(
        gripper_q.reshape(robot.worlds, -1), all_q[:, profile.gripper_coord_indices]
    )
    end_effectors = robot.end_effectors.get_attribute("body_q", robot.state).numpy()
    assert end_effectors.shape[:3] == (robot.worlds, robot.roots, 2 // robot.roots)
    all_bodies = robot.state.body_q.numpy().reshape(robot.worlds, robot.body_count, 7)
    np.testing.assert_allclose(
        end_effectors.reshape(robot.worlds, 2, 7),
        all_bodies[:, robot.end_effector_indices],
    )


def test_runtime_left_right_gripper_actions_remain_independent(rigid_robot):
    import warp as wp

    robot = rigid_robot
    profile = robot.profile
    # Exercise the actual apply_action method and ArticulationView writes,
    # without constructing a cloth solver or taking a physics step.
    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.model = robot.model
    runtime.scene = SimpleNamespace(arms=robot.arms, grippers=robot.grippers, robot_profile=profile)
    runtime.control = robot.model.control()
    runtime._arm_action_count = len(profile.arm_coord_indices)
    runtime._gripper_action_count = len(profile.gripper_coord_indices)
    runtime._arm_lower = robot.arms.get_attribute("joint_limit_lower", robot.model)
    runtime._arm_upper = robot.arms.get_attribute("joint_limit_upper", robot.model)
    runtime._arm_targets = wp.empty_like(runtime._arm_lower)
    runtime._arm_joints_per_articulation = runtime._arm_targets.shape[2]
    runtime._gripper_targets = wp.empty_like(robot.grippers.get_attribute("joint_q", robot.state))
    runtime._gripper_joints_per_articulation = runtime._gripper_targets.shape[2]
    runtime._gripper_closed_positions = wp.array(
        profile.gripper_closed_positions * 2, dtype=wp.float32, device="cpu"
    )
    runtime._gripper_open_positions = wp.array(
        profile.gripper_open_positions * 2, dtype=wp.float32, device="cpu"
    )
    home = np.asarray(profile.home_joint_q, dtype=np.float32)
    arm_commands = np.tile(home[list(profile.arm_coord_indices)], (robot.worlds, 1))
    arm_commands += np.linspace(-0.02, 0.02, runtime._arm_action_count, dtype=np.float32)
    openness = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.float32)[:robot.worlds]
    before = runtime.control.joint_target_q.numpy().reshape(robot.worlds, -1).copy()
    runtime.apply_action(
        wp.array(arm_commands, dtype=wp.float32, device="cpu"),
        wp.array(openness, dtype=wp.float32, device="cpu"),
    )
    actual = runtime.control.joint_target_q.numpy().reshape(robot.worlds, -1)
    expected_arms = np.clip(
        arm_commands,
        runtime._arm_lower.numpy().reshape(robot.worlds, -1),
        runtime._arm_upper.numpy().reshape(robot.worlds, -1),
    )
    np.testing.assert_allclose(actual[:, profile.arm_coord_indices], expected_arms)
    closed = np.asarray(profile.gripper_closed_positions)
    opened = np.asarray(profile.gripper_open_positions)
    expected_grippers = closed + openness[..., None] * (opened - closed)
    np.testing.assert_allclose(
        actual[:, profile.gripper_coord_indices], expected_grippers.reshape(robot.worlds, -1)
    )
    passive = np.setdiff1d(
        np.arange(len(home)), (*profile.arm_coord_indices, *profile.gripper_coord_indices)
    )
    np.testing.assert_array_equal(actual[:, passive], before[:, passive])
