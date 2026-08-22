"""Convert public Newton runtime state into the trajectory dataset model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from clothdojo.trajectory.schema import TrajectoryAction, TrajectoryState

if TYPE_CHECKING:
    from clothdojo.sim.runtime import ShirtFlattenRuntime


def capture_states(
    runtime: ShirtFlattenRuntime,
    sample_index: int,
) -> tuple[TrajectoryState, ...]:
    """Copy one world-local trajectory state for every parallel world."""

    scene = runtime.scene
    state = runtime.state_0
    joint_q = scene.robot.get_attribute("joint_q", state).numpy()[:, 0]
    joint_qd = scene.robot.get_attribute("joint_qd", state).numpy()[:, 0]
    body_q = scene.robot.get_attribute("body_q", state).numpy()[:, 0]
    body_qd = scene.robot.get_attribute("body_qd", state).numpy()[:, 0]
    particle_q = state.particle_q.numpy()
    particle_qd = state.particle_qd.numpy()
    gripper_q = scene.grippers.get_attribute("joint_q", state).numpy()[:, 0]
    gripper_lower = scene.grippers.get_attribute("joint_limit_lower", runtime.model).numpy()[:, 0]
    gripper_upper = scene.grippers.get_attribute("joint_limit_upper", runtime.model).numpy()[:, 0]
    gripper_open = gripper_openness(gripper_q, gripper_lower, gripper_upper)
    grasp = runtime.grasp.capture_all_states()
    ranges = runtime.grasp.particle_ranges()
    return tuple(
        TrajectoryState(
            sample_index=int(sample_index),
            joint_q=joint_q[world].copy(),
            joint_qd=joint_qd[world].copy(),
            body_q=body_q[world].copy(),
            body_qd=body_qd[world].copy(),
            particle_q=particle_q[start:end].copy(),
            particle_qd=particle_qd[start:end].copy(),
            gripper_open_state=gripper_open[world].copy(),
            **grasp[world],
        )
        for world, (start, end) in enumerate(ranges)
    )


def capture_actions(
    runtime: ShirtFlattenRuntime,
    sample_index: int,
    arm_target,
    gripper_open,
) -> tuple[TrajectoryAction, ...]:
    """Copy commands paired with the current trajectory state."""

    arm_values = np.asarray(arm_target.numpy(), dtype=np.float32)
    gripper_values = np.asarray(gripper_open.numpy(), dtype=np.float32)
    expected_arm = (runtime.model.world_count, 12)
    expected_gripper = (runtime.model.world_count, 2)
    if arm_values.shape != expected_arm:
        raise ValueError(f"arm_target shape {arm_values.shape} != {expected_arm}")
    if gripper_values.shape != expected_gripper:
        raise ValueError(f"gripper_open shape {gripper_values.shape} != {expected_gripper}")
    return tuple(
        TrajectoryAction(
            sample_index=int(sample_index),
            joint_target=arm_values[world].copy(),
            gripper_open_command=gripper_values[world].copy(),
        )
        for world in range(runtime.model.world_count)
    )


def gripper_openness(q, lower, upper) -> np.ndarray:
    """Return normalized left/right openness from each leader finger joint."""

    controlled = (0, 2)
    return np.clip(
        (q[:, controlled] - lower[:, controlled]) / (upper[:, controlled] - lower[:, controlled]),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
