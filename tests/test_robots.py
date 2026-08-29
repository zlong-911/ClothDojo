from __future__ import annotations

import numpy as np
import pytest

from clothdojo.robots import (
    ALOHA2,
    PANDA,
    PIPER,
    UR5E_2F85,
    X5,
    robot_profile,
)
from clothdojo.tasks import SHIRT_FLATTEN, SHIRT_FOLD, task_profile


def _project(camera, points: np.ndarray) -> np.ndarray:
    world_from_camera = np.asarray(camera.parent_from_camera, dtype=np.float64)
    homogeneous = np.concatenate(
        (np.asarray(points, dtype=np.float64), np.ones((len(points), 1))), axis=1
    )
    camera_points = homogeneous @ np.linalg.inv(world_from_camera).T
    depth = -camera_points[:, 2]
    vertical_aperture = camera.horizontal_aperture * 480.0 / 640.0
    u = 320.0 + 640.0 * camera.focal_length * camera_points[:, 0] / (
        depth * camera.horizontal_aperture
    )
    v = 240.0 - 480.0 * camera.focal_length * camera_points[:, 1] / (depth * vertical_aperture)
    return np.stack((u, v, depth), axis=1)


def test_piper_profile_reads_the_four_asset_authored_cameras() -> None:
    profile = robot_profile(PIPER)

    assert profile.robot_id == "piper"
    assert "task_name" not in profile.to_json_dict()
    assert "fingerprint" not in profile.to_json_dict()

    assert tuple(camera.name for camera in profile.cameras) == (
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    assert tuple(camera.parent_body for camera in profile.cameras) == (
        None,
        None,
        "left/camera",
        "right/camera",
    )
    np.testing.assert_allclose(
        np.asarray(profile.cameras[0].parent_from_camera)[:3, 3],
        (0.253845, 0.001071, 1.308),
    )
    np.testing.assert_allclose(
        np.asarray(profile.cameras[2].parent_from_camera)[:3, 3],
        (0.0, 0.0, 0.0109),
    )


def test_aloha_static_cameras_frame_both_robots_and_table_workspace() -> None:
    profile = robot_profile(ALOHA2)
    assert tuple(frame.position for frame in profile.installations) == (
        (0.382, 0.559, 0.82),
        (0.382, -0.559, 0.82),
    )
    assert profile.gripper_closed_positions == (0.0079, 0.0079)
    assert profile.gripper_open_positions == (0.037, 0.037)
    assert profile.newton_asset == "robots/aloha2/aloha.xml"
    assert profile.installation_asset == "robots/aloha2/frame.xml"
    assert profile.body_parents[8:10] == (7, 7)
    assert profile.body_parents[18:20] == (17, 17)
    assert tuple(camera.name for camera in profile.cameras) == (
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    assert all(camera.focal_length == 1.93 for camera in profile.cameras)
    assert all(camera.horizontal_aperture == 3.896 for camera in profile.cameras)
    representative_points = np.asarray(
        (
            (0.382, 0.559, 0.82),
            (0.382, -0.559, 0.82),
            (0.4, 0.3, 1.0),
            (0.4, -0.3, 1.0),
            (0.401, 0.0, 0.8),
        )
    )

    high = _project(profile.cameras[0], representative_points)
    assert np.all(high[:, 2] > 0.0)
    assert np.all((0.0 <= high[:, 0]) & (high[:, 0] < 640.0))
    assert np.all((0.0 <= high[:, 1]) & (high[:, 1] < 480.0))

    low = _project(profile.cameras[1], representative_points[2:])
    assert np.all(low[:, 2] > 0.0)
    assert np.all((0.0 <= low[:, 0]) & (low[:, 0] < 640.0))
    assert np.all((0.0 <= low[:, 1]) & (low[:, 1] < 480.0))
    np.testing.assert_allclose(
        np.asarray(profile.cameras[1].parent_from_camera)[:3, 3],
        (0.023833, 0.0, 0.8316055),
    )


def test_aloha_wrist_cameras_frame_open_fingers_at_a_readable_distance() -> None:
    profile = robot_profile(ALOHA2)
    finger_centers = np.asarray(
        (
            (0.0361, 0.0008363, 0.0811727),
            (-0.0361, 0.0008363, 0.0811727),
            (0.0, 0.0030756, 0.0949976),
        )
    )

    for camera in profile.cameras[2:4]:
        projected = _project(camera, finger_centers)
        assert np.all((0.1 <= projected[:, 2]) & (projected[:, 2] <= 0.15))
        assert np.all((120.0 <= projected[:, 0]) & (projected[:, 0] <= 520.0))
        assert np.all((300.0 <= projected[:, 1]) & (projected[:, 1] <= 370.0))


def test_panda_profile_uses_two_pinned_standard_single_arm_assets() -> None:
    profile = robot_profile(PANDA)

    assert profile.newton_asset == "robots/panda/panda.xml"
    assert profile.asset_instances == 2
    assert profile.arm_dofs_per_side == 7
    assert len(profile.arm_coord_indices) == 14
    assert len(profile.home_joint_q) == 18
    assert tuple(frame.position for frame in profile.installations) == (
        (0.25, 0.6, 0.8),
        (0.25, -0.6, 0.8),
    )
    assert profile.body_parents.count(-1) == 2
    assert profile.position_drive_override is not None


def test_task_profiles_are_independent_of_robot_profiles() -> None:
    assert task_profile(SHIRT_FLATTEN).scene_available is True
    assert task_profile(SHIRT_FOLD).scene_available is False

    from clothdojo.sim.scene import build_scene

    with pytest.raises(NotImplementedError, match="assets and randomization"):
        build_scene("assets", robot_id=PIPER, task_name=SHIRT_FOLD)


def test_x5_profile_reuses_the_old_validated_dual_installation() -> None:
    profile = robot_profile(X5)

    assert profile.newton_asset == "robots/x5/x5.xml"
    assert profile.asset_instances == 2
    assert profile.arm_dofs_per_side == 6
    assert len(profile.home_joint_q) == 16
    assert tuple(frame.position for frame in profile.installations) == (
        (0.25, 0.3, 0.8),
        (0.25, -0.3, 0.8),
    )
    assert profile.attach_frames[0].position == (0.152602, 0.0, -0.00014)
    assert profile.gripper_open_positions == (0.044, 0.044)
    assert profile.body_parents.count(-1) == 2
    assert profile.position_drive_override is not None


def test_ur5e_2f85_profile_keeps_closed_chain_coordinates_separate_from_dofs() -> None:
    profile = robot_profile(UR5E_2F85)

    assert profile.newton_asset == "robots/ur5e_2f85/ur5e_2f85.xml"
    assert profile.asset_instances == 2
    assert profile.arm_dofs_per_side == 6
    assert len(profile.home_joint_q) == 44
    assert profile.joint_dof_count == 40
    assert profile.arm_coord_indices[6:] == (22, 23, 24, 25, 26, 27)
    assert profile.arm_dof_indices[6:] == (20, 21, 22, 23, 24, 25)
    assert profile.gripper_coord_indices == (6, 10, 28, 32)
    assert profile.gripper_dof_indices == (6, 10, 26, 30)
    assert profile.body_parents.count(-1) == 2
    assert profile.position_drive_override is not None
