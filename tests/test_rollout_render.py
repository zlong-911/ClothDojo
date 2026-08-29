from __future__ import annotations

import numpy as np
import warp as wp

from clothdojo.rendering import (
    DATASET_CAMERA_NAMES,
    POLICY_RENDER_SIZE,
    policy_observation_settings,
)
from clothdojo.robots import PIPER, robot_profile
from clothdojo.rollout.render import (
    _accumulate_vertex_normals,
    _normalize_vertex_normals,
    _refit_deforming_mesh_bounds,
    _update_wrist_cameras,
)


def test_policy_render_contract_uses_the_model_input_resolution() -> None:
    settings = policy_observation_settings()

    assert POLICY_RENDER_SIZE == (640, 480)
    assert settings["render_width_height"] == [640, 480]
    assert settings["transport_width_height"] == [224, 224]
    assert "calibration_sha256" not in settings


def test_dataset_camera_contract_uses_standard_four_view_order() -> None:
    assert DATASET_CAMERA_NAMES == (
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    cameras = robot_profile(PIPER).cameras
    assert tuple(camera.parent_body for camera in cameras) == (
        None,
        None,
        "left/camera",
        "right/camera",
    )


def test_wrist_camera_transforms_update_every_world() -> None:
    body_q = wp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=wp.transformf,
        device="cpu",
    )
    body_indices = wp.array([0, 1], dtype=int, device="cpu")
    camera_indices = wp.array([1, 3], dtype=int, device="cpu")
    local = wp.array(
        [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2,
        dtype=wp.transformf,
        device="cpu",
    )
    transforms = wp.zeros((4, 2), dtype=wp.transformf, device="cpu")

    wp.launch(
        _update_wrist_cameras,
        dim=(2, 2),
        inputs=[body_q, camera_indices, body_indices, local, 2, transforms],
        device="cpu",
    )

    np.testing.assert_allclose(
        transforms.numpy()[[1, 3], :, :3],
        [
            [[1.5, 0.0, 0.0], [11.5, 0.0, 0.0]],
            [[2.5, 0.0, 0.0], [12.5, 0.0, 0.0]],
        ],
    )


def test_deforming_normals_follow_logical_surface_winding() -> None:
    points = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=wp.vec3f,
        device="cpu",
    )
    indices = wp.array([0, 1, 2], dtype=int, device="cpu")
    normals = wp.zeros(3, dtype=wp.vec3f, device="cpu")

    for winding_sign, expected_z in ((1.0, 1.0), (-1.0, -1.0)):
        normals.zero_()
        wp.launch(
            _accumulate_vertex_normals,
            dim=1,
            inputs=[points, indices, normals, winding_sign],
            device="cpu",
        )
        wp.launch(
            _normalize_vertex_normals,
            dim=3,
            inputs=[normals],
            device="cpu",
        )
        np.testing.assert_allclose(
            normals.numpy(),
            np.asarray([[0.0, 0.0, expected_z]] * 3, dtype=np.float32),
        )


def test_deforming_mesh_bounds_follow_current_vertices() -> None:
    shape_bounds = wp.zeros((2, 2), dtype=wp.vec3f, device="cpu")
    scratch = wp.empty((2, 3), dtype=wp.float32, device="cpu")

    for points, world_offset, expected in (
        (
            [[-1.0, 2.0, 0.5], [3.0, -4.0, 2.5], [0.0, 1.0, -2.0]],
            [0.0, 0.0, 0.0],
            [[-1.0, -4.0, -2.0], [3.0, 2.0, 2.5]],
        ),
        (
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
            [10.0, 20.0, 30.0],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ),
    ):
        vertices = wp.array(points, dtype=wp.vec3f, device="cpu")
        _refit_deforming_mesh_bounds(
            vertices,
            scratch,
            shape_bounds,
            1,
            world_offset=wp.vec3f(*world_offset),
            device="cpu",
        )
        np.testing.assert_allclose(shape_bounds.numpy()[1], expected)

    np.testing.assert_array_equal(shape_bounds.numpy()[0], np.zeros((2, 3), dtype=np.float32))
