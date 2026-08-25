from __future__ import annotations

import numpy as np
import warp as wp

from clothdojo.rollout.render import (
    _refit_deforming_mesh_bounds,
    _update_wrist_cameras,
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
    local = wp.array(
        [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2,
        dtype=wp.transformf,
        device="cpu",
    )
    transforms = wp.zeros((3, 2), dtype=wp.transformf, device="cpu")

    wp.launch(
        _update_wrist_cameras,
        dim=(2, 2),
        inputs=[body_q, body_indices, local, 2, transforms],
        device="cpu",
    )

    np.testing.assert_allclose(
        transforms.numpy()[1:, :, :3],
        [
            [[1.5, 0.0, 0.0], [11.5, 0.0, 0.0]],
            [[2.5, 0.0, 0.0], [12.5, 0.0, 0.0]],
        ],
    )


def test_deforming_mesh_bounds_follow_current_vertices() -> None:
    shape_bounds = wp.zeros((2, 2), dtype=wp.vec3f, device="cpu")
    scratch = wp.empty((2, 3), dtype=wp.float32, device="cpu")

    for points, expected in (
        (
            [[-1.0, 2.0, 0.5], [3.0, -4.0, 2.5], [0.0, 1.0, -2.0]],
            [[-1.0, -4.0, -2.0], [3.0, 2.0, 2.5]],
        ),
        (
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
        ),
    ):
        vertices = wp.array(points, dtype=wp.vec3f, device="cpu")
        _refit_deforming_mesh_bounds(
            vertices,
            scratch,
            shape_bounds,
            1,
            device="cpu",
        )
        np.testing.assert_allclose(shape_bounds.numpy()[1], expected)

    np.testing.assert_array_equal(shape_bounds.numpy()[0], np.zeros((2, 3), dtype=np.float32))
