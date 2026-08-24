from __future__ import annotations

import numpy as np
import warp as wp

from clothdojo.sim.grasp import _reset_grasp, _update_grasp


def test_grasp_matches_expert_four_seed_two_ring_patch_and_release() -> None:
    device = "cpu"
    capacity = 32
    shape2 = (1, 2)
    shape3 = (1, 2, capacity)
    attached = wp.zeros(shape2, dtype=wp.int32, device=device)
    previous_open = wp.ones(shape2, dtype=wp.float32, device=device)
    pin_count = wp.zeros(shape2, dtype=wp.int32, device=device)
    pins = wp.full(shape3, -1, dtype=wp.int32, device=device)
    pin_local = wp.zeros(shape3, dtype=wp.vec3, device=device)
    frontier = wp.full(shape3, -1, dtype=wp.int32, device=device)
    next_frontier = wp.full(shape3, -1, dtype=wp.int32, device=device)

    # Ten particles form a line. Four nearest seeds plus two expansion rings
    # must select [0..5]; the former nearest-plus-one-ring implementation only
    # selected [0, 1].
    particle_q = wp.array(
        [(0.001 * index, 0.0, 0.0) for index in range(10)],
        dtype=wp.vec3,
        device=device,
    )
    neighbor_offsets = wp.array(
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 18], dtype=wp.int32, device=device
    )
    neighbor_indices = wp.array(
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8],
        dtype=wp.int32,
        device=device,
    )
    gripper_q = wp.array([[[0.0044, -0.04, 0.04, -0.04]]], dtype=wp.float32, device=device)
    lower = wp.array([[[0.0, -0.04, 0.0, -0.04]]], dtype=wp.float32, device=device)
    upper = wp.array([[[0.04, 0.0, 0.04, 0.0]]], dtype=wp.float32, device=device)
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)

    def update() -> None:
        wp.launch(
            _update_grasp,
            dim=(1, 2),
            inputs=[
                gripper_q,
                lower,
                upper,
                wp.array([[0, 0]], dtype=wp.int32, device=device),
                body_q,
                particle_q,
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([10], dtype=wp.int32, device=device),
                neighbor_offsets,
                neighbor_indices,
                attached,
                previous_open,
                pin_count,
                pins,
                pin_local,
                frontier,
                next_frontier,
                wp.array([(0.0, 0.0, 0.0)] * 2, dtype=wp.vec3, device=device),
                0.1,
                0.7,
                0.01**2,
                4,
                2,
                32,
                0.025,
            ],
            device=device,
        )

    update()

    assert attached.numpy().tolist() == [[1, 0]]
    assert pin_count.numpy().tolist() == [[6, 0]]
    np.testing.assert_array_equal(pins.numpy()[0, 0, :6], np.arange(6))

    opened = gripper_q.numpy()
    opened[0, 0, 0] = 0.04
    gripper_q.assign(opened)
    update()
    assert attached.numpy().tolist() == [[0, 0]]
    assert pin_count.numpy().tolist() == [[0, 0]]

    wp.launch(
        _reset_grasp,
        dim=(1, 2),
        inputs=[
            wp.array([True], dtype=wp.bool, device=device),
            attached,
            previous_open,
            pin_count,
            pins,
            pin_local,
        ],
        device=device,
    )
    assert np.all(pins.numpy() == -1)
