from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp

from clothdojo.sim.grasp import (
    ParticleGrasp,
    _attach_local_offsets,
    _reset_grasp,
    _reset_release_collision,
    _update_grasp,
)


def test_particle_grasp_uses_robot_profile_attach_centers() -> None:
    frames = (
        SimpleNamespace(position=(0.13, 0.0, -0.003)),
        SimpleNamespace(position=(0.14, 0.01, -0.004)),
    )
    scene = SimpleNamespace(robot_profile=SimpleNamespace(attach_frames=frames))

    np.testing.assert_allclose(
        _attach_local_offsets(scene),
        ((0.13, 0.0, -0.003), (0.14, 0.01, -0.004)),
    )


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
    release_collision_countdown = wp.zeros(shape2, dtype=wp.int32, device=device)
    shape_flags = wp.array([4, 4], dtype=wp.int32, device=device)
    collision_shapes = wp.array([[[0], [1]]], dtype=wp.int32, device=device)
    collision_shape_count = wp.ones(shape2, dtype=wp.int32, device=device)

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
    closed = wp.array([0.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
    open_positions = wp.array([0.04, -0.04, 0.04, -0.04], dtype=wp.float32, device=device)
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)

    def update() -> None:
        wp.launch(
            _update_grasp,
            dim=(1, 2),
            inputs=[
                gripper_q,
                closed,
                open_positions,
                4,
                2,
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
                release_collision_countdown,
                shape_flags,
                collision_shapes,
                collision_shape_count,
                0.1,
                0.7,
                3,
                4,
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
    assert release_collision_countdown.numpy().tolist() == [[3, 0]]
    assert shape_flags.numpy().tolist() == [0, 4]

    update()
    update()
    assert release_collision_countdown.numpy().tolist() == [[1, 0]]
    assert shape_flags.numpy().tolist() == [0, 4]

    update()
    assert release_collision_countdown.numpy().tolist() == [[0, 0]]
    assert shape_flags.numpy().tolist() == [4, 4]

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

    flags = shape_flags.numpy()
    flags[0] = 0
    shape_flags.assign(flags)
    release_collision_countdown.assign(np.array([[2, 0]], dtype=np.int32))
    wp.launch(
        _reset_release_collision,
        dim=(1, 2),
        inputs=[
            wp.array([True], dtype=wp.bool, device=device),
            release_collision_countdown,
            shape_flags,
            collision_shapes,
            collision_shape_count,
            4,
        ],
        device=device,
    )
    assert release_collision_countdown.numpy().tolist() == [[0, 0]]
    assert shape_flags.numpy().tolist() == [4, 4]


def test_gripper_collision_shapes_are_isolated_by_world_side_and_link() -> None:
    grasp = ParticleGrasp.__new__(ParticleGrasp)
    grasp.world_count = 2
    grasp.scene = SimpleNamespace(robot_profile=SimpleNamespace(
        body_names=("left/link6", "left/link7", "right/link6", "left/link5"),
        body_parents=(-1, 0, -1, -1),
        home_joint_q=(0.0, 0.0),
        arm_coord_indices=(0, 1),
        arm_dofs_per_side=1,
    ))
    grasp.model = SimpleNamespace(
        body_count=8,
        joint_q_start=wp.array([0, 1, 2, 3, 4], dtype=wp.int32, device="cpu"),
        joint_world=wp.array([0, 0, 1, 1], dtype=wp.int32, device="cpu"),
        joint_child=wp.array([0, 2, 4, 6], dtype=wp.int32, device="cpu"),
        body_label=[
            "/robot/left/link6",
            "/robot/left/link7",
            "/robot/right/link6",
            "/robot/left/link5",
            "/robot/left/link6",
            "/robot/left/link8",
            "/robot/right/link7",
            "/robot/right/link5",
        ],
        body_world=wp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=wp.int32, device="cpu"),
        shape_body=wp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=wp.int32, device="cpu"),
        shape_world=wp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=wp.int32, device="cpu"),
        # Bit 4 is COLLIDE_PARTICLES. Shapes 1 and 6 are proxy/disabled shapes.
        shape_flags=wp.array([4, 2, 4, 4, 4, 4, 0, 4], dtype=wp.int32, device="cpu"),
    )

    indices, counts, collision_bit = grasp._gripper_collision_shapes()

    assert collision_bit == 4
    np.testing.assert_array_equal(counts, [[1, 1], [2, 0]])
    np.testing.assert_array_equal(indices[0, 0, :1], [0])
    np.testing.assert_array_equal(indices[0, 1, :1], [2])
    np.testing.assert_array_equal(indices[1, 0, :2], [4, 5])
