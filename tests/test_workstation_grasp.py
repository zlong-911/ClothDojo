"""Actual workstation gripper views drive the cloth grasp lifecycle on CPU."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.robots import ALOHA2, FR3_DUO, PIPER, UR5E_2F85, X5, robot_profile
from clothdojo.sim.grasp import ParticleGrasp
from clothdojo.sim.scene import _add_robot, _register_solver_attributes


@pytest.mark.parametrize("robot_id", [PIPER, ALOHA2, FR3_DUO, UR5E_2F85, X5])
def test_workstation_grasp_left_right_and_worlds_are_independent(robot_id):
    import newton
    import warp as wp

    profile = robot_profile(robot_id)
    newton.use_coord_layout_targets = True
    template = newton.ModelBuilder()
    _register_solver_attributes(newton, template)
    _bodies, _arms, gripper_joints, _arm_coords, end_effectors = _add_robot(
        newton, wp, template,
        Path(__file__).resolve().parents[1] / "assets" / profile.asset,
        profile, load_visual_shapes=False,
    )
    # A single particle at each attach center is enough to test capture and
    # release. No cloth solver, garment asset or GPU is needed.
    for _side in range(2):
        template.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.001)
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    builder.replicate(template, 2)
    model = builder.finalize(device="cpu")
    # The isolated probe has vertices but no garment triangles. Production
    # scenes always provide this array; expose an empty adjacency here.
    model.tri_indices = wp.empty((0, 3), dtype=wp.int32, device="cpu")
    grippers = newton.selection.ArticulationView(model, "*", include_joints=gripper_joints)
    scene = SimpleNamespace(
        model=model, grippers=grippers, robot_profile=profile,
        end_effector_body_indices=end_effectors, garment_particles_per_world=(0, 2),
        spec=SimpleNamespace(control_fps=50.0, sim_substeps=6),
    )
    grasp = ParticleGrasp(scene)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_q = state.body_q.numpy()
    particle_centers = []
    for world in range(2):
        for side, frame in enumerate(profile.attach_frames):
            body = body_q[world * len(profile.body_names) + end_effectors[side]]
            transform = wp.transform(body[:3], body[3:])
            particle_centers.append(tuple(wp.transform_point(transform, wp.vec3(frame.position))))
    state.particle_q.assign(np.asarray(particle_centers, dtype=np.float32))

    collision_shapes = grasp._collision_shapes.numpy()
    collision_counts = grasp._collision_shape_count.numpy()
    assert np.all(collision_counts > 0), "every wrist subtree must select cloth-colliding shapes"
    selected = [
        set(collision_shapes[world, side, :collision_counts[world, side]])
        for world in range(2) for side in range(2)
    ]
    for index, shapes in enumerate(selected):
        assert all(shapes.isdisjoint(other) for other in selected[index + 1:])
    if robot_id == PIPER:
        # The unified wrist subtree preserves Piper's three-link selection
        # and deliberately adds its one independently colliding wrist camera.
        # Both now share the short release grace; no upper-arm shape is added.
        shape_body = model.shape_body.numpy()
        shape_world = model.shape_world.numpy()
        shape_flags = model.shape_flags.numpy()
        labels = tuple("/" + str(label).lstrip("/") for label in model.body_label)
        for world in range(2):
            for side, side_name in enumerate(("left", "right")):
                previous = {
                    shape for shape, body in enumerate(shape_body)
                    if body >= 0 and shape_world[shape] == world
                    and shape_flags[shape] & grasp._particle_collision_bit
                    and f"/{side_name}/" in labels[body]
                    and any(
                        labels[body].endswith(f"/{link}") for link in ("link6", "link7", "link8")
                    )
                }
                actual = selected[world * 2 + side]
                assert previous.issubset(actual)
                extra = actual - previous
                assert len(extra) == 1
                assert labels[shape_body[next(iter(extra))]].endswith(f"/{side_name}/camera")

    def update(openness):
        opened = np.asarray(profile.gripper_open_positions)
        closed = np.asarray(profile.gripper_closed_positions)
        targets = closed + np.asarray(openness)[..., None] * (opened - closed)
        shape = grippers.get_attribute("joint_q", state).shape
        grippers.set_attribute(
            "joint_q", state,
            wp.array(targets.reshape(shape), dtype=wp.float32, device="cpu"),
        )
        grasp.update(state, state)

    update(((1.0, 1.0), (1.0, 1.0)))
    np.testing.assert_array_equal(grasp.attached.numpy(), [[0, 0], [0, 0]])
    np.testing.assert_allclose(grasp.previous_open.numpy(), 1.0)
    update(((0.0, 1.0), (1.0, 0.0)))
    np.testing.assert_array_equal(grasp.attached.numpy(), [[1, 0], [0, 1]])
    np.testing.assert_array_equal(grasp.pin_count.numpy(), [[1, 0], [0, 1]])
    assert grasp.pins.numpy()[0, 0, 0] == 0
    assert grasp.pins.numpy()[1, 1, 0] == 3

    # Release those two while independently capturing the opposite two.
    update(((1.0, 0.0), (0.0, 1.0)))
    np.testing.assert_array_equal(grasp.attached.numpy(), [[0, 1], [1, 0]])
    np.testing.assert_array_equal(grasp.pin_count.numpy(), [[0, 1], [1, 0]])
    grace = grasp._release_collision_off_substeps
    np.testing.assert_array_equal(
        grasp.release_collision_countdown.numpy(), [[grace, 0], [0, grace]]
    )
    flags = model.shape_flags.numpy()
    for world in range(2):
        for side in range(2):
            enabled = world != side
            for shape in selected[world * 2 + side]:
                assert bool(flags[shape] & grasp._particle_collision_bit) == enabled

    update(((1.0, 1.0), (1.0, 1.0)))
    np.testing.assert_array_equal(grasp.attached.numpy(), [[0, 0], [0, 0]])
    np.testing.assert_array_equal(grasp.pin_count.numpy(), [[0, 0], [0, 0]])
