import ast
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.robots import ALOHA2, PIPER, robot_profile
from clothdojo.sim.check import _gripper_openness
from clothdojo.sim.runtime import (
    ShirtFlattenRuntime,
    _prune_static_soft_contact_pairs,
    _write_arm_targets,
    _write_gripper_targets,
)
from clothdojo.sim.scene import (
    ARM_JOINTS,
    GRIPPER_JOINTS,
    ShirtFlattenSpec,
    _add_robot,
    _add_table,
    _body_indices,
    _centered_world_offsets,
    _garment_visual_mesh,
    _joint_dofs,
    _joint_indices,
    _nocs_visual_mesh,
    _register_solver_attributes,
    _select_piper_robot,
    _vertex_normals,
    build_scene,
    rest_vertices_to_nocs,
)


def test_white_garment_visual_changes_only_texture_rgb() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.2]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    nocs = _garment_visual_mesh(
        vertices, triangles, vertices, cloth_visual="nocs"
    )
    white = _garment_visual_mesh(
        vertices, triangles, vertices, cloth_visual="white"
    )

    for nocs_value, white_value in zip(nocs[:4], white[:4], strict=True):
        np.testing.assert_array_equal(nocs_value, white_value)
    assert np.any(nocs[4][..., :3] != 255)
    np.testing.assert_array_equal(white[4], np.full_like(white[4], 255))


def test_scene_sets_safe_openusd_concurrency_before_newton_import() -> None:
    env = os.environ.copy()
    env.pop("PXR_WORK_THREAD_LIMIT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import clothdojo.sim.scene; print(os.environ['PXR_WORK_THREAD_LIMIT'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "1"


def test_scene_preserves_host_openusd_concurrency_after_newton_import() -> None:
    env = os.environ.copy()
    env.pop("PXR_WORK_THREAD_LIMIT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import newton; from pxr import Work; before = Work.GetConcurrencyLimit(); "
            "import clothdojo.sim.scene; print(before, Work.GetConcurrencyLimit())",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    before, after = (int(value) for value in result.stdout.split())
    assert after == before


def test_sim_runtime_does_not_depend_on_trajectory_dataset() -> None:
    path = Path(__file__).parents[1] / "src" / "clothdojo" / "sim" / "runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("clothdojo.trajectory") for module in imports)


def test_sim_runtime_enables_cuda_graph_by_default() -> None:
    assert ShirtFlattenRuntime.__init__.__kwdefaults__ == {
        "use_cuda_graph": True,
        "rigid_deterministic": False,
        "rigid_cuda_graph": False,
    }

    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.spec = SimpleNamespace(sim_substeps=3)
    with pytest.raises(RuntimeError, match="even sim_substeps"):
        runtime._capture_step_graph()


def test_aloha_action_targets_follow_two_articulation_layout() -> None:
    import warp as wp

    arm = np.arange(12, dtype=np.float32).reshape(1, 12) / 10.0
    arm_target = wp.zeros((1, 2, 6), dtype=wp.float32, device="cpu")
    arm_limit = wp.array(np.full((1, 2, 6), 100.0, dtype=np.float32), device="cpu")
    wp.launch(
        _write_arm_targets,
        dim=(1, 12),
        inputs=[wp.array(arm, device="cpu"), -arm_limit, arm_limit, 6],
        outputs=[arm_target],
        device="cpu",
    )
    np.testing.assert_allclose(arm_target.numpy().reshape(1, 12), arm)

    gripper_target = wp.zeros((1, 2, 2), dtype=wp.float32, device="cpu")
    wp.launch(
        _write_gripper_targets,
        dim=(1, 4),
        inputs=[
            wp.array([[0.0, 1.0]], dtype=wp.float32, device="cpu"),
            wp.array([0.0079] * 4, dtype=wp.float32, device="cpu"),
            wp.array([0.037] * 4, dtype=wp.float32, device="cpu"),
            2,
            2,
        ],
        outputs=[gripper_target],
        device="cpu",
    )
    np.testing.assert_allclose(gripper_target.numpy(), [[[0.0079, 0.0079], [0.037, 0.037]]])


def test_runtime_groups_rigid_state_by_physics_world() -> None:
    import warp as wp

    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.model = SimpleNamespace(
        world_count=2,
        joint_coord_count=6,
        joint_dof_count=6,
        body_count=4,
    )
    runtime.state_0 = SimpleNamespace(
        joint_q=wp.array(np.arange(6, dtype=np.float32), device="cpu"),
        joint_qd=wp.array(np.arange(6, dtype=np.float32) + 10.0, device="cpu"),
        body_q=wp.array(np.arange(28, dtype=np.float32).reshape(4, 7), device="cpu"),
        body_qd=wp.array(np.arange(24, dtype=np.float32).reshape(4, 6), device="cpu"),
    )

    state = runtime.rigid_state_batch()

    assert state["joint_q"].shape == (2, 3)
    assert state["joint_qd"].shape == (2, 3)
    assert state["body_q"].shape == (2, 2, 7)
    assert state["body_qd"].shape == (2, 2, 6)
    np.testing.assert_array_equal(state["joint_q"][1], (3.0, 4.0, 5.0))
    np.testing.assert_array_equal(
        state["body_q"][1], np.arange(14, 28, dtype=np.float32).reshape(2, 7)
    )


def test_runtime_step_graph_keeps_bvh_rebuild_outside_substeps() -> None:
    events = []
    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.state_0 = "current"
    runtime.state_1 = "next"
    runtime.model = SimpleNamespace(world_count=1)
    runtime._arm_action_count = 12
    runtime.spec = SimpleNamespace(sim_substeps=2, control_fps=50)
    runtime.grasp = SimpleNamespace(
        update=lambda particle, rigid: events.append(("grasp", particle, rigid))
    )
    runtime.cloth = SimpleNamespace(rebuild_bvh=lambda state: events.append(("rebuild", state)))
    runtime.apply_action = lambda arm, gripper: events.append(("action", arm, gripper))
    runtime._step_rigid = lambda dt: events.append(("rigid", dt))
    runtime._step_cloth = lambda dt: events.append(("cloth", dt))

    runtime._step_eager("arm", "gripper")

    assert [event[0] for event in events] == [
        "action",
        "rigid",
        "grasp",
        "cloth",
        "rigid",
        "grasp",
        "cloth",
    ]

    events.clear()
    runtime._step_graph = None
    runtime._control_steps_since_cloth_bvh_rebuild = 0
    runtime._validate_action = lambda *args: None
    runtime._step_eager = lambda arm, gripper: events.append(("step", arm, gripper))
    runtime.step("arm-0", "gripper-0")
    runtime.step("arm-1", "gripper-1")
    assert [event[0] for event in events] == ["step", "step", "rebuild"]


def test_runtime_preserves_previous_rigid_state_until_cloth_step() -> None:
    events = []

    class State:
        def clear_forces(self):
            events.append("clear")

    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.state_0 = State()
    runtime.state_1 = State()
    runtime.rigid_contacts = object()
    runtime.control = object()
    runtime.rigid_collision = SimpleNamespace(
        collide=lambda state, contacts: events.append(("collide", state, contacts))
    )
    runtime.rigid = SimpleNamespace(
        step=lambda state_in, state_out, control, contacts, dt: events.append(
            ("step", state_in, state_out, control, contacts, dt)
        )
    )

    state_0 = runtime.state_0
    state_1 = runtime.state_1
    runtime._step_rigid(0.01)

    assert runtime.state_0 is state_0
    assert runtime.state_1 is state_1
    assert events[-1] == (
        "step",
        state_0,
        state_1,
        runtime.control,
        runtime.rigid_contacts,
        0.01,
    )


def test_cloth_collision_combines_current_particles_and_updated_rigid_state() -> None:
    current_particles = object()
    output_particles = object()
    events = []

    class State:
        def __init__(self, particle_q):
            self.particle_q = particle_q

        def clear_forces(self):
            pass

    state_0 = State(current_particles)
    state_1 = State(output_particles)
    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.state_0 = state_0
    runtime.state_1 = state_1
    runtime.contacts = object()
    runtime.grasp = SimpleNamespace(enforce=lambda *args: events.append(("enforce", *args)))

    def collide(state, contacts):
        events.append(("collide", state, state.particle_q, contacts))

    def cloth_step(state_in, state_out, control, contacts, dt):
        events.append(("cloth", state_in, state_out, state_out.particle_q, control, contacts, dt))

    runtime.collision = SimpleNamespace(collide=collide)
    runtime.cloth = SimpleNamespace(step=cloth_step)

    runtime._step_cloth(0.01)

    assert events[1] == ("collide", state_1, current_particles, runtime.contacts)
    assert events[2] == (
        "cloth",
        state_0,
        state_1,
        output_particles,
        None,
        runtime.contacts,
        0.01,
    )
    assert runtime.state_0 is state_1
    assert runtime.state_1 is state_0


def test_runtime_prunes_shapes_with_static_particle_collision_flags() -> None:
    import newton
    import warp as wp

    model = SimpleNamespace(
        device="cpu",
        shape_flags=wp.array(
            [int(newton.ShapeFlags.COLLIDE_PARTICLES), 0],
            dtype=wp.int32,
            device="cpu",
        ),
    )
    empty = wp.array(np.empty((0, 2), dtype=np.int32), dtype=wp.vec2i, device="cpu")
    pipeline = SimpleNamespace(
        soft_rigid_contact_pairs=wp.array([[0, 0], [0, 1]], dtype=wp.vec2i, device="cpu"),
        soft_edge_rigid_pairs=empty,
        soft_face_rigid_pairs=empty,
        soft_contact_max=2,
        _soft_rigid_contact_pair_count=2,
        _soft_contact_max=2,
    )

    _prune_static_soft_contact_pairs(model, pipeline)

    np.testing.assert_array_equal(
        pipeline.soft_rigid_contact_pairs.numpy(), np.asarray([[0, 0]], dtype=np.int32)
    )
    assert pipeline._soft_rigid_contact_pair_count == 1
    assert pipeline._soft_contact_max == 1


def test_scene_self_contact_defaults_match_expert_data() -> None:
    spec = ShirtFlattenSpec()

    assert spec.self_contact_max_candidates_per_primitive == 63
    assert spec.self_contact_friction_mu == 0.35
    assert spec.self_contact_friction_epsilon == 0.01
    assert spec.suppress_self_friction_during_untangling is True

    with pytest.raises(ValueError, match="max_candidates"):
        ShirtFlattenSpec(self_contact_max_candidates_per_primitive=0)
    with pytest.raises(ValueError, match="friction_mu"):
        ShirtFlattenSpec(self_contact_friction_mu=-0.1)
    with pytest.raises(ValueError, match="friction_epsilon"):
        ShirtFlattenSpec(self_contact_friction_epsilon=0.0)


def test_parallel_physics_world_offsets_are_centered_and_isolated() -> None:
    offsets = np.asarray(_centered_world_offsets(4), dtype=np.float32)

    np.testing.assert_allclose(offsets.mean(axis=0), 0.0)
    assert len(np.unique(offsets[:, :2], axis=0)) == 4
    distances = np.linalg.norm(offsets[:, None, :2] - offsets[None, :, :2], axis=-1)
    assert np.min(distances[distances > 0.0]) == pytest.approx(3.0)


def test_piper_asset_is_a_small_self_contained_usd_closure() -> None:
    root = Path(__file__).parents[1] / "assets" / "robots" / "piper"
    if not root.is_dir():
        pytest.skip("project-local assets are installed separately from Git")
    expected = {
        "piper.usda",
        "piper_dual_m.usda",
        "payloads/base.usda",
        "payloads/collisions.usdc",
        "payloads/geometries.usd",
        "payloads/installation.usdc",
        "payloads/installation_materials.usda",
        "payloads/instances.usda",
        "payloads/materials.usda",
        "payloads/robot.usda",
        "payloads/Physics/mujoco.usda",
        "payloads/Physics/physics.usda",
        "payloads/Physics/physx.usda",
    }

    assert {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()} == expected
    assert "@./piper_dual_m.usda@" in (root / "piper.usda").read_text(encoding="utf-8")


def test_piper_collision_layer_preserves_expert_geometry_provenance() -> None:
    from pxr import Usd

    root = Path(__file__).parents[1] / "assets" / "robots" / "piper"
    stage = Usd.Stage.Open(str(root / "piper.usda"))
    assert stage is not None
    collision_prims = [
        prim
        for prim in stage.Traverse()
        if "/ClothDojoLegacyCollision/" in str(prim.GetPath())
        and prim.HasAttribute("clothdojo:collisionRole")
    ]
    roles = Counter(
        str(prim.GetAttribute("clothdojo:collisionRole").Get()) for prim in collision_prims
    )
    assert roles == {"contact": 99, "self_collision_proxy": 27}
    assert sorted(
        int(prim.GetAttribute("clothdojo:sourceShapeIndex").Get()) for prim in collision_prims
    ) == list(range(126))
    # The composed-stage layer order is not stable across USD versions; read
    # the collision layer directly for its immutable migration signature.
    collision_layer = Usd.Stage.Open(str(root / "payloads" / "collisions.usdc"))
    assert collision_layer is not None
    metadata = collision_layer.GetRootLayer().customLayerData
    assert metadata["sourceGeometrySignatureSha256"] == (
        "b2902d562abab777a9087938ffa346dd1c27a0123c24656c326f8585b7cce1d4"
    )
    assert metadata["sourceNewtonVersion"] == "1.4.0"


@pytest.mark.parametrize("load_visual_shapes", [False, True])
def test_piper_import_selects_only_expert_collision_envelope(load_visual_shapes: bool) -> None:
    import newton
    import warp as wp

    root = Path(__file__).parents[1] / "assets" / "robots" / "piper"
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)

    _add_robot(
        newton,
        wp,
        builder,
        root / "piper.usda",
        robot_profile(PIPER),
        load_visual_shapes=load_visual_shapes,
    )

    labels = tuple(str(label) for label in builder.shape_label)
    migrated = [
        index for index, label in enumerate(labels) if "/ClothDojoLegacyCollision/" in label
    ]
    original = [
        index for index, label in enumerate(labels) if "/ClothDojoLegacyCollision/" not in label
    ]
    shape_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    contact = [index for index in migrated if "/contact_" in labels[index]]
    proxy = [index for index in migrated if "/self_collision_proxy_" in labels[index]]

    assert len(contact) == 99
    assert len(proxy) == 27
    assert Counter(int(builder.shape_type[index]) for index in migrated) == {7: 4, 8: 10, 10: 112}
    assert all(
        int(builder.shape_flags[index]) & (shape_bit | particle_bit) == shape_bit | particle_bit
        for index in contact
    )
    assert all(
        int(builder.shape_flags[index]) & (shape_bit | particle_bit) == shape_bit for index in proxy
    )
    assert all(not int(builder.shape_flags[index]) & shape_bit for index in original)
    assert sum(bool(int(flags) & shape_bit) for flags in builder.shape_flags) == 126
    assert sum(bool(int(flags) & particle_bit) for flags in builder.shape_flags) == 99
    np.testing.assert_allclose(
        np.asarray(builder.shape_material_mu)[migrated], 0.6, rtol=0.0, atol=1.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(builder.shape_material_ke)[migrated], 10000.0
    )
    np.testing.assert_allclose(
        np.asarray(builder.shape_material_kd)[migrated], 200.0
    )
    np.testing.assert_allclose(
        np.asarray(builder.shape_margin)[migrated], 0.0004, rtol=0.0, atol=1.0e-8
    )


def test_robot_cloth_ghost_mode_preserves_table_particle_collision() -> None:
    import newton
    import warp as wp

    root = Path(__file__).parents[1] / "assets" / "robots" / "piper"
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(
        newton,
        wp,
        builder,
        root / "piper.usda",
        robot_profile(PIPER),
        load_visual_shapes=False,
        robot_cloth_collision=False,
    )
    _add_table(newton, wp, builder, ShirtFlattenSpec())

    labels = tuple(str(label) for label in builder.shape_label)
    robot_shapes = [
        index for index, label in enumerate(labels) if "/ClothDojoLegacyCollision/" in label
    ]
    table_shapes = [index for index, label in enumerate(labels) if label == "table"]
    shape_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)

    assert len(robot_shapes) == 126
    assert len(table_shapes) == 1
    assert sum(bool(int(flags) & shape_bit) for flags in builder.shape_flags) == 127
    assert sum(bool(int(flags) & particle_bit) for flags in builder.shape_flags) == 1
    assert all(not int(builder.shape_flags[index]) & particle_bit for index in robot_shapes)
    assert int(builder.shape_flags[table_shapes[0]]) & particle_bit


def test_robot_rigid_ghost_mode_preserves_table_shape_collision() -> None:
    import newton
    import warp as wp

    root = Path(__file__).parents[1] / "assets" / "robots" / "aloha2"
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(
        newton,
        wp,
        builder,
        root / "aloha2.usda",
        robot_profile(ALOHA2),
        load_visual_shapes=False,
        robot_cloth_collision=False,
        robot_rigid_collision=False,
    )
    robot_shape_count = int(builder.shape_count)
    _add_table(newton, wp, builder, ShirtFlattenSpec())

    shape_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    assert all(
        not int(builder.shape_flags[index]) & (shape_bit | particle_bit)
        for index in range(robot_shape_count)
    )
    assert int(builder.shape_flags[robot_shape_count]) & shape_bit
    assert int(builder.shape_flags[robot_shape_count]) & particle_bit


def test_joint_selection_resolves_named_one_dof_joints() -> None:
    builder = SimpleNamespace(
        joint_label=["root", "/robot/left/joint1", "/robot/right/joint1"],
        joint_dof_dim=[(0, 0), (0, 1), (0, 1)],
    )

    joints = _joint_indices(builder, ("left/joint1", "right/joint1"))

    assert joints == [1, 2]
    assert _joint_dofs(builder, joints) == [0, 1]


def test_joint_selection_rejects_ambiguous_asset_labels() -> None:
    builder = SimpleNamespace(
        joint_label=["/a/left/joint1", "/b/left/joint1"],
        joint_dof_dim=[(0, 1), (0, 1)],
    )

    with pytest.raises(ValueError, match="expected one Piper joint"):
        _joint_indices(builder, ("left/joint1",))


def test_end_effector_selection_preserves_left_right_order() -> None:
    builder = SimpleNamespace(
        body_label=[
            "/robot/left/base/link6",
            "/robot/right/base/link6",
        ]
    )

    assert _body_indices(builder, ("left/link6", "right/link6")) == [0, 1]


def test_piper_usd_authors_control_and_gripper_physics() -> None:
    import newton
    import warp as wp

    root = Path(__file__).parents[1] / "assets" / "robots" / "piper"
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(
        newton,
        wp,
        builder,
        root / "piper.usda",
        robot_profile(PIPER),
        load_visual_shapes=False,
    )
    arm_joints, gripper_joints, arm_dofs = _select_piper_robot(builder)
    gripper_dofs = _joint_dofs(builder, gripper_joints)

    assert arm_joints == _joint_indices(builder, ARM_JOINTS)
    assert gripper_joints == _joint_indices(builder, GRIPPER_JOINTS)
    np.testing.assert_allclose(
        np.asarray(builder.joint_limit_lower)[arm_dofs][[0, 6]], -0.5 * np.pi
    )
    np.testing.assert_allclose(
        np.asarray(builder.joint_limit_upper)[arm_dofs][[0, 6]], 0.5 * np.pi
    )
    np.testing.assert_allclose(
        np.asarray(builder.joint_q)[gripper_dofs], (0.04, -0.04, 0.04, -0.04)
    )
    np.testing.assert_allclose(
        np.asarray(builder.joint_target_q)[gripper_dofs], (0.04, -0.04, 0.04, -0.04)
    )
    assert np.asarray(builder.joint_target_mode)[gripper_dofs].tolist() == [
        int(newton.JointTargetMode.POSITION),
        int(newton.JointTargetMode.VELOCITY),
        int(newton.JointTargetMode.POSITION),
        int(newton.JointTargetMode.VELOCITY),
    ]
    np.testing.assert_allclose(
        np.asarray(builder.joint_target_ke)[arm_dofs],
        (31.0035, 45.6903, 28.8618, 17.8393, 18.7203, 17.8206) * 2,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(builder.joint_target_kd)[arm_dofs],
        (5.58063, 8.22425, 5.19513, 3.21108, 3.36966, 3.20771) * 2,
        rtol=1.0e-6,
    )
    assert builder.constraint_mimic_joint0 == [gripper_joints[1], gripper_joints[3]]
    assert builder.constraint_mimic_joint1 == [gripper_joints[0], gripper_joints[2]]
    assert builder.constraint_mimic_coef0 == [0.0, 0.0]
    assert builder.constraint_mimic_coef1 == [-1.0, -1.0]
    assert set(builder.custom_attributes["mujoco:gravcomp"].values) == set(range(21))
    assert set(builder.custom_attributes["mujoco:jnt_actgravcomp"].values) == set(arm_dofs)


def test_scene_requires_at_least_one_world() -> None:
    with pytest.raises(ValueError, match="world_count"):
        ShirtFlattenSpec(world_count=0)


def test_visual_scene_uses_one_noncolliding_model_for_every_world() -> None:
    root = Path(__file__).parents[1] / "assets"
    garment = root / "garments" / "converted" / ShirtFlattenSpec().garment_id
    if not garment.is_dir():
        pytest.skip("project-local converted garments are installed separately from Git")

    import newton

    scene = build_scene(
        root,
        spec=ShirtFlattenSpec(world_count=2),
        device="cpu",
        visuals=True,
    )
    visual_shapes = [
        index
        for index, label in enumerate(scene.model.shape_label)
        if str(label).endswith("garment_visual")
    ]
    collision_bits = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(
        newton.ShapeFlags.COLLIDE_PARTICLES
    )
    flags = scene.model.shape_flags.numpy()

    assert scene.model.world_count == 2
    assert len(visual_shapes) == 2
    assert all(not int(flags[index]) & collision_bits for index in visual_shapes)


def test_gripper_openness_uses_one_leader_joint_per_side() -> None:
    q = np.asarray([[0.02, -0.01, 0.04, -0.03]], dtype=np.float32)
    lower = np.asarray([[0.0, -0.04, 0.0, -0.04]], dtype=np.float32)
    upper = np.asarray([[0.04, 0.0, 0.04, 0.0]], dtype=np.float32)

    openness = _gripper_openness(q, lower, upper)

    np.testing.assert_allclose(openness, [[0.5, 1.0]])


def test_vertex_normals_are_area_weighted_and_normalized() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    normals = _vertex_normals(vertices, triangles)

    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (4, 1)))


def test_nocs_matches_expert_data_encoding_and_reserves_black() -> None:
    vertices = np.asarray(
        [[-1.0, 2.0, 7.0], [0.0, 4.0, 7.0], [1.0, 6.0, 7.0]],
        dtype=np.float32,
    )

    colors = rest_vertices_to_nocs(vertices)

    np.testing.assert_array_equal(
        np.rint(colors * 255.0).astype(np.uint8),
        [[1, 1, 1], [128, 128, 1], [255, 255, 1]],
    )


def test_nocs_visual_mesh_uses_face_varying_public_texture_atlas() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    visual, indices, normals, uvs, texture = _nocs_visual_mesh(
        vertices,
        triangles,
        vertices,
    )

    np.testing.assert_array_equal(visual, vertices[triangles.reshape(-1)])
    np.testing.assert_array_equal(indices, np.arange(6, dtype=np.int32))
    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (6, 1)))
    assert uvs.shape == (6, 2)
    assert texture.dtype == np.uint8
    assert texture.shape == (8, 16, 4)
    expected = np.rint(rest_vertices_to_nocs(vertices)[triangles.reshape(-1)] * 255).astype(
        np.uint8
    )
    pixel_x = np.floor(uvs[:, 0] * texture.shape[1]).astype(int)
    pixel_y = np.floor((1.0 - uvs[:, 1]) * texture.shape[0]).astype(int)
    np.testing.assert_array_equal(texture[pixel_y, pixel_x, :3], expected)
