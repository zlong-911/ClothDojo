"""FR3 uses the accepted UR5e gripper, without moving VMKit or the TCP."""
from pathlib import Path

import numpy as np
import pytest

from clothdojo.robots import robot_profile
from clothdojo.sim.scene import _register_solver_attributes

ASSETS = Path(__file__).parents[1] / "assets/robots"


@pytest.fixture(scope="module")
def models():
    import newton

    result = {}
    for robot in ("fr3_duo", "ur5e_2f85"):
        builder = newton.ModelBuilder()
        _register_solver_attributes(newton, builder)
        imported = builder.add_usd(
            str(ASSETS / robot / f"{robot}.usda"), floating=False,
            load_visual_shapes=False, skip_mesh_approximation=True,
            schema_resolvers=[newton.usd.SchemaResolverNewton(), newton.usd.SchemaResolverPhysx()],
        )
        result[robot] = builder, imported
    return result


def test_fr3_gripper_dynamics_and_closed_loops_match_ur5(models):
    import newton

    target, target_map = models["fr3_duo"]
    source, source_map = models["ur5e_2f85"]
    assert not target.constraint_mimic_joint0
    assert target.joint_type.count(newton.JointType.BALL) == 4
    fields = (
        "joint_target_ke", "joint_target_kd", "joint_effort_limit", "joint_armature",
        "joint_damping", "joint_friction", "joint_limit_lower", "joint_limit_upper",
        "joint_limit_ke", "joint_limit_kd", "joint_velocity_limit",
    )
    for side in ("left", "right"):
        for i in range(14):
            tj = target_map["path_joint_map"][f"/Robot/Duo/Robotiq/{side}/Joints/joint_{i:02d}"]
            sj = source_map["path_joint_map"][f"/Robot/Joints/joint_{i + 9:03d}"]
            assert target.joint_type[tj] == source.joint_type[sj]
            if target.joint_type[tj] != newton.JointType.REVOLUTE:
                continue
            td, sd = target.joint_qd_start[tj], source.joint_qd_start[sj]
            for field in fields:
                assert getattr(target, field)[td] == pytest.approx(getattr(source, field)[sd])
            for field in ("mujoco:dof_passive_stiffness", "mujoco:dof_springref"):
                tv = target.custom_attributes[field].values.get(td)
                sv = source.custom_attributes[field].values.get(sd)
                assert (tv.value if tv else None) == (sv.value if sv else None)


def test_fr3_gripper_inertia_and_geometry_are_copied_from_ur5(models):
    from pxr import Usd

    target, target_map = models["fr3_duo"]
    source, source_map = models["ur5e_2f85"]
    donor = Usd.Stage.Open(str(ASSETS / "ur5e_2f85/ur5e_2f85.usda"))
    stage = Usd.Stage.Open(str(ASSETS / "fr3_duo/fr3_duo.usda"))
    for i in range(8, 21):
        source_path = f"/Robot/Bodies/body_{i:03d}"
        prim = donor.GetPrimAtPath(source_path)
        name = prim.GetAttribute("clothdojo:bodyLabel").Get().split("/")[-1]
        sb = source_map["path_body_map"][source_path]
        for side in ("left", "right"):
            target_path = f"/Robot/Duo/Robotiq/{side}/Bodies/{name}"
            tb = target_map["path_body_map"][target_path]
            for field in ("body_mass", "body_com", "body_inertia"):
                np.testing.assert_allclose(
                    getattr(target, field)[tb], getattr(source, field)[sb], atol=3e-7, rtol=1e-5,
                )
            for shape in Usd.PrimRange(prim.GetChild("Geometry")):
                relative = shape.GetPath().MakeRelativePath(prim.GetPath())
                copied = stage.GetPrimAtPath(f"{target_path}/{relative}")
                assert copied and copied.GetTypeName() == shape.GetTypeName()
                for key in ("points", "faceVertexIndices", "faceVertexCounts",
                            "clothdojo:shapeFlags", "clothdojo:material"):
                    attr = shape.GetAttribute(key)
                    if attr and attr.HasAuthoredValue():
                        assert copied.GetAttribute(key).Get() == attr.Get()


def test_fr3_camera_attach_mount_and_arm_poses_unchanged():
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    path = ASSETS / "fr3_duo/fr3_duo.usda"
    stage = Usd.Stage.Open(str(path))
    baseline_layer = Sdf.Layer.CreateAnonymous()
    baseline_layer.TransferContent(stage.GetRootLayer())
    baseline_layer.subLayerPaths = [
        str((path.parent / layer).resolve()) for layer in stage.GetRootLayer().subLayerPaths
        if "robotiq_installation" not in layer
    ]
    baseline = Usd.Stage.Open(baseline_layer)
    old_cache, new_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
    checked = 0
    for old in baseline.Traverse():
        current = stage.GetPrimAtPath(old.GetPath())
        if not current or not current.IsActive() or not old.IsA(UsdGeom.Xformable):
            continue
        np.testing.assert_array_equal(
            old_cache.GetLocalToWorldTransform(old), new_cache.GetLocalToWorldTransform(current),
        )
        if old.HasAPI(UsdPhysics.RigidBodyAPI):
            for attr in old.GetAttributes():
                if attr.GetName().startswith("physics:") and attr.HasAuthoredValue():
                    assert current.GetAttribute(attr.GetName()).Get() == attr.Get()
        checked += 1
    assert checked > 100
    profile = robot_profile("fr3_duo")
    assert tuple(frame.parent_body for frame in profile.attach_frames) == ("left/tcp", "right/tcp")


def test_fr3_smooth_gripper_cycle_has_stable_passive_links(models):
    import newton
    import warp as wp
    newton.use_coord_layout_targets = True
    builder, _ = models["fr3_duo"]
    model = builder.finalize(device="cpu")
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    solver = newton.solvers.SolverMuJoCo(
        model, use_mujoco_cpu=True, use_mujoco_contacts=True, disable_contacts=True,
        disable_sensors=True, integrator="implicitfast", solver="newton",
        iterations=100, ls_iterations=50,
    )
    profile = robot_profile("fr3_duo")
    home = model.joint_q.numpy().copy()
    controlled = (*profile.arm_coord_indices, *profile.gripper_coord_indices)
    passive = [builder.joint_qd_start[j] for j, kind in enumerate(builder.joint_type)
               if kind == newton.JointType.REVOLUTE
               and builder.joint_q_start[j] not in controlled]
    loops = [j for j, kind in enumerate(builder.joint_type) if kind == newton.JointType.BALL]
    passive_motion = np.zeros(len(passive))
    max_passive_speed = 0.
    for frame in range(250):
        t = frame / 50
        fraction = np.clip(t - 1, 0, 1) if t < 3 else np.clip(4 - t, 0, 1)
        fraction *= fraction * (3 - 2 * fraction)
        target = home.copy()
        target[list(profile.gripper_coord_indices)] = .8 * fraction
        control.joint_target_q.assign(target)
        for _ in range(6):
            state.clear_forces()
            next_state.clear_forces()
            solver.step(state, next_state, control, None, 1 / 300)
            state, next_state = next_state, state
        speed = np.abs(state.joint_qd.numpy()[passive])
        assert np.all(np.isfinite(speed))
        max_passive_speed = max(max_passive_speed, float(speed.max()))
        passive_motion = np.maximum(passive_motion, speed)
        if frame == 149:
            np.testing.assert_allclose(
                state.joint_q.numpy()[list(profile.gripper_coord_indices)], .8, atol=.01,
            )
        poses = state.body_q.numpy()
        for j in loops:
            anchors = [
                np.asarray(wp.transform_point(
                    wp.transform(*poses[body]), wp.transform_get_translation(local),
                ))
                for body, local in ((builder.joint_parent[j], builder.joint_X_p[j]),
                                    (builder.joint_child[j], builder.joint_X_c[j]))
            ]
            assert np.linalg.norm(anchors[0] - anchors[1]) < .001
    assert max_passive_speed < 2.0
    assert np.all(passive_motion > .001)
    np.testing.assert_allclose(
        state.joint_q.numpy()[list(profile.gripper_coord_indices)], 0, atol=.01,
    )
