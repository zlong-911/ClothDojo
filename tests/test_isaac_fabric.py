from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clothdojo.data.isaac_render import _author_cameras
from clothdojo.robots import robot_ids, robot_profile
from clothdojo.teleop.isaac_fabric import (
    _fabric_gpu_direct_supported,
    _find_render_body_paths,
    _UsdRigidBodyWriter,
)
from clothdojo.teleop.isaac_stage import _author_robot


def _label(prim, name):
    from pxr import Sdf

    prim.CreateAttribute("clothdojo:bodyLabel", Sdf.ValueTypeNames.String).Set(name)


def _world_matrices(stage):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache()
    return {
        str(prim.GetPath()): np.asarray(cache.GetLocalToWorldTransform(prim))
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.IsA(UsdGeom.Xformable)
    }


def test_fabric_gpu_direct_path_is_limited_to_cuda_zero():
    assert _fabric_gpu_direct_supported("cuda:0")
    assert not _fabric_gpu_direct_supported("cuda:1")
    assert not _fabric_gpu_direct_supported("cuda:7")


@pytest.mark.parametrize("robot_id", robot_ids())
def test_real_robot_poses_visuals_and_cameras_follow_newton_fk(robot_id):
    import newton
    import warp as wp
    from pxr import Gf, Usd, UsdGeom

    from clothdojo.sim.scene import _add_robot, _register_solver_attributes

    assets = Path(__file__).parents[1] / "assets"
    profile = robot_profile(robot_id)
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(newton, wp, builder, assets / profile.asset, profile, load_visual_shapes=False)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    _author_robot(stage, assets, profile)
    cameras = _author_cameras(stage, profile.body_names, profile.cameras)
    paths = _find_render_body_paths(stage, builder.body_label)
    assert len(paths) == len(profile.body_names)
    assert len(set(paths.values())) == len(paths)
    for index, path in paths.items():
        label = stage.GetPrimAtPath(path).GetAttribute("clothdojo:bodyLabel")
        if label:
            assert label.Get() == profile.body_names[index]

    before = _world_matrices(stage)
    resets = {
        path: UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetResetXformStack()
        for path in before
    }
    writer = _UsdRigidBodyWriter(stage, builder.body_label)
    for path, value in _world_matrices(stage).items():
        np.testing.assert_allclose(value, before[path], atol=1e-12, rtol=0)

    # Every visual/instance/camera follows its nearest USD body unless a reset
    # stack cuts inheritance. Static installation nodes must remain untouched.
    owners = {}
    body_paths = set(paths.values())
    for path in before:
        prim = stage.GetPrimAtPath(path)
        while prim and not prim.IsPseudoRoot():
            candidate = str(prim.GetPath())
            if candidate in body_paths:
                owners[path] = candidate
                break
            xform = UsdGeom.Xformable(prim)
            if xform and xform.GetResetXformStack():
                break
            prim = prim.GetParent()

    model = builder.finalize(device="cpu")
    state = model.state()
    for offset in (0.0, 0.12, -0.08):
        coordinates = np.asarray(profile.home_joint_q, dtype=np.float32).copy()
        coordinates[np.asarray(profile.arm_coord_indices)] += offset
        newton.eval_fk(
            model, wp.array(coordinates, dtype=wp.float32, device="cpu"), model.joint_qd, state
        )
        poses = state.body_q.numpy()
        writer.update(poses)
        actual = _world_matrices(stage)
        for index, path in paths.items():
            np.testing.assert_allclose(actual[path][3, :3], poses[index, :3], atol=1e-12)
            rotation = Gf.Transform(Gf.Matrix4d(actual[path])).GetRotation().GetQuat()
            quaternion = np.asarray((*rotation.GetImaginary(), rotation.GetReal()))
            expected = poses[index, 3:7].astype(float)
            expected /= np.linalg.norm(expected)
            assert abs(float(quaternion @ expected)) == pytest.approx(1.0, abs=1e-10)
            # A rigid pose update must not stretch or shrink the body geometry.
            np.testing.assert_allclose(
                actual[path][:3, :3] @ actual[path][:3, :3].T,
                before[path][:3, :3] @ before[path][:3, :3].T,
                atol=1e-10, rtol=0,
            )
        for path, matrix in actual.items():
            owner = owners.get(path)
            expected = before[path]
            if owner is not None:
                expected = before[path] @ np.linalg.inv(before[owner]) @ actual[owner]
            np.testing.assert_allclose(matrix, expected, atol=1e-10, rtol=0, err_msg=path)
            assert UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetResetXformStack() == resets[path]
        if offset == 0:
            for path in cameras:
                np.testing.assert_allclose(actual[path], before[path], atol=2e-5, rtol=0)


@pytest.mark.parametrize("reset", [False, True])
def test_scaled_parent_and_intermediate_group_preserve_body_and_child_geometry(reset):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Robot")
    root.AddTranslateOp().Set((3, 4, 5))
    root.AddScaleOp().Set((0.01, 0.01, 0.01))
    parent = UsdGeom.Xform.Define(stage, "/World/Robot/body_000")
    _label(parent.GetPrim(), "left/base")
    parent.AddScaleOp().Set((100, 100, 100))
    group = UsdGeom.Xform.Define(stage, "/World/Robot/body_000/mount")
    group.AddTranslateOp().Set((0.1, 0.2, 0.3))
    child = UsdGeom.Xform.Define(stage, "/World/Robot/body_000/mount/body_001")
    _label(child.GetPrim(), "left/tool")
    child.AddScaleOp().Set((2, 3, 4))
    child.SetResetXformStack(reset)
    camera = UsdGeom.Camera.Define(stage, f"{child.GetPath()}/camera")
    UsdGeom.Xformable(camera).AddTranslateOp().Set((0.1, 0.2, 0.3))
    # USD parent ordering must work independently of supplied body indices.
    writer = _UsdRigidBodyWriter(stage, ("left/tool", "left/base"))
    sine = np.sqrt(0.5)
    poses = np.asarray(((1, 2, 3, 0, 0, sine, sine), (4, 5, 6, 0, 0, 0, 1)), dtype=float)
    for _ in range(2):
        writer.update(poses)
        cache = UsdGeom.XformCache()
        expected = ((0, 2, 0, 0), (-3, 0, 0, 0), (0, 0, 4, 0), (1, 2, 3, 1))
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(child.GetPrim()), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(camera.GetPrim()).ExtractTranslation(),
            (0.4, 2.2, 4.2), atol=1e-12,
        )
        assert child.GetResetXformStack() is reset
        # Episode rebinding at an already replayed pose must not accumulate scale.
        writer = _UsdRigidBodyWriter(stage, ("left/tool", "left/base"))


def test_body_labels_resolve_same_named_prims_and_reject_duplicates():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    for side in ("left", "right"):
        prim = UsdGeom.Xform.Define(stage, f"/World/Robot/arms/{side}/body_000").GetPrim()
        _label(prim, f"{side}/base")
    assert _find_render_body_paths(stage, ("right/base", "left/base")) == {
        0: "/World/Robot/arms/right/body_000", 1: "/World/Robot/arms/left/body_000"
    }
    duplicate = UsdGeom.Xform.Define(stage, "/World/Robot/duplicate").GetPrim()
    _label(duplicate, "left/base")
    with pytest.raises(RuntimeError, match="duplicate Isaac rigid-body label"):
        _UsdRigidBodyWriter(stage, ("left/base", "right/base"))
