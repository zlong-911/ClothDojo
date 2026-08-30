import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).parents[1] / "assets/robots/ur5e_2f85"


def _transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


@pytest.fixture
def stage():
    from pxr import Usd

    return Usd.Stage.Open(str(ROOT / "ur5e_2f85.usda"))


def test_ur5_has_exact_source_camera_geometry_and_hidden_colliders(stage):
    from pxr import UsdGeom, UsdPhysics

    for wrist in (6, 27):
        root = f"/Robot/Bodies/body_{wrist:03}/WristCameraAssembly"
        for suffix, triangles in (
            ("AdapterVisual", 1034),
            ("D415/Visual", 425160),
            ("AdapterCollision", 186),
        ):
            prim = stage.GetPrimAtPath(f"{root}/{suffix}")
            assert sum(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()) == triangles * 3
        for suffix in ("AdapterVisual", "D415/Visual"):
            prim = stage.GetPrimAtPath(f"{root}/{suffix}")
            assert prim.GetAttribute("clothdojo:shapeFlags").Get() == 1
            assert not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        for suffix in ("AdapterCollision", "D415/Collision"):
            prim = stage.GetPrimAtPath(f"{root}/{suffix}")
            assert prim.GetAttribute("clothdojo:shapeFlags").Get() == 6
            assert UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"
            assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == 0
        for name, expected in (
            ("white_material", (0.8, 0.8, 0.8)),
            ("green_material", (0, 0.8, 0.3)),
        ):
            shader = stage.GetPrimAtPath(f"{root}/Materials/{name}/PreviewSurface")
            np.testing.assert_allclose(shader.GetAttribute("inputs:diffuseColor").Get(), expected)


def test_ur5_coupling_and_attach_shift_only_seven_mm(stage):
    from pxr import Usd, UsdGeom

    before = Usd.Stage.Open(str(ROOT / "payloads/workstation.usdc"))
    cache = UsdGeom.XformCache()
    flange = _transform((0, 0.1, 0), (-math.pi / 2, 0, 0))
    for side, wrist in (("left", 6), ("right", 27)):
        wrist_prim = before.GetPrimAtPath(f"/Robot/Bodies/body_{wrist:03}")
        wrist_world = np.asarray(cache.GetLocalToWorldTransform(wrist_prim)).T
        expected_shift = (wrist_world @ flange)[:3, 2] * 0.007
        count = 0
        for prim in before.GetPrimAtPath("/Robot/Bodies").GetChildren():
            label = str(prim.GetAttribute("clothdojo:bodyLabel").Get())
            if not label.startswith(side + "/"):
                continue
            old = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            new = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prim.GetPath()))).T
            np.testing.assert_allclose(new[:3, :3], old[:3, :3], atol=1e-12)
            shift = expected_shift if label.startswith(side + "/gripper_") else np.zeros(3)
            np.testing.assert_allclose(new[:3, 3] - old[:3, 3], shift, atol=1e-12)
            count += label.startswith(side + "/gripper_")
        assert count == 14
        base = 8 if side == "left" else 29
        attach_path = f"/Robot/Bodies/body_{base:03}/Semantics/attach_{side}"
        old_attach = np.asarray(cache.GetLocalToWorldTransform(before.GetPrimAtPath(attach_path))).T
        new_attach = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(attach_path))).T
        np.testing.assert_allclose(
            new_attach[:3, 3] - old_attach[:3, 3], expected_shift, atol=1e-12
        )
        np.testing.assert_array_equal(new_attach[:3, :3], old_attach[:3, :3])
    for name in ("cam_high", "cam_low"):
        path = "/Robot/Semantics/" + name
        np.testing.assert_array_equal(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)),
            cache.GetLocalToWorldTransform(before.GetPrimAtPath(path)),
        )


def test_ur5_camera_pose_and_lens_follow_published_d415_chain(stage):
    from pxr import UsdGeom

    flange = _transform((0, 0.1, 0), (-math.pi / 2, 0, 0))
    camera_from_flange = (
        _transform((0, -0.067, 0.0171), (0, -math.pi / 2 + math.pi / 30, math.pi / 2))
        @ _transform((0, 0.02, 0.0115))
        @ _transform((0, 0.015, 0))
        @ _transform(rpy=(-math.pi / 2, 0, -math.pi / 2))
        @ _transform(rpy=(math.pi, 0, 0))
    )
    camera_source = ET.parse(ROOT / "provenance/wrist_camera/picknik_ur.xacro").find(
        ".//sensor[@name='wrist_mounted_camera_rgbd_sensor']/camera"
    )
    source_fov = float(camera_source.findtext("horizontal_fov"))
    cache = UsdGeom.XformCache()
    for side, wrist, base in (("left", 6, 8), ("right", 27, 29)):
        wrist_world = np.asarray(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/Robot/Bodies/body_{wrist:03}"))
        ).T
        camera = UsdGeom.Camera(
            stage.GetPrimAtPath(f"/Robot/Bodies/body_{base:03}/cam_{side}_wrist")
        )
        actual = np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T
        np.testing.assert_allclose(actual, wrist_world @ flange @ camera_from_flange, atol=1e-12)
        f = camera.GetFocalLengthAttr().Get()
        h = camera.GetHorizontalApertureAttr().Get()
        v = camera.GetVerticalApertureAttr().Get()
        assert 2 * math.atan(h / (2 * f)) == pytest.approx(source_fov, abs=1e-7)
        assert v / h == pytest.approx(480 / 640)


def test_ur5_accessories_do_not_change_body_mass_or_control_layout(stage):
    import newton
    from pxr import Sdf, Usd

    from clothdojo.sim.scene import _register_solver_attributes

    newton.use_coord_layout_targets = True

    # Same authoritative body parameters with the accessory references removed:
    # this isolates any importer-side mesh-density contribution.
    stripped = Sdf.Layer.CreateAnonymous()
    stripped.TransferContent(stage.GetRootLayer())
    stripped.subLayerPaths = [str(ROOT / "payloads/workstation.usdc")]
    edits = Sdf.BatchNamespaceEdit()
    for wrist in (6, 27):
        edits.Add(f"/Robot/Bodies/body_{wrist:03}/WristCameraAssembly", Sdf.Path.emptyPath)
    assert stripped.Apply(edits)
    without = Usd.Stage.Open(stripped)
    models = []
    for source in (without, stage):
        builder = newton.ModelBuilder()
        _register_solver_attributes(newton, builder)
        builder.add_usd(
            source,
            floating=False,
            load_visual_shapes=False,
            load_static_visual_shapes=False,
            skip_mesh_approximation=True,
        )
        assert (builder.body_count, builder.joint_coord_count, builder.joint_dof_count) == (
            42,
            44,
            40,
        )
        models.append(builder.finalize("cpu"))
    for field in (
        "body_mass",
        "body_com",
        "body_inertia",
        "joint_q",
        "joint_target_ke",
        "joint_target_kd",
        "joint_effort_limit",
    ):
        np.testing.assert_array_equal(
            getattr(models[0], field).numpy(), getattr(models[1], field).numpy()
        )
