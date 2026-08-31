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


def test_ur5_camera_adapter_reuses_arm_black_material(stage):
    from pxr import UsdShade

    source = UsdShade.Material(
        stage.GetPrimAtPath("/Robot/Materials/SourceMJCF/Material_source_00_black")
    )
    source_shader = source.GetSurfaceOutput().GetConnectedSource()[0]
    for wrist in (6, 27):
        root = f"/Robot/Bodies/body_{wrist:03}/WristCameraAssembly"
        actual, _ = UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath(f"{root}/AdapterVisual")
        ).ComputeBoundMaterial()
        assert str(actual.GetPath()) == f"{root}/Materials/UR5Black"
        actual_shader = actual.GetSurfaceOutput().GetConnectedSource()[0]
        assert str(actual_shader.GetPrim().GetPath()).startswith(str(actual.GetPath()))
        for value in source_shader.GetInputs():
            assert actual_shader.GetInput(value.GetBaseName()).Get() == value.Get()


def test_d415_material_partition_and_uv_preserve_original_mesh(stage):
    import trimesh
    from pxr import UsdGeom, UsdShade

    source = trimesh.load_mesh(ROOT / "provenance/wrist_camera/d415.stl", process=False)
    for wrist in (6, 27):
        root = f"/Robot/Bodies/body_{wrist:03}/WristCameraAssembly"
        visual = UsdGeom.Mesh(stage.GetPrimAtPath(f"{root}/D415/Visual"))
        np.testing.assert_array_equal(
            visual.GetPointsAttr().Get(), source.vertices.astype(np.float32)
        )
        np.testing.assert_array_equal(visual.GetFaceVertexIndicesAttr().Get(), source.faces.ravel())
        groups = UsdShade.MaterialBindingAPI(visual.GetPrim()).GetMaterialBindSubsets()
        assert {group.GetPrim().GetName() for group in groups} == {
            "Housing", "Interior", "FrontCover",
        }
        indices = np.concatenate([np.asarray(group.GetIndicesAttr().Get()) for group in groups])
        np.testing.assert_array_equal(np.sort(indices), np.arange(len(source.faces)))
        cover = next(group for group in groups if group.GetPrim().GetName() == "FrontCover")
        assert len(cover.GetIndicesAttr().Get()) == 1592
        uv = np.asarray(UsdGeom.PrimvarsAPI(visual).GetPrimvar("st").Get())
        cover_vertices = source.faces[np.asarray(cover.GetIndicesAttr().Get())].ravel()
        assert np.isfinite(uv).all()
        assert uv[cover_vertices].min() >= -1e-6
        assert uv[cover_vertices].max() <= 1 + 1e-6
        texture = stage.GetPrimAtPath(f"{root}/Materials/D415FrontCover/Texture")
        assert Path(texture.GetAttribute("inputs:file").Get().resolvedPath).is_file()
        assert texture.GetAttribute("inputs:sourceColorSpace").Get() == "sRGB"


def test_d415_bake_uses_original_three_windows_and_projector():
    from PIL import Image

    with Image.open(ROOT / "components/d415_front.png") as image:
        assert image.size == (1536, 264)

        def pixel(x_mm, y_mm):
            return image.getpixel((round((x_mm + 45.9) / 91.8 * 1535),
                                   round((7.9 - y_mm) / 15.8 * 263)))

        for x in (-35, 20, 35, -8):
            assert pixel(x, 0) == (56, 63, 69)
        for x, y in ((0, 0), (-32, 0), (-8, 3), (35, 3)):
            assert pixel(x, y) == (28, 30, 33)


def test_d415_texture_is_rgb_albedo_with_unmodified_srgb_import(stage):
    from newton._src.usd.utils import resolve_material_properties_for_prim
    from PIL import Image
    from pxr import UsdShade

    expected = ROOT / "components/d415_front.png"
    with Image.open(expected) as image:
        assert image.mode == "RGB"  # No alpha or packed roughness/metalness data.
    for wrist in (6, 27):
        root = f"/Robot/Bodies/body_{wrist:03}/WristCameraAssembly"
        subset = stage.GetPrimAtPath(f"{root}/D415/Visual/FrontCover")
        material, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
        shader, _, _ = material.ComputeSurfaceSource()
        texture, channel, _ = shader.GetInput("diffuseColor").GetConnectedSource()
        assert channel == "rgb"
        assert texture.GetPrim().GetAttribute("info:id").Get() == "UsdUVTexture"
        assert texture.GetInput("sourceColorSpace").Get() == "sRGB"
        uv, channel, _ = texture.GetInput("st").GetConnectedSource()
        assert channel == "result" and uv.GetInput("varname").Get() == "st"
        for name in ("normal", "roughness", "metallic", "opacity"):
            attribute = shader.GetInput(name)
            assert not attribute or not attribute.HasConnectedSource()
        imported = resolve_material_properties_for_prim(subset)
        assert Path(imported["texture"]).resolve() == expected.resolve()
        assert imported["color"] is None  # No extra diffuse multiplier over RGB.
        assert imported["roughness"] == pytest.approx(.18)


def test_ur5_tool_half_turn_preserves_attach_world_frame_and_seven_mm_offset(stage):
    from pxr import Usd, UsdGeom

    before = Usd.Stage.Open(str(ROOT / "payloads/workstation.usdc"))
    cache = UsdGeom.XformCache()
    flange = _transform((0, 0.1, 0), (-math.pi / 2, 0, 0))
    for side, wrist in (("left", 6), ("right", 27)):
        wrist_prim = before.GetPrimAtPath(f"/Robot/Bodies/body_{wrist:03}")
        wrist_world = np.asarray(cache.GetLocalToWorldTransform(wrist_prim)).T
        expected_shift = (wrist_world @ flange)[:3, 2] * 0.007
        tool = wrist_world @ flange
        half_turn = tool @ _transform(rpy=(0, 0, math.pi)) @ np.linalg.inv(tool)
        count = 0
        for prim in before.GetPrimAtPath("/Robot/Bodies").GetChildren():
            label = str(prim.GetAttribute("clothdojo:bodyLabel").Get())
            if not label.startswith(side + "/"):
                continue
            old = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            new = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prim.GetPath()))).T
            # Compare the tool change in the original installation frame.
            new = new.copy()
            new[0, 3] += 0.1
            expected = old.copy()
            if label.startswith(side + "/gripper_"):
                expected[:3, 3] += expected_shift
            if label.startswith(side + "/gripper_") or label == side + "/wrist_3_link":
                expected = half_turn @ expected
            # The source matrices came from float32 FK with non-unit round-off.
            # The new default is re-baked from joint frames, not matrix-edited.
            np.testing.assert_allclose(new[:3, 3], expected[:3, 3], atol=2e-6, rtol=0)
            relative = Rotation.from_matrix(old[:3, :3]).inv() * Rotation.from_matrix(
                new[:3, :3]
            )
            if label.startswith(side + "/gripper_") or label == side + "/wrist_3_link":
                assert relative.magnitude() == pytest.approx(math.pi, abs=1e-5)
            else:
                np.testing.assert_allclose(new, old, atol=1e-12, rtol=0)
            count += label.startswith(side + "/gripper_")
        assert count == 14
        base = 8 if side == "left" else 29
        attach_path = f"/Robot/Bodies/body_{base:03}/Semantics/attach_{side}"
        old_attach = np.asarray(cache.GetLocalToWorldTransform(before.GetPrimAtPath(attach_path))).T
        new_attach = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(attach_path))).T
        np.testing.assert_allclose(
            new_attach[:3, 3] - old_attach[:3, 3], expected_shift + np.array((-0.1, 0, 0)),
            atol=2e-6, rtol=0
        )
        # Both half-turns cancel in world space, but the physical tool/camera turns.
        relative = Rotation.from_matrix(old_attach[:3, :3]).inv() * Rotation.from_matrix(
            new_attach[:3, :3]
        )
        assert relative.magnitude() < 1e-5
        old_local = np.asarray(
            UsdGeom.Xformable(before.GetPrimAtPath(attach_path)).GetLocalTransformation()
        ).T
        new_local = np.asarray(
            UsdGeom.Xformable(stage.GetPrimAtPath(attach_path)).GetLocalTransformation()
        ).T
        np.testing.assert_allclose(
            new_local, old_local @ _transform(rpy=(0, 0, math.pi)), atol=1e-12
        )
    for name in ("cam_high", "cam_low"):
        path = "/Robot/Semantics/" + name
        expected = np.array(cache.GetLocalToWorldTransform(before.GetPrimAtPath(path)))
        if name == "cam_high":
            expected[3, 0] -= .1
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)), expected,
            atol=1e-12, rtol=0,
        )


def test_ur5_half_turn_is_fixed_in_home_state_and_drive_targets(stage):
    from pxr import Usd

    from clothdojo.robots import robot_profile

    before = Usd.Stage.Open(str(ROOT / "payloads/workstation.usdc"))
    expected = np.array(before.GetDefaultPrim().GetAttribute("clothdojo:homeJointQ").Get())
    expected[[5, 27]] -= math.pi
    actual = np.asarray(stage.GetDefaultPrim().GetAttribute("clothdojo:homeJointQ").Get())
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(robot_profile("ur5e_2f85").home_joint_q, actual, atol=1e-6)
    for joint_id, coord in ((6, 5), (29, 27)):
        joint = stage.GetPrimAtPath(f"/Robot/Joints/joint_{joint_id:03}")
        angle = math.degrees(actual[coord])
        assert joint.GetAttribute("physics:lowerLimit").Get() < angle
        assert angle < joint.GetAttribute("physics:upperLimit").Get()
        for attr in ("state:angular:physics:position", "drive:angular:physics:targetPosition"):
            assert joint.GetAttribute(attr).Get() == pytest.approx(angle, abs=2e-6)


def test_ur5_workstation_rebuild_does_not_accumulate_rotations_or_offsets(stage, tmp_path):
    import importlib.util

    from pxr import Sdf, Usd, UsdGeom

    spec = importlib.util.spec_from_file_location(
        "ur5_wrist_builder", ROOT / "build_wrist_cameras.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layer = Sdf.Layer.CreateAnonymous()
    layer.TransferContent(stage.GetRootLayer())
    layer.subLayerPaths = [str(ROOT / "payloads/workstation.usdc")]
    # Keep asset-relative component references resolvable, without editing source.
    (tmp_path / "components").symlink_to(ROOT / "components", target_is_directory=True)
    scratch_path = tmp_path / "workstation.usda"
    layer.Export(str(scratch_path))
    scratch = Usd.Stage.Open(str(scratch_path))
    layer = scratch.GetRootLayer()
    baseline = Usd.Stage.Open(str(ROOT / "payloads/workstation.usdc"))
    component = Usd.Stage.Open(str(ROOT / "components/wrist_camera.usdc"))
    optical = np.asarray(UsdGeom.Xformable(
        component.GetPrimAtPath("/WristCameraAssembly/ColorOpticalUSD")
    ).GetLocalTransformation()).T
    module.assemble_workstation(scratch, baseline, optical)
    once = layer.ExportToString()
    # Regeneration must not bake the workstation translation into body-local poses.
    cache = UsdGeom.XformCache()
    for prim in stage.GetPrimAtPath("/Robot/Bodies").GetChildren():
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(scratch.GetPrimAtPath(prim.GetPath())),
            cache.GetLocalToWorldTransform(prim), atol=5e-6, rtol=0,
        )
    for name in ("cam_left_wrist", "cam_right_wrist"):
        original = next(prim for prim in stage.Traverse() if prim.GetName() == name)
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(scratch.GetPrimAtPath(original.GetPath())),
            cache.GetLocalToWorldTransform(original), atol=5e-6, rtol=0,
        )
    module.assemble_workstation(scratch, baseline, optical)
    assert layer.ExportToString() == once


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
        np.testing.assert_allclose(actual, wrist_world @ flange @ camera_from_flange, atol=5e-6)
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
