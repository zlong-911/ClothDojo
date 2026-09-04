from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from clothdojo.robots import (
    ALOHA2,
    FR3_DUO,
    PIPER,
    UR5E_2F85,
    X5,
    robot_ids,
    robot_profile,
)
from clothdojo.tasks import SHIRT_FLATTEN, SHIRT_FOLD, task_profile


def _project(camera, points: np.ndarray) -> np.ndarray:
    world_from_camera = np.asarray(camera.parent_from_camera, dtype=np.float64)
    homogeneous = np.concatenate(
        (np.asarray(points, dtype=np.float64), np.ones((len(points), 1))), axis=1
    )
    camera_points = homogeneous @ np.linalg.inv(world_from_camera).T
    depth = -camera_points[:, 2]
    vertical_aperture = camera.horizontal_aperture * 480.0 / 640.0
    u = 320.0 + 640.0 * camera.focal_length * camera_points[:, 0] / (
        depth * camera.horizontal_aperture
    )
    v = 240.0 - 480.0 * camera.focal_length * camera_points[:, 1] / (depth * vertical_aperture)
    return np.stack((u, v, depth), axis=1)


def test_piper_profile_reads_the_four_asset_authored_cameras() -> None:
    profile = robot_profile(PIPER)

    assert profile.robot_id == "piper"
    assert "task_name" not in profile.to_json_dict()
    assert "fingerprint" not in profile.to_json_dict()

    assert tuple(camera.name for camera in profile.cameras) == (
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    assert tuple(camera.parent_body for camera in profile.cameras) == (
        None,
        None,
        "left/camera",
        "right/camera",
    )
    np.testing.assert_allclose(
        np.asarray(profile.cameras[0].parent_from_camera)[:3, 3],
        (0.253845, 0.001071, 1.308),
    )
    np.testing.assert_allclose(
        np.asarray(profile.cameras[2].parent_from_camera)[:3, 3],
        (0.0, 0.0, 0.0109),
    )


def test_every_robot_uses_the_same_asset_authored_workstation_contract() -> None:
    profiles = tuple(
        robot_profile(robot_id)
        for robot_id in (PIPER, ALOHA2, X5, FR3_DUO, UR5E_2F85)
    )

    assert all(profile.asset.endswith(".usda") for profile in profiles)
    assert all("newton_asset" not in profile.to_json_dict() for profile in profiles)
    assert all("installation_asset" not in profile.to_json_dict() for profile in profiles)
    assert set(robot_ids()) == {PIPER, ALOHA2, X5, FR3_DUO, UR5E_2F85}
    with pytest.raises(ValueError, match="retired"):
        robot_profile("panda")


def test_aloha_static_cameras_frame_both_robots_and_table_workspace() -> None:
    profile = robot_profile(ALOHA2)
    np.testing.assert_allclose(
        tuple(frame.position for frame in profile.root_body_frames),
        ((0.382, 0.559, 0.82), (0.382, -0.559, 0.82)),
    )
    assert profile.gripper_closed_positions == (0.0079, 0.0079)
    assert profile.gripper_open_positions == (0.037, 0.037)
    assert profile.asset == "robots/aloha2/aloha2.usda"
    assert profile.body_parents[8:10] == (7, 7)
    assert profile.body_parents[18:20] == (17, 17)
    assert tuple(camera.name for camera in profile.cameras) == (
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    assert all(camera.focal_length == 1.93 for camera in profile.cameras)
    assert all(camera.horizontal_aperture == 3.896 for camera in profile.cameras)
    representative_points = np.asarray(
        (
            (0.382, 0.559, 0.82),
            (0.382, -0.559, 0.82),
            (0.4, 0.3, 1.0),
            (0.4, -0.3, 1.0),
            (0.401, 0.0, 0.8),
        )
    )

    high = _project(profile.cameras[0], representative_points)
    assert np.all(high[:, 2] > 0.0)
    assert np.all((0.0 <= high[:, 0]) & (high[:, 0] < 640.0))
    assert np.all((0.0 <= high[:, 1]) & (high[:, 1] < 480.0))

    low = _project(profile.cameras[1], representative_points[2:])
    assert np.all(low[:, 2] > 0.0)
    assert np.all((0.0 <= low[:, 0]) & (low[:, 0] < 640.0))
    assert np.all((0.0 <= low[:, 1]) & (low[:, 1] < 480.0))
    np.testing.assert_allclose(
        np.asarray(profile.cameras[1].parent_from_camera)[:3, 3],
        (0.023833, 0.0, 0.8316055),
    )


def test_aloha_wrist_cameras_frame_open_fingers_at_a_readable_distance() -> None:
    profile = robot_profile(ALOHA2)
    finger_centers = np.asarray(
        (
            (0.0361, 0.0008363, 0.0811727),
            (-0.0361, 0.0008363, 0.0811727),
            (0.0, 0.0030756, 0.0949976),
        )
    )

    for camera in profile.cameras[2:4]:
        projected = _project(camera, finger_centers)
        assert np.all((0.1 <= projected[:, 2]) & (projected[:, 2] <= 0.15))
        assert np.all((120.0 <= projected[:, 0]) & (projected[:, 0] <= 520.0))
        assert np.all((300.0 <= projected[:, 1]) & (projected[:, 1] <= 370.0))


def test_fr3_duo_profile_uses_one_shared_articulation_and_accepted_home() -> None:
    profile = robot_profile(FR3_DUO)

    assert profile.asset == "robots/fr3_duo/fr3_duo.usda"
    assert profile.arm_dofs_per_side == 7
    assert len(profile.arm_coord_indices) == 14
    assert len(profile.home_joint_q) == 46
    assert profile.joint_dof_count == 42
    assert profile.arm_coord_indices == tuple(range(7)) + tuple(range(15, 22))
    assert profile.gripper_coord_indices == profile.gripper_dof_indices == (7, 11, 22, 26)
    assert profile.gripper_coords_per_side == 2
    assert profile.gripper_open_positions == (0.0, 0.0)
    assert profile.gripper_closed_positions == pytest.approx((0.8, 0.8))
    assert profile.body_parents.count(-1) == 1
    assert profile.body_names[0] == "fr3_duo_mount_mounting_point"
    np.testing.assert_allclose(
        tuple(frame.position for frame in profile.root_body_frames),
        ((0.13, 0.0, 0.95),), atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(profile.home_joint_q)[list(profile.arm_coord_indices)],
        (0.189741, -0.494105, -0.292850, -2.623325, 0.740971, 1.771250, 0.065042,
         -0.020020, -0.474083, 0.139937, -2.615131, -0.792055, 1.719218, -0.018384086),
        atol=2e-7, rtol=0,
    )
    assert tuple(frame.parent_body for frame in profile.attach_frames) == ("left/tcp", "right/tcp")
    for frame in profile.attach_frames:
        np.testing.assert_allclose(frame.position, (0.0, 0.0, 0.054319), atol=1e-9)
        np.testing.assert_allclose(frame.rotation_xyzw, (0.0, 0.0, 0.0, 1.0), atol=1e-9)


def test_fr3_duo_keeps_vmkit_cameras_and_adds_virtual_low_without_an_extra_stand() -> None:
    profile = robot_profile(FR3_DUO)
    assert tuple(camera.name for camera in profile.cameras) == (
        "cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist",
    )
    assert tuple(camera.parent_body for camera in profile.cameras) == (
        "zed_mini_left_camera_frame", None, "left/d405_color_frame", "right/d405_color_frame",
    )
    native_cameras = tuple(camera for camera in profile.cameras if camera.name != "cam_low")
    np.testing.assert_allclose(
        [camera.focal_length for camera in native_cameras], (2.4, 2.529072, 2.529072),
        atol=1e-6,
    )
    np.testing.assert_allclose([camera.horizontal_aperture for camera in native_cameras], 4.8)
    np.testing.assert_allclose(
        [camera.vertical_aperture for camera in native_cameras], (2.7, 2.716981, 2.716981),
        atol=1e-6,
    )
    stage = Usd.Stage.Open(str(Path(__file__).parents[1] / "assets" / profile.asset))
    cameras = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]
    assert len(cameras) == 4
    assert {prim.GetName() for prim in cameras} == {
        "head_cam", "left_wrist_cam", "right_wrist_cam", "cam_low",
    }
    assert stage.GetPrimAtPath("/Robot/Installation/TableMount")
    assert not stage.GetPrimAtPath("/Robot/Installation/CameraStand")
    for prim in cameras:
        name = prim.GetAttribute("clothdojo:cameraName").Get() or prim.GetName()
        spec = next(camera for camera in profile.cameras if camera.name == name)
        np.testing.assert_allclose(
            spec.parent_from_camera, np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T,
            atol=1e-7,
        )


def test_fr3_duo_attach_matches_ur5_robotiq_base_local_grasp_depth() -> None:
    stage = Usd.Stage.Open(
        str(Path(__file__).parents[1] / "assets" / robot_profile(FR3_DUO).asset)
    )
    cache = UsdGeom.XformCache()
    bases = {
        str(prim.GetAttribute("clothdojo:bodyLabel").Get()): prim
        for prim in stage.Traverse()
        if str(prim.GetAttribute("clothdojo:bodyLabel").Get()).endswith("/gripper_base")
    }
    checked = set()
    for prim in stage.Traverse():
        if prim.GetAttribute("clothdojo:role").Get() != "end_effector_attach":
            continue
        side = str(prim.GetAttribute("clothdojo:parentBody").Get()).split("/")[0]
        relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(
            bases[f"{side}/gripper_base"]
        ).GetInverse()
        # The new base default is baked from float32 FK; the retained TCP
        # keeps its original USD transform, so allow sub-micron round-off.
        np.testing.assert_allclose(relative.ExtractTranslation(), (0, 0, 0.157719), atol=1e-6)
        # Menagerie closes along Y; the retained FR3 attach closes along X.
        np.testing.assert_allclose(
            np.asarray(relative)[:3, :3], ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), atol=2e-6,
        )
        checked.add(side)
    assert checked == {"left", "right"}


@pytest.mark.parametrize("robot_id", (ALOHA2, FR3_DUO, UR5E_2F85))
def test_workstation_usd_default_transforms_match_authored_home_fk(robot_id) -> None:
    import newton
    import warp as wp
    from pxr import Gf

    from clothdojo.sim.scene import _add_robot

    profile = robot_profile(robot_id)
    path = Path(__file__).parents[1] / "assets" / profile.asset
    if robot_id == ALOHA2:
        np.testing.assert_allclose(
            np.asarray(profile.home_joint_q)[list(profile.arm_coord_indices)],
            (0.0, -0.96, 1.16, 0.0, -0.3, 0.0) * 2,
            rtol=0.0,
            atol=1.0e-7,
        )
    with wp.ScopedDevice("cpu"):
        builder = newton.ModelBuilder()
        _add_robot(newton, wp, builder, path, profile, load_visual_shapes=False)
        model = builder.finalize(device="cpu")
        if robot_id == FR3_DUO:
            assert model.articulation_count == 1
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        poses = dict(zip(profile.body_names, state.body_q.numpy(), strict=True))

    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    checked = set()
    for prim in stage.Traverse():
        label = prim.GetAttribute("clothdojo:bodyLabel")
        name = str(label.Get()) if label and label.Get() else ""
        if name not in poses:
            continue
        pose = poses[name]
        expected = Gf.Matrix4d().SetRotate(
            Gf.Quatd(float(pose[6]), *map(float, pose[3:6]))
        )
        expected.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(prim), expected, rtol=0.0, atol=2.0e-6,
            err_msg=name,
        )
        checked.add(name)
    assert checked == set(profile.body_names)


def test_ur5_usd_home_keeps_grasp_positions_and_synchronized_drive_targets() -> None:
    expected_positions = {
        UR5E_2F85: {
            # PickNik's flush camera adapter adds 7 mm along the tool axis;
            # the wrist half-turn keeps the grasp positions and original coupling.
                "left/gripper_base": (0.6001029, 0.4000614, 1.0430008),
                "right/gripper_base": (0.6000490, -0.3999888, 1.0430002),
        },
    }
    for robot_id, positions in expected_positions.items():
        profile = robot_profile(robot_id)
        stage = Usd.Stage.Open(str(Path(__file__).parents[1] / "assets" / profile.asset))
        cache = UsdGeom.XformCache()
        actual = {}
        for prim in stage.Traverse():
            label = prim.GetAttribute("clothdojo:bodyLabel")
            name = str(label.Get()) if label and label.Get() else ""
            if name in positions:
                actual[name] = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            if prim.IsA(UsdPhysics.RevoluteJoint):
                kind = "angular"
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                kind = "linear"
            else:
                continue
            state = prim.GetAttribute(f"state:{kind}:physics:position")
            target = prim.GetAttribute(f"drive:{kind}:physics:targetPosition")
            if state and state.HasAuthoredValue() and target:
                assert float(target.Get()) == pytest.approx(float(state.Get()), abs=1.0e-6)
        assert set(actual) == set(positions)
        for name, expected in positions.items():
            np.testing.assert_allclose(actual[name], expected, rtol=0.0, atol=2.0e-6)


def test_ur5_has_stable_bases_and_the_complete_legacy_camera_stand() -> None:
    project = Path(__file__).parents[1]
    camera_position = (0.2179578738615954, -0.0225080347427568, 1.34981)

    for robot_id, base_labels in (
        (UR5E_2F85, ("left/base", "right/base")),
    ):
        profile = robot_profile(robot_id)
        stage = Usd.Stage.Open(str(project / "assets" / profile.asset))
        body_prims = {}
        for prim in stage.Traverse():
            label = prim.GetAttribute("clothdojo:bodyLabel")
            name = str(label.Get()) if label and label.Get() else ""
            if name in base_labels:
                body_prims[name] = prim
        assert set(body_prims) == set(base_labels)

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            (UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy),
        )
        for side, label in zip((1.0, -1.0), base_labels, strict=True):
            prim = body_prims[label]
            origin = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
            np.testing.assert_allclose(origin, (0.15, side * 0.55, 0.8), atol=2.0e-6)
            bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            outer_edge = bounds.GetMax()[1] if side > 0 else bounds.GetMin()[1]
            assert 0.7 - abs(outer_edge) >= 0.05

        stand = stage.GetPrimAtPath("/Robot/Installation/CameraStand")
        assert stand
        assert stand.GetAttribute("clothdojo:columnAxisWorld").Get() == (0.161, 0.0)
        assert stage.GetPrimAtPath(
            "/Robot/Installation/CameraStand/base_br/Camera_stand/base/Mesh_1"
        )
        assert stage.GetPrimAtPath(
            "/Robot/Installation/CameraStand/base_br/rsd455/RSD455/Visual/Case_front"
        )
        for prim in Usd.PrimRange(stand):
            assert not prim.HasAPI(UsdPhysics.CollisionAPI)
            assert not prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            assert not prim.HasAPI(UsdPhysics.MassAPI)
            if prim.IsA(UsdGeom.Mesh):
                assert prim.GetAttribute("clothdojo:shapeFlags").Get() == 1
                assert prim.GetAttribute("physics:collisionEnabled").Get() is False

        base_plate = stage.GetPrimAtPath(
            "/Robot/Installation/CameraStand/base_br/Camera_stand/base/Mesh_1"
        )
        base_bounds = bbox_cache.ComputeWorldBound(base_plate).ComputeAlignedRange()
        assert base_bounds.GetMin()[2] == pytest.approx(0.8, abs=2.0e-6)
        assert base_bounds.GetMax()[2] == pytest.approx(0.81, abs=2.0e-6)
        cam_high = stage.GetPrimAtPath("/Robot/Semantics/cam_high")
        assert cam_high.GetAttribute("clothdojo:cameraModel").Get() == "RealSense D455"
        np.testing.assert_allclose(
            UsdGeom.XformCache().GetLocalToWorldTransform(cam_high).ExtractTranslation(),
            camera_position,
        )


def test_task_profiles_are_independent_of_robot_profiles(monkeypatch) -> None:
    assert task_profile(SHIRT_FLATTEN).scene_available is True
    assert task_profile(SHIRT_FOLD).scene_available is True

    from clothdojo.sim import scene

    marker = object()
    monkeypatch.setattr(scene, "_build_shirt_flatten", lambda *_args, **_kwargs: marker)

    assert scene.build_scene("assets", robot_id=PIPER, task_name=SHIRT_FOLD) is marker


def test_x5_profile_uses_the_kai0_integrated_base_with_robodojo_x5() -> None:
    profile = robot_profile(X5)

    assert profile.asset == "robots/x5/x5.usda"
    assert profile.arm_dofs_per_side == 6
    assert len(profile.home_joint_q) == 16
    np.testing.assert_allclose(
        tuple(frame.position for frame in profile.root_body_frames),
        ((0.25, 0.28625912, 0.81), (0.25, -0.28625912, 0.81)),
    )
    assert profile.attach_frames[0].position == (0.152602, 0.0, -0.00014)
    assert profile.gripper_open_positions == (0.044, 0.044)
    assert profile.body_parents.count(-1) == 2


def test_x5_references_complete_robodojo_arms_and_cameras() -> None:
    stage = Usd.Stage.Open(
        str(Path(__file__).parents[1] / "assets/robots/x5/x5.usda")
    )

    stand = stage.GetPrimAtPath("/Robot/Installation/CameraStand")
    assert stand
    assert not stage.GetPrimAtPath("/Robot/Installation/Kai0CameraBottomBracket")
    assert stage.GetPrimAtPath("/Robot/Installation/CameraStand/Foot")
    assert stage.GetPrimAtPath("/Robot/Installation/CameraStand/D455/CaseFront")
    assert not stage.GetPrimAtPath("/Robot/Installation/CameraStand/Base")
    assert stand.GetAttribute("clothdojo:footHolePatternMm").Get() == [54.1, 54.1, 5.0]
    np.testing.assert_allclose(
        UsdGeom.Xformable(stand).GetLocalTransformation().ExtractTranslation(),
        (0.261, 0.0, 0.81),
    )
    cam_high = stage.GetPrimAtPath("/Robot/Semantics/cam_high")
    assert cam_high.GetAttribute("clothdojo:cameraModel").Get() == "RealSense D455"
    np.testing.assert_allclose(
        UsdGeom.Xformable(cam_high).GetLocalTransformation().ExtractTranslation(),
        (0.3179578738615954, -0.0225080347427568, 1.44981),
    )
    np.testing.assert_allclose(
        np.asarray(UsdGeom.Xformable(cam_high).GetLocalTransformation())[:3, :3],
        ((0, -1, 0), (0.8660254037844386, 0, 0.5), (-0.5, 0, 0.8660254037844386)),
    )
    assert stand.GetAttribute("clothdojo:columnExtensionMeters").Get() == pytest.approx(0.1)
    stand_bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
    ).ComputeWorldBound(stand).ComputeAlignedRange()
    assert stand_bounds.GetMin()[2] == pytest.approx(0.804, abs=2e-6)
    assert stand_bounds.GetMax()[2] == pytest.approx(1.48951, abs=2e-6)
    for side in ("left", "right"):
        arm = stage.GetPrimAtPath(f"/Robot/Arms/{side}")
        assert arm
        assert stage.GetPrimAtPath(f"/Robot/Arms/{side}/camera_base")
        prim = stage.GetPrimAtPath(
            f"/Robot/Arms/{side}/camera/visuals/camera/World/mesh"
        )
        mesh = UsdGeom.Mesh(prim)
        assert prim.GetAttribute("clothdojo:shapeFlags").Get() == 1
        assert prim.GetAttribute("physics:collisionEnabled").Get() is False
        assert len(mesh.GetPointsAttr().Get()) == 554_009
        assert len(mesh.GetFaceVertexCountsAttr().Get()) == 231_186
        mount_joint = UsdPhysics.FixedJoint(
            stage.GetPrimAtPath(f"/Robot/Arms/{side}/joints/hand_to_camera_mount")
        )
        camera_joint = UsdPhysics.FixedJoint(
            stage.GetPrimAtPath(f"/Robot/Arms/{side}/joints/camera_joint")
        )
        np.testing.assert_allclose(mount_joint.GetLocalPos0Attr().Get(), (0.057, 0.0, 0.0))
        np.testing.assert_allclose(
            camera_joint.GetLocalPos0Attr().Get(), (-0.0275, 0.0, 0.05)
        )
        root_joint = UsdPhysics.FixedJoint(
            stage.GetPrimAtPath(f"/Robot/Arms/{side}/joints/root_joint")
        )
        assert root_joint
        assert root_joint.GetBody1Rel().GetTargets() == [
            root_joint.GetPath().GetParentPath().GetParentPath().AppendChild("base_link")
        ]


def test_x5_preserves_upstream_dark_material_and_matches_base_to_column() -> None:
    folder = Path(__file__).parents[1] / "assets/robots/x5"
    stage = Usd.Stage.Open(str(folder / "x5.usda"))
    source = Usd.Stage.Open(str(folder / "robodojo/ARX.usd"))
    original = source.GetPrimAtPath("/X5A/Looks/defaultmaterial/Shader")
    for side in ("left", "right"):
        count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/Robot/Arms/{side}")):
            path = str(prim.GetPath())
            if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in path:
                continue
            body = path.split("/")[4]
            if body in ("link3", "camera"):
                continue  # Known unresolved upstream bindings, not restored here.
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            assert str(material.GetPath()) == f"/Robot/Arms/{side}/Looks/defaultmaterial"
            shader, _, _ = material.ComputeSurfaceSource()
            for attr in original.GetAttributes():
                assert shader.GetPrim().GetAttribute(attr.GetName()).Get() == attr.Get()
            count += 1
        assert count == 9
    paths = (
        "/Robot/Installation/Kai0BaseIntegrated",
        "/Robot/Installation/CameraStand/Column",
    )
    materials = [
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial()[0]
        for path in paths
    ]
    assert all(materials)
    assert materials[0].GetPath() == materials[1].GetPath()


def test_ur5e_2f85_profile_keeps_closed_chain_coordinates_separate_from_dofs() -> None:
    profile = robot_profile(UR5E_2F85)

    assert profile.asset == "robots/ur5e_2f85/ur5e_2f85.usda"
    assert profile.arm_dofs_per_side == 6
    assert len(profile.home_joint_q) == 44
    assert profile.joint_dof_count == 40
    assert profile.arm_coord_indices[6:] == (22, 23, 24, 25, 26, 27)
    assert profile.arm_dof_indices[6:] == (20, 21, 22, 23, 24, 25)
    assert profile.gripper_coord_indices == (6, 10, 28, 32)
    assert profile.gripper_dof_indices == (6, 10, 26, 30)
    assert profile.body_parents.count(-1) == 2


@pytest.mark.parametrize("robot_id,mesh_count", [(ALOHA2, 55), (UR5E_2F85, 68)])
def test_mjcf_workstations_keep_bound_source_materials(robot_id, mesh_count) -> None:
    import mujoco

    folder = Path(__file__).parents[1] / "assets/robots" / robot_id
    source_files = ("aloha.xml", "frame.xml") if robot_id == ALOHA2 else ("ur5e_2f85.xml",)
    source_materials = {}
    for filename in source_files:
        model = mujoco.MjModel.from_xml_path(str(folder / filename))
        for index in range(model.nmat):
            source_materials[model.mat(index).name] = (
                model.mat_rgba[index, :3].copy(),
                float(model.mat_specular[index]), float(model.mat_shininess[index]),
            )
    stage = Usd.Stage.Open(
        str(Path(__file__).parents[1] / "assets" / robot_profile(robot_id).asset)
    )
    meshes = [
        prim for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
        and str(prim.GetAttribute("clothdojo:shapeLabel").Get()).endswith("_visual")
    ]
    assert len(meshes) == mesh_count
    for prim in meshes:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert material, prim.GetPath()
        shader, _, _ = material.ComputeSurfaceSource()
        assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
        assert shader.GetInput("useSpecularWorkflow").Get() == 1
        assert not shader.GetInput("metallic")  # shininess is not metalness
        name = material.GetPrim().GetName().split("_", 3)[3]
        if name == "geom_rgba_default":
            source_rgb, specular, shininess = np.full(3, .5), .5, .5
        else:
            source_rgb, specular, shininess = source_materials[name]
        source_rgb = np.asarray(source_rgb, dtype=np.float64)
        expected_linear = np.where(
            source_rgb <= .04045, source_rgb / 12.92,
            ((source_rgb + .055) / 1.055) ** 2.4,
        )
        # Fixed MuJoCo 3.11 Filament legacy-material semantics, not an eye fit.
        np.testing.assert_allclose(shader.GetInput("specularColor").Get(), specular * .2)
        assert shader.GetInput("roughness").Get() == pytest.approx(1 - shininess * .1)
        color = shader.GetInput("diffuseColor").Get()
        np.testing.assert_allclose(color, expected_linear, atol=1e-7)
        np.testing.assert_allclose(color, UsdGeom.Mesh(prim).GetDisplayColorAttr().Get()[0])


def test_aloha_extrusion_keeps_source_stl_facet_normals() -> None:
    root = Path(__file__).parents[1] / "assets/robots/aloha2"
    stage = Usd.Stage.Open(str(root / "aloha2.usda"))
    prim = next(
        p for p in stage.Traverse()
        if p.GetAttribute("clothdojo:shapeLabel").Get()
        == "aloha_installation_frame/worldbody/world_geom_1_visual"
    )
    raw = (root / "assets/extrusion_150.stl").read_bytes()
    records = np.frombuffer(
        raw, offset=84,
        dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")],
    )
    mesh = UsdGeom.Mesh(prim)
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.uniform
    np.testing.assert_allclose(mesh.GetNormalsAttr().Get(), records["normal"], atol=1e-6)


def test_ur5e_gripper_keeps_stl_hard_edges_without_changing_obj_normals() -> None:
    stage = Usd.Stage.Open(
        str(Path(__file__).parents[1] / "assets/robots/ur5e_2f85/ur5e_2f85.usda")
    )
    interpolations = {"uniform": 0, "vertex": 0}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and str(
            prim.GetAttribute("clothdojo:shapeLabel").Get()
        ).endswith("_visual"):
            interpolations[UsdGeom.Mesh(prim).GetNormalsInterpolation()] += 1
    assert interpolations == {"uniform": 28, "vertex": 40}
