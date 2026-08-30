import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom

ROOT = Path(__file__).parents[1] / "assets/robots/x5"


@pytest.fixture
def stage():
    return Usd.Stage.Open(str(ROOT / "x5.usda"))


def test_x5_wrist_cameras_use_complete_robodojo_simulation_preset(stage):
    from scipy.spatial.transform import Rotation

    source = json.loads((ROOT / "provenance/robodojo_wrist_camera.json").read_text())
    assert source["model"] == "D435"
    assert source["stream"] == "color"
    # Fixed source robot_config.yml plus runtime's intrinsic XYZ convention.
    # PINHOLE child pose is identity in USD axes: no extra ROS or roll transform.
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler("XYZ", [0, -80, -90], degrees=True).as_matrix().T
    assert source["source_euler_intrinsic_xyz_degrees"] == [0, -80, -90]
    assert source["output_roll_degrees"] == 0
    assert np.linalg.det(expected[:3, :3]) == pytest.approx(1)
    np.testing.assert_allclose(source["usd_rotation_rows"], expected[:3, :3], atol=1e-15)
    expected[3, :3] = (0, 0, .001)
    assert source["position_m"] == [0, 0, .001]
    assert source["resolution"] == [640, 480]
    for side in ("left", "right"):
        prim = stage.GetPrimAtPath(f"/Robot/Arms/{side}/camera/cam_{side}_wrist")
        camera = UsdGeom.Camera(prim)
        np.testing.assert_allclose(
            UsdGeom.Xformable(prim).GetLocalTransformation(), expected, atol=1e-12,
        )
        assert camera.GetFocalLengthAttr().Get() == source["focal_length"] == 13
        assert camera.GetHorizontalApertureAttr().Get() == pytest.approx(20.955)
        assert source["horizontal_aperture"] == 20.955
        assert camera.GetVerticalApertureAttr().Get() == pytest.approx(15.71625)
        assert source["vertical_aperture"] == 15.71625
        np.testing.assert_allclose(camera.GetClippingRangeAttr().Get(), [.0001, 10])
        assert source["clipping_range_m"] == [.0001, 10]
        np.testing.assert_allclose(source["output_fx_fy_cx_cy_pixels"], [
            640 * 13 / 20.955, 480 * 13 / 15.71625, 320, 240,
        ])
        assert camera.GetHorizontalApertureOffsetAttr().Get() == 0
        assert camera.GetVerticalApertureOffsetAttr().Get() == 0
        assert prim.GetAttribute("clothdojo:parentBody").Get() == f"{side}/camera"
        assert prim.GetAttribute("clothdojo:cameraModel").Get() == "D435"
        assert prim.GetAttribute("clothdojo:cameraOutputRollDegrees").Get() == 0
        assert prim.GetAttribute("clothdojo:cameraSource").Get() == (
            "provenance/robodojo_wrist_camera.json"
        )
        assert prim.GetAttribute("clothdojo:cameraStream").Get() == "color"
        assert source["calibration"] == "robodojo_simulation_preset"
        assert prim.GetAttribute("clothdojo:cameraCalibration").Get() == (
            source["calibration"]
        )


def test_x5_physical_camera_mesh_registration_is_independent_of_virtual_optics(stage):
    from scipy.spatial import cKDTree

    source = json.loads((ROOT / "provenance/robodojo_wrist_camera.json").read_text())
    registration = source["mesh_registration"]
    rotation = np.asarray(registration["official_dae_to_camera_body_rotation_rows"])
    np.testing.assert_array_equal(rotation, ((0, -1, 0), (0, 0, -1), (1, 0, 0)))
    assert np.linalg.det(rotation) == 1
    landmarks = np.asarray(registration["official_dae_landmarks_m"]) @ rotation
    cache = UsdGeom.XformCache()
    for side in ("left", "right"):
        body = stage.GetPrimAtPath(f"/Robot/Arms/{side}/camera")
        mesh = stage.GetPrimAtPath(f"{body.GetPath()}/visuals/camera/World/mesh")
        transform = np.asarray(
            cache.GetLocalToWorldTransform(mesh) * cache.GetLocalToWorldTransform(body).GetInverse()
        )
        points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
        points = points @ transform[:3, :3] + transform[3, :3]
        # Eight fixed official DAE vertices across housing, internal optics and
        # connectors guard the rigid registration, not just equal bounding boxes.
        assert cKDTree(points).query(landmarks)[0].max() < 2e-7
        np.testing.assert_allclose(
            points.max(0) - points.min(0), [.0250547, .0899143, .025], atol=2e-7,
        )


def test_x5_camera_fix_keeps_home_assembly_attach_and_third_person_cameras(stage):
    baseline = Usd.Stage.Open(str(ROOT / "payloads/workstation.usdc"))
    np.testing.assert_array_equal(
        stage.GetDefaultPrim().GetAttribute("clothdojo:homeJointQ").Get(),
        baseline.GetDefaultPrim().GetAttribute("clothdojo:homeJointQ").Get(),
    )
    cache = UsdGeom.XformCache()
    checked = 0
    for prim in baseline.Traverse():
        if not UsdGeom.Xformable(prim) or prim.GetName() in ("cam_left_wrist", "cam_right_wrist"):
            continue
        np.testing.assert_array_equal(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prim.GetPath())),
            cache.GetLocalToWorldTransform(prim), err_msg=str(prim.GetPath()),
        )
        checked += 1
    assert checked > 40


def test_x5_robodojo_camera_keeps_grasp_centered_and_points_down_at_home():
    from clothdojo.sim.scene import ShirtFlattenSpec
    from clothdojo.sim.workstation_review import prepare_workstation_review

    review = prepare_workstation_review("x5", assets_root=ROOT.parents[1], device="cpu")
    table = ShirtFlattenSpec()
    table_z = table.table_position[2] + table.table_half_extents[2]
    for side in ("left", "right"):
        camera = review.report["cameras"][f"cam_{side}_wrist"]
        world = np.asarray(camera["newton_home_world_row_matrix"])
        front = -world[2, :3]
        front /= np.linalg.norm(front)
        # The source virtual camera adds ten degrees to the physical mount tilt.
        assert math.degrees(math.asin(front[2])) == pytest.approx(-30, abs=.01)
        assert math.degrees(math.atan2(front[1], front[0])) == pytest.approx(0, abs=.01)
        assert world[1, 2] > .86  # Upstream pose is upright; no additional roll.
        origin = world[3, :3]
        distance = (table_z - origin[2]) / front[2]
        assert distance > 0
        hit = origin + distance * front
        assert np.all(np.abs(hit[:2] - table.table_position[:2]) < table.table_half_extents[:2])
        attach = np.asarray(review.report["attach"][side]["newton_home_world_row_matrix"])
        assert np.dot(front, attach[3, :3] - origin) > 0
        local = np.r_[attach[3, :3], 1] @ np.linalg.inv(world)
        pixel = (
            320 + 640 * 13 / 20.955 * local[0] / -local[2],
            240 - 480 * 13 / 15.71625 * local[1] / -local[2],
        )
        np.testing.assert_allclose(pixel, (320, 288.81), atol=.02)


def test_x5_camera_builder_is_idempotent_and_retains_correct_convention():
    spec = importlib.util.spec_from_file_location("x5_builder", ROOT / "build_workstation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/camera").GetPrim()
    module._author_camera(parent, "cam_left_wrist")
    first = stage.GetRootLayer().ExportToString()
    module._author_camera(parent, "cam_left_wrist")
    assert stage.GetRootLayer().ExportToString() == first
    fixed = Usd.Stage.Open(str(ROOT / "x5.usda"))
    expected = fixed.GetPrimAtPath("/Robot/Arms/left/camera/cam_left_wrist")
    actual = stage.GetPrimAtPath("/camera/cam_left_wrist")
    for attr in actual.GetAttributes():
        if attr.HasAuthoredValue():
            assert attr.Get() == expected.GetAttribute(attr.GetName()).Get()


def test_x5_restored_apertures_expose_existing_legacy_policy_ray_boundary():
    from clothdojo.robots import robot_profile
    from clothdojo.rollout.render import _camera_rays

    class RecordRays:
        def compute_camera_rays_pinhole(self, width, height, **kwargs):
            assert (width, height) == (640, 480)
            return kwargs

    cameras = robot_profile("x5").cameras
    result = _camera_rays(RecordRays(), cameras, 640, 480)
    # This asset change does not change the historical benchmark renderer.
    # All four apertures are now 4:3, so it again takes its legacy H-as-V branch.
    # This is a documented boundary, NOT a camera-calibration correctness claim.
    assert "horizontal_aperture" not in result
    np.testing.assert_allclose(result["camera_fovs"], [
        2 * math.atan(camera.horizontal_aperture / (2 * camera.focal_length))
        for camera in cameras
    ])
