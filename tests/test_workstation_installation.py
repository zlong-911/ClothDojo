"""Installation changes must agree in USD, runtime FK, and replay metadata."""
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from clothdojo.robots import robot_profile
from clothdojo.sim.scene import ShirtFlattenSpec
from clothdojo.sim.workstation_review import prepare_workstation_review

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _stage(robot):
    return Usd.Stage.Open(str(ASSETS / "robots" / robot / f"{robot}.usda"))


def _bounds(prim):
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        prim
    ).ComputeAlignedRange()
    return np.array((box.GetMin(), box.GetMax()))


def test_fr3_column_returns_to_table_edge_without_moving_arm_mount_height():
    stage = _stage("fr3_duo")
    mount = stage.GetPrimAtPath("/Robot/Installation/TableMount")
    profile = mount.GetChild("varidee_profile_120x80_150")
    bounds = _bounds(profile)
    np.testing.assert_allclose(bounds[:, 2], (.8, .95), atol=2e-7, rtol=0)
    np.testing.assert_allclose(bounds[1, :2] - bounds[0, :2], (.12, .08), atol=2e-7)
    spec = ShirtFlattenSpec()
    table_rear = spec.table_position[0] - spec.table_half_extents[0]
    assert bounds[0, 0] - table_rear == pytest.approx(.0505, abs=1e-6)
    assert len(list(mount.GetChildren())) == 19  # profile, six top screws, six bolts/washers
    for index in range(3):
        for side in range(2):
            assert mount.GetChild(f"bottom_bolt_M8x70_{index}_{side}").IsActive()
            assert mount.GetChild(f"bottom_washer_M8_{index}_{side}").IsActive()
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        label = prim.GetAttribute("clothdojo:bodyLabel")
        if label and label.Get() in ("left/fr3v2_link0", "right/fr3v2_link0"):
            position = np.asarray(cache.GetLocalToWorldTransform(prim))[3, :3]
            np.testing.assert_allclose(position[[0, 2]], (.167, 1.0), atol=2e-7)


def test_ur5_bases_stay_on_table_and_physical_head_camera_moves_with_stand():
    stage = _stage("ur5e_2f85")
    before = Usd.Stage.Open(str(ASSETS / "robots/ur5e_2f85/payloads/workstation.usdc"))
    spec = ShirtFlattenSpec()
    table_rear = spec.table_position[0] - spec.table_half_extents[0]
    for name in ("body_000", "body_021"):
        bounds = _bounds(stage.GetPrimAtPath(f"/Robot/Bodies/{name}"))
        assert bounds[0, 2] == pytest.approx(.8, abs=1e-6)
        assert bounds[0, 0] - table_rear == pytest.approx(.0545, abs=2e-6)
    cache = UsdGeom.XformCache()
    for path in ("/Robot/Installation/CameraStand", "/Robot/Semantics/cam_high"):
        expected = np.array(cache.GetLocalToWorldTransform(before.GetPrimAtPath(path)))
        expected[3, 0] -= .1
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)), expected, atol=1e-12,
        )
    assert _bounds(stage.GetPrimAtPath("/Robot/Installation/CameraStand"))[0, 0] > table_rear


@pytest.mark.parametrize("robot", ("fr3_duo", "ur5e_2f85"))
def test_relocated_usd_root_attach_and_cameras_match_newton_and_replay_frames(robot):
    stage = _stage(robot)
    profile = robot_profile(robot)
    review = prepare_workstation_review(robot, assets_root=ASSETS)
    assert review.report["table_source"] == "ShirtFlattenSpec"
    poses = review.state.body_q.numpy()
    roots = [i for i, parent in enumerate(profile.body_parents) if parent == -1]
    for index, frame in zip(roots, profile.root_body_frames, strict=True):
        np.testing.assert_allclose(poses[index, :3], frame.position, atol=2e-6)
        assert abs(np.dot(poses[index, 3:], frame.rotation_xyzw)) == pytest.approx(1, abs=1e-6)
    for side in ("left", "right"):
        frame = review.report["attach"][side]
        np.testing.assert_allclose(
            frame["newton_home_world_row_matrix"], frame["usd_default_world_row_matrix"],
            atol=5e-6, rtol=0,
        )
    cache = UsdGeom.XformCache()
    cameras = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]
    for camera in profile.cameras:
        prim = next(p for p in cameras if p.GetName() == camera.name or
                    p.GetAttribute("clothdojo:cameraName").Get() == camera.name)
        np.testing.assert_allclose(
            review.report["cameras"][camera.name]["newton_home_world_row_matrix"],
            cache.GetLocalToWorldTransform(prim), atol=5e-6, rtol=0,
        )
