"""The FR3 low observer is a fixed Piper view, not additional VMkit hardware."""

from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from clothdojo.rendering import DATASET_CAMERA_NAMES
from clothdojo.robots import FR3_DUO, PIPER, robot_profile

ASSETS = Path(__file__).resolve().parents[1] / "assets"
LOW_PATH = "/Robot/cameras/cam_low"


@pytest.fixture(scope="module")
def camera_stages():
    return {
        "fr3": Usd.Stage.Open(str(ASSETS / "robots/fr3_duo/fr3_duo.usda")),
        "piper": Usd.Stage.Open(str(ASSETS / "robots/piper/piper.usda")),
        "payload": Usd.Stage.Open(str(ASSETS / "robots/fr3_duo/payloads/workstation.usdc")),
    }


def _assert_same_camera_parameters(first, second):
    for name in UsdGeom.Camera.GetSchemaAttributeNames(includeInherited=False):
        assert first.GetAttribute(name).Get() == second.GetAttribute(name).Get(), name


def test_fr3_low_camera_matches_piper_world_pose_and_every_optical_parameter(camera_stages):
    low = camera_stages["fr3"].GetPrimAtPath(LOW_PATH)
    source = next(
        prim for prim in camera_stages["piper"].Traverse()
        if prim.IsA(UsdGeom.Camera) and prim.GetName() == "cam_low"
    )
    _assert_same_camera_parameters(low, source)
    cache = UsdGeom.XformCache()
    np.testing.assert_array_equal(
        cache.GetLocalToWorldTransform(low), cache.GetLocalToWorldTransform(source)
    )
    np.testing.assert_array_equal(
        UsdGeom.Xformable(low).GetLocalTransformation(), cache.GetLocalToWorldTransform(source)
    )
    assert UsdGeom.Xformable(low).GetResetXformStack()


def test_fr3_low_observer_is_exposed_as_an_independent_fourth_policy_camera():
    fr3 = robot_profile(FR3_DUO)
    piper = robot_profile(PIPER)
    assert tuple(camera.name for camera in fr3.cameras) == DATASET_CAMERA_NAMES
    low = next(camera for camera in fr3.cameras if camera.name == "cam_low")
    assert low.parent_body is None
    assert low == next(camera for camera in piper.cameras if camera.name == "cam_low")


def test_fr3_observer_adds_no_hardware_or_piper_runtime_dependency(camera_stages):
    stage = camera_stages["fr3"]
    scope = stage.GetPrimAtPath("/Robot/cameras")
    assert [str(prim.GetPath()) for prim in Usd.PrimRange(scope)] == [
        "/Robot/cameras", LOW_PATH,
    ]
    for prim in Usd.PrimRange(scope):
        assert not prim.IsA(UsdGeom.Gprim)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.GetAttribute("clothdojo:parentBody")
    assert all("piper" not in layer.identifier for layer in stage.GetUsedLayers())
    assert not camera_stages["payload"].GetPrimAtPath(LOW_PATH)


def test_fr3_native_three_cameras_keep_payload_geometry_and_optics(camera_stages):
    stage = camera_stages["fr3"]
    payload = camera_stages["payload"]
    native = [prim for prim in payload.Traverse() if prim.IsA(UsdGeom.Camera)]
    assert len(native) == 3
    assert len([prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]) == 4
    cache = UsdGeom.XformCache()
    for source in native:
        current = stage.GetPrimAtPath(source.GetPath())
        _assert_same_camera_parameters(current, source)
        expected = np.array(cache.GetLocalToWorldTransform(source))
        np.testing.assert_allclose(cache.GetLocalToWorldTransform(current), expected, atol=1e-12)
        np.testing.assert_array_equal(
            UsdGeom.Xformable(current).GetLocalTransformation(),
            UsdGeom.Xformable(source).GetLocalTransformation(),
        )
