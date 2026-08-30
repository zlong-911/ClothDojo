"""Check the review camera against USD optics, not ViewerGL's orbit camera."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom


@pytest.fixture
def review_module():
    path = Path(__file__).parents[1] / "scripts/robots/check_cameras.py"
    spec = importlib.util.spec_from_file_location("camera_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_camera():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.CreateFocalLengthAttr(12)
    camera.CreateHorizontalApertureAttr(4.8)
    camera.CreateVerticalApertureAttr(2.7)
    camera.CreateHorizontalApertureOffsetAttr(.03)
    camera.CreateVerticalApertureOffsetAttr(-.02)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.04, 10))
    return stage, camera


def test_native_review_preserves_offsets_and_two_axis_projection(review_module, source_camera):
    _, usd = source_camera
    camera = review_module.native_camera(usd.GetPrim(), np.eye(4))
    center = review_module.projected_point(camera, (0, 0, -1), 640, 480)
    assert center["in_frustum"]
    np.testing.assert_allclose(
        center["pixel_xy"], (320 * (1 - 2 * .03 / 4.8), 240 * (1 - 2 * .02 / 2.7)),
        atol=1e-5,
    )
    for point in ((0, 0, 1), (0, 0, -.01), (0, 0, -11)):
        assert not review_module.projected_point(camera, point, 640, 480)["in_frustum"]
    # Native frustum at a 4:3 raster does not widen the 16:9 source vertical FOV.
    point = (.1, .05, -1)
    projected = review_module.projected_point(camera, point, 640, 480)
    np.testing.assert_allclose(
        np.asarray(projected["pixel_xy"]) - center["pixel_xy"],
        (640 * 12 / 4.8 * .1, -480 * 12 / 2.7 * .05), atol=1e-5,
    )


def test_gl_review_preserves_camera_roll_and_exact_vertical_axis(review_module, source_camera):
    import pyglet

    pyglet.options["headless"] = True
    _, usd = source_camera
    world = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 37))
    world.SetTranslateOnly(Gf.Vec3d(.7, -.4, 1.2))
    camera = review_module.native_camera(usd.GetPrim(), np.asarray(world))
    view = review_module.gl_camera(camera, 640, 480)
    np.testing.assert_allclose(view.get_front(), (0, 0, -1), atol=1e-7)
    np.testing.assert_allclose(view.get_up(), np.asarray(world)[1, :3], atol=1e-7)
    np.testing.assert_allclose(view.get_right(), np.asarray(world)[0, :3], atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(world) @ view.get_view_matrix().reshape(4, 4), np.eye(4), atol=1e-7,
    )
    projection = view.get_projection_matrix().reshape(4, 4)
    assert projection[0, 0] == pytest.approx(2 * 12 / 4.8)
    assert projection[1, 1] == pytest.approx(2 * 12 / 2.7)
    assert view.near == pytest.approx(.04)
    assert view.far == pytest.approx(10)
    with pytest.raises(ValueError, match="meter-scale"):
        view.get_view_matrix(scaling=100)


def test_native_review_does_not_edit_source_camera(review_module, source_camera):
    stage, camera = source_camera
    before = stage.GetRootLayer().ExportToString()
    world = np.eye(4)
    world[3, :3] = (.7, .3, 1.2)
    result = review_module.native_camera(camera.GetPrim(), world)
    np.testing.assert_array_equal(result.transform, world)
    assert stage.GetRootLayer().ExportToString() == before
