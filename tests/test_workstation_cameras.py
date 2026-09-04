import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from clothdojo.rendering import CAMERA_NAMES, DATASET_CAMERA_NAMES, CameraSpec
from clothdojo.rollout.render import ShirtFlattenCameraObservation, _camera_rays, _clip_camera_rays


def _camera(name="cam_high", parent=None, *, vertical_aperture=16.875):
    matrix = np.eye(4)
    matrix[2, 3] = 0.1
    return CameraSpec(
        name=name,
        parent_body=parent,
        parent_from_camera=tuple(tuple(row) for row in matrix),
        focal_length=20.0,
        horizontal_aperture=30.0,
        vertical_aperture=vertical_aperture,
    )


def test_native_camera_rays_preserve_both_apertures_on_cpu():
    from newton._src.sensors.warp_raytrace.utils import Utils

    utils = Utils(SimpleNamespace(device="cpu"))
    rays = _camera_rays(utils, (_camera(),), 4, 3).numpy()
    for row, column in ((0, 0), (1, 2), (2, 3)):
        expected = np.array(
            [((column + 0.5) / 4 - 0.5) * 30 / 20, (0.5 - (row + 0.5) / 3) * 16.875 / 20, -1.0]
        )
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(rays[0, row, column, 1], expected, atol=1.0e-7)


def test_legacy_four_by_three_rays_keep_existing_benchmark_pixels():
    from newton._src.sensors.warp_raytrace.utils import Utils

    utils = Utils(SimpleNamespace(device="cpu"))
    actual = _camera_rays(utils, (_camera(vertical_aperture=22.5),), 4, 3).numpy()
    # Explicitly retain the historical horizontal-as-vertical FOV behavior.
    # This does NOT assert equality with the camera's physical USD intrinsics.
    original = utils.compute_camera_rays_pinhole(
        4, 3, camera_fovs=(2 * math.atan(30 / (2 * 20)),)
    ).numpy()
    np.testing.assert_array_equal(actual, original)


def test_near_clip_preserves_projection_and_uses_an_optical_axis_plane():
    from newton._src.sensors.warp_raytrace.utils import Utils

    cameras = tuple(replace(_camera(), near_clip=near) for near in (0.0, 0.01, 0.04))
    rays = _camera_rays(Utils(SimpleNamespace(device="cpu")), cameras, 8, 6)
    before = rays.numpy()
    _clip_camera_rays(rays, cameras)
    after = rays.numpy()
    np.testing.assert_array_equal(after[..., 1, :], before[..., 1, :])
    for index, camera in enumerate(cameras):
        origins, directions = after[index, ..., 0, :], after[index, ..., 1, :]
        np.testing.assert_allclose(origins[..., 2], -camera.near_clip, atol=1e-8)
        np.testing.assert_allclose(np.cross(origins, directions), 0, atol=1e-8)
    # An oblique ray must start farther than near in Euclidean distance.
    assert np.linalg.norm(after[1, 0, 0, 0]) > cameras[1].near_clip


@pytest.mark.parametrize("near", [-1, float("nan"), float("inf")])
def test_invalid_near_clip_is_rejected(near):
    with pytest.raises(ValueError, match="near clipping"):
        _clip_camera_rays(None, (replace(_camera(), near_clip=near),))


@pytest.mark.parametrize("robot_id", ["piper", "aloha2", "x5", "ur5e_2f85", "fr3_duo"])
def test_runtime_camera_near_clip_matches_usd(robot_id):
    from pathlib import Path

    from pxr import Usd, UsdGeom

    from clothdojo.robots import robot_profile

    profile = robot_profile(robot_id)
    stage = Usd.Stage.Open(str(Path(__file__).resolve().parents[1] / "assets" / profile.asset))
    authored = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            role = prim.GetAttribute("clothdojo:cameraName")
            name = role.Get() if role and role.Get() else prim.GetName()
            authored[name] = float(UsdGeom.Camera(prim).GetClippingRangeAttr().Get()[0])
    for camera in profile.cameras:
        assert camera.near_clip == authored[camera.name]


@pytest.fixture
def camera_scene(monkeypatch):
    import newton.sensors

    import clothdojo.rollout.render as render

    class Sensor:
        def __init__(self, *, model):
            self.default_render_config = SimpleNamespace()
            self.utils = SimpleNamespace(
                create_default_light=lambda **kwargs: None,
                compute_camera_rays_pinhole=lambda *args, **kwargs: None,
                create_color_image_output=lambda width, height, count: (width, height, count),
                to_rgba_from_color=lambda size: wp.zeros(
                    (model.world_count, size[2], size[0] * size[1], 4),
                    dtype=wp.uint8,
                    device="cpu",
                ),
            )

        def update(self, *args, **kwargs):
            pass

    monkeypatch.setattr(newton.sensors, "SensorTiledCamera", Sensor)
    monkeypatch.setattr(
        render,
        "_SmoothClothVisual",
        lambda *args: SimpleNamespace(sync=lambda runtime: None),
    )
    cameras = tuple(
        _camera(name, parent)
        for name, parent in zip(CAMERA_NAMES, ("head", "left/wrist", "right/wrist"), strict=True)
    )
    model = SimpleNamespace(
        world_count=2,
        body_count=6,
        device="cpu",
        body_label=["/Robot/head", "/Robot/left/wrist", "/Robot/right/wrist"] * 2,
        bvh_refit_shapes=lambda state: None,
    )
    return SimpleNamespace(
        model=model,
        world_offsets=np.array([[0, 0, 0], [10, 0, 0]]),
        robot_profile=SimpleNamespace(robot_id="fr3_duo", cameras=cameras),
    )


def test_missing_cam_low_fails_before_sensor_creation(camera_scene, monkeypatch):
    import newton.sensors

    def forbidden_sensor(**kwargs):
        pytest.fail("missing-camera validation must precede sensor allocation")

    monkeypatch.setattr(newton.sensors, "SensorTiledCamera", forbidden_sensor)
    with pytest.raises(ValueError, match=r"fr3_duo.*cam_low"):
        ShirtFlattenCameraObservation(camera_scene, camera_names=DATASET_CAMERA_NAMES)


@pytest.mark.parametrize("names", [CAMERA_NAMES, ("cam_high",)])
def test_every_body_mounted_camera_follows_its_parent(camera_scene, names):
    observation = ShirtFlattenCameraObservation(camera_scene, width=4, height=3, camera_names=names)
    transforms = [
        wp.transform((10 * world + body, 2, 3), wp.quat_identity())
        for world in range(2)
        for body in range(3)
    ]
    runtime = SimpleNamespace(
        scene=camera_scene,
        state_0=SimpleNamespace(body_q=wp.array(transforms, dtype=wp.transformf, device="cpu")),
    )
    result = observation(runtime)
    actual = observation._transforms.numpy()
    for camera_index in range(len(names)):
        for world in range(2):
            np.testing.assert_allclose(
                actual[camera_index, world, :3], [10 * world + camera_index, 2, 3.1]
            )
    assert set(result) == {f"observation.images.{name}" for name in names}
    assert result["observation.images.cam_high"].shape == (2, 3, 4, 3)


def test_isaac_reuses_native_aliased_cameras_without_changing_clipping():
    from pxr import Gf, Sdf, Usd, UsdGeom

    from clothdojo.data.isaac_render import _author_cameras

    stage = Usd.Stage.CreateInMemory()
    paths = []
    for index, name in enumerate(CAMERA_NAMES):
        camera = UsdGeom.Camera.Define(stage, f"/Robot/native_camera_{index}")
        camera.GetPrim().CreateAttribute("clothdojo:cameraName", Sdf.ValueTypeNames.String).Set(
            name
        )
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
        paths.append(str(camera.GetPath()))
    assert _author_cameras(stage, (), tuple(_camera(name) for name in CAMERA_NAMES)) == tuple(paths)
    assert len([prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]) == 3
    for path in paths:
        reused = UsdGeom.Camera(stage.GetPrimAtPath(path))
        assert reused.GetFocalLengthAttr().Get() == 20.0
        assert reused.GetHorizontalApertureAttr().Get() == 30.0
        assert reused.GetVerticalApertureAttr().Get() == 16.875
        assert reused.GetClippingRangeAttr().Get() == Gf.Vec2f(0.05, 20.0)


def test_isaac_authors_vertical_aperture_from_camera_spec(monkeypatch):
    from pxr import Usd, UsdGeom

    import clothdojo.teleop.isaac_fabric as fabric
    from clothdojo.data.isaac_render import _author_cameras

    monkeypatch.setattr(fabric, "_find_render_body_paths", lambda *args: {})
    stage = Usd.Stage.CreateInMemory()
    paths = _author_cameras(stage, (), (_camera(),))
    camera = UsdGeom.Camera(stage.GetPrimAtPath(paths[0]))
    assert camera.GetHorizontalApertureAttr().Get() == 30.0
    assert camera.GetVerticalApertureAttr().Get() == 16.875
