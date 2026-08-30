from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.robots import robot_ids, robot_profile
from clothdojo.sim.workstation_review import (
    ALL_VIEWS,
    VIEWS,
    _apply_authored_appearance,
    _restart_headless_event_loop,
    _validate_frame,
    _view_camera,
    prepare_workstation_review,
    render_workstation_review,
)

ASSETS = Path(__file__).resolve().parents[1] / "assets"


@pytest.mark.parametrize("robot_id", robot_ids())
def test_review_uses_registered_home_and_attach_without_temporary_inputs(robot_id):
    import newton
    import warp as wp

    profile = robot_profile(robot_id)
    source = ASSETS / profile.asset
    source_before = source.read_bytes()
    review = prepare_workstation_review(robot_id, assets_root=ASSETS)
    report = review.report
    assert report["source"] == str(source)
    assert report["device"] == "cpu"
    assert report["physics_steps"] == 0
    assert report["source_assets_modified"] is False
    assert report["context_is_visual_only"] is True
    assert source.read_bytes() == source_before
    np.testing.assert_allclose(report["home_q"], profile.home_joint_q, atol=2e-6, rtol=0)
    assert report["robot_body_count"] == len(profile.body_names)
    assert report["total_shapes_with_context"] == report["robot_shape_count"] + 6
    flags = review.model.shape_flags.numpy()[report["robot_shape_count"] :]
    assert np.all(flags & int(newton.ShapeFlags.VISIBLE))
    assert not np.any(flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
    assert not np.any(flags & int(newton.ShapeFlags.COLLIDE_PARTICLES))
    assert set(report["attach"]) == {"left", "right"}
    assert set(report["cameras"]) == {camera.name for camera in profile.cameras}
    poses = review.state.body_q.numpy()
    for camera in profile.cameras:
        local = np.asarray(camera.parent_from_camera)
        expected_origin = local[:3, 3]
        if camera.parent_body is not None:
            pose = poses[profile.body_names.index(camera.parent_body)]
            expected_origin = pose[:3] + np.asarray(
                wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(*expected_origin))
            )
        frame = report["cameras"][camera.name]
        assert frame["parent_body"] == camera.parent_body
        np.testing.assert_allclose(
            np.asarray(frame["newton_home_world_row_matrix"])[3, :3], expected_origin,
            atol=2e-6,
        )
    for side, frame in zip(("left", "right"), profile.attach_frames, strict=True):
        attach = report["attach"][side]
        assert attach["parent_body"] == frame.parent_body
        np.testing.assert_allclose(np.asarray(attach["local_row_matrix"])[3, :3], frame.position)
        world = np.asarray(attach["newton_home_world_row_matrix"])
        assert np.all(np.isfinite(world))
        # FK quaternion products accumulate float32 round-off along the chain.
        np.testing.assert_allclose(world[:3, :3] @ world[:3, :3].T, np.eye(3), atol=1e-5)
    for view in VIEWS:
        eye, direction, fov = _view_camera(view, report["attach"])
        assert eye[2] >= 0.85
        assert np.linalg.norm(direction) > 0
        assert 0 < fov < 90
    for view in ("left_wrist", "right_wrist"):
        eye, direction, fov = _view_camera(view, report["attach"], report["cameras"])
        assert np.all(np.isfinite(eye))
        assert eye[2] >= 0.845 - 1e-8
        assert np.linalg.norm(direction) >= 0.5 - 1e-8
        assert 0 < fov < 90


def test_attach_closeup_stays_above_table_for_downward_tcp():
    matrix = np.diag((1.0, -1.0, -1.0, 1.0))
    matrix[3, :3] = (0.7, 0.2, 0.82)
    eye, direction, _ = _view_camera(
        "left",
        {
            "left": {"newton_home_world_row_matrix": matrix.tolist()},
        },
    )
    assert eye[2] >= 0.9
    assert direction[2] < 0


def test_wrist_views_use_the_camera_front_and_outboard_side_without_changing_defaults():
    assert VIEWS == ("overall", "left", "right")
    assert ALL_VIEWS == (*VIEWS, "left_wrist", "right_wrist")
    attach, cameras = {}, {}
    for side, sign in (("left", 1), ("right", -1)):
        tip = np.eye(4)
        tip[3, :3] = (0.7, sign * 0.4, 0.89)
        camera = np.eye(4)
        camera[3, :3] = (0.62, sign * 0.4, 1.04)
        attach[side] = {"newton_home_world_row_matrix": tip.tolist()}
        cameras[f"cam_{side}_wrist"] = {"newton_home_world_row_matrix": camera.tolist()}
    for side, sign in (("left", 1), ("right", -1)):
        eye, direction, _ = _view_camera(f"{side}_wrist", attach, cameras)
        tip = np.asarray(attach[side]["newton_home_world_row_matrix"])[3, :3]
        camera = np.asarray(cameras[f"cam_{side}_wrist"]["newton_home_world_row_matrix"])
        assert np.dot(eye - camera[3, :3], -camera[2, :3]) > 0
        assert sign * (eye[1] - tip[1]) > 0
        np.testing.assert_allclose(eye + direction, 0.65 * camera[3, :3] + 0.35 * tip)
    with pytest.raises(ValueError, match="asset-authored"):
        _view_camera("left_wrist", attach)


def test_review_reads_profile_and_geometry_from_the_same_deployed_assets(tmp_path, monkeypatch):
    import clothdojo.sim.workstation_review as module

    deployed = tmp_path / "deployed_assets"
    deployed.mkdir()
    (deployed / "robots").symlink_to(ASSETS / "robots", target_is_directory=True)
    requested_roots = []

    def profile_at_root(robot_id, *, assets_root=None):
        requested_roots.append(assets_root)
        return robot_profile(robot_id, assets_root=assets_root)

    monkeypatch.setattr(module, "robot_profile", profile_at_root)
    review = prepare_workstation_review("piper", assets_root=deployed)
    assert requested_roots == [deployed]
    assert review.report["source"] == str(deployed / "robots/piper/piper.usda")


def test_preview_preserves_bound_material_and_hides_authored_collision_proxy():
    import newton
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    proxy = UsdGeom.Mesh.Define(stage, "/Proxy")
    proxy.CreatePurposeAttr(UsdGeom.Tokens.proxy)
    proxy.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.4, 0.5)])
    visual = UsdGeom.Mesh.Define(stage, "/Visual")
    visual.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.9, 0.9)])
    material = UsdShade.Material.Define(stage, "/Material")
    shader = UsdShade.Shader.Define(stage, "/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    visible = int(newton.ShapeFlags.VISIBLE)
    collision = int(newton.ShapeFlags.COLLIDE_SHAPES)
    builder = SimpleNamespace(
        shape_label=["/Proxy", "/Visual"],
        shape_flags=[visible | collision, visible],
        shape_color=[(1.0, 1.0, 1.0), (0.1, 0.1, 0.1)],
    )
    before = stage.GetRootLayer().ExportToString()
    changes = _apply_authored_appearance(builder, stage)
    assert builder.shape_flags == [collision, visible]
    # USD displayColor defaults to linear RGB; imported bound colors stay intact.
    np.testing.assert_allclose(
        builder.shape_color, ((0.58383149, 0.66518508, 0.73535698), (0.1, 0.1, 0.1)),
        atol=1e-7,
    )
    assert changes["authored_display_color_fallbacks"] == ["/Proxy"]
    assert changes["hidden_authored_proxy_shapes"] == ["/Proxy"]
    assert stage.GetRootLayer().ExportToString() == before


def test_review_rejects_invalid_options_before_initializing_gl(tmp_path):
    with pytest.raises(ValueError, match="egl_device"):
        render_workstation_review(
            "piper", assets_root=ASSETS, output_dir=tmp_path, device="cpu", egl_device=-1
        )
    with pytest.raises(ValueError, match="views"):
        render_workstation_review(
            "piper",
            assets_root=ASSETS,
            output_dir=tmp_path,
            device="cpu",
            egl_device=0,
            views=("typo",),
        )


def test_headless_review_restarts_pyglets_public_exit_flag_between_viewers():
    loop = SimpleNamespace(is_running=False, has_exit=True)
    _restart_headless_event_loop(loop)
    assert loop.has_exit is False
    loop.has_exit = True  # Closing the last window requests exit again.
    _restart_headless_event_loop(loop)
    assert loop.has_exit is False
    loop.is_running = True
    with pytest.raises(RuntimeError, match="own headless"):
        _restart_headless_event_loop(loop)


@pytest.mark.parametrize("color", ((0, 0, 0), (255, 255, 255), (10, 20, 30)))
def test_review_rejects_blank_frame_even_when_dimensions_are_correct(color):
    frame = np.full((900, 1200, 3), color, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="blank"):
        _validate_frame(frame)
    frame[1, 1] = (40, 50, 60)
    _validate_frame(frame)


def test_review_module_has_no_temporary_imports_or_preview_paths():
    import clothdojo.sim.workstation_review as module

    source = Path(module.__file__).read_text()
    assert "outputs/" not in source
    assert "sys.path" not in source
    assert "audit.json" not in source
