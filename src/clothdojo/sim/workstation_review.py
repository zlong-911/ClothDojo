"""Read-only Newton home/attach review of the registered workstation USDs."""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clothdojo.robots import robot_profile
from clothdojo.sim.scene import ShirtFlattenSpec, _add_robot

VIEWS = ("overall", "left", "right")
ALL_VIEWS = (*VIEWS, "left_wrist", "right_wrist")


@dataclass(frozen=True)
class WorkstationReview:
    model: object
    state: object
    report: dict


def _pose_matrix(position, rotation_xyzw):
    from pxr import Gf

    matrix = Gf.Matrix4d().SetRotate(
        Gf.Quatd(float(rotation_xyzw[3]), *map(float, rotation_xyzw[:3]))
    )
    matrix.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
    return matrix


def _apply_authored_appearance(builder, stage) -> dict:
    from newton import ShapeFlags
    from newton._src.usd.utils import resolve_material_properties_for_prim
    from pxr import UsdGeom

    display_colors, hidden_proxies = [], []
    for index, path in enumerate(builder.shape_label):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.proxy:
            if builder.shape_flags[index] & int(ShapeFlags.VISIBLE):
                hidden_proxies.append(path)
            builder.shape_flags[index] &= ~int(ShapeFlags.VISIBLE)
        # Newton 1.5 resolves bound materials but omits uniform displayColor.
        # Only bridge this omission in memory; never replace a bound material.
        if resolve_material_properties_for_prim(prim).get("color") is not None:
            continue
        attribute = prim.GetAttribute("primvars:displayColor")
        values = attribute.Get() if attribute and attribute.HasAuthoredValue() else None
        if values is not None and len(values) == 1:
            builder.shape_color[index] = tuple(map(float, values[0]))
            display_colors.append(path)
    return {
        "authored_display_color_fallbacks": display_colors,
        "hidden_authored_proxy_shapes": hidden_proxies,
    }


def _add_context(newton, wp, builder, robot_id, assets_root, spec):
    """Keep the accepted review backdrop separate from robot/physics truth."""
    cfg = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False, has_particle_collision=False, density=0.0
    )

    def box(label, center, half, color):
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
            hx=half[0],
            hy=half[1],
            hz=half[2],
            color=color,
            cfg=cfg,
            label=label,
        )

    if robot_id == "fr3_duo":
        import trimesh

        path = assets_root / "robots/fr3_duo/tabletop_context_mm.stl"
        mesh = trimesh.load(str(path), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32) * 0.001
        bounds = np.array((vertices.min(axis=0), vertices.max(axis=0)))
        expected = np.asarray(spec.table_position) + np.asarray(
            (
                -np.asarray(spec.table_half_extents),
                spec.table_half_extents,
            )
        )
        if not np.allclose(bounds, expected, atol=2.0e-6, rtol=0):
            raise ValueError("FR3 review hole plate does not match the scene table dimensions")
        builder.add_shape_mesh(
            body=-1,
            mesh=newton.Mesh(vertices, np.asarray(mesh.faces).reshape(-1)),
            cfg=cfg,
            color=(0.65, 0.68, 0.73),
            label="review/table",
        )
        table_source = str(path)
    else:
        box("review/table", spec.table_position, spec.table_half_extents, (0.65, 0.68, 0.73))
        table_source = "ShirtFlattenSpec"
    box("review/ground", (0.65, 0, -0.02), (3, 3, 0.02), (0.12, 0.14, 0.17))
    for index, (x, y) in enumerate(((0.1, -0.62), (0.1, 0.62), (1.2, -0.62), (1.2, 0.62))):
        box(f"review/leg_{index}", (x, y, 0.375), (0.025, 0.025, 0.375), (0.25, 0.28, 0.32))
    return {"table_source": table_source, "context_is_visual_only": True}


def prepare_workstation_review(
    robot_id: str,
    *,
    assets_root: str | Path,
    device: str = "cpu",
) -> WorkstationReview:
    """Import the production USD and compute home FK, without opening OpenGL."""
    import newton
    import warp as wp
    from pxr import Usd, UsdGeom

    assets_root = Path(assets_root).resolve()
    profile = robot_profile(robot_id, assets_root=assets_root)
    source = assets_root / profile.asset
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"could not open workstation USD: {source}")
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=wp.vec3(0))
        _add_robot(newton, wp, builder, source, profile, load_visual_shapes=True)
        robot_body_count, robot_shape_count = builder.body_count, builder.shape_count
        home_q = list(map(float, builder.joint_q))
        appearance = _apply_authored_appearance(builder, stage)
        context = _add_context(newton, wp, builder, robot_id, assets_root, ShirtFlattenSpec())
        if builder.body_count != robot_body_count or list(map(float, builder.joint_q)) != home_q:
            raise RuntimeError("review context changed robot state")
        model = builder.finalize(device=device)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    poses = state.body_q.numpy()
    body_indices = {name: index for index, name in enumerate(profile.body_names)}
    attach_prims = {
        str(prim.GetAttribute("clothdojo:parentBody").Get()): prim
        for prim in stage.Traverse()
        if prim.GetAttribute("clothdojo:role")
        and prim.GetAttribute("clothdojo:role").Get() == "end_effector_attach"
    }
    cache = UsdGeom.XformCache()
    attach = {}
    for side, frame in zip(("left", "right"), profile.attach_frames, strict=True):
        pose = poses[body_indices[frame.parent_body]]
        local = _pose_matrix(frame.position, frame.rotation_xyzw)
        world = np.asarray(local * _pose_matrix(pose[:3], pose[3:]))
        usd_world = np.asarray(cache.GetLocalToWorldTransform(attach_prims[frame.parent_body]))
        attach[side] = {
            "parent_body": frame.parent_body,
            "local_row_matrix": np.asarray(local).tolist(),
            "newton_home_world_row_matrix": world.tolist(),
            "usd_default_world_row_matrix": usd_world.tolist(),
            "usd_default_to_home_position_error_m": float(
                np.linalg.norm(usd_world[3, :3] - world[3, :3])
            ),
        }
    cameras = {}
    for camera in profile.cameras:
        world = np.asarray(camera.parent_from_camera).T
        if camera.parent_body is not None:
            pose = poses[body_indices[camera.parent_body]]
            world = world @ np.asarray(_pose_matrix(pose[:3], pose[3:]))
        cameras[camera.name] = {
            "parent_body": camera.parent_body,
            "newton_home_world_row_matrix": world.tolist(),
        }
    return WorkstationReview(
        model,
        state,
        {
            "robot": robot_id,
            "source": str(source),
            "renderer": "Newton ViewerGL",
            "physics_steps": 0,
            "source_assets_modified": False,
            "device": str(model.device),
            "home_q": home_q,
            "arm_joint_names": list(profile.arm_joint_names),
            "arm_home_q_rad": [home_q[index] for index in profile.arm_coord_indices],
            "gripper_home_q": [home_q[index] for index in profile.gripper_coord_indices],
            "robot_body_count": robot_body_count,
            "robot_shape_count": robot_shape_count,
            "total_shapes_with_context": model.shape_count,
            "visible_shapes": int(
                np.count_nonzero(model.shape_flags.numpy() & int(newton.ShapeFlags.VISIBLE))
            ),
            "attach": attach,
            "cameras": cameras,
            "axis_colors": {"X": "red", "Y": "green", "Z": "blue"},
            "matrix_convention": "USD row-vector matrices; translation in last row",
            **appearance,
            **context,
        },
    )


def _view_camera(view: str, attach: dict, cameras: dict | None = None):
    if view not in ALL_VIEWS:
        raise ValueError(f"unknown review view {view!r}; expected {ALL_VIEWS}")
    if view == "overall":
        eye = np.array((2.8, -2.6, 2.4))
        return eye, np.array((0.5, 0, 0.94)) - eye, 40.0
    if view.endswith("_wrist"):
        side = view.split("_")[0]
        name = f"cam_{side}_wrist"
        if cameras is None or name not in cameras:
            raise ValueError(f"{view} requires the asset-authored {name} frame")
        camera = np.asarray(cameras[name]["newton_home_world_row_matrix"])
        tip = np.asarray(attach[side]["newton_home_world_row_matrix"])[3, :3]
        origin = camera[3, :3]
        front = -camera[2, :3]  # USD cameras look along local -Z.
        front /= np.linalg.norm(front)
        separation = origin - tip
        away = separation - np.dot(separation, front) * front
        # Step to the camera's side of the gripper so the palm does not hide
        # its shell/adapter; an axial mount instead uses the camera's local up.
        if np.linalg.norm(away) < 1e-6:
            away = camera[1, :3]
        away /= np.linalg.norm(away)
        other = "right" if side == "left" else "left"
        other_tip = np.asarray(attach[other]["newton_home_world_row_matrix"])[3, :3]
        outward = tip - other_tip
        outward -= np.dot(outward, front) * front
        outward /= np.linalg.norm(outward)
        target = origin * 0.65 + tip * 0.35
        # The outboard side avoids looking through the other arm in a dual
        # workstation; bias the target toward the wrist to retain the adapter.
        offset = 0.6 * front + 0.5 * away + 0.9 * outward
        offset /= np.linalg.norm(offset)
        distance = max(0.5, 2.2 * np.linalg.norm(separation))
        eye = target + distance * offset
        spec = ShirtFlattenSpec()
        floor = spec.table_position[2] + spec.table_half_extents[2] + 0.045
        if eye[2] < floor:
            eye[2] = floor
            # Retain the framing distance when a downward optical axis would
            # otherwise put the external review viewpoint inside the table.
            horizontal = eye[:2] - target[:2]
            horizontal_distance = np.sqrt(max(distance**2 - (eye[2] - target[2]) ** 2, 0))
            eye[:2] = target[:2] + horizontal * horizontal_distance / np.linalg.norm(horizontal)
        return eye, target - eye, 42.0
    matrix = np.asarray(attach[view]["newton_home_world_row_matrix"])
    rotation = matrix[:3, :3].T
    target = matrix[3, :3] - rotation[:, 2] * 0.025
    eye = target + rotation @ np.array((0.27, 0.23, 0.32))
    # Keep downward-facing TCP close-ups above the table, as in the accepted review.
    if eye[2] < 0.85:
        eye[2] = max(0.9, target[2] + 0.18)
    fov = float(np.rad2deg(2 * np.arctan(27.0 / (2 * 55.0))))
    return eye, target - eye, fov


def _restart_headless_event_loop(event_loop) -> None:
    if event_loop.is_running:
        raise RuntimeError("workstation review requires its own headless pyglet event loop")
    # RendererGL.close() closes the last window, which leaves pyglet's global
    # has_exit flag true. Newton 1.5 checks that flag before every render but
    # does not reset it when constructing a new viewer. We manually step this
    # headless viewer, so mirror the public reset performed by EventLoop.run().
    event_loop.has_exit = False


def _validate_frame(rgb: np.ndarray) -> None:
    if rgb.shape != (900, 1200, 3):
        raise RuntimeError(f"unexpected ViewerGL frame dimensions: {rgb.shape}")
    if not np.any(rgb != rgb[0, 0]):
        raise RuntimeError("ViewerGL returned a blank uniform frame; review was not rendered")


def render_workstation_review(
    robot_id: str,
    *,
    assets_root: str | Path,
    output_dir: str | Path,
    device: str,
    egl_device: int,
    views: tuple[str, ...] = VIEWS,
) -> dict:
    """Write overall, attach, or wrist-assembly PNGs and their FK report.

    ``device`` is a Warp device (including ``cpu``); ``egl_device`` is pyglet's
    physical EGL ordinal, independent of CUDA_VISIBLE_DEVICES. The caller must
    choose both explicitly. Use a fresh process if changing EGL devices.
    Optional wrist views are external inspection viewpoints, not new robot
    cameras. No view changes the source USD or its camera calibration.
    """
    if egl_device < 0:
        raise ValueError("egl_device must be non-negative")
    if not views or any(view not in ALL_VIEWS for view in views):
        raise ValueError(f"views must be a non-empty subset of {ALL_VIEWS}")
    import pyglet

    pyglet.options["headless"] = True
    pyglet.options["headless_device"] = egl_device
    import newton
    import warp as wp
    from PIL import Image
    from pyglet import gl

    review = prepare_workstation_review(robot_id, assets_root=assets_root, device=device)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    starts, ends, colors = [], [], []
    for frame in review.report["attach"].values():
        matrix = np.asarray(frame["newton_home_world_row_matrix"])
        for index, color in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            starts.append(matrix[3, :3])
            ends.append(matrix[3, :3] + matrix[index, :3] * 0.075)
            colors.append(color)
    with wp.ScopedDevice(device):
        _restart_headless_event_loop(pyglet.app.event_loop)
        viewer = newton.viewer.ViewerGL(width=1200, height=900, headless=True)
        try:
            viewer.set_model(review.model)
            viewer.show_collision = viewer.show_joints = viewer.show_com = False
            viewer.show_visual = True
            viewer.show_static = False  # Do not force invisible static colliders visible.
            viewer.renderer.arrow_scale = 1.0
            arrays = [
                wp.array(values, dtype=wp.vec3, device=device) for values in (starts, ends, colors)
            ]
            images = {}
            for view in views:
                eye, direction, fov = _view_camera(
                    view, review.report["attach"], review.report["cameras"]
                )
                direction /= np.linalg.norm(direction)
                viewer.set_camera(
                    wp.vec3(*eye),
                    float(np.rad2deg(np.arcsin(direction[2]))),
                    float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
                )
                viewer.camera.fov = fov
                for _ in range(8):
                    viewer.begin_frame(0.0)
                    viewer.log_state(review.state)
                    viewer.log_arrows("attach_XYZ", *arrays)
                    viewer.end_frame()
                rgb = viewer.get_frame().numpy()
                _validate_frame(rgb)
                path = output_dir / f"{robot_id}_{view}.png"
                Image.fromarray(rgb).save(path)
                images[view] = str(path)
            report = {
                **review.report,
                "egl_device": egl_device,
                "gl_renderer": ctypes.string_at(gl.glGetString(gl.GL_RENDERER)).decode(),
                "images": images,
            }
            (output_dir / f"{robot_id}.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
        finally:
            viewer.close()
