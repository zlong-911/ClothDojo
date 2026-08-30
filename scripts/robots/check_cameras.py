"""Read-only home views through every native workstation USD camera.

Uses Newton ViewerGL and the USD frustum (including roll, both apertures,
offsets and clipping). This is an asset inspection, not rollout pixel parity
or an Isaac RTX material certification. No source camera/geometry is changed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import numpy as np
from pxr import Gf, Usd, UsdGeom

from clothdojo.robots import robot_ids
from clothdojo.sim.scene import ShirtFlattenSpec
from clothdojo.sim.workstation_review import (
    _restart_headless_event_loop,
    prepare_workstation_review,
)


def native_camera(prim, world_row_matrix):
    camera = UsdGeom.Camera(prim).GetCamera(Usd.TimeCode.Default())
    camera.transform = Gf.Matrix4d(*np.asarray(world_row_matrix).ravel().tolist())
    return camera


def gl_camera(source, width, height):
    from newton._src.viewer.camera import Camera

    class UsdCamera(Camera):
        def __init__(self):
            self.frustum = source.frustum
            self.world = np.asarray(source.transform)
            super().__init__(
                pos=self.world[3, :3], width=width, height=height,
                near=source.clippingRange.min, far=source.clippingRange.max,
            )

        def get_front(self):
            return self._as_vec3(-self.world[2, :3]).normalize()

        def get_up(self):
            return self._as_vec3(self.world[1, :3]).normalize()

        def get_right(self):
            return self._as_vec3(self.world[0, :3]).normalize()

        def get_view_matrix(self, scaling=1.0):
            if scaling != 1.0:
                raise ValueError("native camera inspection must retain meter-scale geometry")
            # Gf uses row matrices; flattened rows are GL's column-major storage.
            return np.asarray(self.frustum.ComputeViewMatrix(), dtype=np.float32).ravel()

        def get_projection_matrix(self):
            return np.asarray(self.frustum.ComputeProjectionMatrix(), dtype=np.float32).ravel()

    return UsdCamera()


def projected_point(camera, point, width, height):
    frustum = camera.frustum
    clip = np.r_[point, 1.0] @ np.asarray(frustum.ComputeViewMatrix()) @ np.asarray(
        frustum.ComputeProjectionMatrix()
    )
    if abs(clip[3]) < 1e-12:
        return {"pixel_xy": None, "in_frustum": False}
    ndc = clip[:3] / clip[3]
    return {
        "pixel_xy": [float((ndc[0] + 1) * width / 2), float((1 - ndc[1]) * height / 2)],
        "in_frustum": bool(clip[3] > 0 and np.all(np.abs(ndc) <= 1)),
    }


def render(robot_id, args):
    import newton
    import pyglet
    import warp as wp
    from PIL import Image, ImageDraw
    from pyglet import gl

    width, height = 640, 480
    review = prepare_workstation_review(robot_id, assets_root=args.assets, device=args.device)
    source_path = Path(review.report["source"])
    source_before = source_path.read_bytes()
    stage = Usd.Stage.Open(str(source_path))
    prims = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            name = prim.GetAttribute("clothdojo:cameraName").Get() or prim.GetName()
            if name in review.report["cameras"]:
                if name in prims:
                    raise ValueError(f"duplicate native camera {name}")
                prims[name] = prim
    folder = args.output / robot_id
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "robot": robot_id, "source": str(source_path), "renderer": "Newton ViewerGL",
        "pose": "authored home", "physics_steps": 0, "output_size": [width, height],
        "projection": "USD frustum, no aspect conform or horizon leveling",
        "source_assets_modified": False, "cameras": {},
    }
    sheet = Image.new("RGB", (width * 2, (height + 32) * 2), (32, 35, 40))
    draw = ImageDraw.Draw(sheet)
    table = ShirtFlattenSpec()
    table_z = table.table_position[2] + table.table_half_extents[2]
    with wp.ScopedDevice(args.device):
        _restart_headless_event_loop(pyglet.app.event_loop)
        viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
        try:
            viewer.set_model(review.model)
            viewer.show_collision = viewer.show_joints = viewer.show_com = False
            viewer.show_visual = True
            viewer.show_static = False  # Preserve authored invisible collision geometry.
            for index, (name, frame) in enumerate(review.report["cameras"].items()):
                prim = prims[name]
                camera = native_camera(prim, frame["newton_home_world_row_matrix"])
                viewer.camera = gl_camera(camera, width, height)
                for _ in range(8):
                    viewer.begin_frame(0.0)
                    viewer.log_state(review.state)
                    viewer.end_frame()
                rgb = viewer.get_frame().numpy()
                if rgb.shape != (height, width, 3):
                    raise RuntimeError(f"incorrect framebuffer shape {rgb.shape}")
                image = Image.fromarray(rgb)
                path = folder / f"{name}.png"
                image.save(path)
                position = np.asarray(camera.transform)[3, :3]
                front = -np.asarray(camera.transform)[2, :3]
                distance = (table_z - position[2]) / front[2] if abs(front[2]) > 1e-9 else -1
                report["cameras"][name] = {
                    "prim": str(prim.GetPath()), "parent_body": frame["parent_body"],
                    "image": str(path), "world_row_matrix": np.asarray(camera.transform).tolist(),
                    "focal_length": camera.focalLength,
                    "apertures": [camera.horizontalAperture, camera.verticalAperture],
                    "aperture_offsets": [
                        camera.horizontalApertureOffset, camera.verticalApertureOffset,
                    ],
                    "clipping": [camera.clippingRange.min, camera.clippingRange.max],
                    "projection_row_matrix": np.asarray(
                        camera.frustum.ComputeProjectionMatrix()
                    ).tolist(),
                    "optical_axis_table_intersection": (
                        (position + distance * front).tolist() if distance > 0 else None
                    ),
                    "attach_projections": {
                        side: projected_point(
                            camera, np.asarray(attach["newton_home_world_row_matrix"])[3, :3],
                            width, height,
                        ) for side, attach in review.report["attach"].items()
                    },
                    "rgb_std": float(rgb.std()),
                    "black_pixel_fraction": float(np.mean(rgb.max(axis=2) < 3)),
                    "uniform": bool(np.all(rgb == rgb[0, 0])),
                }
                x, y = index % 2 * width, index // 2 * (height + 32)
                draw.text((x + 10, y + 9), f"{robot_id} / {name}", fill="white")
                sheet.paste(image, (x, y + 32))
                print(robot_id, name, "rgb_std", round(float(rgb.std()), 3), flush=True)
            report["gl_renderer"] = ctypes.string_at(gl.glGetString(gl.GL_RENDERER)).decode()
        finally:
            viewer.close()
    if source_path.read_bytes() != source_before:
        raise RuntimeError("workstation changed during read-only camera inspection")
    sheet.save(folder / "overview.jpg", quality=95)
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("all", *robot_ids()), default="all")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/native_camera_review"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--egl-device", type=int, required=True)
    args = parser.parse_args()
    args.assets, args.output = args.assets.resolve(), args.output.resolve()
    import pyglet

    pyglet.options["headless"] = True
    pyglet.options["headless_device"] = args.egl_device
    for robot_id in robot_ids() if args.robot == "all" else (args.robot,):
        render(robot_id, args)


if __name__ == "__main__":
    main()
