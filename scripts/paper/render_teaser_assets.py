#!/usr/bin/env python3
"""Render canonical garment assets for the paper teaser.

Each garment is shown as a physically grounded studio flat-lay from the same
front three-quarter view.  One centimeters-to-meters conversion is shared by
all meshes so their relative physical sizes remain visible.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--pathtrace-spp", type=int, default=256)
    parser.add_argument("--assets-root", type=Path, default=Path("assets"))
    parser.add_argument("--visual-assets-root", type=Path, default=Path("data/visual-assets"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/isaac-paper"))
    return parser.parse_args()


def _canonical_points(vertices: np.ndarray) -> np.ndarray:
    """Map Style3D centimeters into one shared flat-lay world frame."""

    points = np.asarray(vertices, dtype=np.float32)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    # Style3D authored these vertices in centimeters.  A single conversion
    # preserves the relative physical size of every garment; do not normalize
    # individual meshes to their tile.
    points = (points - center) * 0.01
    # Source x/y are garment width/height and source z is depth.  Lay the
    # garment on the authored table (top surface z=0.8 m), anchoring only the
    # vertical offset so every mesh casts a plausible contact shadow.  The
    # shared 0.01 scale remains untouched.
    depth = points[:, 2] - points[:, 2].min()
    return np.column_stack(
        (
            0.651 - points[:, 1],
            -points[:, 0],
            0.802 + depth,
        )
    ).astype(np.float32)


def _look_at_camera(
    *,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[tuple[float, ...], ...]:
    """Return a USD camera-to-world transform with local -Z looking at target."""

    eye_vector = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye_vector
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    right /= np.linalg.norm(right)
    back = -forward
    up = np.cross(back, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = back
    matrix[:3, 3] = eye_vector
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _stage_flatlay(stage, visual_assets_root: Path) -> None:
    """Turn the task stage into a coherent studio flat-lay set."""

    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    from clothdojo.teleop.isaac_stage import _define_textured_cube, _set_cube

    for path in ("/World/Robot", "/World/Table", "/World/Ground"):
        prim = stage.GetPrimAtPath(path)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()
    geniesim_hdri = (
        visual_assets_root / "geniesim-3.2/background/common/HDR/room/ferndale_studio_01_4k.hdr"
    ).resolve()
    if not geniesim_hdri.is_file():
        raise FileNotFoundError(f"GenieSim studio HDRI is missing: {geniesim_hdri}")
    dome = stage.GetPrimAtPath("/World/Lighting/Dome")
    if not dome:
        raise RuntimeError("flat-lay dome light is unavailable")
    dome.GetAttribute("inputs:texture:file").Set(Sdf.AssetPath(str(geniesim_hdri)))
    dome.GetAttribute("inputs:exposure").Set(-0.35)
    surface = _define_textured_cube(
        stage,
        "/World/AssetStudioSurface",
        (2.0, 2.0, 0.025),
        physical_uvs=True,
    )
    _set_cube(surface, (0.651, 0.0, 0.775), (2.0, 2.0, 0.025))
    table_material = UsdShade.Material.Get(stage, "/World/Looks/RandomizedTable")
    if not table_material:
        raise RuntimeError("flat-lay table material is unavailable")
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(table_material)
    for name, position, radius, intensity in (
        ("StudioKey", (1.35, -1.15, 2.75), 0.72, 3200.0),
        ("StudioFill", (-0.25, 1.10, 2.15), 0.95, 1050.0),
    ):
        light = UsdLux.SphereLight.Define(stage, f"/World/Lighting/{name}")
        light.CreateRadiusAttr(radius)
        light.CreateIntensityAttr(intensity)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*position))


def _paper_cloth_appearance(job: dict, catalog, appearance_type):
    """Resolve an optional, catalog-validated paper-only cloth override."""

    material_id = job.get("paper_cloth_material")
    color_channel = job.get("paper_cloth_color_channel")
    if material_id is None and color_channel is None:
        return None
    if not isinstance(material_id, str) or not material_id:
        raise ValueError("paper_cloth_material must be a non-empty string")
    material = catalog.material(material_id)
    if material.role != "cloth":
        raise ValueError(f"paper cloth material is not cloth: {material_id}")
    if color_channel is None:
        if len(material.color_channels) != 1:
            raise ValueError(f"paper_cloth_color_channel is required for {material_id}")
        color_channel = material.color_channels[0]
    if color_channel not in material.color_channels:
        raise ValueError(f"unknown paper cloth color channel {color_channel!r} for {material_id}")
    return appearance_type(material=material_id, color_channel=color_channel)


def main() -> None:
    args = _arguments()
    plan_path = args.plan.expanduser().resolve()
    output = args.output.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    visual_assets_root = args.visual_assets_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = [job for job in plan["jobs"] if job.get("group") == "assets"]
    if args.job:
        selected = set(args.job)
        jobs = [job for job in jobs if job.get("name") in selected]
        missing = selected.difference(job.get("name") for job in jobs)
        if missing:
            raise ValueError(f"unknown asset jobs: {sorted(missing)}")
    if not jobs:
        raise ValueError("paper render plan has no asset jobs")
    # Reopening the render stage for another garment in the same Kit process is
    # unstable in Isaac Sim 5.  Isolate each asset while keeping a multi-job CLI
    # convenient and reproducible.
    if len(jobs) > 1:
        for job in jobs:
            subprocess.run(
                (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(output),
                    "--gpu",
                    str(args.gpu),
                    "--job",
                    job["name"],
                    "--width",
                    str(args.width),
                    "--height",
                    str(args.height),
                    "--pathtrace-spp",
                    str(args.pathtrace_spp),
                    "--assets-root",
                    str(assets_root),
                    "--visual-assets-root",
                    str(visual_assets_root),
                    "--cache-root",
                    str(cache_root),
                ),
                check=True,
            )
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

    from clothdojo.assets.materials import load_material_catalog
    from clothdojo.data.isaac_render import (
        IsaacRandomizedCamera,
        LightingAppearance,
        MaterialAppearance,
        _preload_python_libstdcpp,
        _require_isaac5_runtime,
        _require_visual_assets,
        _start_isaac_app,
        sample_visual_layout,
    )
    from clothdojo.rendering import CameraSpec

    _require_isaac5_runtime()
    _require_visual_assets(visual_assets_root, load_material_catalog(), include_room=False)
    _preload_python_libstdcpp()
    from clothdojo.teleop.isaac_compat import configure_replicator_warp

    configure_replicator_warp()
    portable = cache_root / "clothdojo" / "isaac5-paper" / f"{socket.gethostname()}-gpu-{args.gpu}"
    portable.mkdir(parents=True, exist_ok=True)
    app = _start_isaac_app(args.gpu, portable)
    camera_spec = CameraSpec(
        name="asset_view",
        parent_body=None,
        parent_from_camera=_look_at_camera(
            eye=(1.12, -0.25, 2.30),
            target=(0.651, 0.0, 0.80),
        ),
        focal_length=26.0,
        horizontal_aperture=24.0,
        vertical_aperture=24.0,
        near_clip=0.01,
    )
    records = []
    existing_manifest = output / "asset_manifest.json"
    if args.job and existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        selected = set(args.job)
        records = [
            record for record in existing.get("records", []) if record.get("name") not in selected
        ]
    try:
        from clothdojo.robots import robot_profile
        from clothdojo.sim.cloth import load_garment_mesh
        from clothdojo.trajectory import TrajectoryReader

        profile = robot_profile("piper", assets_root=assets_root)
        catalog = load_material_catalog()
        for index, job in enumerate(jobs):
            reader = TrajectoryReader(Path(job["trajectory"]).expanduser().resolve())
            asset_id = reader.manifest.metadata["asset_id"]
            asset_path = assets_root / "garments/converted" / asset_id / f"{asset_id}.usdc"
            vertices, _, _, _ = load_garment_mesh(asset_path)
            state = replace(
                reader.state_at(0),
                particle_q=_canonical_points(vertices),
                particle_qd=np.zeros_like(vertices, dtype=np.float32),
            )
            layouts: list[dict] = []
            camera = IsaacRandomizedCamera(
                app,
                state,
                asset_path=asset_path,
                assets_root=assets_root,
                visual_assets_root=visual_assets_root,
                device="cuda:0",
                base_seed=int(job.get("seed", index)),
                include_room=False,
                randomize=("cloth", "table"),
                layouts=layouts,
                robot_profile=profile,
                cameras=(camera_spec,),
                render_resolution=(args.width, args.height),
                pathtrace_spp=args.pathtrace_spp,
            )
            try:
                layout = sample_visual_layout(
                    int(job.get("seed", index)),
                    index,
                    source_episode_id=f"paper-asset-{asset_id}",
                    include_room=False,
                    randomize=("cloth", "table"),
                )
                layout = replace(
                    layout,
                    cloth=(
                        _paper_cloth_appearance(job, catalog, MaterialAppearance) or layout.cloth
                    ),
                    table=MaterialAppearance(
                        material="marble_01",
                        color_channel="Diffuse",
                    ),
                    lighting=LightingAppearance(
                        environment="studio_small_08",
                        exposure=-0.18,
                        rotation_z=32.0,
                    ),
                )
                camera.begin_episode(
                    index,
                    source_episode_id=layout.source_episode_id,
                    layout=layout,
                )
                _stage_flatlay(camera._stage, visual_assets_root)
                for _ in range(3):
                    app.update()
                image = camera(state)["observation.images.asset_view"][0]
                path = output / "assets" / f"{job['name']}_00.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image.astype(np.uint8), "RGB").save(path)
                records.append(
                    {
                        "name": job["name"],
                        "asset_id": asset_id,
                        "image": str(path.relative_to(output)),
                        "appearance": layouts[-1],
                    }
                )
                print(f"rendered {path}", flush=True)
            finally:
                camera.close()
        (output / "asset_manifest.json").write_text(
            json.dumps({"schema": "clothdojo_paper_asset_renders_v1", "records": records}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    finally:
        if app.is_running():
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
