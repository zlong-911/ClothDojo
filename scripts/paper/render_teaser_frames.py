#!/usr/bin/env python3
"""Render selected ClothDojo trajectory states for the paper teaser.

Unlike the dataset exporter, this utility renders only the requested states as
high-resolution, lossless PNG files with paper-specific cameras and rooms.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_PAPER_ROOMS = ("azure", "blue", "teal")


@dataclass(frozen=True)
class _PaperEnvironment:
    asset_pack: str
    source: Path
    floor_prim: str | None = None


@dataclass(frozen=True)
class _PaperTableMaterial:
    source: Path
    material_prim: str


_PAPER_ENVIRONMENTS = {
    "isaac_office": _PaperEnvironment("isaac-5.0", Path("Isaac/Environments/Office/office.usd")),
    "isaac_hospital": _PaperEnvironment(
        "isaac-5.0", Path("Isaac/Environments/Hospital/hospital.usd")
    ),
    "isaac_warehouse": _PaperEnvironment(
        "isaac-5.0",
        Path("Isaac/Environments/Simple_Warehouse/warehouse.usd"),
    ),
    "isaac_digital_twin_warehouse": _PaperEnvironment(
        "isaac-5.0",
        Path("Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd"),
        "Structure/floor",
    ),
    "geniesim_laboratory": _PaperEnvironment(
        "geniesim-3.2",
        Path("background/common/laboratory/Aligned.usd"),
        "body/visual",
    ),
    "geniesim_home_03": _PaperEnvironment(
        "geniesim-3.2",
        Path("background/common/home_03/Aligned.usd"),
    ),
    "geniesim_factory": _PaperEnvironment(
        "geniesim-3.2",
        Path("background/common/factory/background.usd"),
        "SM_Ground",
    ),
    "geniesim_restaurant": _PaperEnvironment(
        "geniesim-3.2",
        Path("background/common/restaurant/Aligned.usd"),
    ),
    "geniesim_dining_room": _PaperEnvironment(
        "geniesim-3.2",
        Path("background/common/dining_room/benchmark_dining_room_000/Aligned.usd"),
    ),
    "geniesim_market": _PaperEnvironment(
        "geniesim-3.2",
        Path("background/common/market/Aligned.usd"),
    ),
    "geniesim_home_02": _PaperEnvironment(
        "geniesim-3.2",
        Path("background/common/home_02/Aligned.usd"),
        "part_00/visual",
    ),
    "geniesim_gdk_market": _PaperEnvironment(
        "geniesim-3.2",
        Path("scenes/GDK/demo.usda"),
    ),
}

_PAPER_TABLE_MATERIALS = {
    "geniesim_dining_white_marble": _PaperTableMaterial(
        Path("background/common/dining_room/benchmark_dining_room_000/Aligned.usd"),
        "/World/Looks/ml_part_02",
    ),
    "geniesim_dining_gray_marble": _PaperTableMaterial(
        Path("background/common/dining_room/benchmark_dining_room_000/Aligned.usd"),
        "/World/Looks/ml_part_00",
    ),
    "geniesim_dining_light_wood": _PaperTableMaterial(
        Path("background/common/dining_room/benchmark_dining_room_000/Aligned.usd"),
        "/World/Looks/ml_part_06",
    ),
    "geniesim_lab_blue_terrazzo": _PaperTableMaterial(
        Path("background/common/laboratory/Aligned.usd"),
        "/World/Looks/ml_body",
    ),
    "geniesim_lab_cream": _PaperTableMaterial(
        Path("background/common/laboratory/Aligned.usd"),
        "/World/Looks/ml_part_01",
    ),
    "geniesim_lab_black_marble": _PaperTableMaterial(
        Path("background/common/laboratory/Aligned.usd"),
        "/World/Looks/ml_part_03",
    ),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--variant-order", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--pathtrace-spp", type=int, default=256)
    parser.add_argument("--assets-root", type=Path, default=Path("assets"))
    parser.add_argument("--visual-assets-root", type=Path, default=Path("data/visual-assets"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/isaac-paper"))
    parser.add_argument("--include-room", action="store_true")
    return parser.parse_args()


def _read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "clothdojo_paper_teaser_render_v1":
        raise ValueError(f"unsupported teaser render plan: {path}")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("teaser render plan must contain jobs")
    names = [job.get("name") for job in jobs]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every teaser render job needs a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("teaser render job names must be unique")
    return plan


def _frame_index(value: Any, state_count: int) -> int:
    if isinstance(value, bool):
        raise TypeError("frame selectors cannot be boolean")
    if isinstance(value, int):
        index = value
    elif isinstance(value, float) and 0.0 <= value <= 1.0:
        index = round(value * (state_count - 1))
    else:
        raise ValueError("frame selectors must be indices or fractions in [0, 1]")
    if index < 0 or index >= state_count:
        raise IndexError(f"frame {index} is outside [0, {state_count})")
    return index


def _save_rgb(path: Path, array: np.ndarray) -> None:
    image = np.asarray(array)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"unexpected camera image shape: {image.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image[:, :, :3].astype(np.uint8, copy=False), "RGB").save(path)


def _paper_layout(
    base_layout,
    *,
    job: dict[str, Any],
    include_room: bool,
):
    """Keep teaser variation visible without allowing large color or exposure shifts."""

    from clothdojo.assets.materials import load_material_catalog
    from clothdojo.data.isaac_render import LightingAppearance, MaterialAppearance

    rng = np.random.default_rng(base_layout.seed)
    selected_room = job.get("room") or _PAPER_ROOMS[base_layout.seed % len(_PAPER_ROOMS)]
    exposure_bias = float(job.get("paper_exposure_bias", 0.0))
    cloth = base_layout.cloth
    material_id = job.get("paper_cloth_material")
    color_channel = job.get("paper_cloth_color_channel")
    if material_id is not None or color_channel is not None:
        if not isinstance(material_id, str) or not material_id:
            raise ValueError("paper_cloth_material must be a non-empty string")
        material = load_material_catalog().material(material_id)
        if material.role != "cloth":
            raise ValueError(f"paper cloth material is not cloth: {material_id}")
        if color_channel is None:
            if len(material.color_channels) != 1:
                raise ValueError(f"paper_cloth_color_channel is required for {material_id}")
            color_channel = material.color_channels[0]
        if color_channel not in material.color_channels:
            raise ValueError(
                f"unknown paper cloth color channel {color_channel!r} for {material_id}"
            )
        cloth = MaterialAppearance(material=material_id, color_channel=color_channel)
    return replace(
        base_layout,
        cloth=cloth,
        lighting=LightingAppearance(
            environment="studio_small_08",
            exposure=round(
                0.03 + exposure_bias + float(rng.uniform(-0.10, 0.10)),
                6,
            ),
            rotation_z=base_layout.lighting.rotation_z,
        ),
        room=f"Simple_Room_{selected_room}" if include_room else None,
    )


def _look_at_camera(job: dict[str, Any]):
    """Author an independent paper camera instead of reusing PIPER calibration."""

    from clothdojo.rendering import CameraSpec

    eye = np.asarray(job.get("camera_eye", (2.55, -1.55, 1.80)), dtype=np.float64)
    target = np.asarray(job.get("camera_target", (0.55, 0.0, 0.86)), dtype=np.float64)
    if eye.shape != (3,) or target.shape != (3,):
        raise ValueError("paper camera eye and target must contain three values")
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    right /= np.linalg.norm(right)
    back = -forward
    up = np.cross(back, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = back
    matrix[:3, 3] = eye
    aperture = float(job.get("camera_aperture", 36.0))
    return CameraSpec(
        name="paper_view",
        parent_body=None,
        parent_from_camera=tuple(tuple(float(value) for value in row) for row in matrix),
        focal_length=float(job.get("camera_focal_length", 40.0)),
        horizontal_aperture=aperture,
        vertical_aperture=aperture,
        near_clip=0.01,
    )


def _job_camera(profile, job: dict[str, Any]):
    """Select either the paper-only camera or one native workstation camera."""

    camera_name = str(job.get("camera", "paper_view"))
    if camera_name == "paper_view":
        return _look_at_camera(job)
    matches = tuple(camera for camera in profile.cameras if camera.name == camera_name)
    if len(matches) != 1:
        available = ", ".join(camera.name for camera in profile.cameras)
        raise ValueError(
            f"camera {camera_name!r} is unavailable for {profile.robot_id}; "
            f"available native cameras: {available}"
        )
    camera = matches[0]
    focal_length = job.get("camera_focal_length")
    if focal_length is not None:
        focal_length = float(focal_length)
        if focal_length <= 0.0:
            raise ValueError("camera_focal_length must be positive")
        camera = replace(camera, focal_length=focal_length)
    return camera


def _apply_paper_camera_optics(stage, job: dict[str, Any]) -> dict[str, float]:
    """Optionally focus the paper camera on the workcell using physical DOF."""

    f_stop = job.get("camera_f_stop")
    if f_stop is None:
        return {}
    f_stop = float(f_stop)
    if f_stop <= 0.0:
        raise ValueError("camera_f_stop must be positive")
    eye = np.asarray(job.get("camera_eye", (2.55, -1.55, 1.80)), dtype=np.float64)
    target = np.asarray(job.get("camera_target", (0.55, 0.0, 0.86)), dtype=np.float64)
    focus_distance = float(job.get("camera_focus_distance", np.linalg.norm(target - eye)))
    if focus_distance <= 0.0:
        raise ValueError("camera_focus_distance must be positive")

    from pxr import UsdGeom

    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cameras/paper_view"))
    if not camera:
        raise RuntimeError("paper camera prim is unavailable")
    camera.CreateFStopAttr().Set(f_stop)
    camera.CreateFocusDistanceAttr().Set(focus_distance)
    return {"f_stop": f_stop, "focus_distance": focus_distance}


def _apply_native_camera_focal_override(stage, camera_path: str, job: dict[str, Any]) -> None:
    """Override only the lens of an embedded workstation camera for paper probes."""

    focal_length = job.get("camera_focal_length")
    if focal_length is None or str(job.get("camera", "paper_view")) == "paper_view":
        return
    focal_length = float(focal_length)
    if focal_length <= 0.0:
        raise ValueError("camera_focal_length must be positive")

    from pxr import UsdGeom

    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    if not camera:
        raise RuntimeError(f"native paper camera is unavailable: {camera_path}")
    camera.GetFocalLengthAttr().Set(focal_length)


def _define_preview_material(
    stage,
    path: str,
    *,
    color: tuple[float, float, float],
    metallic: float,
    roughness: float,
):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_nocs_cloth(stage, asset_path: Path) -> None:
    """Render the garment's rest-space coordinates as a shaded NOCS texture."""

    from pxr import UsdGeom, UsdShade

    from clothdojo.sim.cloth import load_garment_mesh
    from clothdojo.teleop.isaac_stage import _nocs_material, _write_nocs_primvar

    cloth_prim = stage.GetPrimAtPath("/World/Cloth")
    if not cloth_prim:
        raise RuntimeError("paper NOCS render requires /World/Cloth")
    rest_vertices, _triangles, _panel_vertices, _panel_indices = load_garment_mesh(asset_path)
    cloth = UsdGeom.Mesh(cloth_prim)
    points = cloth.GetPointsAttr().Get()
    if points is None or len(points) != len(rest_vertices):
        raise RuntimeError("paper NOCS render requires one rest vertex per cloth particle")
    _write_nocs_primvar(cloth, rest_vertices)
    material = _nocs_material(stage, "/World/Looks/PaperClothNocs")
    UsdShade.MaterialBindingAPI.Apply(cloth_prim).Bind(material)
    print("paper cloth visual: shaded NOCS", flush=True)


def _bind_paper_solid_surfaces(stage, job: dict[str, Any]) -> dict[str, Any]:
    """Optionally replace busy table/ground textures with neutral paper surfaces."""

    from pxr import UsdShade

    result: dict[str, Any] = {}
    for field, prim_path, look_path, default_roughness in (
        ("paper_table_color", "/World/Table", "/World/Looks/PaperTable", 0.62),
        ("paper_ground_color", "/World/Ground", "/World/Looks/PaperGround", 0.82),
    ):
        value = job.get(field)
        if value is None:
            continue
        color = tuple(float(component) for component in value)
        if len(color) != 3 or any(not 0.0 <= component <= 1.0 for component in color):
            raise ValueError(f"{field} must contain three values in [0, 1]")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise RuntimeError(f"paper surface prim is unavailable: {prim_path}")
        roughness = float(job.get(f"{field}_roughness", default_roughness))
        if not 0.0 <= roughness <= 1.0:
            raise ValueError(f"{field}_roughness must be in [0, 1]")
        material = _define_preview_material(
            stage,
            look_path,
            color=color,
            metallic=0.0,
            roughness=roughness,
        )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        result[field] = {"color": list(color), "roughness": roughness}
    return result


def _stage_workcell(stage) -> None:
    """Complete the floating tabletop as a grounded industrial workstation."""

    from pxr import UsdGeom, UsdShade

    from clothdojo.teleop.isaac_stage import _define_textured_cube, _set_cube

    steel = _define_preview_material(
        stage,
        "/World/Looks/PaperWorkcellSteel",
        color=(0.055, 0.065, 0.075),
        metallic=0.78,
        roughness=0.28,
    )
    rubber = _define_preview_material(
        stage,
        "/World/Looks/PaperWorkcellRubber",
        color=(0.028, 0.032, 0.036),
        metallic=0.0,
        roughness=0.72,
    )

    def cube(
        name: str,
        position: tuple[float, float, float],
        half_size: tuple[float, float, float],
        material,
    ) -> None:
        geometry = _define_textured_cube(
            stage, f"/World/PaperWorkcell/{name}", half_size, physical_uvs=True
        )
        _set_cube(geometry, position, half_size)
        UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)

    for x in (0.13, 1.17):
        for y in (-0.60, 0.60):
            suffix = f"{'Front' if x > 0.7 else 'Rear'}{'Left' if y < 0 else 'Right'}"
            cube(f"Leg{suffix}", (x, y, 0.385), (0.035, 0.035, 0.385), steel)
            # The feet now contact the imported room floor directly.
            cube(f"Foot{suffix}", (x, y, 0.018), (0.075, 0.075, 0.018), rubber)
    cube("FrontApron", (1.17, 0.0, 0.725), (0.035, 0.60, 0.055), steel)
    cube("RearApron", (0.13, 0.0, 0.725), (0.035, 0.60, 0.055), steel)
    cube("LeftBrace", (0.65, -0.60, 0.43), (0.52, 0.025, 0.025), steel)
    cube("RightBrace", (0.65, 0.60, 0.43), (0.52, 0.025, 0.025), steel)
    cube("CableTray", (0.03, 0.0, 0.66), (0.055, 0.50, 0.055), steel)

    ground = stage.GetPrimAtPath("/World/Ground")
    if ground:
        UsdGeom.Imageable(ground).MakeInvisible()


def _bind_geniesim_table_material(
    stage,
    job: dict[str, Any],
    *,
    visual_assets_root: Path,
    seed: int,
) -> dict[str, str]:
    """Bind one deterministic material from a curated GenieSim PBR pool."""

    pool = job.get("geniesim_table_material_pool", list(_PAPER_TABLE_MATERIALS))
    if (
        not isinstance(pool, list)
        or not pool
        or any(not isinstance(value, str) or not value for value in pool)
        or len(pool) != len(set(pool))
    ):
        raise ValueError("geniesim_table_material_pool must contain unique material ids")
    unknown = set(pool).difference(_PAPER_TABLE_MATERIALS)
    if unknown:
        raise ValueError(f"unknown GenieSim table materials: {sorted(unknown)}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5441424C]))
    material_id = pool[int(rng.integers(len(pool)))]
    spec = _PAPER_TABLE_MATERIALS[material_id]
    source = (visual_assets_root / "geniesim-3.2" / spec.source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GenieSim table material source is missing: {source}")

    from pxr import Sdf, UsdShade

    destination = "/World/Looks/PaperGenieSimTable"
    existing = stage.GetPrimAtPath(destination)
    if existing:
        stage.RemovePrim(existing.GetPath())
    prim = stage.OverridePrim(destination)
    prim.GetReferences().AddReference(str(source), Sdf.Path(spec.material_prim))
    material = UsdShade.Material(prim)
    if not material:
        raise RuntimeError(f"referenced GenieSim material is invalid: {material_id}")
    table = stage.GetPrimAtPath("/World/Table")
    if not table:
        raise RuntimeError("paper table prim is missing")
    UsdShade.MaterialBindingAPI.Apply(table).Bind(material)
    return {
        "id": material_id,
        "source": str(source),
        "material_prim": spec.material_prim,
    }


def _validate_workspace_clearance(stage, root, minimum: float) -> float:
    """Reject room geometry that enters the paper workcell clearance volume."""

    from pxr import Usd, UsdGeom

    if minimum < 0.0:
        raise ValueError("paper environment clearance cannot be negative")
    # Includes the 2.10 x 1.76 m safety mat, robot kits, and a requested wall gap.
    bounds_min = np.asarray((-0.40 - minimum, -0.88 - minimum, 0.05))
    bounds_max = np.asarray((1.70 + minimum, 0.88 + minimum, 2.15))
    nearest = float("inf")
    collisions: list[str] = []
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        authored_points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if len(authored_points) == 0 or len(counts) == 0 or len(indices) == 0:
            continue
        points = np.asarray(authored_points, dtype=np.float64)
        transform = np.asarray(xforms.GetLocalToWorldTransform(prim), dtype=np.float64)
        homogeneous = np.concatenate(
            (points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
        )
        points = (homogeneous @ transform)[:, :3]
        offset = 0
        triangles: list[tuple[int, int, int]] = []
        for count in counts:
            face = indices[offset : offset + count]
            offset += count
            triangles.extend(
                (int(face[0]), int(face[index]), int(face[index + 1]))
                for index in range(1, int(count) - 1)
            )
        if not triangles:
            continue
        vertices = points[np.asarray(triangles, dtype=np.int64)]
        normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        vertical = (lengths > 1.0e-9) & (np.abs(normals[:, 2]) <= 0.85 * lengths)
        tri_min = vertices.min(axis=1)
        tri_max = vertices.max(axis=1)
        height_overlap = (tri_max[:, 2] >= bounds_min[2]) & (tri_min[:, 2] <= bounds_max[2])
        candidates = vertical & height_overlap
        if not np.any(candidates):
            continue
        tri_min = tri_min[candidates]
        tri_max = tri_max[candidates]
        dx = np.maximum.reduce(
            (
                bounds_min[0] - tri_max[:, 0],
                tri_min[:, 0] - bounds_max[0],
                np.zeros(tri_min.shape[0]),
            )
        )
        dy = np.maximum.reduce(
            (
                bounds_min[1] - tri_max[:, 1],
                tri_min[:, 1] - bounds_max[1],
                np.zeros(tri_min.shape[0]),
            )
        )
        nearest = min(nearest, float(np.min(np.hypot(dx, dy))))
        overlap = (dx == 0.0) & (dy == 0.0)
        if np.any(overlap):
            collisions.append(str(prim.GetPath()))
    if collisions:
        paths = ", ".join(sorted(set(collisions))[:4])
        raise ValueError(
            "paper environment enters the workcell clearance volume "
            f"({minimum:.3f} m margin): {paths}"
        )
    return nearest + minimum if np.isfinite(nearest) else float("inf")


def _set_paper_environment(
    stage,
    job: dict[str, Any],
    *,
    visual_assets_root: Path,
) -> dict[str, Any]:
    """Reference a distinct paper-only environment around the workstation."""

    environment = str(job.get("environment", "robodojo"))
    yaw = float(job.get("room_yaw", 0.0))
    room = stage.GetPrimAtPath("/World/RoboDojoRoom")
    light_scale = float(job.get("environment_light_scale", 1.0))
    light_temperature = float(job.get("light_temperature", 5100.0))

    def add_soft_lights(scale: float) -> None:
        from pxr import Gf, UsdGeom, UsdLux

        for name, position, radius, intensity, tint in (
            ("Key", (0.70, -1.10, 3.05), 0.95, 3400.0, light_temperature),
            (
                "Fill",
                (1.90, 1.65, 2.45),
                1.15,
                1450.0,
                light_temperature + 500.0,
            ),
        ):
            light = UsdLux.SphereLight.Define(stage, f"/World/Paper{name}Light")
            light.CreateRadiusAttr(radius)
            light.CreateIntensityAttr(intensity * scale)
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(tint)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*position))

    if environment != "robodojo":
        if room:
            stage.RemovePrim(room.GetPath())
        try:
            spec = _PAPER_ENVIRONMENTS[environment]
        except KeyError as exc:
            raise ValueError(f"unknown paper environment: {environment}") from exc
        source = (visual_assets_root / spec.asset_pack / spec.source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"paper environment is not cached locally: {source}")
        from pxr import Gf, Usd, UsdGeom

        # Keep the placement transform outside the referenced prim.  Several
        # official environments author their own root translate/rotate ops;
        # adding same-named ops directly to that prim is invalid USD.
        anchor = UsdGeom.Xform.Define(stage, "/World/PaperEnvironment")
        room = UsdGeom.Xform.Define(stage, "/World/PaperEnvironment/Asset")
        room.GetPrim().GetReferences().AddReference(str(source))
        for relative_path in job.get("hide_environment_prims", ()):
            parts = Path(str(relative_path)).parts
            if not parts or any(part in ("", ".", "..") for part in parts):
                raise ValueError(f"invalid paper environment prim path: {relative_path!r}")
            target = stage.GetPrimAtPath("/World/PaperEnvironment/Asset/" + "/".join(parts))
            if not target:
                raise ValueError(f"paper environment prim does not exist: {relative_path!r}")
            target.SetActive(False)
        xform = UsdGeom.Xformable(anchor)
        translation = tuple(
            float(value) for value in job.get("environment_translation", (0.0, 0.0, 0.0))
        )
        if len(translation) != 3:
            raise ValueError("paper environment translation must contain three values")
        scale = float(job.get("environment_scale", 1.0))
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddRotateZOp().Set(yaw)
        xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

        geometry: dict[str, Any] = {
            "environment": environment,
            "translation": list(translation),
            "scale": scale,
            "lighting": {
                "soft_light_scale": light_scale,
                "key_temperature": light_temperature,
                "fill_temperature": light_temperature + 500.0,
            },
        }
        if spec.floor_prim is not None:
            floor = stage.GetPrimAtPath(f"/World/PaperEnvironment/Asset/{spec.floor_prim}")
            if not floor:
                raise ValueError(f"paper environment floor prim does not exist: {spec.floor_prim}")
            bbox = (
                UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                )
                .ComputeWorldBound(floor)
                .ComputeAlignedBox()
            )
            floor_z = float(bbox.GetMax()[2])
            tolerance = float(job.get("environment_floor_tolerance", 0.002))
            if abs(floor_z) > tolerance:
                raise ValueError(
                    f"{environment} floor is at z={floor_z:.6f} m; expected z=0 "
                    f"within {tolerance:.6f} m"
                )
            geometry["floor_z"] = floor_z
            geometry["floor_prim"] = spec.floor_prim
        clearance = job.get("environment_min_workspace_clearance")
        if clearance is not None:
            clearance = float(clearance)
            measured = _validate_workspace_clearance(stage, room.GetPrim(), clearance)
            geometry["minimum_workspace_clearance"] = clearance
            geometry["measured_workspace_clearance"] = measured

        # The architectural assets are enclosed, so an exterior dome alone
        # leaves the manipulation table nearly black.  Add two large, soft
        # practical sources inside the room for a legible paper render while
        # retaining path-traced shadows and authored materials.
        add_soft_lights(light_scale)
        print(f"validated environment geometry: {geometry}", flush=True)
        return geometry
    if not room:
        return {"environment": environment}
    rotation = room.GetAttribute("xformOp:rotateZ")
    if not rotation:
        raise RuntimeError("paper room is missing its authored rotation")
    rotation.Set(-90.0 + float(yaw))
    add_soft_lights(1.2 * light_scale)
    return {"environment": environment}


def _apply_privileged_black_scene(stage, job: dict[str, Any]) -> dict[str, Any]:
    """Reduce a privileged NOCS pass to lit task geometry on a black field."""

    if not job.get("paper_privileged_black_scene", False):
        return {}
    if job.get("cloth_visual", "normal") != "nocs":
        raise ValueError("paper_privileged_black_scene requires cloth_visual=nocs")

    from pxr import Gf, UsdGeom, UsdLux

    for path in ("/World/RoboDojoRoom", "/World/PaperEnvironment"):
        prim = stage.GetPrimAtPath(path)
        if prim:
            stage.RemovePrim(prim.GetPath())
    dome = stage.GetPrimAtPath("/World/Lighting/Dome")
    if dome:
        UsdGeom.Imageable(dome).MakeInvisible()

    lights = (
        ("PrivilegedKey", (0.55, -0.85, 3.05), 1.10, 3200.0),
        ("PrivilegedFill", (1.60, 1.35, 2.50), 1.20, 1250.0),
        ("PrivilegedRim", (-0.45, 0.15, 2.10), 0.72, 1850.0),
    )
    for name, position, radius, intensity in lights:
        light = UsdLux.SphereLight.Define(stage, f"/World/{name}")
        light.CreateRadiusAttr(radius)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*position))
    return {"background": "black", "lighting": "three-point neutral"}


def _apply_privileged_white_scene(stage, job: dict[str, Any]) -> dict[str, Any]:
    """Place a privileged NOCS pass on a seamless white studio field."""

    if not job.get("paper_privileged_white_scene", False):
        return {}
    if job.get("cloth_visual", "normal") != "nocs":
        raise ValueError("paper_privileged_white_scene requires cloth_visual=nocs")

    from pxr import Gf, UsdGeom, UsdLux, UsdShade

    from clothdojo.teleop.isaac_stage import _define_textured_cube, _set_cube

    for path in ("/World/RoboDojoRoom", "/World/PaperEnvironment"):
        prim = stage.GetPrimAtPath(path)
        if prim:
            stage.RemovePrim(prim.GetPath())
    backdrop_material = _define_preview_material(
        stage,
        "/World/Looks/PrivilegedWhiteBackdrop",
        color=(0.985, 0.985, 0.99),
        metallic=0.0,
        roughness=0.96,
    )
    backdrop = _define_textured_cube(
        stage,
        "/World/PrivilegedWhiteBackdrop",
        (4.0, 4.0, 0.01),
        physical_uvs=True,
    )
    _set_cube(backdrop, (0.65, 0.0, -0.01), (4.0, 4.0, 0.01))
    UsdShade.MaterialBindingAPI.Apply(backdrop.GetPrim()).Bind(backdrop_material)

    for name, position, radius, intensity in (
        ("PrivilegedWhiteKey", (0.55, -0.85, 3.10), 1.15, 2300.0),
        ("PrivilegedWhiteFill", (1.55, 1.35, 2.55), 1.30, 900.0),
    ):
        light = UsdLux.SphereLight.Define(stage, f"/World/{name}")
        light.CreateRadiusAttr(radius)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*position))
    return {"background": "white", "lighting": "two-point neutral"}


def main() -> None:
    args = _arguments()
    plan_path = args.plan.expanduser().resolve()
    output = args.output.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    visual_assets_root = args.visual_assets_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    plan = _read_plan(plan_path)
    selected = set(args.job)
    frame_jobs = [job for job in plan["jobs"] if job.get("group") != "assets"]
    jobs = [job for job in frame_jobs if not selected or job["name"] in selected]
    missing = selected.difference(job["name"] for job in jobs)
    if missing:
        raise ValueError(f"unknown teaser render jobs: {sorted(missing)}")

    if args.variant_order is not None:
        if len(jobs) != 1:
            raise ValueError("--variant-order requires exactly one selected job")
        job = jobs[0]
        focal_lengths = job.get("camera_focal_lengths")
        if not isinstance(focal_lengths, list):
            raise ValueError("--variant-order requires camera_focal_lengths")
        if len(focal_lengths) != len(job["frames"]):
            raise ValueError("camera_focal_lengths must match frame selectors")
        if not 0 <= args.variant_order < len(focal_lengths):
            raise ValueError("variant order is outside camera_focal_lengths")
        job = {
            **job,
            "frames": [job["frames"][args.variant_order]],
            "camera_focal_length": float(focal_lengths[args.variant_order]),
            "_output_order": args.variant_order,
        }
        job.pop("camera_focal_lengths")
        jobs = [job]

    # Isaac Sim does not reliably tear down and rebuild these large stages in
    # one Kit process.  Keep the convenient multi-job CLI while isolating each
    # formal render and preserving the shared manifest between children.
    if args.variant_order is None and (len(jobs) > 1 or "camera_focal_lengths" in jobs[0]):
        for job in jobs:
            focal_lengths = job.get("camera_focal_lengths")
            orders = range(len(focal_lengths)) if focal_lengths is not None else (None,)
            for variant_order in orders:
                command = [
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
                ]
                if variant_order is not None:
                    command.extend(("--variant-order", str(variant_order)))
                if args.include_room:
                    command.append("--include-room")
                subprocess.run(command, check=True)
        return

    records: list[dict[str, Any]] = []
    existing_manifest = output / "manifest.json"
    if selected and existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        records = [
            record
            for record in existing.get("records", [])
            if not (
                record.get("job") in selected
                and (args.variant_order is None or record.get("output_order") == args.variant_order)
            )
        ]

    # Isaac Fabric addresses the selected physical device as logical cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

    from clothdojo.assets.materials import load_material_catalog
    from clothdojo.data.isaac_render import (
        VISUAL_COMPONENTS,
        IsaacRandomizedCamera,
        _preload_python_libstdcpp,
        _require_isaac5_runtime,
        _require_visual_assets,
        _start_isaac_app,
        overlay_kinematic_arm_reference,
        sample_visual_layout,
    )

    _require_isaac5_runtime()
    _require_visual_assets(
        visual_assets_root,
        load_material_catalog(),
        include_room=args.include_room,
    )
    _preload_python_libstdcpp()
    from clothdojo.teleop.isaac_compat import configure_replicator_warp

    configure_replicator_warp()
    portable = cache_root / "clothdojo" / "isaac5-paper" / f"{socket.gethostname()}-gpu-{args.gpu}"
    portable.mkdir(parents=True, exist_ok=True)
    app = _start_isaac_app(args.gpu, portable)
    try:
        from clothdojo.robots import robot_profile
        from clothdojo.trajectory import TrajectoryReader

        for episode_index, job in enumerate(jobs):
            print(f"starting {job['name']}", flush=True)
            trajectory = Path(job["trajectory"]).expanduser().resolve()
            reader = TrajectoryReader(trajectory)
            asset_id = reader.manifest.metadata.get("asset_id")
            if not isinstance(asset_id, str) or not asset_id:
                raise RuntimeError(f"trajectory has no asset_id: {trajectory}")
            profile = robot_profile(reader.manifest.scene.robot_id, assets_root=assets_root)
            selected_camera = _job_camera(profile, job)
            cameras = (selected_camera,)
            print(
                f"loaded {profile.robot_id}: states={reader.manifest.state_count}, "
                f"bodies={len(profile.body_names)}, asset={asset_id}",
                flush=True,
            )
            asset_path = assets_root / "garments" / "converted" / asset_id / f"{asset_id}.usdc"
            layouts: list[dict[str, Any]] = []
            kinematic_reference = None
            if job.get("use_kinematic_reference"):
                metrics_path = trajectory / "retarget_metrics.npz"
                with np.load(metrics_path) as metrics:
                    kinematic_reference = (
                        np.asarray(metrics["reference_joint_q"], dtype=np.float32),
                        np.asarray(metrics["reference_joint_qd"], dtype=np.float32),
                    )
            camera = IsaacRandomizedCamera(
                app,
                reader.state_at(0),
                asset_path=asset_path,
                assets_root=assets_root,
                visual_assets_root=visual_assets_root,
                device="cuda:0",
                base_seed=int(job.get("seed", plan.get("seed", 0))),
                include_room=args.include_room,
                randomize=VISUAL_COMPONENTS,
                layouts=layouts,
                robot_profile=profile,
                cameras=cameras,
                render_resolution=(args.width, args.height),
                pathtrace_spp=args.pathtrace_spp,
                room_light_saturation=0.12,
                room_light_intensity_scale=0.12,
            )
            _apply_native_camera_focal_override(
                camera._stage,
                camera._camera_paths[0],
                job,
            )
            for _ in range(2):
                app.update()
            print(f"camera ready for {job['name']}", flush=True)
            try:
                appearance_episode_index = int(job.get("appearance_episode_index", episode_index))
                layout = sample_visual_layout(
                    int(job.get("seed", plan.get("seed", 0))),
                    appearance_episode_index,
                    source_episode_id=str(job.get("appearance_id", job["name"])),
                    include_room=args.include_room,
                    randomize=VISUAL_COMPONENTS,
                )
                layout = _paper_layout(
                    layout,
                    job=job,
                    include_room=args.include_room,
                )
                camera.begin_episode(
                    appearance_episode_index,
                    source_episode_id=str(job.get("appearance_id", job["name"])),
                    layout=layout,
                )
                if job.get("cloth_visual", "normal") == "nocs":
                    _bind_nocs_cloth(camera._stage, asset_path)
                    for _ in range(4):
                        app.update()
                camera_optics = (
                    _apply_paper_camera_optics(camera._stage, job)
                    if selected_camera.name == "paper_view"
                    else {}
                )
                table_material = _bind_geniesim_table_material(
                    camera._stage,
                    job,
                    visual_assets_root=visual_assets_root,
                    seed=layout.seed,
                )
                paper_surfaces = _bind_paper_solid_surfaces(camera._stage, job)
                environment_geometry = _set_paper_environment(
                    camera._stage,
                    job,
                    visual_assets_root=visual_assets_root,
                )
                privileged_scene = _apply_privileged_black_scene(camera._stage, job)
                white_privileged_scene = _apply_privileged_white_scene(camera._stage, job)
                if privileged_scene and white_privileged_scene:
                    raise ValueError("privileged scene cannot be both black and white")
                privileged_scene = privileged_scene or white_privileged_scene
                if privileged_scene:
                    environment_geometry["privileged_scene"] = privileged_scene
                _stage_workcell(camera._stage)
                for _ in range(30 if job.get("environment") != "robodojo" else 3):
                    app.update()
                focal_lengths = job.get("camera_focal_lengths")
                if focal_lengths is not None:
                    if selected_camera.name != "paper_view":
                        raise ValueError("camera_focal_lengths requires paper_view")
                    if len(focal_lengths) != len(job["frames"]):
                        raise ValueError(
                            "camera_focal_lengths must match the number of frame selectors"
                        )
                for order, selector in enumerate(job["frames"]):
                    if focal_lengths is not None:
                        focal_job = {**job, "camera_focal_length": float(focal_lengths[order])}
                        camera_optics = _apply_paper_camera_optics(camera._stage, focal_job)
                        for _ in range(2):
                            app.update()
                    index = _frame_index(selector, reader.manifest.state_count)
                    state = reader.state_at(index)
                    if kinematic_reference is not None:
                        state = overlay_kinematic_arm_reference(
                            state,
                            reference_joint_q=kinematic_reference[0][index],
                            reference_joint_qd=kinematic_reference[1][index],
                            robot_profile=profile,
                        )
                    observation = camera(state)
                    camera_name = selected_camera.name
                    image_key = f"observation.images.{camera_name}"
                    if image_key not in observation:
                        raise ValueError(
                            f"camera {camera_name!r} is unavailable for {profile.robot_id}"
                        )
                    image = observation[image_key][0]
                    output_order = int(job.get("_output_order", order))
                    relative = Path(job["group"]) / f"{job['name']}_{output_order:02d}.png"
                    _save_rgb(output / relative, image)
                    records.append(
                        {
                            "group": job["group"],
                            "job": job["name"],
                            "trajectory": str(trajectory),
                            "task": reader.manifest.scene.task_name,
                            "robot": profile.robot_id,
                            "asset_id": asset_id,
                            "frame_selector": selector,
                            "frame_index": index,
                            "output_order": output_order,
                            "camera": camera_name,
                            "camera_spec": {
                                "name": selected_camera.name,
                                "parent_body": selected_camera.parent_body,
                                "parent_from_camera": selected_camera.parent_from_camera,
                                "focal_length": selected_camera.focal_length,
                                "horizontal_aperture": selected_camera.horizontal_aperture,
                                "vertical_aperture": selected_camera.vertical_aperture,
                                "near_clip": selected_camera.near_clip,
                                **camera_optics,
                            },
                            "image": str(relative),
                            "appearance": layouts[-1],
                            "paper_table_material": table_material,
                            "paper_solid_surfaces": paper_surfaces,
                            "environment_geometry": environment_geometry,
                        }
                    )
                    manifest = {
                        "schema": "clothdojo_paper_teaser_renders_v1",
                        "plan": str(plan_path),
                        "include_room": args.include_room,
                        "lighting_style": "paper_mild",
                        "room_light_saturation": 0.12,
                        "room_light_intensity_scale": 0.12,
                        "records": records,
                    }
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "manifest.json").write_text(
                        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                    )
                    print(f"rendered {relative} at state {index}", flush=True)
            except BaseException:
                # SimulationApp's fast shutdown can suppress the ordinary
                # interpreter traceback.  Emit it before tearing down the
                # camera so failed paper renders never look successful.
                import traceback

                traceback.print_exc()
                raise
            finally:
                camera.close()
    finally:
        if app.is_running():
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
