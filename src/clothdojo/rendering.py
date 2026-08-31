"""Shared camera calibration and authored robot appearance."""

from __future__ import annotations

from dataclasses import dataclass

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DATASET_CAMERA_NAMES = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
# Keep rollout rasterization aligned with the existing paired benchmark. OpenPI
# downsamples these views to the square transport size below; a lower native
# render size is faster but is not pixel-equivalent and therefore needs its own
# paired rollout validation before it can replace this contract.
POLICY_RENDER_SIZE = (640, 480)
POLICY_TRANSPORT_SIZE = (224, 224)


@dataclass(frozen=True, slots=True)
class CameraSpec:
    name: str
    parent_body: str | None
    parent_from_camera: tuple[tuple[float, float, float, float], ...]
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    near_clip: float = 0.0


def _apply_authored_appearance(builder, stage) -> dict:
    """Bridge Newton USD appearance omissions for both review and policy RGB."""
    from newton import ShapeFlags
    from newton._src.usd.utils import (
        _color_to_display_space,
        resolve_material_properties_for_prim,
    )
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
        properties = resolve_material_properties_for_prim(prim)
        if properties.get("color") is not None or properties.get("texture") is not None:
            continue
        attribute = prim.GetAttribute("primvars:displayColor")
        values = attribute.Get() if attribute and attribute.HasAuthoredValue() else None
        if values is not None and len(values) == 1:
            builder.shape_color[index] = _color_to_display_space(
                tuple(map(float, values[0])), attribute
            )
            display_colors.append(path)
    return {
        "authored_display_color_fallbacks": display_colors,
        "hidden_authored_proxy_shapes": hidden_proxies,
    }


def policy_observation_settings() -> dict[str, object]:
    """Return the fixed visual contract used by policy rollout workers."""
    return {
        "renderer": "newton-sensor-tiled-camera",
        "camera_names": list(CAMERA_NAMES),
        "render_width_height": list(POLICY_RENDER_SIZE),
        "transport_width_height": list(POLICY_TRANSPORT_SIZE),
        "cloth_visual": "nocs",
    }
