"""Renderer-independent dataset camera calibration."""

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


def policy_observation_settings() -> dict[str, object]:
    """Return the fixed visual contract used by policy rollout workers."""
    return {
        "renderer": "newton-sensor-tiled-camera",
        "camera_names": list(CAMERA_NAMES),
        "render_width_height": list(POLICY_RENDER_SIZE),
        "transport_width_height": list(POLICY_TRANSPORT_SIZE),
        "cloth_visual": "nocs",
    }
