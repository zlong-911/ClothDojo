"""Renderer-independent dataset camera calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DATASET_CAMERA_NAMES = (*CAMERA_NAMES, "cam_third_person")
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


CAMERA_MATRICES = (
    (
        (0.0, 0.8660254037844386, -0.5, 0.253845),
        (-1.0, 0.0, 0.0, 0.001071),
        (0.0, 0.5, 0.8660254037844386, 1.308),
        (0.0, 0.0, 0.0, 1.0),
    ),
    (
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0109),
        (0.0, 0.0, 0.0, 1.0),
    ),
    (
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0109),
        (0.0, 0.0, 0.0, 1.0),
    ),
    (
        (0.0, -0.374606593415912, 0.9271838545667874, 1.25),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.9271838545667874, 0.374606593415912, 1.35),
        (0.0, 0.0, 0.0, 1.0),
    ),
)

DATASET_CAMERAS = tuple(
    CameraSpec(name, parent, matrix, focal_length, horizontal_aperture)
    for name, parent, matrix, focal_length, horizontal_aperture in zip(
        DATASET_CAMERA_NAMES,
        (None, "left/camera", "right/camera", None),
        CAMERA_MATRICES,
        (10.0, 13.0, 13.0, 10.0),
        (14.266, 15.71625, 15.71625, 14.266),
        strict=True,
    )
)


def policy_observation_settings() -> dict[str, object]:
    """Return the fixed visual contract used by policy rollout workers."""

    cameras = [
        {
            "name": camera.name,
            "parent_body": camera.parent_body,
            "parent_from_camera": camera.parent_from_camera,
            "focal_length": camera.focal_length,
            "horizontal_aperture": camera.horizontal_aperture,
        }
        for camera in DATASET_CAMERAS
        if camera.name in CAMERA_NAMES
    ]
    calibration = hashlib.sha256(
        json.dumps(cameras, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "renderer": "newton-sensor-tiled-camera",
        "camera_names": list(CAMERA_NAMES),
        "render_width_height": list(POLICY_RENDER_SIZE),
        "transport_width_height": list(POLICY_TRANSPORT_SIZE),
        "calibration_sha256": calibration,
        "cloth_visual": "nocs",
    }
