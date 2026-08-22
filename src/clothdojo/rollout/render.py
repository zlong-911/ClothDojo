"""Three-view RGB observation provider for shirt-flattening rollout."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")

_CAMERA_MATRICES = (
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
)


@wp.kernel
def _update_wrist_cameras(
    body_q: wp.array(dtype=wp.transformf),
    body_indices: wp.array(dtype=int),
    local_transforms: wp.array(dtype=wp.transformf),
    transforms: wp.array2d(dtype=wp.transformf),
):
    wrist = wp.tid()
    transforms[wrist + 1, 0] = wp.transform_multiply(
        body_q[body_indices[wrist]], local_transforms[wrist]
    )


class ShirtFlattenCameraObservation:
    """Render policy RGB from a visual twin of the physics model."""

    def __init__(
        self,
        physics_scene,
        render_scene,
        *,
        width: int = 640,
        height: int = 480,
    ) -> None:
        from newton.sensors import SensorTiledCamera

        if physics_scene.model.world_count != 1 or render_scene.model.world_count != 1:
            raise ValueError("the first camera rollout slice requires one world")
        for field in ("body_count", "particle_count", "tri_count"):
            if int(getattr(physics_scene.model, field)) != int(getattr(render_scene.model, field)):
                raise ValueError(f"physics/render topology differs: {field}")
        if width < 1 or height < 1 or width * 3 != height * 4:
            raise ValueError("shirt-flattening cameras require a positive 4:3 resolution")
        self.model = render_scene.model
        self.width = int(width)
        self.height = int(height)
        self._sensor = SensorTiledCamera(model=self.model)
        self._sensor.default_render_config.enable_backface_culling = False
        self._sensor.default_render_config.enable_shadows = True
        self._sensor.default_render_config.enable_particles = False
        self._sensor.utils.create_default_light(enable_shadows=True)

        fovs = (
            _vertical_fov(10.0, 14.266),
            _vertical_fov(13.0, 15.71625),
            _vertical_fov(13.0, 15.71625),
        )
        self._rays = self._sensor.utils.compute_camera_rays_pinhole(
            self.width,
            self.height,
            camera_fovs=fovs,
        )
        local = tuple(_matrix_to_transform(matrix) for matrix in _CAMERA_MATRICES)
        self._transforms = wp.array(
            [[local[0]], [local[1]], [local[2]]],
            dtype=wp.transformf,
            device=self.model.device,
        )
        self._wrist_local = wp.array(local[1:], dtype=wp.transformf, device=self.model.device)
        self._wrist_bodies = wp.array(
            [_body_index(self.model, "left", "camera"), _body_index(self.model, "right", "camera")],
            dtype=int,
            device=self.model.device,
        )
        self._color = self._sensor.utils.create_color_image_output(
            self.width, self.height, len(CAMERA_NAMES)
        )
        self._rgba = self._sensor.utils.to_rgba_from_color(self._color)

    def __call__(self, runtime) -> dict[str, np.ndarray]:
        wp.launch(
            _update_wrist_cameras,
            dim=2,
            inputs=[
                runtime.state_0.body_q,
                self._wrist_bodies,
                self._wrist_local,
                self._transforms,
            ],
            device=self.model.device,
        )
        self.model.bvh_refit_shapes(runtime.state_0)
        self._sensor.update(
            runtime.state_0,
            self._transforms,
            self._rays,
            color_image=self._color,
        )
        rgb = (
            self._rgba[:, :, :, :3]
            .numpy()
            .reshape(1, len(CAMERA_NAMES), self.height, self.width, 3)
        )
        return {
            f"observation.images.{name}": rgb[:, index].copy()
            for index, name in enumerate(CAMERA_NAMES)
        }


def _vertical_fov(focal_length: float, vertical_aperture: float) -> float:
    return 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))


def _matrix_to_transform(matrix):
    values = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    return wp.transform_from_matrix(wp.mat44f(*map(float, values.reshape(-1))))


def _body_index(model, side: str, link: str) -> int:
    matches = [
        index
        for index, label in enumerate(model.body_label)
        if f"/{side}/" in str(label) and str(label).endswith(f"/{link}")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one render body named {side}/{link}, found {len(matches)}")
    return matches[0]
