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
def _accumulate_vertex_normals(
    points: wp.array(dtype=wp.vec3f),
    indices: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3f),
):
    triangle = wp.tid()
    i0 = indices[triangle * 3]
    i1 = indices[triangle * 3 + 1]
    i2 = indices[triangle * 3 + 2]
    face_normal = wp.cross(points[i1] - points[i0], points[i2] - points[i0])
    wp.atomic_add(normals, i0, face_normal)
    wp.atomic_add(normals, i1, face_normal)
    wp.atomic_add(normals, i2, face_normal)


@wp.kernel
def _normalize_vertex_normals(normals: wp.array(dtype=wp.vec3f)):
    vertex = wp.tid()
    length = wp.length(normals[vertex])
    if length > 1.0e-12:
        normals[vertex] = normals[vertex] / length


@wp.kernel
def _sync_visual_vertices(
    particle_points: wp.array(dtype=wp.vec3f),
    particle_normals: wp.array(dtype=wp.vec3f),
    source_indices: wp.array(dtype=int),
    visual_points: wp.array(dtype=wp.vec3f),
    visual_normals: wp.array(dtype=wp.vec3f),
):
    vertex = wp.tid()
    source = source_indices[vertex]
    visual_points[vertex] = particle_points[source]
    visual_normals[vertex] = particle_normals[source]


@wp.kernel
def _reset_mesh_bounds(bounds: wp.array2d(dtype=wp.float32)):
    axis = wp.tid()
    bounds[0, axis] = 1.0e30
    bounds[1, axis] = -1.0e30


@wp.kernel
def _accumulate_mesh_bounds(
    points: wp.array(dtype=wp.vec3f),
    bounds: wp.array2d(dtype=wp.float32),
):
    point = points[wp.tid()]
    wp.atomic_min(bounds, 0, 0, point[0])
    wp.atomic_min(bounds, 0, 1, point[1])
    wp.atomic_min(bounds, 0, 2, point[2])
    wp.atomic_max(bounds, 1, 0, point[0])
    wp.atomic_max(bounds, 1, 1, point[1])
    wp.atomic_max(bounds, 1, 2, point[2])


@wp.kernel
def _write_mesh_shape_bounds(
    bounds: wp.array2d(dtype=wp.float32),
    shape_index: int,
    shape_bounds: wp.array2d(dtype=wp.vec3f),
):
    shape_bounds[shape_index, 0] = wp.vec3f(bounds[0, 0], bounds[0, 1], bounds[0, 2])
    shape_bounds[shape_index, 1] = wp.vec3f(bounds[1, 0], bounds[1, 1], bounds[1, 2])


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
        self._sensor.default_render_config.enable_textures = True
        self._sensor.utils.create_default_light(enable_shadows=True)
        self._smooth_cloth = _SmoothClothVisual(self._sensor, render_scene)

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
        self._smooth_cloth.sync(runtime)
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


class _SmoothClothVisual:
    """Keep one render-only mesh on Newton's smooth static-mesh path."""

    def __init__(self, sensor, render_scene) -> None:
        matches = [
            index
            for index, label in enumerate(render_scene.model.shape_label)
            if str(label).endswith("garment_visual")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one render-only garment mesh, found {len(matches)}")
        shape_index = matches[0]
        source = render_scene.model.shape_source[shape_index]
        mesh = getattr(source, "mesh", None)
        if mesh is None:
            raise RuntimeError("render-only garment mesh was not finalized")

        # Newton 1.5 copies authored Mesh normals into its render context but
        # does not expose a public update hook. Keep this one version-pinned
        # adapter here instead of copying or patching Newton's renderer.
        context = getattr(sensor, "_SensorTiledCamera__render_context", None)
        mesh_data_store = getattr(context, "_RenderContext__mesh_data", None)
        if context is None or mesh_data_store is None:
            raise RuntimeError("Newton 1.5 tiled-camera internals changed")
        mesh_data_id = int(context.shape_mesh_data_ids.numpy()[shape_index])
        if mesh_data_id < 0:
            raise RuntimeError("render-only garment mesh has no vertex-normal buffer")
        normals = mesh_data_store[mesh_data_id].normals
        if mesh.points.shape[0] != normals.shape[0]:
            raise RuntimeError("render-only garment points and normals differ")
        if not context.has_triangle_mesh:
            raise RuntimeError("Newton render model has no deformable triangle mesh to replace")

        # The physics triangles and this visual mesh are coincident. Disable
        # only the flat-shaded render copy; physics state and topology remain
        # untouched and the static Mesh path supplies the rendered garment.
        context._RenderContext__triangle_points = None
        context._RenderContext__triangle_indices = None
        context.triangle_mesh = None
        self._model = render_scene.model
        self._shape_index = shape_index
        self._mesh = mesh
        self._points = mesh.points
        self._normals = normals
        self._source_indices = wp.array(
            np.ascontiguousarray(render_scene.model.tri_indices.numpy().reshape(-1)),
            dtype=int,
            device=render_scene.model.device,
        )
        if self._source_indices.shape[0] != self._points.shape[0]:
            raise RuntimeError("NOCS visual vertices do not match physics triangle corners")
        self._particle_normals = wp.zeros(
            int(render_scene.model.particle_count),
            dtype=wp.vec3f,
            device=render_scene.model.device,
        )
        if render_scene.model.bvh_shape_bounds is None:
            raise RuntimeError("Newton render model has no shape BVH bounds")
        self._bounds = wp.empty(
            (2, 3),
            dtype=wp.float32,
            device=render_scene.model.device,
        )

    def sync(self, runtime) -> None:
        ranges = runtime.grasp.particle_ranges()
        if len(ranges) != 1:
            raise RuntimeError("smooth garment rendering requires one physics world")
        start, end = ranges[0]
        particle_q = runtime.state_0.particle_q
        if start != 0 or end != particle_q.shape[0]:
            raise RuntimeError("garment particles are not the complete particle state")
        if self._particle_normals.shape[0] != particle_q.shape[0]:
            raise RuntimeError("physics and NOCS garment vertex counts differ")
        self._particle_normals.zero_()
        wp.launch(
            _accumulate_vertex_normals,
            dim=self._source_indices.shape[0] // 3,
            inputs=[particle_q, self._source_indices, self._particle_normals],
            device=self._model.device,
        )
        wp.launch(
            _normalize_vertex_normals,
            dim=particle_q.shape[0],
            inputs=[self._particle_normals],
            device=self._model.device,
        )
        wp.launch(
            _sync_visual_vertices,
            dim=self._points.shape[0],
            inputs=[
                particle_q,
                self._particle_normals,
                self._source_indices,
                self._points,
                self._normals,
            ],
            device=self._model.device,
        )
        self._mesh.refit()
        _refit_deforming_mesh_bounds(
            particle_q,
            self._bounds,
            self._model.bvh_shape_bounds,
            self._shape_index,
            device=self._model.device,
        )


def _refit_deforming_mesh_bounds(points, bounds, shape_bounds, shape_index: int, *, device) -> None:
    """Refresh one deforming static Mesh's local bounds without rebuilding the scene BVH."""

    wp.launch(
        _reset_mesh_bounds,
        dim=3,
        inputs=[bounds],
        device=device,
    )
    wp.launch(
        _accumulate_mesh_bounds,
        dim=points.shape[0],
        inputs=[points, bounds],
        device=device,
    )
    wp.launch(
        _write_mesh_shape_bounds,
        dim=1,
        inputs=[bounds, shape_index, shape_bounds],
        device=device,
    )


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
