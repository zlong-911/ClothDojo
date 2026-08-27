"""Three-view RGB observation provider for shirt-flattening rollout."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from clothdojo.rendering import CAMERA_MATRICES as _CAMERA_MATRICES
from clothdojo.rendering import CAMERA_NAMES, DATASET_CAMERA_NAMES, DATASET_CAMERAS


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
    world_offset: wp.vec3f,
    visual_points: wp.array(dtype=wp.vec3f),
    visual_normals: wp.array(dtype=wp.vec3f),
):
    vertex = wp.tid()
    source = source_indices[vertex]
    visual_points[vertex] = particle_points[source] - world_offset
    visual_normals[vertex] = particle_normals[source]


@wp.kernel
def _reset_mesh_bounds(bounds: wp.array2d(dtype=wp.float32)):
    axis = wp.tid()
    bounds[0, axis] = 1.0e30
    bounds[1, axis] = -1.0e30


@wp.kernel
def _accumulate_mesh_bounds(
    points: wp.array(dtype=wp.vec3f),
    world_offset: wp.vec3f,
    bounds: wp.array2d(dtype=wp.float32),
):
    point = points[wp.tid()] - world_offset
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
    bodies_per_world: int,
    transforms: wp.array2d(dtype=wp.transformf),
):
    wrist, world = wp.tid()
    transforms[wrist + 1, world] = wp.transform_multiply(
        body_q[world * bodies_per_world + body_indices[wrist]],
        local_transforms[wrist],
    )


class ShirtFlattenCameraObservation:
    """Render policy RGB from the model that also owns simulation state."""

    def __init__(
        self,
        scene,
        *,
        width: int = 640,
        height: int = 480,
        camera_names: tuple[str, ...] = CAMERA_NAMES,
    ) -> None:
        from newton.sensors import SensorTiledCamera

        self.world_count = int(scene.model.world_count)
        if self.world_count < 1:
            raise ValueError("camera scene must have a positive world count")
        if width < 1 or height < 1 or width * 3 != height * 4:
            raise ValueError("shirt-flattening cameras require a positive 4:3 resolution")
        self.model = scene.model
        self.width = int(width)
        self.height = int(height)
        self.camera_names = tuple(camera_names)
        if self.camera_names not in (
            CAMERA_NAMES,
            DATASET_CAMERA_NAMES,
            (CAMERA_NAMES[0],),
        ):
            raise ValueError(
                "camera_names must be policy cameras, dataset cameras or only cam_high"
            )
        self._sensor = SensorTiledCamera(model=self.model)
        self._sensor.default_render_config.enable_backface_culling = False
        self._sensor.default_render_config.enable_shadows = True
        self._sensor.default_render_config.enable_particles = False
        self._sensor.default_render_config.enable_textures = True
        self._sensor.utils.create_default_light(enable_shadows=True)
        self._smooth_cloth = _SmoothClothVisual(self._sensor, scene)

        all_fovs = tuple(
            _vertical_fov(camera.focal_length, camera.horizontal_aperture)
            for camera in DATASET_CAMERAS
        )
        camera_indices = tuple(DATASET_CAMERA_NAMES.index(name) for name in self.camera_names)
        fovs = tuple(all_fovs[index] for index in camera_indices)
        self._rays = self._sensor.utils.compute_camera_rays_pinhole(
            self.width,
            self.height,
            camera_fovs=fovs,
        )
        world_offsets = np.asarray(scene.world_offsets, dtype=np.float32)
        if world_offsets.shape != (self.world_count, 3):
            raise RuntimeError("camera world offsets do not match the Newton world count")
        local = tuple(_matrix_to_transform(matrix) for matrix in _CAMERA_MATRICES)
        high_transforms = tuple(
            _matrix_to_transform(_translated_matrix(_CAMERA_MATRICES[0], offset))
            for offset in world_offsets
        )
        if self.camera_names in (CAMERA_NAMES, DATASET_CAMERA_NAMES):
            camera_transforms = [
                high_transforms,
                [local[1]] * self.world_count,
                [local[2]] * self.world_count,
            ]
            if self.camera_names == DATASET_CAMERA_NAMES:
                camera_transforms.append(
                    tuple(
                        _matrix_to_transform(
                            _translated_matrix(_CAMERA_MATRICES[3], offset)
                        )
                        for offset in world_offsets
                    )
                )
            self._transforms = wp.array(
                camera_transforms,
                dtype=wp.transformf,
                device=self.model.device,
            )
            self._wrist_local = wp.array(
                local[1:3], dtype=wp.transformf, device=self.model.device
            )
            if int(self.model.body_count) % self.world_count:
                raise RuntimeError("render body count is not homogeneous across worlds")
            self._bodies_per_world = int(self.model.body_count) // self.world_count
            self._wrist_bodies = wp.array(
                [
                    _body_index(self.model, "left", "camera", self._bodies_per_world),
                    _body_index(self.model, "right", "camera", self._bodies_per_world),
                ],
                dtype=int,
                device=self.model.device,
            )
        else:
            self._transforms = wp.array(
                [high_transforms], dtype=wp.transformf, device=self.model.device
            )
            self._wrist_local = None
            self._wrist_bodies = None
            self._bodies_per_world = 0
        self._color = self._sensor.utils.create_color_image_output(
            self.width, self.height, len(self.camera_names)
        )
        self._rgba = self._sensor.utils.to_rgba_from_color(self._color)

    def __call__(self, runtime) -> dict[str, np.ndarray]:
        if runtime.scene.model is not self.model:
            raise ValueError("camera and runtime must use the same Newton model")
        self._smooth_cloth.sync(runtime)
        if self._wrist_bodies is not None:
            wp.launch(
                _update_wrist_cameras,
                dim=(2, self.world_count),
                inputs=[
                    runtime.state_0.body_q,
                    self._wrist_bodies,
                    self._wrist_local,
                    self._bodies_per_world,
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
            .reshape(self.world_count, len(self.camera_names), self.height, self.width, 3)
        )
        return {
            f"observation.images.{name}": rgb[:, index].copy()
            for index, name in enumerate(self.camera_names)
        }


class _SmoothClothVisual:
    """Keep one independent render-only NOCS mesh per Newton world."""

    def __init__(self, sensor, scene) -> None:
        matches = [
            index
            for index, label in enumerate(scene.model.shape_label)
            if str(label).endswith("garment_visual")
        ]
        world_count = int(scene.model.world_count)
        if len(matches) != world_count:
            raise RuntimeError(
                f"expected one render-only garment mesh per world, found {len(matches)}"
            )

        # Newton 1.5 copies authored Mesh normals into its render context but
        # does not expose a public update hook. Keep this one version-pinned
        # adapter here instead of copying or patching Newton's renderer.
        context = getattr(sensor, "_SensorTiledCamera__render_context", None)
        mesh_data_store = getattr(context, "_RenderContext__mesh_data", None)
        if context is None or mesh_data_store is None:
            raise RuntimeError("Newton 1.5 tiled-camera internals changed")
        if not context.has_triangle_mesh:
            raise RuntimeError("Newton render model has no deformable triangle mesh to replace")

        mesh_data_ids = context.shape_mesh_data_ids.numpy()
        visual_mesh_ids = [int(mesh_data_ids[shape]) for shape in matches]
        if any(mesh_id < 0 for mesh_id in visual_mesh_ids):
            raise RuntimeError("render-only garment mesh has no vertex-normal buffer")
        if len(set(visual_mesh_ids)) != world_count:
            raise RuntimeError("render-only garment worlds unexpectedly share one Mesh buffer")

        # The physics triangles and this visual mesh are coincident. Disable
        # only the flat-shaded render copy; physics state and topology remain
        # untouched and the static Mesh path supplies the rendered garment.
        context._RenderContext__triangle_points = None
        context._RenderContext__triangle_indices = None
        context.triangle_mesh = None
        self._model = scene.model
        triangles = np.asarray(scene.model.tri_indices.numpy(), dtype=np.int32).reshape(-1, 3)
        particle_world = np.asarray(scene.model.particle_world.numpy(), dtype=np.int32)
        triangle_worlds = particle_world[triangles]
        if np.any(triangle_worlds != triangle_worlds[:, :1]):
            raise RuntimeError("render garment triangles cross Newton world boundaries")
        self._all_source_indices = wp.array(
            np.ascontiguousarray(triangles.reshape(-1)),
            dtype=int,
            device=scene.model.device,
        )
        self._particle_normals = wp.zeros(
            int(scene.model.particle_count),
            dtype=wp.vec3f,
            device=scene.model.device,
        )
        self._particle_ranges = []
        if scene.model.bvh_shape_bounds is None:
            raise RuntimeError("Newton render model has no shape BVH bounds")
        self._visuals = []
        for world, (shape_index, mesh_data_id) in enumerate(
            zip(matches, visual_mesh_ids, strict=True)
        ):
            source = scene.model.shape_source[shape_index]
            mesh = getattr(source, "mesh", None)
            if mesh is None:
                raise RuntimeError("render-only garment mesh was not finalized")
            normals = mesh_data_store[mesh_data_id].normals
            source_indices_host = np.ascontiguousarray(
                triangles[triangle_worlds[:, 0] == world].reshape(-1)
            )
            if mesh.points.shape[0] != normals.shape[0]:
                raise RuntimeError("render-only garment points and normals differ")
            if len(source_indices_host) != mesh.points.shape[0]:
                raise RuntimeError("NOCS visual vertices do not match physics triangle corners")
            particle_indices = np.flatnonzero(particle_world == world)
            if not len(particle_indices) or np.any(np.diff(particle_indices) != 1):
                raise RuntimeError("garment particles must be contiguous within each world")
            self._visuals.append(
                (
                    shape_index,
                    mesh,
                    mesh.points,
                    normals,
                    wp.array(
                        source_indices_host,
                        dtype=int,
                        device=scene.model.device,
                    ),
                    int(particle_indices[0]),
                    int(particle_indices[-1]) + 1,
                    wp.empty(
                        (2, 3),
                        dtype=wp.float32,
                        device=scene.model.device,
                    ),
                    wp.vec3f(*map(float, scene.world_offsets[world])),
                )
            )
            self._particle_ranges.append(
                (int(particle_indices[0]), int(particle_indices[-1]) + 1)
            )

    def sync(self, runtime) -> None:
        particle_q = runtime.state_0.particle_q
        if self._particle_normals.shape[0] != particle_q.shape[0]:
            raise RuntimeError("physics and NOCS garment vertex counts differ")
        if any(end > particle_q.shape[0] for _, end in self._particle_ranges):
            raise RuntimeError("render state and NOCS garment particle ranges differ")
        self._particle_normals.zero_()
        wp.launch(
            _accumulate_vertex_normals,
            dim=self._all_source_indices.shape[0] // 3,
            inputs=[particle_q, self._all_source_indices, self._particle_normals],
            device=self._model.device,
        )
        wp.launch(
            _normalize_vertex_normals,
            dim=particle_q.shape[0],
            inputs=[self._particle_normals],
            device=self._model.device,
        )
        for (
            shape_index,
            mesh,
            points,
            normals,
            source_indices,
            start,
            end,
            bounds,
            world_offset,
        ) in self._visuals:
            wp.launch(
                _sync_visual_vertices,
                dim=points.shape[0],
                inputs=[
                    particle_q,
                    self._particle_normals,
                    source_indices,
                    world_offset,
                    points,
                    normals,
                ],
                device=self._model.device,
            )
            mesh.refit()
            _refit_deforming_mesh_bounds(
                particle_q[start:end],
                bounds,
                self._model.bvh_shape_bounds,
                shape_index,
                world_offset=world_offset,
                device=self._model.device,
            )


def _refit_deforming_mesh_bounds(
    points,
    bounds,
    shape_bounds,
    shape_index: int,
    *,
    world_offset,
    device,
) -> None:
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
        inputs=[points, world_offset, bounds],
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


def _translated_matrix(matrix, offset):
    values = np.asarray(matrix, dtype=np.float32).reshape(4, 4).copy()
    values[:3, 3] += np.asarray(offset, dtype=np.float32)
    return values


def _body_index(model, side: str, link: str, bodies_per_world: int) -> int:
    matches = [
        index
        for index, label in enumerate(model.body_label[:bodies_per_world])
        if f"/{side}/" in str(label) and str(label).endswith(f"/{link}")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one render body named {side}/{link}, found {len(matches)}")
    return matches[0]
