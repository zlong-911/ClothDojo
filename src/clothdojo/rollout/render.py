"""Three-view RGB observation provider for shirt-flattening rollout."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from clothdojo.rendering import CAMERA_NAMES, DATASET_CAMERA_NAMES


@wp.kernel
def _accumulate_vertex_normals(
    points: wp.array(dtype=wp.vec3f),
    indices: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3f),
    winding_sign: float,
):
    triangle = wp.tid()
    i0 = indices[triangle * 3]
    i1 = indices[triangle * 3 + 1]
    i2 = indices[triangle * 3 + 2]
    face_normal = winding_sign * wp.cross(points[i1] - points[i0], points[i2] - points[i0])
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
    camera_indices: wp.array(dtype=int),
    body_indices: wp.array(dtype=int),
    local_transforms: wp.array(dtype=wp.transformf),
    bodies_per_world: int,
    transforms: wp.array2d(dtype=wp.transformf),
):
    wrist, world = wp.tid()
    transforms[camera_indices[wrist], world] = wp.transform_multiply(
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
        self.robot_profile = scene.robot_profile
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
        cameras_by_name = {camera.name: camera for camera in self.robot_profile.cameras}
        missing = tuple(name for name in self.camera_names if name not in cameras_by_name)
        if missing:
            raise ValueError(
                f"robot {self.robot_profile.robot_id} does not define cameras: "
                f"{', '.join(missing)}"
            )
        cameras = tuple(cameras_by_name[name] for name in self.camera_names)
        self._sensor = SensorTiledCamera(model=self.model)
        self._sensor.default_render_config.enable_backface_culling = False
        self._sensor.default_render_config.enable_shadows = True
        self._sensor.default_render_config.enable_particles = False
        self._sensor.default_render_config.enable_textures = True
        self._sensor.utils.create_default_light(enable_shadows=True)
        self._smooth_cloth = _SmoothClothVisual(self._sensor, scene)

        self._rays = _camera_rays(self._sensor.utils, cameras, self.width, self.height)
        _clip_camera_rays(self._rays, cameras)
        world_offsets = np.asarray(scene.world_offsets, dtype=np.float32)
        if world_offsets.shape != (self.world_count, 3):
            raise RuntimeError("camera world offsets do not match the Newton world count")
        local = tuple(
            _matrix_to_transform(camera.parent_from_camera) for camera in cameras
        )
        # This includes FR3's body-mounted head camera, even for cam_high-only
        # rendering; body_q already contains each world's translation.
        wrist_camera_indices = tuple(
            index for index, camera in enumerate(cameras) if camera.parent_body
        )
        camera_transforms = [
            (
                [local[index]] * self.world_count
                if camera.parent_body
                else tuple(
                    _matrix_to_transform(
                        _translated_matrix(camera.parent_from_camera, offset)
                    )
                    for offset in world_offsets
                )
            )
            for index, camera in enumerate(cameras)
        ]
        self._transforms = wp.array(
            camera_transforms,
            dtype=wp.transformf,
            device=self.model.device,
        )
        if wrist_camera_indices:
            self._wrist_local = wp.array(
                [local[index] for index in wrist_camera_indices],
                dtype=wp.transformf,
                device=self.model.device,
            )
            self._wrist_camera_indices = wp.array(
                wrist_camera_indices, dtype=int, device=self.model.device
            )
            if int(self.model.body_count) % self.world_count:
                raise RuntimeError("render body count is not homogeneous across worlds")
            self._bodies_per_world = int(self.model.body_count) // self.world_count
            self._wrist_bodies = wp.array(
                [
                    _body_index(
                        self.model, cameras[index].parent_body, self._bodies_per_world
                    )
                    for index in wrist_camera_indices
                ],
                dtype=int,
                device=self.model.device,
            )
        else:
            self._wrist_local = None
            self._wrist_camera_indices = None
            self._wrist_bodies = None
            self._bodies_per_world = 0
        self._color = self._sensor.utils.create_color_image_output(
            self.width, self.height, len(self.camera_names)
        )
        self._rgba = self._sensor.utils.to_rgba_from_color(self._color)

    def set_cloth_surface_winding_reversed(self, reversed_: bool) -> None:
        """Select the logical front face for the next recorded state."""

        self._smooth_cloth.set_surface_winding_reversed(reversed_)

    def __call__(self, runtime) -> dict[str, np.ndarray]:
        if runtime.scene.model is not self.model:
            raise ValueError("camera and runtime must use the same Newton model")
        self._smooth_cloth.sync(runtime)
        if self._wrist_bodies is not None:
            wp.launch(
                _update_wrist_cameras,
                dim=(len(self._wrist_bodies), self.world_count),
                inputs=[
                    runtime.state_0.body_q,
                    self._wrist_camera_indices,
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
        self._winding_sign = 1.0
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

    def set_surface_winding_reversed(self, reversed_: bool) -> None:
        if not isinstance(reversed_, bool):
            raise TypeError("surface winding flag must be bool")
        self._winding_sign = -1.0 if reversed_ else 1.0

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
            inputs=[
                particle_q,
                self._all_source_indices,
                self._particle_normals,
                self._winding_sign,
            ],
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


@wp.kernel
def _move_ray_origins_to_near_plane(
    rays: wp.array4d(dtype=wp.vec3f),
    near: wp.array(dtype=wp.float32),
):
    camera, row, column = wp.tid()
    direction = rays[camera, row, column, 1]
    # USD cameras look along -Z. Clipping is an optical-axis plane, not
    # a sphere of constant Euclidean ray distance.
    rays[camera, row, column, 0] = direction * (near[camera] / -direction[2])


def _clip_camera_rays(rays, cameras) -> None:
    """Apply USD near clipping without moving the camera or changing its FOV.

    SensorTiledCamera accepts explicit per-pixel ray origins but has no near
    plane argument. This RGB provider traces from the near plane; raw sensor
    depth, if requested separately, is consequently relative to that plane.
    """
    near = np.asarray([camera.near_clip for camera in cameras], dtype=np.float32)
    if not np.isfinite(near).all() or np.any(near < 0):
        raise ValueError("camera near clipping must be finite and nonnegative")
    if np.any(near > 0):
        wp.launch(
            _move_ray_origins_to_near_plane,
            dim=rays.shape[:3],
            inputs=[rays, wp.array(near, dtype=wp.float32, device=rays.device)],
            device=rays.device,
        )


def _camera_rays(utils, cameras, width: int, height: int):
    if all(
        math.isclose(
            camera.vertical_aperture,
            camera.horizontal_aperture * height / width,
            rel_tol=1.0e-6,
        )
        for camera in cameras
    ):
        # Preserve the existing benchmark pixels for the legacy 4:3 assets.
        # That path historically treats horizontal FOV as vertical FOV; it is
        # NOT equivalent to the USD intrinsics. Correcting it needs a separate
        # paired benchmark, not an incidental workstation migration change.
        return utils.compute_camera_rays_pinhole(
            width,
            height,
            camera_fovs=tuple(
                _vertical_fov(camera.focal_length, camera.horizontal_aperture)
                for camera in cameras
            ),
        )
    # Native non-4:3 cameras (FR3 Duo) retain both authored axes rather than
    # deriving an aperture from the fixed policy raster's aspect ratio.
    return utils.compute_camera_rays_pinhole(
        width,
        height,
        focal_length=tuple(camera.focal_length for camera in cameras),
        horizontal_aperture=tuple(camera.horizontal_aperture for camera in cameras),
        vertical_aperture=tuple(camera.vertical_aperture for camera in cameras),
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


def _body_index(model, semantic_name: str | None, bodies_per_world: int) -> int:
    if semantic_name is None:
        raise ValueError("wrist camera must define a parent body")
    matches = [
        index
        for index, label in enumerate(model.body_label[:bodies_per_world])
        if _matches_semantic_body(str(label), semantic_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one render body named {semantic_name}, found {len(matches)}"
        )
    return matches[0]


def _matches_semantic_body(label: str, semantic_name: str) -> bool:
    label_parts = label.strip("/").split("/")
    semantic_parts = semantic_name.strip("/").split("/")
    if not label_parts or not semantic_parts or label_parts[-1] != semantic_parts[-1]:
        return False
    positions = iter(label_parts[:-1])
    return all(part in positions for part in semantic_parts[:-1])
