"""A fixed-allocation connected cloth patch grasp for the Piper grippers."""

from __future__ import annotations

import numpy as np
import warp as wp

from clothdojo.sim.scene import ShirtFlattenScene

_ATTACH_KE = 0.5
_ATTACH_KD = 0.017
_ATTACH_FORCE_MAX = 0.2
_ATTACH_TIME_CONSTANT_S = 0.00467279


@wp.kernel
def _reset_grasp(
    world_mask: wp.array(dtype=bool),
    attached: wp.array2d(dtype=int),
    previous_open: wp.array2d(dtype=float),
    pin_count: wp.array2d(dtype=int),
    pins: wp.array3d(dtype=int),
    pin_local: wp.array3d(dtype=wp.vec3),
):
    world, side = wp.tid()
    if not world_mask[world]:
        return
    attached[world, side] = 0
    previous_open[world, side] = 1.0
    pin_count[world, side] = 0
    for slot in range(pins.shape[2]):
        pins[world, side, slot] = -1
        pin_local[world, side, slot] = wp.vec3(0.0)


@wp.kernel
def _update_grasp(
    gripper_q: wp.array3d(dtype=float),
    gripper_lower: wp.array3d(dtype=float),
    gripper_upper: wp.array3d(dtype=float),
    ee_body: wp.array2d(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    particle_q: wp.array(dtype=wp.vec3),
    particle_start: wp.array(dtype=int),
    particle_end: wp.array(dtype=int),
    neighbor_offsets: wp.array(dtype=int),
    neighbor_indices: wp.array(dtype=int),
    attached: wp.array2d(dtype=int),
    previous_open: wp.array2d(dtype=float),
    pin_count: wp.array2d(dtype=int),
    pins: wp.array3d(dtype=int),
    pin_local: wp.array3d(dtype=wp.vec3),
    local_offsets: wp.array(dtype=wp.vec3),
    attach_threshold: float,
    release_threshold: float,
    capture_radius_sq: float,
):
    world, side = wp.tid()
    joint = side * 2
    lower = gripper_lower[world, 0, joint]
    upper = gripper_upper[world, 0, joint]
    openness = wp.clamp((gripper_q[world, 0, joint] - lower) / (upper - lower), 0.0, 1.0)

    if attached[world, side] != 0:
        if openness > release_threshold:
            attached[world, side] = 0
            pin_count[world, side] = 0
        previous_open[world, side] = openness
        return

    crossed = previous_open[world, side] >= attach_threshold and openness < attach_threshold
    previous_open[world, side] = openness
    if not crossed:
        return

    transform = body_q[ee_body[world, side]]
    rotation = wp.transform_get_rotation(transform)
    center = wp.transform_point(transform, local_offsets[side])
    start = particle_start[world]
    end = particle_end[world]
    best_particle = int(-1)  # noqa: RUF046, UP018
    best_distance = float(1.0e30)  # noqa: UP018
    for particle in range(start, end):
        distance = wp.length_sq(particle_q[particle] - center)
        if distance <= capture_radius_sq and distance < best_distance:
            best_particle = particle
            best_distance = distance

    selected_count = int(0)  # noqa: RUF046, UP018
    if best_particle >= 0:
        pins[world, side, selected_count] = best_particle
        selected_count += 1
        for edge in range(
            neighbor_offsets[best_particle], neighbor_offsets[best_particle + 1]
        ):
            if selected_count < pins.shape[2]:
                pins[world, side, selected_count] = neighbor_indices[edge]
                selected_count += 1

    if selected_count > 0:
        frame_position = wp.transform_get_translation(transform) + wp.quat_rotate(
            rotation, local_offsets[side]
        )
        for slot in range(selected_count):
            particle = pins[world, side, slot]
            pin_local[world, side, slot] = wp.quat_rotate_inv(
                rotation, particle_q[particle] - frame_position
            )
        pin_count[world, side] = selected_count
        attached[world, side] = 1


@wp.kernel
def _enforce_grasp(
    pin_count: wp.array2d(dtype=int),
    pins: wp.array3d(dtype=int),
    pin_local: wp.array3d(dtype=wp.vec3),
    ee_body: wp.array2d(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    local_offsets: wp.array(dtype=wp.vec3),
    dt: float,
    attach_ke: float,
    attach_kd: float,
    attach_force_max: float,
    position_time_constant: float,
    velocity_time_constant: float,
):
    world, side, slot = wp.tid()
    if slot >= pin_count[world, side]:
        return
    body = ee_body[world, side]
    transform = body_q[body]
    rotation = wp.transform_get_rotation(transform)
    offset = wp.quat_rotate(rotation, local_offsets[side])
    frame_position = wp.transform_get_translation(transform) + offset
    velocity = body_qd[body]
    linear = wp.spatial_top(velocity)
    omega = wp.spatial_bottom(velocity)
    com = wp.transform_point(transform, body_com[body])
    frame_velocity = linear + wp.cross(omega, frame_position - com)

    particle = pins[world, side, slot]
    local = wp.quat_rotate(rotation, pin_local[world, side, slot])
    target_position = frame_position + local
    target_velocity = frame_velocity + wp.cross(omega, local)
    position_delta = target_position - particle_q[particle]
    velocity_delta = target_velocity - particle_qd[particle]
    force = position_delta * attach_ke + velocity_delta * attach_kd
    force_length = wp.length(force)
    if attach_force_max > 0.0 and force_length > attach_force_max:
        force = force * (attach_force_max / force_length)
    wp.atomic_add(particle_f, particle, force)
    position_blend = 1.0 - wp.exp(-dt / position_time_constant)
    velocity_blend = 1.0 - wp.exp(-dt / velocity_time_constant)
    particle_q[particle] = particle_q[particle] + position_delta * position_blend
    particle_qd[particle] = particle_qd[particle] + velocity_delta * velocity_blend


@wp.kernel
def _enforce_static_patch(
    enabled: wp.array(dtype=int),
    pin_count: wp.array(dtype=int),
    pins: wp.array2d(dtype=int),
    targets: wp.array2d(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
):
    world, slot = wp.tid()
    if enabled[0] == 0 or slot >= pin_count[world]:
        return
    particle = pins[world, slot]
    particle_q[particle] = targets[world, slot]
    particle_qd[particle] = wp.vec3(0.0)
    particle_f[particle] = wp.vec3(0.0)


class StaticParticlePatch:
    """Kinematically hold one fixed connected particle patch per world."""

    capacity = 32

    def __init__(self, scene: ShirtFlattenScene):
        self.world_count = int(scene.model.world_count)
        self.device = scene.model.device
        self.enabled = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.pin_count = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.pins = wp.full(
            (self.world_count, self.capacity),
            -1,
            dtype=wp.int32,
            device=self.device,
        )
        self.targets = wp.zeros(
            (self.world_count, self.capacity),
            dtype=wp.vec3,
            device=self.device,
        )

    def configure(self, patches: list[np.ndarray], particle_q) -> None:
        if len(patches) != self.world_count:
            raise ValueError("static particle patches must contain one patch per world")
        positions = particle_q.numpy()
        counts = np.asarray([len(patch) for patch in patches], dtype=np.int32)
        if np.any(counts < 1) or np.any(counts > self.capacity):
            raise ValueError(f"static particle patch size must be in [1, {self.capacity}]")
        pins = np.full((self.world_count, self.capacity), -1, dtype=np.int32)
        targets = np.zeros((self.world_count, self.capacity, 3), dtype=np.float32)
        for world, patch in enumerate(patches):
            values = np.asarray(patch, dtype=np.int32)
            if len(np.unique(values)) != len(values):
                raise ValueError("static particle patch cannot contain duplicate particles")
            pins[world, : len(values)] = values
            targets[world, : len(values)] = positions[values]
        self.pin_count.assign(counts)
        self.pins.assign(pins)
        self.targets.assign(targets)
        self.enabled.assign(np.ones(1, dtype=np.int32))

    def disable(self) -> None:
        self.enabled.zero_()

    def enforce(self, state) -> None:
        wp.launch(
            _enforce_static_patch,
            dim=(self.world_count, self.capacity),
            inputs=[
                self.enabled,
                self.pin_count,
                self.pins,
                self.targets,
                state.particle_q,
                state.particle_qd,
                state.particle_f,
            ],
            device=self.device,
        )


class ParticleGrasp:
    """Capture and softly pin one connected garment patch per gripper."""

    pin_capacity = 32

    def __init__(self, scene: ShirtFlattenScene):
        self.scene = scene
        self.model = scene.model
        self.world_count = int(self.model.world_count)
        self.device = self.model.device
        shape2 = (self.world_count, 2)
        shape3 = (self.world_count, 2, self.pin_capacity)
        self.attached = wp.zeros(shape2, dtype=wp.int32, device=self.device)
        self.previous_open = wp.ones(shape2, dtype=wp.float32, device=self.device)
        self.pin_count = wp.zeros(shape2, dtype=wp.int32, device=self.device)
        self.pins = wp.full(shape3, -1, dtype=wp.int32, device=self.device)
        self.pin_local = wp.zeros(shape3, dtype=wp.vec3, device=self.device)
        self._local_offsets = wp.array(
            ((0.0, 0.0, 0.13503), (0.0, 0.0, 0.13503)),
            dtype=wp.vec3,
            device=self.device,
        )
        self._gripper_lower = scene.grippers.get_attribute("joint_limit_lower", self.model)
        self._gripper_upper = scene.grippers.get_attribute("joint_limit_upper", self.model)
        bodies_per_world = int(self.model.body_count) // self.world_count
        self._ee_body = wp.array(
            [
                [body + world * bodies_per_world for body in scene.end_effector_body_indices]
                for world in range(self.world_count)
            ],
            dtype=wp.int32,
            device=self.device,
        )
        self._particle_start, self._particle_end = self._particle_ranges()
        self._neighbor_offsets, self._neighbor_indices = self._particle_neighbors()

    def reset(self, world_mask) -> None:
        wp.launch(
            _reset_grasp,
            dim=(self.world_count, 2),
            inputs=[
                world_mask,
                self.attached,
                self.previous_open,
                self.pin_count,
                self.pins,
                self.pin_local,
            ],
            device=self.device,
        )

    def pre_step(self, state) -> None:
        wp.launch(
            _update_grasp,
            dim=(self.world_count, 2),
            inputs=[
                self.scene.grippers.get_attribute("joint_q", state),
                self._gripper_lower,
                self._gripper_upper,
                self._ee_body,
                state.body_q,
                state.particle_q,
                self._particle_start,
                self._particle_end,
                self._neighbor_offsets,
                self._neighbor_indices,
                self.attached,
                self.previous_open,
                self.pin_count,
                self.pins,
                self.pin_local,
                self._local_offsets,
                0.1,
                0.7,
                0.01**2,
            ],
            device=self.device,
        )

    def enforce(self, state_0, state_1, dt: float) -> None:
        wp.launch(
            _enforce_grasp,
            dim=(self.world_count, 2, self.pin_capacity),
            inputs=[
                self.pin_count,
                self.pins,
                self.pin_local,
                self._ee_body,
                state_1.body_q,
                state_1.body_qd,
                self.model.body_com,
                state_0.particle_q,
                state_0.particle_qd,
                state_0.particle_f,
                self._local_offsets,
                dt,
                _ATTACH_KE,
                _ATTACH_KD,
                _ATTACH_FORCE_MAX,
                _ATTACH_TIME_CONSTANT_S,
                _ATTACH_TIME_CONSTANT_S,
            ],
            device=self.device,
        )

    def capture_all_states(self) -> tuple[dict[str, np.ndarray], ...]:
        attached = self.attached.numpy()
        previous_open = self.previous_open.numpy()
        counts = self.pin_count.numpy()
        pins = self.pins.numpy()
        local = self.pin_local.numpy()
        starts = self._particle_start.numpy()
        return tuple(
            {
                "grasp_attached": attached[world].astype(np.int32, copy=True),
                "grasp_previous_open": previous_open[world].astype(np.float32, copy=True),
                "grasp_pin_count": counts[world].astype(np.int32, copy=True),
                "grasp_pins": self._local_pins(pins[world], counts[world], int(starts[world])),
                "grasp_pin_local": local[world].astype(np.float32, copy=True),
            }
            for world in range(self.world_count)
        )

    def particle_ranges(self) -> tuple[tuple[int, int], ...]:
        starts = self._particle_start.numpy()
        ends = self._particle_end.numpy()
        return tuple((int(start), int(end)) for start, end in zip(starts, ends, strict=True))

    def connected_patch(self, particle: int) -> np.ndarray:
        """Return the center particle and the same one-ring patch used by grasping."""

        particle = int(particle)
        if not 0 <= particle < int(self.model.particle_count):
            raise IndexError("grasp particle is outside the model")
        offsets = self._neighbor_offsets.numpy()
        neighbors = self._neighbor_indices.numpy()
        return np.concatenate(
            (
                np.asarray([particle], dtype=np.int32),
                np.asarray(
                    neighbors[int(offsets[particle]) : int(offsets[particle + 1])],
                    dtype=np.int32,
                ),
            )
        )

    def _particle_ranges(self):
        per_world = int(self.model.particle_count) // self.world_count
        local_start, local_end = self.scene.garment_particles_per_world
        starts = [world * per_world + local_start for world in range(self.world_count)]
        ends = [world * per_world + local_end for world in range(self.world_count)]
        return (
            wp.array(starts, dtype=wp.int32, device=self.device),
            wp.array(ends, dtype=wp.int32, device=self.device),
        )

    def _particle_neighbors(self):
        neighbors = [set() for _ in range(int(self.model.particle_count))]
        for a, b, c in self.model.tri_indices.numpy().reshape(-1, 3):
            a, b, c = int(a), int(b), int(c)
            neighbors[a].update((b, c))
            neighbors[b].update((a, c))
            neighbors[c].update((a, b))
        offsets = np.zeros(len(neighbors) + 1, dtype=np.int32)
        flat = []
        for particle, adjacent in enumerate(neighbors):
            flat.extend(sorted(adjacent))
            offsets[particle + 1] = len(flat)
        return (
            wp.array(offsets, dtype=wp.int32, device=self.device),
            wp.array(flat, dtype=wp.int32, device=self.device),
        )

    @staticmethod
    def _local_pins(pins: np.ndarray, counts: np.ndarray, start: int) -> np.ndarray:
        result = np.full(pins.shape, -1, dtype=np.int32)
        for side, count in enumerate(counts):
            result[side, : int(count)] = pins[side, : int(count)] - start
        return result
