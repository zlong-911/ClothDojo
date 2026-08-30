"""Small public-API runtime for the MuJoCo and Style3D parity baseline."""

from __future__ import annotations

import numpy as np
import warp as wp

from clothdojo.sim.grasp import ParticleGrasp
from clothdojo.sim.scene import ShirtFlattenScene

_CLOTH_BVH_REBUILD_CONTROL_STEPS = 2
_CUDA_GRAPH_CONDITIONAL_MIN_DRIVER = (12, 4)


@wp.kernel
def _write_arm_targets(
    action: wp.array2d(dtype=float),
    lower: wp.array3d(dtype=float),
    upper: wp.array3d(dtype=float),
    joints_per_articulation: int,
    target: wp.array3d(dtype=float),
):
    world, joint = wp.tid()
    articulation = joint // joints_per_articulation
    articulation_joint = joint % joints_per_articulation
    target[world, articulation, articulation_joint] = wp.clamp(
        action[world, joint],
        lower[world, articulation, articulation_joint],
        upper[world, articulation, articulation_joint],
    )


@wp.kernel
def _write_gripper_targets(
    openness: wp.array2d(dtype=float),
    closed_positions: wp.array(dtype=float),
    open_positions: wp.array(dtype=float),
    joints_per_articulation: int,
    joints_per_side: int,
    target: wp.array3d(dtype=float),
):
    world, joint = wp.tid()
    articulation = joint // joints_per_articulation
    articulation_joint = joint % joints_per_articulation
    side = joint // joints_per_side
    amount = wp.clamp(openness[world, side], 0.0, 1.0)
    target[world, articulation, articulation_joint] = closed_positions[joint] + amount * (
        open_positions[joint] - closed_positions[joint]
    )


@wp.func
def _selected_world(world: int, world_count: int, mask: wp.array(dtype=bool)) -> bool:
    slot = world if world >= 0 else world_count
    return mask[slot]


@wp.kernel
def _reset_particles(
    world_count: int,
    world_mask: wp.array(dtype=bool),
    particle_world: wp.array(dtype=int),
    initial_q: wp.array(dtype=wp.vec3),
    initial_qd: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()
    if _selected_world(particle_world[particle], world_count, world_mask):
        q[particle] = initial_q[particle]
        qd[particle] = initial_qd[particle]
        force[particle] = wp.vec3(0.0)


@wp.kernel
def _reset_style3d_dx(
    world_count: int,
    world_mask: wp.array(dtype=bool),
    particle_world: wp.array(dtype=int),
    dx: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()
    if _selected_world(particle_world[particle], world_count, world_mask):
        dx[particle] = wp.vec3(0.0)


@wp.kernel
def _damp_particles(
    retention: float,
    particle_qd: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()
    particle_qd[particle] = particle_qd[particle] * retention


def _prune_static_soft_contact_pairs(model, collision_pipeline) -> None:
    """Remove particle-shape pairs disabled by immutable scene shape flags.

    Newton 1.5 intentionally builds every world-compatible particle-shape pair
    so applications can toggle ``COLLIDE_PARTICLES`` after construction.  This
    scene never changes shape flags at runtime, and render/proxy shapes make
    those dormant pairs dominate the contact buffer.  Keep this version-pinned
    adaptation next to pipeline construction and validate Newton's default
    capacity before narrowing its private pair count.
    """

    import newton

    pairs = collision_pipeline.soft_rigid_contact_pairs
    if len(pairs) == 0:
        return
    full_surface_count = len(collision_pipeline.soft_edge_rigid_pairs) + len(
        collision_pipeline.soft_face_rigid_pairs
    )
    expected_capacity = len(pairs) + full_surface_count
    if collision_pipeline.soft_contact_max != expected_capacity:
        raise RuntimeError("static soft-contact pruning requires Newton 1.5's default capacity")
    host_pairs = np.asarray(pairs.numpy(), dtype=np.int32)
    shape_flags = np.asarray(model.shape_flags.numpy())
    keep = (shape_flags[host_pairs[:, 1]] & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0
    if bool(keep.all()):
        return
    filtered = np.ascontiguousarray(host_pairs[keep])
    collision_pipeline.soft_rigid_contact_pairs = wp.array(
        filtered,
        dtype=wp.vec2i,
        device=model.device,
    )
    collision_pipeline._soft_rigid_contact_pair_count = len(filtered)
    collision_pipeline._soft_contact_max = len(filtered) + full_surface_count


class ShirtFlattenRuntime:
    """Own state, controls, contacts and the one intentional two-solver step."""

    def __init__(
        self,
        scene: ShirtFlattenScene,
        *,
        use_cuda_graph: bool = True,
        rigid_deterministic: bool = False,
    ):
        import newton

        if not isinstance(use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be a bool")
        if not isinstance(rigid_deterministic, bool):
            raise TypeError("rigid_deterministic must be a bool")
        self.scene = scene
        self.model = scene.model
        self.spec = scene.spec
        self.use_cuda_graph = use_cuda_graph
        self._step_graph = None
        self._control_steps_since_cloth_bvh_rebuild = 0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        collision_options = {
            "broad_phase": "explicit",
            "rigid_contact_max": self.spec.rigid_contact_max,
            "include_static_kinematic_pairs": False,
            "deterministic": True,
        }
        self.rigid_collision = newton.CollisionPipeline(
            self.model,
            soft_contact_max=0,
            **collision_options,
        )
        self.rigid_contacts = self.rigid_collision.contacts()
        self.collision = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.spec.cloth_contact_margin,
            **collision_options,
        )
        _prune_static_soft_contact_pairs(self.model, self.collision)
        self.contacts = self.collision.contacts()
        self.rigid = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=2048,
            nconmax=2048,
            solver="newton",
            integrator="implicitfast",
            cone="pyramidal",
            iterations=100,
            ls_iterations=50,
            impratio=1.0,
            use_mujoco_cpu=False,
            use_mujoco_contacts=False,
            # ClothDojo does not consume MuJoCo sensors. State synchronization
            # remains enabled because cold trajectory restore also updates the
            # robot coordinates owned by Newton state.
            disable_sensors=True,
            # Full cloth stepping does not promise bitwise repeatability. The
            # rigid-only retarget path opts into MuJoCo-Warp's deterministic
            # kernels so repeated action replays have a meaningful hard gate.
            deterministic=(
                wp.DeterministicMode.RUN_TO_RUN
                if rigid_deterministic
                else wp.DeterministicMode.NOT_GUARANTEED
            ),
        )
        if (
            self.rigid.mjw_model is not None
            and wp.get_device(self.model.device).is_cuda
            and wp.get_cuda_driver_version() < _CUDA_GRAPH_CONDITIONAL_MIN_DRIVER
        ):
            # MuJoCo Warp enables conditional graph nodes by default, but
            # NVIDIA drivers older than CUDA 12.4 cannot capture them. Keep
            # ClothDojo's outer whole-step CUDA Graph and use MuJoCo Warp's
            # fixed-iteration graph on those hosts.
            self.rigid.mjw_model.opt.graph_conditional = False
        self.cloth = newton.solvers.SolverStyle3D(
            self.model,
            iterations=self.spec.style3d_iterations,
            linear_iterations=self.spec.style3d_linear_iterations,
            self_contact_max_candidates_per_primitive=(
                self.spec.self_contact_max_candidates_per_primitive
            ),
            self_contact_friction_mu=self.spec.self_contact_friction_mu,
            self_contact_friction_epsilon=self.spec.self_contact_friction_epsilon,
            suppress_self_friction_during_untangling=(
                self.spec.suppress_self_friction_during_untangling
            ),
        )
        self.cloth.collision.body_contact_max = self.contacts.soft_contact_max
        self.cloth.collision.radius = self.spec.self_contact_radius
        self.cloth.collision.friction_epsilon = self.spec.cloth_body_friction_epsilon
        self.cloth.collision.stiff_vf = self.spec.self_contact_vf_stiffness
        self.cloth.collision.stiff_ee = self.spec.self_contact_ee_stiffness
        self.cloth.collision.stiff_ef = self.spec.self_contact_ef_stiffness

        self._all_worlds = wp.ones(
            self.model.world_count + 1,
            dtype=wp.bool,
            device=self.model.device,
        )
        self._arm_lower = scene.arms.get_attribute("joint_limit_lower", self.model)
        self._arm_upper = scene.arms.get_attribute("joint_limit_upper", self.model)
        self._arm_targets = wp.empty_like(self._arm_lower)
        self._arm_action_count = len(scene.robot_profile.arm_coord_indices)
        if int(np.prod(self._arm_targets.shape[1:])) != self._arm_action_count:
            raise RuntimeError("bimanual arm view differs from its robot profile")
        self._arm_joints_per_articulation = int(self._arm_targets.shape[2])
        self._gripper_targets = wp.empty_like(scene.grippers.get_attribute("joint_q", self.state_0))
        self._gripper_action_count = len(scene.robot_profile.gripper_coord_indices)
        if int(np.prod(self._gripper_targets.shape[1:])) != self._gripper_action_count:
            raise RuntimeError("bimanual gripper view differs from its robot profile")
        self._gripper_joints_per_articulation = int(self._gripper_targets.shape[2])
        self._gripper_closed_positions = wp.array(
            scene.robot_profile.gripper_closed_positions * 2,
            dtype=wp.float32,
            device=self.model.device,
        )
        self._gripper_open_positions = wp.array(
            scene.robot_profile.gripper_open_positions * 2,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.grasp = ParticleGrasp(scene)
        self.reset()
        self._graph_arm_target = wp.empty(
            (self.model.world_count, self._arm_action_count),
            dtype=wp.float32,
            device=self.model.device,
        )
        self._graph_gripper_open = wp.empty(
            (self.model.world_count, 2),
            dtype=wp.float32,
            device=self.model.device,
        )
        if self.use_cuda_graph and wp.get_device(self.model.device).is_cuda:
            self._capture_step_graph()

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.spec.control_fps)

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._step_graph is not None

    def apply_action(self, arm_target, gripper_open) -> None:
        """Apply one profile-sized arm action and two gripper commands per world."""

        arm_shape = (self.model.world_count, self._arm_action_count)
        self._validate_action("arm_target", arm_target, arm_shape)
        self._validate_action("gripper_open", gripper_open, (self.model.world_count, 2))
        wp.launch(
            _write_arm_targets,
            dim=arm_shape,
            inputs=[
                arm_target,
                self._arm_lower,
                self._arm_upper,
                self._arm_joints_per_articulation,
            ],
            outputs=[self._arm_targets],
            device=self.model.device,
        )
        self.scene.arms.set_attribute("joint_target_q", self.control, self._arm_targets)
        wp.launch(
            _write_gripper_targets,
            dim=(self.model.world_count, self._gripper_action_count),
            inputs=[
                gripper_open,
                self._gripper_closed_positions,
                self._gripper_open_positions,
                self._gripper_joints_per_articulation,
                self.scene.robot_profile.gripper_coords_per_side,
            ],
            outputs=[self._gripper_targets],
            device=self.model.device,
        )
        self.scene.grippers.set_attribute("joint_target_q", self.control, self._gripper_targets)

    def step_rigid(self, arm_target, gripper_open) -> None:
        """Advance only target-robot rigid dynamics for one control interval."""

        self._validate_action(
            "arm_target", arm_target, (self.model.world_count, self._arm_action_count)
        )
        self._validate_action("gripper_open", gripper_open, (self.model.world_count, 2))
        self.apply_action(arm_target, gripper_open)
        sub_dt = self.control_dt / float(self.spec.sim_substeps)
        for _ in range(self.spec.sim_substeps):
            self._step_rigid(sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def restore_rigid_state(self, joint_q: np.ndarray, joint_qd: np.ndarray) -> None:
        """Cold-restore one rigid boundary for disposable action shooting."""

        if self.model.world_count != 1:
            raise ValueError("rigid trajectory restore requires one world")
        self.restore_rigid_state_batch(
            np.asarray(joint_q, dtype=np.float32)[None],
            np.asarray(joint_qd, dtype=np.float32)[None],
        )

    def restore_rigid_state_batch(self, joint_q: np.ndarray, joint_qd: np.ndarray) -> None:
        """Cold-restore one rigid boundary per world for batched shooting."""

        import newton

        world_count = int(self.model.world_count)
        coord_count = int(self.model.joint_coord_count) // world_count
        dof_count = int(self.model.joint_dof_count) // world_count
        arrays = {
            "joint_q": np.asarray(joint_q, dtype=np.float32),
            "joint_qd": np.asarray(joint_qd, dtype=np.float32),
        }
        expected = {
            "joint_q": (world_count, coord_count),
            "joint_qd": (world_count, dof_count),
        }
        for name, value in arrays.items():
            if value.shape != expected[name] or not np.isfinite(value).all():
                raise ValueError(
                    f"{name} must be finite with shape {expected[name]}, got {value.shape}"
                )
        for state in (self.state_0, self.state_1):
            state.joint_q.assign(arrays["joint_q"].reshape(-1))
            state.joint_qd.assign(arrays["joint_qd"].reshape(-1))
            state.clear_forces()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        # Episode trajectories do not store MuJoCo warm-start buffers. Every
        # shooting branch therefore starts from the same canonical cold state.
        self.rigid.reset(self.state_0, flags=0)

    def rigid_state(self) -> dict[str, np.ndarray]:
        """Copy the current one-world articulated state in canonical layout."""

        if self.model.world_count != 1:
            raise ValueError("canonical rigid state capture requires one world")
        return {name: value[0] for name, value in self.rigid_state_batch().items()}

    def rigid_state_batch(self) -> dict[str, np.ndarray]:
        """Copy articulated state grouped by physics world."""

        world_count = int(self.model.world_count)
        counts = {
            "joint_q": int(self.model.joint_coord_count) // world_count,
            "joint_qd": int(self.model.joint_dof_count) // world_count,
            "body_q": int(self.model.body_count) // world_count,
            "body_qd": int(self.model.body_count) // world_count,
        }
        result = {}
        for name, count in counts.items():
            value = np.asarray(getattr(self.state_0, name).numpy(), dtype=np.float32)
            result[name] = value.reshape((world_count, count, *value.shape[1:])).copy()
        return result

    def inverse_dynamics(self, desired_joint_acceleration: np.ndarray) -> np.ndarray:
        """Return actuator effort for a desired generalized acceleration.

        Newton 1.5 does not expose inverse dynamics on ``SolverMuJoCo``. This
        version-pinned adapter uses the same MuJoCo Warp model and coordinate
        conversion as forward stepping. It supports unactuated mimic joints:
        their accelerations participate in inverse dynamics, while only
        actuator-backed coordinates receive an output effort.
        """

        solver = self.rigid
        if solver.use_mujoco_cpu or solver.mjw_model is None or solver.mjw_data is None:
            raise RuntimeError("inverse dynamics requires the MuJoCo Warp solver")
        mapping = np.asarray(solver.mjc_actuator_to_newton_idx.numpy(), dtype=np.int64)
        if np.any(mapping < 0):
            raise RuntimeError("inverse dynamics requires position-controlled actuators")
        mj_model = solver.mj_model
        actuator_joint = np.asarray(mj_model.actuator_trnid[:, 0], dtype=np.int64)
        actuator_dof = np.asarray(mj_model.jnt_dofadr[actuator_joint], dtype=np.int64)
        dof_map = np.asarray(solver.mjc_dof_to_newton_dof.numpy(), dtype=np.int64)
        world_count = int(solver.mjw_data.nworld)
        dofs_per_world = int(self.model.joint_dof_count) // world_count
        local_dof_map = dof_map % dofs_per_world
        if np.any(dof_map < 0) or not np.array_equal(
            mapping % dofs_per_world, local_dof_map[0, actuator_dof]
        ):
            raise RuntimeError("Newton and MuJoCo scalar joint layouts differ")

        desired = np.asarray(desired_joint_acceleration, dtype=np.float32)
        expected = (world_count, dofs_per_world)
        if desired.shape != expected or not np.isfinite(desired).all():
            raise ValueError(f"desired_joint_acceleration must be finite with shape {expected}")
        solver._update_mjc_data(solver.mjw_data, self.model, self.state_0)
        solver.mjw_data.qacc.assign(np.take_along_axis(desired, local_dof_map, axis=1))
        solver.mjw_model.opt.timestep.fill_(self.control_dt / float(self.spec.sim_substeps))
        import mujoco_warp

        mujoco_warp.inverse(solver.mjw_model, solver.mjw_data)
        mj_effort = np.asarray(solver.mjw_data.qfrc_inverse.numpy(), dtype=np.float64)
        mj_effort += np.asarray(solver.mjw_data.qfrc_constraint.numpy(), dtype=np.float64)
        effort = np.zeros_like(desired, dtype=np.float64)
        effort[:, mapping % dofs_per_world] = mj_effort[:, actuator_dof]
        return effort

    def step(self, arm_target, gripper_open) -> None:
        self._validate_action(
            "arm_target", arm_target, (self.model.world_count, self._arm_action_count)
        )
        self._validate_action("gripper_open", gripper_open, (self.model.world_count, 2))
        if self._step_graph is not None:
            wp.copy(self._graph_arm_target, arm_target)
            wp.copy(self._graph_gripper_open, gripper_open)
            wp.capture_launch(self._step_graph)
        else:
            self._step_eager(arm_target, gripper_open)
        self._control_steps_since_cloth_bvh_rebuild += 1
        if self._control_steps_since_cloth_bvh_rebuild >= _CLOTH_BVH_REBUILD_CONTROL_STEPS:
            # Newton 1.5's Style3D collision owns one global BVH for every
            # world. Rebuilding every frame is cheap for B1 but dominates B8;
            # two control steps bounds refit aging while retaining batched
            # throughput. The rebuild remains asynchronous on the same stream.
            self.cloth.rebuild_bvh(self.state_0)
            self._control_steps_since_cloth_bvh_rebuild = 0

    def _step_eager(self, arm_target, gripper_open) -> None:
        self.apply_action(arm_target, gripper_open)
        sub_dt = self.control_dt / float(self.spec.sim_substeps)
        for _ in range(self.spec.sim_substeps):
            self._step_rigid(sub_dt)
            self.grasp.update(self.state_0, self.state_1)
            self._step_cloth(sub_dt)

    def _capture_step_graph(self) -> None:
        if self.spec.sim_substeps % 2:
            raise RuntimeError("CUDA Graph requires an even sim_substeps state-swap cycle")
        self._graph_arm_target.zero_()
        self._graph_gripper_open.fill_(1.0)
        # Compile and allocate every lazy Newton/Warp path before capture, then
        # restore the exact initial state so graph setup is not a simulation step.
        self._step_eager(self._graph_arm_target, self._graph_gripper_open)
        wp.synchronize_device(self.model.device)
        self.reset()
        wp.synchronize_device(self.model.device)
        with wp.ScopedCapture(device=self.model.device, force_module_load=False) as capture:
            self._step_eager(self._graph_arm_target, self._graph_gripper_open)
        self._step_graph = capture.graph
        self.reset()
        wp.synchronize_device(self.model.device)

    def step_cloth(self, *, velocity_retention: float = 1.0, static_patch=None) -> None:
        """Advance cloth against a stationary robot during initial-state preparation."""

        if not 0.0 <= velocity_retention <= 1.0:
            raise ValueError("velocity_retention must be in [0, 1]")
        self.cloth.rebuild_bvh(self.state_0)
        sub_dt = self.control_dt / float(self.spec.sim_substeps)
        for _ in range(self.spec.sim_substeps):
            self._step_cloth(sub_dt, static_patch=static_patch)
        if velocity_retention < 1.0:
            wp.launch(
                _damp_particles,
                dim=self.model.particle_count,
                inputs=[float(velocity_retention), self.state_0.particle_qd],
                device=self.model.device,
            )

    def advance_cloth(self, steps: int, *, graph=None, static_patch=None):
        """Advance a cloth-only phase, reusing one captured Newton 1.5 step graph."""

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("cloth step count must be a non-negative integer")
        if not steps:
            return graph
        device = wp.get_device(self.model.device)
        if not device.is_cuda:
            for _ in range(steps):
                self.step_cloth(static_patch=static_patch)
            return None
        if graph is None:
            self.step_cloth(static_patch=static_patch)
            steps -= 1
            if steps:
                with wp.ScopedCapture(device=device) as capture:
                    self.step_cloth(static_patch=static_patch)
                graph = capture.graph
                steps -= 1
        for _ in range(steps):
            wp.capture_launch(graph)
        return graph

    def cloth_contact_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Count current garment contacts with the table and robot per world."""

        self.collision.collide(self.state_0, self.contacts)
        count = int(self.contacts.soft_contact_count.numpy()[0])
        count = min(count, len(self.contacts.soft_contact_particle))
        particles = self.contacts.soft_contact_particle.numpy()[:count]
        shapes = self.contacts.soft_contact_shape.numpy()[:count]
        shape_bodies = self.model.shape_body.numpy()
        particle_world = self.model.particle_world.numpy()
        table = np.zeros(self.model.world_count, dtype=np.int32)
        robot = np.zeros(self.model.world_count, dtype=np.int32)
        for particle, shape in zip(particles, shapes, strict=True):
            particle = int(particle)
            shape = int(shape)
            if particle < 0 or shape < 0:
                continue
            world = int(particle_world[particle])
            target = table if int(shape_bodies[shape]) < 0 else robot
            target[world] += 1
        return table, robot

    def reset(self, world_mask=None) -> None:
        """Reset selected worlds using Newton 1.5's trailing-global mask layout."""

        import newton

        world_mask = self._world_mask(world_mask)
        for state in (self.state_0, self.state_1):
            self.rigid.reset(state, world_mask=world_mask)
            wp.launch(
                _reset_particles,
                dim=self.model.particle_count,
                inputs=[
                    self.model.world_count,
                    world_mask,
                    self.model.particle_world,
                    self.model.particle_q,
                    self.model.particle_qd,
                ],
                outputs=[state.particle_q, state.particle_qd, state.particle_f],
                device=self.model.device,
            )
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        wp.launch(
            _reset_style3d_dx,
            dim=self.model.particle_count,
            inputs=[
                self.model.world_count,
                world_mask,
                self.model.particle_world,
            ],
            outputs=[self.cloth.dx],
            device=self.model.device,
        )
        self.cloth.rebuild_bvh(self.state_0)
        self._control_steps_since_cloth_bvh_rebuild = 0
        self.grasp.reset(world_mask)
        articulation_mask = world_mask[: self.model.world_count]
        self.scene.arms.set_attribute(
            "joint_target_q",
            self.control,
            self.scene.arms.get_attribute("joint_q", self.state_0),
            mask=articulation_mask,
        )
        self.scene.grippers.set_attribute(
            "joint_target_q",
            self.control,
            self.scene.grippers.get_attribute("joint_q", self.state_0),
            mask=articulation_mask,
        )

    def restore_particles(self, particle_q: np.ndarray, particle_qd: np.ndarray) -> None:
        """Restore one garment state per world and reset transient solver state."""

        q = np.asarray(particle_q)
        qd = np.asarray(particle_qd)
        ranges = self.grasp.particle_ranges()
        counts = {end - start for start, end in ranges}
        if len(counts) != 1:
            raise RuntimeError("shirt-flattening worlds have different particle counts")
        expected = (self.model.world_count, counts.pop(), 3)
        if q.shape != expected or qd.shape != expected:
            raise ValueError(
                f"restored particle_q/particle_qd must both have shape {expected}, "
                f"got {q.shape}/{qd.shape}"
            )
        if q.dtype != np.float32 or qd.dtype != np.float32:
            raise TypeError("restored particle states must use float32")
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("restored particle states must be finite")

        full_q = self.state_0.particle_q.numpy()
        full_qd = self.state_0.particle_qd.numpy()
        world_offsets = np.asarray(self.scene.world_offsets, dtype=np.float32)
        if world_offsets.shape != (self.model.world_count, 3):
            raise RuntimeError("scene world offsets do not match the Newton world count")
        for world, (start, end) in enumerate(ranges):
            full_q[start:end] = q[world] + world_offsets[world]
            full_qd[start:end] = qd[world]
        for state in (self.state_0, self.state_1):
            state.particle_q.assign(full_q)
            state.particle_qd.assign(full_qd)
            state.particle_f.zero_()
        self.cloth.dx.zero_()
        self.cloth.rebuild_bvh(self.state_0)
        self._control_steps_since_cloth_bvh_rebuild = 0
        self.grasp.reset(self._all_worlds)

    def restore_state(self, frame) -> None:
        """Cold-restore one canonical trajectory boundary without reallocating GPU state."""

        import newton

        if self.model.world_count != 1:
            raise ValueError("trajectory cold restore requires one world")
        arrays = self._validated_restore_arrays(frame)
        self.grasp.validate_restore_state(frame)
        robot_values = {
            "joint_q": arrays["joint_q"][None, None],
            "joint_qd": arrays["joint_qd"][None, None],
        }
        start, end = self.grasp.particle_ranges()[0]
        if (start, end) != (0, int(self.model.particle_count)):
            raise RuntimeError("single-world garment particles must cover the particle state")
        for state in (self.state_0, self.state_1):
            for name, value in robot_values.items():
                self.scene.robot.get_attribute(name, state).assign(value)
            state.particle_q.assign(arrays["particle_q"])
            state.particle_qd.assign(arrays["particle_qd"])
            state.clear_forces()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        # flags=0 keeps the restored joint coordinates while clearing MuJoCo's
        # acceleration, force, activation and control warm-start buffers.
        self.rigid.reset(self.state_0, flags=0)
        self.cloth.dx.zero_()
        self.cloth.rebuild_bvh(self.state_0)
        self._control_steps_since_cloth_bvh_rebuild = 0
        self.grasp.restore_state(frame)
        self.scene.arms.set_attribute(
            "joint_target_q",
            self.control,
            self.scene.arms.get_attribute("joint_q", self.state_0),
        )
        self.scene.grippers.set_attribute(
            "joint_target_q",
            self.control,
            self.scene.grippers.get_attribute("joint_q", self.state_0),
        )
        wp.synchronize_device(self.model.device)

    def _validated_restore_arrays(self, frame) -> dict[str, np.ndarray]:
        expected = {
            "joint_q": tuple(self.scene.robot.get_attribute("joint_q", self.state_0).shape[2:]),
            "joint_qd": tuple(self.scene.robot.get_attribute("joint_qd", self.state_0).shape[2:]),
            "body_q": (
                int(self.scene.robot.get_attribute("body_q", self.state_0).shape[2]),
                7,
            ),
            "body_qd": (
                int(self.scene.robot.get_attribute("body_qd", self.state_0).shape[2]),
                6,
            ),
        }
        start, end = self.grasp.particle_ranges()[0]
        expected.update(
            {
                "particle_q": (end - start, 3),
                "particle_qd": (end - start, 3),
            }
        )
        arrays = {}
        for name, shape in expected.items():
            value = np.asarray(getattr(frame, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            if value.dtype != np.float32:
                raise TypeError(f"{name} dtype {value.dtype} != float32")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            arrays[name] = np.ascontiguousarray(value)
        gripper_open = np.asarray(frame.gripper_open_state)
        if gripper_open.shape != (2,) or gripper_open.dtype != np.float32:
            raise ValueError("gripper_open_state must be float32 with shape (2,)")
        if not np.isfinite(gripper_open).all() or np.any(
            (gripper_open < 0.0) | (gripper_open > 1.0)
        ):
            raise ValueError("gripper_open_state must be finite and in [0, 1]")
        return arrays

    def _step_rigid(self, dt: float) -> None:
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.rigid_collision.collide(self.state_0, self.rigid_contacts)
        self.rigid.step(
            self.state_0,
            self.state_1,
            self.control,
            self.rigid_contacts,
            dt,
        )

    def _collide_cloth_with_updated_rigid_state(self) -> None:
        """Collide current cloth against the rigid state produced this substep."""

        # MuJoCo writes the updated rigid state to state_1 while the current
        # cloth remains in state_0. Borrow state_1 as a zero-copy contact view:
        # its body transforms stay current, and only its particle position
        # reference is temporarily redirected to the current cloth buffer.
        particle_q_out = self.state_1.particle_q
        self.state_1.particle_q = self.state_0.particle_q
        try:
            self.collision.collide(self.state_1, self.contacts)
        finally:
            self.state_1.particle_q = particle_q_out

    def _step_cloth(self, dt: float, *, static_patch=None) -> None:
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.grasp.enforce(self.state_0, self.state_1, dt)
        if static_patch is not None:
            static_patch.enforce(self.state_0)
        self._collide_cloth_with_updated_rigid_state()
        self.cloth.step(self.state_0, self.state_1, None, self.contacts, dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        if static_patch is not None:
            static_patch.enforce(self.state_0)

    def _world_mask(self, world_mask):
        if world_mask is None:
            return self._all_worlds
        if not isinstance(world_mask, wp.array) or world_mask.dtype != wp.bool:
            raise TypeError("world_mask must be a Warp bool array")
        expected = (self.model.world_count + 1,)
        if world_mask.shape != expected:
            raise ValueError(f"world_mask shape {world_mask.shape} != {expected}")
        if world_mask.device != wp.get_device(self.model.device):
            raise ValueError("world_mask must be on the model device")
        return world_mask

    def _validate_action(self, name: str, value, shape: tuple[int, int]) -> None:
        if not isinstance(value, wp.array) or value.dtype != wp.float32:
            raise TypeError(f"{name} must be a Warp float32 array")
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if value.device != wp.get_device(self.model.device):
            raise ValueError(f"{name} must be on the model device")
