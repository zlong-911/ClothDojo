"""Small public-API runtime for the MuJoCo and Style3D parity baseline."""

from __future__ import annotations

import numpy as np
import warp as wp

from clothdojo.sim.grasp import ParticleGrasp
from clothdojo.sim.scene import ShirtFlattenScene


@wp.kernel
def _write_arm_targets(
    action: wp.array2d(dtype=float),
    lower: wp.array3d(dtype=float),
    upper: wp.array3d(dtype=float),
    target: wp.array3d(dtype=float),
):
    world, joint = wp.tid()
    target[world, 0, joint] = wp.clamp(
        action[world, joint], lower[world, 0, joint], upper[world, 0, joint]
    )


@wp.kernel
def _write_gripper_targets(
    openness: wp.array2d(dtype=float),
    open_positions: wp.array(dtype=float),
    target: wp.array3d(dtype=float),
):
    world, joint = wp.tid()
    side = joint // 2
    target[world, 0, joint] = wp.clamp(openness[world, side], 0.0, 1.0) * open_positions[joint]


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


class ShirtFlattenRuntime:
    """Own state, controls, contacts and the one intentional two-solver step."""

    def __init__(self, scene: ShirtFlattenScene):
        import newton

        self.scene = scene
        self.model = scene.model
        self.spec = scene.spec
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
            # Style3D is not bitwise repeatable, so do not impose a stronger
            # mode on the rigid solver or its unused MuJoCo Warp sensor module.
            deterministic=wp.DeterministicMode.NOT_GUARANTEED,
        )
        self.cloth = newton.solvers.SolverStyle3D(
            self.model,
            iterations=self.spec.style3d_iterations,
            linear_iterations=self.spec.style3d_linear_iterations,
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
        self._gripper_targets = wp.empty(
            (self.model.world_count, 1, 4),
            dtype=wp.float32,
            device=self.model.device,
        )
        self._gripper_open_positions = wp.array(
            (0.04, -0.04, 0.04, -0.04),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.grasp = ParticleGrasp(scene)
        self.reset()

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.spec.control_fps)

    def apply_action(self, arm_target, gripper_open) -> None:
        """Apply batched device actions shaped ``(worlds, 12)`` and ``(worlds, 2)``."""

        self._validate_action("arm_target", arm_target, (self.model.world_count, 12))
        self._validate_action("gripper_open", gripper_open, (self.model.world_count, 2))
        wp.launch(
            _write_arm_targets,
            dim=(self.model.world_count, 12),
            inputs=[arm_target, self._arm_lower, self._arm_upper],
            outputs=[self._arm_targets],
            device=self.model.device,
        )
        self.scene.arms.set_attribute("joint_target_q", self.control, self._arm_targets)
        wp.launch(
            _write_gripper_targets,
            dim=(self.model.world_count, 4),
            inputs=[gripper_open, self._gripper_open_positions],
            outputs=[self._gripper_targets],
            device=self.model.device,
        )
        self.scene.grippers.set_attribute("joint_target_q", self.control, self._gripper_targets)

    def step(self, arm_target, gripper_open) -> None:
        self.grasp.pre_step(self.state_0)
        self.apply_action(arm_target, gripper_open)
        sub_dt = self.control_dt / float(self.spec.sim_substeps)
        for _ in range(self.spec.sim_substeps):
            self._step_rigid(sub_dt)
            self._step_cloth(sub_dt)

    def step_cloth(self) -> None:
        """Advance cloth against a stationary robot during initial-state preparation."""

        sub_dt = self.control_dt / float(self.spec.sim_substeps)
        for _ in range(self.spec.sim_substeps):
            self._step_cloth(sub_dt)

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
        for world, (start, end) in enumerate(ranges):
            full_q[start:end] = q[world]
            full_qd[start:end] = qd[world]
        for state in (self.state_0, self.state_1):
            state.particle_q.assign(full_q)
            state.particle_qd.assign(full_qd)
            state.particle_f.zero_()
        self.cloth.dx.zero_()
        self.cloth.rebuild_bvh(self.state_0)
        self.grasp.reset(self._all_worlds)

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
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            getattr(self.state_0, name).assign(getattr(self.state_1, name))

    def _step_cloth(self, dt: float) -> None:
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.grasp.enforce(self.state_0, self.state_1, dt)
        self.collision.collide(self.state_0, self.contacts)
        self.cloth.step(self.state_0, self.state_1, None, self.contacts, dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

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
