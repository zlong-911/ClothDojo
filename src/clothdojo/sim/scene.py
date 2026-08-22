"""The minimal Piper shirt-flattening scene on Newton 1.5."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from clothdojo.assets import GarmentLibrary
from clothdojo.sim.cloth import add_style3d_garment

# OpenUSD 26.3 can crash while this asset is imported with its full Work/TBB
# concurrency. Set its supported startup control only before pxr is loaded;
# _limit_openusd_workers() handles host applications that loaded it already.
if "pxr.Work" not in sys.modules:
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

ARM_JOINTS = tuple(f"{side}/joint{joint}" for side in ("left", "right") for joint in range(1, 7))
GRIPPER_JOINTS = tuple(f"{side}/joint{joint}" for side in ("left", "right") for joint in (7, 8))
END_EFFECTOR_LINKS = ("left/link6", "right/link6")


@dataclass(frozen=True, slots=True)
class ShirtFlattenSpec:
    """Physics values that define the first migrated scene."""

    garment_id: str = "rand_ZZCWITH5HY"
    world_count: int = 1
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    table_position: tuple[float, float, float] = (0.651, 0.0, 0.775)
    table_half_extents: tuple[float, float, float] = (0.631, 0.7, 0.025)
    garment_position: tuple[float, float, float] = (0.65, 0.0, 1.162)
    garment_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    garment_scale: float = 1.0e-3
    cloth_density: float = 0.18
    particle_radius: float = 0.0025
    stretch_stiffness_uv: tuple[float, float, float] = (50.0, 50.0, 2.5)
    bend_stiffness_uv: tuple[float, float, float] = (1.25e-6, 6.25e-7, 3.125e-7)
    soft_contact_ke: float = 3800.0
    soft_contact_kd: float = 0.475
    soft_contact_mu: float = 0.6
    rigid_contact_ke: float = 10000.0
    rigid_contact_kd: float = 200.0
    rigid_contact_margin: float = 0.0004
    robot_friction: float = 0.6
    table_friction: float = 0.8
    control_fps: int = 50
    sim_substeps: int = 6
    cloth_contact_margin: float = 0.01
    style3d_iterations: int = 4
    style3d_linear_iterations: int = 10
    self_contact_radius: float = 0.002
    cloth_body_friction_epsilon: float = 0.003
    self_contact_vf_stiffness: float = 0.05
    self_contact_ee_stiffness: float = 0.002
    self_contact_ef_stiffness: float = 0.03
    rigid_contact_max: int = 4096

    def __post_init__(self) -> None:
        if self.world_count < 1:
            raise ValueError("world_count must be positive")
        if self.control_fps < 1 or self.sim_substeps < 1:
            raise ValueError("control_fps and sim_substeps must be positive")


@dataclass(frozen=True, slots=True)
class ShirtFlattenScene:
    model: object
    robot: object
    arms: object
    grippers: object
    end_effectors: object
    end_effector_body_indices: tuple[int, int]
    garment_particles_per_world: tuple[int, int]
    spec: ShirtFlattenSpec


def build_shirt_flatten(
    assets_root: str | Path,
    *,
    spec: ShirtFlattenSpec | None = None,
    device: str = "cuda:0",
) -> ShirtFlattenScene:
    """Build the physics-only scene used by simulation."""

    return _build_shirt_flatten(
        assets_root,
        spec=spec,
        device=device,
        load_visual_shapes=False,
    )


def build_shirt_flatten_render_scene(
    assets_root: str | Path,
    *,
    spec: ShirtFlattenSpec | None = None,
    device: str = "cuda:0",
) -> ShirtFlattenScene:
    """Build a separate visual model with topology matching the physics scene."""

    return _build_shirt_flatten(
        assets_root,
        spec=spec,
        device=device,
        load_visual_shapes=True,
    )


def _build_shirt_flatten(
    assets_root: str | Path,
    *,
    spec: ShirtFlattenSpec | None,
    device: str,
    load_visual_shapes: bool,
) -> ShirtFlattenScene:
    """Build the scene once, then replicate it into colocated independent worlds."""

    _limit_openusd_workers()
    import newton
    import warp as wp

    spec = spec or ShirtFlattenSpec()
    assets_root = Path(assets_root).expanduser().absolute()
    robot_path = assets_root / "robots" / "piper" / "piper.usda"
    if not robot_path.is_file():
        raise FileNotFoundError(f"missing Piper asset: {robot_path}")
    garment_path = GarmentLibrary(assets_root).converted(spec.garment_id)

    newton.use_coord_layout_targets = True
    template = newton.ModelBuilder(gravity=spec.gravity)
    _register_solver_attributes(newton, template)
    robot_body_count = _add_robot(
        newton,
        wp,
        template,
        robot_path,
        spec,
        load_visual_shapes=load_visual_shapes,
    )
    arm_joints, gripper_joints = _configure_robot(newton, template, spec, robot_body_count)
    end_effector_links = _body_indices(template, END_EFFECTOR_LINKS)
    _add_table(newton, wp, template, spec)
    particle_range = add_style3d_garment(template, garment_path, spec)

    builder = newton.ModelBuilder(gravity=spec.gravity)
    _register_solver_attributes(newton, builder)
    # Newton 1.5 recommends viewer offsets instead of physical world spacing.
    builder.replicate(template, spec.world_count)
    model = builder.finalize(device=device, requires_grad=False)
    model.soft_contact_ke = spec.soft_contact_ke
    model.soft_contact_kd = spec.soft_contact_kd
    model.soft_contact_mu = spec.soft_contact_mu

    robot = newton.selection.ArticulationView(model, "*")
    arms = newton.selection.ArticulationView(model, "*", include_joints=arm_joints)
    grippers = newton.selection.ArticulationView(model, "*", include_joints=gripper_joints)
    end_effectors = newton.selection.ArticulationView(model, "*", include_links=end_effector_links)
    if robot.count != spec.world_count:
        raise RuntimeError(
            "expected one Piper articulation per world, "
            f"got {robot.count} for {spec.world_count} worlds"
        )
    return ShirtFlattenScene(
        model,
        robot,
        arms,
        grippers,
        end_effectors,
        tuple(end_effector_links),
        particle_range,
        spec,
    )


def _limit_openusd_workers() -> None:
    """Verify that OpenUSD was limited before its Work runtime initialized."""

    from pxr import Work

    if Work.GetConcurrencyLimit() != 1:
        raise RuntimeError(
            "OpenUSD initialized before ClothDojo could set "
            "PXR_WORK_THREAD_LIMIT=1; set it before importing Newton"
        )


def _register_solver_attributes(newton, builder) -> None:
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)


def _add_robot(
    newton,
    wp,
    builder,
    path: Path,
    spec: ShirtFlattenSpec,
    *,
    load_visual_shapes: bool,
) -> int:
    body_start = int(builder.body_count)
    shape_start = int(builder.shape_count)
    builder.add_usd(
        source=str(path),
        floating=False,
        enable_self_collisions=False,
        ignore_paths=[".*/root_joint"],
        load_visual_shapes=load_visual_shapes,
        xform=wp.transform_identity(),
    )
    body_end = int(builder.body_count)
    collision_shapes = 0
    collision_bits = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    for shape in range(shape_start, int(builder.shape_count)):
        if not int(builder.shape_flags[shape]) & collision_bits:
            continue
        collision_shapes += 1
        builder.shape_flags[shape] |= particle_bit
        builder.shape_material_mu[shape] = spec.robot_friction
        builder.shape_material_ke[shape] = spec.rigid_contact_ke
        builder.shape_material_kd[shape] = spec.rigid_contact_kd
        # In 1.5 kf also drives MuJoCo's elliptic friction reference. Cloth
        # contact uses mu, so keep this unrelated gain disabled explicitly.
        builder.shape_material_kf[shape] = 0.0
        builder.shape_margin[shape] = spec.rigid_contact_margin
    if collision_shapes == 0 or body_end == body_start:
        raise RuntimeError("Piper USD did not provide rigid bodies and collision shapes")
    return body_end - body_start


def _configure_robot(newton, builder, spec: ShirtFlattenSpec, body_count: int):
    arm_joints = _joint_indices(builder, ARM_JOINTS)
    gripper_joints = _joint_indices(builder, GRIPPER_JOINTS)
    arm_dofs = _joint_dofs(builder, arm_joints)
    gripper_dofs = _joint_dofs(builder, gripper_joints)

    arm_ke = (31.0035, 45.6903, 28.8618, 17.8393, 18.7203, 17.8206) * 2
    arm_kd = (5.58063, 8.22425, 5.19513, 3.21108, 3.36966, 3.20771) * 2
    arm_effort = (32.0, 32.0, 18.0, 5.0, 5.0, 2.0) * 2
    for dof, ke, kd, effort in zip(arm_dofs, arm_ke, arm_kd, arm_effort, strict=True):
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        builder.joint_target_ke[dof] = ke
        builder.joint_target_kd[dof] = kd
        builder.joint_effort_limit[dof] = effort
        builder.joint_armature[dof] = 0.1
    for slot, dof in enumerate(gripper_dofs):
        builder.joint_limit_lower[dof] = (0.0, -0.04, 0.0, -0.04)[slot]
        builder.joint_limit_upper[dof] = (0.04, 0.0, 0.04, 0.0)[slot]
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        builder.joint_target_ke[dof] = 2500.0 if slot % 2 == 0 else 0.0
        builder.joint_target_kd[dof] = 80.0
        builder.joint_effort_limit[dof] = 30.0
        builder.joint_armature[dof] = 0.3

    gravcomp = builder.custom_attributes["mujoco:gravcomp"]
    gravcomp.values = dict.fromkeys(range(body_count), 1.0)
    actgravcomp = builder.custom_attributes["mujoco:jnt_actgravcomp"]
    actgravcomp.values = dict.fromkeys(arm_dofs, True)
    return arm_joints, gripper_joints


def _add_table(newton, wp, builder, spec: ShirtFlattenSpec) -> None:
    cfg = newton.ModelBuilder.ShapeConfig(
        mu=spec.table_friction,
        ke=spec.rigid_contact_ke,
        kd=spec.rigid_contact_kd,
        kf=0.0,
        margin=spec.rigid_contact_margin,
        has_particle_collision=True,
        has_shape_collision=True,
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*spec.table_position), wp.quat_identity()),
        hx=spec.table_half_extents[0],
        hy=spec.table_half_extents[1],
        hz=spec.table_half_extents[2],
        cfg=cfg,
        color=(0.85, 0.85, 0.85),
        label="table",
    )


def _joint_indices(builder, names: tuple[str, ...]) -> list[int]:
    result = []
    for name in names:
        matches = [
            index
            for index, label in enumerate(builder.joint_label)
            if str(label).endswith(f"/{name}")
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one Piper joint named {name!r}, found {len(matches)}")
        result.append(matches[0])
    return sorted(result)


def _body_indices(builder, names: tuple[str, ...]) -> list[int]:
    result = []
    for name in names:
        side, link = name.split("/", maxsplit=1)
        matches = [
            index
            for index, label in enumerate(builder.body_label)
            if f"/{side}/" in str(label) and str(label).endswith(f"/{link}")
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one Piper body named {name!r}, found {len(matches)}")
        result.append(matches[0])
    return sorted(result)


def _joint_dofs(builder, joint_indices: list[int]) -> list[int]:
    starts = []
    offset = 0
    selected = set(joint_indices)
    for joint, dimensions in enumerate(builder.joint_dof_dim):
        count = int(dimensions[0]) + int(dimensions[1])
        if joint in selected:
            if count != 1:
                raise ValueError(f"controlled Piper joint {joint} has {count} degrees of freedom")
            starts.append(offset)
        offset += count
    if len(starts) != len(joint_indices):
        raise ValueError("failed to resolve every controlled Piper joint")
    return starts
