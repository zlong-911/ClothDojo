"""The minimal Piper shirt-flattening scene on Newton 1.5."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path

# Physics-only processes default OpenUSD to one worker before any robot USD is loaded.
# Host applications such as Isaac own an already initialized task runtime and
# retain their application-level concurrency.
if "pxr.Work" not in sys.modules:
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import numpy as np

from clothdojo.assets import GarmentLibrary
from clothdojo.robots import (
    PIPER,
    RobotProfile,
    robot_profile,
)
from clothdojo.sim.cloth import add_style3d_garment, load_garment_mesh
from clothdojo.tasks import SHIRT_FLATTEN, task_profile

ARM_JOINTS = tuple(f"{side}/joint{joint}" for side in ("left", "right") for joint in range(1, 7))
GRIPPER_JOINTS = tuple(f"{side}/joint{joint}" for side in ("left", "right") for joint in (7, 8))
END_EFFECTOR_LINKS = ("left/link6", "right/link6")
WORLD_LAYOUT = "centered_grid"
WORLD_SPACING_M = 3.0


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
    garment_scale: float = 1.0e-2
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
    table_friction: float = 0.8
    control_fps: int = 50
    sim_substeps: int = 6
    cloth_contact_margin: float = 0.01
    style3d_iterations: int = 4
    style3d_linear_iterations: int = 10
    self_contact_radius: float = 0.002
    self_contact_max_candidates_per_primitive: int = 63
    self_contact_friction_mu: float = 0.35
    self_contact_friction_epsilon: float = 0.01
    suppress_self_friction_during_untangling: bool = True
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
        if self.self_contact_max_candidates_per_primitive < 1:
            raise ValueError("self_contact_max_candidates_per_primitive must be positive")
        if self.self_contact_friction_mu < 0.0:
            raise ValueError("self_contact_friction_mu must be non-negative")
        if self.self_contact_friction_epsilon <= 0.0:
            raise ValueError("self_contact_friction_epsilon must be positive")


@dataclass(frozen=True, slots=True)
class ShirtFlattenScene:
    model: object
    robot: object
    arms: object
    grippers: object
    end_effectors: object
    end_effector_body_indices: tuple[int, int]
    arm_joint_coord_indices: tuple[int, ...]
    garment_particles_per_world: tuple[int, int]
    spec: ShirtFlattenSpec
    world_offsets: tuple[tuple[float, float, float], ...]
    robot_profile: RobotProfile


def _build_shirt_flatten(
    assets_root: str | Path,
    *,
    spec: ShirtFlattenSpec | None = None,
    device: str = "cuda:0",
    visuals: bool = False,
    robot_cloth_collision: bool = True,
    robot_id: str = PIPER,
) -> ShirtFlattenScene:
    """Build one simulation model, optionally including render-only visuals.

    Visual shapes use Newton 1.5 collision flags and therefore remain in the
    same model without participating in rigid or particle contact.  Callers
    that never render can leave ``visuals`` disabled to avoid loading them.
    Template generation can disable ``robot_cloth_collision`` while retaining
    the table-particle and rigid-shape collision paths.
    """

    import newton
    import warp as wp

    spec = spec or ShirtFlattenSpec()
    assets_root = Path(assets_root).expanduser().absolute()
    profile = robot_profile(robot_id)
    robot_path = assets_root / profile.newton_asset
    if not robot_path.is_file():
        raise FileNotFoundError(f"missing {profile.robot_id} asset: {robot_path}")
    robot_visual_path = assets_root / profile.newton_visual_asset
    if visuals and not robot_visual_path.is_file():
        raise FileNotFoundError(
            f"missing {profile.robot_id} visual asset: {robot_visual_path}"
        )
    installation_path = (
        assets_root / profile.installation_asset
        if profile.installation_asset is not None
        else None
    )
    if visuals and installation_path is not None and not installation_path.is_file():
        raise FileNotFoundError(
            f"missing {profile.robot_id} installation asset: {installation_path}"
        )
    garment_path = GarmentLibrary(assets_root).converted(spec.garment_id)

    newton.use_coord_layout_targets = True
    template = newton.ModelBuilder(gravity=spec.gravity)
    _register_solver_attributes(newton, template)
    if profile.robot_id == "piper":
        _robot_body_count = _add_robot(
            newton,
            wp,
            template,
            robot_path,
            load_visual_shapes=visuals,
            robot_cloth_collision=robot_cloth_collision,
        )
        arm_joints, gripper_joints, arm_coords = _select_piper_robot(template)
        end_effector_links = _body_indices(template, END_EFFECTOR_LINKS)
    else:
        (
            _robot_body_count,
            arm_joints,
            gripper_joints,
            arm_coords,
            end_effector_links,
        ) = _add_profile_robot(
            newton,
            wp,
            template,
            robot_path,
            profile,
            load_visual_shapes=visuals,
        )
    if visuals and installation_path is not None:
        _add_installation_visual(newton, wp, template, installation_path, profile)
    _add_table(newton, wp, template, spec)
    garment_mesh = load_garment_mesh(garment_path)
    particle_range = add_style3d_garment(template, garment_path, spec, mesh=garment_mesh)
    builder = newton.ModelBuilder(gravity=spec.gravity)
    _register_solver_attributes(newton, builder)
    world_offsets = _centered_world_offsets(spec.world_count)
    world_xforms = tuple(
        wp.transform(offset, wp.quat_identity()) for offset in world_offsets
    )
    if visuals:
        # The NOCS proxy deforms independently in each world. Build one public
        # ModelBuilder world per proxy so Newton owns distinct Mesh buffers;
        # the physics path below still takes 1.5's optimized replicate path.
        for world_index in range(spec.world_count):
            world = newton.ModelBuilder(gravity=spec.gravity)
            _register_solver_attributes(newton, world)
            world.add_builder(template)
            _add_smooth_garment_visual(
                newton,
                wp,
                world,
                particle_range,
                garment_mesh[0],
                mesh_identity=world_index,
            )
            builder.add_world(world, xform=world_xforms[world_index])
    else:
        # SolverStyle3D 1.5 owns an ungrouped self-collision BVH. Physically
        # separate worlds so its broad phase cannot create cross-world cloth
        # contacts; project boundaries convert back to world-local coordinates.
        builder.replicate(template, spec.world_count, xforms=world_xforms)
    model = builder.finalize(device=device, requires_grad=False)
    model.soft_contact_ke = spec.soft_contact_ke
    model.soft_contact_kd = spec.soft_contact_kd
    model.soft_contact_mu = spec.soft_contact_mu

    robot = newton.selection.ArticulationView(model, "*")
    arms = newton.selection.ArticulationView(model, "*", include_joints=arm_joints)
    grippers = newton.selection.ArticulationView(model, "*", include_joints=gripper_joints)
    end_effector_view_links = (
        end_effector_links if profile.robot_id == "piper" else [end_effector_links[0]]
    )
    end_effectors = newton.selection.ArticulationView(
        model, "*", include_links=end_effector_view_links
    )
    articulations_per_world = profile.body_parents.count(-1)
    expected_articulations = spec.world_count * articulations_per_world
    if robot.count != expected_articulations:
        raise RuntimeError(
            f"expected {expected_articulations} {profile.robot_id} articulations, "
            f"got {robot.count}"
        )
    return ShirtFlattenScene(
        model=model,
        robot=robot,
        arms=arms,
        grippers=grippers,
        end_effectors=end_effectors,
        end_effector_body_indices=tuple(end_effector_links),
        arm_joint_coord_indices=tuple(arm_coords),
        garment_particles_per_world=particle_range,
        spec=spec,
        world_offsets=world_offsets,
        robot_profile=profile,
    )


def build_scene(
    assets_root: str | Path,
    *,
    robot_id: str = PIPER,
    task_name: str = SHIRT_FLATTEN,
    spec: ShirtFlattenSpec | None = None,
    device: str = "cuda:0",
    visuals: bool = False,
    robot_cloth_collision: bool = True,
) -> ShirtFlattenScene:
    """Build a scene from independent robot and task identities."""

    task = task_profile(task_name)
    if not task.scene_available:
        raise NotImplementedError(
            f"{task_name} task assets and randomization are not defined yet"
        )
    if task_name != SHIRT_FLATTEN:
        raise ValueError(f"no scene builder is registered for task {task_name!r}")
    return _build_shirt_flatten(
        assets_root,
        robot_id=robot_id,
        spec=spec,
        device=device,
        visuals=visuals,
        robot_cloth_collision=robot_cloth_collision,
    )


def _add_profile_robot(
    newton,
    wp,
    builder,
    kinematic_path: Path,
    profile: RobotProfile,
    *,
    load_visual_shapes: bool,
) -> tuple[int, list[int], list[int], list[int], list[int]]:
    """Add one bimanual MJCF or a symmetric pair of single-arm MJCFs."""

    body_start = int(builder.body_count)
    if len(profile.installations) != 2:
        raise RuntimeError("target robot profile must define two installations")
    q_start = int(builder.joint_coord_count)
    dof_start = int(builder.joint_dof_count)
    frames = (
        (profile.asset_frame,)
        if profile.asset_instances == 1
        else profile.installations
    )
    for frame in frames:
        builder.add_mjcf(
            str(kinematic_path),
            xform=wp.transform(frame.position, frame.rotation_xyzw),
            floating=False,
            parse_meshes=load_visual_shapes,
            parse_visuals=load_visual_shapes,
            hide_visuals=not load_visual_shapes,
            enable_self_collisions=False,
            override_root_xform=False,
        )
    body_count = int(builder.body_count) - body_start
    q_count = int(builder.joint_coord_count) - q_start
    dof_count = int(builder.joint_dof_count) - dof_start
    if (
        body_count != len(profile.body_names)
        or q_count != len(profile.home_joint_q)
        or dof_count != profile.joint_dof_count
    ):
        raise RuntimeError(
            f"pinned {profile.robot_id} MJCF topology differs from its robot profile"
        )
    for index, label in enumerate(profile.body_names, start=body_start):
        builder.body_label[index] = label
    builder.joint_q[q_start : q_start + q_count] = list(profile.home_joint_q)
    _drop_replaced_mujoco_actuators(builder, profile)
    _apply_position_drive_override(newton, builder, profile, dof_start=dof_start)
    per_side = profile.arm_dofs_per_side
    arm_joints = _joint_indices_for_coordinates(
        newton,
        builder,
        [q_start + value for value in profile.arm_coord_indices[:per_side]],
    )
    gripper_joints = _joint_indices_for_coordinates(
        newton,
        builder,
        [q_start + value for value in profile.gripper_coord_indices[:2]],
    )
    end_effectors = _body_indices(
        builder, [frame.parent_body for frame in profile.attach_frames]
    )
    return (
        body_count,
        arm_joints,
        gripper_joints,
        [q_start + value for value in profile.arm_coord_indices],
        end_effectors,
    )


def _joint_indices_for_coordinates(newton, builder, coordinates: list[int]) -> list[int]:
    result = []
    for coordinate in coordinates:
        matches = [
            index
            for index, (start, joint_type) in enumerate(
                zip(builder.joint_q_start, builder.joint_type, strict=True)
            )
            if int(start) == coordinate and joint_type != newton.JointType.FIXED
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"robot coordinate {coordinate} matched {len(matches)} movable joints"
            )
        result.append(matches[0])
    return result


def _apply_position_drive_override(
    newton, builder, profile: RobotProfile, *, dof_start: int
) -> None:
    drive = profile.position_drive_override
    if drive is None:
        return
    arm_values = (
        drive.arm_stiffness * 2,
        drive.arm_damping * 2,
        drive.arm_effort_limit * 2,
    )
    gripper_values = (
        drive.gripper_stiffness * 2,
        drive.gripper_damping * 2,
        drive.gripper_effort_limit * 2,
    )
    for dofs, (stiffness, damping, effort) in (
        (profile.arm_dof_indices, arm_values),
        (profile.gripper_dof_indices, gripper_values),
    ):
        for dof, ke, kd, limit in zip(
            dofs, stiffness, damping, effort, strict=True
        ):
            index = dof_start + dof
            builder.joint_target_mode[index] = newton.JointTargetMode.POSITION
            builder.joint_target_ke[index] = ke
            builder.joint_target_kd[index] = kd
            builder.joint_effort_limit[index] = limit


def _drop_replaced_mujoco_actuators(builder, profile: RobotProfile) -> None:
    """Remove upstream actuators replaced by Newton scalar position drives."""

    if profile.position_drive_override is None:
        return
    frequency = "mujoco:actuator"
    counts = getattr(builder, "_custom_frequency_counts", None)
    attributes = getattr(builder, "custom_attributes", None)
    if not isinstance(counts, dict) or not isinstance(attributes, dict):
        raise RuntimeError("Newton 1.5 MuJoCo actuator metadata is unavailable")
    counts[frequency] = 0
    for attribute in attributes.values():
        if attribute.frequency == frequency:
            attribute.values.clear()


def _add_installation_visual(
    newton,
    wp,
    builder,
    path: Path,
    profile: RobotProfile,
) -> None:
    """Add the robot-owned static mounting structure without contact."""

    body_count = int(builder.body_count)
    joint_count = int(builder.joint_count)
    shape_start = int(builder.shape_count)
    frame = profile.asset_frame
    builder.add_mjcf(
        str(path),
        xform=wp.transform(frame.position, frame.rotation_xyzw),
        floating=False,
        parse_meshes=True,
        parse_visuals=True,
        enable_self_collisions=False,
    )
    if int(builder.body_count) != body_count or int(builder.joint_count) != joint_count:
        raise RuntimeError("installation visual must not add articulated bodies")
    if int(builder.shape_count) - shape_start != 31:
        raise RuntimeError("pinned ALOHA installation visual must contain 31 meshes")
    collision_bits = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(
        newton.ShapeFlags.COLLIDE_PARTICLES
    )
    for shape in range(shape_start, int(builder.shape_count)):
        builder.shape_flags[shape] &= ~collision_bits


def _centered_world_offsets(
    world_count: int,
) -> tuple[tuple[float, float, float], ...]:
    """Place physics worlds on a compact centered grid for Style3D isolation."""

    if world_count < 1:
        raise ValueError("world_count must be positive")
    columns = ceil(sqrt(world_count))
    rows = ceil(world_count / columns)
    offsets = []
    for world in range(world_count):
        row, column = divmod(world, columns)
        offsets.append(
            (
                (column - 0.5 * (columns - 1)) * WORLD_SPACING_M,
                (row - 0.5 * (rows - 1)) * WORLD_SPACING_M,
                0.0,
            )
        )
    return tuple(offsets)


def _register_solver_attributes(newton, builder) -> None:
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)


def _add_robot(
    newton,
    wp,
    builder,
    path: Path,
    *,
    load_visual_shapes: bool,
    robot_cloth_collision: bool = True,
) -> int:
    body_start = int(builder.body_count)
    shape_start = int(builder.shape_count)
    collision_roles, contact, initial_joint_q = _robot_asset_contract(path)
    imported = builder.add_usd(
        source=str(path),
        floating=False,
        # The baked expert geometry owns its self-collision proxy filters below.
        enable_self_collisions=True,
        ignore_paths=[".*/root_joint"],
        load_visual_shapes=load_visual_shapes,
        xform=wp.transform_identity(),
    )
    body_end = int(builder.body_count)
    path_shape_map = imported["path_shape_map"]
    path_joint_map = imported["path_joint_map"]
    missing_joints = sorted(set(initial_joint_q).difference(path_joint_map))
    if missing_joints:
        raise RuntimeError(
            f"robot USD did not import {len(missing_joints)} authored joint states"
        )
    for joint_path, position in initial_joint_q.items():
        joint = int(path_joint_map[joint_path])
        builder.joint_q[int(builder.joint_q_start[joint])] = position
    missing = sorted(set(collision_roles).difference(path_shape_map))
    if missing:
        raise RuntimeError(
            f"robot USD did not import {len(missing)} authored collision-role shapes"
        )
    contact_shapes = tuple(
        int(path_shape_map[path])
        for path, role in collision_roles.items()
        if role == "contact"
    )
    proxy_shapes = tuple(
        int(path_shape_map[path])
        for path, role in collision_roles.items()
        if role == "self_collision_proxy"
    )
    collision_shapes = _configure_robot_collision_geometry(
        newton,
        builder,
        body_start=body_start,
        body_end=body_end,
        shape_start=shape_start,
        contact_shapes=contact_shapes,
        proxy_shapes=proxy_shapes,
        robot_cloth_collision=robot_cloth_collision,
    )
    for shape in collision_shapes:
        builder.shape_material_mu[shape] = contact.mu
        builder.shape_material_ke[shape] = contact.ke
        builder.shape_material_kd[shape] = contact.kd
        builder.shape_material_kf[shape] = contact.kf
        builder.shape_margin[shape] = contact.margin
    if not collision_shapes or body_end == body_start:
        raise RuntimeError("Piper USD did not provide rigid bodies and collision shapes")
    return body_end - body_start


@dataclass(frozen=True, slots=True)
class _RobotContactProperties:
    mu: float
    ke: float
    kd: float
    kf: float
    margin: float


def _robot_asset_contract(
    path: Path,
) -> tuple[dict[str, str], _RobotContactProperties, dict[str, float]]:
    """Read project collision, contact and reset semantics from the robot USD."""

    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None or not stage.GetDefaultPrim():
        raise RuntimeError(f"could not open robot USD: {path}")
    root = stage.GetDefaultPrim()
    role_by_path = {}
    for prim in stage.Traverse():
        role_attr = prim.GetAttribute("clothdojo:collisionRole")
        if role_attr and role_attr.HasAuthoredValue():
            role = str(role_attr.Get())
            if role not in {"contact", "self_collision_proxy"}:
                raise RuntimeError(f"unsupported robot collision role {role!r}")
            role_by_path[str(prim.GetPath())] = role
    if not role_by_path or set(role_by_path.values()) != {
        "contact",
        "self_collision_proxy",
    }:
        raise RuntimeError("robot USD must author contact and self-collision proxy roles")

    initial_joint_q = {}
    for prim in stage.Traverse():
        linear = prim.GetAttribute("state:linear:physics:position")
        angular = prim.GetAttribute("state:angular:physics:position")
        linear_value = linear.Get() if linear and linear.HasAuthoredValue() else None
        angular_value = angular.Get() if angular and angular.HasAuthoredValue() else None
        if linear_value is not None and angular_value is not None:
            raise RuntimeError(f"robot joint {prim.GetPath()} has two scalar position states")
        if linear_value is not None:
            initial_joint_q[str(prim.GetPath())] = float(linear_value)
        elif angular_value is not None:
            initial_joint_q[str(prim.GetPath())] = float(np.deg2rad(angular_value))

    material_targets = root.GetRelationship("clothdojo:physicsMaterial").GetTargets()
    if len(material_targets) != 1:
        raise RuntimeError("robot USD must identify exactly one physics material")
    material = stage.GetPrimAtPath(material_targets[0])
    if not material:
        raise RuntimeError("robot USD physics material target does not exist")

    def required(prim, name: str) -> float:
        attribute = prim.GetAttribute(name)
        value = attribute.Get() if attribute else None
        if value is None:
            raise RuntimeError(f"robot USD is missing {name}")
        return float(value)

    return (
        role_by_path,
        _RobotContactProperties(
            mu=required(material, "physics:dynamicFriction"),
            ke=required(material, "newton:contactStiffness"),
            kd=required(material, "newton:contactDamping"),
            kf=required(material, "newton:contactFrictionGain"),
            margin=required(root, "clothdojo:contactMargin"),
        ),
        initial_joint_q,
    )


def _configure_robot_collision_geometry(
    newton,
    builder,
    *,
    body_start: int,
    body_end: int,
    shape_start: int,
    contact_shapes: tuple[int, ...],
    proxy_shapes: tuple[int, ...],
    robot_cloth_collision: bool,
) -> tuple[int, ...]:
    """Apply authored collision roles and reproduce topology-derived proxy filters."""

    shape_end = int(builder.shape_count)
    if not contact_shapes or not proxy_shapes:
        raise RuntimeError("robot USD collision-role sets must both be non-empty")
    if any(shape < shape_start or shape >= shape_end for shape in contact_shapes + proxy_shapes):
        raise RuntimeError("robot USD collision role resolved outside its imported shape range")

    shape_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    collision_bits = shape_bit | particle_bit
    migrated = set(contact_shapes) | set(proxy_shapes)
    for shape in range(shape_start, shape_end):
        if shape not in migrated:
            # The expert runtime disabled authored/visual colliders before adding
            # its baked contact envelope. Preserve visibility but remove contact.
            builder.shape_flags[shape] &= ~collision_bits
            builder.shape_collision_group[shape] = 0
    for shape in contact_shapes:
        contact_bits = shape_bit | (particle_bit if robot_cloth_collision else 0)
        builder.shape_flags[shape] = (
            int(builder.shape_flags[shape]) & ~collision_bits
        ) | contact_bits
        builder.shape_collision_group[shape] = 1
    for shape in proxy_shapes:
        builder.shape_flags[shape] = (
            int(builder.shape_flags[shape]) & ~collision_bits
        ) | shape_bit
        builder.shape_collision_group[shape] = 2

    # Contact geometry collides with cloth and external rigid bodies, never
    # with another part of the robot. The lighter proxy envelope owns only
    # articulation self-collision, matching the data-collection scene.
    for slot, left_shape in enumerate(contact_shapes):
        for right_shape in contact_shapes[slot + 1 :]:
            builder.add_shape_collision_filter_pair(left_shape, right_shape)

    robot_bodies = set(range(body_start, body_end))
    ignored_body_pairs = {(body, body) for body in robot_bodies}
    for parent, child in zip(builder.joint_parent, builder.joint_child, strict=True):
        parent = int(parent)
        child = int(child)
        if parent in robot_bodies and child in robot_bodies:
            ignored_body_pairs.add((min(parent, child), max(parent, child)))

    fixed_type = int(newton.JointType.FIXED)
    fixed_parent = {body: body for body in robot_bodies}

    def find(body: int) -> int:
        while fixed_parent[body] != body:
            fixed_parent[body] = fixed_parent[fixed_parent[body]]
            body = fixed_parent[body]
        return body

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            fixed_parent[right_root] = left_root

    for joint_type, parent, child in zip(
        builder.joint_type, builder.joint_parent, builder.joint_child, strict=True
    ):
        parent = int(parent)
        child = int(child)
        if int(joint_type) == fixed_type and parent in robot_bodies and child in robot_bodies:
            union(parent, child)
    fixed_components: dict[int, list[int]] = {}
    for body in robot_bodies:
        fixed_components.setdefault(find(body), []).append(body)
    for component in fixed_components.values():
        for slot, left_body in enumerate(component):
            for right_body in component[slot:]:
                ignored_body_pairs.add((min(left_body, right_body), max(left_body, right_body)))

    children_by_parent: dict[int, set[int]] = {}
    for parent, child in zip(builder.joint_parent, builder.joint_child, strict=True):
        parent = int(parent)
        child = int(child)
        if parent in robot_bodies and child in robot_bodies:
            children_by_parent.setdefault(parent, set()).add(child)
    for root in _body_indices(builder, END_EFFECTOR_LINKS):
        subtree = {root}
        frontier = [root]
        while frontier:
            parent = frontier.pop()
            for child in children_by_parent.get(parent, ()):
                if child not in subtree:
                    subtree.add(child)
                    frontier.append(child)
        ordered = sorted(subtree)
        for slot, left_body in enumerate(ordered):
            for right_body in ordered[slot:]:
                ignored_body_pairs.add((min(left_body, right_body), max(left_body, right_body)))

    for slot, left_shape in enumerate(proxy_shapes):
        left_body = int(builder.shape_body[left_shape])
        for right_shape in proxy_shapes[slot + 1 :]:
            right_body = int(builder.shape_body[right_shape])
            if (min(left_body, right_body), max(left_body, right_body)) in ignored_body_pairs:
                builder.add_shape_collision_filter_pair(left_shape, right_shape)

    return contact_shapes + proxy_shapes


def _select_piper_robot(builder):
    """Select Piper control groups without changing asset-authored physics."""

    arm_joints = _joint_indices(builder, ARM_JOINTS)
    gripper_joints = _joint_indices(builder, GRIPPER_JOINTS)
    arm_dofs = _joint_dofs(builder, arm_joints)
    return arm_joints, gripper_joints, arm_dofs


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


def _add_smooth_garment_visual(
    newton,
    wp,
    builder,
    particle_range: tuple[int, int],
    rest_vertices: np.ndarray,
    *,
    mesh_identity: int = 0,
) -> None:
    """Add a smooth render-only mesh with the expert-data NOCS appearance."""

    start, end = particle_range
    vertices = np.asarray(builder.particle_q[start:end], dtype=np.float32)
    indices = np.asarray(builder.tri_indices, dtype=np.int32).reshape(-1, 3)
    if indices.size == 0 or np.any(indices < start) or np.any(indices >= end):
        raise RuntimeError("garment triangles do not match the garment particle range")
    local_indices = np.ascontiguousarray(indices - start)
    if np.asarray(rest_vertices).shape != vertices.shape:
        raise RuntimeError("garment rest and simulated vertex counts differ")
    visual_vertices, visual_indices, normals, uvs, texture = _nocs_visual_mesh(
        vertices,
        local_indices,
        rest_vertices,
    )
    if mesh_identity:
        # Model finalization deduplicates byte-identical meshes. Each NOCS
        # proxy needs its own Warp BVH because its vertices deform per world;
        # one temporary ULP marker gives it a distinct public Mesh identity
        # and is overwritten from particle state before the first render.
        for _ in range(mesh_identity):
            visual_vertices[0, 0] = np.nextafter(visual_vertices[0, 0], np.float32(np.inf))
    mesh = newton.Mesh(
        vertices=visual_vertices,
        indices=visual_indices,
        normals=normals,
        uvs=uvs,
        texture=texture,
        compute_inertia=False,
        is_solid=False,
    )
    builder.add_shape_mesh(
        body=-1,
        xform=wp.transform_identity(),
        mesh=mesh,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_particle_collision=False,
            has_shape_collision=False,
        ),
        color=(1.0, 1.0, 1.0),
        label="garment_visual",
    )


def rest_vertices_to_nocs(rest_vertices: np.ndarray) -> np.ndarray:
    """Encode canonical garment coordinates in display/sRGB NOCS.

    Black remains reserved for the background, matching the expert-data
    renderer used to collect the SFT demonstrations.
    """

    vertices = np.asarray(rest_vertices, dtype=np.float32).reshape((-1, 3))
    if len(vertices) == 0:
        raise ValueError("garment rest mesh has no vertices")
    minimum = vertices.min(axis=0)
    span = vertices.max(axis=0) - minimum
    span[span < 1.0e-9] = 1.0
    return ((1.0 + np.rint((vertices - minimum) / span * 254.0)) / 255.0).astype(np.float32)


def _nocs_visual_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    rest_vertices: np.ndarray,
    *,
    tile_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bake per-vertex NOCS into a small public-API texture atlas.

    Newton 1.5 exposes per-mesh textures and UVs but not vertex colors. Each
    dense cloth triangle therefore owns one tiny atlas tile. Render vertices
    are face-varying while their order still matches the flattened physics
    triangle indices, allowing the camera adapter to update them on the GPU.
    """

    points = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    rest = np.asarray(rest_vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if rest.shape != points.shape:
        raise ValueError("rest_vertices must match vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("triangles must have non-empty shape (M, 3)")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("triangle index is outside the vertex array")
    if tile_size < 6:
        raise ValueError("NOCS atlas tiles must be at least 6 pixels")

    source_indices = np.ascontiguousarray(faces.reshape(-1))
    visual_vertices = np.ascontiguousarray(points[source_indices])
    visual_indices = np.arange(len(source_indices), dtype=np.int32)
    visual_normals = np.ascontiguousarray(_vertex_normals(points, faces)[source_indices])

    face_count = len(faces)
    columns = int(np.ceil(np.sqrt(face_count)))
    rows = int(np.ceil(face_count / columns))
    width = columns * tile_size
    height = rows * tile_size
    texture = np.zeros((height, width, 4), dtype=np.uint8)
    texture[..., 3] = 255
    corner_colors = np.rint(rest_vertices_to_nocs(rest)[source_indices] * 255.0).reshape(
        face_count, 3, 3
    )
    uvs = np.empty((face_count, 3, 2), dtype=np.float32)
    low = 1
    high = tile_size - 2
    pixel_axis = np.arange(tile_size, dtype=np.float32)
    s = (pixel_axis[None, :] - low) / (high - low)
    t = (pixel_axis[:, None] - low) / (high - low)
    for face in range(face_count):
        row, column = divmod(face, columns)
        y0 = row * tile_size
        x0 = column * tile_size
        colors = corner_colors[face]
        tile = (
            colors[0][None, None, :]
            + s[..., None] * (colors[1] - colors[0])[None, None, :]
            + t[..., None] * (colors[2] - colors[0])[None, None, :]
        )
        texture[y0 : y0 + tile_size, x0 : x0 + tile_size, :3] = np.clip(
            np.rint(tile), 0.0, 255.0
        ).astype(np.uint8)
        pixel_x = np.asarray([x0 + low, x0 + high, x0 + low], dtype=np.float32)
        pixel_y = np.asarray([y0 + low, y0 + low, y0 + high], dtype=np.float32)
        uvs[face, :, 0] = (pixel_x + 0.5) / width
        # Newton's mesh sampler flips authored V before indexing image rows.
        uvs[face, :, 1] = 1.0 - (pixel_y + 0.5) / height

    return (
        visual_vertices,
        visual_indices,
        visual_normals,
        np.ascontiguousarray(uvs.reshape(-1, 2)),
        texture,
    )


def _vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return area-weighted per-vertex normals for a triangle mesh."""

    points = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(indices, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("indices must have shape (M, 3)")
    if triangles.size and (np.any(triangles < 0) or np.any(triangles >= len(points))):
        raise ValueError("triangle index is outside the vertex array")
    normals = np.zeros_like(points)
    face_normals = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > 1.0e-12
    normals[nonzero] /= lengths[nonzero, None]
    return np.ascontiguousarray(normals, dtype=np.float32)


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
            if str(label).strip("/") == name
            or (f"/{side}/" in str(label) and str(label).endswith(f"/{link}"))
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one robot body named {name!r}, found {len(matches)}")
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
