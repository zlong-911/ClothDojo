"""Reachable front/back augmentation for canonical dual-Piper trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from clothdojo.data.mirror import (
    MIRROR_SUFFIX,
    PIPER_ARM_SIGNS,
    mirror_action_arrays,
    mirror_robot_state_arrays,
)
from clothdojo.data.retarget import (
    _retarget_motion,
    _robot_velocity,
    _RobotKinematics,
    piper_attach_poses,
)
from clothdojo.data.retarget_dynamics import RobotDynamicsRetargeter
from clothdojo.robots import PIPER, robot_profile
from clothdojo.tasks import SHIRT_FOLD
from clothdojo.trajectory import TrajectoryManifest, TrajectoryReader
from clothdojo.trajectory.reader import _DERIVED_OVERRIDE_FIELDS

FRONT_BACK_COLLECTION_SCHEMA = "clothdojo_front_back_augmentation_collection"
FRONT_BACK_TRAJECTORY_SCHEMA = "clothdojo_front_back_augmentation"
FRONT_BACK_FORMAT_VERSION = 1
FRONT_BACK_SUFFIX = "__front_back"

_REFLECT_X = np.diag((-1.0, 1.0, 1.0)).astype(np.float32)
_REFLECT_Y = np.diag((1.0, -1.0, 1.0)).astype(np.float32)
_DEFAULT_POSITION_TOLERANCE_M = 0.005
_DEFAULT_TABLE_MARGIN_M = 0.01
_DEFAULT_PLACEMENT_CANDIDATES = 64
_DEFAULT_MAX_PLACEMENT_SHIFT_X_M = 0.15
_DEFAULT_MAX_PLACEMENT_SHIFT_Y_M = 0.05
_DEFAULT_MIN_BASE_CLEARANCE_M = 0.10
_DEFAULT_MIN_INTER_ARM_CLEARANCE_M = 0.03
_COLLISION_PENETRATION_TOLERANCE_M = 5.0e-4
_FREE_MOTION_TABLE_CLEARANCE_M = 0.08
_FREE_MOTION_LOCAL_PLANAR_ADJUSTMENT_M = 0.05
_FREE_MOTION_MAX_PLANAR_ADJUSTMENT_M = 0.25
_FREE_MOTION_MAX_YIELD_PLANAR_ADJUSTMENT_M = 0.25
_FREE_MOTION_MAX_EXTRA_LIFT_M = 0.08
_FREE_MOTION_STANDBY_TRIGGER_DISTANCE_M = 0.20
_FREE_MOTION_TERMINAL_RETREAT_MAX_FRAMES = 60
_CLOTH_RELATIVE_MAX_POSITION_CORRECTION_M = 0.005
_CLOTH_RELATIVE_SMOOTHING_FRAMES = 21
_LOCAL_REFINEMENT_TRIGGER_PENETRATION_M = 2.0e-3
_LOCAL_REFINEMENT_TRIGGER_POSITION_ERROR_M = 0.01
_LOCAL_REFINEMENT_STEPS_M = (0.005,)
_LOCAL_REFINEMENT_MAX_JOINT_STEP_RATIO = 1.10
_LOCAL_REFINEMENT_MAX_MOTION_RATIO = 1.15
_IK_BRANCH_WRIST_OFFSETS_RAD = (1.5, -1.5)
_DYNAMIC_CANDIDATE_ATTEMPTS = 3
_MOTION_ENVELOPE_RELATIVE_TOLERANCE = 0.06
_JOINT_SMOOTHING_PASSES = (1, 2, 4, 8)
_JOINT_SMOOTHING_KERNEL = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float32) / 16.0


@dataclass(frozen=True, slots=True)
class SpatialTransform:
    linear: np.ndarray
    translation: np.ndarray

    def points(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) @ self.linear.T + self.translation).astype(
            np.float32
        )

    def vectors(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) @ self.linear.T).astype(np.float32)

    def to_json_dict(self) -> dict:
        return {
            "linear": self.linear.astype(float).tolist(),
            "translation": self.translation.astype(float).tolist(),
        }


def augment_front_back(
    source: str | Path,
    output: str | Path,
    *,
    seed: int,
    assets: str | Path = "assets",
    device: str = "cuda:0",
    placement_candidates: int = _DEFAULT_PLACEMENT_CANDIDATES,
    table_margin_m: float = _DEFAULT_TABLE_MARGIN_M,
    position_tolerance_m: float = _DEFAULT_POSITION_TOLERANCE_M,
    max_placement_shift_x_m: float = _DEFAULT_MAX_PLACEMENT_SHIFT_X_M,
    max_placement_shift_y_m: float = _DEFAULT_MAX_PLACEMENT_SHIFT_Y_M,
    min_base_clearance_m: float = _DEFAULT_MIN_BASE_CLEARANCE_M,
    min_inter_arm_clearance_m: float = _DEFAULT_MIN_INTER_ARM_CLEARANCE_M,
) -> dict:
    """Publish original/front-back/left-right/composed Piper overlays."""

    source_root = Path(source).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    rejection_report = output_root.with_name(f"{output_root.name}.rejections.json")
    assets_root = Path(assets).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"front/back augmentation output already exists: {output_root}")
    if rejection_report.exists():
        raise FileExistsError(f"front/back rejection report already exists: {rejection_report}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("front/back output must be outside the source collection")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("front/back seed must be an integer")
    if (
        isinstance(placement_candidates, bool)
        or not isinstance(placement_candidates, int)
        or placement_candidates < 1
    ):
        raise ValueError("placement_candidates must be a positive integer")
    for name, value in (
        ("table_margin_m", table_margin_m),
        ("position_tolerance_m", position_tolerance_m),
        ("min_base_clearance_m", min_base_clearance_m),
        ("min_inter_arm_clearance_m", min_inter_arm_clearance_m),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (
        ("max_placement_shift_x_m", max_placement_shift_x_m),
        ("max_placement_shift_y_m", max_placement_shift_y_m),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be nonnegative and finite")

    source_trajectories = _source_trajectories(source_root)
    collisions = sorted(
        path.name
        for path in source_trajectories
        if path.name.endswith((FRONT_BACK_SUFFIX, MIRROR_SUFFIX))
    )
    if collisions:
        raise ValueError(f"source already contains augmentation trajectory ids: {collisions}")

    profile = robot_profile(PIPER, assets_root=assets_root)
    kinematics = _RobotKinematics(
        profile,
        assets_root / profile.asset,
        device=device,
    )
    collision = _RigidCollisionValidator(assets_root, device=device)
    dynamics_by_timing: dict[tuple[float, int], RobotDynamicsRetargeter] = {}
    staging = output_root.with_name(f".{output_root.name}.tmp-{uuid4().hex}")
    trajectories_root = staging / "trajectories"
    annotations_root = trajectories_root / "annotations"
    records: list[dict] = []
    fallback_pairs: list[dict] = []
    rejections: list[dict] = []
    try:
        annotations_root.mkdir(parents=True)
        for index, source_trajectory in enumerate(source_trajectories, start=1):
            reader = TrajectoryReader(source_trajectory)
            _validate_source(reader, source_trajectory)
            timing = (
                float(reader.manifest.control_fps),
                int(reader.manifest.sim_substeps),
            )
            dynamics = dynamics_by_timing.get(timing)
            if dynamics is None:
                dynamics = RobotDynamicsRetargeter(
                    assets_root,
                    garment_id=str(reader.manifest.metadata["asset_id"]),
                    control_fps=timing[0],
                    sim_substeps=timing[1],
                    device=device,
                    target_robot=PIPER,
                )
                dynamics_by_timing[timing] = dynamics
            try:
                record = _augment_trajectory(
                    source_trajectory,
                    trajectories_root,
                    seed=seed,
                    placement_candidates=placement_candidates,
                    table_margin_m=table_margin_m,
                    position_tolerance_m=position_tolerance_m,
                    max_placement_shift_x_m=max_placement_shift_x_m,
                    max_placement_shift_y_m=max_placement_shift_y_m,
                    min_base_clearance_m=min_base_clearance_m,
                    min_inter_arm_clearance_m=min_inter_arm_clearance_m,
                    kinematics=kinematics,
                    dynamics=dynamics,
                    collision=collision,
                )
            except _PlacementRejected as exc:
                rejections.append(
                    {
                        "source_id": source_trajectory.name,
                        "source_manifest_sha256": _sha256(source_trajectory / "manifest.json"),
                        **exc.diagnostic,
                    }
                )
                fallback_pairs.append(
                    _write_identity_and_bilateral_pair(source_trajectory, trajectories_root)
                )
                _copy_annotations(
                    source_root,
                    annotations_root,
                    source_trajectory.name,
                    include_front_back=False,
                )
                continue
            records.append(record)
            _copy_annotations(source_root, annotations_root, source_trajectory.name)
            print(
                f"[front-back {device}] {index}/{len(source_trajectories)} "
                f"{source_trajectory.name}: candidate={record['candidate_index']}",
                flush=True,
            )

        collection = {
            "schema": FRONT_BACK_COLLECTION_SCHEMA,
            "format_version": FRONT_BACK_FORMAT_VERSION,
            "status": "complete",
            "source": str(source_root),
            "source_is_runtime_dependency": True,
            "source_snapshot_sha256": _source_snapshot_sha256(records, rejections),
            "seed": seed,
            "placement_candidates": placement_candidates,
            "table_margin_m": float(table_margin_m),
            "position_tolerance_m": float(position_tolerance_m),
            "max_placement_shift_x_m": float(max_placement_shift_x_m),
            "max_placement_shift_y_m": float(max_placement_shift_y_m),
            "min_base_clearance_m": float(min_base_clearance_m),
            "min_inter_arm_clearance_m": float(min_inter_arm_clearance_m),
            "input_trajectories": len(source_trajectories),
            "accepted_source_trajectories": len(records),
            "rejected_source_trajectories": len(rejections),
            "output_trajectories": 4 * len(records) + 2 * len(fallback_pairs),
            "variants": ["original", "front_back", "left_right", "front_back_left_right"],
            "trajectory_groups": records,
            "fallback_pairs": fallback_pairs,
            "rejections": rejections,
        }
        _write_json(staging / "augmentation.json", collection)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output": str(output_root),
        "input_trajectories": len(source_trajectories),
        "accepted_source_trajectories": len(records),
        "rejected_source_trajectories": len(rejections),
        "output_trajectories": 4 * len(records) + 2 * len(fallback_pairs),
        "seed": seed,
        "max_placement_shift_x_m": float(max_placement_shift_x_m),
        "max_placement_shift_y_m": float(max_placement_shift_y_m),
        "min_base_clearance_m": float(min_base_clearance_m),
        "min_inter_arm_clearance_m": float(min_inter_arm_clearance_m),
    }


class _PlacementRejected(RuntimeError):
    def __init__(self, diagnostic: dict):
        super().__init__(str(diagnostic.get("reason", "placement rejected")))
        self.diagnostic = diagnostic


def _augment_trajectory(
    source: Path,
    output_root: Path,
    *,
    seed: int,
    placement_candidates: int,
    table_margin_m: float,
    position_tolerance_m: float,
    max_placement_shift_x_m: float,
    max_placement_shift_y_m: float,
    min_base_clearance_m: float,
    min_inter_arm_clearance_m: float,
    kinematics,
    dynamics,
    collision,
) -> dict:
    reader = TrajectoryReader(source)
    robot_fields = {
        "sample_index": np.arange(reader.manifest.state_count, dtype=np.int64),
        **reader.state_arrays(_DERIVED_OVERRIDE_FIELDS),
    }
    actions = reader.action_arrays()
    particles = reader.state_arrays(("particle_q",))["particle_q"]
    source_poses = piper_attach_poses(robot_fields["body_q"])
    gripper_state = np.asarray(robot_fields["gripper_open_state"], dtype=np.float32)
    attached = np.asarray(robot_fields["grasp_attached"], dtype=np.int32)
    table = _table_contract(reader.manifest)
    source_center = _initial_xy_bbox_center(particles[0])
    source_max_joint_step = _maximum_arm_joint_step(robot_fields["joint_q"])
    source_motion_envelope = _motion_envelope(
        robot_fields["joint_qd"],
        reader.manifest,
    )
    home_attach_positions = kinematics.attach_poses_batch(
        np.asarray(kinematics.home, dtype=np.float32)[None]
    )[0, :, :3]
    candidates, candidate_contract = _placement_transforms(
        particles,
        source_center,
        table,
        seed=_trajectory_seed(seed, source.name),
        count=placement_candidates,
        margin_m=table_margin_m,
        max_shift_x_m=max_placement_shift_x_m,
        max_shift_y_m=max_placement_shift_y_m,
    )
    profile = robot_profile(PIPER)
    base_indices = tuple(
        profile.body_names.index(name) for name in ("left/base_link", "right/base_link")
    )
    base_positions = np.asarray(robot_fields["body_q"], dtype=np.float32)[
        0, base_indices, :3
    ]
    failures: list[dict] = []

    def reject(
        candidate_index: int,
        refinement_index: int,
        stage: str,
        diagnostic: dict,
    ) -> None:
        entry = {
            "candidate_index": candidate_index,
            "refinement_index": refinement_index,
            "refinement_offset_xy": (transform.translation[:2] - coarse_transform.translation[:2])
            .astype(float)
            .tolist(),
            "stage": stage,
            **diagnostic,
        }
        failures.append(entry)
        motion_ratios = ""
        if "max_velocity_ratio" in entry or "max_acceleration_ratio" in entry:
            motion_ratios = (
                f" velocity_ratio={entry.get('max_velocity_ratio', float('nan')):.3f}"
                f" acceleration_ratio={entry.get('max_acceleration_ratio', float('nan')):.3f}"
                f" velocity_frame={entry.get('max_velocity_frame', -1)}"
                f" velocity_joint={entry.get('max_velocity_arm_joint_index', -1)}"
                f" acceleration_frame={entry.get('max_acceleration_end_frame', -1)}"
                f" acceleration_joint={entry.get('max_acceleration_arm_joint_index', -1)}"
                f" joint_step_frame={entry.get('max_frame_joint_step_end_frame', -1)}"
                f" joint_step_joint={entry.get('max_frame_joint_step_arm_joint_index', -1)}"
                f" attach_error={entry.get('max_attach_position_error_m', float('nan')):.6f}"
                f" joint_step={entry.get('max_frame_joint_step_rad', float('nan')):.6f}"
                f" joint_step_limit={entry.get('joint_jump_limit_rad', float('nan')):.6f}"
                f" joint_margin={entry.get('minimum_joint_limit_margin_rad', float('nan')):.6f}"
            )
        collision = entry.get("worst_collision")
        collision_details = ""
        if collision is not None:
            collision_details = (
                f" collision_frame={collision.get('frame', -1)}"
                f" collision_pair={'/'.join(collision.get('body_pair', ())) or 'unknown'}"
                f" penetration={collision.get('penetration_m', float('nan')):.6f}"
            )
        print(
            f"[front-back] {source.name} candidate={candidate_index} "
            f"refinement={refinement_index} stage={stage} "
            f"reason={entry.get('reason', 'rejected')}{motion_ratios}{collision_details}",
            flush=True,
        )

    selected = None
    for candidate_index, coarse_transform in enumerate(candidates):
        attempts = [coarse_transform]
        for refinement_index, transform in enumerate(attempts):
            base_clearance = _validate_initial_base_clearance(
                transform.points(particles[0]),
                base_positions,
                min_clearance_m=min_base_clearance_m,
            )
            if not base_clearance["valid"]:
                reject(
                    candidate_index,
                    refinement_index,
                    "initial_base_clearance",
                    base_clearance,
                )
                continue
            target_poses = source_poses.copy()
            target_poses[..., :3] = transform.points(source_poses[..., :3])
            target_poses[..., 3:7] = _transform_gripper_rotations(
                source_poses[..., 3:7], transform.linear
            )
            target_poses[..., :3], cloth_relative_plan = _cloth_relative_attach_positions(
                target_poses[..., :3],
                transform.points(particles),
                attached,
                robot_fields["grasp_pin_count"],
                robot_fields["grasp_pins"],
                robot_fields["grasp_pin_local"],
                target_poses[..., 3:7],
            )
            target_poses[..., :3], free_motion_plan = _plan_free_attach_positions(
                target_poses[..., :3],
                attached,
                table_top_z=float(table["top_z"]),
                home_positions=home_attach_positions,
            )
            motion = _retarget_motion(
                target_poses,
                gripper_state,
                attached,
                kinematics=kinematics,
                position_tolerance_m=position_tolerance_m,
                control_fps=float(reader.manifest.control_fps),
            )
            motion, limit_validation = _select_smoothed_motion(
                motion,
                attached=attached,
                manifest=reader.manifest,
                kinematics=kinematics,
                profile=profile,
                position_tolerance_m=position_tolerance_m,
                source_max_joint_step_rad=source_max_joint_step,
                source_motion_envelope=source_motion_envelope,
            )
            if not limit_validation["valid"]:
                if refinement_index == 0 and _should_refine_kinematics(
                    limit_validation,
                    position_tolerance_m=position_tolerance_m,
                ):
                    attempts.extend(
                        _local_refinement_transforms(
                            coarse_transform,
                            source_center,
                            candidate_contract,
                        )
                    )
                reject(candidate_index, refinement_index, "kinematics", limit_validation)
                continue
            collision_validation = collision.validate(motion.joint_q)
            inter_arm_clearance = (
                collision.validate_inter_arm_clearance(
                    motion.joint_q,
                    min_clearance_m=min_inter_arm_clearance_m,
                )
                if collision_validation["valid"]
                else None
            )
            ik_branch = {"label": "home", "side": None, "wrist_offset_rad": 0.0}
            branch_attempts = [
                _ik_branch_attempt_diagnostic(
                    ik_branch,
                    limit_validation,
                    collision_validation,
                    inter_arm_clearance,
                )
            ]
            if not collision_validation["valid"] or not inter_arm_clearance["valid"]:
                valid_branch_results = []
                for branch, initial_joint_q in _ik_branch_initializations(
                    profile,
                    collision_validation,
                    inter_arm_clearance,
                ):
                    branch_motion = _retarget_motion(
                        target_poses,
                        gripper_state,
                        attached,
                        kinematics=kinematics,
                        position_tolerance_m=position_tolerance_m,
                        control_fps=float(reader.manifest.control_fps),
                        initial_joint_q=initial_joint_q,
                    )
                    branch_motion, branch_limits = _select_smoothed_motion(
                        branch_motion,
                        attached=attached,
                        manifest=reader.manifest,
                        kinematics=kinematics,
                        profile=profile,
                        position_tolerance_m=position_tolerance_m,
                        source_max_joint_step_rad=source_max_joint_step,
                        source_motion_envelope=source_motion_envelope,
                    )
                    branch_collision = (
                        collision.validate(branch_motion.joint_q)
                        if branch_limits["valid"]
                        else None
                    )
                    branch_clearance = (
                        collision.validate_inter_arm_clearance(
                            branch_motion.joint_q,
                            min_clearance_m=min_inter_arm_clearance_m,
                        )
                        if branch_collision is not None and branch_collision["valid"]
                        else None
                    )
                    naturalness = _ik_naturalness_score(branch_motion.joint_q, profile)
                    branch = {**branch, "naturalness": naturalness}
                    branch_attempts.append(
                        _ik_branch_attempt_diagnostic(
                            branch, branch_limits, branch_collision, branch_clearance
                        )
                    )
                    if (
                        branch_limits["valid"]
                        and branch_collision["valid"]
                        and branch_clearance["valid"]
                    ):
                        valid_branch_results.append(
                            (
                                naturalness["cost"],
                                branch_motion,
                                branch_limits,
                                branch_collision,
                                branch_clearance,
                                branch,
                            )
                        )
                if valid_branch_results:
                    (
                        _naturalness_cost,
                        motion,
                        limit_validation,
                        collision_validation,
                        inter_arm_clearance,
                        ik_branch,
                    ) = min(valid_branch_results, key=lambda result: result[0])
                else:
                    collision_validation = {
                        **collision_validation,
                        "ik_branch_attempts": branch_attempts,
                    }
            if not collision_validation["valid"]:
                if refinement_index == 0 and _should_refine(collision_validation):
                    attempts.extend(
                        _local_refinement_transforms(
                            coarse_transform,
                            source_center,
                            candidate_contract,
                        )
                    )
                reject(
                    candidate_index,
                    refinement_index,
                    "kinematic_collision",
                    collision_validation,
                )
                continue
            if not inter_arm_clearance["valid"]:
                reject(
                    candidate_index,
                    refinement_index,
                    "kinematic_inter_arm_clearance",
                    {**inter_arm_clearance, "ik_branch_attempts": branch_attempts},
                )
                continue
            reference_qd = _robot_velocity(
                motion.joint_q,
                robot_profile(PIPER),
                float(reader.manifest.control_fps),
            )

            dynamics_attempts = []
            for _dynamic_attempt in range(_DYNAMIC_CANDIDATE_ATTEMPTS):
                dynamic = _reconstruct_dynamics(dynamics, motion, reference_qd, attached)
                front_back_dynamics = _validate_front_back_dynamics(dynamic.validation)
                dynamic_limits = _enforce_motion_envelope(
                    _validate_motion_limits(
                        dynamic.joint_q,
                        dynamic.joint_qd,
                        reader.manifest,
                        kinematics,
                        position_errors=np.where(attached, dynamic.attach_errors, 0.0),
                        position_tolerance_m=0.01,
                        source_max_joint_step_rad=source_max_joint_step,
                    ),
                    source_motion_envelope,
                )
                dynamic_collision = collision.validate(dynamic.joint_q)
                dynamic_inter_arm_clearance = (
                    collision.validate_inter_arm_clearance(
                        dynamic.joint_q,
                        min_clearance_m=min_inter_arm_clearance_m,
                    )
                    if dynamic_collision["valid"]
                    else None
                )
                dynamics_attempts.append(
                    {
                        "robot_dynamics": dynamic.validation,
                        "front_back_validation": front_back_dynamics,
                        "joint_limits": dynamic_limits,
                        "collision": dynamic_collision,
                        "inter_arm_clearance": dynamic_inter_arm_clearance,
                    }
                )
                if (
                    front_back_dynamics["valid"]
                    and dynamic_limits["valid"]
                    and dynamic_collision["valid"]
                    and dynamic_inter_arm_clearance["valid"]
                ):
                    break
            dynamic.validation["front_back_validation"] = front_back_dynamics
            dynamic.validation["cold_reconstruction_attempts"] = len(dynamics_attempts)
            if len(dynamics_attempts) > 1:
                dynamic.validation["prior_failed_attempts"] = dynamics_attempts[:-1]
            if not front_back_dynamics["valid"]:
                reject(
                    candidate_index,
                    refinement_index,
                    "robot_dynamics",
                    {
                        "valid": False,
                        "reason": "robot_dynamics_invalid",
                        "robot_dynamics": dynamic.validation,
                        "robot_dynamics_attempts": dynamics_attempts,
                    },
                )
                continue
            if not dynamic_limits["valid"]:
                if refinement_index == 0:
                    attempts.extend(
                        _local_refinement_transforms(
                            coarse_transform,
                            source_center,
                            candidate_contract,
                        )
                    )
                reject(
                    candidate_index,
                    refinement_index,
                    "dynamic_limits",
                    {**dynamic_limits, "robot_dynamics_attempts": dynamics_attempts},
                )
                continue
            if not dynamic_collision["valid"]:
                if refinement_index == 0 and _should_refine(dynamic_collision):
                    attempts.extend(
                        _local_refinement_transforms(
                            coarse_transform,
                            source_center,
                            candidate_contract,
                        )
                    )
                reject(
                    candidate_index,
                    refinement_index,
                    "dynamic_collision",
                    {
                        **dynamic_collision,
                        "joint_limits": dynamic_limits,
                        "robot_dynamics_attempts": dynamics_attempts,
                    },
                )
                continue
            if not dynamic_inter_arm_clearance["valid"]:
                reject(
                    candidate_index,
                    refinement_index,
                    "dynamic_inter_arm_clearance",
                    dynamic_inter_arm_clearance,
                )
                continue
            selected = (
                candidate_index,
                refinement_index,
                transform,
                cloth_relative_plan,
                free_motion_plan,
                limit_validation,
                dynamic,
                dynamic_limits,
                dynamic_collision,
                dynamic_inter_arm_clearance,
                base_clearance,
                ik_branch,
                branch_attempts,
            )
            break
        if selected is not None:
            break
    if selected is None:
        raise _PlacementRejected(
            {
                "reason": "no_valid_placement",
                "candidate_contract": candidate_contract,
                "candidate_failures": failures,
            }
        )

    (
        candidate_index,
        refinement_index,
        transform,
        cloth_relative_plan,
        free_motion_plan,
        kinematic_validation,
        dynamic,
        dynamic_limits,
        dynamic_collision,
        dynamic_inter_arm_clearance,
        base_clearance,
        ik_branch,
        branch_attempts,
    ) = selected

    transformed_particles = transform.points(particles)
    table_validation = _validate_table_bounds(
        transformed_particles,
        table,
        margin_m=table_margin_m,
    )
    if not table_validation["valid"]:
        raise _PlacementRejected(
            {
                "reason": "transformed_cloth_outside_table",
                "candidate_index": candidate_index,
                "table_validation": table_validation,
            }
        )

    source_id = source.name
    front_back_id = f"{source_id}{FRONT_BACK_SUFFIX}"
    mirror_id = f"{source_id}{MIRROR_SUFFIX}"
    composed_id = f"{front_back_id}{MIRROR_SUFFIX}"
    _write_identity_overlay(source, output_root / source_id)
    front_back_states = _dynamic_state_arrays(robot_fields, dynamic)
    front_back_actions = {
        "sample_index": np.arange(reader.manifest.action_count, dtype=np.int64),
        "joint_target": np.asarray(dynamic.joint_target, dtype=np.float32),
        "gripper_open_command": np.asarray(dynamic.gripper_open_command, dtype=np.float32),
    }
    validation = {
        "training_eligible": True,
        "kinematics": {
            **kinematic_validation,
            "ik_branch": ik_branch,
            "ik_branch_attempts": branch_attempts,
        },
        "robot_dynamics": dynamic.validation,
        "joint_limits": dynamic_limits,
        "collision": dynamic_collision,
        "inter_arm_clearance": dynamic_inter_arm_clearance,
        "initial_base_clearance": base_clearance,
        "table": table_validation,
    }
    front_back_manifest = _write_variant_overlay(
        reader.manifest,
        output_root / front_back_id,
        base=source,
        states=front_back_states,
        actions=front_back_actions,
        transform=transform,
        augmentation={
            "schema": FRONT_BACK_TRAJECTORY_SCHEMA,
            "format_version": FRONT_BACK_FORMAT_VERSION,
            "kind": "front_back",
            "source_trajectory_id": source_id,
            "source_manifest_sha256": _sha256(source / "manifest.json"),
            "source_center_xy": source_center.astype(float).tolist(),
            "target_center_xy": transform.points(
                np.asarray((*source_center, 0.0), dtype=np.float32)
            )[:2]
            .astype(float)
            .tolist(),
            "candidate_index": candidate_index,
            "refinement_index": refinement_index,
            "refinement_offset_xy": (transform.translation[:2] - coarse_transform.translation[:2])
            .astype(float)
            .tolist(),
            "seed": seed,
            "position_only_ik": True,
            "grasp_frame_usage": "cloth_patch_reconstruction_only",
            "cloth_relative_attach": cloth_relative_plan,
            "ik_branch": ik_branch,
            "free_motion_replanned": True,
            "free_motion_plan": free_motion_plan,
            "robot_dynamics_rollout": True,
            "style3d_solver_used": False,
            "collision_checked": True,
            "surface_winding_reversed": True,
            "spatial_transform": transform.to_json_dict(),
            "validation": validation,
        },
    )
    mirrored_source_states = mirror_robot_state_arrays(robot_fields)
    mirrored_source_actions = mirror_action_arrays(actions)
    _write_variant_overlay(
        reader.manifest,
        output_root / mirror_id,
        base=source,
        states=mirrored_source_states,
        actions=mirrored_source_actions,
        transform=SpatialTransform(_REFLECT_Y, np.zeros(3, dtype=np.float32)),
        augmentation=_bilateral_metadata(source_id, source),
    )
    composed_states = mirror_robot_state_arrays(front_back_states)
    composed_actions = mirror_action_arrays(front_back_actions)
    _write_variant_overlay(
        front_back_manifest,
        output_root / composed_id,
        base=output_root / front_back_id,
        states=composed_states,
        actions=composed_actions,
        transform=SpatialTransform(_REFLECT_Y, np.zeros(3, dtype=np.float32)),
        augmentation={
            "schema": FRONT_BACK_TRAJECTORY_SCHEMA,
            "format_version": FRONT_BACK_FORMAT_VERSION,
            "kind": "front_back_then_bilateral_mirror",
            "source_trajectory_id": source_id,
            "front_back_trajectory_id": front_back_id,
            "candidate_index": candidate_index,
            "refinement_index": refinement_index,
            "refinement_offset_xy": (transform.translation[:2] - coarse_transform.translation[:2])
            .astype(float)
            .tolist(),
            "arm_joint_signs": PIPER_ARM_SIGNS.astype(int).tolist(),
            "left_right_swapped": True,
            "position_only_ik": True,
            "grasp_frame_usage": "cloth_patch_reconstruction_only",
            "cloth_relative_attach": cloth_relative_plan,
            "ik_branch": ik_branch,
            "free_motion_replanned": True,
            "free_motion_plan": free_motion_plan,
            "robot_dynamics_rollout": True,
            "style3d_solver_used": False,
            "collision_checked": True,
            "surface_winding_reversed": False,
            "validation": validation,
        },
    )
    return {
        "source_id": source_id,
        "original_id": source_id,
        "front_back_id": front_back_id,
        "mirror_id": mirror_id,
        "front_back_mirror_id": composed_id,
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "output_manifest_sha256": {
            source_id: _sha256(output_root / source_id / "manifest.json"),
            front_back_id: _sha256(output_root / front_back_id / "manifest.json"),
            mirror_id: _sha256(output_root / mirror_id / "manifest.json"),
            composed_id: _sha256(output_root / composed_id / "manifest.json"),
        },
        "action_count": reader.manifest.action_count,
        "candidate_index": candidate_index,
        "refinement_index": refinement_index,
        "refinement_offset_xy": (transform.translation[:2] - coarse_transform.translation[:2])
        .astype(float)
        .tolist(),
        "ik_branch": ik_branch,
        "source_center_xy": source_center.astype(float).tolist(),
        "target_center_xy": transform.points(np.asarray((*source_center, 0.0), dtype=np.float32))[
            :2
        ]
        .astype(float)
        .tolist(),
    }


def _reconstruct_dynamics(dynamics, motion, reference_qd: np.ndarray, attached: np.ndarray):
    return dynamics.reconstruct(
        motion.joint_q,
        reference_qd,
        motion.target_gripper_open,
        motion.target_attach_positions,
        motion.target_approach_directions,
        motion.pregrasp_mask,
        motion.approach_validation_mask,
        attached,
        target_up_directions=motion.target_up_directions,
    )


def _validate_front_back_dynamics(validation: dict) -> dict:
    """Validate task tracking while leaving free-arm safety to geometric checks."""

    limits = validation.get("limits", {})
    event_validation = validation.get("gripper_event_validation", {})
    checks = {
        "gripper_events": bool(event_validation.get("valid", False)),
        "attached_position": validation.get("max_attached_position_error_m", np.inf)
        <= limits.get("attach_position_error_m", -np.inf),
        "action_replay_position": validation.get("action_replay_position_error_rad", np.inf)
        <= limits.get("action_replay_position_error_rad", -np.inf),
        "action_replay_velocity": validation.get("action_replay_velocity_error_rad_s", np.inf)
        <= limits.get("action_replay_velocity_error_rad_s", -np.inf),
        "action_replay_gripper_position": validation.get(
            "action_replay_gripper_openness_error", np.inf
        )
        <= limits.get("action_replay_gripper_openness_error", -np.inf),
        "action_replay_gripper_velocity": validation.get(
            "action_replay_gripper_openness_velocity_error_s", np.inf
        )
        <= limits.get("action_replay_gripper_openness_velocity_error_s", -np.inf),
    }
    valid = bool(all(checks.values()))
    return {
        "valid": valid,
        "reason": "accepted" if valid else "task_dynamics_invalid",
        "checks": checks,
        "attached_targets_enforced": True,
        "detached_targets_are_collision_avoidance_guidance": True,
        "detached_motion_verified_on_realized_joint_path": True,
    }


def _dynamic_state_arrays(source: dict[str, np.ndarray], result) -> dict[str, np.ndarray]:
    count = len(result.joint_q)
    return {
        "sample_index": np.arange(count, dtype=np.int64),
        "joint_q": np.asarray(result.joint_q, dtype=np.float32),
        "joint_qd": np.asarray(result.joint_qd, dtype=np.float32),
        "body_q": np.asarray(result.body_q, dtype=np.float32),
        "body_qd": np.asarray(result.body_qd, dtype=np.float32),
        "gripper_open_state": np.asarray(result.gripper_open_state, dtype=np.float32),
        "grasp_attached": np.asarray(source["grasp_attached"], dtype=np.int32),
        "grasp_previous_open": np.asarray(source["grasp_previous_open"], dtype=np.float32),
        "grasp_pin_count": np.zeros((count, 2), dtype=np.int32),
        "grasp_pins": np.full_like(source["grasp_pins"], -1, dtype=np.int32),
        "grasp_pin_local": np.zeros_like(source["grasp_pin_local"], dtype=np.float32),
    }


def _write_identity_overlay(source: Path, output: Path) -> TrajectoryManifest:
    reader = TrajectoryReader(source)
    states = {
        "sample_index": np.arange(reader.manifest.state_count, dtype=np.int64),
        **reader.state_arrays(_DERIVED_OVERRIDE_FIELDS),
    }
    actions = reader.action_arrays()
    return _write_variant_overlay(
        reader.manifest,
        output,
        base=source,
        states=states,
        actions=actions,
        transform=None,
        augmentation={
            "schema": FRONT_BACK_TRAJECTORY_SCHEMA,
            "format_version": FRONT_BACK_FORMAT_VERSION,
            "kind": "identity_overlay",
            "source_trajectory_id": source.name,
            "source_manifest_sha256": _sha256(source / "manifest.json"),
            "surface_winding_reversed": False,
        },
    )


def _write_identity_and_bilateral_pair(source: Path, output_root: Path) -> dict:
    """Keep safe symmetry variants when front/back retargeting is rejected."""

    reader = TrajectoryReader(source)
    source_id = source.name
    mirror_id = f"{source_id}{MIRROR_SUFFIX}"
    _write_identity_overlay(source, output_root / source_id)
    robot_fields = {
        "sample_index": np.arange(reader.manifest.state_count, dtype=np.int64),
        **reader.state_arrays(_DERIVED_OVERRIDE_FIELDS),
    }
    _write_variant_overlay(
        reader.manifest,
        output_root / mirror_id,
        base=source,
        states=mirror_robot_state_arrays(robot_fields),
        actions=mirror_action_arrays(reader.action_arrays()),
        transform=SpatialTransform(_REFLECT_Y, np.zeros(3, dtype=np.float32)),
        augmentation=_bilateral_metadata(source_id, source),
    )
    return {
        "source_id": source_id,
        "original_id": source_id,
        "mirror_id": mirror_id,
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "output_manifest_sha256": {
            source_id: _sha256(output_root / source_id / "manifest.json"),
            mirror_id: _sha256(output_root / mirror_id / "manifest.json"),
        },
        "action_count": reader.manifest.action_count,
        "front_back_rejected": True,
    }


def _write_variant_overlay(
    source_manifest: TrajectoryManifest,
    output: Path,
    *,
    base: Path,
    states: dict[str, np.ndarray],
    actions: dict[str, np.ndarray],
    transform: SpatialTransform | None,
    augmentation: dict,
) -> TrajectoryManifest:
    metadata = copy.deepcopy(source_manifest.metadata)
    metadata["augmentation"] = augmentation
    base_path = (
        os.path.relpath(base.resolve(), output.resolve())
        if base.resolve().parent == output.resolve().parent
        else str(base.resolve())
    )
    metadata["storage"] = {
        "kind": "derived_overlay_v1",
        "base_trajectory": base_path,
        "base_manifest_sha256": _sha256(base / "manifest.json"),
        **({"base_spatial_transform": transform.to_json_dict()} if transform is not None else {}),
    }
    manifest = copy.deepcopy(source_manifest)
    manifest.metadata = metadata
    output.mkdir(parents=True)
    _write_npz(output / "states.npz", states)
    _write_npz(output / "actions.npz", actions)
    _write_json(output / "manifest.json", manifest.to_json_dict())
    published = TrajectoryReader(output)
    for kind, arrays in (
        ("state", published.state_arrays(_DERIVED_OVERRIDE_FIELDS)),
        ("action", published.action_arrays()),
    ):
        for field, values in arrays.items():
            if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
                raise RuntimeError(f"front/back overlay contains non-finite {kind}: {field}")
    return manifest


def _bilateral_metadata(source_id: str, source: Path) -> dict:
    return {
        "schema": FRONT_BACK_TRAJECTORY_SCHEMA,
        "format_version": FRONT_BACK_FORMAT_VERSION,
        "kind": "bilateral_mirror",
        "source_trajectory_id": source_id,
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "mirror_plane": {"axis": "y", "offset_m": 0.0},
        "arm_joint_signs": PIPER_ARM_SIGNS.astype(int).tolist(),
        "left_right_swapped": True,
        "body_orientation_frame": "piper_home_body",
        "surface_winding_reversed": True,
        "kinematics_used": False,
        "physics_used": False,
    }


def _initial_xy_bbox_center(particle_q: np.ndarray) -> np.ndarray:
    values = np.asarray(particle_q, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("initial particle_q must be finite with shape [P, 3]")
    return (0.5 * (values[:, :2].min(axis=0) + values[:, :2].max(axis=0))).astype(np.float32)


def _transform_gripper_rotations(rotations: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Transform gripper frames through a reflection using local-X symmetry."""

    quaternions = np.asarray(rotations, dtype=np.float64)
    transform = np.asarray(linear, dtype=np.float64).reshape(3, 3)
    if quaternions.ndim != 3 or quaternions.shape[1:] != (2, 4):
        raise ValueError("gripper rotations must have shape [N, 2, 4]")
    if not np.isfinite(quaternions).all() or not np.isfinite(transform).all():
        raise ValueError("gripper rotations and transform must be finite")
    if not np.allclose(transform.T @ transform, np.eye(3), atol=1.0e-6):
        raise ValueError("gripper rotation transform must be orthogonal")
    determinant = float(np.linalg.det(transform))
    if not np.isclose(abs(determinant), 1.0, atol=1.0e-6):
        raise ValueError("gripper rotation transform determinant must have unit magnitude")
    local_symmetry = np.diag((-1.0, 1.0, 1.0)) if determinant < 0.0 else np.eye(3)
    matrices = _quaternion_matrices(quaternions)
    transformed = transform @ matrices @ local_symmetry
    result = _rotation_quaternions(transformed)
    for side in range(2):
        for index in range(1, len(result)):
            if float(np.dot(result[index - 1, side], result[index, side])) < 0.0:
                result[index, side] *= -1.0
    return result.astype(np.float32)


def _cloth_relative_attach_positions(
    target_positions: np.ndarray,
    transformed_particles: np.ndarray,
    grasp_attached: np.ndarray,
    grasp_pin_count: np.ndarray,
    grasp_pins: np.ndarray,
    grasp_pin_local: np.ndarray,
    target_rotations: np.ndarray,
    *,
    max_correction_m: float = _CLOTH_RELATIVE_MAX_POSITION_CORRECTION_M,
    smoothing_frames: int = _CLOTH_RELATIVE_SMOOTHING_FRAMES,
) -> tuple[np.ndarray, dict]:
    """Follow recorded cloth patches without letting soft-pin error drive the arm."""

    targets = np.asarray(target_positions, dtype=np.float32)
    particles = np.asarray(transformed_particles, dtype=np.float32)
    attached = np.asarray(grasp_attached, dtype=np.int32) != 0
    counts = np.asarray(grasp_pin_count, dtype=np.int32)
    pins = np.asarray(grasp_pins, dtype=np.int32)
    local = np.asarray(grasp_pin_local, dtype=np.float32)
    rotations = np.asarray(target_rotations, dtype=np.float32)
    frame_count = len(targets)
    if targets.shape != (frame_count, 2, 3) or particles.ndim != 3:
        raise ValueError("cloth-relative targets require [N, 2, 3] targets and [N, P, 3] cloth")
    if particles.shape[0] != frame_count or particles.shape[2] != 3:
        raise ValueError("cloth particles must share the target timeline")
    if attached.shape != (frame_count, 2) or counts.shape != (frame_count, 2):
        raise ValueError("cloth grasp state must have shape [N, 2]")
    if pins.shape[:2] != (frame_count, 2) or local.shape != (*pins.shape, 3):
        raise ValueError("cloth grasp pins and local offsets have incompatible shapes")
    if rotations.shape != (frame_count, 2, 4):
        raise ValueError("cloth-relative rotations must have shape [N, 2, 4]")
    if not np.isfinite(max_correction_m) or max_correction_m < 0.0:
        raise ValueError("cloth-relative correction limit must be finite and nonnegative")
    if not isinstance(smoothing_frames, int) or smoothing_frames < 1 or smoothing_frames % 2 == 0:
        raise ValueError("cloth-relative smoothing window must be a positive odd integer")

    raw = np.zeros_like(targets, dtype=np.float64)
    valid = np.zeros((frame_count, 2), dtype=bool)
    matrices = _quaternion_matrices(rotations)
    particle_count = particles.shape[1]
    capacity = pins.shape[2]
    for frame in range(frame_count):
        for side in range(2):
            count = int(counts[frame, side])
            if not attached[frame, side] or not 0 < count <= capacity:
                continue
            selected = pins[frame, side, :count]
            if np.any((selected < 0) | (selected >= particle_count)):
                continue
            reconstructed = particles[frame, selected].mean(axis=0) - (
                matrices[frame, side] @ local[frame, side, :count].mean(axis=0)
            )
            correction = reconstructed - targets[frame, side]
            norm = float(np.linalg.norm(correction))
            if norm > max_correction_m > 0.0:
                correction *= max_correction_m / norm
            elif max_correction_m == 0.0:
                correction.fill(0.0)
            raw[frame, side] = correction
            valid[frame, side] = True

    smoothed = np.zeros_like(raw)
    half = smoothing_frames // 2
    kernel = np.hanning(smoothing_frames + 2)[1:-1]
    kernel /= kernel.sum()
    for side in range(2):
        for start, end in _true_runs(valid[:, side]):
            values = raw[start : end + 1, side]
            if len(values) == 1:
                continue
            padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
            for axis in range(3):
                smoothed[start : end + 1, side, axis] = np.convolve(
                    padded[:, axis], kernel, mode="valid"
                )
            taper_frames = min(smoothing_frames, (len(values) + 1) // 2)
            ramp = np.linspace(0.0, 1.0, taper_frames, dtype=np.float64)
            ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            envelope = np.ones(len(values), dtype=np.float64)
            envelope[:taper_frames] = np.minimum(envelope[:taper_frames], ramp)
            envelope[-taper_frames:] = np.minimum(envelope[-taper_frames:], ramp[::-1])
            smoothed[start : end + 1, side] *= envelope[:, None]
    norms = np.linalg.norm(smoothed, axis=-1)
    scale = np.minimum(1.0, max_correction_m / np.maximum(norms, 1.0e-12))
    smoothed *= scale[..., None]
    result = targets.copy()
    result[valid] += smoothed[valid].astype(np.float32)
    applied = np.linalg.norm(smoothed[valid], axis=-1)
    return result, {
        "schema": "cloth_patch_relative_attach_v1",
        "source": "recorded_grasp_pins_and_pin_local",
        "soft_pin_feedback_bounded": True,
        "max_position_correction_m": float(max_correction_m),
        "smoothing_frames": int(smoothing_frames),
        "boundary_taper_frames": int(smoothing_frames),
        "corrected_state_count": int(np.count_nonzero(valid)),
        "max_applied_correction_m": float(applied.max(initial=0.0)),
        "mean_applied_correction_m": float(applied.mean()) if len(applied) else 0.0,
    }


def _quaternion_matrices(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1.0e-12):
        raise ValueError("rotation quaternions must be finite and nonzero")
    x, y, z, w = np.moveaxis(values / norms, -1, 0)
    result = np.empty((*values.shape[:-1], 3, 3), dtype=np.float64)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def _rotation_quaternions(rotations: np.ndarray) -> np.ndarray:
    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError("rotation matrices must be finite with shape [..., 3, 3]")
    result = np.empty((*matrices.shape[:-2], 4), dtype=np.float64)
    flat_matrices = matrices.reshape(-1, 3, 3)
    flat_result = result.reshape(-1, 4)
    for index, matrix in enumerate(flat_matrices):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            value = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                value = (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            elif axis == 1:
                scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                value = (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            else:
                scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                value = (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
        flat_result[index] = value
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result


def _plan_free_attach_positions(
    target_positions: np.ndarray,
    grasp_attached: np.ndarray,
    *,
    table_top_z: float,
    clearance_m: float = _FREE_MOTION_TABLE_CLEARANCE_M,
    planar_adjustment_m: float = _FREE_MOTION_MAX_PLANAR_ADJUSTMENT_M,
    home_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Replace detached targets with bounded, local fixed-horizon transfers."""

    targets = np.asarray(target_positions, dtype=np.float32)
    attached = np.asarray(grasp_attached, dtype=np.int32) != 0
    if targets.ndim != 3 or targets.shape[1:] != (2, 3) or len(targets) == 0:
        raise ValueError("target_positions must have nonempty shape [N, 2, 3]")
    if attached.shape != targets.shape[:2]:
        raise ValueError("grasp_attached must have shape [N, 2]")
    if not np.isfinite(targets).all():
        raise ValueError("target_positions must be finite")
    if not np.isfinite(table_top_z):
        raise ValueError("table_top_z must be finite")
    if not np.isfinite(clearance_m) or clearance_m <= 0.0:
        raise ValueError("clearance_m must be positive and finite")
    home = targets[0].copy() if home_positions is None else np.asarray(home_positions)
    if home.shape != (2, 3) or not np.isfinite(home).all():
        raise ValueError("home_positions must contain two finite Cartesian positions")
    if (
        not np.isfinite(planar_adjustment_m)
        or planar_adjustment_m < 0.0
        or planar_adjustment_m > _FREE_MOTION_MAX_PLANAR_ADJUSTMENT_M
    ):
        raise ValueError("planar adjustment must stay inside the configured local bound")

    planned = targets.copy()
    segments: list[dict] = []
    overwritten = np.zeros(2, dtype=np.int64)
    runs_by_side = [_true_runs(~attached[:, side]) for side in range(2)]
    schedules = _coordinated_free_motion_schedule(runs_by_side, len(targets))
    for side in range(2):
        for free_start, free_end in runs_by_side[side]:
            anchor_start = max(0, free_start - 1)
            anchor_end = min(len(targets) - 1, free_end + 1)
            terminal_hold = free_end == len(targets) - 1
            available_frames = anchor_end - anchor_start + 1
            route_start = home[side] if free_start == 0 else targets[anchor_start, side]
            start_step = (
                targets[anchor_start, side] - targets[anchor_start - 1, side]
                if anchor_start > 0
                else np.zeros(3, dtype=np.float32)
            )
            # Approach a grasp at rest. Carrying the following attached-frame
            # velocity backward into the free path can overshoot the pregrasp
            # pose and sweep a wrist through nearby static geometry.
            end_step = np.zeros(3, dtype=np.float32)
            schedule = schedules[(side, free_start, free_end)]
            coordinated_staging = bool(
                schedule["yield_end_fraction"] is not None
                or schedule["hold_end_fraction"] > 0.68 + 1.0e-9
            )
            other_side = 1 - side
            other_terminal_start = next(
                (
                    start
                    for start, end in runs_by_side[other_side]
                    if end == len(targets) - 1
                ),
                None,
            )
            near_simultaneous_terminal_release = bool(
                terminal_hold
                and other_terminal_start is not None
                and abs(free_start - other_terminal_start) <= 1
            )
            terminal_after_other_release = bool(
                terminal_hold
                and not near_simultaneous_terminal_release
                and other_terminal_start is not None
                and free_start >= other_terminal_start
            )
            terminal_static_hold = bool(
                near_simultaneous_terminal_release or terminal_after_other_release
            )
            other_attached = attached[free_start : free_end + 1, other_side]
            other_targets = targets[free_start : free_end + 1, other_side][other_attached]
            nearest_other_distance = (
                float(np.min(np.linalg.norm(other_targets[:, :2] - route_start[:2], axis=-1)))
                if len(other_targets)
                else float("inf")
            )
            crosses_center = bool(route_start[1] * home[side, 1] < 0.0)
            near_other_path = bool(
                coordinated_staging
                and nearest_other_distance < _FREE_MOTION_STANDBY_TRIGGER_DISTANCE_M
            )
            home_standby = crosses_center or near_other_path
            standby_position = home[side] if terminal_hold or home_standby else None
            route_planar_limit = (
                planar_adjustment_m
                if terminal_hold or home_standby
                else min(planar_adjustment_m, _FREE_MOTION_LOCAL_PLANAR_ADJUSTMENT_M)
            )
            standby_distance = (
                route_planar_limit
                if standby_position is None
                else min(
                    float(np.linalg.norm(standby_position[:2] - route_start[:2])),
                    route_planar_limit,
                )
            )
            local_transfer = not terminal_hold and not coordinated_staging
            terminal_retreat_end_fraction = min(
                0.45,
                _FREE_MOTION_TERMINAL_RETREAT_MAX_FRAMES / max(1, available_frames - 1),
            )
            if terminal_static_hold:
                route = np.repeat(route_start[None], available_frames, axis=0).astype(
                    np.float32
                )
            elif terminal_hold:
                route = _retreat_terminal_pose(
                    route_start,
                    available_frames,
                    side=side,
                    crosses_center=crosses_center,
                    table_top_z=table_top_z,
                    clearance_m=clearance_m,
                    start_step=start_step,
                    standby_position=standby_position,
                    planar_adjustment_m=route_planar_limit,
                )
            elif local_transfer:
                route = _lift_transfer_descend(
                    route_start,
                    targets[anchor_end, side],
                    available_frames,
                    table_top_z=table_top_z,
                    clearance_m=clearance_m,
                    start_step=start_step,
                    end_step=end_step,
                )
            else:
                route = _retreat_wait_approach(
                    route_start,
                    targets[anchor_end, side],
                    anchor_end - anchor_start + 1,
                    side=side,
                    table_top_z=table_top_z,
                    clearance_m=clearance_m,
                    retreat_end_fraction=schedule["retreat_end_fraction"],
                    hold_end_fraction=schedule["hold_end_fraction"],
                    yield_end_fraction=schedule["yield_end_fraction"],
                    yield_resume_fraction=schedule["yield_resume_fraction"],
                    start_step=start_step,
                    end_step=end_step,
                    standby_position=standby_position,
                    planar_adjustment_m=route_planar_limit,
                )
            source_slice = slice(free_start - anchor_start, free_end - anchor_start + 1)
            planned[free_start : free_end + 1, side] = route[source_slice]
            overwritten[side] += free_end - free_start + 1
            segments.append(
                {
                    "side": "left" if side == 0 else "right",
                    "free_start_frame": free_start,
                    "free_end_frame": free_end,
                    "anchor_start_frame": anchor_start,
                    "anchor_end_frame": anchor_end,
                    "available_state_frames": available_frames,
                    "terminal_hold": terminal_hold,
                    "near_simultaneous_terminal_release": near_simultaneous_terminal_release,
                    "terminal_after_other_release": terminal_after_other_release,
                    "crosses_center": crosses_center,
                    "near_other_attached_path": near_other_path,
                    "nearest_other_attached_path_m": (
                        nearest_other_distance if np.isfinite(nearest_other_distance) else None
                    ),
                    "route": (
                        "terminal_hold_near_simultaneous_release"
                        if near_simultaneous_terminal_release
                        else "terminal_hold_after_other_release"
                        if terminal_after_other_release
                        else "terminal_cross_center_retreat"
                        if terminal_hold and crosses_center
                        else "terminal_retreat"
                        if terminal_hold
                        else "local_transfer"
                        if local_transfer
                        else "coordinated_staging"
                    ),
                    "retreat_end_fraction": (
                        0.0
                        if terminal_static_hold
                        else terminal_retreat_end_fraction
                        if terminal_hold
                        else schedule["retreat_end_fraction"]
                    ),
                    "hold_end_fraction": (
                        1.0 if terminal_hold else schedule["hold_end_fraction"]
                    ),
                    "yield_end_fraction": (
                        None if terminal_hold else schedule["yield_end_fraction"]
                    ),
                    "yield_resume_fraction": (
                        None if terminal_hold else schedule["yield_resume_fraction"]
                    ),
                    "clearance_z_m": float(np.max(route[:, 2], initial=-np.inf)),
                    "standby_planar_adjustment_m": (
                        0.0
                        if terminal_static_hold
                        else float(standby_distance)
                        if terminal_hold or not local_transfer
                        else 0.0
                    ),
                    "path_length_m": float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=-1))),
                }
            )

    if not np.array_equal(planned[attached], targets[attached]):
        raise RuntimeError("free-motion planning modified an attached target")
    return planned, {
        "schema": "bounded_local_free_attach_planner_v4",
        "fixed_timeline": True,
        "attached_targets_preserved": True,
        "initial_free_pose": "robot_home_attach_position",
        "table_clearance_m": float(clearance_m),
        "local_adjustment": {
            "maximum_planar_m": float(planar_adjustment_m),
            "hard_maximum_planar_m": _FREE_MOTION_MAX_PLANAR_ADJUSTMENT_M,
            "same_side_planar_m": _FREE_MOTION_LOCAL_PLANAR_ADJUSTMENT_M,
            "maximum_yield_planar_m": _FREE_MOTION_MAX_YIELD_PLANAR_ADJUSTMENT_M,
            "maximum_extra_lift_m": _FREE_MOTION_MAX_EXTRA_LIFT_M,
            "cross_center_standby_target": "robot_home_attach_position",
            "other_path_trigger_distance_m": _FREE_MOTION_STANDBY_TRIGGER_DISTANCE_M,
            "hold_fraction": [0.32, 0.68],
        },
        "overwritten_state_count": overwritten.astype(int).tolist(),
        "segments": segments,
    }


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("run mask must be one-dimensional")
    padded = np.pad(values.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end - 1)) for start, end in edges.reshape(-1, 2)]


def _coordinated_free_motion_schedule(
    runs_by_side: list[list[tuple[int, int]]],
    frame_count: int,
) -> dict[tuple[int, int, int], dict[str, float | None]]:
    """Stagger overlapping releases so one arm clears before the other approaches."""

    schedules = {
        (side, start, end): {
            "retreat_end_fraction": 0.32,
            "hold_end_fraction": 0.68,
            "yield_end_fraction": None,
            "yield_resume_fraction": None,
        }
        for side, runs in enumerate(runs_by_side)
        for start, end in runs
    }
    for left in runs_by_side[0]:
        for right in runs_by_side[1]:
            overlap_start = max(left[0], right[0])
            overlap_end = min(left[1], right[1])
            if overlap_start > overlap_end or left[1] == right[1]:
                continue
            early_side, early = (0, left) if left[1] < right[1] else (1, right)
            late_side, late = (1, right) if early_side == 0 else (0, left)
            if early[1] == frame_count - 1 or not (early[0] < late[0] <= early[1]):
                continue
            # Start the attaching arm early enough to leave a useful
            # deceleration window, while giving the yielding arm the original
            # longer retreat. Their short overlap is accepted only if the
            # downstream frame-by-frame clearance checks pass.
            approach_fraction = 0.10 if late[1] == frame_count - 1 else 0.25
            approach_start = overlap_start + round(
                approach_fraction * (overlap_end - overlap_start)
            )
            yield_complete = overlap_start + round(0.55 * (overlap_end - overlap_start))
            early_anchor_start = max(0, early[0] - 1)
            early_anchor_end = min(frame_count - 1, early[1] + 1)
            late_anchor_start = max(0, late[0] - 1)
            late_anchor_end = min(frame_count - 1, late[1] + 1)
            early_fraction = (approach_start - early_anchor_start) / max(
                1, early_anchor_end - early_anchor_start
            )
            yield_end_fraction = (yield_complete - late_anchor_start) / max(
                1, late_anchor_end - late_anchor_start
            )
            late_duration = max(1, late_anchor_end - late_anchor_start)
            yield_resume_frame = yield_complete + max(2, round(0.02 * late_duration))
            yield_resume_fraction = (yield_resume_frame - late_anchor_start) / late_duration
            early_schedule = schedules[(early_side, *early)]
            early_schedule["hold_end_fraction"] = float(
                np.clip(
                    max(early_schedule["hold_end_fraction"], early_fraction),
                    0.68,
                    0.90,
                )
            )
            late_schedule = schedules[(late_side, *late)]
            previous_yield_end = late_schedule["yield_end_fraction"]
            previous_yield_resume = late_schedule["yield_resume_fraction"]
            late_schedule["yield_end_fraction"] = float(
                np.clip(
                    yield_end_fraction
                    if previous_yield_end is None
                    else min(previous_yield_end, yield_end_fraction),
                    0.10,
                    0.32,
                )
            )
            late_schedule["yield_resume_fraction"] = float(
                np.clip(
                    yield_resume_fraction
                    if previous_yield_resume is None
                    else max(previous_yield_resume, yield_resume_fraction),
                    0.11,
                    0.50,
                )
            )
    return schedules


def _lift_transfer_descend(
    start: np.ndarray,
    end: np.ndarray,
    count: int,
    *,
    table_top_z: float,
    clearance_m: float,
    start_step: np.ndarray | None = None,
    end_step: np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("free-motion route count must be a positive integer")
    first = np.asarray(start, dtype=np.float64).reshape(3)
    last = np.asarray(end, dtype=np.float64).reshape(3)
    if not np.isfinite(first).all() or not np.isfinite(last).all():
        raise ValueError("free-motion route endpoints must be finite")
    if count == 1:
        return first[None].astype(np.float32)
    clearance_z = max(float(first[2]), float(last[2]), float(table_top_z + clearance_m))
    waypoints = np.stack(
        (
            first,
            np.asarray((first[0], first[1], clearance_z)),
            np.asarray((last[0], last[1], clearance_z)),
            last,
        )
    )
    lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=-1)
    active = lengths > 1.0e-9
    if not np.any(active):
        return np.repeat(first[None], count, axis=0).astype(np.float32)
    waypoints = np.concatenate((waypoints[:-1][active], waypoints[-1:]), axis=0)
    lengths = lengths[active]
    boundaries = np.concatenate(([0.0], np.cumsum(lengths) / np.sum(lengths)))
    start_delta = _finite_route_step(start_step)
    end_delta = _finite_route_step(end_step)
    result = np.empty((count, 3), dtype=np.float64)
    for index, progress in enumerate(np.linspace(0.0, 1.0, count)):
        segment = min(
            int(np.searchsorted(boundaries[1:], progress, side="right")),
            len(lengths) - 1,
        )
        width = boundaries[segment + 1] - boundaries[segment]
        fraction = np.clip((progress - boundaries[segment]) / width, 0.0, 1.0)
        duration_frames = (boundaries[segment + 1] - boundaries[segment]) * (count - 1)
        start_tangent = start_delta * duration_frames if segment == 0 else np.zeros(3)
        end_tangent = end_delta * duration_frames if segment == len(lengths) - 1 else np.zeros(3)
        result[index] = _quintic_hermite(
            waypoints[segment],
            waypoints[segment + 1],
            fraction,
            start_tangent,
            end_tangent,
        )
    result[0] = first
    result[-1] = last
    return result.astype(np.float32)


def _retreat_wait_approach(
    start: np.ndarray,
    end: np.ndarray,
    count: int,
    *,
    side: int,
    table_top_z: float,
    clearance_m: float,
    retreat_end_fraction: float = 0.32,
    hold_end_fraction: float = 0.68,
    yield_end_fraction: float | None = None,
    yield_resume_fraction: float | None = None,
    start_step: np.ndarray | None = None,
    end_step: np.ndarray | None = None,
    standby_position: np.ndarray | None = None,
    planar_adjustment_m: float = _FREE_MOTION_MAX_PLANAR_ADJUSTMENT_M,
) -> np.ndarray:
    """Route a free hand through a bounded local standby point."""

    if side not in (0, 1):
        raise ValueError("free-motion side must be 0 (left) or 1 (right)")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("free-motion route count must be a positive integer")
    first = np.asarray(start, dtype=np.float64).reshape(3)
    last = np.asarray(end, dtype=np.float64).reshape(3)
    if not np.isfinite(first).all() or not np.isfinite(last).all():
        raise ValueError("free-motion route endpoints must be finite")
    if not 0.0 < retreat_end_fraction < hold_end_fraction < 1.0:
        raise ValueError("free-motion phase fractions must be strictly ordered")
    if (yield_end_fraction is None) != (yield_resume_fraction is None):
        raise ValueError("free-motion yield fractions must be provided together")
    if (
        not np.isfinite(planar_adjustment_m)
        or planar_adjustment_m < 0.0
        or planar_adjustment_m > _FREE_MOTION_MAX_PLANAR_ADJUSTMENT_M
    ):
        raise ValueError("free-motion planar adjustment exceeds its local bound")
    if count == 1:
        return first[None].astype(np.float32)

    clearance_z = max(
        float(first[2]),
        float(last[2]),
        float(table_top_z + clearance_m),
    )
    endpoint_z = max(float(first[2]), float(last[2]))
    if clearance_z - endpoint_z > _FREE_MOTION_MAX_EXTRA_LIFT_M + 1.0e-9:
        raise ValueError("required table clearance exceeds the bounded free-motion lift")
    staging_z = clearance_z
    if standby_position is None:
        planar_delta = np.asarray((0.0, 1.0 if side == 0 else -1.0))
    else:
        standby = np.asarray(standby_position, dtype=np.float64).reshape(3)
        if not np.isfinite(standby).all():
            raise ValueError("free-motion standby position must be finite")
        planar_delta = standby[:2] - first[:2]
    planar_distance = float(np.linalg.norm(planar_delta))
    if planar_distance > planar_adjustment_m > 0.0:
        planar_delta *= planar_adjustment_m / planar_distance
    elif planar_adjustment_m == 0.0:
        planar_delta.fill(0.0)
    elif standby_position is None:
        planar_delta *= planar_adjustment_m
    staging = np.asarray((*tuple(first[:2] + planar_delta), staging_z))
    approach_above_end = hold_end_fraction + 0.4375 * (1.0 - hold_end_fraction)
    if yield_end_fraction is None:
        waypoints = np.stack(
            (
                first,
                np.asarray((first[0], first[1], staging_z)),
                staging,
                staging,
                np.asarray((last[0], last[1], staging_z)),
                last,
            )
        )
        retreat_vertical_end = 0.5625 * retreat_end_fraction
        boundaries = np.asarray(
            (
                0.0,
                retreat_vertical_end,
                retreat_end_fraction,
                hold_end_fraction,
                approach_above_end,
                1.0,
            )
        )
    else:
        if not 0.0 < yield_end_fraction < yield_resume_fraction < hold_end_fraction:
            raise ValueError("free-motion yield phases must precede the staging hold")
        staging_end = min(
            hold_end_fraction - 0.05,
            max(retreat_end_fraction, yield_resume_fraction + 0.18),
        )
        if staging_end <= yield_resume_fraction:
            raise ValueError("free-motion yield leaves no time to reach staging")
        yield_delta = planar_delta.copy()
        yield_distance = float(np.linalg.norm(yield_delta))
        if yield_distance > _FREE_MOTION_MAX_YIELD_PLANAR_ADJUSTMENT_M:
            yield_delta *= _FREE_MOTION_MAX_YIELD_PLANAR_ADJUSTMENT_M / yield_distance
        yield_point = np.asarray((*tuple(first[:2] + yield_delta), staging_z))
        waypoints = np.stack(
            (
                first,
                yield_point,
                yield_point,
                staging,
                staging,
                np.asarray((last[0], last[1], staging_z)),
                last,
            )
        )
        boundaries = np.asarray(
            (
                0.0,
                yield_end_fraction,
                yield_resume_fraction,
                staging_end,
                hold_end_fraction,
                approach_above_end,
                1.0,
            )
        )
    start_delta = _finite_route_step(start_step)
    end_delta = _finite_route_step(end_step)
    result = np.empty((count, 3), dtype=np.float64)
    for index, progress in enumerate(np.linspace(0.0, 1.0, count)):
        segment = min(
            int(np.searchsorted(boundaries[1:], progress, side="right")),
            len(boundaries) - 2,
        )
        width = boundaries[segment + 1] - boundaries[segment]
        fraction = np.clip((progress - boundaries[segment]) / width, 0.0, 1.0)
        duration_frames = (boundaries[segment + 1] - boundaries[segment]) * (count - 1)
        start_tangent = start_delta * duration_frames if segment == 0 else np.zeros(3)
        end_tangent = (
            end_delta * duration_frames if segment == len(boundaries) - 2 else np.zeros(3)
        )
        result[index] = _quintic_hermite(
            waypoints[segment],
            waypoints[segment + 1],
            fraction,
            start_tangent,
            end_tangent,
        )
    result[0] = first
    result[-1] = last
    return result.astype(np.float32)


def _finite_route_step(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float64)
    step = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(step).all():
        raise ValueError("free-motion boundary step must be finite")
    return step


def _quintic_hermite(
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
) -> np.ndarray:
    """Interpolate with prescribed endpoint velocity and zero endpoint acceleration."""

    t = float(fraction)
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    v0 = np.asarray(start_tangent, dtype=np.float64)
    v1 = np.asarray(end_tangent, dtype=np.float64)
    displacement = last - first - v0
    velocity_change = v1 - v0
    c3 = 10.0 * displacement - 4.0 * velocity_change
    c4 = 7.0 * velocity_change - 15.0 * displacement
    c5 = 6.0 * displacement - 3.0 * velocity_change
    return first + v0 * t + c3 * t**3 + c4 * t**4 + c5 * t**5


def _retreat_terminal_pose(
    start: np.ndarray,
    count: int,
    *,
    side: int,
    crosses_center: bool = False,
    table_top_z: float,
    clearance_m: float,
    start_step: np.ndarray | None = None,
    standby_position: np.ndarray | None = None,
    planar_adjustment_m: float = _FREE_MOTION_MAX_PLANAR_ADJUSTMENT_M,
) -> np.ndarray:
    """Lift a terminally detached hand toward its own bounded standby lane."""

    if side not in (0, 1):
        raise ValueError("free-motion side must be 0 (left) or 1 (right)")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("free-motion route count must be a positive integer")
    first = np.asarray(start, dtype=np.float64).reshape(3)
    if not np.isfinite(first).all():
        raise ValueError("free-motion route endpoint must be finite")
    if (
        not np.isfinite(planar_adjustment_m)
        or planar_adjustment_m < 0.0
        or planar_adjustment_m > _FREE_MOTION_MAX_PLANAR_ADJUSTMENT_M
    ):
        raise ValueError("free-motion planar adjustment exceeds its local bound")
    if count == 1:
        return first[None].astype(np.float32)

    clearance_z = max(
        float(first[2]) + (_FREE_MOTION_MAX_EXTRA_LIFT_M if crosses_center else 0.0),
        float(table_top_z + clearance_m),
    )
    if clearance_z - float(first[2]) > _FREE_MOTION_MAX_EXTRA_LIFT_M + 1.0e-9:
        raise ValueError("required table clearance exceeds the bounded free-motion lift")
    if standby_position is None:
        planar_delta = np.asarray((0.0, 1.0 if side == 0 else -1.0))
    else:
        standby = np.asarray(standby_position, dtype=np.float64).reshape(3)
        if not np.isfinite(standby).all():
            raise ValueError("free-motion standby position must be finite")
        planar_delta = standby[:2] - first[:2]
    planar_distance = float(np.linalg.norm(planar_delta))
    if planar_distance > planar_adjustment_m > 0.0:
        planar_delta *= planar_adjustment_m / planar_distance
    elif planar_adjustment_m == 0.0:
        planar_delta.fill(0.0)
    elif standby_position is None:
        planar_delta *= planar_adjustment_m

    above_release = np.asarray((first[0], first[1], clearance_z))
    if crosses_center and planar_adjustment_m > 0.0:
        forward_distance = 0.72 * planar_adjustment_m
        lateral_distance = np.sqrt(planar_adjustment_m**2 - forward_distance**2)
        forward = np.asarray((first[0] + forward_distance, first[1], clearance_z))
        standby = np.asarray(
            (
                first[0] + forward_distance,
                first[1] + (lateral_distance if side == 0 else -lateral_distance),
                clearance_z,
            )
        )
        path = np.stack((first, forward, standby))
    else:
        standby = np.asarray((*tuple(first[:2] + planar_delta), clearance_z))
        path = np.stack((first, above_release, standby))
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=-1)
    active = lengths > 1.0e-9
    if not np.any(active):
        return np.repeat(first[None], count, axis=0).astype(np.float32)
    path = np.concatenate((path[:-1][active], path[-1:]), axis=0)
    lengths = lengths[active]
    retreat_end_fraction = min(
        0.45,
        _FREE_MOTION_TERMINAL_RETREAT_MAX_FRAMES / (count - 1),
    )
    path_boundaries = np.concatenate(([0.0], np.cumsum(lengths) / np.sum(lengths)))
    boundaries = np.concatenate((path_boundaries * retreat_end_fraction, (1.0,)))
    waypoints = np.concatenate((path, standby[None]), axis=0)
    start_delta = _finite_route_step(start_step)
    result = np.empty((count, 3), dtype=np.float64)
    for index, progress in enumerate(np.linspace(0.0, 1.0, count)):
        segment = min(
            int(np.searchsorted(boundaries[1:], progress, side="right")),
            len(boundaries) - 2,
        )
        width = boundaries[segment + 1] - boundaries[segment]
        fraction = np.clip((progress - boundaries[segment]) / width, 0.0, 1.0)
        duration_frames = width * (count - 1)
        segment_start_step = start_delta * duration_frames if segment == 0 else np.zeros(3)
        result[index] = _quintic_hermite(
            waypoints[segment],
            waypoints[segment + 1],
            fraction,
            segment_start_step,
            np.zeros(3),
        )
    result[0] = first
    result[-1] = standby
    return result.astype(np.float32)


def _placement_transforms(
    particle_q: np.ndarray,
    source_center_xy: np.ndarray,
    table: dict,
    *,
    seed: int,
    count: int,
    margin_m: float,
    max_shift_x_m: float = _DEFAULT_MAX_PLACEMENT_SHIFT_X_M,
    max_shift_y_m: float = _DEFAULT_MAX_PLACEMENT_SHIFT_Y_M,
) -> tuple[list[SpatialTransform], dict]:
    values = np.asarray(particle_q, dtype=np.float32)
    source_center = np.asarray(source_center_xy, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("particle_q must be finite with shape [T, P, 3]")
    if source_center.shape != (2,) or not np.isfinite(source_center).all():
        raise ValueError("source_center_xy must be a finite XY vector")
    max_shift = np.asarray((max_shift_x_m, max_shift_y_m), dtype=np.float32)
    if not np.isfinite(max_shift).all() or np.any(max_shift < 0.0):
        raise ValueError("placement shift limits must be nonnegative and finite")
    base_translation = np.asarray((2.0 * source_center[0], 0.0, 0.0), dtype=np.float32)
    reflected = values @ _REFLECT_X.T + base_translation
    table_min = np.asarray(table["center_xy"], dtype=np.float32) - np.asarray(
        table["half_extents_xy"], dtype=np.float32
    )
    table_max = np.asarray(table["center_xy"], dtype=np.float32) + np.asarray(
        table["half_extents_xy"], dtype=np.float32
    )
    table_lower = table_min + margin_m - reflected[..., :2].min(axis=(0, 1))
    table_upper = table_max - margin_m - reflected[..., :2].max(axis=(0, 1))
    lower = np.maximum(table_lower, -max_shift)
    upper = np.minimum(table_upper, max_shift)
    if np.any(lower > upper):
        raise _PlacementRejected(
            {
                "reason": "cloth_has_no_bounded_feasible_translation",
                "table_translation_lower_xy": table_lower.astype(float).tolist(),
                "table_translation_upper_xy": table_upper.astype(float).tolist(),
                "translation_lower_xy": lower.astype(float).tolist(),
                "translation_upper_xy": upper.astype(float).tolist(),
                "max_placement_shift_xy_m": max_shift.astype(float).tolist(),
            }
        )
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(count)))
    strata = []
    for row in range(side):
        for column in range(side):
            jitter = rng.random(2)
            unit = (np.asarray((column, row), dtype=np.float64) + jitter) / side
            strata.append(lower + unit * (upper - lower))
    preferred = np.clip(np.zeros(2, dtype=np.float32), lower, upper)
    rng.shuffle(strata)
    scale = np.maximum(max_shift, 1.0e-6)
    strata.sort(key=lambda item: float(np.linalg.norm(np.asarray(item) / scale)))
    candidates = [preferred]
    candidates.extend(np.asarray(item, dtype=np.float32) for item in strata)
    unique: list[np.ndarray] = []
    for item in candidates:
        if not any(np.allclose(item, other, atol=1.0e-7, rtol=0.0) for other in unique):
            unique.append(item)
        if len(unique) == count:
            break
    transforms = [
        SpatialTransform(
            _REFLECT_X.copy(),
            base_translation + np.asarray((delta[0], delta[1], 0.0), dtype=np.float32),
        )
        for delta in unique
    ]
    return transforms, {
        "table_translation_lower_xy": table_lower.astype(float).tolist(),
        "table_translation_upper_xy": table_upper.astype(float).tolist(),
        "translation_lower_xy": lower.astype(float).tolist(),
        "translation_upper_xy": upper.astype(float).tolist(),
        "max_placement_shift_xy_m": max_shift.astype(float).tolist(),
        "candidate_count": len(transforms),
        "sampling": "zero_first_seeded_stratified_bounded_xy",
        "local_refinement_trigger_penetration_m": (_LOCAL_REFINEMENT_TRIGGER_PENETRATION_M),
        "local_refinement_trigger_position_error_m": (_LOCAL_REFINEMENT_TRIGGER_POSITION_ERROR_M),
        "local_refinement_steps_m": list(_LOCAL_REFINEMENT_STEPS_M),
    }


def _validate_initial_base_clearance(
    particle_q: np.ndarray,
    base_positions: np.ndarray,
    *,
    min_clearance_m: float,
) -> dict:
    particles = np.asarray(particle_q, dtype=np.float32)
    bases = np.asarray(base_positions, dtype=np.float32)
    if particles.ndim != 2 or particles.shape[1] != 3 or not np.isfinite(particles).all():
        raise ValueError("initial particle_q must be finite with shape [P, 3]")
    if bases.shape != (2, 3) or not np.isfinite(bases).all():
        raise ValueError("base_positions must be finite with shape [2, 3]")
    if not np.isfinite(min_clearance_m) or min_clearance_m <= 0.0:
        raise ValueError("min_clearance_m must be positive and finite")
    distances = np.linalg.norm(particles[:, None, :] - bases[None, :, :], axis=-1)
    minimum_by_base = distances.min(axis=0)
    worst_side = int(np.argmin(minimum_by_base))
    minimum = float(minimum_by_base[worst_side])
    return {
        "valid": minimum >= min_clearance_m,
        "reason": "accepted" if minimum >= min_clearance_m else "cloth_too_close_to_base",
        "minimum_clearance_m": minimum,
        "required_clearance_m": float(min_clearance_m),
        "minimum_by_base_m": minimum_by_base.astype(float).tolist(),
        "worst_base": "left/base_link" if worst_side == 0 else "right/base_link",
    }


def _should_refine(collision_validation: dict) -> bool:
    worst = collision_validation.get("worst_collision")
    if not isinstance(worst, dict):
        return False
    penetration = worst.get("penetration_m")
    return bool(
        isinstance(penetration, (int, float))
        and np.isfinite(penetration)
        and penetration <= _LOCAL_REFINEMENT_TRIGGER_PENETRATION_M
    )


def _should_refine_kinematics(
    validation: dict,
    *,
    position_tolerance_m: float,
) -> bool:
    error = validation.get("max_attach_position_error_m")
    step = validation.get("max_frame_joint_step_rad")
    step_limit = validation.get("joint_jump_limit_rad")
    margin = validation.get("minimum_joint_limit_margin_rad")
    envelope_enforced = bool(validation.get("source_envelope_enforced", True))
    envelope_invalid = envelope_enforced and (
        validation.get("velocity_envelope_valid") is False
        or validation.get("acceleration_envelope_valid") is False
    )
    motion_ratios = (
        ("velocity_envelope_valid", "max_velocity_ratio", "allowed_max_velocity_ratio"),
        (
            "acceleration_envelope_valid",
            "max_acceleration_ratio",
            "allowed_max_acceleration_ratio",
        ),
    )
    motion_close = not envelope_enforced or all(
        validation.get(valid_key) is not False
        or (
            isinstance(validation.get(value_key), (int, float))
            and np.isfinite(validation[value_key])
            and isinstance(validation.get(allowed_key), (int, float))
            and np.isfinite(validation[allowed_key])
            and validation[allowed_key] > 0.0
            and validation[value_key]
            <= _LOCAL_REFINEMENT_MAX_MOTION_RATIO * validation[allowed_key]
        )
        for valid_key, value_key, allowed_key in motion_ratios
    )
    return bool(
        all(
            isinstance(value, (int, float)) and np.isfinite(value)
            for value in (error, step, step_limit, margin, position_tolerance_m)
        )
        and position_tolerance_m > 0.0
        and error <= _LOCAL_REFINEMENT_TRIGGER_POSITION_ERROR_M
        and (error > position_tolerance_m or step > step_limit or envelope_invalid)
        and (
            error > position_tolerance_m
            or step <= _LOCAL_REFINEMENT_MAX_JOINT_STEP_RATIO * step_limit
        )
        and motion_close
        and margin >= -1.0e-5
    )


def _local_refinement_transforms(
    coarse: SpatialTransform,
    source_center_xy: np.ndarray,
    candidate_contract: dict,
) -> list[SpatialTransform]:
    source_center = np.asarray(source_center_xy, dtype=np.float32).reshape(2)
    lower = np.asarray(candidate_contract["translation_lower_xy"], dtype=np.float32)
    upper = np.asarray(candidate_contract["translation_upper_xy"], dtype=np.float32)
    base_translation = np.asarray((2.0 * source_center[0], 0.0), dtype=np.float32)
    coarse_delta = np.asarray(coarse.translation[:2], dtype=np.float32) - base_translation
    y_direction = -1.0 if coarse_delta[1] <= 0.0 else 1.0
    x_direction = -1.0 if coarse_delta[0] <= 0.0 else 1.0
    directions = (
        (0.0, y_direction),
        (0.0, -y_direction),
        (x_direction, 0.0),
        (-x_direction, 0.0),
    )
    offsets = [
        (step * direction[0], step * direction[1])
        for direction in directions
        for step in _LOCAL_REFINEMENT_STEPS_M
    ]
    result = []
    for offset in offsets:
        delta = coarse_delta + np.asarray(offset, dtype=np.float32)
        if np.any(delta < lower) or np.any(delta > upper):
            continue
        translation = np.asarray(coarse.translation, dtype=np.float32).copy()
        translation[:2] += np.asarray(offset, dtype=np.float32)
        result.append(SpatialTransform(coarse.linear.copy(), translation))
    return result


def _validate_motion_limits(
    joint_q: np.ndarray,
    joint_qd: np.ndarray,
    manifest: TrajectoryManifest,
    kinematics,
    *,
    position_errors: np.ndarray,
    position_tolerance_m: float,
    source_max_joint_step_rad: float,
) -> dict:
    profile = robot_profile(PIPER)
    arm = np.asarray(profile.arm_coord_indices, dtype=np.int64)
    arm_dof = np.asarray(profile.arm_dof_indices, dtype=np.int64)
    q = np.asarray(joint_q, dtype=np.float32)[:, arm]
    qd = np.asarray(joint_qd, dtype=np.float32)[:, arm_dof]
    scene = manifest.metadata.get("scene_config", {})
    robot = scene.get("robot", {}) if isinstance(scene, dict) else {}
    velocity_limit = np.asarray(robot.get("arm_joint_velocity_limit", ()), dtype=np.float32)
    acceleration_limit = np.asarray(robot.get("arm_joint_acceleration_limit", ()), dtype=np.float32)
    if velocity_limit.shape != (len(arm),) or acceleration_limit.shape != (len(arm),):
        raise RuntimeError("source scene_config lacks Piper velocity/acceleration limits")
    if (
        not np.isfinite(velocity_limit).all()
        or not np.isfinite(acceleration_limit).all()
        or np.any(velocity_limit <= 0.0)
        or np.any(acceleration_limit <= 0.0)
    ):
        raise RuntimeError("source Piper velocity/acceleration limits must be positive and finite")
    fps = float(manifest.control_fps)
    acceleration = np.diff(qd, axis=0) * fps
    lower = np.asarray(kinematics._arm_lower, dtype=np.float32)
    upper = np.asarray(kinematics._arm_upper, dtype=np.float32)
    max_position_error = float(np.max(position_errors, initial=0.0))
    max_velocity_ratio = float(np.max(np.abs(qd) / velocity_limit, initial=0.0))
    max_acceleration_ratio = float(np.max(np.abs(acceleration) / acceleration_limit, initial=0.0))
    velocity_ratios = np.abs(qd) / velocity_limit
    acceleration_ratios = np.abs(acceleration) / acceleration_limit
    velocity_index = (
        np.unravel_index(int(np.argmax(velocity_ratios)), velocity_ratios.shape)
        if velocity_ratios.size
        else (0, 0)
    )
    acceleration_index = (
        np.unravel_index(int(np.argmax(acceleration_ratios)), acceleration_ratios.shape)
        if acceleration_ratios.size
        else (0, 0)
    )
    max_joint_step = _maximum_arm_joint_step(joint_q)
    joint_jump_limit = max(0.25, 2.0 * float(source_max_joint_step_rad))
    joint_margin = np.minimum(q - lower, upper - q)
    minimum_joint_margin = float(np.min(joint_margin, initial=np.inf))
    valid = bool(
        max_position_error <= position_tolerance_m
        and max_joint_step <= joint_jump_limit
        and minimum_joint_margin >= -1.0e-5
    )
    position_error_index = np.unravel_index(
        int(np.argmax(position_errors)), np.asarray(position_errors).shape
    )
    joint_steps = np.abs(np.diff(q, axis=0))
    joint_step_index = (
        np.unravel_index(int(np.argmax(joint_steps)), joint_steps.shape)
        if joint_steps.size
        else (0, 0)
    )
    return {
        "valid": valid,
        "reason": "motion_limits" if not valid else "accepted",
        "max_attach_position_error_m": max_position_error,
        "max_attach_position_error_frame": int(position_error_index[0]),
        "max_attach_position_error_side": (
            "left" if int(position_error_index[1]) == 0 else "right"
        ),
        "max_velocity_ratio": max_velocity_ratio,
        "max_velocity_frame": int(velocity_index[0]),
        "max_velocity_arm_joint_index": int(velocity_index[1]),
        "max_acceleration_ratio": max_acceleration_ratio,
        "max_acceleration_end_frame": int(acceleration_index[0]) + 1,
        "max_acceleration_arm_joint_index": int(acceleration_index[1]),
        "planner_velocity_acceleration_limits_are_diagnostic": True,
        "max_frame_joint_step_rad": max_joint_step,
        "max_frame_joint_step_end_frame": int(joint_step_index[0]) + 1,
        "max_frame_joint_step_arm_joint_index": int(joint_step_index[1]),
        "joint_jump_limit_rad": joint_jump_limit,
        "minimum_joint_limit_margin_rad": minimum_joint_margin,
    }


def _motion_envelope(joint_qd: np.ndarray, manifest: TrajectoryManifest) -> dict:
    profile = robot_profile(PIPER)
    arm_dof = np.asarray(profile.arm_dof_indices, dtype=np.int64)
    values = np.asarray(joint_qd, dtype=np.float32)[:, arm_dof]
    scene = manifest.metadata.get("scene_config", {})
    robot = scene.get("robot", {}) if isinstance(scene, dict) else {}
    velocity_limit = np.asarray(robot.get("arm_joint_velocity_limit", ()), dtype=np.float32)
    acceleration_limit = np.asarray(robot.get("arm_joint_acceleration_limit", ()), dtype=np.float32)
    if velocity_limit.shape != (values.shape[1],) or acceleration_limit.shape != (
        values.shape[1],
    ):
        raise RuntimeError("source scene_config lacks Piper velocity/acceleration limits")
    acceleration = np.diff(values, axis=0) * float(manifest.control_fps)
    return {
        "max_velocity_ratio": float(np.max(np.abs(values) / velocity_limit, initial=0.0)),
        "max_acceleration_ratio": float(
            np.max(np.abs(acceleration) / acceleration_limit, initial=0.0)
        ),
    }


def _enforce_motion_envelope(validation: dict, source_envelope: dict) -> dict:
    velocity_limit = (1.0 + _MOTION_ENVELOPE_RELATIVE_TOLERANCE) * max(
        1.0, float(source_envelope["max_velocity_ratio"])
    )
    acceleration_limit = (1.0 + _MOTION_ENVELOPE_RELATIVE_TOLERANCE) * max(
        1.0, float(source_envelope["max_acceleration_ratio"])
    )
    velocity_valid = float(validation["max_velocity_ratio"]) <= velocity_limit
    acceleration_valid = float(validation["max_acceleration_ratio"]) <= acceleration_limit
    valid = bool(validation["valid"] and velocity_valid and acceleration_valid)
    return {
        **validation,
        "valid": valid,
        "reason": validation["reason"],
        "source_envelope_enforced": True,
        "source_max_velocity_ratio": float(source_envelope["max_velocity_ratio"]),
        "source_max_acceleration_ratio": float(source_envelope["max_acceleration_ratio"]),
        "relative_tolerance": _MOTION_ENVELOPE_RELATIVE_TOLERANCE,
        "allowed_max_velocity_ratio": velocity_limit,
        "allowed_max_acceleration_ratio": acceleration_limit,
        "velocity_envelope_valid": velocity_valid,
        "acceleration_envelope_valid": acceleration_valid,
        **({"reason": "motion_envelope"} if validation["valid"] and not valid else {}),
    }


def _annotate_motion_envelope(validation: dict, source_envelope: dict) -> dict:
    """Record reference-path motion ratios without making them a hard gate.

    The IK path is guidance for the action-only dynamics reconstruction.  Its
    finite-difference velocity and acceleration can be noisy even when the
    attached Cartesian targets and joint continuity are valid.  The executed
    dynamics trajectory is checked separately with ``_enforce_motion_envelope``.
    """

    annotated = _enforce_motion_envelope(validation, source_envelope)
    return {
        **annotated,
        "valid": bool(validation["valid"]),
        "reason": validation["reason"],
        "source_envelope_enforced": False,
        "source_envelope_diagnostic": True,
    }


def _select_smoothed_motion(
    motion,
    *,
    attached: np.ndarray,
    manifest: TrajectoryManifest,
    kinematics,
    profile,
    position_tolerance_m: float,
    source_max_joint_step_rad: float,
    source_motion_envelope: dict,
):
    """Use the least smoothing that satisfies all existing hard checks.

    Newton 1.5 provides per-frame IK but no trajectory objective.  A short
    symmetric binomial filter removes frame-local IK chatter while Cartesian
    error is recomputed after every pass.  No tolerance is relaxed, and the
    unsmoothed path remains the diagnostic result if every filtered path fails.
    """

    fps = float(manifest.control_fps)
    attached_mask = np.asarray(attached, dtype=bool)
    if attached_mask.shape != motion.attach_errors.shape:
        raise ValueError("attached mask must match motion attach errors")

    def validate(candidate):
        return _annotate_motion_envelope(
            _validate_motion_limits(
                candidate.joint_q,
                _robot_velocity(candidate.joint_q, profile, fps),
                manifest,
                kinematics,
                position_errors=np.where(attached_mask, candidate.attach_errors, 0.0),
                position_tolerance_m=position_tolerance_m,
                source_max_joint_step_rad=source_max_joint_step_rad,
            ),
            source_motion_envelope,
        )

    raw_validation = validate(motion)
    attempts = [_smoothing_attempt_summary(0, raw_validation)]
    if raw_validation["valid"] or len(motion.joint_q) < 5:
        return motion, {**raw_validation, "joint_smoothing_attempts": attempts}

    arm = np.asarray(profile.arm_coord_indices, dtype=np.int64)
    arms = arm.reshape(2, -1)
    original = np.asarray(motion.joint_q, dtype=np.float32)
    filtered = original[:, arm].copy()
    previous_passes = 0
    detached_attempts = []
    global_result = None
    for passes in _JOINT_SMOOTHING_PASSES:
        for _ in range(passes - previous_passes):
            padded = np.pad(filtered, ((2, 2), (0, 0)), mode="edge")
            filtered = sum(
                weight * padded[offset : offset + len(filtered)]
                for offset, weight in enumerate(_JOINT_SMOOTHING_KERNEL)
            ).astype(np.float32)
        previous_passes = passes
        # The recorded first and last configurations are safety anchors.  In
        # normal trajectories they lie in stationary holds, so retaining them
        # does not reintroduce an endpoint impulse.
        filtered[0] = original[0, arm]
        filtered[-1] = original[-1, arm]
        detached_joint_q = original.copy()
        for side in range(2):
            free = ~attached_mask[:, side]
            detached_joint_q[np.ix_(free, arms[side])] = filtered[
                np.ix_(free, np.arange(side * arms.shape[1], (side + 1) * arms.shape[1]))
            ]
        detached_attach_poses = kinematics.attach_poses_batch(detached_joint_q)
        detached_candidate = replace(
            motion,
            joint_q=detached_joint_q,
            attach_poses=detached_attach_poses,
            attach_errors=np.linalg.norm(
                detached_attach_poses[..., :3] - motion.target_attach_positions,
                axis=-1,
            ).astype(np.float32),
        )
        detached_validation = validate(detached_candidate)
        detached_attempts.append(_smoothing_attempt_summary(passes, detached_validation))
        if detached_validation["valid"]:
            return detached_candidate, {
                **detached_validation,
                "detached_joint_smoothing_passes": passes,
                "joint_smoothing_kernel": _JOINT_SMOOTHING_KERNEL.astype(float).tolist(),
                "detached_joint_smoothing_attempts": detached_attempts,
                "joint_smoothing_attempts": attempts,
            }
        joint_q = original.copy()
        joint_q[:, arm] = filtered
        attach_poses = kinematics.attach_poses_batch(joint_q)
        attach_errors = np.linalg.norm(
            attach_poses[..., :3] - motion.target_attach_positions,
            axis=-1,
        ).astype(np.float32)
        candidate = replace(
            motion,
            joint_q=joint_q,
            attach_poses=attach_poses,
            attach_errors=attach_errors,
        )
        candidate_validation = validate(candidate)
        attempts.append(_smoothing_attempt_summary(passes, candidate_validation))
        if candidate_validation["valid"] and global_result is None:
            global_result = (
                candidate,
                {
                    **candidate_validation,
                    "joint_smoothing_passes": passes,
                    "joint_smoothing_kernel": _JOINT_SMOOTHING_KERNEL.astype(float).tolist(),
                    "joint_smoothing_attempts": list(attempts),
                },
            )
    if global_result is not None:
        return global_result

    interpolated = _interpolate_detached_joint_motion(
        motion,
        attached_mask,
        kinematics=kinematics,
        profile=profile,
    )
    interpolated_validation = validate(interpolated)
    if interpolated_validation["valid"]:
        return interpolated, {
            **interpolated_validation,
            "detached_joint_interpolation": True,
            "detached_joint_smoothing_attempts": detached_attempts,
            "joint_smoothing_attempts": attempts,
        }
    return motion, {
        **raw_validation,
        "detached_joint_smoothing_attempts": detached_attempts,
        "joint_smoothing_attempts": attempts,
    }


def _interpolate_detached_joint_motion(motion, attached, *, kinematics, profile):
    """Replace free-frame IK branches by bounded joint-space transitions.

    Attached configurations are immutable task anchors.  For a terminal free
    run, the endpoint is the first frame from which the Cartesian retreat has
    reached its final standby position; holding that solved configuration
    avoids stretching a short retreat across the remainder of the episode.
    Geometry is intentionally not accepted here: the caller still runs the
    normal frame-by-frame collision and clearance gates.
    """

    attached_mask = np.asarray(attached, dtype=bool)
    original = np.asarray(motion.joint_q, dtype=np.float32)
    targets = np.asarray(motion.target_attach_positions, dtype=np.float32)
    if attached_mask.shape != targets.shape[:2]:
        raise ValueError("attached mask must match target attach positions")
    if original.ndim != 2 or len(original) != len(targets):
        raise ValueError("joint motion must share the target timeline")

    joint_q = original.copy()
    arms = np.asarray(profile.arm_coord_indices, dtype=np.int64).reshape(2, -1)
    for side in range(2):
        for free_start, free_end in _true_runs(~attached_mask[:, side]):
            anchor_start = max(0, free_start - 1)
            if free_end == len(targets) - 1:
                final_target = targets[-1, side]
                at_final = np.all(
                    np.isclose(targets[:, side], final_target, atol=1.0e-6, rtol=0.0),
                    axis=-1,
                )
                stable_to_end = np.logical_and.accumulate(at_final[::-1])[::-1]
                settled = np.flatnonzero(
                    stable_to_end & (np.arange(len(targets)) >= free_start)
                )
                anchor_end = int(settled[0]) if len(settled) else free_end
                anchor_end = max(anchor_start + 1, anchor_end)
            else:
                anchor_end = min(len(targets) - 1, free_end + 1)
            count = anchor_end - anchor_start + 1
            phase = np.linspace(0.0, 1.0, count, dtype=np.float64)
            blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
            start_q = original[anchor_start, arms[side]].astype(np.float64)
            end_q = original[anchor_end, arms[side]].astype(np.float64)
            joint_q[anchor_start : anchor_end + 1, arms[side]] = (
                start_q[None] + blend[:, None] * (end_q - start_q)[None]
            ).astype(np.float32)
            if free_end == len(targets) - 1:
                joint_q[anchor_end:, arms[side]] = end_q.astype(np.float32)

    attach_poses = kinematics.attach_poses_batch(joint_q)
    attach_errors = np.linalg.norm(
        attach_poses[..., :3] - targets,
        axis=-1,
    ).astype(np.float32)
    return replace(
        motion,
        joint_q=joint_q,
        attach_poses=attach_poses,
        attach_errors=attach_errors,
    )


def _smoothing_attempt_summary(passes: int, validation: dict) -> dict:
    return {
        "passes": passes,
        "valid": bool(validation["valid"]),
        "reason": validation["reason"],
        "max_attach_position_error_m": float(validation["max_attach_position_error_m"]),
        "max_velocity_ratio": float(validation["max_velocity_ratio"]),
        "max_acceleration_ratio": float(validation["max_acceleration_ratio"]),
        "max_frame_joint_step_rad": float(validation["max_frame_joint_step_rad"]),
    }


def _maximum_arm_joint_step(joint_q: np.ndarray) -> float:
    arm = np.asarray(robot_profile(PIPER).arm_coord_indices, dtype=np.int64)
    values = np.asarray(joint_q, dtype=np.float32)[:, arm]
    if len(values) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(values, axis=0)), initial=0.0))


def _ik_branch_initializations(
    profile,
    collision_validation: dict,
    inter_arm_clearance: dict | None = None,
):
    """Return deterministic wrist seeds only for the side implicated by collision."""

    worst_collision = collision_validation.get("worst_collision") or {}
    worst_clearance = (inter_arm_clearance or {}).get("worst_clearance") or {}
    bodies = tuple(
        str(value)
        for value in (*worst_collision.get("body_pair", ()), *worst_clearance.get("body_pair", ()))
    )
    arms = np.asarray(profile.arm_coord_indices, dtype=np.int64).reshape(2, -1)
    home = np.asarray(profile.home_joint_q, dtype=np.float32)
    for side_index, side in enumerate(("left", "right")):
        if not any(body.startswith(f"{side}/") for body in bodies):
            continue
        wrist_coordinate = int(arms[side_index, 3])
        for offset in _IK_BRANCH_WRIST_OFFSETS_RAD:
            seed = home.copy()
            seed[wrist_coordinate] += offset
            yield (
                {
                    "label": f"{side}_wrist_{offset:+.1f}",
                    "side": side,
                    "wrist_offset_rad": float(offset),
                },
                seed,
            )


def _ik_naturalness_score(joint_q: np.ndarray, profile) -> dict:
    """Score a collision-safe IK branch by continuity and distance from home."""

    arm_indices = np.asarray(profile.arm_coord_indices, dtype=np.int64)
    values = np.asarray(joint_q, dtype=np.float64)[:, arm_indices]
    home = np.asarray(profile.home_joint_q, dtype=np.float64)[arm_indices]
    steps = np.diff(values, axis=0)
    max_step = float(np.max(np.abs(steps), initial=0.0))
    rms_step = float(np.sqrt(np.mean(np.square(steps)))) if len(steps) else 0.0
    rms_home_offset = float(np.sqrt(np.mean(np.square(values - home))))
    cost = max_step + rms_step + 0.05 * rms_home_offset
    return {
        "schema": "joint_continuity_and_home_posture_v1",
        "cost": float(cost),
        "max_frame_joint_step_rad": max_step,
        "rms_frame_joint_step_rad": rms_step,
        "rms_home_offset_rad": rms_home_offset,
    }


def _ik_branch_attempt_diagnostic(
    branch: dict,
    limits: dict,
    collision: dict | None,
    inter_arm_clearance: dict | None = None,
) -> dict:
    return {
        **branch,
        "motion_valid": bool(limits.get("valid", False)),
        "max_attach_position_error_m": limits.get("max_attach_position_error_m"),
        "max_frame_joint_step_rad": limits.get("max_frame_joint_step_rad"),
        "collision_valid": None if collision is None else bool(collision.get("valid", False)),
        "worst_collision": None if collision is None else collision.get("worst_collision"),
        "inter_arm_clearance_valid": (
            None
            if inter_arm_clearance is None
            else bool(inter_arm_clearance.get("valid", False))
        ),
        "worst_inter_arm_clearance": (
            None
            if inter_arm_clearance is None
            else inter_arm_clearance.get("worst_clearance")
        ),
    }


def _geometric_surface_separation(
    point0: np.ndarray,
    point1: np.ndarray,
    normal: np.ndarray,
    contact_margin0: float,
    contact_margin1: float,
    shape_margin0: float,
    shape_margin1: float,
) -> tuple[float, float]:
    """Return physical and solver-effective separations for a Newton contact.

    Newton's contact margin contains both geometry thickness (for example a
    capsule radius) and an artificial shape margin.  The physical collision
    surface therefore retains the former and removes only the latter.
    """

    support_separation = float(np.dot(normal, point1 - point0))
    radius0 = max(0.0, float(contact_margin0) - float(shape_margin0))
    radius1 = max(0.0, float(contact_margin1) - float(shape_margin1))
    geometric = support_separation - radius0 - radius1
    solver_effective = support_separation - float(contact_margin0) - float(contact_margin1)
    return geometric, solver_effective


class _RigidCollisionValidator:
    """Reject penetrations beyond the workstation's home fixture baseline."""

    def __init__(self, assets_root: Path, *, device: str, batch_size: int = 16):
        from clothdojo.sim.runtime import ShirtFlattenRuntime
        from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

        self.profile = robot_profile(PIPER, assets_root=assets_root)
        scene = build_scene(
            assets_root,
            # Inflated arm margins used by clearance validation can produce
            # more than the simulation-oriented default of 4096 contacts for
            # a 16-world validation batch.  Truncating those contacts would
            # make the safety check unsound, so give this offline validator a
            # deliberately larger buffer.
            spec=ShirtFlattenSpec(world_count=batch_size, rigid_contact_max=16384),
            device=device,
            robot_id=PIPER,
            visuals=False,
            include_cloth=False,
            robot_cloth_collision=False,
            robot_rigid_collision=True,
        )
        self.runtime = ShirtFlattenRuntime(scene, use_cuda_graph=False)
        self.batch_size = batch_size
        self.body_count = len(self.profile.body_names)
        self.shape_count = int(self.runtime.model.shape_count) // batch_size
        self.shape_body = np.asarray(self.runtime.model.shape_body.numpy(), dtype=np.int32)
        self.original_shape_margin = np.asarray(
            self.runtime.model.shape_margin.numpy(), dtype=np.float32
        )
        self.moving_bodies = {
            index
            for index, name in enumerate(self.profile.body_names)
            if name != "world_base" and not name.endswith("/base_link")
        }
        home_q = np.asarray(self.profile.home_joint_q, dtype=np.float32)
        home_qd = np.zeros(self.profile.joint_dof_count, dtype=np.float32)
        self._restore(
            np.repeat(home_q[None], batch_size, axis=0),
            np.repeat(home_qd[None], batch_size, axis=0),
        )
        self.baseline = self._penetrations()

    def validate(self, joint_q: np.ndarray) -> dict:
        q = np.asarray(joint_q, dtype=np.float32)
        zero = np.zeros((self.batch_size, self.profile.joint_dof_count), dtype=np.float32)
        worst = None
        for start in range(0, len(q), self.batch_size):
            count = min(self.batch_size, len(q) - start)
            batch = np.repeat(q[start : start + 1], self.batch_size, axis=0)
            batch[:count] = q[start : start + count]
            self._restore(batch, zero)
            for item in self._penetrations():
                if item["world"] >= count:
                    continue
                key = tuple(item["shape_pair"])
                baseline = self.baseline.get(key, 0.0)
                if item["penetration_m"] <= baseline + _COLLISION_PENETRATION_TOLERANCE_M:
                    continue
                candidate = {**item, "frame": start + item["world"], "baseline_m": baseline}
                if worst is None or candidate["penetration_m"] > worst["penetration_m"]:
                    worst = candidate
        return {
            "valid": worst is None,
            "reason": "collision" if worst is not None else "accepted",
            "penetration_tolerance_m": _COLLISION_PENETRATION_TOLERANCE_M,
            "worst_collision": worst,
        }

    def validate_inter_arm_clearance(
        self,
        joint_q: np.ndarray,
        *,
        min_clearance_m: float,
        temporal_subsamples: int = 2,
    ) -> dict:
        """Validate a physical cross-arm surface gap with inflated proxy envelopes."""

        if not np.isfinite(min_clearance_m) or min_clearance_m <= 0.0:
            raise ValueError("inter-arm clearance must be positive and finite")
        if temporal_subsamples < 1:
            raise ValueError("temporal_subsamples must be positive")
        q = _subsample_joint_path(joint_q, temporal_subsamples)
        expanded = self.original_shape_margin.copy()
        for shape, body in enumerate(self.shape_body):
            if body < 0:
                continue
            name = self.profile.body_names[body % self.body_count]
            if name.startswith("left/") or name.startswith("right/"):
                expanded[shape] = max(expanded[shape], 0.5 * float(min_clearance_m))
        self.runtime.model.shape_margin.assign(expanded)
        zero = np.zeros((self.batch_size, self.profile.joint_dof_count), dtype=np.float32)
        worst = None
        try:
            for start in range(0, len(q), self.batch_size):
                count = min(self.batch_size, len(q) - start)
                batch = np.repeat(q[start : start + 1], self.batch_size, axis=0)
                batch[:count] = q[start : start + count]
                self._restore(batch, zero)
                for item in self._inter_arm_clearances(min_clearance_m):
                    if item["world"] >= count:
                        continue
                    candidate = {
                        **item,
                        "sampled_frame": start + item["world"],
                        "source_frame": (start + item["world"]) / temporal_subsamples,
                    }
                    if worst is None or candidate["shortfall_m"] > worst["shortfall_m"]:
                        worst = candidate
        finally:
            self.runtime.model.shape_margin.assign(self.original_shape_margin)
        return {
            "valid": worst is None,
            "reason": "inter_arm_clearance" if worst is not None else "accepted",
            "required_clearance_m": float(min_clearance_m),
            "temporal_subsamples": int(temporal_subsamples),
            "worst_clearance": worst,
        }

    def _restore(self, joint_q: np.ndarray, joint_qd: np.ndarray) -> None:
        self.runtime.restore_rigid_state_batch(joint_q, joint_qd)
        self.runtime.rigid_collision.collide(
            self.runtime.state_0,
            self.runtime.rigid_contacts,
        )

    def _penetrations(self):
        contacts = self.runtime.rigid_contacts
        raw_count = int(contacts.rigid_contact_count.numpy()[0])
        capacity = int(contacts.rigid_contact_max)
        if raw_count > capacity:
            raise RuntimeError(
                f"rigid collision contact buffer overflowed: {raw_count} > {capacity}"
            )
        count = raw_count
        if count == 0:
            return {} if not hasattr(self, "baseline") else []
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        point0 = contacts.rigid_contact_point0.numpy()[:count]
        point1 = contacts.rigid_contact_point1.numpy()[:count]
        normal = contacts.rigid_contact_normal.numpy()[:count]
        margin0 = contacts.rigid_contact_margin0.numpy()[:count]
        margin1 = contacts.rigid_contact_margin1.numpy()[:count]
        shape_body = self.runtime.model.shape_body.numpy()
        shape_margin = self.runtime.model.shape_margin.numpy()
        body_q = self.runtime.state_0.body_q.numpy()
        values = []
        baseline_mode = not hasattr(self, "baseline")
        baseline = {}
        for slot in range(count):
            s0, s1 = int(shape0[slot]), int(shape1[slot])
            if s0 < 0 or s1 < 0:
                continue
            b0, b1 = int(shape_body[s0]), int(shape_body[s1])
            local_bodies = tuple(body % self.body_count if body >= 0 else -1 for body in (b0, b1))
            if not any(body in self.moving_bodies for body in local_bodies):
                continue
            world = (
                b0 // self.body_count
                if b0 >= 0
                else b1 // self.body_count
                if b1 >= 0
                else s0 // self.shape_count
            )
            p0 = _world_point(body_q, b0, point0[slot])
            p1 = _world_point(body_q, b1, point1[slot])
            separation, solver_separation = _geometric_surface_separation(
                p0,
                p1,
                normal[slot],
                margin0[slot],
                margin1[slot],
                shape_margin[s0],
                shape_margin[s1],
            )
            if separation >= 0.0:
                continue
            key = tuple(sorted((s0 % self.shape_count, s1 % self.shape_count)))
            penetration = -separation
            if baseline_mode:
                baseline[key] = max(baseline.get(key, 0.0), penetration)
            else:
                values.append(
                    {
                        "world": int(world),
                        "shape_pair": list(key),
                        "body_pair": [
                            self.profile.body_names[body] if body >= 0 else "static"
                            for body in local_bodies
                        ],
                        "penetration_m": penetration,
                        "solver_effective_penetration_m": max(0.0, -solver_separation),
                        "artificial_shape_margin_sum_m": float(
                            shape_margin[s0] + shape_margin[s1]
                        ),
                    }
                )
        return baseline if baseline_mode else values

    def _inter_arm_clearances(self, required_clearance_m: float) -> list[dict]:
        contacts = self.runtime.rigid_contacts
        count = int(contacts.rigid_contact_count.numpy()[0])
        if count > int(contacts.rigid_contact_max):
            raise RuntimeError("inter-arm clearance contact buffer overflowed")
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        point0 = contacts.rigid_contact_point0.numpy()[:count]
        point1 = contacts.rigid_contact_point1.numpy()[:count]
        normal = contacts.rigid_contact_normal.numpy()[:count]
        margin0 = contacts.rigid_contact_margin0.numpy()[:count]
        margin1 = contacts.rigid_contact_margin1.numpy()[:count]
        shape_margin = self.runtime.model.shape_margin.numpy()
        body_q = self.runtime.state_0.body_q.numpy()
        values = []
        for slot in range(count):
            s0, s1 = int(shape0[slot]), int(shape1[slot])
            if s0 < 0 or s1 < 0:
                continue
            b0, b1 = int(self.shape_body[s0]), int(self.shape_body[s1])
            if b0 < 0 or b1 < 0:
                continue
            local0, local1 = b0 % self.body_count, b1 % self.body_count
            name0, name1 = self.profile.body_names[local0], self.profile.body_names[local1]
            cross_arm = (name0.startswith("left/") and name1.startswith("right/")) or (
                name0.startswith("right/") and name1.startswith("left/")
            )
            if not cross_arm:
                continue
            p0 = _world_point(body_q, b0, point0[slot])
            p1 = _world_point(body_q, b1, point1[slot])
            clearance, _solver_separation = _geometric_surface_separation(
                p0,
                p1,
                normal[slot],
                margin0[slot],
                margin1[slot],
                shape_margin[s0],
                shape_margin[s1],
            )
            shortfall = float(required_clearance_m) - clearance
            if shortfall <= 0.0:
                continue
            world = b0 // self.body_count
            values.append(
                {
                    "world": int(world),
                    "shape_pair": [s0 % self.shape_count, s1 % self.shape_count],
                    "body_pair": [name0, name1],
                    "clearance_m": float(clearance),
                    "shortfall_m": float(shortfall),
                }
            )
        return values


def _world_point(body_q: np.ndarray, body: int, point: np.ndarray) -> np.ndarray:
    if body < 0:
        return np.asarray(point, dtype=np.float32)
    pose = body_q[body]
    return pose[:3] + _quat_rotate(pose[3:7], point)


def _subsample_joint_path(joint_q: np.ndarray, subdivisions: int) -> np.ndarray:
    q = np.asarray(joint_q, dtype=np.float32)
    if q.ndim != 2 or len(q) == 0:
        raise ValueError("joint path must have nonempty shape [N, Q]")
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("joint path subdivisions must be a positive integer")
    if len(q) == 1 or subdivisions == 1:
        return q.copy()
    result = np.empty(((len(q) - 1) * subdivisions + 1, q.shape[1]), dtype=np.float32)
    for frame in range(len(q) - 1):
        for slot in range(subdivisions):
            fraction = slot / subdivisions
            result[frame * subdivisions + slot] = q[frame] + fraction * (q[frame + 1] - q[frame])
    result[-1] = q[-1]
    return result


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    v = np.asarray(vector, dtype=np.float32)
    xyz = q[:3]
    return v + 2.0 * np.cross(xyz, np.cross(xyz, v) + q[3] * v)


def _table_contract(manifest: TrajectoryManifest) -> dict:
    scene = manifest.metadata.get("scene_config")
    table = scene.get("table") if isinstance(scene, dict) else None
    if not isinstance(table, dict):
        raise RuntimeError("source scene_config table contract is missing")
    center = np.asarray(table.get("pos_xy"), dtype=np.float64)
    size = np.asarray(table.get("size"), dtype=np.float64)
    z_base = table.get("z_base")
    if (
        center.shape != (2,)
        or size.shape != (3,)
        or not isinstance(z_base, (int, float))
        or not np.isfinite(center).all()
        or not np.isfinite(size).all()
        or not np.isfinite(z_base)
    ):
        raise RuntimeError("source scene_config table contract is invalid")
    if np.any(size <= 0.0):
        raise RuntimeError("source table half extents must be positive")
    if not np.isclose(center[1], 0.0, atol=1.0e-7, rtol=0.0):
        raise RuntimeError("bilateral mirror requires the source table to be centered at y=0")
    return {
        "center_xy": center.astype(float).tolist(),
        "half_extents_xy": size[:2].astype(float).tolist(),
        "top_z": float(z_base + size[2]),
    }


def _validate_table_bounds(
    particle_q: np.ndarray,
    table: dict,
    *,
    margin_m: float,
) -> dict:
    values = np.asarray(particle_q, dtype=np.float32)
    lower = np.asarray(table["center_xy"]) - np.asarray(table["half_extents_xy"]) + margin_m
    upper = np.asarray(table["center_xy"]) + np.asarray(table["half_extents_xy"]) - margin_m
    minimum = values[..., :2].min(axis=(0, 1))
    maximum = values[..., :2].max(axis=(0, 1))
    return {
        "valid": bool(np.all(minimum >= lower) and np.all(maximum <= upper)),
        "minimum_xy": minimum.astype(float).tolist(),
        "maximum_xy": maximum.astype(float).tolist(),
        "allowed_minimum_xy": lower.astype(float).tolist(),
        "allowed_maximum_xy": upper.astype(float).tolist(),
    }


def _validate_source(reader: TrajectoryReader, source: Path) -> None:
    if reader.manifest.scene.robot_id != PIPER:
        raise ValueError(f"front/back augmentation requires Piper: {source}")
    if reader.manifest.scene.task_name != SHIRT_FOLD:
        raise ValueError(f"front/back augmentation requires shirt_fold: {source}")
    if reader.manifest.action_count < 1:
        raise ValueError(f"front/back augmentation requires actions: {source}")
    asset_id = reader.manifest.metadata.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise RuntimeError(f"front/back source asset_id is missing: {source}")


def _source_trajectories(source_root: Path) -> list[Path]:
    root = source_root / "trajectories"
    if not root.is_dir():
        raise RuntimeError("front/back source must contain trajectories/")
    trajectories = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            trajectories.append(manifest_path.parent)
    if not trajectories:
        raise RuntimeError("front/back source has no complete trajectories")
    return trajectories


def _copy_annotations(
    source_root: Path,
    output: Path,
    source_id: str,
    *,
    include_front_back: bool = True,
) -> None:
    source = source_root / "trajectories" / "annotations" / f"{source_id}.segments.json"
    if not source.is_file():
        return
    content = source.read_bytes()
    trajectory_ids = [source_id, f"{source_id}{MIRROR_SUFFIX}"]
    if include_front_back:
        trajectory_ids.extend(
            (
                f"{source_id}{FRONT_BACK_SUFFIX}",
                f"{source_id}{FRONT_BACK_SUFFIX}{MIRROR_SUFFIX}",
            )
        )
    for trajectory_id in trajectory_ids:
        (output / f"{trajectory_id}.segments.json").write_bytes(content)


def _trajectory_seed(seed: int, trajectory_id: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(seed).to_bytes(16, "little", signed=True))
    digest.update(b"\0")
    digest.update(trajectory_id.encode("utf-8"))
    return int.from_bytes(digest.digest(), "little")


def _source_snapshot_sha256(records: list[dict], rejections: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in records:
        digest.update(item["source_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["source_manifest_sha256"].encode("ascii"))
        digest.update(b"\n")
    for item in rejections:
        digest.update(item["source_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["source_manifest_sha256"].encode("ascii"))
        digest.update(b"\0rejected\n")
    return digest.hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as file:
        np.savez_compressed(file, **arrays)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
