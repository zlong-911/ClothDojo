"""Dynamics-consistent counterfactual replay from Piper to ALOHA."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import numpy as np

from clothdojo.data.retarget_dynamics import (
    PREGRASP_APPROACH_TOLERANCE_RAD,
    AlohaDynamicsRetargeter,
    DynamicsRetargetResult,
)
from clothdojo.robots import (
    ALOHA2,
    PANDA,
    PIPER,
    UR5E_2F85,
    X5,
    robot_profile,
)
from clothdojo.tasks import SHIRT_FLATTEN
from clothdojo.trajectory import TrajectoryManifest, TrajectoryReader

RETARGET_COLLECTION_SCHEMA = "clothdojo_retarget_collection"
RETARGET_SCHEMA_VERSION = 1
ALOHA_SOURCE_REVISION = "da76818e269b82289eba39808e2fb91d679d6994"
ALOHA_MJCF_SHA256 = "8f8fe3a476ebbd9c7aa1b6940dda8f01da9b8b5adde2d52a614862c9069c2e7b"
ALOHA_FRAME_MJCF_SHA256 = "6dc4abb19b1275c1ea413aab7991af99645170e96323cc75da663fe5d20b2445"
ALOHA_LICENSE_SHA256 = "21868447315ac2b10e592d08958c1035b19a32cf39e9165b23033f68390098cb"
PANDA_MJCF_SHA256 = "96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273"
PANDA_LICENSE_SHA256 = "a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9"
UR5E_2F85_MJCF_SHA256 = "d0399d65466c742fc5a6241a48feb12e034b47c77803c12b0021d560e02870bb"
UR5E_LICENSE_SHA256 = "5ec71ccf66c8d03261448f2441a586765b97f2248c860a1ae19689fb1c45cee6"
ROBOTIQ_2F85_LICENSE_SHA256 = "3d5edd18d1e22a17666e9bb84270171ed5ac90b192d24e0177fd171708ea2316"
X5_SOURCE_REVISION = "cfb06d1dadcdf03ccc923102b0e3f3b0a68dfc43"
X5_MJCF_SHA256 = "acfd2a391e2871fcb91c18d8a8049e3e1687b625e317e10ce9016b072d13ac09"
X5_ROBODOJO_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
X5_SDK_LICENSE_SHA256 = "ad851f58a4ed28b01d815d74c7c54d18042760b122c89f7e6b4dc7e0126492f6"

_APPROACH_TOLERANCE_RAD = PREGRASP_APPROACH_TOLERANCE_RAD
_IK_POSITION_WEIGHT = 100.0
_IK_ROTATION_WEIGHT = 3.0
_ALOHA_PROFILE = robot_profile(ALOHA2)
_PIPER_PROFILE = robot_profile(PIPER)
_SOURCE_EE_BODY_INDICES = np.asarray(
    tuple(
        _PIPER_PROFILE.body_names.index(frame.parent_body)
        for frame in _PIPER_PROFILE.attach_frames
    ),
    dtype=np.int64,
)
_ALOHA_HOME = np.asarray(_ALOHA_PROFILE.home_joint_q, dtype=np.float32)
_ALOHA_ARM_INDICES = np.asarray(_ALOHA_PROFILE.arm_coord_indices, dtype=np.int64)
_ALOHA_GRIPPER_INDICES = np.asarray(_ALOHA_PROFILE.gripper_coord_indices, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class _AssetPins:
    source: str
    revision: str
    mjcf_sha256: str
    licenses: tuple[tuple[str, str], ...]
    installation_sha256: str | None = None


_TARGET_ASSET_PINS = {
    ALOHA2: _AssetPins(
        "google-deepmind/mujoco_menagerie",
        ALOHA_SOURCE_REVISION,
        ALOHA_MJCF_SHA256,
        (("LICENSE", ALOHA_LICENSE_SHA256),),
        ALOHA_FRAME_MJCF_SHA256,
    ),
    PANDA: _AssetPins(
        "google-deepmind/mujoco_menagerie",
        ALOHA_SOURCE_REVISION,
        PANDA_MJCF_SHA256,
        (("LICENSE", PANDA_LICENSE_SHA256),),
    ),
    UR5E_2F85: _AssetPins(
        "google-deepmind/mujoco_menagerie",
        ALOHA_SOURCE_REVISION,
        UR5E_2F85_MJCF_SHA256,
        (
            ("LICENSE.ur5e", UR5E_LICENSE_SHA256),
            ("LICENSE.robotiq_2f85", ROBOTIQ_2F85_LICENSE_SHA256),
        ),
    ),
    X5: _AssetPins(
        "RoboDojo-Benchmark/RoboDojo",
        X5_SOURCE_REVISION,
        X5_MJCF_SHA256,
        (
            ("LICENSE.robodojo", X5_ROBODOJO_LICENSE_SHA256),
            ("LICENSE.arx5-sdk", X5_SDK_LICENSE_SHA256),
        ),
    ),
}
_TARGET_SOURCE_ASSETS = {
    ALOHA2: ("robots/aloha2/aloha.xml", "robots/aloha2/frame.xml"),
    PANDA: ("robots/panda/panda.xml", None),
    UR5E_2F85: ("robots/ur5e_2f85/ur5e_2f85.xml", None),
    X5: ("robots/x5/x5.xml", None),
}


class _Kinematics(Protocol):
    home: np.ndarray
    arm_indices: np.ndarray
    gripper_indices: np.ndarray
    gripper_closed_positions: np.ndarray
    gripper_open_positions: np.ndarray

    def solve(
        self,
        seed: np.ndarray,
        target_positions: np.ndarray,
        target_rotations: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def attach_positions_batch(self, joint_q: np.ndarray) -> np.ndarray: ...

    def attach_poses_batch(self, joint_q: np.ndarray) -> np.ndarray: ...

    def body_state_batch(
        self, joint_q: np.ndarray, joint_qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _Dynamics(Protocol):
    control_fps: float
    sim_substeps: int

    def reconstruct(
        self,
        reference_joint_q: np.ndarray,
        reference_joint_qd: np.ndarray,
        reference_gripper_open: np.ndarray,
        target_attach_positions: np.ndarray,
        target_approach_directions: np.ndarray,
        pregrasp_mask: np.ndarray,
        approach_validation_mask: np.ndarray,
        attached: np.ndarray,
    ) -> DynamicsRetargetResult: ...


@dataclass(frozen=True, slots=True)
class _Motion:
    joint_q: np.ndarray
    target_gripper_open: np.ndarray
    target_attach_positions: np.ndarray
    target_approach_directions: np.ndarray
    pregrasp_mask: np.ndarray
    approach_validation_mask: np.ndarray
    attach_errors: np.ndarray
    approach_errors: np.ndarray


def retarget(
    source: str | Path,
    output: str | Path,
    *,
    target_robot: str = ALOHA2,
    assets: str | Path = "assets",
    device: str = "cuda:0",
    position_tolerance_m: float = 0.005,
) -> dict:
    """Retarget one trajectory or collection to a fixed robot profile."""

    if position_tolerance_m <= 0.0:
        raise ValueError("position_tolerance_m must be positive")
    source_root = Path(source).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"retarget output already exists: {output_root}")
    trajectories, is_collection = _source_trajectories(source_root)
    target_profile = robot_profile(target_robot)
    try:
        asset_pins = _TARGET_ASSET_PINS[target_profile.robot_id]
    except KeyError as exc:
        raise ValueError(f"retarget does not support {target_robot!r}") from exc
    assets_root = Path(assets).expanduser().resolve()
    source_asset, source_installation = _TARGET_SOURCE_ASSETS[target_profile.robot_id]
    mjcf = assets_root / source_asset
    if not mjcf.is_file() or _sha256(mjcf) != asset_pins.mjcf_sha256:
        raise FileNotFoundError(
            f"pinned {target_profile.robot_id} asset is missing or modified: {mjcf}"
        )
    for license_name, expected_sha256 in asset_pins.licenses:
        license_path = mjcf.parent / license_name
        if not license_path.is_file() or _sha256(license_path) != expected_sha256:
            raise FileNotFoundError(
                f"pinned {target_profile.robot_id} license is missing or modified: {license_path}"
            )
    installation = None
    if asset_pins.installation_sha256 is not None:
        if source_installation is None:
            raise RuntimeError("target robot does not define its installation asset")
        installation = assets_root / source_installation
        if not installation.is_file() or _sha256(installation) != asset_pins.installation_sha256:
            raise FileNotFoundError(
                f"pinned installation asset is missing or modified: {installation}"
            )

    staging = output_root.with_name(f".{output_root.name}.tmp-{uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        kinematics = _RobotKinematics(
            target_profile,
            assets_root / target_profile.asset,
            device=device,
        )
        first_reader = TrajectoryReader(trajectories[0])
        if first_reader.manifest.scene.task_name != SHIRT_FLATTEN:
            raise RuntimeError(
                f"retarget source task must be {SHIRT_FLATTEN!r}"
            )
        first_asset_id = first_reader.manifest.metadata.get("asset_id")
        if not isinstance(first_asset_id, str) or not first_asset_id:
            raise RuntimeError("retarget source trajectory asset_id is missing")
        dynamics = AlohaDynamicsRetargeter(
            assets_root,
            garment_id=first_asset_id,
            control_fps=first_reader.manifest.control_fps,
            sim_substeps=first_reader.manifest.sim_substeps,
            device=device,
            target_robot=target_robot,
        )
        results = []
        for trajectory in trajectories:
            relative = trajectory.relative_to(source_root) if is_collection else Path(".")
            destination = staging / relative
            results.append(
                _retarget_trajectory(
                    trajectory,
                    destination,
                    kinematics=kinematics,
                    dynamics=dynamics,
                    position_tolerance_m=position_tolerance_m,
                    target_robot=target_robot,
                )
            )
        if is_collection:
            source_meta = _optional_collection_meta(source_root)
            trajectory_entries = [
                {
                    "path": str(path.relative_to(staging)),
                    "asset_id": result["metadata"]["asset_id"],
                    "action_count": result["action_count"],
                    "source_trajectory": result["metadata"]["retarget"]["source_trajectory"],
                }
                for path, result in zip(
                    (staging / item.relative_to(source_root) for item in trajectories),
                    results,
                    strict=True,
                )
            ]
            all_training_eligible = all(
                item["metadata"]["retarget"]["validation"]["training_eligible"] for item in results
            )
            collection = {
                "schema": RETARGET_COLLECTION_SCHEMA,
                "format_version": RETARGET_SCHEMA_VERSION,
                "status": "complete",
                "source": str(source_root),
                "target_robot": target_profile.to_json_dict(),
                "attach_alignment": "position_with_pregrasp_approach",
                "trajectory_count": len(results),
                "expected_trajectories": len(results),
                "training_eligible": all_training_eligible,
                "scenario": source_meta.get(
                    "scenario",
                    {"scenario_id": "shirt_flatten", "prompt": "Flatten the shirt."},
                ),
                "policy": {},
                "trajectories": trajectory_entries,
                "kinematic_valid_count": sum(
                    bool(item["metadata"]["retarget"]["validation"]["kinematic_valid"])
                    for item in results
                ),
                "dynamics_valid_count": sum(
                    bool(item["metadata"]["retarget"]["validation"]["dynamics_valid"])
                    for item in results
                ),
            }
            _write_json(staging / "meta.json", collection)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output": str(output_root),
        "target_robot": target_robot,
        "attach_alignment": "position_with_pregrasp_approach",
        "trajectory_count": len(results),
        "kinematic_valid_count": sum(
            bool(item["metadata"]["retarget"]["validation"]["kinematic_valid"]) for item in results
        ),
        "dynamics_valid_count": sum(
            bool(item["metadata"]["retarget"]["validation"]["dynamics_valid"]) for item in results
        ),
        "robot_dynamics_rollout": True,
        "physical_rollout": False,
    }


def _retarget_trajectory(
    source: Path,
    output: Path,
    *,
    kinematics: _Kinematics,
    dynamics: _Dynamics,
    position_tolerance_m: float,
    target_robot: str = ALOHA2,
) -> dict:
    reader = TrajectoryReader(source)
    source_robot = reader.manifest.scene.robot_id
    if source_robot != PIPER:
        raise RuntimeError(f"retarget source robot must be {PIPER!r}")
    source_task = reader.manifest.scene.task_name
    if source_task != SHIRT_FLATTEN:
        raise RuntimeError(f"retarget source task must be {SHIRT_FLATTEN!r}")
    target_profile = robot_profile(target_robot)
    if float(reader.manifest.control_fps) != float(dynamics.control_fps) or int(
        reader.manifest.sim_substeps
    ) != int(dynamics.sim_substeps):
        raise ValueError("retarget collection mixes incompatible simulation timing")
    asset_id = reader.manifest.metadata.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise RuntimeError("retarget source trajectory asset_id is missing")
    states = reader.state_arrays(
        (
            "body_q",
            "gripper_open_state",
            "grasp_attached",
            "grasp_previous_open",
            "grasp_pins",
            "grasp_pin_local",
        )
    )
    body_q = states["body_q"]
    if body_q.shape[1:] != (len(_PIPER_PROFILE.body_names), 7):
        raise RuntimeError("source trajectory does not contain the expected Piper body state")
    source_attach = piper_attach_poses(body_q)
    gripper_state = np.asarray(states["gripper_open_state"], dtype=np.float32)
    grasp_attached = np.asarray(states["grasp_attached"], dtype=np.int32)
    motion = _retarget_motion(
        source_attach,
        gripper_state,
        grasp_attached,
        kinematics=kinematics,
        position_tolerance_m=position_tolerance_m,
    )
    fps = float(reader.manifest.control_fps)
    kinematic_attach = kinematics.attach_positions_batch(motion.joint_q)
    kinematic_attach_errors = np.linalg.norm(
        kinematic_attach - motion.target_attach_positions,
        axis=-1,
    ).astype(np.float32)
    kinematic_poses = kinematics.attach_poses_batch(motion.joint_q)
    kinematic_approach = _quat_rotate(
        kinematic_poses[..., 3:7], np.asarray((0.0, 0.0, 1.0))
    ).astype(np.float32)
    kinematic_approach_errors = _direction_angle(
        kinematic_approach, motion.target_approach_directions
    ).astype(np.float32)
    active_kinematic_approach_errors = kinematic_approach_errors[
        motion.approach_validation_mask
    ]
    reference_joint_qd = _robot_velocity(motion.joint_q, target_profile, fps)
    dynamics_result = dynamics.reconstruct(
        motion.joint_q,
        reference_joint_qd,
        motion.target_gripper_open,
        motion.target_attach_positions,
        motion.target_approach_directions,
        motion.pregrasp_mask,
        motion.approach_validation_mask,
        grasp_attached,
    )
    max_error = float(kinematic_attach_errors.max(initial=0.0))
    max_approach_error = float(active_kinematic_approach_errors.max(initial=0.0))
    kinematic_valid = bool(
        max_error <= position_tolerance_m and max_approach_error <= _APPROACH_TOLERANCE_RAD
    )
    dynamics_poses = _attach_poses_from_body(
        dynamics_result.body_q,
        target_profile.attach_frames,
        tuple(
            target_profile.body_names.index(frame.parent_body)
            for frame in target_profile.attach_frames
        ),
    )
    dynamics_approach = _quat_rotate(dynamics_poses[..., 3:7], np.asarray((0.0, 0.0, 1.0))).astype(
        np.float32
    )
    dynamics_approach_errors = _direction_angle(
        dynamics_approach, motion.target_approach_directions
    ).astype(np.float32)
    active_dynamics_approach_errors = dynamics_approach_errors[
        motion.approach_validation_mask
    ]
    max_dynamics_approach_error = float(active_dynamics_approach_errors.max(initial=0.0))
    dynamics_valid = bool(
        dynamics_result.validation["dynamics_valid"]
        and max_dynamics_approach_error <= _APPROACH_TOLERANCE_RAD
    )
    validation = {
        "kinematic_valid": kinematic_valid,
        "dynamics_valid": dynamics_valid,
        "training_eligible": bool(kinematic_valid and dynamics_valid),
        "source_timeline_preserved": True,
        "position_tolerance_m": float(position_tolerance_m),
        "pregrasp_approach_tolerance_rad": float(_APPROACH_TOLERANCE_RAD),
        "max_attach_position_error_m": max_error,
        "max_pregrasp_approach_error_rad": max_approach_error,
        "max_dynamics_pregrasp_approach_error_rad": max_dynamics_approach_error,
        "p95_attach_position_error_m": float(
            np.quantile(kinematic_attach_errors, 0.95) if kinematic_attach_errors.size else 0.0
        ),
        "source_state_count": reader.manifest.state_count,
        "output_state_count": len(motion.joint_q),
        "collision_checked": False,
        "robot_dynamics": dynamics_result.validation,
    }
    state_arrays = {
        "sample_index": np.arange(len(motion.joint_q), dtype=np.int64),
        "joint_q": dynamics_result.joint_q,
        "joint_qd": dynamics_result.joint_qd,
        "body_q": dynamics_result.body_q,
        "body_qd": dynamics_result.body_qd,
        "gripper_open_state": dynamics_result.gripper_open_state,
        "grasp_attached": grasp_attached,
        "grasp_previous_open": np.asarray(states["grasp_previous_open"], dtype=np.float32),
        "grasp_pin_count": np.zeros((len(motion.joint_q), 2), dtype=np.int32),
        "grasp_pins": np.full_like(states["grasp_pins"], -1, dtype=np.int32),
        "grasp_pin_local": np.zeros_like(states["grasp_pin_local"], dtype=np.float32),
    }
    action_arrays = {
        "sample_index": np.arange(len(motion.joint_q) - 1, dtype=np.int64),
        "joint_target": dynamics_result.joint_target,
        "gripper_open_command": dynamics_result.gripper_open_command,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "states.npz").open("wb") as file:
        np.savez_compressed(file, **state_arrays)
    if len(motion.joint_q) > 1:
        with (output / "actions.npz").open("wb") as file:
            np.savez_compressed(file, **action_arrays)
    metrics = {
        "source_attach_position": motion.target_attach_positions,
        "reference_joint_q": motion.joint_q,
        "reference_joint_qd": reference_joint_qd,
        "kinematic_attach_position": kinematic_attach.astype(np.float32),
        "kinematic_attach_position_error": kinematic_attach_errors,
        "dynamics_attach_position": dynamics_result.attach_positions,
        "dynamics_attach_position_error": dynamics_result.attach_errors,
        "source_attach_rotation": source_attach[..., 3:7],
        "target_approach_direction": motion.target_approach_directions,
        "pregrasp_mask": motion.pregrasp_mask,
        "approach_validation_mask": motion.approach_validation_mask,
        "kinematic_approach_direction": kinematic_approach,
        "kinematic_approach_error": kinematic_approach_errors,
        "dynamics_approach_direction": dynamics_approach,
        "dynamics_approach_error": dynamics_approach_errors,
    }
    with (output / "retarget_metrics.npz").open("wb") as file:
        np.savez_compressed(file, **metrics)
    metadata = {
        "asset_id": asset_id,
        "producer": "clothdojo.data.retarget",
        "joint_names": list(target_profile.arm_joint_names),
        "source_metadata": dict(reader.manifest.metadata),
        "storage": {
            "kind": "derived_overlay",
            "base_trajectory": str(source),
            "base_manifest_sha256": _sha256(source / "manifest.json"),
        },
        "retarget": {
            "kind": "dynamics_compensated_counterfactual_replay",
            "source_trajectory": str(source),
            "source_robot": source_robot,
            "target_robot": target_robot,
            "attach_alignment": "position_with_pregrasp_approach",
            "target_asset": {
                "source": _TARGET_ASSET_PINS[target_robot].source,
                "revision": _TARGET_ASSET_PINS[target_robot].revision,
                "mjcf_sha256": _TARGET_ASSET_PINS[target_robot].mjcf_sha256,
                "licenses": dict(_TARGET_ASSET_PINS[target_robot].licenses),
            },
            "physical_rollout": False,
            "robot_dynamics_rollout": True,
            "style3d_solver_used": False,
            "collision_checked": False,
            "metrics": "retarget_metrics.npz",
            "validation": validation,
        },
    }
    manifest = TrajectoryManifest(
        status="complete",
        control_fps=reader.manifest.control_fps,
        sim_substeps=reader.manifest.sim_substeps,
        units=reader.manifest.units,
        max_actions=reader.manifest.max_actions,
        state_count=len(motion.joint_q),
        action_count=len(motion.joint_q) - 1,
        state_shapes={
            **{
                field: list(state_arrays[field].shape[1:])
                for field in state_arrays
                if field != "sample_index"
            },
            "particle_q": list(reader.manifest.state_shapes["particle_q"]),
            "particle_qd": list(reader.manifest.state_shapes["particle_qd"]),
        },
        action_shapes={
            field: list(action_arrays[field].shape[1:])
            for field in action_arrays
            if field != "sample_index"
        },
        scene=type(reader.manifest.scene)(
            task_name=reader.manifest.scene.task_name,
            robot_id=target_profile.robot_id,
            asset_fingerprints={
                "robot_kinematics": _TARGET_ASSET_PINS[target_robot].mjcf_sha256,
                **(
                    {
                        "installation_visual": _TARGET_ASSET_PINS[
                            target_robot
                        ].installation_sha256
                    }
                    if _TARGET_ASSET_PINS[target_robot].installation_sha256 is not None
                    else {}
                ),
            },
            config_fingerprints={
                **{
                    f"source/{key}": value
                    for key, value in reader.manifest.scene.config_fingerprints.items()
                },
            },
        ),
        metadata=metadata,
    ).to_json_dict()
    _write_json(output / "manifest.json", manifest)
    return manifest


def piper_attach_positions(body_q: np.ndarray) -> np.ndarray:
    """Return world-space Piper semantic grasp centers from recorded body poses."""

    return piper_attach_poses(body_q)[..., :3]


def piper_attach_poses(body_q: np.ndarray) -> np.ndarray:
    """Return world-space Piper semantic attach poses from recorded body poses."""

    return _attach_poses_from_body(
        body_q,
        _PIPER_PROFILE.attach_frames,
        _SOURCE_EE_BODY_INDICES,
    )


def _retarget_motion(
    source_attach_poses: np.ndarray,
    gripper_open_state: np.ndarray,
    grasp_attached: np.ndarray,
    *,
    kinematics: _Kinematics,
    position_tolerance_m: float,
) -> _Motion:
    source_poses = np.asarray(source_attach_poses, dtype=np.float32)
    grippers = np.asarray(gripper_open_state, dtype=np.float32)
    attached = np.asarray(grasp_attached, dtype=np.int32) != 0
    if source_poses.ndim != 3 or source_poses.shape[1:] != (2, 7) or len(source_poses) == 0:
        raise ValueError("source attach poses must have non-empty shape [N, 2, 7]")
    if grippers.shape != (len(source_poses), 2):
        raise ValueError("gripper state must have shape [N, 2]")
    if not np.isfinite(grippers).all() or np.any((grippers < 0.0) | (grippers > 1.0)):
        raise ValueError("gripper state must be finite and normalized to [0, 1]")
    if attached.shape != (len(source_poses), 2):
        raise ValueError("grasp attached state must have shape [N, 2]")
    targets = source_poses[..., :3]
    source_approach = _quat_rotate(source_poses[..., 3:7], np.asarray((0.0, 0.0, 1.0))).astype(
        np.float32
    )
    pregrasp = _pregrasp_mask(grippers, attached)
    approach_validation = _pregrasp_mask(grippers, attached, lead_multiplier=1)
    target_gripper_open = grippers.copy()

    alignment_fraction = _pregrasp_alignment_fraction(grippers, pregrasp)
    target_approach = source_approach.copy()
    frame_count = len(targets)
    joint_q = np.empty((frame_count, len(kinematics.home)), dtype=np.float32)
    transition_start = np.zeros((2, 3), dtype=np.float64)
    previous_active = np.zeros(2, dtype=bool)
    seed = np.asarray(kinematics.home, dtype=np.float32).copy()
    for index in range(frame_count):
        positioned = kinematics.solve(seed, targets[index])
        active = pregrasp[index]
        if np.any(active):
            achieved = kinematics.attach_poses_batch(positioned[None])[0]
            achieved_directions = _quat_rotate(achieved[:, 3:7], np.asarray((0.0, 0.0, 1.0)))
            starting = active & ~previous_active
            transition_start[starting] = achieved_directions[starting]
            desired_directions = target_approach[index].copy()
            for side in np.flatnonzero(active):
                desired_directions[side] = _slerp_direction(
                    transition_start[side],
                    source_approach[index, side],
                    alignment_fraction[index, side],
                )
            target_rotations = achieved[:, 3:7].copy()
            target_rotations[active] = _align_quaternion_axis(
                target_rotations[active],
                desired_directions[active],
            )
            seed = kinematics.solve(positioned, targets[index], target_rotations)
            solved = kinematics.attach_poses_batch(seed[None])[0]
            solved_directions = _quat_rotate(solved[:, 3:7], np.asarray((0.0, 0.0, 1.0)))
            leading = active & (alignment_fraction[index] < 1.0)
            target_approach[index, leading] = solved_directions[leading]
        else:
            seed = positioned
        joint_q[index] = seed
        previous_active = active
    _set_gripper_positions(
        joint_q,
        target_gripper_open,
        kinematics.gripper_indices,
        kinematics.gripper_closed_positions,
        kinematics.gripper_open_positions,
    )
    achieved_poses = kinematics.attach_poses_batch(joint_q)
    errors = np.linalg.norm(achieved_poses[..., :3] - targets, axis=-1)
    achieved_approach = _quat_rotate(achieved_poses[..., 3:7], np.asarray((0.0, 0.0, 1.0)))
    approach_errors = _direction_angle(achieved_approach, target_approach)
    return _Motion(
        joint_q=joint_q,
        target_gripper_open=target_gripper_open,
        target_attach_positions=targets,
        target_approach_directions=target_approach,
        pregrasp_mask=pregrasp,
        approach_validation_mask=approach_validation,
        attach_errors=errors,
        approach_errors=approach_errors,
    )


def _pregrasp_mask(
    gripper_open_state: np.ndarray,
    attached: np.ndarray,
    *,
    lead_multiplier: int = 3,
) -> np.ndarray:
    openness = np.asarray(gripper_open_state, dtype=np.float32)
    attached_mask = np.asarray(attached, dtype=bool)
    if openness.ndim != 2 or openness.shape[1] != 2 or attached_mask.shape != openness.shape:
        raise ValueError("gripper state and attach mask must have shape [N, 2]")
    if lead_multiplier < 1:
        raise ValueError("pregrasp lead multiplier must be positive")
    pregrasp = np.zeros_like(attached_mask)
    closing = np.zeros_like(attached_mask)
    closing[1:] = openness[1:] < openness[:-1] - 1.0e-4
    for side in range(2):
        free_start = 0 if len(attached_mask) and not attached_mask[0, side] else None
        for index in range(1, len(attached_mask)):
            if attached_mask[index - 1, side] and not attached_mask[index, side]:
                free_start = index
            elif not attached_mask[index - 1, side] and attached_mask[index, side]:
                if free_start is not None and index > free_start and closing[index - 1, side]:
                    closing_start = index - 1
                    while closing_start > free_start and closing[closing_start - 1, side]:
                        closing_start -= 1
                    closing_frames = index - closing_start
                    lead_start = max(
                        free_start,
                        closing_start - lead_multiplier * closing_frames,
                    )
                    pregrasp[lead_start:index, side] = True
                free_start = None
    return pregrasp


def _pregrasp_alignment_fraction(
    gripper_open_state: np.ndarray,
    pregrasp_mask: np.ndarray,
) -> np.ndarray:
    openness = np.asarray(gripper_open_state, dtype=np.float64)
    mask = np.asarray(pregrasp_mask, dtype=bool)
    if openness.shape != mask.shape:
        raise ValueError("gripper state and pregrasp mask must have the same shape")
    if openness.ndim != 2 or openness.shape[1] != 2:
        raise ValueError("gripper state and pregrasp mask must have shape [N, 2]")
    closing = np.zeros_like(mask)
    closing[1:] = openness[1:] < openness[:-1] - 1.0e-4
    result = np.zeros_like(openness)
    for side in range(2):
        active = mask[:, side]
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        for start, end in zip(starts, ends, strict=True):
            closing_indices = np.flatnonzero(closing[start : end + 1, side])
            closing_start = int(start + closing_indices[0]) if len(closing_indices) else int(end)
            span = max(closing_start - int(start), 1)
            for index in range(int(start), int(end) + 1):
                result[index, side] = min((index - start) / span, 1.0)
    return result.astype(np.float32)


def _slerp_direction(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    initial = np.asarray(start, dtype=np.float64)
    target = np.asarray(end, dtype=np.float64)
    initial /= np.linalg.norm(initial)
    target /= np.linalg.norm(target)
    fraction = float(np.clip(amount, 0.0, 1.0))
    dot = float(np.clip(np.dot(initial, target), -1.0, 1.0))
    if dot > 1.0 - 1.0e-8:
        value = (1.0 - fraction) * initial + fraction * target
        return value / np.linalg.norm(value)
    if dot < -1.0 + 1.0e-8:
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(initial)))] = 1.0
        orthogonal = np.cross(initial, basis)
        orthogonal /= np.linalg.norm(orthogonal)
        return np.cos(np.pi * fraction) * initial + np.sin(np.pi * fraction) * orthogonal
    angle = np.arccos(dot)
    scale = np.sin(angle)
    return (np.sin((1.0 - fraction) * angle) * initial + np.sin(fraction * angle) * target) / scale


def _set_gripper_positions(
    joint_q: np.ndarray,
    open_state: np.ndarray,
    gripper_indices: np.ndarray,
    closed_positions: np.ndarray,
    open_positions: np.ndarray,
) -> None:
    openness = np.clip(open_state, 0.0, 1.0)
    closed = np.asarray(closed_positions, dtype=np.float32).reshape(1, 1, 2)
    opened = np.asarray(open_positions, dtype=np.float32).reshape(1, 1, 2)
    positions = closed + openness[:, :, None] * (opened - closed)
    joint_q[:, gripper_indices] = positions.reshape(len(joint_q), 4)


class _RobotKinematics:
    def __init__(self, profile, asset: Path, *, device: str, iterations: int = 40) -> None:
        import newton
        import newton.ik as ik
        import warp as wp

        from clothdojo.sim.scene import _add_robot

        self.newton = newton
        self.wp = wp
        self.profile = profile
        self.home = np.asarray(profile.home_joint_q, dtype=np.float32)
        self.arm_indices = np.asarray(profile.arm_coord_indices, dtype=np.int64)
        self.arm_dof_indices = np.asarray(profile.arm_dof_indices, dtype=np.int64)
        self.gripper_indices = np.asarray(profile.gripper_coord_indices, dtype=np.int64)
        self.gripper_closed_positions = np.asarray(
            profile.gripper_closed_positions, dtype=np.float32
        )
        self.gripper_open_positions = np.asarray(profile.gripper_open_positions, dtype=np.float32)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        _body_count, _arms, _grippers, _coords, end_effectors = _add_robot(
            newton,
            wp,
            builder,
            asset,
            profile,
            load_visual_shapes=False,
        )
        self.end_effectors = tuple(end_effectors)
        self.model = builder.finalize(device=device, requires_grad=False)
        self.iterations = int(iterations)
        self.q_count = len(self.home)
        self._q_in = wp.zeros((1, self.q_count), dtype=wp.float32, device=self.model.device)
        self._q_out = wp.zeros_like(self._q_in)
        self._left_targets = wp.zeros((1,), dtype=wp.vec3, device=self.model.device)
        self._right_targets = wp.zeros((1,), dtype=wp.vec3, device=self.model.device)
        self._left_rotation_targets = wp.zeros((1,), dtype=wp.vec4, device=self.model.device)
        self._right_rotation_targets = wp.zeros((1,), dtype=wp.vec4, device=self.model.device)

        def objectives(*, include_rotation: bool):
            values = [
                ik.IKObjectivePosition(
                    self.end_effectors[0],
                    wp.vec3(*profile.attach_frames[0].position),
                    self._left_targets,
                    _IK_POSITION_WEIGHT,
                ),
                ik.IKObjectivePosition(
                    self.end_effectors[1],
                    wp.vec3(*profile.attach_frames[1].position),
                    self._right_targets,
                    _IK_POSITION_WEIGHT,
                ),
            ]
            if include_rotation:
                values.extend(
                    (
                        ik.IKObjectiveRotation(
                            self.end_effectors[0],
                            wp.quat(*profile.attach_frames[0].rotation_xyzw),
                            self._left_rotation_targets,
                            weight=_IK_ROTATION_WEIGHT,
                        ),
                        ik.IKObjectiveRotation(
                            self.end_effectors[1],
                            wp.quat(*profile.attach_frames[1].rotation_xyzw),
                            self._right_rotation_targets,
                            weight=_IK_ROTATION_WEIGHT,
                        ),
                    )
                )
            # Newton represents MJCF CONNECT constraints with auxiliary ball
            # coordinates that are not MuJoCo velocity DOFs. Its joint-limit IK
            # objective assumes equal coordinate/DOF layouts, so closed-chain
            # profiles apply scalar arm limits after solving instead.
            if int(self.model.joint_coord_count) == int(self.model.joint_dof_count):
                values.append(
                    ik.IKObjectiveJointLimit(
                        self.model.joint_limit_lower,
                        self.model.joint_limit_upper,
                        10.0,
                    )
                )
            return values

        self._position_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=objectives(include_rotation=False),
            jacobian_mode=ik.IKJacobianType.MIXED,
            lambda_initial=0.1,
        )
        self._pose_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=objectives(include_rotation=True),
            jacobian_mode=ik.IKJacobianType.MIXED,
            lambda_initial=0.1,
        )
        self._fk_q = wp.zeros(self.q_count, dtype=wp.float32, device=self.model.device)
        self.dof_count = int(profile.joint_dof_count)
        if int(self.model.joint_dof_count) != self.dof_count:
            raise RuntimeError(
                f"pinned {profile.robot_id} velocity topology differs from its profile"
            )
        self._fk_qd = wp.zeros(self.dof_count, dtype=wp.float32, device=self.model.device)
        self._fk_state = self.model.state()

    def solve(
        self,
        seed: np.ndarray,
        target_positions: np.ndarray,
        target_rotations: np.ndarray | None = None,
    ) -> np.ndarray:
        seed = np.asarray(seed, dtype=np.float32).reshape(self.q_count)
        targets = np.asarray(target_positions, dtype=np.float32).reshape(2, 3)
        self._left_targets.assign(targets[None, 0])
        self._right_targets.assign(targets[None, 1])
        solver = self._position_solver
        if target_rotations is not None:
            rotations = np.asarray(target_rotations, dtype=np.float32).reshape(2, 4)
            self._left_rotation_targets.assign(rotations[None, 0])
            self._right_rotation_targets.assign(rotations[None, 1])
            solver = self._pose_solver
        self._q_in.assign(seed[None])
        solver.reset()
        solver.step(self._q_in, self._q_out, iterations=self.iterations)
        result = self._q_out.numpy()[0].astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError(
                f"Newton IK returned non-finite {self.profile.robot_id} coordinates"
            )
        lower = self.model.joint_limit_lower.numpy()[self.arm_dof_indices]
        upper = self.model.joint_limit_upper.numpy()[self.arm_dof_indices]
        result[self.arm_indices] = np.clip(result[self.arm_indices], lower, upper)
        result[self.gripper_indices] = seed[self.gripper_indices]
        return result

    def attach_positions_batch(self, joint_q: np.ndarray) -> np.ndarray:
        return self.attach_poses_batch(joint_q)[..., :3]

    def attach_poses_batch(self, joint_q: np.ndarray) -> np.ndarray:
        values = np.asarray(joint_q, dtype=np.float32)
        result = np.empty((len(values), 2, 7), dtype=np.float32)
        for index, coordinates in enumerate(values):
            self._fk_q.assign(coordinates)
            self.newton.eval_fk(
                self.model,
                self._fk_q,
                self.model.joint_qd,
                self._fk_state,
            )
            poses = self._fk_state.body_q.numpy()[list(self.end_effectors)]
            result[index] = _attach_poses_from_body(
                poses[None],
                self.profile.attach_frames,
                (0, 1),
            )[0]
        return result

    def body_state_batch(
        self, joint_q: np.ndarray, joint_qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(joint_q, dtype=np.float32)
        velocities = np.asarray(joint_qd, dtype=np.float32)
        if positions.shape[1:] != (self.q_count,) or velocities.shape != (
            len(positions),
            self.dof_count,
        ):
            raise ValueError("robot joint position and velocity arrays differ from the profile")
        body_count = len(self.profile.body_names)
        body_q = np.empty((len(positions), body_count, 7), dtype=np.float32)
        body_qd = np.empty((len(positions), body_count, 6), dtype=np.float32)
        for index, (q, qd) in enumerate(zip(positions, velocities, strict=True)):
            self._fk_q.assign(q)
            self._fk_qd.assign(qd)
            self.newton.eval_fk(
                self.model,
                self._fk_q,
                self._fk_qd,
                self._fk_state,
            )
            body_q[index] = self._fk_state.body_q.numpy()
            body_qd[index] = self._fk_state.body_qd.numpy()
        return body_q, body_qd


def _attach_poses_from_body(body_q: np.ndarray, attach_frames, body_indices) -> np.ndarray:
    poses = np.asarray(body_q, dtype=np.float64)[:, list(body_indices)]
    offsets = np.asarray([frame.position for frame in attach_frames], dtype=np.float64)
    local_rotations = np.asarray([frame.rotation_xyzw for frame in attach_frames], dtype=np.float64)
    result = np.empty(poses.shape, dtype=np.float64)
    result[..., :3] = poses[..., :3] + _quat_rotate(poses[..., 3:7], offsets)
    result[..., 3:7] = _quat_multiply(poses[..., 3:7], local_rotations)
    return result.astype(np.float32)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.broadcast_to(np.asarray(right, dtype=np.float64), lhs.shape)
    xyz = (
        lhs[..., 3:4] * rhs[..., :3]
        + rhs[..., 3:4] * lhs[..., :3]
        + np.cross(lhs[..., :3], rhs[..., :3])
    )
    scalar = lhs[..., 3:4] * rhs[..., 3:4] - np.sum(
        lhs[..., :3] * rhs[..., :3], axis=-1, keepdims=True
    )
    result = np.concatenate((xyz, scalar), axis=-1)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _align_quaternion_axis(
    quaternion: np.ndarray,
    target_direction: np.ndarray,
) -> np.ndarray:
    current = np.asarray(quaternion, dtype=np.float64)
    desired = np.asarray(target_direction, dtype=np.float64)
    desired /= np.linalg.norm(desired, axis=-1, keepdims=True)
    approach = _quat_rotate(current, np.asarray((0.0, 0.0, 1.0)))
    cross = np.cross(approach, desired)
    dot = np.sum(approach * desired, axis=-1)
    delta = np.concatenate((cross, (1.0 + dot)[..., None]), axis=-1)
    opposite = dot < -1.0 + 1.0e-8
    for index in np.flatnonzero(opposite):
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(approach[index])))] = 1.0
        axis = np.cross(approach[index], basis)
        axis /= np.linalg.norm(axis)
        delta[index] = (*axis, 0.0)
    delta /= np.linalg.norm(delta, axis=-1, keepdims=True)
    return _quat_multiply(delta, current).astype(np.float32)


def _direction_angle(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    lhs = np.asarray(actual, dtype=np.float64)
    rhs = np.broadcast_to(np.asarray(target, dtype=np.float64), lhs.shape).copy()
    lhs /= np.linalg.norm(lhs, axis=-1, keepdims=True)
    rhs /= np.linalg.norm(rhs, axis=-1, keepdims=True)
    dot = np.sum(lhs * rhs, axis=-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.broadcast_to(np.asarray(vector, dtype=np.float64), q[..., :3].shape)
    xyz = q[..., :3]
    return v + 2.0 * np.cross(xyz, np.cross(xyz, v) + q[..., 3:4] * v)


def _source_trajectories(source: Path) -> tuple[list[Path], bool]:
    manifest = source / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read source manifest {manifest}: {exc}") from exc
        if data.get("format_version") == 1:
            return [source], False
    trajectory_root = source / "trajectories"
    if not trajectory_root.is_dir():
        raise RuntimeError("source must be one trajectory or a collection with trajectories/")
    trajectories = sorted(
        path.parent for path in trajectory_root.glob("*/manifest.json") if path.is_file()
    )
    if not trajectories:
        raise RuntimeError(f"source collection has no trajectories: {source}")
    return trajectories, True


def load_retarget_collection(source: str | Path, *, require_complete: bool = True) -> dict:
    root = Path(source).expanduser().resolve()
    try:
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read retarget collection: {root}: {exc}") from exc
    if meta.get("schema") != RETARGET_COLLECTION_SCHEMA:
        raise ValueError("unsupported retarget collection schema")
    if meta.get("format_version") != RETARGET_SCHEMA_VERSION:
        raise ValueError("unsupported retarget collection version")
    if require_complete and meta.get("status") != "complete":
        raise RuntimeError("retarget collection is not complete")
    trajectories = meta.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("retarget collection trajectory index is invalid")
    if require_complete:
        for item in trajectories:
            reader = TrajectoryReader(root / item["path"])
            if reader.manifest.action_count != item.get("action_count"):
                raise ValueError("retarget trajectory action count differs")
    return meta


def _optional_collection_meta(source: Path) -> dict:
    path = source / "meta.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if len(array) == 1:
        return np.zeros_like(array)
    return np.gradient(array, 1.0 / float(fps), axis=0).astype(np.float32)


def _robot_velocity(joint_q: np.ndarray, profile, fps: float) -> np.ndarray:
    """Differentiate scalar controlled coordinates into the Newton DOF layout."""

    coordinates = np.concatenate(
        (
            np.asarray(profile.arm_coord_indices, dtype=np.int64),
            np.asarray(profile.gripper_coord_indices, dtype=np.int64),
        )
    )
    dofs = np.concatenate(
        (
            np.asarray(profile.arm_dof_indices, dtype=np.int64),
            np.asarray(profile.gripper_dof_indices, dtype=np.int64),
        )
    )
    result = np.zeros((len(joint_q), int(profile.joint_dof_count)), dtype=np.float32)
    result[:, dofs] = _finite_difference(np.asarray(joint_q, dtype=np.float32)[:, coordinates], fps)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
