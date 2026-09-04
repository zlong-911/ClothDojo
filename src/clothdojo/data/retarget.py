"""Dynamics-consistent counterfactual replay from Piper to fixed workstations."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

import numpy as np

from clothdojo.data.retarget_dynamics import (
    DynamicsRetargetResult,
    RobotDynamicsRetargeter,
)
from clothdojo.robots import (
    ALOHA2,
    FR3_DUO,
    PIPER,
    UR5E_2F85,
    X5,
    robot_profile,
)
from clothdojo.tasks import SHIRT_FLATTEN, SHIRT_FOLD
from clothdojo.trajectory import TrajectoryManifest, TrajectoryReader

RETARGET_COLLECTION_SCHEMA = "clothdojo_retarget_collection"
RETARGET_SCHEMA_VERSION = 1
_SUPPORTED_SOURCE_TASKS = frozenset((SHIRT_FLATTEN, SHIRT_FOLD))

_IK_POSITION_WEIGHT = 100.0
_IK_APPROACH_MARKER_M = 0.1
_ROLL_PREFERENCE_RATE_RAD_S = 1.0
_PIPER_PROFILE = robot_profile(PIPER)
_SOURCE_EE_BODY_INDICES = np.asarray(
    tuple(
        _PIPER_PROFILE.body_names.index(frame.parent_body) for frame in _PIPER_PROFILE.attach_frames
    ),
    dtype=np.int64,
)


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
        *,
        align_roll: bool = False,
        orientation_mask: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def attach_positions_batch(self, joint_q: np.ndarray) -> np.ndarray: ...

    def attach_poses_batch(self, joint_q: np.ndarray) -> np.ndarray: ...


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
        *,
        target_up_directions: np.ndarray | None = None,
    ) -> DynamicsRetargetResult: ...


@dataclass(frozen=True, slots=True)
class _Motion:
    joint_q: np.ndarray
    target_gripper_open: np.ndarray
    target_attach_positions: np.ndarray
    target_approach_directions: np.ndarray
    pregrasp_mask: np.ndarray
    approach_validation_mask: np.ndarray
    attach_poses: np.ndarray
    attach_errors: np.ndarray
    target_up_directions: np.ndarray | None = None


def _validate_fr3_asset(assets_root: Path) -> dict:
    """Read the accepted USD and its actual sources, without inventing an MJCF pin."""

    from clothdojo.assets.workstations import _usd_dependencies

    root = assets_root.resolve()
    entry = root / "robots/fr3_duo/fr3_duo.usda"
    payload = entry.parent / "payloads/workstation.usdc"
    provenance = entry.parent / "provenance/dynamics.json"
    files, _renderer_modules = _usd_dependencies(entry, root)
    if payload not in files:
        raise RuntimeError(f"FR3 workstation does not reference its accepted payload: {payload}")
    # Use the same semantic reader as the runtime registry, including home,
    # attach, gripper topology and cameras; do not maintain another definition.
    robot_profile(FR3_DUO, assets_root=root)
    with provenance.open(encoding="utf-8") as stream:
        sources = json.load(stream)
    if not isinstance(sources, dict):
        raise ValueError(f"FR3 dynamics provenance must be an object: {provenance}")
    for field in ("franka_revision", "robotiq_revision"):
        if not isinstance(sources.get(field), str) or not sources[field].strip():
            raise ValueError(f"FR3 dynamics provenance is missing {field}: {provenance}")
    return {
        "format": "usd",
        "asset": entry.relative_to(root).as_posix(),
        "payload": payload.relative_to(root).as_posix(),
        "dynamics_provenance": provenance.relative_to(root).as_posix(),
        "sources": {
            "workstation": {
                "source": "https://franka.de/3d-assets",
                "asset": "FR3Duo_withVMkit.usd",
            },
            "arm_dynamics": {
                "source": "frankarobotics/franka_description",
                "revision": sources["franka_revision"],
            },
            "gripper_dynamics": {
                "source": "google-deepmind/mujoco_menagerie/robotiq_2f85",
                "revision": sources["robotiq_revision"],
            },
        },
    }


def _validate_target_asset(assets_root: Path, target_robot: str) -> dict:
    """Validate the USD actually consumed by IK and dynamics, not a legacy XML."""
    from clothdojo.assets.workstations import _usd_dependencies

    if target_robot == FR3_DUO:
        return _validate_fr3_asset(assets_root)
    if target_robot not in (ALOHA2, X5, UR5E_2F85):
        raise ValueError(f"retarget does not support {target_robot!r}")
    root = assets_root.resolve()
    entry = root / "robots" / target_robot / f"{target_robot}.usda"
    files, _renderer_modules = _usd_dependencies(entry, root)
    robot_profile(target_robot, assets_root=root)
    return {
        "format": "usd",
        "asset": entry.relative_to(root).as_posix(),
        "runtime_files": sorted(path.relative_to(root).as_posix() for path in files),
    }


def retarget(
    source: str | Path,
    output: str | Path,
    *,
    target_robot: str = ALOHA2,
    assets: str | Path = "assets",
    device: str = "cuda:0",
    devices: tuple[str, ...] | None = None,
    workers_per_device: int = 1,
    position_tolerance_m: float = 0.005,
) -> dict:
    """Retarget one trajectory or collection to a fixed robot profile."""

    if not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0.0:
        raise ValueError("position_tolerance_m must be positive and finite")
    worker_devices = tuple(devices) if devices is not None else (device,)
    if not worker_devices or len(set(worker_devices)) != len(worker_devices):
        raise ValueError("retarget devices must be nonempty and unique")
    if (
        isinstance(workers_per_device, bool)
        or not isinstance(workers_per_device, int)
        or workers_per_device < 1
    ):
        raise ValueError("workers_per_device must be a positive integer")
    if (len(worker_devices) > 1 or workers_per_device > 1) and any(
        re.fullmatch(r"cuda:[0-9]+", item) is None for item in worker_devices
    ):
        raise ValueError("parallel retarget requires explicit cuda:N devices")
    # Fill one slot on every GPU before assigning a second slot to any GPU.
    # MPS, when configured by the caller, lets these independent CUDA contexts
    # overlap their small IK/physics kernels without changing the rollout.
    worker_devices = worker_devices * workers_per_device
    source_root = Path(source).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"retarget output already exists: {output_root}")
    trajectories, is_collection = _source_trajectories(source_root)
    empty = [path for path in trajectories if TrajectoryReader(path).manifest.action_count == 0]
    if empty:
        raise ValueError(
            "retarget requires at least one action per trajectory; empty trajectories: "
            + ", ".join(str(path) for path in empty)
        )
    assets_root = Path(assets).expanduser().resolve()
    target_asset_metadata = _validate_target_asset(assets_root, target_robot)
    target_profile = robot_profile(target_robot, assets_root=assets_root)

    staging = output_root.with_name(f".{output_root.name}.tmp-{uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        tasks = [
            (
                index,
                trajectory,
                staging / (trajectory.relative_to(source_root) if is_collection else Path(".")),
            )
            for index, trajectory in enumerate(trajectories)
        ]
        worker_options = {
            "assets_root": assets_root,
            "target_robot": target_robot,
            "target_asset_metadata": target_asset_metadata,
            "position_tolerance_m": position_tolerance_m,
        }
        if len(worker_devices) == 1 or len(tasks) == 1:
            indexed_results = _retarget_partition(
                tasks,
                device=worker_devices[0],
                **worker_options,
            )
        else:
            partitions = _retarget_partitions(tasks, len(worker_devices))
            indexed_results = []
            # CUDA contexts must be created in spawned children, never forked.
            # Each worker owns its engines; multiple workers may share a GPU.
            with ProcessPoolExecutor(
                max_workers=len(partitions),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = [
                    pool.submit(
                        _retarget_partition,
                        partition,
                        device=worker_device,
                        **worker_options,
                    )
                    for partition, worker_device in zip(partitions, worker_devices, strict=False)
                ]
                for future in as_completed(futures):
                    indexed_results.extend(future.result())
        results = [result for _index, result in sorted(indexed_results)]
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
                "attach_alignment": "position_with_soft_roll",
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
        "attach_alignment": "position_with_soft_roll",
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


def _retarget_partitions(tasks: list[tuple[int, Path, Path]], count: int) -> list[list]:
    """Balance independent trajectories by action count, longest first."""
    weighted = [(TrajectoryReader(task[1]).manifest.action_count, task) for task in tasks]
    partitions = [[] for _ in range(min(count, len(tasks)))]
    loads = [0] * len(partitions)
    for weight, task in sorted(weighted, key=lambda item: (-item[0], item[1][0])):
        slot = min(range(len(loads)), key=loads.__getitem__)
        partitions[slot].append(task)
        loads[slot] += max(1, weight)
    return partitions


def _retarget_partition(
    tasks: list[tuple[int, Path, Path]],
    *,
    assets_root: Path,
    target_robot: str,
    target_asset_metadata: dict,
    position_tolerance_m: float,
    device: str,
) -> list[tuple[int, dict]]:
    import warp as wp

    # Model construction also has temporary Warp allocations. Scope the whole
    # worker, not only the final model, to avoid using another worker's GPU.
    with wp.ScopedDevice(device):
        profile = robot_profile(target_robot, assets_root=assets_root)
        kinematics = _RobotKinematics(profile, assets_root / profile.asset, device=device)
        timed_tasks = []
        for task in tasks:
            manifest = TrajectoryReader(task[1]).manifest
            timed_tasks.append(((manifest.control_fps, manifest.sim_substeps), task, manifest))
        # There are no cloth particles/contacts in these engines, so different
        # source garments share exactly the same target-robot dynamics.
        dynamics = None
        current_timing = None
        results = []
        for completed, (timing, (index, source, destination), manifest) in enumerate(
            sorted(timed_tasks, key=lambda item: item[0]),
            start=1,
        ):
            if manifest.scene.task_name not in _SUPPORTED_SOURCE_TASKS:
                raise RuntimeError(
                    "retarget source task must be shirt_flatten or shirt_fold"
                )
            asset_id = manifest.metadata.get("asset_id")
            if not isinstance(asset_id, str) or not asset_id:
                raise RuntimeError("retarget source trajectory asset_id is missing")
            started = perf_counter()
            if timing != current_timing:
                dynamics = RobotDynamicsRetargeter(
                    assets_root,
                    garment_id=asset_id,
                    control_fps=timing[0],
                    sim_substeps=timing[1],
                    device=device,
                    target_robot=target_robot,
                )
                current_timing = timing
            results.append(
                (
                    index,
                    _retarget_trajectory(
                        source,
                        destination,
                        kinematics=kinematics,
                        dynamics=dynamics,
                        position_tolerance_m=position_tolerance_m,
                        target_robot=target_robot,
                        target_asset_metadata=target_asset_metadata,
                    ),
                )
            )
            print(
                f"[retarget {device}] {completed}/{len(tasks)} {source.name}: "
                f"{perf_counter() - started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        return results


def _retarget_trajectory(
    source: Path,
    output: Path,
    *,
    kinematics: _Kinematics,
    dynamics: _Dynamics,
    position_tolerance_m: float,
    target_robot: str = ALOHA2,
    target_asset_metadata: dict | None = None,
) -> dict:
    reader = TrajectoryReader(source)
    source_robot = reader.manifest.scene.robot_id
    if source_robot != PIPER:
        raise RuntimeError(f"retarget source robot must be {PIPER!r}")
    source_task = reader.manifest.scene.task_name
    if source_task not in _SUPPORTED_SOURCE_TASKS:
        raise RuntimeError("retarget source task must be shirt_flatten or shirt_fold")
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
        control_fps=float(reader.manifest.control_fps),
    )
    fps = float(reader.manifest.control_fps)
    kinematic_poses = motion.attach_poses
    kinematic_attach = kinematic_poses[..., :3]
    kinematic_attach_errors = motion.attach_errors.astype(np.float32)
    kinematic_approach = _quat_rotate(
        kinematic_poses[..., 3:7], np.asarray((0.0, 0.0, 1.0))
    ).astype(np.float32)
    kinematic_approach_errors = _direction_angle(
        kinematic_approach, motion.target_approach_directions
    ).astype(np.float32)
    active_kinematic_approach_errors = kinematic_approach_errors[motion.approach_validation_mask]
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
        target_up_directions=motion.target_up_directions,
    )
    max_error = float(kinematic_attach_errors.max(initial=0.0))
    max_approach_error = float(active_kinematic_approach_errors.max(initial=0.0))
    up_error = 0.0
    if motion.target_up_directions is not None:
        up_error = float(
            _direction_angle(
                _quat_rotate(kinematic_poses[..., 3:7], np.asarray((0.0, 1.0, 0.0))),
                motion.target_up_directions,
            )[motion.approach_validation_mask].max(initial=0.0)
        )
    kinematic_valid = bool(max_error <= position_tolerance_m)
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
    active_dynamics_approach_errors = dynamics_approach_errors[motion.approach_validation_mask]
    max_dynamics_approach_error = float(active_dynamics_approach_errors.max(initial=0.0))
    dynamics_valid = bool(dynamics_result.validation["dynamics_valid"])
    validation = {
        "kinematic_valid": kinematic_valid,
        "dynamics_valid": dynamics_valid,
        "training_eligible": bool(kinematic_valid and dynamics_valid),
        "source_timeline_preserved": True,
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_constraint_scope": "none",
        "orientation_preference": "bounded_local_z_y_up",
        "roll_preference_rate_rad_s": _ROLL_PREFERENCE_RATE_RAD_S,
        "max_attach_position_error_m": max_error,
        "max_pregrasp_approach_error_rad": max_approach_error,
        "max_kinematic_up_axis_error_rad": up_error,
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
    if motion.target_up_directions is not None:
        metrics["target_up_direction"] = motion.target_up_directions
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
            "attach_alignment": "position_with_soft_roll",
            "target_asset": target_asset_metadata
            if target_asset_metadata is not None
            else {
                "format": "usd",
                "asset": target_profile.asset,
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
            asset_fingerprints={},
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
    control_fps: float = 50.0,
    initial_joint_q: np.ndarray | None = None,
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
    if not np.isfinite(control_fps) or control_fps <= 0:
        raise ValueError("control_fps must be positive and finite")
    # These persisted masks describe enforced orientation, not grasp events.
    # Position is the only task-space constraint, including before attachment.
    pregrasp = np.zeros_like(attached)
    approach_validation = np.zeros_like(attached)
    target_gripper_open = grippers.copy()
    target_approach = np.empty_like(targets)
    target_up = np.empty_like(target_approach)
    frame_count = len(targets)
    joint_q = np.empty((frame_count, len(kinematics.home)), dtype=np.float32)
    seed = np.asarray(
        kinematics.home if initial_joint_q is None else initial_joint_q,
        dtype=np.float32,
    ).copy()
    if seed.shape != np.asarray(kinematics.home).shape or not np.isfinite(seed).all():
        raise ValueError("initial_joint_q must match the finite kinematic home layout")
    for index in range(frame_count):
        positioned = kinematics.solve(seed, targets[index])
        achieved = kinematics.attach_poses_batch(positioned[None])[0]
        target_rotations = _upright_rotations(achieved[None, :, 3:7])[0]
        seed = _prefer_upright_roll(
            kinematics,
            positioned,
            targets[index],
            achieved,
            target_rotations,
            max_step_rad=_ROLL_PREFERENCE_RATE_RAD_S / control_fps,
            position_tolerance_m=position_tolerance_m,
        )
        target_approach[index] = _quat_rotate(target_rotations, np.asarray((0.0, 0.0, 1.0)))
        target_up[index] = _quat_rotate(target_rotations, np.asarray((0.0, 1.0, 0.0)))
        joint_q[index] = seed
    _set_gripper_positions(
        joint_q,
        target_gripper_open,
        kinematics.gripper_indices,
        kinematics.gripper_closed_positions,
        kinematics.gripper_open_positions,
    )
    achieved_poses = kinematics.attach_poses_batch(joint_q)
    errors = np.linalg.norm(achieved_poses[..., :3] - targets, axis=-1)
    return _Motion(
        joint_q=joint_q,
        target_gripper_open=target_gripper_open,
        target_attach_positions=targets,
        target_approach_directions=target_approach,
        pregrasp_mask=pregrasp,
        approach_validation_mask=approach_validation,
        attach_poses=achieved_poses,
        attach_errors=errors,
        target_up_directions=target_up,
    )


def _prefer_upright_roll(
    kinematics: _Kinematics,
    positioned: np.ndarray,
    targets: np.ndarray,
    achieved: np.ndarray,
    upright: np.ndarray,
    *,
    max_step_rad: float,
    position_tolerance_m: float,
) -> np.ndarray:
    """Try a small local-Z twist, then restore position and accept per hand.

    No source orientation enters this preference. Infeasible roll loses to
    position/continuity, rather than becoming an orientation tracking failure.
    """
    current = achieved[:, 3:7]
    alignment = np.clip(np.abs(np.sum(current * upright, axis=-1)), 0, 1)
    angle = 2 * np.arccos(alignment)
    if np.all(angle < 1e-6):
        return positioned.copy()
    fraction = np.minimum(1, max_step_rad / np.maximum(angle, 1e-8))
    preferred = _upright_rotations(current[None], alignment_fraction=fraction[None])[0]
    proposal = kinematics.solve(positioned, targets, preferred, align_roll=True)
    arms = np.asarray(kinematics.arm_indices).reshape(2, -1)
    step = proposal[arms] - positioned[arms]
    scale = np.minimum(1, max_step_rad / np.maximum(np.max(np.abs(step), axis=1), 1e-8))
    bounded = positioned.copy()
    bounded[arms] += step * scale[:, None]
    # The secondary step must not buy a nicer wrist angle with TCP drift.
    candidate = kinematics.solve(bounded, targets)
    poses = kinematics.attach_poses_batch(candidate[None])[0]
    baseline_error = np.linalg.norm(achieved[:, :3] - targets, axis=-1)
    error = np.linalg.norm(poses[:, :3] - targets, axis=-1)
    old_y = _quat_rotate(current, np.asarray((0.0, 1.0, 0.0)))[:, 2]
    new_y = _quat_rotate(poses[:, 3:7], np.asarray((0.0, 1.0, 0.0)))[:, 2]
    old_z = _quat_rotate(current, np.asarray((0.0, 0.0, 1.0)))
    new_z = _quat_rotate(poses[:, 3:7], np.asarray((0.0, 0.0, 1.0)))
    step_size = np.max(np.abs(candidate[arms] - positioned[arms]), axis=1)
    accept = (
        (error <= np.minimum(position_tolerance_m, baseline_error + 1e-5))
        & (step_size <= max_step_rad + 1e-6)
        & (new_y >= old_y)
        # Do not turn the roll preference into a pitch/yaw preference after
        # reprojection. This checks only this optional step, never source Z.
        & (np.linalg.norm(new_z - old_z, axis=-1) <= 1e-3)
    )
    result = positioned.copy()
    for side in np.flatnonzero(accept):
        result[arms[side]] = candidate[arms[side]]
    return result


def _set_gripper_positions(
    joint_q: np.ndarray,
    open_state: np.ndarray,
    gripper_indices: np.ndarray,
    closed_positions: np.ndarray,
    open_positions: np.ndarray,
) -> None:
    openness = np.clip(open_state, 0.0, 1.0)
    closed = np.asarray(closed_positions, dtype=np.float32).reshape(1, 1, -1)
    opened = np.asarray(open_positions, dtype=np.float32).reshape(1, 1, -1)
    positions = closed + openness[:, :, None] * (opened - closed)
    joint_q[:, gripper_indices] = positions.reshape(len(joint_q), len(gripper_indices))


class _RobotKinematics:
    def __init__(
        self,
        profile,
        asset: Path,
        *,
        device: str,
        iterations: int = 40,
    ) -> None:
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
        self._arm_lower = self.model.joint_limit_lower.numpy()[self.arm_dof_indices]
        self._arm_upper = self.model.joint_limit_upper.numpy()[self.arm_dof_indices]
        self._solve_graphs = {}
        self.iterations = int(iterations)
        self.q_count = len(self.home)
        self._q_in = wp.zeros((1, self.q_count), dtype=wp.float32, device=self.model.device)
        self._q_out = wp.zeros_like(self._q_in)
        self._left_targets = wp.zeros((1,), dtype=wp.vec3, device=self.model.device)
        self._right_targets = wp.zeros((1,), dtype=wp.vec3, device=self.model.device)
        self._left_approach_targets = wp.zeros((1,), dtype=wp.vec3, device=self.model.device)
        self._right_approach_targets = wp.zeros((1,), dtype=wp.vec3, device=self.model.device)
        self._up_targets = [
            wp.zeros((1,), dtype=wp.vec3, device=self.model.device) for _ in range(2)
        ]
        approach_offsets = []
        for frame in profile.attach_frames:
            local_axis = _quat_rotate(
                np.asarray(frame.rotation_xyzw, dtype=np.float64),
                np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
            )
            approach_offsets.append(
                np.asarray(frame.position, dtype=np.float64) + _IK_APPROACH_MARKER_M * local_axis
            )

        self._orientation_objectives = {False: [], True: []}

        def objectives(*, include_approach: bool, include_up: bool = False):
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
            if include_approach:
                for side, target in enumerate(
                    (
                        self._left_approach_targets,
                        self._right_approach_targets,
                    )
                ):
                    objective = ik.IKObjectivePosition(
                        self.end_effectors[side],
                        wp.vec3(*approach_offsets[side]),
                        target,
                        _IK_POSITION_WEIGHT,
                    )
                    values.append(objective)
                    self._orientation_objectives[include_up].append((side, objective))
            if include_up:
                for side, frame in enumerate(profile.attach_frames):
                    up = _quat_rotate(np.asarray(frame.rotation_xyzw), np.asarray((0.0, 1.0, 0.0)))
                    offset = np.asarray(frame.position) + _IK_APPROACH_MARKER_M * up
                    objective = ik.IKObjectivePosition(
                        self.end_effectors[side],
                        wp.vec3(*offset),
                        self._up_targets[side],
                        _IK_POSITION_WEIGHT,
                    )
                    values.append(objective)
                    self._orientation_objectives[include_up].append((side, objective))
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
            objectives=objectives(include_approach=False),
            jacobian_mode=ik.IKJacobianType.MIXED,
            lambda_initial=0.1,
        )
        self._pose_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=objectives(include_approach=True),
            jacobian_mode=ik.IKJacobianType.MIXED,
            lambda_initial=0.1,
        )
        self._upright_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=objectives(include_approach=True, include_up=True),
            jacobian_mode=ik.IKJacobianType.MIXED,
            lambda_initial=0.1,
        )
        self._fk_q = wp.zeros(self.q_count, dtype=wp.float32, device=self.model.device)
        if int(self.model.joint_dof_count) != int(profile.joint_dof_count):
            raise RuntimeError(
                f"pinned {profile.robot_id} velocity topology differs from its profile"
            )
        self._fk_state = self.model.state()

    def solve(
        self,
        seed: np.ndarray,
        target_positions: np.ndarray,
        target_rotations: np.ndarray | None = None,
        *,
        align_roll: bool = False,
        orientation_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        seed = np.asarray(seed, dtype=np.float32).reshape(self.q_count)
        targets = np.asarray(target_positions, dtype=np.float32).reshape(2, 3)
        self._left_targets.assign(targets[None, 0])
        self._right_targets.assign(targets[None, 1])
        solver = self._position_solver
        graph_key = ("position",)
        if target_rotations is not None:
            active = (
                np.ones(2, dtype=bool)
                if orientation_mask is None
                else np.asarray(orientation_mask, dtype=bool)
            )
            if active.shape != (2,):
                raise ValueError("orientation mask must contain one flag per hand")
            # Newton 1.5 reads objective weights at kernel launch, so each
            # orientation mask needs its own graph. Target arrays stay mutable.
            for side, objective in self._orientation_objectives[align_roll]:
                objective.weight = _IK_POSITION_WEIGHT if active[side] else 0.0
            rotations = np.asarray(target_rotations, dtype=np.float32).reshape(2, 4)
            directions = _quat_rotate(
                rotations,
                np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            )
            approach_targets = targets + _IK_APPROACH_MARKER_M * directions
            self._left_approach_targets.assign(approach_targets[None, 0])
            self._right_approach_targets.assign(approach_targets[None, 1])
            solver = self._pose_solver
            graph_key = ("upright" if align_roll else "pose", *active.tolist())
            if align_roll:
                up = _quat_rotate(rotations, np.asarray((0.0, 1.0, 0.0)))
                for side in range(2):
                    self._up_targets[side].assign(
                        (targets[side] + _IK_APPROACH_MARKER_M * up[side])[None]
                    )
                solver = self._upright_solver
        self._q_in.assign(seed[None])
        if self.model.device.is_cuda:
            graph = self._solve_graphs.get(graph_key)
            if graph is None:
                solver.reset()
                solver.step(self._q_in, self._q_out, iterations=self.iterations)
                self.wp.synchronize_device(self.model.device)
                with self.wp.ScopedCapture(
                    device=self.model.device,
                    force_module_load=False,
                ) as capture:
                    solver.reset()
                    solver.step(self._q_in, self._q_out, iterations=self.iterations)
                graph = self._solve_graphs[graph_key] = capture.graph
            self.wp.capture_launch(graph)
        else:
            solver.reset()
            solver.step(self._q_in, self._q_out, iterations=self.iterations)
        result = self._q_out.numpy()[0].astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError(f"Newton IK returned non-finite {self.profile.robot_id} coordinates")
        result[self.arm_indices] = np.clip(
            result[self.arm_indices],
            self._arm_lower,
            self._arm_upper,
        )
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


def _upright_rotations(
    rotations: np.ndarray,
    *,
    initial_rotations: np.ndarray | None = None,
    alignment_fraction: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate only about local Z, maximizing +Y height away from the poles.

    When Z is within about 6 degrees of vertical, parallel-transport the
    preceding Y instead. Blend back to world-up between about 6 and 12 degrees
    so a poorly defined projected up vector cannot cause an abrupt half turn.
    """
    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 4) or not len(values):
        raise ValueError("upright rotations require nonempty [N, 2, 4] quaternions")
    if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=-1) < 1e-8):
        raise ValueError("upright rotations must be finite nonzero quaternions")
    values = values / np.linalg.norm(values, axis=-1, keepdims=True)
    fractions = (
        np.ones(values.shape[:2])
        if alignment_fraction is None
        else np.asarray(alignment_fraction, dtype=np.float64)
    )
    if fractions.shape != values.shape[:2] or not np.isfinite(fractions).all():
        raise ValueError("upright alignment fractions must match frames and hands")
    result = np.empty_like(values)
    previous = (
        values[0].copy()
        if initial_rotations is None
        else np.asarray(initial_rotations, dtype=np.float64).copy()
    )
    if (
        previous.shape != (2, 4)
        or not np.isfinite(previous).all()
        or np.any(np.linalg.norm(previous, axis=-1) < 1e-8)
    ):
        raise ValueError("initial upright rotations must contain two nonzero quaternions")
    previous /= np.linalg.norm(previous, axis=-1, keepdims=True)
    for index, current in enumerate(values):
        z = _quat_rotate(current, np.asarray((0.0, 0.0, 1.0)))
        transported = _align_quaternion_axis(previous, z)
        y = _quat_rotate(transported, np.asarray((0.0, 1.0, 0.0)))
        projected = np.asarray((0.0, 0.0, 1.0)) - z[:, 2, None] * z
        strength = np.linalg.norm(projected, axis=-1)
        up = projected / np.maximum(strength[:, None], 1e-8)
        angle = np.arctan2(np.sum(z * np.cross(y, up), axis=-1), np.sum(y * up, axis=-1))
        blend = np.clip((strength - 0.1) / 0.1, 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        half_angle = 0.5 * blend * angle * np.clip(fractions[index], 0.0, 1.0)
        delta = np.concatenate(
            (z * np.sin(half_angle)[:, None], np.cos(half_angle)[:, None]), axis=-1
        )
        previous = _quat_multiply(delta, transported)
        result[index] = previous
    return result.astype(np.float32)


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
