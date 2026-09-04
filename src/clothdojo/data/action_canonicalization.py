"""Create an independent trajectory snapshot with runtime-bounded actions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from uuid import uuid4

import numpy as np

from clothdojo.data.annotations import build_good_segment_plan
from clothdojo.robots import PIPER, robot_profile
from clothdojo.trajectory import RUNTIME_ACTION_SEMANTICS

CANONICALIZATION_SCHEMA = "clothdojo_action_canonicalization"
CANONICALIZATION_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PiperActionBounds:
    joint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    fingerprint: str


@dataclass(frozen=True, slots=True)
class TrajectoryPlan:
    source: Path
    manifest: dict
    bounds: PiperActionBounds
    counts: dict
    state_bytes: int
    source_fingerprint: dict[str, str | int]


def canonicalize_bounded_actions(
    joint_target: np.ndarray,
    gripper_open_command: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact float32 targets produced by runtime range clipping."""

    arms = np.asarray(joint_target)
    grippers = np.asarray(gripper_open_command)
    lower_values = np.asarray(lower)
    upper_values = np.asarray(upper)
    if arms.dtype != np.float32 or grippers.dtype != np.float32:
        raise TypeError("trajectory actions must use float32")
    if lower_values.dtype != np.float32 or upper_values.dtype != np.float32:
        raise TypeError("runtime action bounds must use float32")
    if arms.ndim != 2 or arms.shape[1:] != lower_values.shape:
        raise ValueError("joint_target and runtime bounds have different layouts")
    if upper_values.shape != lower_values.shape or np.any(lower_values >= upper_values):
        raise ValueError("runtime action bounds are invalid")
    if grippers.shape != (len(arms), 2):
        raise ValueError("gripper_open_command must have shape [frames, 2]")
    if not np.isfinite(arms).all() or not np.isfinite(grippers).all():
        raise ValueError("trajectory actions must be finite")
    return (
        np.ascontiguousarray(np.clip(arms, lower_values, upper_values), dtype=np.float32),
        np.ascontiguousarray(np.clip(grippers, 0.0, 1.0), dtype=np.float32),
    )


def canonicalize_action_collection(
    source: str | Path,
    output: str | Path | None = None,
    *,
    plan_only: bool = False,
    expected: dict[str, int] | None = None,
) -> dict:
    """Publish a complete snapshot while leaving every source file untouched."""

    source_root = Path(source).expanduser().resolve()
    output_root = None if output is None else Path(output).expanduser().resolve()
    if output_root is not None:
        _validate_output_target(source_root, output_root)
    plans, skipped_incomplete = _build_plans(source_root)
    report = _plan_report(
        source_root,
        output_root,
        plans,
        skipped_incomplete=skipped_incomplete,
    )
    if expected is not None:
        _validate_expected_report(report, expected)
    if plan_only:
        return report
    if output_root is None:
        raise ValueError("canonical action publication requires an output directory")
    required_bytes = report["state_bytes"] + 1024**3
    storage_root = output_root.parent
    while not storage_root.exists():
        storage_root = storage_root.parent
    free_bytes = shutil.disk_usage(storage_root).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"canonical snapshot needs at least {required_bytes} free bytes, got {free_bytes}"
        )

    staging = output_root.with_name(f".{output_root.name}.tmp-{uuid4().hex}")
    trajectories_root = staging / "trajectories"
    try:
        trajectories_root.mkdir(parents=True)
        workers = min(4, len(plans))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(
                pool.map(
                    lambda plan: _write_trajectory_snapshot(
                        plan,
                        trajectories_root / plan.source.name,
                    ),
                    plans,
                )
            )
        annotations = _copy_annotations(source_root, trajectories_root)
        manifest = {
            "schema": CANONICALIZATION_SCHEMA,
            "format_version": CANONICALIZATION_FORMAT_VERSION,
            "status": "complete",
            "source": str(source_root),
            "source_is_runtime_dependency": False,
            "action_semantics": RUNTIME_ACTION_SEMANTICS,
            "operation": "joint_and_gripper_range_clip",
            "trajectories": records,
            "annotations": annotations,
            **{
                key: report[key]
                for key in (
                    "input_trajectories",
                    "skipped_incomplete_trajectories",
                    "frames",
                    "changed_frames",
                    "changed_arm_elements",
                    "changed_gripper_elements",
                    "maximum_arm_clip_rad",
                    "maximum_gripper_clip",
                    "state_bytes",
                )
            },
        }
        _write_json(staging / "action_canonicalization.json", manifest)
        _validate_snapshot(staging, source_root, plans, records, annotations)
        _fsync_tree_directories(staging)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
        _fsync_directory(output_root.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**report, "output": str(output_root)}


def _build_plans(source_root: Path) -> tuple[list[TrajectoryPlan], int]:
    trajectories_root = source_root / "trajectories"
    if not trajectories_root.is_dir():
        raise RuntimeError("canonical action source must contain trajectories/")
    plans = []
    skipped = 0
    bounds_cache: dict[str, PiperActionBounds] = {}
    for manifest_path in sorted(trajectories_root.glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete":
            skipped += 1
            continue
        source = manifest_path.parent
        if {path.name for path in source.iterdir()} != {
            "manifest.json",
            "actions.npz",
            "states.npz",
        }:
            raise RuntimeError(f"source trajectory must contain exactly three files: {source}")
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict) or "action_semantics" in metadata:
            raise RuntimeError(f"source action_semantics must be absent: {source}")
        states_path = source / "states.npz"
        if not states_path.is_file():
            raise RuntimeError(f"complete trajectory states.npz is missing: {source}")
        bounds = _resolved_bounds(manifest, bounds_cache)
        source_fingerprint = _source_fingerprint(source)
        actions = _load_actions(source / "actions.npz", manifest)
        arms, grippers = canonicalize_bounded_actions(
            actions["joint_target"],
            actions["gripper_open_command"],
            lower=bounds.lower,
            upper=bounds.upper,
        )
        plans.append(
            TrajectoryPlan(
                source=source,
                manifest=manifest,
                bounds=bounds,
                counts=_change_counts(actions, arms, grippers),
                state_bytes=states_path.stat().st_size,
                source_fingerprint=source_fingerprint,
            )
        )
        if _source_fingerprint(source) != source_fingerprint:
            raise RuntimeError(f"source trajectory changed while planning: {source}")
    if not plans:
        raise RuntimeError(f"source collection has no complete trajectories: {source_root}")
    return plans, skipped


def _plan_report(
    source_root: Path,
    output_root: Path | None,
    plans: list[TrajectoryPlan],
    *,
    skipped_incomplete: int,
) -> dict:
    totals = _empty_totals()
    contracts: dict[str, int] = {}
    for plan in plans:
        _add_counts(totals, plan.counts)
        contracts[plan.bounds.fingerprint] = contracts.get(plan.bounds.fingerprint, 0) + 1
    if len(contracts) != 1:
        raise RuntimeError(f"source trajectories use different runtime limits: {contracts}")
    annotations = _tree_fingerprint(source_root / "trajectories" / "annotations")
    return {
        "source": str(source_root),
        "output": None if output_root is None else str(output_root),
        "input_trajectories": len(plans),
        "skipped_incomplete_trajectories": skipped_incomplete,
        "action_semantics": RUNTIME_ACTION_SEMANTICS,
        "runtime_limit_groups": contracts,
        "annotation_files": annotations["files"],
        "annotation_bytes": annotations["bytes"],
        "state_bytes": sum(plan.state_bytes for plan in plans),
        **totals,
    }


def _write_trajectory_snapshot(plan: TrajectoryPlan, output: Path) -> dict:
    source = plan.source
    before = _source_fingerprint(source)
    if before != plan.source_fingerprint:
        raise RuntimeError(f"source trajectory changed after planning: {source}")
    actions = _load_actions(source / "actions.npz", plan.manifest)
    arms, grippers = canonicalize_bounded_actions(
        actions["joint_target"],
        actions["gripper_open_command"],
        lower=plan.bounds.lower,
        upper=plan.bounds.upper,
    )

    output.mkdir()
    source_states = source / "states.npz"
    output_states = output / "states.npz"
    source_states_sha256 = _copy_verified_state(source_states, output_states)

    actions_path = output / "actions.npz"
    with actions_path.open("wb") as file:
        np.savez_compressed(
            file,
            sample_index=actions["sample_index"],
            joint_target=arms,
            gripper_open_command=grippers,
        )
        file.flush()
        os.fsync(file.fileno())
    manifest = copy.deepcopy(plan.manifest)
    metadata = manifest.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError(f"trajectory metadata must be an object: {source}")
    metadata["action_semantics"] = RUNTIME_ACTION_SEMANTICS
    _write_json(output / "manifest.json", manifest)

    after = _source_fingerprint(source)
    if after != before:
        raise RuntimeError(f"source trajectory changed during publication: {source}")
    if os.path.samefile(source_states, output_states):
        raise RuntimeError("snapshot states must not be hard-linked to the source")
    if output_states.stat().st_size != source_states.stat().st_size:
        raise RuntimeError(f"copied trajectory state size differs: {source.name}")
    if _npz_archive_directory_fingerprint(output_states) != before["states_archive"]:
        raise RuntimeError(f"copied trajectory state archive differs: {source.name}")
    output_actions = _load_actions(actions_path, manifest)
    if not np.array_equal(output_actions["sample_index"], actions["sample_index"]):
        raise RuntimeError(f"snapshot action timeline differs: {source.name}")
    if not np.array_equal(output_actions["joint_target"], arms) or not np.array_equal(
        output_actions["gripper_open_command"], grippers
    ):
        raise RuntimeError(f"snapshot canonical actions differ: {source.name}")
    return {
        "trajectory_id": source.name,
        "source_manifest_sha256": before["manifest"],
        "source_actions_sha256": before["actions"],
        "source_states_archive_fingerprint": before["states_archive"],
        "source_states_sha256": source_states_sha256,
        "output_manifest_sha256": _sha256(output / "manifest.json"),
        "output_actions_sha256": _sha256(actions_path),
        "runtime_limits_sha256": plan.bounds.fingerprint,
        **plan.counts,
    }


def _validate_snapshot(
    staging: Path,
    source_root: Path,
    plans: list[TrajectoryPlan],
    records: list[dict],
    annotations: dict,
) -> None:
    if len(records) != len(plans):
        raise RuntimeError("snapshot trajectory count differs")
    for plan, record in zip(plans, records, strict=True):
        if record.get("trajectory_id") != plan.source.name:
            raise RuntimeError("snapshot trajectory order differs")
        output = staging / "trajectories" / plan.source.name
        source_states = plan.source / "states.npz"
        output_states = output / "states.npz"
        if not output_states.is_file() or os.path.samefile(source_states, output_states):
            raise RuntimeError(f"snapshot state independence differs: {plan.source.name}")
        if output_states.is_symlink() or source_states.is_symlink():
            raise RuntimeError(f"snapshot states must be regular files: {plan.source.name}")
        if {path.name for path in output.iterdir()} != {
            "manifest.json",
            "actions.npz",
            "states.npz",
        }:
            raise RuntimeError(f"snapshot trajectory file set differs: {plan.source.name}")
        if _source_fingerprint(plan.source)["manifest"] != record["source_manifest_sha256"]:
            raise RuntimeError(f"source manifest changed: {plan.source.name}")
        if _source_fingerprint(plan.source) != plan.source_fingerprint:
            raise RuntimeError(f"source trajectory changed before publication: {plan.source.name}")
        expected_manifest = copy.deepcopy(plan.manifest)
        expected_manifest.setdefault("metadata", {})["action_semantics"] = (
            RUNTIME_ACTION_SEMANTICS
        )
        if _read_json(output / "manifest.json") != expected_manifest:
            raise RuntimeError(f"snapshot manifest differs: {plan.source.name}")
    output_annotations = _tree_fingerprint(staging / "trajectories" / "annotations")
    if output_annotations != annotations:
        raise RuntimeError("snapshot annotations differ from source")
    if _tree_fingerprint(source_root / "trajectories" / "annotations") != annotations:
        raise RuntimeError("source annotations changed during publication")
    source_selection = build_good_segment_plan(source_root, min_segment_frames=1)
    output_selection = build_good_segment_plan(staging, min_segment_frames=1)
    if source_selection != output_selection:
        raise RuntimeError("snapshot annotation selection differs from source")
    trajectory_dirs = {
        path.name
        for path in (staging / "trajectories").iterdir()
        if path.is_dir() and path.name != "annotations"
    }
    if trajectory_dirs != {plan.source.name for plan in plans}:
        raise RuntimeError("snapshot trajectory directory set differs")
    current_manifests = {
        path.parent.name
        for path in (source_root / "trajectories").glob("*/manifest.json")
        if _read_json(path).get("status") == "complete"
    }
    if current_manifests != {plan.source.name for plan in plans}:
        raise RuntimeError("source trajectory inventory changed before publication")


def _resolved_bounds(
    manifest: dict,
    cache_by_source: dict[str, PiperActionBounds],
) -> PiperActionBounds:
    metadata = manifest.get("metadata")
    scene = manifest.get("scene")
    if not isinstance(metadata, dict) or not isinstance(scene, dict):
        raise RuntimeError("Piper trajectory manifest metadata/scene is invalid")
    robot_id = scene.get("robot_id", scene.get("robot_name"))
    if robot_id != "piper":
        raise RuntimeError(f"action canonicalization requires Piper, got {robot_id!r}")
    names_value = metadata.get("joint_names")
    scene_config = metadata.get("scene_config")
    robot = scene_config.get("robot") if isinstance(scene_config, dict) else None
    if not isinstance(names_value, list) or not isinstance(robot, dict):
        raise RuntimeError("Piper manifest must embed joint_names and scene_config.robot")
    names = tuple(str(value) for value in names_value)
    runtime_names, runtime_lower, runtime_upper = _runtime_piper_bounds()
    if names != runtime_names:
        raise RuntimeError("Piper manifest joint_names differ from the runtime profile order")
    usd_value = robot.get("usd_path")
    if not isinstance(usd_value, str) or not usd_value:
        raise RuntimeError("Piper manifest robot.usd_path is missing")
    usd_path = Path(usd_value).expanduser().resolve()
    if not usd_path.is_file():
        raise RuntimeError(f"Piper source USD is unavailable: {usd_path}")
    overrides = robot.get("arm_joint_position_limit_overrides_rad", [])
    cache_key = _json_sha256(
        {
            "joint_names": names,
            "usd_path": str(usd_path),
            "usd_sha256": _sha256(usd_path),
            "overrides": overrides,
        }
    )
    cached = cache_by_source.get(cache_key)
    if cached is not None:
        return cached
    lower_by_name, upper_by_name = _usd_revolute_limits(usd_path)
    _apply_limit_overrides(lower_by_name, upper_by_name, overrides)
    missing = sorted(set(names) - set(lower_by_name))
    if missing:
        raise RuntimeError(f"Piper USD lacks manifest arm joints: {missing}")
    lower = np.asarray([lower_by_name[name] for name in names], dtype=np.float32)
    upper = np.asarray([upper_by_name[name] for name in names], dtype=np.float32)
    if not np.array_equal(lower, runtime_lower) or not np.array_equal(upper, runtime_upper):
        raise RuntimeError("trajectory joint limits differ from the current Piper runtime")
    fingerprint = _json_sha256(
        {
            "joint_names": names,
            "lower_rad_float32": lower.tolist(),
            "upper_rad_float32": upper.tolist(),
        }
    )
    result = PiperActionBounds(
        joint_names=names,
        lower=lower,
        upper=upper,
        fingerprint=fingerprint,
    )
    cache_by_source[cache_key] = result
    return result


@cache
def _runtime_piper_bounds() -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    profile = robot_profile(PIPER)
    runtime_usd = Path(__file__).resolve().parents[3] / "assets" / profile.asset
    lower_by_name, upper_by_name = _usd_revolute_limits(runtime_usd)
    names = profile.arm_joint_names
    lower = np.asarray([lower_by_name[name] for name in names], dtype=np.float32)
    upper = np.asarray([upper_by_name[name] for name in names], dtype=np.float32)
    lower.setflags(write=False)
    upper.setflags(write=False)
    return names, lower, upper


def _usd_revolute_limits(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"cannot open Piper USD: {path}")
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for prim in stage.Traverse():
        parts = str(prim.GetPath()).strip("/").split("/")
        if len(parts) < 2 or parts[-2] not in {"left", "right"}:
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            continue
        key = f"{parts[-2]}/{parts[-1]}"
        lower_degrees = joint.GetLowerLimitAttr().Get()
        upper_degrees = joint.GetUpperLimitAttr().Get()
        if lower_degrees is None or upper_degrees is None:
            continue
        lower[key] = float(np.deg2rad(float(lower_degrees)))
        upper[key] = float(np.deg2rad(float(upper_degrees)))
    return lower, upper


def _apply_limit_overrides(
    lower: dict[str, float],
    upper: dict[str, float],
    overrides,
) -> None:
    if isinstance(overrides, dict):
        items = [(name, *values) for name, values in overrides.items()]
    elif isinstance(overrides, list):
        items = overrides
    else:
        raise RuntimeError("Piper joint-limit overrides must be a list or object")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise RuntimeError("Piper joint-limit override entries must be [name, lower, upper]")
        name, lo, hi = item
        key = str(name)
        if key in seen:
            raise RuntimeError(f"duplicate Piper joint-limit override: {key}")
        seen.add(key)
        if key not in lower:
            raise RuntimeError(f"Piper joint-limit override names an unknown joint: {key}")
        lo_value = float(lo)
        hi_value = float(hi)
        if not np.isfinite(lo_value) or not np.isfinite(hi_value) or lo_value >= hi_value:
            raise RuntimeError(f"Piper joint-limit override is invalid: {key}")
        if lo_value < lower[key] or hi_value > upper[key]:
            raise RuntimeError(f"Piper joint-limit override widens the asset range: {key}")
        lower[key] = lo_value
        upper[key] = hi_value


def _load_actions(path: Path, manifest: dict) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "sample_index",
                "joint_target",
                "gripper_open_command",
            }:
                raise RuntimeError(f"trajectory action archive fields differ: {path}")
            actions = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read trajectory actions: {path}") from exc
    count = manifest.get("action_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RuntimeError(f"trajectory action_count is invalid: {path.parent}")
    if actions["sample_index"].dtype != np.int64 or not np.array_equal(
        actions["sample_index"], np.arange(count, dtype=np.int64)
    ):
        raise RuntimeError(f"trajectory action sample_index differs: {path}")
    if actions["joint_target"].shape != (count, 12) or actions[
        "gripper_open_command"
    ].shape != (count, 2):
        raise RuntimeError(f"trajectory action shapes differ: {path}")
    if (
        actions["joint_target"].dtype != np.float32
        or actions["gripper_open_command"].dtype != np.float32
    ):
        raise RuntimeError(f"trajectory action dtypes differ: {path}")
    return actions


def _source_fingerprint(source: Path) -> dict[str, str | int]:
    states_path = source / "states.npz"
    stat = states_path.stat()
    return {
        "manifest": _sha256(source / "manifest.json"),
        "actions": _sha256(source / "actions.npz"),
        "states_archive": _npz_archive_directory_fingerprint(states_path),
        "states_size": stat.st_size,
        "states_mtime_ns": stat.st_mtime_ns,
        "states_inode": stat.st_ino,
    }


def _npz_archive_directory_fingerprint(path: Path) -> str:
    """Fingerprint an NPZ container without reading its array payloads."""

    try:
        with zipfile.ZipFile(path) as archive:
            members = [
                {
                    "name": info.filename,
                    "crc32": info.CRC,
                    "compressed_size": info.compress_size,
                    "size": info.file_size,
                    "compression": info.compress_type,
                }
                for info in archive.infolist()
            ]
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"cannot fingerprint trajectory archive: {path}") from exc
    value = {"archive_size": path.stat().st_size, "members": members}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _copy_verified_state(source: Path, output: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as input_file, output.open("xb") as output_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            output_file.write(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    shutil.copystat(source, output, follow_symlinks=False)
    _fsync_file(output)
    source_sha256 = digest.hexdigest()
    if _sha256(output) != source_sha256:
        raise RuntimeError(f"copied trajectory state SHA-256 differs: {source}")
    return source_sha256


def _copy_annotations(source_root: Path, trajectories_root: Path) -> dict:
    source = source_root / "trajectories" / "annotations"
    before = _tree_fingerprint(source)
    if source.is_dir():
        shutil.copytree(source, trajectories_root / "annotations", copy_function=shutil.copy2)
        for path in (trajectories_root / "annotations").rglob("*"):
            if path.is_file():
                _fsync_file(path)
    after = _tree_fingerprint(source)
    if after != before:
        raise RuntimeError("source annotations changed during publication")
    output = _tree_fingerprint(trajectories_root / "annotations")
    if output != before:
        raise RuntimeError("copied annotations differ from source")
    return before


def _change_counts(actions: dict, arms: np.ndarray, grippers: np.ndarray) -> dict:
    source_arm = np.asarray(actions["joint_target"])
    source_gripper = np.asarray(actions["gripper_open_command"])
    arm_changed = source_arm != arms
    gripper_changed = source_gripper != grippers
    arm_delta = np.abs(source_arm.astype(np.float64) - arms.astype(np.float64))
    gripper_delta = np.abs(
        source_gripper.astype(np.float64) - grippers.astype(np.float64)
    )
    changed_frames = np.any(arm_changed, axis=1) | np.any(gripper_changed, axis=1)
    return {
        "frames": len(arms),
        "changed_frames": int(np.count_nonzero(changed_frames)),
        "changed_arm_elements": int(np.count_nonzero(arm_changed)),
        "changed_gripper_elements": int(np.count_nonzero(gripper_changed)),
        "maximum_arm_clip_rad": float(arm_delta.max(initial=0.0)),
        "maximum_gripper_clip": float(gripper_delta.max(initial=0.0)),
    }


def _empty_totals() -> dict:
    return {
        "frames": 0,
        "changed_frames": 0,
        "changed_arm_elements": 0,
        "changed_gripper_elements": 0,
        "maximum_arm_clip_rad": 0.0,
        "maximum_gripper_clip": 0.0,
    }


def _add_counts(total: dict, values: dict) -> None:
    for key in (
        "frames",
        "changed_frames",
        "changed_arm_elements",
        "changed_gripper_elements",
    ):
        total[key] += values[key]
    for key in ("maximum_arm_clip_rad", "maximum_gripper_clip"):
        total[key] = max(total[key], values[key])


def _validate_output_target(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"canonical action output already exists: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("canonical action output must be outside the source collection")


def _validate_expected_report(report: dict, expected: dict[str, int]) -> None:
    unknown = set(expected) - set(report)
    if unknown:
        raise ValueError(f"expected snapshot report has unknown fields: {sorted(unknown)}")
    mismatched = {
        key: (report[key], value)
        for key, value in expected.items()
        if report[key] != value
    }
    if mismatched:
        raise RuntimeError(f"source collection differs from the audited plan: {mismatched}")


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _tree_fingerprint(root: Path) -> dict:
    if not root.exists():
        return {"files": 0, "bytes": 0, "sha256": _json_sha256([])}
    if not root.is_dir():
        raise RuntimeError(f"expected directory: {root}")
    records = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": size,
                "sha256": _sha256(path),
            }
        )
    return {
        "files": len(records),
        "bytes": total_bytes,
        "sha256": _json_sha256(records),
    }


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON value must be an object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
