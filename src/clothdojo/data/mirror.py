"""Deterministic bilateral mirroring for canonical dual-Piper trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import numpy as np

from clothdojo.trajectory import TrajectoryManifest, TrajectoryReader

MIRROR_COLLECTION_SCHEMA = "clothdojo_bilateral_mirror_collection"
MIRROR_TRAJECTORY_SCHEMA = "clothdojo_bilateral_mirror"
MIRROR_FORMAT_VERSION = 1
MIRROR_SUFFIX = "__mirror"

PIPER_ARM_SIGNS = np.asarray((-1.0, 1.0, 1.0, -1.0, 1.0, -1.0), dtype=np.float32)
_POLAR_SIGNS = np.asarray((1.0, -1.0, 1.0), dtype=np.float32)
_AXIAL_SIGNS = np.asarray((-1.0, 1.0, -1.0), dtype=np.float32)
_QUATERNION_SIGNS = np.asarray((-1.0, 1.0, -1.0, 1.0), dtype=np.float32)
_BODY_VELOCITY_SIGNS = np.concatenate((_POLAR_SIGNS, _AXIAL_SIGNS))
_PIPER_BODY_COUNT = 21
_PIPER_SIDE_BODY_COUNT = 10
_PIPER_HOME_SIDE_QUATERNIONS = np.asarray(
    (
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.047970932, 0.7054777, 0.7054777, -0.04797097),
        (-0.5213338, 0.4777144, 0.47771442, 0.5213338),
        (-4.1748827e-9, 0.67559016, 9.466274e-9, 0.7372774),
        (-0.5213338, 0.4777144, 0.47771442, 0.5213338),
        (0.4777144, 0.47771442, 0.5213338, 0.5213338),
        (0.7781813, 0.00022213276, 0.62803954, 0.0003304479),
        (0.4777126, 0.521334, 0.5213338, -0.4777164),
        (-0.47771454, 0.5213357, -0.521332, -0.47771436),
    ),
    dtype=np.float32,
)
_PIPER_HOME_BODY_QUATERNIONS = np.concatenate(
    (
        np.asarray(((0.0, 0.0, 0.0, 1.0),), dtype=np.float32),
        _PIPER_HOME_SIDE_QUATERNIONS,
        _PIPER_HOME_SIDE_QUATERNIONS,
    )
)


def augment_mirror(source: str | Path, output: str | Path) -> dict:
    """Publish one independent original-plus-mirror canonical collection."""

    source_root = Path(source).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"mirror output already exists: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("mirror output must be outside the source collection")

    source_trajectories, skipped_incomplete = _complete_trajectories(source_root)
    source_ids = {path.name for path in source_trajectories}
    collisions = sorted(
        source_id
        for source_id in source_ids
        if source_id.endswith(MIRROR_SUFFIX) or f"{source_id}{MIRROR_SUFFIX}" in source_ids
    )
    if collisions:
        raise ValueError(f"source collection already contains mirror trajectory ids: {collisions}")

    staging = output_root.with_name(f".{output_root.name}.tmp-{uuid4().hex}")
    trajectories_root = staging / "trajectories"
    annotations_root = trajectories_root / "annotations"
    pairs = []
    annotation_files = 0
    try:
        annotations_root.mkdir(parents=True)
        def augment_one(source_trajectory: Path) -> tuple[dict, int, bool]:
            source_id = source_trajectory.name
            mirror_id = f"{source_id}{MIRROR_SUFFIX}"
            source_manifest = source_trajectory / "manifest.json"
            legacy_manifest_sha256 = _sha256(source_manifest)

            copied = trajectories_root / source_id
            shutil.copytree(source_trajectory, copied, copy_function=shutil.copy2)
            normalized_legacy = _normalize_copied_legacy_manifest(
                source_manifest,
                copied / "manifest.json",
            )
            source_manifest_sha256 = _sha256(copied / "manifest.json")
            if not normalized_legacy and source_manifest_sha256 != legacy_manifest_sha256:
                raise RuntimeError(f"copied source manifest differs: {source_id}")

            mirror = trajectories_root / mirror_id
            mirror_manifest = _mirror_trajectory(
                copied,
                mirror,
                source_id=source_id,
                source_manifest_sha256=source_manifest_sha256,
                legacy_source_manifest_sha256=(
                    legacy_manifest_sha256 if normalized_legacy else None
                ),
            )
            annotation = (
                source_root / "trajectories" / "annotations" / f"{source_id}.segments.json"
            )
            copied_annotations = 0
            if annotation.is_file():
                content = annotation.read_bytes()
                (annotations_root / annotation.name).write_bytes(content)
                (annotations_root / f"{mirror_id}.segments.json").write_bytes(content)
                copied_annotations = 2

            return (
                {
                    "source_id": source_id,
                    "mirror_id": mirror_id,
                    "source_manifest_sha256": source_manifest_sha256,
                    "mirror_manifest_sha256": _sha256(mirror / "manifest.json"),
                    "action_count": mirror_manifest.action_count,
                    **(
                        {"legacy_source_manifest_sha256": legacy_manifest_sha256}
                        if normalized_legacy
                        else {}
                    ),
                },
                copied_annotations,
                normalized_legacy,
            )

        normalized_legacy_manifests = 0
        with ThreadPoolExecutor(max_workers=min(8, len(source_trajectories))) as pool:
            for pair, copied_annotations, normalized_legacy in pool.map(
                augment_one, source_trajectories
            ):
                pairs.append(pair)
                annotation_files += copied_annotations
                normalized_legacy_manifests += int(normalized_legacy)

        collection = {
            "schema": MIRROR_COLLECTION_SCHEMA,
            "format_version": MIRROR_FORMAT_VERSION,
            "status": "complete",
            "source": str(source_root),
            "source_is_runtime_dependency": False,
            "source_snapshot_sha256": _source_snapshot_sha256(pairs),
            "mirror_plane": {"axis": "y", "offset_m": 0.0},
            "arm_joint_signs": PIPER_ARM_SIGNS.astype(int).tolist(),
            "body_orientation_frame": "piper_home_body",
            "surface_winding_reversed": True,
            "input_trajectories": len(pairs),
            "output_trajectories": 2 * len(pairs),
            "skipped_incomplete_trajectories": skipped_incomplete,
            "annotation_files": annotation_files,
            "normalized_legacy_manifests": normalized_legacy_manifests,
            "trajectory_pairs": pairs,
        }
        _write_json(staging / "augmentation.json", collection)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output": str(output_root),
        "input_trajectories": len(pairs),
        "output_trajectories": 2 * len(pairs),
        "skipped_incomplete_trajectories": skipped_incomplete,
        "annotation_files": annotation_files,
        "normalized_legacy_manifests": normalized_legacy_manifests,
    }


def mirror_state_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mirror one complete canonical Piper state archive."""

    _require_fields(
        arrays,
        {
            "sample_index",
            "joint_q",
            "joint_qd",
            "body_q",
            "body_qd",
            "particle_q",
            "particle_qd",
            "gripper_open_state",
            "grasp_attached",
            "grasp_previous_open",
            "grasp_pin_count",
            "grasp_pins",
            "grasp_pin_local",
        },
        "state",
    )
    _require_shape(arrays["joint_q"], (16,), "joint_q")
    _require_shape(arrays["joint_qd"], (16,), "joint_qd")
    _require_shape(arrays["body_q"], (_PIPER_BODY_COUNT, 7), "body_q")
    _require_shape(arrays["body_qd"], (_PIPER_BODY_COUNT, 6), "body_qd")
    _require_shape(arrays["particle_q"], (None, 3), "particle_q")
    _require_shape(arrays["particle_qd"], (None, 3), "particle_qd")
    for name in ("gripper_open_state", "grasp_attached", "grasp_previous_open", "grasp_pin_count"):
        _require_shape(arrays[name], (2,), name)
    _require_shape(arrays["grasp_pins"], (2, None), "grasp_pins")
    _require_shape(arrays["grasp_pin_local"], (2, None, 3), "grasp_pin_local")

    result = mirror_robot_state_arrays(
        {
            name: values
            for name, values in arrays.items()
            if name not in {"particle_q", "particle_qd"}
        }
    )
    particle_q = np.asarray(arrays["particle_q"]).copy()
    particle_q[..., 1] *= -1.0
    particle_qd = np.asarray(arrays["particle_qd"]).copy()
    particle_qd[..., 1] *= -1.0
    result.update({"particle_q": particle_q, "particle_qd": particle_qd})
    _require_finite(result, "mirrored state")
    return {name: result[name] for name in arrays}


def mirror_robot_state_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mirror Piper state fields while leaving derived cloth fields to an overlay."""

    expected = {
        "sample_index",
        "joint_q",
        "joint_qd",
        "body_q",
        "body_qd",
        "gripper_open_state",
        "grasp_attached",
        "grasp_previous_open",
        "grasp_pin_count",
        "grasp_pins",
        "grasp_pin_local",
    }
    _require_fields(arrays, expected, "robot state")
    _require_shape(arrays["joint_q"], (16,), "joint_q")
    _require_shape(arrays["joint_qd"], (16,), "joint_qd")
    _require_shape(arrays["body_q"], (_PIPER_BODY_COUNT, 7), "body_q")
    _require_shape(arrays["body_qd"], (_PIPER_BODY_COUNT, 6), "body_qd")
    for name in ("gripper_open_state", "grasp_attached", "grasp_previous_open", "grasp_pin_count"):
        _require_shape(arrays[name], (2,), name)
    _require_shape(arrays["grasp_pins"], (2, None), "grasp_pins")
    _require_shape(arrays["grasp_pin_local"], (2, None, 3), "grasp_pin_local")

    body_permutation = np.concatenate(
        (
            np.asarray((0,), dtype=np.int64),
            np.arange(1 + _PIPER_SIDE_BODY_COUNT, _PIPER_BODY_COUNT),
            np.arange(1, 1 + _PIPER_SIDE_BODY_COUNT),
        )
    )
    body_q = np.asarray(arrays["body_q"][:, body_permutation]).copy()
    body_q[..., :3] *= _POLAR_SIGNS
    source_home = _PIPER_HOME_BODY_QUATERNIONS[body_permutation]
    relative_orientation = _quaternion_multiply_xyzw(
        body_q[..., 3:7],
        _quaternion_conjugate_xyzw(source_home),
    )
    relative_orientation *= _QUATERNION_SIGNS
    body_q[..., 3:7] = _quaternion_multiply_xyzw(
        relative_orientation,
        _PIPER_HOME_BODY_QUATERNIONS,
    )
    body_qd = np.asarray(arrays["body_qd"][:, body_permutation]).copy()
    body_qd *= _BODY_VELOCITY_SIGNS

    pin_local = np.asarray(arrays["grasp_pin_local"][:, ::-1]).copy()
    # link6's home frame maps its local x axis to world y. Its local reflection
    # is therefore H^-1 S H = diag(-1, 1, 1), consistent with body_q above.
    # The grasp-frame offset lies on local z and is unchanged by this reflection.
    pin_local[..., 0] *= -1.0

    result = {
        "sample_index": np.asarray(arrays["sample_index"]).copy(),
        "joint_q": _mirror_piper_joint_coordinates(arrays["joint_q"]),
        "joint_qd": _mirror_piper_joint_coordinates(arrays["joint_qd"]),
        "body_q": body_q,
        "body_qd": body_qd,
        "gripper_open_state": np.asarray(arrays["gripper_open_state"][:, ::-1]).copy(),
        "grasp_attached": np.asarray(arrays["grasp_attached"][:, ::-1]).copy(),
        "grasp_previous_open": np.asarray(
            arrays["grasp_previous_open"][:, ::-1]
        ).copy(),
        "grasp_pin_count": np.asarray(arrays["grasp_pin_count"][:, ::-1]).copy(),
        "grasp_pins": np.asarray(arrays["grasp_pins"][:, ::-1]).copy(),
        "grasp_pin_local": pin_local,
    }
    _require_finite(result, "mirrored robot state")
    return result


def mirror_action_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mirror one complete canonical Piper action archive."""

    _require_fields(
        arrays,
        {"sample_index", "joint_target", "gripper_open_command"},
        "action",
    )
    _require_shape(arrays["joint_target"], (12,), "joint_target")
    _require_shape(arrays["gripper_open_command"], (2,), "gripper_open_command")
    joint_target = np.asarray(arrays["joint_target"]).reshape(-1, 2, 6)[:, ::-1].copy()
    joint_target *= PIPER_ARM_SIGNS
    result = {
        "sample_index": np.asarray(arrays["sample_index"]).copy(),
        "joint_target": joint_target.reshape(-1, 12),
        "gripper_open_command": np.asarray(
            arrays["gripper_open_command"][:, ::-1]
        ).copy(),
    }
    _require_finite(result, "mirrored action")
    return result


def _mirror_piper_joint_coordinates(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    result = np.empty_like(source)
    result[:, 0:6] = source[:, 8:14] * PIPER_ARM_SIGNS
    result[:, 6:8] = source[:, 14:16]
    result[:, 8:14] = source[:, 0:6] * PIPER_ARM_SIGNS
    result[:, 14:16] = source[:, 6:8]
    return result


def _quaternion_conjugate_xyzw(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result[..., :3] *= -1.0
    return result


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.broadcast_arrays(np.asarray(left), np.asarray(right))
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _mirror_trajectory(
    source: Path,
    output: Path,
    *,
    source_id: str,
    source_manifest_sha256: str,
    legacy_source_manifest_sha256: str | None = None,
) -> TrajectoryManifest:
    reader = TrajectoryReader(source)
    if reader.manifest.scene.robot_id != "piper":
        raise ValueError(
            f"mirror augmentation requires Piper trajectories: {source_id} has "
            f"robot={reader.manifest.scene.robot_id!r}"
        )
    if reader.manifest.action_count < 1:
        raise ValueError(f"mirror augmentation requires a non-empty trajectory: {source_id}")
    source_states = reader.state_arrays()
    source_actions = reader.action_arrays()
    states = mirror_state_arrays(source_states)
    actions = mirror_action_arrays(source_actions)
    _validate_involution(source_states, source_actions, states, actions)

    metadata = copy.deepcopy(reader.manifest.metadata)
    if "augmentation" in metadata:
        raise ValueError(f"source trajectory already owns augmentation metadata: {source_id}")
    metadata["augmentation"] = {
        "schema": MIRROR_TRAJECTORY_SCHEMA,
        "format_version": MIRROR_FORMAT_VERSION,
        "kind": "bilateral_mirror",
        "source_trajectory_id": source_id,
        "source_manifest_sha256": source_manifest_sha256,
        **(
            {"legacy_source_manifest_sha256": legacy_source_manifest_sha256}
            if legacy_source_manifest_sha256 is not None
            else {}
        ),
        "mirror_plane": {"axis": "y", "offset_m": 0.0},
        "arm_joint_signs": PIPER_ARM_SIGNS.astype(int).tolist(),
        "left_right_swapped": True,
        "body_orientation_frame": "piper_home_body",
        "surface_winding_reversed": True,
        "kinematics_used": False,
        "physics_used": False,
    }
    manifest = copy.deepcopy(reader.manifest)
    manifest.metadata = metadata
    output.mkdir(parents=True)
    _write_npz(output / "states.npz", states)
    if manifest.action_count:
        _write_npz(output / "actions.npz", actions)
    _write_json(output / "manifest.json", manifest.to_json_dict())
    del states, actions, source_states, source_actions, reader
    TrajectoryReader(output).validate_training_data()
    return manifest


def _normalize_copied_legacy_manifest(source: Path, copied: Path) -> bool:
    """Normalize the one legacy EpisodeV1 manifest shape in the output copy."""

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read trajectory manifest: {source}") from exc
    if payload.get("format_version") == MIRROR_FORMAT_VERSION:
        return False
    if (
        payload.get("schema_version") != "newton_cloth_episode"
        or payload.get("state_profile") != "full"
        or "format_version" in payload
    ):
        raise ValueError(f"unsupported legacy trajectory manifest: {source}")
    payload.pop("schema_version")
    payload.pop("state_profile")
    payload["format_version"] = MIRROR_FORMAT_VERSION
    manifest = TrajectoryManifest.from_json_dict(payload)
    _write_json(copied, manifest.to_json_dict())
    return True


def _validate_involution(
    source_states: dict[str, np.ndarray],
    source_actions: dict[str, np.ndarray],
    mirrored_states: dict[str, np.ndarray],
    mirrored_actions: dict[str, np.ndarray],
) -> None:
    restored_states = mirror_state_arrays(mirrored_states)
    restored_actions = mirror_action_arrays(mirrored_actions)
    for name, values in source_states.items():
        if not np.allclose(restored_states[name], values, rtol=1.0e-6, atol=1.0e-6):
            raise RuntimeError(f"state mirror transform is not involutive: {name}")
    for name, values in source_actions.items():
        if not np.array_equal(restored_actions[name], values):
            raise RuntimeError(f"action mirror transform is not involutive: {name}")


def _complete_trajectories(source_root: Path) -> tuple[list[Path], int]:
    trajectory_root = source_root / "trajectories"
    if not trajectory_root.is_dir():
        raise RuntimeError("mirror source must be a canonical collection with trajectories/")
    complete = []
    skipped = 0
    for manifest_path in sorted(trajectory_root.glob("*/manifest.json")):
        if manifest_path.parent.name == "annotations":
            continue
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read trajectory manifest: {manifest_path}") from exc
        if not isinstance(value, dict) or value.get("status") != "complete":
            skipped += 1
            continue
        complete.append(manifest_path.parent)
    if not complete:
        raise RuntimeError(f"source collection has no complete trajectories: {source_root}")
    return complete, skipped


def _require_fields(arrays: dict[str, np.ndarray], expected: set[str], kind: str) -> None:
    if set(arrays) != expected:
        raise ValueError(
            f"{kind} archive fields differ: missing={sorted(expected - set(arrays))}, "
            f"unexpected={sorted(set(arrays) - expected)}"
        )


def _require_shape(array: np.ndarray, trailing: tuple[int | None, ...], name: str) -> None:
    actual = tuple(np.asarray(array).shape[1:])
    if len(actual) != len(trailing) or any(
        expected is not None and value != expected
        for value, expected in zip(actual, trailing, strict=True)
    ):
        raise ValueError(f"Piper trajectory {name} shape {actual} != {trailing}")


def _require_finite(arrays: dict[str, np.ndarray], kind: str) -> None:
    for name, values in arrays.items():
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            raise ValueError(f"{kind} contains non-finite values: {name}")


def _source_snapshot_sha256(pairs: list[dict]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(pair["source_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(pair["source_manifest_sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as file:
        np.savez_compressed(file, **arrays)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
