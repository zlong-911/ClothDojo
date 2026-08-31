"""Dataset and normalization artifact checks for OpenPI training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_TASK = "flatten the shirt"
MIN_SEGMENT_FRAMES = 65
TRAINING_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def inspect_openpi_dataset(
    value: Any, *, expected_task: str = EXPECTED_TASK
) -> dict[str, Any]:
    """Inspect the canonical LeRobot artifact used by OpenPI training."""

    root = Path(str(value)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OpenPI dataset does not exist: {root}")
    info_path = root / "meta" / "info.json"
    sidecar_path = root / "meta" / "clothdojo.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    info = _read_object(info_path, "LeRobot info")
    sidecar = _read_object(sidecar_path, "ClothDojo dataset sidecar")

    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"OpenPI requires pinned LeRobot v2.1: {info_path}")
    total_episodes = _positive_int(info.get("total_episodes"), "total_episodes")
    total_frames = _positive_int(info.get("total_frames"), "total_frames")
    if info.get("total_tasks") != 1:
        raise ValueError(f"OpenPI dataset must declare exactly one task: {info_path}")
    _validate_features(info.get("features"), root)

    if sidecar.get("schema") != "clothdojo_lerobot_export":
        raise ValueError(f"unsupported ClothDojo dataset sidecar: {sidecar_path}")
    if sidecar.get("format_version") != 1 or sidecar.get("export_status") != "complete":
        raise ValueError(f"ClothDojo dataset export is incomplete: {sidecar_path}")
    if sidecar.get("training_eligible", True) is not True:
        raise ValueError(f"ClothDojo dataset is ineligible for training: {sidecar_path}")
    selection = sidecar.get("selection")
    if not isinstance(selection, dict):
        raise TypeError(f"dataset selection provenance must be an object: {sidecar_path}")
    minimum = _positive_int(selection.get("min_segment_frames"), "min_segment_frames")
    if minimum < MIN_SEGMENT_FRAMES:
        raise ValueError(
            f"dataset min_segment_frames must be >= {MIN_SEGMENT_FRAMES}, got {minimum}"
        )

    tasks = _read_jsonl(tasks_path, "LeRobot tasks")
    task = tasks[0].get("task") if len(tasks) == 1 else None
    if (
        not isinstance(task, str)
        or task.strip().casefold().rstrip(".")
        != expected_task.strip().casefold().rstrip(".")
        or tasks[0].get("task_index") != 0
    ):
        raise ValueError(
            f"OpenPI dataset task must be exactly {expected_task!r}: {tasks_path}"
        )
    episodes = _read_jsonl(episodes_path, "LeRobot episodes")
    if len(episodes) != total_episodes:
        raise ValueError("LeRobot episode count differs from info.json")
    episodes.sort(key=lambda row: row.get("episode_index", -1))
    if [row.get("episode_index") for row in episodes] != list(range(total_episodes)):
        raise ValueError("LeRobot episode indices must be contiguous and zero-based")
    lengths = [_positive_int(row.get("length"), "episode length") for row in episodes]
    if sum(lengths) != total_frames:
        raise ValueError("LeRobot episode lengths do not sum to total_frames")
    if min(lengths) < MIN_SEGMENT_FRAMES:
        raise ValueError(
            f"dataset contains an episode shorter than {MIN_SEGMENT_FRAMES} frames"
        )

    sidecar_episodes = sidecar.get("episodes")
    counts = sidecar.get("counts")
    if not isinstance(sidecar_episodes, list) or len(sidecar_episodes) != total_episodes:
        raise ValueError("dataset sidecar episode list is incomplete")
    if not isinstance(counts, dict):
        raise TypeError("dataset sidecar counts must be an object")
    if (
        counts.get("output_episodes") != total_episodes
        or counts.get("output_frames") != total_frames
    ):
        raise ValueError("dataset sidecar counts differ from info.json")
    if counts.get("min_segment_frames") != minimum:
        raise ValueError("dataset sidecar minimum-length provenance is inconsistent")
    for index, (record, length) in enumerate(zip(sidecar_episodes, lengths, strict=True)):
        if record.get("episode_index") != index or record.get("length") != length:
            raise ValueError("dataset sidecar episode extents differ from LeRobot metadata")

    return {
        "root": str(root),
        "info_path": str(info_path),
        "info_sha256": _sha256(info_path),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "tasks_path": str(tasks_path),
        "tasks_sha256": _sha256(tasks_path),
        "episodes_path": str(episodes_path),
        "episodes_sha256": _sha256(episodes_path),
        "episodes": total_episodes,
        "frames": total_frames,
        "fps": _positive_int(info.get("fps"), "fps"),
        "min_segment_frames": minimum,
        "minimum_episode_frames": min(lengths),
        "task": expected_task,
        "prompt": task,
        "training_cameras": list(TRAINING_CAMERAS),
        "state_dim": 14,
        "action_dim": 14,
    }


def inspect_openpi_norm_stats(
    value: Any,
    *,
    dataset: dict[str, Any],
    config_name: str,
    data_kwargs: dict[str, Any],
    action_horizon: int,
) -> dict[str, Any]:
    """Require normalization statistics to match the exact reviewed dataset."""

    root = Path(str(value)).expanduser().resolve()
    stats_path = root / "norm_stats.json"
    provenance_path = root / "clothdojo.json"
    stats = _read_object(stats_path, "OpenPI normalization statistics")
    provenance = _read_object(provenance_path, "OpenPI norm-stat provenance")
    if provenance.get("schema") != "clothdojo_openpi_norm_stats":
        raise ValueError(f"unsupported norm-stat provenance: {provenance_path}")
    if provenance.get("format_version") != 1:
        raise ValueError(f"unsupported norm-stat format version: {provenance_path}")
    expected = {
        "dataset_info_sha256": dataset["info_sha256"],
        "dataset_sidecar_sha256": dataset["sidecar_sha256"],
        "config_name": config_name,
        "data_kwargs": data_kwargs,
        "action_horizon": action_horizon,
        "frames": dataset["frames"],
        "episodes": dataset["episodes"],
    }
    for key, expected_value in expected.items():
        actual_value = provenance.get(key)
        if key == "data_kwargs" and isinstance(actual_value, dict):
            # Text never enters state/action normalization. Keep every other
            # transform setting exact, and retain the original provenance below.
            actual_value = {k: v for k, v in actual_value.items() if k != "default_prompt"}
            expected_value = {k: v for k, v in expected_value.items() if k != "default_prompt"}
        if actual_value != expected_value:
            raise ValueError(
                f"norm-stat provenance mismatch for {key}: "
                f"{provenance.get(key)!r} != {expected_value!r}"
            )
    values = stats.get("norm_stats")
    if not isinstance(values, dict) or set(values) != {"state", "actions"}:
        raise ValueError(f"norm_stats.json must contain state and actions: {stats_path}")
    for name in ("state", "actions"):
        record = values[name]
        if not isinstance(record, dict):
            raise TypeError(f"norm stats {name} must be an object")
        for statistic in ("mean", "std", "q01", "q99"):
            vector = record.get(statistic)
            if not isinstance(vector, list) or len(vector) != 14:
                raise ValueError(f"norm stats {name}.{statistic} must be 14D")
    return {
        "root": str(root),
        "stats_path": str(stats_path),
        "stats_sha256": _sha256(stats_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        **expected,
        "data_kwargs": provenance["data_kwargs"],
    }


def _validate_features(value: Any, root: Path) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"LeRobot features must be an object: {root}")
    for key in ("observation.state", "action"):
        feature = value.get(key)
        if (
            not isinstance(feature, dict)
            or feature.get("dtype") != "float32"
            or feature.get("shape") != [14]
        ):
            raise ValueError(f"OpenPI feature {key} must be float32[14]: {root}")
    for key in TRAINING_CAMERAS:
        feature = value.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"OpenPI training camera must be a video feature: {key}")
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"OpenPI training camera must have HWC RGB shape: {key}")


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object: {path}")
    return value


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {context}: {path}: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{context} must contain JSON objects: {path}")
    return rows


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
