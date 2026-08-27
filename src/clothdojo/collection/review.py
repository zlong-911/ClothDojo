"""Trajectory review catalog and concurrency-safe annotation writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from clothdojo.collection.config import CollectionProfile
from clothdojo.data.annotations import ANNOTATION_SCHEMA, KNOWN_LABELS
from clothdojo.data.rollout_review import REVIEW_CAMERA
from clothdojo.trajectory.schema import TrajectoryManifest

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
ReviewStatus = Literal["in_progress", "complete"]


class AnnotationConflictError(RuntimeError):
    """Raised when a reviewer saves over a newer annotation revision."""


class ReviewCatalog:
    def __init__(self, profile: CollectionProfile):
        self.profile = profile
        self._lock = threading.RLock()
        self._episode_cache: tuple[tuple[tuple[int, int], ...], list[dict[str, Any]]] | None = None

    def episodes(self, *, page: int = 1, page_size: int = 50) -> dict:
        if page < 1:
            raise ValueError("page must be positive")
        if not 1 <= page_size <= 200:
            raise ValueError("page_size must be between 1 and 200")
        items = self._episode_items()
        total = len(items)
        page_count = max(1, (total + page_size - 1) // page_size)
        page = min(page, page_count)
        start = (page - 1) * page_size
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "page_count": page_count,
            "episodes": items[start : start + page_size],
        }

    def annotation(
        self,
        episode_id: str,
        *,
        manifest: TrajectoryManifest | None = None,
    ) -> dict[str, Any]:
        episode_id = _identity(episode_id, "episode id")
        manifest = manifest or self._complete_manifest(episode_id)
        path = self._annotation_path(episode_id)
        if not path.is_file():
            return {
                "episode_id": episode_id,
                "action_count": manifest.action_count,
                "review_status": "unreviewed",
                "segments": [],
                "updated_at": None,
                "revision": None,
            }
        content = path.read_bytes()
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid annotation JSON: {path}") from exc
        status, segments, updated_at = _validate_annotation(value, manifest.action_count)
        return {
            "episode_id": episode_id,
            "action_count": manifest.action_count,
            "review_status": status,
            "segments": segments,
            "updated_at": updated_at,
            "revision": hashlib.sha256(content).hexdigest(),
        }

    def save_annotation(
        self,
        episode_id: str,
        *,
        review_status: ReviewStatus,
        segments: object,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        episode_id = _identity(episode_id, "episode id")
        manifest = self._complete_manifest(episode_id)
        normalized = _segments(segments, manifest.action_count)
        if review_status not in {"in_progress", "complete"}:
            raise ValueError("review_status must be in_progress or complete")
        path = self._annotation_path(episode_id)
        with self._lock:
            current_revision = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            )
            if current_revision != expected_revision:
                raise AnnotationConflictError(
                    "annotation changed after it was loaded; reload before saving"
                )
            payload = {
                "schema": ANNOTATION_SCHEMA,
                "review_status": review_status,
                "segments": normalized,
                "updated_at": _utc_now(),
            }
            _atomic_json(path, payload)
            self._episode_cache = None
        return self.annotation(episode_id, manifest=manifest)

    def video(self, episode_id: str, camera: str) -> Path:
        episode_id = _identity(episode_id, "episode id")
        camera = _identity(camera, "camera")
        path = self.profile.review_root / "videos" / camera / f"{episode_id}.mp4"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"review camera is unavailable: {episode_id}/{camera}")
        return path

    def _episode_items(self) -> list[dict[str, Any]]:
        root = self.profile.trajectory_root
        if not root.is_dir():
            return []
        stamp = (
            _file_set_stamp(root, "*/manifest.json"),
            _file_set_stamp(root / "annotations", "*.segments.json"),
            _file_set_stamp(self.profile.review_root, "videos/*/*.mp4"),
        )
        with self._lock:
            if self._episode_cache is not None and self._episode_cache[0] == stamp:
                return self._episode_cache[1]
            items = []
            for manifest_path in root.glob("*/manifest.json"):
                episode_id = manifest_path.parent.name
                if not _ID.fullmatch(episode_id) or manifest_path.parent.is_symlink():
                    continue
                try:
                    manifest = _manifest(manifest_path)
                except (OSError, RuntimeError, TypeError, ValueError):
                    continue
                annotation = self.annotation(episode_id, manifest=manifest)
                video = self.profile.review_root / "videos" / REVIEW_CAMERA / f"{episode_id}.mp4"
                review_ready = video.is_file() and not video.is_symlink()
                items.append(
                    {
                        "id": episode_id,
                        "status": manifest.status,
                        "action_count": manifest.action_count,
                        "control_fps": manifest.control_fps,
                        "asset_id": manifest.metadata.get("asset_id"),
                        "metadata": dict(manifest.metadata),
                        "review_ready": review_ready,
                        "cameras": [REVIEW_CAMERA] if review_ready else [],
                        "review_status": annotation["review_status"],
                        "annotation": annotation,
                        "bad_segment_count": sum(
                            segment["label"] == "bad" for segment in annotation["segments"]
                        ),
                    }
                )
            items.sort(key=lambda item: item["id"], reverse=True)
            self._episode_cache = (stamp, items)
            return items

    def _complete_manifest(self, episode_id: str) -> TrajectoryManifest:
        path = self.profile.trajectory_root / episode_id
        if path.is_symlink() or path.parent != self.profile.trajectory_root:
            raise ValueError("trajectory path is outside the profile")
        try:
            manifest = _manifest(path / "manifest.json")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"unknown trajectory: {episode_id}") from exc
        if manifest.status != "complete":
            raise ValueError("only complete trajectories can be reviewed")
        return manifest

    def _annotation_path(self, episode_id: str) -> Path:
        root = self.profile.trajectory_root
        annotations = root / "annotations"
        if annotations.is_symlink() or annotations.parent != root:
            raise ValueError("annotation directory is outside the trajectory root")
        return annotations / f"{episode_id}.segments.json"


def _manifest(path: Path) -> TrajectoryManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("trajectory manifest must be an object")
    return TrajectoryManifest.from_json_dict(value)


def _validate_annotation(value: object, action_count: int) -> tuple[str, list[dict], str]:
    if not isinstance(value, dict) or value.get("schema") != ANNOTATION_SCHEMA:
        raise RuntimeError(f"annotation schema must be {ANNOTATION_SCHEMA}")
    status = value.get("review_status")
    if status not in {"in_progress", "complete"}:
        raise RuntimeError("annotation review_status is invalid")
    updated_at = value.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        raise RuntimeError("annotation updated_at is invalid")
    try:
        segments = _segments(value.get("segments"), action_count)
    except ValueError as exc:
        raise RuntimeError(f"annotation segments are invalid: {exc}") from exc
    return status, segments, updated_at


def _segments(value: object, action_count: int) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("segments must be a list")
    parsed = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"start_frame", "end_frame", "label"}:
            raise ValueError("each segment must contain start_frame, end_frame and label")
        start = raw["start_frame"]
        end = raw["end_frame"]
        label = raw["label"]
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("segment start_frame must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError("segment end_frame must be an integer")
        if label not in KNOWN_LABELS:
            raise ValueError(f"segment label must be one of {sorted(KNOWN_LABELS)}")
        if start < 0 or end < start or end >= action_count:
            raise ValueError(f"segment [{start}, {end}] is outside [0, {action_count})")
        parsed.append({"start_frame": start, "end_frame": end, "label": label})
    parsed.sort(key=lambda item: (item["start_frame"], item["end_frame"], item["label"]))
    merged: list[dict] = []
    for item in parsed:
        if (
            merged
            and merged[-1]["label"] == item["label"]
            and item["start_frame"] <= merged[-1]["end_frame"] + 1
        ):
            merged[-1]["end_frame"] = max(merged[-1]["end_frame"], item["end_frame"])
        else:
            merged.append(dict(item))
    return merged


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _identity(value: object, name: str) -> str:
    text = str(value)
    if not _ID.fullmatch(text):
        raise ValueError(f"invalid {name}: {value!r}")
    return text


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _file_set_stamp(path: Path, pattern: str) -> tuple[tuple[str, int, int], ...]:
    if not path.is_dir():
        return ()
    result = []
    for item in path.glob(pattern):
        try:
            stat = item.stat()
        except FileNotFoundError:
            continue
        result.append((str(item.relative_to(path)), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(result))
