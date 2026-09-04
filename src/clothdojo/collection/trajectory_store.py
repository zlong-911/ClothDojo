"""Shared filesystem operations for operator-deleted trajectories."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def trajectory_path(dataset_root: Path, trajectory_id: str) -> Path:
    """Resolve one flat trajectory without following an operator-controlled link."""

    if not trajectory_id or any(character not in _ID_CHARS for character in trajectory_id):
        raise ValueError(f"invalid trajectory id: {trajectory_id!r}")
    root = (dataset_root / "trajectories").resolve()
    candidate = root / trajectory_id
    if (
        candidate.is_symlink()
        or candidate.resolve().parent != root
        or not (candidate / "manifest.json").is_file()
    ):
        raise FileNotFoundError(f"unknown trajectory: {trajectory_id}")
    return candidate


def remove_trajectory(
    dataset_root: Path,
    trajectory_id: str,
    *,
    disposition: Literal["delete", "reject"],
) -> dict[str, object]:
    """Remove one trajectory and every mutable review artifact keyed by its id."""

    source = trajectory_path(dataset_root, trajectory_id)
    archived_path: str | None = None
    if disposition == "reject":
        rejected_root = (dataset_root / "rejected").resolve()
        rejected_root.mkdir(parents=True, exist_ok=True)
        destination = rejected_root / trajectory_id
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"rejected trajectory already exists: {destination}")
        rejection = source / "rejection.json"
        _atomic_json(
            rejection,
            {
                "schema": "clothdojo_rejected_teleop_attempt",
                "format_version": 1,
                "reason": "operator_deleted",
                "rejected_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "original_path": str(source),
            },
        )
        try:
            os.replace(source, destination)
        except BaseException:
            rejection.unlink(missing_ok=True)
            raise
        archived_path = str(destination)
    elif disposition == "delete":
        shutil.rmtree(source)
    else:
        raise ValueError(f"unknown trajectory disposition: {disposition!r}")

    removed_videos = 0
    videos = dataset_root / "review" / "videos"
    if videos.is_dir() and not videos.is_symlink():
        for video in videos.glob(f"*/{trajectory_id}.mp4"):
            if not video.is_file() or video.is_symlink():
                continue
            try:
                video.unlink()
            except FileNotFoundError:
                continue
            removed_videos += 1

    annotation = dataset_root / "trajectories" / "annotations" / (
        f"{trajectory_id}.segments.json"
    )
    removed_annotation = annotation.is_file() and not annotation.is_symlink()
    if removed_annotation:
        annotation.unlink()

    return {
        "trajectory_id": trajectory_id,
        "archived_path": archived_path,
        "removed_review_videos": removed_videos,
        "removed_annotation": removed_annotation,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
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
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
