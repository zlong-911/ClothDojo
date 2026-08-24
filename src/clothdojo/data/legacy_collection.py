"""One-way import of the validated pre-ClothDojo trajectory collection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clothdojo.data.annotations import build_good_segment_plan
from clothdojo.data.review_source import discover_review_sources
from clothdojo.trajectory import TrajectoryReader
from clothdojo.trajectory.schema import TRAJECTORY_FORMAT_VERSION, TrajectoryManifest

IMPORT_SCHEMA = "clothdojo_legacy_collection_import"
IMPORT_FORMAT_VERSION = 1
LEGACY_MANIFEST_SCHEMA = "newton_cloth_episode"
LEGACY_STATE_PROFILE = "full"


@dataclass(frozen=True, slots=True)
class LegacyCollectionPlan:
    source_root: Path
    complete_trajectories: int
    incomplete_trajectories: int
    source_frames: int
    annotation_files: int
    review_exports: int
    copied_files: int
    copied_bytes: int
    source_snapshot_sha256: str

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_root"] = str(self.source_root)
        return value


Progress = Callable[[int, int, int, int], None]


def inspect_legacy_collection(
    source_root: str | Path, *, include_reviews: bool = True
) -> LegacyCollectionPlan:
    """Validate the old collection without making it a runtime dependency."""

    root = Path(source_root).expanduser().resolve()
    trajectories = root / "trajectories"
    if not trajectories.is_dir():
        raise NotADirectoryError(f"legacy trajectory directory is missing: {trajectories}")

    complete = 0
    incomplete = 0
    source_frames = 0
    manifests: list[tuple[Path, dict[str, Any]]] = []
    files: list[Path] = []
    for path in sorted(trajectories.glob("*/manifest.json")):
        value = _read_object(path)
        _canonical_manifest(value, path)
        status = value.get("status")
        action_count = _nonnegative_integer(value.get("action_count"), "action_count", path)
        if status == "complete":
            complete += 1
            source_frames += action_count
            expected = ("actions.npz", "states.npz")
            missing = [name for name in expected if not (path.parent / name).is_file()]
            if missing:
                raise RuntimeError(f"legacy trajectory files are missing: {path.parent}: {missing}")
            files.extend(path.parent / name for name in expected)
        elif status in {"recording", "aborted"} and action_count == 0:
            incomplete += 1
        else:
            raise RuntimeError(f"unsupported legacy trajectory status: {path}: {status!r}")
        manifests.append((path, value))
    if not manifests or complete == 0:
        raise RuntimeError(f"legacy collection has no complete trajectories: {root}")

    # Import validates annotation coordinates but does not apply the downstream
    # training export's 65-frame selection floor.
    selection = build_good_segment_plan(root, min_segment_frames=1)
    if selection.source_trajectories != complete or selection.source_frames != source_frames:
        raise RuntimeError("legacy annotation plan differs from trajectory inventory")
    annotations = sorted((trajectories / "annotations").glob("*.segments.json"))
    files.extend(annotations)

    review_exports = 0
    if include_reviews:
        reviews = discover_review_sources(root)
        complete_ids = {
            path.parent.name for path, value in manifests if value.get("status") == "complete"
        }
        if set(reviews) != complete_ids:
            raise RuntimeError("legacy review exports do not match complete trajectories")
        review_exports = len(reviews)
        files.extend(
            path
            for source in reviews.values()
            for path in sorted(source.root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        )

    snapshot = hashlib.sha256()
    for path, value in manifests:
        _update_snapshot(snapshot, path.relative_to(root), _stable_json(value))
    for path in annotations:
        _update_snapshot(snapshot, path.relative_to(root), path.read_bytes())
    copied_bytes = sum(path.stat().st_size for path in files)
    return LegacyCollectionPlan(
        source_root=root,
        complete_trajectories=complete,
        incomplete_trajectories=incomplete,
        source_frames=source_frames,
        annotation_files=len(annotations),
        review_exports=review_exports,
        copied_files=len(files),
        copied_bytes=copied_bytes,
        source_snapshot_sha256=snapshot.hexdigest(),
    )


def import_legacy_collection(
    source_root: str | Path,
    output_root: str | Path,
    *,
    include_reviews: bool = True,
    workers: int = 8,
    progress: Progress | None = None,
) -> LegacyCollectionPlan:
    """Copy and normalize a legacy collection through a resumable staging tree."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    plan = inspect_legacy_collection(source_root, include_reviews=include_reviews)
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"collection output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.importing"
    staging.mkdir(exist_ok=True)
    plan_path = staging / ".migration-plan.json"
    expected_plan = {
        "schema": IMPORT_SCHEMA,
        "format_version": IMPORT_FORMAT_VERSION,
        "include_reviews": include_reviews,
        "plan": plan.to_json_dict(),
    }
    if plan_path.is_file():
        if _read_object(plan_path) != expected_plan:
            raise RuntimeError(f"unfinished import has a different source plan: {staging}")
    elif any(staging.iterdir()):
        raise RuntimeError(f"unfinished import has no migration plan: {staging}")
    else:
        _atomic_json(plan_path, expected_plan)

    source = plan.source_root
    copy_pairs: list[tuple[Path, Path]] = []
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted((source / "trajectories").glob("*/manifest.json")):
        value = _read_object(manifest_path)
        manifests.append((manifest_path, value))
        target = staging / "trajectories" / manifest_path.parent.name
        if value.get("status") == "complete":
            copy_pairs.extend(
                (manifest_path.parent / name, target / name)
                for name in ("actions.npz", "states.npz")
            )
    for path in sorted((source / "trajectories/annotations").glob("*.segments.json")):
        copy_pairs.append((path, staging / path.relative_to(source)))
    if include_reviews:
        for path in sorted((source / "reviews").rglob("*")):
            if path.is_file() and not path.is_symlink():
                copy_pairs.append((path, staging / path.relative_to(source)))

    copied_files = 0
    copied_bytes = 0
    progress_lock = threading.Lock()

    def finished(size: int) -> None:
        nonlocal copied_files, copied_bytes
        with progress_lock:
            copied_files += 1
            copied_bytes += size
            if progress is not None:
                progress(copied_files, len(copy_pairs), copied_bytes, plan.copied_bytes)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_copy_file_resumable, source_path, target_path): source_path
            for source_path, target_path in copy_pairs
        }
        for future in as_completed(futures):
            finished(future.result())

    for manifest_path, value in manifests:
        target = staging / "trajectories" / manifest_path.parent.name / "manifest.json"
        _atomic_json(target, _canonical_manifest(value, manifest_path).to_json_dict())

    imported_plan = build_good_segment_plan(staging, min_segment_frames=1)
    if (
        imported_plan.source_trajectories != plan.complete_trajectories
        or imported_plan.source_frames != plan.source_frames
        or imported_plan.annotation_files != plan.annotation_files
    ):
        raise RuntimeError("imported trajectory or annotation inventory differs from source")
    for manifest_path in sorted((staging / "trajectories").glob("*/manifest.json")):
        value = _read_object(manifest_path)
        TrajectoryReader(manifest_path.parent, require_complete=value["status"] == "complete")
    if include_reviews:
        reviews = discover_review_sources(staging)
        if len(reviews) != plan.review_exports:
            raise RuntimeError("imported review inventory differs from source")

    receipt = {
        "schema": IMPORT_SCHEMA,
        "format_version": IMPORT_FORMAT_VERSION,
        "status": "complete",
        "source_is_runtime_dependency": False,
        "include_reviews": include_reviews,
        "plan": plan.to_json_dict(),
    }
    _atomic_json(staging / "migration.json", receipt)
    plan_path.unlink()
    staging.rename(output)
    return plan


def _canonical_manifest(value: dict[str, Any], path: Path) -> TrajectoryManifest:
    if value.get("schema_version") != LEGACY_MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported legacy manifest schema: {path}")
    if value.get("state_profile") != LEGACY_STATE_PROFILE:
        raise RuntimeError(f"unsupported legacy state profile: {path}")
    canonical = dict(value)
    canonical.pop("schema_version", None)
    canonical.pop("state_profile", None)
    canonical["format_version"] = TRAJECTORY_FORMAT_VERSION
    manifest = TrajectoryManifest.from_json_dict(canonical)
    if manifest.to_json_dict() != canonical:
        raise RuntimeError(f"legacy manifest contains fields outside the canonical schema: {path}")
    return manifest


def _copy_file_resumable(source: Path, target: Path) -> int:
    size = source.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if target.stat().st_size != size:
            raise RuntimeError(f"resumed import file has the wrong size: {target}")
        return size
    temporary = target.parent / f".{target.name}.copying"
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != size:
            raise RuntimeError(f"copied file has the wrong size: {temporary}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return size


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON value must be an object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _nonnegative_integer(value: object, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"legacy {name} is invalid: {path}: {value!r}")
    return value


def _stable_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _update_snapshot(digest, relative: Path, content: bytes) -> None:
    digest.update(str(relative).encode())
    digest.update(b"\0")
    digest.update(content)
    digest.update(b"\0")
