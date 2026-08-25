"""Publish an immutable LeRobot v3 dataset as a pinned-v2.1 indexed view.

This migration is intentionally narrow.  It splits the numeric frame table into
the episode layout expected by the pinned LeRobot package and indexes existing
chunked MP4 files with adjusted timestamps.  Media is materialized inside the
new dataset (normally by hard link), so the v3 source is provenance rather than
a runtime dependency.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clothdojo.data.lerobot import EXPORT_FORMAT_VERSION, EXPORT_SCHEMA


@dataclass(frozen=True, slots=True)
class IndexedView:
    target_key: str
    source_root: Path
    source_key: str


@dataclass(frozen=True, slots=True)
class IndexedViewSpec:
    config_path: Path
    source_root: Path
    output_root: Path
    repo_id: str
    views: tuple[IndexedView, ...]


def compose_indexed_view_spec(config_path: str | Path) -> IndexedViewSpec:
    """Resolve one explicit, versioned v3-to-v2.1 migration config."""

    from omegaconf import OmegaConf

    path = Path(config_path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("indexed-view config must be a mapping")
    if raw.get("schema") != "clothdojo_lerobot_v3_indexed_view":
        raise ValueError("unsupported indexed-view config schema")
    if raw.get("format_version") != 1:
        raise ValueError("unsupported indexed-view config version")
    source = _directory(raw.get("source_path"), "source_path")
    output_text = str(raw.get("output_path", "")).strip()
    if not output_text:
        raise ValueError("output_path must be explicit")
    output = Path(output_text).expanduser().resolve()
    repo_id = str(raw.get("repo_id", output.name)).strip()
    if not repo_id:
        raise ValueError("repo_id must be non-empty")
    entries = raw.get("views")
    if not isinstance(entries, list) or not entries:
        raise ValueError("indexed-view config requires at least one camera view")
    views = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("indexed-view camera entry must be a mapping")
        target_key = str(entry.get("target_key", ""))
        source_key = str(entry.get("source_key", ""))
        if not target_key.startswith("observation.images.") or not source_key.startswith(
            "observation.images."
        ):
            raise ValueError("camera keys must start with observation.images.")
        views.append(
            IndexedView(
                target_key=target_key,
                source_root=_directory(entry.get("source_path", source), "view source"),
                source_key=source_key,
            )
        )
    if len({view.target_key for view in views}) != len(views):
        raise ValueError("indexed-view target camera keys must be unique")
    spec = IndexedViewSpec(path, source, output, repo_id, tuple(views))
    inspect_indexed_view(spec)
    return spec


def inspect_indexed_view(spec: IndexedViewSpec) -> dict[str, Any]:
    """Validate source revisions, episode alignment and camera mappings."""

    source_info = _object(spec.source_root / "meta" / "info.json")
    if source_info.get("codebase_version") != "v3.0":
        raise ValueError("indexed-view source must be LeRobot v3.0")
    source_episodes = _episode_table(spec.source_root)
    total_episodes = _positive_int(source_info.get("total_episodes"), "total_episodes")
    total_frames = _positive_int(source_info.get("total_frames"), "total_frames")
    if source_episodes.num_rows != total_episodes:
        raise ValueError("v3 episode metadata count differs from info.json")
    if sum(source_episodes["length"].to_pylist()) != total_frames:
        raise ValueError("v3 episode lengths do not sum to total_frames")
    identity = _episode_identity(source_episodes)
    for view in spec.views:
        info = _object(view.source_root / "meta" / "info.json")
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"view source must be LeRobot v3.0: {view.source_root}")
        feature = info.get("features", {}).get(view.source_key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"source camera is not a video feature: {view.source_key}")
        if _episode_identity(_episode_table(view.source_root)) != identity:
            raise ValueError("indexed-view sources have different episode boundaries")
    return {
        "compatible": True,
        "schema": "clothdojo_lerobot_v3_indexed_view_plan",
        "format_version": 1,
        "source_path": str(spec.source_root),
        "output_path": str(spec.output_root),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "views": [
            {
                "target_key": view.target_key,
                "source_path": str(view.source_root),
                "source_key": view.source_key,
            }
            for view in spec.views
        ],
    }


def publish_indexed_view(spec: IndexedViewSpec) -> dict[str, Any]:
    """Transactionally publish the configured v2.1 dataset."""

    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq

    plan = inspect_indexed_view(spec)
    output = spec.output_root
    if output.exists():
        raise FileExistsError(f"indexed-view output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp"
    if staging.exists():
        raise FileExistsError(f"unfinished indexed-view staging directory exists: {staging}")
    staging.mkdir()
    try:
        source_info = _object(spec.source_root / "meta" / "info.json")
        episode_table = _episode_table(spec.source_root)
        episode_rows = episode_table.to_pylist()
        view_rows = {
            view.target_key: _episode_table(view.source_root).to_pylist()
            for view in spec.views
        }
        frame_table = pads.dataset(spec.source_root / "data", format="parquet").to_table()
        if frame_table.num_rows != plan["total_frames"]:
            raise ValueError("v3 frame table count differs from info.json")
        indices = frame_table["index"].to_pylist()
        if indices != list(range(frame_table.num_rows)):
            order = pc.sort_indices(frame_table, sort_keys=[("index", "ascending")])
            frame_table = pc.take(frame_table, order)
            if frame_table["index"].to_pylist() != list(range(frame_table.num_rows)):
                raise ValueError("v3 frame indices are not contiguous")

        chunks_size = 1000
        data_template = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        video_template = (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        features = {
            key: value
            for key, value in source_info["features"].items()
            if value.get("dtype") != "video"
        }
        for view in spec.views:
            source_feature = _object(view.source_root / "meta" / "info.json")["features"]
            features[view.target_key] = source_feature[view.source_key]

        media_modes: set[str] = set()
        shared_media: dict[tuple[str, int, int], Path] = {}
        media_index = []
        episodes_json = []
        episode_stats = []
        for episode_index, row in enumerate(episode_rows):
            if row.get("episode_index") != episode_index:
                raise ValueError("v3 episode indices are not contiguous")
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            length = int(row["length"])
            if end - start != length:
                raise ValueError(f"v3 episode frame range is invalid: {episode_index}")
            table = frame_table.slice(start, length)
            _require_episode_columns(table, episode_index, length, start)

            data_path = staging / _v21_path(
                data_template, episode_index, chunks_size=chunks_size
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, data_path)

            episode_offsets = {}
            for view in spec.views:
                current = view_rows[view.target_key][episode_index]
                episode_offsets[view.target_key] = float(
                    current[f"videos/{view.source_key}/from_timestamp"]
                )
                chunk_index = int(current[f"videos/{view.source_key}/chunk_index"])
                file_index = int(current[f"videos/{view.source_key}/file_index"])
                source_video = view.source_root / _v3_video_path(
                    _object(view.source_root / "meta" / "info.json")["video_path"],
                    view.source_key,
                    chunk_index,
                    file_index,
                )
                identity = (view.target_key, chunk_index, file_index)
                shared = shared_media.get(identity)
                if shared is None:
                    shared = (
                        staging
                        / "videos"
                        / ".shared"
                        / view.target_key
                        / f"chunk-{chunk_index:03d}"
                        / f"file-{file_index:03d}.mp4"
                    )
                    shared.parent.mkdir(parents=True, exist_ok=True)
                    media_modes.add(_link_or_copy(source_video, shared))
                    shared_media[identity] = shared
                episode_video = staging / _v21_path(
                    video_template,
                    episode_index,
                    chunks_size=chunks_size,
                    video_key=view.target_key,
                )
                episode_video.parent.mkdir(parents=True, exist_ok=True)
                media_modes.add(_link_or_relative_symlink(shared, episode_video))

            media_index.append(
                {"episode_index": episode_index, "timestamp_offsets": episode_offsets}
            )

            episodes_json.append(
                {
                    "episode_index": episode_index,
                    "tasks": list(row.get("tasks", [])),
                    "length": length,
                }
            )
            episode_stats.append(
                {
                    "episode_index": episode_index,
                    "stats": _v21_stats(row, spec.views, view_rows, episode_index),
                }
            )

        tasks = _tasks(spec.source_root)
        info = dict(source_info)
        info.update(
            {
                "codebase_version": "v2.1",
                "total_episodes": plan["total_episodes"],
                "total_frames": plan["total_frames"],
                "total_tasks": len(tasks),
                "total_videos": plan["total_episodes"] * len(spec.views),
                "total_chunks": math.ceil(plan["total_episodes"] / chunks_size),
                "chunks_size": chunks_size,
                "splits": {"train": f"0:{plan['total_episodes']}"},
                "data_path": data_template,
                "video_path": video_template,
                "features": features,
            }
        )
        meta = staging / "meta"
        _write_json(meta / "info.json", info)
        _write_jsonl(meta / "tasks.jsonl", tasks)
        _write_jsonl(meta / "episodes.jsonl", episodes_json)
        _write_jsonl(meta / "episodes_stats.jsonl", episode_stats)
        _write_jsonl(meta / "clothdojo_video_index.jsonl", media_index)
        _write_json(
            meta / "clothdojo.json",
            {
                "schema": EXPORT_SCHEMA,
                "format_version": EXPORT_FORMAT_VERSION,
                "export_status": "complete",
                "training_eligible": True,
                "selection": {"rule": "all_source_episodes"},
                "source": {
                    "format": "LeRobot v3.0",
                    "dataset_root": str(spec.source_root),
                    "info_sha256": _sha256(spec.source_root / "meta" / "info.json"),
                    "runtime_dependency": False,
                },
                "migration": {
                    "kind": "immutable_chunked_media_index",
                    "config_sha256": _sha256(spec.config_path),
                    "media_modes": sorted(media_modes),
                    "shared_video_files": len(shared_media),
                    "timestamp_rule": "per_camera_v3_video_from_timestamp",
                    "video_index": "meta/clothdojo_video_index.jsonl",
                },
                "counts": {
                    "output_episodes": plan["total_episodes"],
                    "output_frames": plan["total_frames"],
                },
                "episode_sets": {
                    "good": list(range(plan["total_episodes"])),
                    "full_trajectory": list(range(plan["total_episodes"])),
                    "partial_trajectory": [],
                },
                "views": [
                    {
                        "target_key": view.target_key,
                        "source_key": view.source_key,
                        "source_info_sha256": _sha256(
                            view.source_root / "meta" / "info.json"
                        ),
                    }
                    for view in spec.views
                ],
            },
        )
        _validate_publication(staging, spec.repo_id, plan)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**plan, "output_path": str(output)}


def _validate_publication(root: Path, repo_id: str, plan: dict[str, Any]) -> None:
    from lerobot.common.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    if meta.total_episodes != plan["total_episodes"]:
        raise RuntimeError("pinned LeRobot sees a different episode count")
    dataset = LeRobotDataset(repo_id, root=root, download_videos=False)
    if len(dataset) != plan["total_frames"]:
        raise RuntimeError("pinned LeRobot sees a different frame count")


def _episode_table(root: Path) -> Any:
    import pyarrow.dataset as pads

    path = root / "meta" / "episodes"
    if not path.is_dir():
        raise FileNotFoundError(f"v3 episode metadata is missing: {path}")
    table = pads.dataset(path, format="parquet").to_table()
    import pyarrow.compute as pc

    order = pc.sort_indices(table, sort_keys=[("episode_index", "ascending")])
    return pc.take(table, order)


def _episode_identity(table: Any) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (
            int(row["episode_index"]),
            int(row["length"]),
            int(row["dataset_from_index"]),
            int(row["dataset_to_index"]),
        )
        for row in table.select(
            ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
        ).to_pylist()
    )


def _v21_stats(
    base_row: dict[str, Any],
    views: tuple[IndexedView, ...],
    view_rows: dict[str, list[dict[str, Any]]],
    episode_index: int,
) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for name, value in base_row.items():
        if not name.startswith("stats/"):
            continue
        feature, metric = name.removeprefix("stats/").rsplit("/", 1)
        if feature.startswith("observation.images."):
            continue
        stats.setdefault(feature, {})[metric] = value
    for view in views:
        row = view_rows[view.target_key][episode_index]
        prefix = f"stats/{view.source_key}/"
        stats[view.target_key] = {
            name.removeprefix(prefix): value
            for name, value in row.items()
            if name.startswith(prefix)
        }
    return stats


def _tasks(root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = root / "meta" / "tasks.parquet"
    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda row: row["task_index"])
    return rows


def _require_episode_columns(table: Any, episode: int, length: int, start: int) -> None:
    if table.num_rows != length:
        raise ValueError(f"v3 episode {episode} has the wrong frame count")
    if table["episode_index"].to_pylist() != [episode] * length:
        raise ValueError(f"v3 episode_index column is invalid: {episode}")
    if table["frame_index"].to_pylist() != list(range(length)):
        raise ValueError(f"v3 frame_index column is invalid: {episode}")
    if table["index"].to_pylist() != list(range(start, start + length)):
        raise ValueError(f"v3 global index column is invalid: {episode}")


def _v21_path(template: str, episode: int, *, chunks_size: int, video_key: str = "") -> Path:
    path = Path(
        template.format(
            episode_chunk=episode // chunks_size,
            episode_index=episode,
            video_key=video_key,
        )
    )
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"dataset path escapes root: {path}")
    return path


def _v3_video_path(template: str, key: str, chunk: int, file: int) -> Path:
    path = Path(
        template.format(video_key=key, chunk_index=chunk, file_index=file)
    )
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"v3 video path escapes root: {path}")
    return path


def _link_or_copy(source: Path, target: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"source video is missing: {source}")
    try:
        os.link(source, target)
        return "immutable_hardlink_import"
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EXDEV, errno.EOPNOTSUPP}:
            raise
        shutil.copy2(source, target)
        return "immutable_copy_import"


def _link_or_relative_symlink(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "immutable_hardlink_index"
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EXDEV, errno.EOPNOTSUPP, errno.EMLINK}:
            raise
        target.symlink_to(os.path.relpath(source, target.parent))
        return "internal_relative_symlink_index"


def _directory(value: Any, field: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON metadata must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
