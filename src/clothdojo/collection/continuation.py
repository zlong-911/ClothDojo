"""Terminal-state continuation plans over canonical ClothDojo trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from clothdojo.collection.config import CollectionConfig, CollectionProfile
from clothdojo.teleop.restoration import CONTINUATION_PLAN_SCHEMA
from clothdojo.trajectory import TrajectoryReader


def build_continuation_plan(
    config: CollectionConfig,
    profile: CollectionProfile,
    *,
    asset_id: str,
    leased_episode_ids: set[str] | None = None,
) -> dict[str, Any]:
    source = _source_profile(config, profile)
    sources = continuation_sources(config, profile)
    excluded = _consumed_source_index(profile, source.id).get(asset_id, set()) | set(
        leased_episode_ids or ()
    )
    candidates = [
        item
        for item in sources
        if item["asset_id"] == asset_id and item["source_episode_id"] not in excluded
    ]
    if not candidates:
        raise ValueError(f"no uncollected continuation state is available for asset {asset_id}")
    identity = candidates[0]["runtime"]
    candidates = [item for item in candidates if item["runtime"] == identity]
    return {
        "schema": CONTINUATION_PLAN_SCHEMA,
        "profile_id": profile.id,
        "source_profile_id": source.id,
        "asset_id": asset_id,
        "runtime": {
            **identity,
            "task_name": profile.scene,
            "source_task_name": source.scene,
        },
        "sources": [
            {
                key: item[key]
                for key in (
                    "source_episode_id",
                    "source_sample_index",
                    "trajectory",
                    "manifest_sha256",
                )
            }
            for item in candidates
        ],
    }


def continuation_sources(
    config: CollectionConfig,
    profile: CollectionProfile,
) -> list[dict[str, Any]]:
    """Return compatible terminal states, reusing the inventory until files change."""

    source = _source_profile(config, profile)
    root = source.trajectory_root.resolve()
    if not root.is_dir():
        return []
    value = _continuation_sources_cached(
        str(root),
        _manifest_stamps(root),
        str(config.assets_root.resolve()),
        _asset_stamps(config.assets_root.resolve()),
        source.id,
        source.scene,
    )
    return copy.deepcopy(value)


def pending_continuation_counts(
    config: CollectionConfig,
    profile: CollectionProfile,
) -> dict[str, int]:
    source = _source_profile(config, profile)
    sources = continuation_sources(config, profile)
    consumed_by_asset = _consumed_source_index(profile, source.id)
    counts: dict[str, int] = {}
    for item in sources:
        asset_id = item["asset_id"]
        consumed = consumed_by_asset.get(asset_id, set())
        if item["source_episode_id"] not in consumed:
            counts[asset_id] = counts.get(asset_id, 0) + 1
    return counts


@lru_cache(maxsize=32)
def _continuation_sources_cached(
    root_text: str,
    manifest_stamps: tuple[tuple[str, int, int], ...],
    assets_root_text: str,
    _assets_stamps: tuple[tuple[str, int, int], ...],
    source_profile_id: str,
    source_scene: str,
) -> tuple[dict[str, Any], ...]:
    root = Path(root_text)
    assets_root = Path(assets_root_text)
    result = []
    for episode_id, mtime_ns, size in manifest_stamps:
        trajectory = root / episode_id
        manifest_path = trajectory / "manifest.json"
        try:
            reader = TrajectoryReader(trajectory)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            continue
        manifest = reader.manifest
        asset_id = manifest.metadata.get("asset_id")
        if (
            not isinstance(asset_id, str)
            or manifest.status != "complete"
            or manifest.action_count <= 0
            or manifest.state_count != manifest.action_count + 1
        ):
            continue
        if manifest.scene.task_name != source_scene or manifest.scene.robot_name != "piper":
            continue
        expected = _asset_fingerprints(assets_root, asset_id)
        recorded = manifest.scene.asset_fingerprints
        if any(recorded.get(key) != value for key, value in expected.items()):
            continue
        result.append(
            {
                "source_episode_id": episode_id,
                "source_sample_index": manifest.state_count - 1,
                "trajectory": str(trajectory),
                "manifest_sha256": _sha256_versioned(str(manifest_path), mtime_ns, size),
                "asset_id": asset_id,
                "source_profile_id": source_profile_id,
                "runtime": {
                    "robot_name": manifest.scene.robot_name,
                    "control_fps": manifest.control_fps,
                    "sim_substeps": manifest.sim_substeps,
                    "asset_fingerprints": expected,
                    "config_fingerprints": dict(manifest.scene.config_fingerprints),
                },
            }
        )
    return tuple(result)


def _source_profile(config: CollectionConfig, profile: CollectionProfile) -> CollectionProfile:
    source_id = profile.continuation_source_profile
    if source_id is None:
        raise ValueError(f"profile is not a continuation workflow: {profile.id}")
    return config.profiles[source_id]


def _consumed_source_index(
    profile: CollectionProfile,
    source_profile_id: str,
) -> dict[str, set[str]]:
    root = profile.trajectory_root.resolve()
    if not root.is_dir():
        return {}
    value = _consumed_sources_cached(
        str(root),
        _manifest_stamps(root),
        source_profile_id,
    )
    result: dict[str, set[str]] = {}
    for asset_id, episode_id in value:
        result.setdefault(asset_id, set()).add(episode_id)
    return result


@lru_cache(maxsize=256)
def _consumed_sources_cached(
    root_text: str,
    manifest_stamps: tuple[tuple[str, int, int], ...],
    source_profile_id: str,
) -> frozenset[tuple[str, str]]:
    root = Path(root_text)
    consumed: set[tuple[str, str]] = set()
    for episode_id, _mtime_ns, _size in manifest_stamps:
        path = root / episode_id / "manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if (
            value.get("status") == "complete"
            and int(value.get("action_count", 0)) > 0
            and metadata.get("source") == "task_continuation"
            and metadata.get("source_profile_id") == source_profile_id
            and isinstance(metadata.get("asset_id"), str)
            and isinstance(metadata.get("source_episode_id"), str)
        ):
            consumed.add((metadata["asset_id"], metadata["source_episode_id"]))
    return frozenset(consumed)


def _asset_fingerprints(assets_root: Path, asset_id: str) -> dict[str, str]:
    return {
        "piper_entry_usda": _sha256(assets_root / "robots/piper/piper.usda"),
        "garment_usdc": _sha256(
            assets_root / "garments" / "converted" / asset_id / f"{asset_id}.usdc"
        ),
    }


def _manifest_stamps(root: Path) -> tuple[tuple[str, int, int], ...]:
    result = []
    for manifest_path in root.glob("*/manifest.json"):
        trajectory = manifest_path.parent
        if trajectory.is_symlink() or trajectory.parent != root:
            continue
        try:
            stat = manifest_path.stat()
        except FileNotFoundError:
            continue
        result.append((trajectory.name, stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(result))


def _asset_stamps(assets_root: Path) -> tuple[tuple[str, int, int], ...]:
    paths = [assets_root / "robots/piper/piper.usda"]
    paths.extend(
        sorted((assets_root / "garments/converted").glob("*/*.usdc"))
    )
    result = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        result.append(
            (path.relative_to(assets_root).as_posix(), stat.st_mtime_ns, stat.st_size)
        )
    return tuple(result)


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_versioned(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4096)
def _sha256_versioned(path: str, _mtime_ns: int, _size: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
