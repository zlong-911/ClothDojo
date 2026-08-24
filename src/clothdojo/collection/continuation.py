"""Terminal-state continuation plans over canonical ClothDojo trajectories."""

from __future__ import annotations

import hashlib
import json
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
    excluded = _consumed_sources(profile, source.id, asset_id) | set(leased_episode_ids or ())
    candidates = [
        item
        for item in continuation_sources(config, profile)
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
    source = _source_profile(config, profile)
    result = []
    root = source.trajectory_root.resolve()
    if not root.is_dir():
        return result
    for manifest_path in sorted(root.glob("*/manifest.json")):
        trajectory = manifest_path.parent
        if trajectory.is_symlink() or trajectory.resolve().parent != root:
            continue
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
        if manifest.scene.task_name != source.scene or manifest.scene.robot_name != "piper":
            continue
        asset_fingerprints = manifest.scene.asset_fingerprints
        expected = _asset_fingerprints(config, asset_id)
        if any(asset_fingerprints.get(key) != value for key, value in expected.items()):
            continue
        result.append(
            {
                "source_episode_id": trajectory.name,
                "source_sample_index": manifest.state_count - 1,
                "trajectory": str(trajectory.resolve()),
                "manifest_sha256": _sha256(manifest_path),
                "asset_id": asset_id,
                "runtime": {
                    "robot_name": manifest.scene.robot_name,
                    "control_fps": manifest.control_fps,
                    "sim_substeps": manifest.sim_substeps,
                    "asset_fingerprints": dict(asset_fingerprints),
                    "config_fingerprints": dict(manifest.scene.config_fingerprints),
                },
            }
        )
    return result


def pending_continuation_counts(
    config: CollectionConfig,
    profile: CollectionProfile,
) -> dict[str, int]:
    source = _source_profile(config, profile)
    consumed_by_asset: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for item in continuation_sources(config, profile):
        asset_id = item["asset_id"]
        consumed = consumed_by_asset.setdefault(
            asset_id,
            _consumed_sources(profile, source.id, asset_id),
        )
        if item["source_episode_id"] not in consumed:
            counts[asset_id] = counts.get(asset_id, 0) + 1
    return counts


def _source_profile(config: CollectionConfig, profile: CollectionProfile) -> CollectionProfile:
    source_id = profile.continuation_source_profile
    if source_id is None:
        raise ValueError(f"profile is not a continuation workflow: {profile.id}")
    return config.profiles[source_id]


def _consumed_sources(
    profile: CollectionProfile,
    source_profile_id: str,
    asset_id: str,
) -> set[str]:
    consumed = set()
    root = profile.trajectory_root.resolve()
    if not root.is_dir():
        return consumed
    for path in root.glob("*/manifest.json"):
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
            and metadata.get("asset_id") == asset_id
            and isinstance(metadata.get("source_episode_id"), str)
        ):
            consumed.add(metadata["source_episode_id"])
    return consumed


def _asset_fingerprints(config: CollectionConfig, asset_id: str | None) -> dict[str, str]:
    value = {
        "piper_entry_usda": _sha256(config.assets_root / "robots/piper/piper.usda"),
    }
    if asset_id is not None:
        garment = (
            config.assets_root
            / "garments"
            / "converted"
            / asset_id
            / f"{asset_id}.usdc"
        )
        value["garment_usdc"] = _sha256(garment)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
