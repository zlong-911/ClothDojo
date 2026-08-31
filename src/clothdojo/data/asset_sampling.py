"""Portable asset-tempered sampling metadata for LeRobot datasets.

The exported LeRobot rows intentionally remain unchanged.  This module builds a
small, content-addressed side index by joining the export sidecar's source
trajectory references with the corresponding collection manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "clothdojo_asset_sampling_index"
FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _dataset_hashes(meta: Path) -> DatasetHashes:
    return DatasetHashes(
        info_sha256=_sha256(meta / "info.json"),
        episodes_sha256=_sha256(meta / "episodes.jsonl"),
        sidecar_sha256=_sha256(meta / "clothdojo.json"),
    )


def _episode_metadata(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank line in episodes.jsonl at line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"episodes.jsonl line {line_number} is not an object")
        result.append(value)
    return result


@dataclass(frozen=True)
class DatasetHashes:
    """Hashes of the metadata that fixes LeRobot row ordering and provenance."""

    info_sha256: str
    episodes_sha256: str
    sidecar_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "episodes_sha256": self.episodes_sha256,
            "info_sha256": self.info_sha256,
            "sidecar_sha256": self.sidecar_sha256,
        }


@dataclass(frozen=True)
class EpisodeAssetEntry:
    """Asset ownership and extents for one contiguous LeRobot episode."""

    episode_index: int
    frame_start: int
    frame_end_exclusive: int
    asset_id: str
    source_trajectory: str
    source_action_start: int
    source_action_end_exclusive: int
    source_manifest_sha256: str

    @property
    def length(self) -> int:
        return self.frame_end_exclusive - self.frame_start

    def to_dict(self) -> dict[str, int | str]:
        return {
            "asset_id": self.asset_id,
            "episode_index": self.episode_index,
            "frame_end_exclusive": self.frame_end_exclusive,
            "frame_start": self.frame_start,
            "source_action_end_exclusive": self.source_action_end_exclusive,
            "source_action_start": self.source_action_start,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_trajectory": self.source_trajectory,
        }


@dataclass(frozen=True)
class AssetSamplingIndex:
    """Content-addressed mapping from LeRobot episodes to garment assets."""

    dataset: DatasetHashes
    episodes: tuple[EpisodeAssetEntry, ...]

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("asset sampling index must contain at least one episode")
        expected_frame_start = 0
        for expected_episode, entry in enumerate(self.episodes):
            if entry.episode_index != expected_episode:
                raise ValueError(
                    "episode indices must be contiguous from zero: "
                    f"expected {expected_episode}, got {entry.episode_index}"
                )
            if entry.frame_start != expected_frame_start:
                raise ValueError(
                    f"episode {entry.episode_index} frame_start must be "
                    f"{expected_frame_start}, got {entry.frame_start}"
                )
            if entry.length <= 0:
                raise ValueError(f"episode {entry.episode_index} must contain frames")
            if entry.source_action_start < 0:
                raise ValueError(f"episode {entry.episode_index} has a negative source extent")
            if entry.source_action_end_exclusive - entry.source_action_start != entry.length:
                raise ValueError(
                    f"episode {entry.episode_index} source and dataset extents differ"
                )
            _required_string(entry.asset_id, field="asset_id")
            _required_string(entry.source_trajectory, field="source_trajectory")
            if len(entry.source_manifest_sha256) != 64:
                raise ValueError("source_manifest_sha256 must be a SHA-256 hex digest")
            expected_frame_start = entry.frame_end_exclusive

    @property
    def total_frames(self) -> int:
        return self.episodes[-1].frame_end_exclusive

    @property
    def episode_to_asset(self) -> dict[int, str]:
        return {entry.episode_index: entry.asset_id for entry in self.episodes}

    @property
    def asset_frame_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for entry in self.episodes:
            counts[entry.asset_id] += entry.length
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset.to_dict(),
            "episodes": [entry.to_dict() for entry in self.episodes],
            "format_version": FORMAT_VERSION,
            "schema": SCHEMA,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


def build_asset_sampling_index(
    dataset_root: str | Path,
    *,
    collection_root: str | Path | None = None,
) -> AssetSamplingIndex:
    """Join LeRobot metadata to source manifests and validate every extent."""

    dataset_root = Path(dataset_root).resolve()
    meta = dataset_root / "meta"
    info_path = meta / "info.json"
    episodes_path = meta / "episodes.jsonl"
    sidecar_path = meta / "clothdojo.json"
    info = _json_object(info_path)
    sidecar = _json_object(sidecar_path)
    if sidecar.get("schema") != "clothdojo_lerobot_export":
        raise ValueError(f"unsupported LeRobot sidecar schema: {sidecar.get('schema')!r}")

    if collection_root is None:
        source = sidecar.get("source")
        if not isinstance(source, dict):
            raise ValueError("sidecar source must be an object")
        collection_root = _required_string(source.get("dataset_root"), field="source.dataset_root")
    collection_root = Path(collection_root).resolve()

    episode_metadata = _episode_metadata(episodes_path)

    sidecar_episodes = sidecar.get("episodes")
    if not isinstance(sidecar_episodes, list):
        raise ValueError("sidecar episodes must be a list")
    total_episodes = _required_int(info.get("total_episodes"), field="info.total_episodes")
    total_frames = _required_int(info.get("total_frames"), field="info.total_frames")
    if len(episode_metadata) != total_episodes or len(sidecar_episodes) != total_episodes:
        raise ValueError(
            "episode count mismatch among info.json, episodes.jsonl, and clothdojo.json"
        )

    manifest_cache: dict[str, tuple[int, str, str]] = {}
    entries: list[EpisodeAssetEntry] = []
    frame_start = 0
    for expected_index, (metadata, source_episode) in enumerate(
        zip(episode_metadata, sidecar_episodes, strict=True)
    ):
        if not isinstance(source_episode, dict):
            raise ValueError(f"sidecar episode {expected_index} is not an object")
        metadata_index = _required_int(
            metadata.get("episode_index"), field="episodes.episode_index"
        )
        source_index = _required_int(
            source_episode.get("episode_index"), field="sidecar.episodes.episode_index"
        )
        if metadata_index != expected_index or source_index != expected_index:
            raise ValueError(
                f"episode indices must be contiguous: expected {expected_index}, "
                f"got metadata={metadata_index}, sidecar={source_index}"
            )
        length = _required_int(metadata.get("length"), field="episodes.length")
        sidecar_length = _required_int(source_episode.get("length"), field="sidecar.length")
        action_start = _required_int(
            source_episode.get("action_start"), field="sidecar.action_start"
        )
        action_end = _required_int(
            source_episode.get("action_end_exclusive"),
            field="sidecar.action_end_exclusive",
        )
        if length <= 0 or sidecar_length != length or action_end - action_start != length:
            raise ValueError(f"episode {expected_index} has inconsistent extents")

        trajectory = _required_string(
            source_episode.get("source_trajectory"), field="sidecar.source_trajectory"
        )
        if trajectory not in manifest_cache:
            manifest_path = collection_root / "trajectories" / trajectory / "manifest.json"
            manifest = _json_object(manifest_path)
            manifest_actions = _required_int(
                manifest.get("action_count"), field="manifest.action_count"
            )
            manifest_metadata = manifest.get("metadata")
            if not isinstance(manifest_metadata, dict):
                raise ValueError(f"manifest metadata must be an object: {manifest_path}")
            asset_id = _required_string(
                manifest_metadata.get("asset_id"), field="manifest.metadata.asset_id"
            )
            manifest_cache[trajectory] = (manifest_actions, asset_id, _sha256(manifest_path))
        manifest_actions, asset_id, manifest_sha256 = manifest_cache[trajectory]
        source_actions = _required_int(
            source_episode.get("source_action_count"), field="sidecar.source_action_count"
        )
        valid_extent = 0 <= action_start < action_end <= manifest_actions
        if source_actions != manifest_actions or not valid_extent:
            raise ValueError(f"episode {expected_index} exceeds source manifest extent")

        frame_end = frame_start + length
        entries.append(
            EpisodeAssetEntry(
                episode_index=expected_index,
                frame_start=frame_start,
                frame_end_exclusive=frame_end,
                asset_id=asset_id,
                source_trajectory=trajectory,
                source_action_start=action_start,
                source_action_end_exclusive=action_end,
                source_manifest_sha256=manifest_sha256,
            )
        )
        frame_start = frame_end

    if frame_start != total_frames:
        raise ValueError(
            f"episode extents cover {frame_start} frames, info.json declares {total_frames}"
        )
    return AssetSamplingIndex(
        dataset=_dataset_hashes(meta),
        episodes=tuple(entries),
    )


def _index_from_dict(value: dict[str, Any]) -> AssetSamplingIndex:
    if value.get("schema") != SCHEMA or value.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported asset sampling index schema or format version")
    dataset = value.get("dataset")
    episodes = value.get("episodes")
    if not isinstance(dataset, dict) or not isinstance(episodes, list):
        raise ValueError("asset sampling index dataset/episodes fields are malformed")
    hashes = DatasetHashes(
        info_sha256=_required_string(dataset.get("info_sha256"), field="dataset.info_sha256"),
        episodes_sha256=_required_string(
            dataset.get("episodes_sha256"), field="dataset.episodes_sha256"
        ),
        sidecar_sha256=_required_string(
            dataset.get("sidecar_sha256"), field="dataset.sidecar_sha256"
        ),
    )
    for field, digest in hashes.to_dict().items():
        if len(digest) != 64:
            raise ValueError(f"dataset.{field} must be a SHA-256 hex digest")
    parsed = []
    for item in episodes:
        if not isinstance(item, dict):
            raise ValueError("asset sampling index episode is not an object")
        parsed.append(
            EpisodeAssetEntry(
                episode_index=_required_int(item.get("episode_index"), field="episode_index"),
                frame_start=_required_int(item.get("frame_start"), field="frame_start"),
                frame_end_exclusive=_required_int(
                    item.get("frame_end_exclusive"), field="frame_end_exclusive"
                ),
                asset_id=_required_string(item.get("asset_id"), field="asset_id"),
                source_trajectory=_required_string(
                    item.get("source_trajectory"), field="source_trajectory"
                ),
                source_action_start=_required_int(
                    item.get("source_action_start"), field="source_action_start"
                ),
                source_action_end_exclusive=_required_int(
                    item.get("source_action_end_exclusive"),
                    field="source_action_end_exclusive",
                ),
                source_manifest_sha256=_required_string(
                    item.get("source_manifest_sha256"), field="source_manifest_sha256"
                ),
            )
        )
    return AssetSamplingIndex(dataset=hashes, episodes=tuple(parsed))


def write_asset_sampling_index(index: AssetSamplingIndex, path: str | Path) -> Path:
    """Atomically publish an index with an embedded self-checking fingerprint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = index.to_dict()
    document["index_sha256"] = index.sha256
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_asset_sampling_index(path: str | Path) -> AssetSamplingIndex:
    """Load an index and verify its embedded fingerprint and invariants."""

    value = _json_object(Path(path))
    recorded_sha256 = value.pop("index_sha256", None)
    index = _index_from_dict(value)
    if recorded_sha256 != index.sha256:
        raise ValueError(
            f"asset sampling index fingerprint mismatch: expected {recorded_sha256!r}, "
            f"computed {index.sha256}"
        )
    return index


def validate_asset_sampling_index(
    index: AssetSamplingIndex,
    dataset_root: str | Path,
    *,
    collection_root: str | Path | None = None,
    validate_source_manifests: bool = False,
) -> None:
    """Verify hashes/extents, optionally re-reading the source collection manifests."""

    dataset_root = Path(dataset_root).resolve()
    meta = dataset_root / "meta"
    if _dataset_hashes(meta) != index.dataset:
        raise ValueError("asset sampling index dataset hash mismatch")
    info = _json_object(meta / "info.json")
    episode_metadata = _episode_metadata(meta / "episodes.jsonl")
    sidecar = _json_object(meta / "clothdojo.json")
    sidecar_episodes = sidecar.get("episodes")
    if not isinstance(sidecar_episodes, list):
        raise ValueError("sidecar episodes must be a list")
    total_episodes = _required_int(info.get("total_episodes"), field="info.total_episodes")
    total_frames = _required_int(info.get("total_frames"), field="info.total_frames")
    if total_episodes != len(index.episodes) or total_episodes != len(episode_metadata):
        raise ValueError("asset sampling index episode count does not match dataset metadata")
    if len(sidecar_episodes) != total_episodes or index.total_frames != total_frames:
        raise ValueError("asset sampling index extents do not match dataset metadata")
    for entry, metadata, source_episode in zip(
        index.episodes, episode_metadata, sidecar_episodes, strict=True
    ):
        if not isinstance(source_episode, dict):
            raise ValueError(f"sidecar episode {entry.episode_index} is not an object")
        expected = (
            entry.episode_index,
            entry.length,
            entry.source_trajectory,
            entry.source_action_start,
            entry.source_action_end_exclusive,
        )
        actual = (
            metadata.get("episode_index"),
            metadata.get("length"),
            source_episode.get("source_trajectory"),
            source_episode.get("action_start"),
            source_episode.get("action_end_exclusive"),
        )
        if actual != expected:
            raise ValueError(
                f"asset sampling index episode {entry.episode_index} does not match dataset extents"
            )
    if validate_source_manifests:
        rebuilt = build_asset_sampling_index(dataset_root, collection_root=collection_root)
        if rebuilt != index:
            raise ValueError("asset sampling index does not match source collection manifests")


def asset_tempered_frame_weights(index: AssetSamplingIndex, *, alpha: float):
    """Return per-frame weights ``n_asset ** (alpha - 1)`` as a CPU tensor."""

    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    import torch

    asset_counts = index.asset_frame_counts
    episode_weights = torch.tensor(
        [asset_counts[entry.asset_id] ** (alpha - 1.0) for entry in index.episodes],
        dtype=torch.double,
    )
    episode_lengths = torch.tensor([entry.length for entry in index.episodes], dtype=torch.long)
    return torch.repeat_interleave(episode_weights, episode_lengths)


def make_asset_tempered_sampler(
    index: AssetSamplingIndex,
    *,
    alpha: float,
    seed: int,
    num_samples: int | None = None,
    replacement: bool = True,
):
    """Construct a deterministic PyTorch ``WeightedRandomSampler``."""

    import torch

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if num_samples is None:
        num_samples = index.total_frames
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        asset_tempered_frame_weights(index, alpha=alpha),
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    )
