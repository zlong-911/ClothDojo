"""Turn sparse bad-range annotations into deterministic good segments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ANNOTATION_SCHEMA = "sim1_segments_v1"
BAD_LABEL = "bad"
KNOWN_LABELS = frozenset({BAD_LABEL, "good"})
# A 50-step Pi0.5 target needs 50 frames to contain one action chunk without
# terminal padding. The downstream STEAM evaluator uses pairs separated by
# k=64, which needs k+1 frames to produce one pair without terminal clamping.
# Both loaders can technically consume shorter episodes; this stricter shared
# floor is a reviewed-data quality policy rather than a runtime requirement.
DEFAULT_MIN_SEGMENT_FRAMES = 65


@dataclass(frozen=True, slots=True)
class GoodSegment:
    """One contiguous, bad-free range on a source trajectory action timeline."""

    source_trajectory: str
    action_start: int
    action_end_exclusive: int
    source_action_count: int
    full_trajectory: bool
    annotation_present: bool
    annotation_sha256: str | None
    review_status: str | None

    @property
    def length(self) -> int:
        return self.action_end_exclusive - self.action_start

    def to_json_dict(self, *, episode_index: int) -> dict[str, Any]:
        return {"episode_index": episode_index, **self.to_range_json_dict()}

    def to_range_json_dict(self) -> dict[str, Any]:
        """Return provenance for this source range without assigning an episode."""

        return {**asdict(self), "length": self.length}


@dataclass(frozen=True, slots=True)
class ExportPlan:
    """Complete selection plan for one good-only LeRobot export."""

    segments: tuple[GoodSegment, ...]
    source_trajectories: int
    source_frames: int
    skipped_incomplete_trajectories: int
    annotation_files: int
    default_good_trajectories: int
    trajectories_with_bad: int
    bad_ranges: int
    dropped_bad_frames: int
    min_segment_frames: int
    dropped_short_ranges: tuple[GoodSegment, ...]
    source_action_counts: tuple[tuple[str, int], ...]
    whole_trajectory_good_only: bool
    skipped_not_whole_trajectory_good: int

    @property
    def output_episodes(self) -> int:
        return len(self.segments)

    @property
    def output_frames(self) -> int:
        return sum(segment.length for segment in self.segments)

    @property
    def full_trajectory_episodes(self) -> int:
        return sum(segment.full_trajectory for segment in self.segments)

    @property
    def represented_source_trajectories(self) -> int:
        return len({segment.source_trajectory for segment in self.segments})

    @property
    def dropped_short_segments(self) -> int:
        return len(self.dropped_short_ranges)

    @property
    def dropped_short_frames(self) -> int:
        return sum(segment.length for segment in self.dropped_short_ranges)

    @property
    def dropped_short_full_trajectory_segments(self) -> int:
        return sum(segment.full_trajectory for segment in self.dropped_short_ranges)

    @property
    def dropped_short_partial_segments(self) -> int:
        return self.dropped_short_segments - self.dropped_short_full_trajectory_segments

    @property
    def dropped_short_source_ids(self) -> tuple[str, ...]:
        """Sources represented only by good ranges rejected by the length floor."""

        represented = {segment.source_trajectory for segment in self.segments}
        sources_with_short = {segment.source_trajectory for segment in self.dropped_short_ranges}
        return tuple(
            source_id
            for source_id, _ in self.source_action_counts
            if source_id in sources_with_short and source_id not in represented
        )

    @property
    def dropped_short_source_trajectories(self) -> int:
        return len(self.dropped_short_source_ids)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_trajectories": self.source_trajectories,
            "represented_source_trajectories": self.represented_source_trajectories,
            "source_frames": self.source_frames,
            "skipped_incomplete_trajectories": self.skipped_incomplete_trajectories,
            "annotation_files": self.annotation_files,
            "default_good_trajectories": self.default_good_trajectories,
            "trajectories_with_bad": self.trajectories_with_bad,
            "bad_ranges": self.bad_ranges,
            "dropped_bad_frames": self.dropped_bad_frames,
            "whole_trajectory_good_only": self.whole_trajectory_good_only,
            "skipped_not_whole_trajectory_good": (
                self.skipped_not_whole_trajectory_good
            ),
            "min_segment_frames": self.min_segment_frames,
            "dropped_short_segments": self.dropped_short_segments,
            "dropped_short_frames": self.dropped_short_frames,
            "dropped_short_full_trajectory_segments": (self.dropped_short_full_trajectory_segments),
            "dropped_short_partial_segments": self.dropped_short_partial_segments,
            "dropped_short_source_trajectories": (self.dropped_short_source_trajectories),
            "output_episodes": self.output_episodes,
            "output_frames": self.output_frames,
            "full_trajectory_episodes": self.full_trajectory_episodes,
            "partial_trajectory_episodes": (
                self.output_episodes - self.full_trajectory_episodes
            ),
            "source_action_counts": dict(self.source_action_counts),
            "dropped_short_source_ids": list(self.dropped_short_source_ids),
            "dropped_short_ranges": [
                segment.to_range_json_dict() for segment in self.dropped_short_ranges
            ],
            "episodes": [
                segment.to_json_dict(episode_index=index)
                for index, segment in enumerate(self.segments)
            ],
        }


def build_good_segment_plan(
    dataset_root: str | Path,
    *,
    min_segment_frames: int = DEFAULT_MIN_SEGMENT_FRAMES,
    whole_trajectory_good_only: bool = False,
) -> ExportPlan:
    """Build good ranges from ``trajectories`` and sparse bad annotations.

    Annotation endpoints are inclusive action indices. A missing annotation,
    an empty annotation, and an explicit ``good`` annotation all retain data.
    By default, ``review_status`` is recorded for provenance but does not affect
    selection. ``whole_trajectory_good_only`` instead requires an explicit
    ``review_status=complete`` annotation with no bad range, matching the
    collection console's whole-trajectory Good state.
    """

    if (
        isinstance(min_segment_frames, bool)
        or not isinstance(min_segment_frames, int)
        or min_segment_frames < 1
    ):
        raise ValueError("min_segment_frames must be a positive integer")

    root = Path(dataset_root)
    trajectories_root = root / "trajectories"
    annotations_root = trajectories_root / "annotations"
    if not trajectories_root.is_dir():
        raise NotADirectoryError(f"trajectory directory is missing: {trajectories_root}")
    if annotations_root.exists() and not annotations_root.is_dir():
        raise NotADirectoryError(f"annotation path is not a directory: {annotations_root}")

    manifest_paths = sorted(
        path
        for path in trajectories_root.glob("*/manifest.json")
        if path.parent.name != "annotations"
    )
    if not manifest_paths:
        raise RuntimeError(f"no trajectory manifests found in {trajectories_root}")

    segments: list[GoodSegment] = []
    source_frames = 0
    skipped_incomplete = 0
    annotation_files = 0
    default_good = 0
    trajectories_with_bad = 0
    bad_ranges = 0
    dropped_bad_frames = 0
    dropped_short_ranges: list[GoodSegment] = []
    skipped_not_whole_trajectory_good = 0
    source_ids: set[str] = set()
    source_action_counts: list[tuple[str, int]] = []

    for manifest_path in manifest_paths:
        manifest = _read_object(manifest_path, "trajectory manifest")
        if manifest.get("status") != "complete":
            skipped_incomplete += 1
            continue
        source_id = manifest_path.parent.name
        if source_id in source_ids:
            raise RuntimeError(f"duplicate trajectory id: {source_id}")
        source_ids.add(source_id)
        action_count = _nonnegative_integer(manifest.get("action_count"), "action_count")
        state_count = _nonnegative_integer(manifest.get("state_count"), "state_count")
        if state_count != action_count + 1:
            raise RuntimeError(
                f"trajectory {source_id} has states={state_count}, actions={action_count}"
            )
        if action_count == 0:
            raise RuntimeError(f"complete trajectory has no actions: {source_id}")
        source_frames += action_count
        source_action_counts.append((source_id, action_count))

        annotation_path = annotations_root / f"{source_id}.segments.json"
        annotation_present = annotation_path.is_file()
        review_status: str | None = None
        drop: list[tuple[int, int]] = []
        if annotation_present:
            annotation_files += 1
            annotation, annotation_sha256 = _read_object_with_sha256(
                annotation_path, "annotation"
            )
            if annotation.get("schema") != ANNOTATION_SCHEMA:
                raise RuntimeError(
                    f"annotation schema must be {ANNOTATION_SCHEMA!r}: {annotation_path}"
                )
            raw_review_status = annotation.get("review_status")
            if raw_review_status is not None and not isinstance(raw_review_status, str):
                raise RuntimeError(f"review_status must be a string: {annotation_path}")
            review_status = raw_review_status
            raw_segments = annotation.get("segments")
            if not isinstance(raw_segments, list):
                raise RuntimeError(f"segments must be a list: {annotation_path}")
            for raw in raw_segments:
                parsed = _parse_annotation_segment(raw, action_count, annotation_path)
                if parsed is not None:
                    drop.append(parsed)
        else:
            default_good += 1
            annotation_sha256 = None

        merged = _merge_ranges(drop)
        if whole_trajectory_good_only and (
            review_status != "complete" or bool(merged)
        ):
            skipped_not_whole_trajectory_good += 1
            continue
        if merged:
            trajectories_with_bad += 1
        bad_ranges += len(merged)
        dropped_bad_frames += sum(end - start for start, end in merged)
        for start, end in _range_complement(merged, action_count):
            segment = GoodSegment(
                source_trajectory=source_id,
                action_start=start,
                action_end_exclusive=end,
                source_action_count=action_count,
                full_trajectory=start == 0 and end == action_count,
                annotation_present=annotation_present,
                annotation_sha256=annotation_sha256,
                review_status=review_status,
            )
            if segment.length < min_segment_frames:
                dropped_short_ranges.append(segment)
                continue
            segments.append(segment)

    orphan_annotations = sorted(
        path.name.removesuffix(".segments.json")
        for path in annotations_root.glob("*.segments.json")
        if path.name.removesuffix(".segments.json") not in source_ids
    )
    if orphan_annotations:
        raise RuntimeError(f"annotations do not match complete trajectories: {orphan_annotations}")
    if not segments:
        raise RuntimeError(
            "bad annotations and the minimum segment length remove every training frame"
        )

    return ExportPlan(
        segments=tuple(segments),
        source_trajectories=len(source_ids),
        source_frames=source_frames,
        skipped_incomplete_trajectories=skipped_incomplete,
        annotation_files=annotation_files,
        default_good_trajectories=default_good,
        trajectories_with_bad=trajectories_with_bad,
        bad_ranges=bad_ranges,
        dropped_bad_frames=dropped_bad_frames,
        min_segment_frames=min_segment_frames,
        dropped_short_ranges=tuple(dropped_short_ranges),
        source_action_counts=tuple(source_action_counts),
        whole_trajectory_good_only=whole_trajectory_good_only,
        skipped_not_whole_trajectory_good=skipped_not_whole_trajectory_good,
    )


def _parse_annotation_segment(
    raw: object,
    action_count: int,
    path: Path,
) -> tuple[int, int] | None:
    if not isinstance(raw, dict):
        raise RuntimeError(f"annotation segment must be an object: {path}: {raw!r}")
    label = raw.get("label")
    if label not in KNOWN_LABELS:
        raise RuntimeError(
            f"annotation label must be one of {sorted(KNOWN_LABELS)}: {path}: {raw!r}"
        )
    if label != BAD_LABEL:
        return None
    start = _integer(raw.get("start_frame"), "start_frame")
    end_inclusive = _integer(raw.get("end_frame"), "end_frame")
    if start < 0 or end_inclusive < start or end_inclusive >= action_count:
        raise RuntimeError(
            f"bad range is outside [0,{action_count}): {path}: {raw!r}"
        )
    return start, end_inclusive + 1


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], end)
        else:
            merged.append((start, end))
    return merged


def _range_complement(
    ranges: list[tuple[int, int]], action_count: int
) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    cursor = 0
    for start, end in (*ranges, (action_count, action_count)):
        if cursor < start:
            kept.append((cursor, start))
        cursor = max(cursor, end)
    return kept


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must be a JSON object: {path}")
    return payload


def _read_object_with_sha256(
    path: Path, context: str
) -> tuple[dict[str, Any], str]:
    try:
        content = path.read_bytes()
        payload = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must be a JSON object: {path}")
    return payload, hashlib.sha256(content).hexdigest()


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be an integer, got {value!r}")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise RuntimeError(f"{name} must be non-negative, got {result}")
    return result
