"""Export reviewed trajectories as one good-only LeRobot v2.1 dataset."""

from __future__ import annotations

import errno
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Protocol

from clothdojo.data.annotations import (
    DEFAULT_MIN_SEGMENT_FRAMES,
    ExportPlan,
    GoodSegment,
    build_good_segment_plan,
)
from clothdojo.data.review_source import (
    ReviewFrameReader,
    ReviewSource,
    discover_review_sources,
)

EXPORT_SCHEMA = "clothdojo_lerobot_export"
EXPORT_FORMAT_VERSION = 1
VIDEO_CODEC = "h264"
VIDEO_CRF = 23
VIDEO_ENCODER_WORKERS = 4


class _Writer(Protocol):
    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ExportResult:
    output_root: Path
    source_trajectories: int
    represented_source_trajectories: int
    source_frames: int
    output_episodes: int
    output_frames: int
    dropped_bad_frames: int
    min_segment_frames: int
    dropped_short_segments: int
    dropped_short_frames: int
    dropped_short_full_trajectory_segments: int
    dropped_short_partial_segments: int
    dropped_short_source_trajectories: int
    full_trajectory_episodes: int
    whole_trajectory_good_only: bool
    skipped_not_whole_trajectory_good: int
    camera_names: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "source_trajectories": self.source_trajectories,
            "represented_source_trajectories": self.represented_source_trajectories,
            "source_frames": self.source_frames,
            "output_episodes": self.output_episodes,
            "output_frames": self.output_frames,
            "dropped_bad_frames": self.dropped_bad_frames,
            "min_segment_frames": self.min_segment_frames,
            "dropped_short_segments": self.dropped_short_segments,
            "dropped_short_frames": self.dropped_short_frames,
            "dropped_short_full_trajectory_segments": (self.dropped_short_full_trajectory_segments),
            "dropped_short_partial_segments": self.dropped_short_partial_segments,
            "dropped_short_source_trajectories": (self.dropped_short_source_trajectories),
            "full_trajectory_episodes": self.full_trajectory_episodes,
            "whole_trajectory_good_only": self.whole_trajectory_good_only,
            "skipped_not_whole_trajectory_good": (
                self.skipped_not_whole_trajectory_good
            ),
            "partial_trajectory_episodes": (
                self.output_episodes - self.full_trajectory_episodes
            ),
            "camera_names": list(self.camera_names),
        }


def republish_review_dataset(
    dataset_root: str | Path,
    reusable_root: str | Path,
    output_root: str | Path,
    *,
    repo_id: str | None = None,
    min_segment_frames: int = DEFAULT_MIN_SEGMENT_FRAMES,
    whole_trajectory_good_only: bool = False,
) -> ExportResult:
    """Publish a stricter reviewed export without re-encoding unchanged media.

    Every selected segment must already exist byte-for-byte in a completed
    ClothDojo LeRobot v2.1 export. Parquet generated indices and metadata are
    rebuilt, while immutable MP4 files are hard-linked into the new atomic
    publication. A hard link has no path dependency on the reusable dataset.
    """

    _require_export_dependencies()
    source_root = Path(dataset_root).resolve()
    reusable = Path(reusable_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"LeRobot output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp"
    if staging.exists():
        raise FileExistsError(
            f"unfinished LeRobot staging directory exists; inspect it first: {staging}"
        )

    plan, sources = inspect_review_export(
        source_root,
        min_segment_frames=min_segment_frames,
        whole_trajectory_good_only=whole_trajectory_good_only,
        require_review_payload=False,
    )
    reusable_sidecar = _read_json_object(reusable / "meta" / "clothdojo.json")
    if (
        reusable_sidecar.get("schema") != EXPORT_SCHEMA
        or reusable_sidecar.get("format_version") != EXPORT_FORMAT_VERSION
        or reusable_sidecar.get("export_status") != "complete"
    ):
        raise RuntimeError(f"reusable dataset is not a completed ClothDojo export: {reusable}")
    reusable_info = _read_json_object(reusable / "meta" / "info.json")
    if reusable_info.get("codebase_version") != "v2.1":
        raise RuntimeError(f"reusable dataset must use LeRobot v2.1: {reusable}")
    reusable_episodes = reusable_sidecar.get("episodes")
    if not isinstance(reusable_episodes, list):
        raise RuntimeError(f"reusable dataset has no episode provenance: {reusable}")
    mapping = _match_reusable_episodes(plan, reusable_episodes, sources=sources)

    reference = sources[sorted(sources)[0]]
    if reusable_info.get("fps") != reference.fps or reusable_info.get(
        "robot_type"
    ) != reference.robot_type:
        raise RuntimeError("reusable dataset fps or robot type differs from review sources")
    reusable_features = reusable_info.get("features")
    if not isinstance(reusable_features, dict) or {
        key: {
            "dtype": value.get("dtype"),
            "shape": value.get("shape"),
            "names": value.get("names"),
        }
        for key, value in reusable_features.items()
        if key in reference.features
    } != reference.features:
        raise RuntimeError("reusable dataset feature schema differs from review sources")

    episode_rows = _read_json_lines(reusable / "meta" / "episodes.jsonl")
    stats_rows = _read_json_lines(reusable / "meta" / "episodes_stats.jsonl")
    episode_by_index = _rows_by_episode_index(episode_rows, "episode metadata")
    stats_by_index = _rows_by_episode_index(stats_rows, "episode statistics")
    chunks_size = reusable_info.get("chunks_size")
    if isinstance(chunks_size, bool) or not isinstance(chunks_size, int) or chunks_size < 1:
        raise RuntimeError("reusable dataset chunks_size is invalid")
    data_template = reusable_info.get("data_path")
    video_template = reusable_info.get("video_path")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise RuntimeError("reusable dataset paths are invalid")

    staging.mkdir()
    new_episode_rows = []
    new_stats_rows = []
    global_index = 0
    media_publication_modes: set[str] = set()
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        for new_index, (segment, old_index) in enumerate(
            zip(plan.segments, mapping, strict=True)
        ):
            old_data = reusable / _v21_path(
                data_template, old_index, chunks_size=chunks_size
            )
            table = parquet.read_table(old_data)
            if table.num_rows != segment.length:
                raise RuntimeError(
                    f"reusable Parquet length differs for episode {old_index}: "
                    f"{table.num_rows} != {segment.length}"
                )
            _require_constant_column(table, "episode_index", old_index)
            _require_range_column(table, "frame_index", 0, segment.length)
            table = table.set_column(
                table.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array([new_index] * segment.length, type=table["episode_index"].type),
            )
            table = table.set_column(
                table.schema.get_field_index("index"),
                "index",
                pa.array(
                    range(global_index, global_index + segment.length),
                    type=table["index"].type,
                ),
            )
            new_data = staging / _v21_path(
                data_template, new_index, chunks_size=chunks_size
            )
            new_data.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_table(table, new_data)

            for key in reference.video_keys:
                old_video = reusable / _v21_path(
                    video_template,
                    old_index,
                    chunks_size=chunks_size,
                    video_key=key,
                )
                if not old_video.is_file() or old_video.is_symlink():
                    raise RuntimeError(f"reusable video is missing or indirect: {old_video}")
                new_video = staging / _v21_path(
                    video_template,
                    new_index,
                    chunks_size=chunks_size,
                    video_key=key,
                )
                new_video.parent.mkdir(parents=True, exist_ok=True)
                media_publication_modes.add(
                    _publish_immutable_video(old_video, new_video)
                )

            old_episode = dict(episode_by_index[old_index])
            if old_episode.get("length") != segment.length:
                raise RuntimeError(f"reusable episode metadata length differs: {old_index}")
            old_episode["episode_index"] = new_index
            new_episode_rows.append(old_episode)
            new_stats_rows.append(
                _reindex_episode_stats(
                    stats_by_index[old_index],
                    episode_index=new_index,
                    global_start=global_index,
                    length=segment.length,
                )
            )
            global_index += segment.length

        target_info = dict(reusable_info)
        target_info.update(
            {
                "total_episodes": plan.output_episodes,
                "total_frames": plan.output_frames,
                "total_videos": plan.output_episodes * len(reference.video_keys),
                "total_chunks": (plan.output_episodes + chunks_size - 1) // chunks_size,
                "splits": {"train": f"0:{plan.output_episodes}"},
            }
        )
        meta = staging / "meta"
        _write_json(meta / "info.json", target_info)
        _write_json_lines(meta / "episodes.jsonl", new_episode_rows)
        _write_json_lines(meta / "episodes_stats.jsonl", new_stats_rows)
        tasks = reusable / "meta" / "tasks.jsonl"
        if not tasks.is_file() or tasks.is_symlink():
            raise RuntimeError(f"reusable task metadata is missing: {tasks}")
        (meta / "tasks.jsonl").write_bytes(tasks.read_bytes())
        _write_sidecar(
            staging,
            source_root,
            plan,
            sources,
            reused_media_from=reusable,
            media_publication_modes=media_publication_modes,
        )
        _validate_republished_output(
            staging,
            plan,
            repo_id=repo_id or f"local/{output.name}",
            video_keys=reference.video_keys,
            reusable_root=reusable,
            mapping=mapping,
            chunks_size=chunks_size,
            video_template=video_template,
            media_publication_modes=media_publication_modes,
        )
        _require_inputs_unchanged(
            source_root,
            plan,
            sources,
            require_review_payload=False,
        )
        staging.rename(output)
    except BaseException:
        # Keep transactional staging for diagnosis; it is never mistaken for a
        # published dataset and a rerun refuses to overwrite it.
        raise

    return ExportResult(
        output_root=output,
        source_trajectories=plan.source_trajectories,
        represented_source_trajectories=plan.represented_source_trajectories,
        source_frames=plan.source_frames,
        output_episodes=plan.output_episodes,
        output_frames=plan.output_frames,
        dropped_bad_frames=plan.dropped_bad_frames,
        min_segment_frames=plan.min_segment_frames,
        dropped_short_segments=plan.dropped_short_segments,
        dropped_short_frames=plan.dropped_short_frames,
        dropped_short_full_trajectory_segments=(plan.dropped_short_full_trajectory_segments),
        dropped_short_partial_segments=plan.dropped_short_partial_segments,
        dropped_short_source_trajectories=plan.dropped_short_source_trajectories,
        full_trajectory_episodes=plan.full_trajectory_episodes,
        whole_trajectory_good_only=plan.whole_trajectory_good_only,
        skipped_not_whole_trajectory_good=plan.skipped_not_whole_trajectory_good,
        camera_names=tuple(key.removeprefix("observation.images.") for key in reference.video_keys),
    )


def inspect_review_export(
    dataset_root: str | Path,
    *,
    min_segment_frames: int = DEFAULT_MIN_SEGMENT_FRAMES,
    whole_trajectory_good_only: bool = False,
    require_review_payload: bool = True,
) -> tuple[ExportPlan, dict[str, ReviewSource]]:
    """Synchronize the trajectory/annotation/review indexes without writing output."""

    plan = build_good_segment_plan(
        dataset_root,
        min_segment_frames=min_segment_frames,
        whole_trajectory_good_only=whole_trajectory_good_only,
    )
    sources = discover_review_sources(
        dataset_root,
        require_payload=require_review_payload,
    )
    source_counts = dict(plan.source_action_counts)
    missing = sorted(set(source_counts) - set(sources))
    extra = sorted(set(sources) - set(source_counts))
    if missing or extra:
        raise RuntimeError(
            f"trajectory/review source sets differ: missing={missing}, extra={extra}"
        )
    for source_id, action_count in source_counts.items():
        if sources[source_id].frame_count != action_count:
            raise RuntimeError(
                f"trajectory/review frame count differs for {source_id}: "
                f"{action_count} != {sources[source_id].frame_count}"
            )
    _require_common_schema(sources.values())
    return plan, sources


def export_review_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    repo_id: str | None = None,
    image_writer_threads: int = 6,
    min_segment_frames: int = DEFAULT_MIN_SEGMENT_FRAMES,
    whole_trajectory_good_only: bool = False,
    _writer_factory: Callable[..., _Writer] | None = None,
    _reader_factory: Callable[[ReviewSource], Any] = ReviewFrameReader,
) -> ExportResult:
    """Convert v3 review caches into a transactional good-only v2.1 dataset."""

    if isinstance(image_writer_threads, bool) or image_writer_threads < 1:
        raise ValueError("image_writer_threads must be positive")
    source_root = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"LeRobot output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp"
    if staging.exists():
        raise FileExistsError(
            f"unfinished LeRobot staging directory exists; inspect it first: {staging}"
        )

    plan, sources = inspect_review_export(
        source_root,
        min_segment_frames=min_segment_frames,
        whole_trajectory_good_only=whole_trajectory_good_only,
    )
    reference = sources[sorted(sources)[0]]
    target_repo_id = repo_id or f"local/{output.name}"
    if _writer_factory is None:
        _require_export_dependencies()
    writer_factory = _writer_factory or _create_writer
    # The official LeRobot factory owns root creation and rejects a pre-created
    # directory. Test/injected writers do not create a filesystem root.
    if _writer_factory is not None:
        staging.mkdir()
    writer = writer_factory(
        repo_id=target_repo_id,
        root=staging,
        fps=reference.fps,
        robot_type=reference.robot_type,
        features=reference.features,
        image_writer_threads=image_writer_threads,
    )

    by_source: dict[str, list[GoodSegment]] = defaultdict(list)
    for segment in plan.segments:
        by_source[segment.source_trajectory].append(segment)
    written_episodes = 0
    written_frames = 0
    for source_id in sorted(sources):
        selected = sorted(by_source[source_id], key=lambda segment: segment.action_start)
        with _reader_factory(sources[source_id]) as reader:
            episode_count, frame_count = _copy_good_segments(reader, selected, writer)
        written_episodes += episode_count
        written_frames += frame_count
    if written_episodes != plan.output_episodes or written_frames != plan.output_frames:
        raise RuntimeError(
            "LeRobot writer input counts differ from plan: "
            f"episodes={written_episodes}/{plan.output_episodes}, "
            f"frames={written_frames}/{plan.output_frames}"
        )

    if _writer_factory is None:
        _validate_v21_output(
            staging,
            plan,
            repo_id=target_repo_id,
            video_keys=reference.video_keys,
        )
    sidecar = _write_sidecar(staging, source_root, plan, sources)
    if not sidecar.is_file():
        raise RuntimeError("ClothDojo export sidecar was not written")
    _require_inputs_unchanged(source_root, plan, sources)
    staging.rename(output)
    return ExportResult(
        output_root=output,
        source_trajectories=plan.source_trajectories,
        represented_source_trajectories=plan.represented_source_trajectories,
        source_frames=plan.source_frames,
        output_episodes=plan.output_episodes,
        output_frames=plan.output_frames,
        dropped_bad_frames=plan.dropped_bad_frames,
        min_segment_frames=plan.min_segment_frames,
        dropped_short_segments=plan.dropped_short_segments,
        dropped_short_frames=plan.dropped_short_frames,
        dropped_short_full_trajectory_segments=(plan.dropped_short_full_trajectory_segments),
        dropped_short_partial_segments=plan.dropped_short_partial_segments,
        dropped_short_source_trajectories=plan.dropped_short_source_trajectories,
        full_trajectory_episodes=plan.full_trajectory_episodes,
        whole_trajectory_good_only=plan.whole_trajectory_good_only,
        skipped_not_whole_trajectory_good=plan.skipped_not_whole_trajectory_good,
        camera_names=tuple(key.removeprefix("observation.images.") for key in reference.video_keys),
    )


def load_episode_indices(
    dataset_root: str | Path, *, trajectory_extent: str = "all"
) -> list[int]:
    """Return canonical good episode indices for STEAM or other consumers."""

    if trajectory_extent not in {"all", "full", "partial"}:
        raise ValueError("trajectory_extent must be 'all', 'full', or 'partial'")
    path = Path(dataset_root) / "meta" / "clothdojo.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read ClothDojo dataset metadata: {path}: {exc}") from exc
    if payload.get("schema") != EXPORT_SCHEMA or payload.get("format_version") != 1:
        raise RuntimeError(f"unsupported ClothDojo dataset metadata: {path}")
    if payload.get("export_status") != "complete":
        raise RuntimeError(f"ClothDojo dataset export is not complete: {path}")
    sets = payload.get("episode_sets")
    if not isinstance(sets, dict):
        raise RuntimeError(f"dataset episode_sets are missing: {path}")
    episodes = payload.get("episodes")
    counts = payload.get("counts")
    if not isinstance(episodes, list) or not isinstance(counts, dict):
        raise RuntimeError(f"dataset episode metadata is missing: {path}")
    total_episodes = counts.get("output_episodes")
    if (
        isinstance(total_episodes, bool)
        or not isinstance(total_episodes, int)
        or total_episodes != len(episodes)
    ):
        raise RuntimeError(f"dataset episode count is invalid: {path}")
    info_path = path.parent / "info.json"
    try:
        lerobot_total = json.loads(info_path.read_text(encoding="utf-8")).get(
            "total_episodes"
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read LeRobot dataset metadata: {info_path}: {exc}") from exc
    if lerobot_total != total_episodes:
        raise RuntimeError(f"ClothDojo and LeRobot episode counts differ: {path}")
    _validate_episode_sets(sets, total_episodes, path)
    set_name = {
        "all": "good",
        "full": "full_trajectory",
        "partial": "partial_trajectory",
    }[trajectory_extent]
    raw = sets[set_name]
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw
    ):
        raise RuntimeError(f"dataset episode set is invalid: {path}: {trajectory_extent}")
    return raw


def _copy_good_segments(
    reader: Iterable[tuple[int, dict[str, Any]]],
    segments: list[GoodSegment],
    writer: _Writer,
) -> tuple[int, int]:
    if not segments:
        return 0, 0
    position = 0
    written_episodes = 0
    written_frames = 0
    for frame_index, frame in reader:
        while position < len(segments) and frame_index >= segments[position].action_end_exclusive:
            writer.save_episode()
            written_episodes += 1
            position += 1
        if position == len(segments):
            continue
        segment = segments[position]
        if segment.action_start <= frame_index < segment.action_end_exclusive:
            writer.add_frame(frame)
            written_frames += 1
    if position < len(segments):
        if segments[position].action_end_exclusive != segments[position].source_action_count:
            raise RuntimeError("review stream ended before the final selected segment")
        writer.save_episode()
        written_episodes += 1
        position += 1
    if position != len(segments):
        raise RuntimeError("review stream did not reach every selected segment")
    return written_episodes, written_frames


def _create_writer(**kwargs) -> _Writer:
    import av
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    av.logging.set_level(av.logging.ERROR)
    writer = LeRobotDataset.create(
        repo_id=kwargs["repo_id"],
        root=kwargs["root"],
        fps=kwargs["fps"],
        robot_type=kwargs["robot_type"],
        features=_writer_features(kwargs["features"]),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=kwargs["image_writer_threads"],
    )
    # The pinned LeRobot writer serializes every camera through SVT-AV1.  Apart
    # from leaving most host cores idle, that encoder has produced truncated
    # streams in this environment. H.264 is a supported LeRobot codec and the
    # cameras are independent, so encode them concurrently without modifying
    # the pinned dependency.
    writer.encode_episode_videos = MethodType(_encode_episode_videos, writer)
    return writer


def _encode_episode_videos(writer: Any, episode_index: int) -> dict[str, str]:
    from lerobot.common.datasets.video_utils import encode_video_frames

    video_paths = {
        key: writer.root / writer.meta.get_video_file_path(episode_index, key)
        for key in writer.meta.video_keys
    }
    image_dirs = {
        key: writer._get_image_file_path(
            episode_index=episode_index,
            image_key=key,
            frame_index=0,
        ).parent
        for key in writer.meta.video_keys
    }
    expected_frames = {
        key: len(list(image_dir.glob("frame_*.png")))
        for key, image_dir in image_dirs.items()
    }

    def encode(key: str) -> None:
        video_path = video_paths[key]
        if video_path.is_file():
            return
        encode_video_frames(
            image_dirs[key],
            video_path,
            writer.fps,
            vcodec=VIDEO_CODEC,
            crf=VIDEO_CRF,
            log_level=None,
            overwrite=True,
        )

    with ThreadPoolExecutor(
        max_workers=min(VIDEO_ENCODER_WORKERS, len(video_paths))
    ) as pool:
        list(pool.map(encode, video_paths))
    for key, path in video_paths.items():
        _validate_video(path, expected_frames[key])
    return {key: str(path) for key, path in video_paths.items()}


def _writer_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Normalize JSON list shapes to the tuple contract used by LeRobot v2.1."""

    normalized = {}
    for key, feature in features.items():
        current = dict(feature)
        shape = current.get("shape")
        if isinstance(shape, list):
            current["shape"] = tuple(shape)
        normalized[key] = current
    return normalized


def _require_export_dependencies() -> None:
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyArrow, PyAV, and the pinned LeRobot package are required for export"
        ) from exc


def _require_common_schema(sources: Iterable[ReviewSource]) -> None:
    identity = None
    for source in sources:
        current = (
            source.fps,
            source.robot_type,
            json.dumps(source.features, sort_keys=True),
            source.video_keys,
        )
        if identity is None:
            identity = current
        elif current != identity:
            raise RuntimeError(f"review source schema differs: {source.root}")


def _source_identity(sources: dict[str, ReviewSource]) -> dict[str, tuple[str, str]]:
    return {
        source_id: (source.info_sha256, source.provenance_sha256)
        for source_id, source in sources.items()
    }


def _require_inputs_unchanged(
    source_root: Path,
    plan: ExportPlan,
    sources: dict[str, ReviewSource],
    *,
    require_review_payload: bool = True,
) -> None:
    current_plan, current_sources = inspect_review_export(
        source_root,
        min_segment_frames=plan.min_segment_frames,
        whole_trajectory_good_only=plan.whole_trajectory_good_only,
        require_review_payload=require_review_payload,
    )
    if current_plan.to_json_dict() != plan.to_json_dict() or _source_identity(
        current_sources
    ) != _source_identity(sources):
        raise RuntimeError("trajectory annotations or review metadata changed during export")


def _write_sidecar(
    root: Path,
    source_root: Path,
    plan: ExportPlan,
    sources: dict[str, ReviewSource],
    *,
    reused_media_from: Path | None = None,
    media_publication_modes: set[str] | None = None,
) -> Path:
    assets = _source_assets(source_root, plan.source_action_counts)
    episodes = []
    full_trajectory = []
    partial = []
    reviewed = []
    default_good = []
    for index, segment in enumerate(plan.segments):
        record = segment.to_json_dict(episode_index=index)
        record.update(
            {
                "quality": "good",
                "source_episode_complete": True,
                "contains_source_terminal": (
                    segment.action_end_exclusive == segment.source_action_count
                ),
                "review_complete": segment.review_status == "complete",
                "review_export": sources[segment.source_trajectory].root.name,
                "review_info_sha256": sources[segment.source_trajectory].info_sha256,
                "review_provenance_sha256": (
                    sources[segment.source_trajectory].provenance_sha256
                ),
            }
        )
        episodes.append(record)
        (full_trajectory if segment.full_trajectory else partial).append(index)
        if segment.annotation_present:
            reviewed.append(index)
        else:
            default_good.append(index)
    payload = {
        "schema": EXPORT_SCHEMA,
        "format_version": EXPORT_FORMAT_VERSION,
        "export_status": "complete",
        "selection": {
            "rule": (
                "explicit_whole_trajectory_good"
                if plan.whole_trajectory_good_only
                else "bad_complement_with_minimum_length"
            ),
            "bad_label": "bad",
            "annotation_endpoints": "inclusive_action_indices",
            "missing_annotation": "good",
            "review_status_affects_selection": plan.whole_trajectory_good_only,
            "required_review_status": (
                "complete" if plan.whole_trajectory_good_only else None
            ),
            "min_segment_frames": plan.min_segment_frames,
            "minimum_length_endpoint": "action_end_exclusive - action_start",
        },
        "dropped_short": {
            "source_trajectories_fully_removed": list(plan.dropped_short_source_ids),
            "ranges": [segment.to_range_json_dict() for segment in plan.dropped_short_ranges],
        },
        "source": {
            "dataset_root": str(source_root),
            "review_format": "LeRobot v3.0",
            "output_format": "LeRobot v2.1",
        },
        "video_encoding": {
            "codec": VIDEO_CODEC,
            "crf": VIDEO_CRF,
            "camera_parallelism": VIDEO_ENCODER_WORKERS,
        },
        "counts": {
            key: value
            for key, value in plan.to_json_dict().items()
            if key
            not in {
                "episodes",
                "source_action_counts",
                "dropped_short_source_ids",
                "dropped_short_ranges",
            }
        },
        "assets": sorted(assets),
        "episode_sets": {
            "good": list(range(plan.output_episodes)),
            "full_trajectory": full_trajectory,
            "partial_trajectory": partial,
            "annotation_present": reviewed,
            "review_complete": [
                index
                for index, segment in enumerate(plan.segments)
                if segment.review_status == "complete"
            ],
            "default_good": default_good,
        },
        "episodes": episodes,
    }
    if reused_media_from is not None:
        modes = media_publication_modes or {"immutable_hardlink_reuse"}
        payload["media_publication"] = {
            "modes": sorted(modes),
            "source_dataset": str(reused_media_from),
            "source_is_runtime_dependency": False,
        }
    path = root / "meta" / "clothdojo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _match_reusable_episodes(
    plan: ExportPlan,
    reusable_episodes: list[object],
    *,
    sources: dict[str, ReviewSource] | None = None,
) -> tuple[int, ...]:
    fields = (
        "source_trajectory",
        "action_start",
        "action_end_exclusive",
        "source_action_count",
        "annotation_sha256",
    )
    indexed: dict[tuple[Any, ...], tuple[int, dict[str, Any]]] = {}
    for raw in reusable_episodes:
        if not isinstance(raw, dict):
            raise RuntimeError("reusable episode provenance must contain objects")
        episode_index = raw.get("episode_index")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise RuntimeError("reusable episode index is invalid")
        key = tuple(raw.get(field) for field in fields)
        if key in indexed:
            raise RuntimeError("reusable episode provenance is ambiguous")
        indexed[key] = episode_index, raw
    mapping = []
    for new_index, segment in enumerate(plan.segments):
        record = segment.to_json_dict(episode_index=new_index)
        key = tuple(record.get(field) for field in fields)
        if key not in indexed:
            raise RuntimeError(
                "selected segment is not an exact member of the reusable export: "
                f"{segment.source_trajectory}[{segment.action_start}:"
                f"{segment.action_end_exclusive}]"
            )
        old_index, reusable_record = indexed[key]
        if sources is not None:
            source = sources[segment.source_trajectory]
            expected_review_identity = {
                "review_export": source.root.name,
                "review_info_sha256": source.info_sha256,
                "review_provenance_sha256": source.provenance_sha256,
            }
            for field, expected in expected_review_identity.items():
                # Exports created before review hashes were added can still be
                # reused once. New exports make every available identity field
                # mandatory on subsequent republication.
                actual = reusable_record.get(field)
                if actual is not None and actual != expected:
                    raise RuntimeError(
                        "reusable review provenance differs for "
                        f"{segment.source_trajectory}: {field}={actual!r} != {expected!r}"
                    )
        mapping.append(old_index)
    return tuple(mapping)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON metadata: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON metadata must be an object: {path}")
    return value


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSONL metadata: {path}: {exc}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeError(f"JSONL metadata must contain objects: {path}")
    return values


def _rows_by_episode_index(
    rows: list[dict[str, Any]], description: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        episode_index = row.get("episode_index")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
            or episode_index in indexed
        ):
            raise RuntimeError(f"reusable {description} has invalid episode indices")
        indexed[episode_index] = row
    return indexed


def _v21_path(
    template: str,
    episode_index: int,
    *,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    try:
        value = template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            video_key=video_key,
        )
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"unsupported LeRobot v2.1 path template: {template}") from exc
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"LeRobot path escapes the dataset: {path}")
    return path


def _require_constant_column(table: Any, name: str, value: int) -> None:
    if name not in table.column_names or any(
        current.as_py() != value for current in table[name]
    ):
        raise RuntimeError(f"reusable Parquet {name} column is invalid")


def _require_range_column(table: Any, name: str, start: int, length: int) -> None:
    if name not in table.column_names or any(
        current.as_py() != expected
        for current, expected in zip(table[name], range(start, start + length), strict=True)
    ):
        raise RuntimeError(f"reusable Parquet {name} column is invalid")


def _reindex_episode_stats(
    raw: dict[str, Any], *, episode_index: int, global_start: int, length: int
) -> dict[str, Any]:
    # JSON round-tripping makes a deep copy without importing a second object
    # model for the LeRobot metadata schema.
    row = json.loads(json.dumps(raw))
    stats = row.get("stats")
    if not isinstance(stats, dict):
        raise RuntimeError("reusable episode statistics are missing")
    row["episode_index"] = episode_index
    count = [length]
    stats["episode_index"] = {
        "min": [episode_index],
        "max": [episode_index],
        "mean": [float(episode_index)],
        "std": [0.0],
        "count": count,
    }
    stats["index"] = {
        "min": [global_start],
        "max": [global_start + length - 1],
        "mean": [global_start + (length - 1) / 2],
        "std": [((length**2 - 1) / 12) ** 0.5],
        "count": count,
    }
    return row


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")


def _write_json_lines(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _validate_republished_output(
    root: Path,
    plan: ExportPlan,
    *,
    repo_id: str,
    video_keys: tuple[str, ...],
    reusable_root: Path,
    mapping: tuple[int, ...],
    chunks_size: int,
    video_template: str,
    media_publication_modes: set[str],
) -> None:
    """Validate rebuilt indices and every reused or copied MP4 publication."""

    info = _read_json_object(root / "meta" / "info.json")
    expected = {
        "codebase_version": "v2.1",
        "total_episodes": plan.output_episodes,
        "total_frames": plan.output_frames,
    }
    if {key: info.get(key) for key in expected} != expected:
        raise RuntimeError("republished LeRobot counts differ from the export plan")
    for new_index, old_index in enumerate(mapping):
        for key in video_keys:
            old_path = reusable_root / _v21_path(
                video_template,
                old_index,
                chunks_size=chunks_size,
                video_key=key,
            )
            new_path = root / _v21_path(
                video_template,
                new_index,
                chunks_size=chunks_size,
                video_key=key,
            )
            old_stat = old_path.stat()
            new_stat = new_path.stat()
            if new_stat.st_size != old_stat.st_size:
                raise RuntimeError(f"republished video size differs: {new_path}")
            if "immutable_copy_reuse" not in media_publication_modes and (
                old_stat.st_dev,
                old_stat.st_ino,
            ) != (new_stat.st_dev, new_stat.st_ino):
                raise RuntimeError(f"republished video is not an exact hard link: {new_path}")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)
    if len(dataset) != plan.output_frames:
        raise RuntimeError("pinned LeRobot could not reload every republished frame")


def _publish_immutable_video(source: Path, target: Path) -> str:
    """Reuse one validated MP4, copying only when the filesystem forbids links."""

    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EXDEV, errno.EOPNOTSUPP}:
            raise
        shutil.copy2(source, target)
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"copied video size differs: {target}") from exc
        return "immutable_copy_reuse"
    return "immutable_hardlink_reuse"


def _source_assets(source_root: Path, source_counts: tuple[tuple[str, int], ...]) -> set[str]:
    """Collect explicit asset identities while the canonical source is available."""

    assets = set()
    for source_id, _ in source_counts:
        path = source_root / "trajectories" / source_id / "manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read source trajectory manifest: {path}: {exc}") from exc
        metadata = manifest.get("metadata")
        asset_id = metadata.get("asset_id") if isinstance(metadata, dict) else None
        if isinstance(asset_id, str) and asset_id:
            assets.add(asset_id)
    return assets


def _validate_v21_output(
    root: Path,
    plan: ExportPlan,
    *,
    repo_id: str,
    video_keys: tuple[str, ...],
) -> None:
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read exported LeRobot info: {info_path}: {exc}") from exc
    expected = {
        "codebase_version": "v2.1",
        "total_episodes": plan.output_episodes,
        "total_frames": plan.output_frames,
    }
    actual = {key: info.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"exported LeRobot counts differ: {actual} != {expected}")
    episodes_path = root / "meta" / "episodes.jsonl"
    try:
        episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read exported episode metadata: {episodes_path}: {exc}"
        ) from exc
    lengths = [episode.get("length") for episode in episodes]
    expected_lengths = [segment.length for segment in plan.segments]
    if lengths != expected_lengths:
        raise RuntimeError("exported LeRobot episode lengths differ from the good-segment plan")
    parquet_files = list((root / "data").rglob("*.parquet"))
    if len(parquet_files) != plan.output_episodes:
        raise RuntimeError("exported LeRobot data file count differs from the episode plan")
    for key in video_keys:
        video_files = sorted((root / "videos").rglob(f"{key}/episode_*.mp4"))
        if len(video_files) != plan.output_episodes or any(
            path.stat().st_size == 0 for path in video_files
        ):
            raise RuntimeError(f"exported LeRobot videos are incomplete: {key}")
        for path, expected_frames in zip(video_files, expected_lengths, strict=True):
            _validate_video(path, expected_frames)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)
    if len(dataset) != plan.output_frames:
        raise RuntimeError("pinned LeRobot could not reload every exported frame")


def _validate_video(path: Path, expected_frames: int) -> None:
    import av

    try:
        with av.open(str(path)) as container:
            actual_frames = sum(1 for _ in container.decode(video=0))
    except (av.error.FFmpegError, OSError) as exc:
        raise RuntimeError(f"cannot decode exported video: {path}: {exc}") from exc
    if actual_frames != expected_frames:
        raise RuntimeError(
            f"exported video frame count differs: {path}: "
            f"{actual_frames} != {expected_frames}"
        )


def _validate_episode_sets(sets: dict[str, Any], total: int, path: Path) -> None:
    required = {
        "good",
        "full_trajectory",
        "partial_trajectory",
        "annotation_present",
        "review_complete",
        "default_good",
    }
    if not required.issubset(sets):
        raise RuntimeError(f"dataset episode sets are incomplete: {path}")
    for name in required:
        values = sets[name]
        if (
            not isinstance(values, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
            or values != sorted(set(values))
            or any(value < 0 or value >= total for value in values)
        ):
            raise RuntimeError(f"dataset episode set is invalid: {path}: {name}")
    good = set(sets["good"])
    full = set(sets["full_trajectory"])
    partial = set(sets["partial_trajectory"])
    if good != set(range(total)) or full & partial or full | partial != good:
        raise RuntimeError(f"dataset good/full/partial sets are inconsistent: {path}")
    for name in {"annotation_present", "review_complete", "default_good"}:
        if not set(sets[name]).issubset(good):
            raise RuntimeError(f"dataset episode set is outside good: {path}: {name}")
