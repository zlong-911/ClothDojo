"""Shared helpers for ordinary LeRobot v2.1 datasets."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType
from typing import Any, Protocol

EXPORT_SCHEMA = "clothdojo_lerobot_export"
EXPORT_FORMAT_VERSION = 1
VIDEO_CODEC = "h264"
VIDEO_CRF = 23
VIDEO_ENCODER = os.environ.get("CLOTHDOJO_VIDEO_ENCODER", VIDEO_CODEC)
if VIDEO_ENCODER not in {VIDEO_CODEC, "h264_nvenc"}:
    raise ValueError("CLOTHDOJO_VIDEO_ENCODER must be 'h264' or 'h264_nvenc'")
VIDEO_NVENC_CQ = int(os.environ.get("CLOTHDOJO_VIDEO_NVENC_CQ", "28"))
if not 0 <= VIDEO_NVENC_CQ <= 51:
    raise ValueError("CLOTHDOJO_VIDEO_NVENC_CQ must be in [0, 51]")
PNG_COMPRESS_LEVEL = int(os.environ.get("CLOTHDOJO_PNG_COMPRESS_LEVEL", "6"))
if not 0 <= PNG_COMPRESS_LEVEL <= 9:
    raise ValueError("CLOTHDOJO_PNG_COMPRESS_LEVEL must be in [0, 9]")
VIDEO_ENCODER_WORKERS = int(os.environ.get("CLOTHDOJO_VIDEO_ENCODER_WORKERS", "4"))
if VIDEO_ENCODER_WORKERS < 1:
    raise ValueError("CLOTHDOJO_VIDEO_ENCODER_WORKERS must be positive")


def merge_lerobot_shards(
    shards: list[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    """Merge disjoint trajectory exports without decoding or re-encoding video."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    roots = [Path(path).expanduser().resolve() for path in shards]
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("merge requires at least two distinct LeRobot shards")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"merged LeRobot output already exists: {destination}")
    staging = destination.parent / f".{destination.name}.tmp"
    if staging.exists():
        raise FileExistsError(f"unfinished merge staging exists: {staging}")

    shard_records = []
    common_info = None
    common_tasks = None
    common_sidecar = None
    shard_contract = set()
    for root in roots:
        info = _read_json_object(root / "meta" / "info.json")
        sidecar = _read_json_object(root / "meta" / "clothdojo.json")
        tasks = (root / "meta" / "tasks.jsonl").read_bytes()
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"LeRobot shard is not v2.1: {root}")
        if (
            sidecar.get("schema") != EXPORT_SCHEMA
            or sidecar.get("format_version") != EXPORT_FORMAT_VERSION
            or sidecar.get("export_status") != "complete"
            or sidecar.get("source", {}).get("kind") != "trajectory_collection"
            or sidecar.get("selection", {}).get("rule") not in {
                "explicit_whole_trajectory_good", "bad_complement_with_minimum_length"
            }
        ):
            raise ValueError(f"merge requires completed selected-trajectory exports: {root}")
        rendering = sidecar.get("rendering")
        shard = rendering.get("trajectory_shard") if isinstance(rendering, dict) else None
        if not isinstance(shard, dict):
            raise ValueError(f"trajectory shard provenance is missing: {root}")
        if any(type(shard.get(key)) is not int for key in ("index", "count")):
            raise ValueError(f"trajectory shard index/count must be integers: {root}")
        shard_contract.add((shard.get("index"), shard.get("count")))
        comparable_info = {
            key: info.get(key)
            for key in (
                "codebase_version", "robot_type", "fps", "chunks_size", "features",
                "data_path", "video_path",
            )
        }
        comparable_sidecar = {
            key: sidecar.get(key)
            for key in ("schema", "format_version", "selection", "source", "scenario")
        }
        base_rendering = dict(rendering)
        base_rendering.pop("trajectory_shard", None)
        base_rendering.pop("parallel_export", None)
        comparable_sidecar["rendering"] = base_rendering
        if common_info is None:
            common_info = comparable_info
            common_tasks = tasks
            common_sidecar = comparable_sidecar
        elif (
            comparable_info != common_info
            or tasks != common_tasks
            or comparable_sidecar != common_sidecar
        ):
            raise ValueError(f"LeRobot shard contracts differ: {root}")
        episodes = _read_json_lines(root / "meta" / "episodes.jsonl")
        stats = _read_json_lines(root / "meta" / "episodes_stats.jsonl")
        provenance = sidecar.get("episodes")
        if not isinstance(provenance, list):
            raise ValueError(f"trajectory shard episode provenance is missing: {root}")
        expected_indices = list(range(len(episodes)))
        if (
            [row.get("episode_index") for row in episodes] != expected_indices
            or [row.get("episode_index") for row in stats] != expected_indices
            or [row.get("episode_index") for row in provenance] != expected_indices
        ):
            raise ValueError(f"trajectory shard episode indices are not contiguous: {root}")
        if info.get("total_episodes") != len(episodes) or info.get("total_frames") != sum(
            row["length"] for row in episodes
        ):
            raise ValueError(f"trajectory shard counts differ from episode metadata: {root}")
        shard_records.append((root, info, episodes, stats, provenance))

    shard_count = len(roots)
    if shard_contract != {(index, shard_count) for index in range(shard_count)}:
        raise ValueError(f"trajectory shard set is incomplete: {sorted(shard_contract)}")
    assert common_info is not None and common_sidecar is not None and common_tasks is not None

    source = common_sidecar["source"]
    selection = common_sidecar["selection"]
    source_root = Path(source["trajectory"])
    from clothdojo.data.annotations import build_good_segment_plan
    from clothdojo.data.rollout import CAMERA_NAMES, _write_selected_sidecar

    plan = build_good_segment_plan(
        source_root,
        min_segment_frames=int(selection["min_segment_frames"]),
        whole_trajectory_good_only=(selection["rule"] == "explicit_whole_trajectory_good"),
    )
    locations = {}
    for record_index, (_root, _info, episodes, stats, provenance) in enumerate(shard_records):
        for local_index, record in enumerate(provenance):
            key = (
                record.get("source_trajectory"),
                record.get("action_start"),
                record.get("action_end_exclusive"),
            )
            if key in locations:
                raise ValueError(f"trajectory segment appears in multiple shards: {key}")
            locations[key] = (record_index, local_index, episodes[local_index], stats[local_index])
    planned_segments = {
        (segment.source_trajectory, segment.action_start, segment.action_end_exclusive): segment
        for segment in plan.segments
    }
    ordered_keys = list(planned_segments)
    if set(locations) != set(ordered_keys):
        missing = sorted(set(ordered_keys) - set(locations))
        extra = sorted(set(locations) - set(ordered_keys))
        raise ValueError(f"trajectory shard coverage differs: missing={missing}, extra={extra}")
    for key, (record_index, local_index, episode, _stats) in locations.items():
        provenance = shard_records[record_index][4][local_index]
        segment = planned_segments[key]
        if episode["length"] != segment.length or any(
            provenance.get(field) != value
            for field, value in segment.to_range_json_dict().items()
        ):
            raise ValueError(f"trajectory shard selection provenance differs: {key}")

    merged_episodes = []
    merged_stats = []
    global_frame_offset = 0
    chunks_size = int(common_info["chunks_size"])
    # Acquire ownership before enabling cleanup; preflight may overlap another merge.
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "meta").mkdir()
        for global_episode, key in enumerate(ordered_keys):
            record_index, local_episode, episode, stats = locations[key]
            root, info, _episodes, _stats, _provenance = shard_records[record_index]
            length = int(episode["length"])
            local_chunk = local_episode // int(info["chunks_size"])
            global_chunk = global_episode // chunks_size
            source_data = root / info["data_path"].format(
                episode_chunk=local_chunk,
                episode_index=local_episode,
            )
            target_data = staging / info["data_path"].format(
                episode_chunk=global_chunk,
                episode_index=global_episode,
            )
            table = pq.read_table(source_data)
            if table.num_rows != length:
                raise RuntimeError(f"LeRobot shard parquet length differs: {source_data}")
            for column, values in {
                "episode_index": pa.array([global_episode] * length, type=pa.int64()),
                "index": pa.array(
                    range(global_frame_offset, global_frame_offset + length),
                    type=pa.int64(),
                ),
            }.items():
                position = table.column_names.index(column)
                table = table.set_column(position, column, values)
            target_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target_data)

            for video_key in common_info["features"]:
                if common_info["features"][video_key].get("dtype") != "video":
                    continue
                source_video = root / info["video_path"].format(
                    episode_chunk=local_chunk,
                    video_key=video_key,
                    episode_index=local_episode,
                )
                target_video = staging / info["video_path"].format(
                    episode_chunk=global_chunk,
                    video_key=video_key,
                    episode_index=global_episode,
                )
                target_video.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_video, target_video)
                except OSError:
                    shutil.copy2(source_video, target_video)

            merged_episode = dict(episode)
            merged_episode["episode_index"] = global_episode
            merged_episodes.append(merged_episode)
            merged_stat = json.loads(json.dumps(stats))
            merged_stat["episode_index"] = global_episode
            _shift_episode_stats(
                merged_stat,
                episode_index=global_episode,
                frame_offset=global_frame_offset,
            )
            merged_stats.append(merged_stat)
            global_frame_offset += length

        info = _read_json_object(roots[0] / "meta" / "info.json")
        info.update(
            {
                "total_episodes": len(merged_episodes),
                "total_frames": global_frame_offset,
                "total_videos": len(merged_episodes) * len(CAMERA_NAMES),
                "total_chunks": math.ceil(len(merged_episodes) / chunks_size),
                "splits": {"train": f"0:{len(merged_episodes)}"},
            }
        )
        _write_json(staging / "meta" / "info.json", info)
        (staging / "meta" / "tasks.jsonl").write_bytes(common_tasks)
        _write_json_lines(staging / "meta" / "episodes.jsonl", merged_episodes)
        _write_json_lines(staging / "meta" / "episodes_stats.jsonl", merged_stats)
        _write_selected_sidecar(
            staging,
            source_root=source_root,
            plan=plan,
            scenario=common_sidecar["scenario"],
            rendering={
                **common_sidecar["rendering"],
                "parallel_export": {"shards": shard_count},
            },
        )

        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        repo_id = f"local/{destination.name}"
        dataset = LeRobotDataset(repo_id=repo_id, root=staging, download_videos=False)
        if len(dataset) != global_frame_offset:
            raise RuntimeError("merged LeRobot dataset cannot be reloaded")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output": str(destination),
        "shards": shard_count,
        "episodes": len(merged_episodes),
        "frames": global_frame_offset,
        "videos": len(merged_episodes) * len(CAMERA_NAMES),
    }


def _shift_episode_stats(record: dict[str, Any], *, episode_index: int, frame_offset: int) -> None:
    stats = record.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("LeRobot episode stats are missing")
    for name, offset in (("episode_index", episode_index), ("index", frame_offset)):
        values = stats.get(name)
        if not isinstance(values, dict):
            raise ValueError(f"LeRobot episode stats are missing: {name}")
        count = values.get("count")
        if name == "episode_index":
            values.update(
                {
                    "min": [episode_index],
                    "max": [episode_index],
                    "mean": [float(episode_index)],
                    "std": [0.0],
                    "count": count,
                }
            )
        else:
            sample_count = int(count[0])
            values.update(
                {
                    "min": [offset],
                    "max": [offset + sample_count - 1],
                    "mean": [offset + 0.5 * (sample_count - 1)],
                    "count": count,
                }
            )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON value must be an object: {path}")
    return value


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON lines: {path}: {exc}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"JSON line must be an object: {path}")
    return values


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_lines(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )


def open_lerobot_dataset(repo_id: str, *, root: str | Path, **kwargs: Any) -> Any:
    """Open one ordinary LeRobot v2.1 dataset with the validated video backend."""

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path(root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read LeRobot dataset metadata: {info_path}: {exc}") from exc
    if not isinstance(info, dict) or info.get("codebase_version") != "v2.1":
        raise ValueError(f"ClothDojo requires standard LeRobot v2.1: {info_path}")
    if (dataset_root / "meta" / "clothdojo_video_index.jsonl").is_file():
        raise ValueError(
            "indexed-media datasets must be converted externally to ordinary LeRobot v2.1"
        )
    kwargs["video_backend"] = "pyav"
    return LeRobotDataset(repo_id, root=dataset_root, **kwargs)


def load_single_task_prompt(dataset_root: str | Path) -> str:
    """Read the one task prompt owned by a standard LeRobot v2.1 dataset."""

    meta = Path(dataset_root).expanduser().resolve() / "meta"
    parquet = meta / "tasks.parquet"
    jsonl = meta / "tasks.jsonl"
    if parquet.is_file():
        import pyarrow.parquet as pq

        values = pq.read_table(parquet, columns=["task"])["task"].to_pylist()
        source = parquet
    else:
        try:
            rows = [
                json.loads(line)
                for line in jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read LeRobot task metadata: {jsonl}: {exc}") from exc
        values = [
            row.get("task")
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("task"), str)
        ]
        source = jsonl
    prompts = {str(value).strip() for value in values if str(value).strip()}
    if len(prompts) != 1:
        raise ValueError(f"LeRobot dataset must contain one non-empty task prompt: {source}")
    return prompts.pop()


class _Writer(Protocol):
    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self) -> None: ...

    def stop_image_writer(self) -> None: ...


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
    episodes = payload.get("episodes")
    counts = payload.get("counts")
    if not isinstance(sets, dict) or not isinstance(episodes, list) or not isinstance(counts, dict):
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
    return sets[
        {"all": "good", "full": "full_trajectory", "partial": "partial_trajectory"}[
            trajectory_extent
        ]
    ]


def _create_writer(**kwargs: Any) -> _Writer:
    import av
    import lerobot.common.datasets.image_writer as image_writer
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    av.logging.set_level(av.logging.ERROR)
    if PNG_COMPRESS_LEVEL != 6:
        image_writer.write_image = _write_temporary_png
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
    writer.encode_episode_videos = MethodType(_encode_episode_videos, writer)
    return writer


def _write_temporary_png(image: Any, path: Path) -> None:
    """Write LeRobot's transient frame with an explicitly selected PNG compression level."""

    import numpy as np
    from lerobot.common.datasets.image_writer import image_array_to_pil_image
    from PIL import Image

    # LeRobot's worker calls queue.task_done only after this callback returns.
    # Preserve its non-raising callback contract so a failed PNG cannot deadlock
    # save_episode; the encoder checks against the recorded frame count below.
    try:
        if isinstance(image, np.ndarray):
            image = image_array_to_pil_image(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"unsupported temporary image type: {type(image)}")
        image.save(path, compress_level=PNG_COMPRESS_LEVEL)
    except Exception:
        logging.exception("cannot write temporary video frame: %s", path)


def _encode_episode_videos(writer: Any, episode_index: int) -> dict[str, str]:
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
    expected_frames = len(writer.episode_buffer["frame_index"])

    def encode(key: str) -> None:
        video_path = video_paths[key]
        if video_path.is_file():
            return
        if VIDEO_ENCODER == "h264_nvenc":
            _encode_nvenc_video_frames(image_dirs[key], video_path, writer.fps)
        else:
            from lerobot.common.datasets.video_utils import encode_video_frames

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
    for path in video_paths.values():
        _validate_video(path, expected_frames)
    return {key: str(path) for key, path in video_paths.items()}


def _encode_nvenc_video_frames(image_dir: Path, video_path: Path, fps: int) -> None:
    """Encode one ordinary H.264 stream with the GPU while preserving LeRobot seekability."""

    import av
    from PIL import Image

    inputs = sorted(image_dir.glob("frame_*.png"))
    if not inputs:
        raise FileNotFoundError(f"no video frames found: {image_dir}")
    with Image.open(inputs[0]) as first:
        width, height = first.size
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path), "w") as output:
        stream = output.add_stream(
            "h264_nvenc",
            fps,
            options={"preset": "p4", "tune": "hq", "rc": "vbr", "cq": str(VIDEO_NVENC_CQ)},
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 2
        stream.codec_context.max_b_frames = 0
        for path in inputs:
            with Image.open(path) as image:
                frame = av.VideoFrame.from_image(image.convert("RGB"))
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _writer_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized = {}
    for key, feature in features.items():
        current = dict(feature)
        shape = current.get("shape")
        if isinstance(shape, list):
            current["shape"] = tuple(shape)
        normalized[key] = current
    return normalized


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
