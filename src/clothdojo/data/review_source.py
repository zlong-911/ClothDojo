"""Read immutable single-episode LeRobot v3 review exports.

ClothDojo targets the LeRobot v2.1 revision pinned by RLINF. The source review
cache was produced by LeRobot v3, so this adapter deliberately reads only the
small v3 surface needed for the one-time migration: JSON metadata, Parquet
policy data, and MP4 frames.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_GENERATED_FEATURES = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)


@dataclass(frozen=True, slots=True)
class ReviewSource:
    source_trajectory: str
    root: Path
    frame_count: int
    fps: int
    robot_type: str
    features: dict[str, dict[str, Any]]
    video_keys: tuple[str, ...]
    info_sha256: str
    provenance_sha256: str


def discover_review_sources(
    dataset_root: str | Path,
    *,
    require_payload: bool = True,
) -> dict[str, ReviewSource]:
    """Index and validate all single-trajectory review exports.

    Metadata-only indexes used to republish an existing export may omit the
    large immutable Parquet and MP4 payloads. Fresh exports retain the strict
    default and require every payload file before any decoding begins.
    """

    reviews_root = Path(dataset_root) / "reviews"
    if not reviews_root.is_dir():
        raise NotADirectoryError(f"review export directory is missing: {reviews_root}")
    sources: dict[str, ReviewSource] = {}
    for root in sorted(reviews_root.glob("review-*")):
        if not root.is_dir():
            continue
        info_path = root / "meta" / "info.json"
        provenance_path = root / "meta" / "newton_cloth_rl.json"
        if not info_path.is_file() or not provenance_path.is_file():
            raise RuntimeError(f"review export metadata is incomplete: {root}")
        info = _read_object(info_path)
        provenance = _read_object(provenance_path)
        if info.get("codebase_version") != "v3.0":
            raise RuntimeError(f"review source must use LeRobot v3.0: {root}")
        if info.get("total_episodes") != 1:
            raise RuntimeError(f"review source must contain one episode: {root}")
        frame_count = _positive_integer(info.get("total_frames"), "total_frames", root)
        fps = _positive_integer(info.get("fps"), "fps", root)
        robot_type = info.get("robot_type")
        if not isinstance(robot_type, str) or not robot_type:
            raise RuntimeError(f"review source has no robot_type: {root}")
        features = info.get("features")
        if not isinstance(features, dict):
            raise RuntimeError(f"review source features must be an object: {root}")
        raw_output_features = {
            key: value for key, value in features.items() if key not in _GENERATED_FEATURES
        }
        if not raw_output_features or any(
            not isinstance(value, dict) for value in raw_output_features.values()
        ):
            raise RuntimeError(f"review source feature schema is invalid: {root}")
        output_features = {
            key: {
                "dtype": value.get("dtype"),
                "shape": value.get("shape"),
                "names": value.get("names"),
            }
            for key, value in raw_output_features.items()
        }
        video_keys = tuple(
            key for key, value in output_features.items() if value.get("dtype") == "video"
        )
        if not video_keys:
            raise RuntimeError(f"review source contains no video features: {root}")

        source_records = provenance.get("source", {}).get("episodes")
        if not isinstance(source_records, list) or len(source_records) != 1:
            raise RuntimeError(f"review provenance must identify one source: {root}")
        source_id = source_records[0].get("episode_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError(f"review provenance has no source episode id: {root}")
        annotation = provenance.get("annotation")
        if not isinstance(annotation, dict) or annotation.get("applied") is not False:
            raise RuntimeError(f"review source must be unfiltered: {root}")
        source_segments = annotation.get("segments")
        expected_segment = {
            "source_episode": source_id,
            "action_start": 0,
            "action_end_exclusive": frame_count,
            "length": frame_count,
            "output_episode_index": 0,
        }
        if not isinstance(source_segments, list) or len(source_segments) != 1:
            raise RuntimeError(f"review source must contain one identity segment: {root}")
        if any(source_segments[0].get(key) != value for key, value in expected_segment.items()):
            raise RuntimeError(f"review source segment is not an identity mapping: {root}")
        if source_id in sources:
            raise RuntimeError(f"duplicate review source for trajectory {source_id}")

        source = ReviewSource(
            source_trajectory=source_id,
            root=root,
            frame_count=frame_count,
            fps=fps,
            robot_type=robot_type,
            features=output_features,
            video_keys=video_keys,
            info_sha256=_sha256(info_path),
            provenance_sha256=_sha256(provenance_path),
        )
        if require_payload:
            _require_source_files(source, info)
        sources[source_id] = source
    if not sources:
        raise RuntimeError(f"no review exports found in {reviews_root}")
    return sources


class ReviewFrameReader:
    """Stream every synchronized frame from one v3 review source."""

    def __init__(self, source: ReviewSource):
        self.source = source

    def __enter__(self) -> ReviewFrameReader:
        try:
            import av
            import pyarrow.parquet as parquet
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyArrow and PyAV are required for LeRobot export") from exc

        info = _read_object(self.source.root / "meta" / "info.json")
        data_path = _format_source_path(info, "data_path")
        table = parquet.read_table(self.source.root / data_path)
        if table.num_rows != self.source.frame_count:
            raise RuntimeError(
                f"review Parquet frame count differs: {self.source.root}: "
                f"{table.num_rows} != {self.source.frame_count}"
            )
        rows = table.to_pylist()
        _validate_generated_columns(rows, self.source)
        episode_path = _episode_metadata_path(self.source)
        episode_rows = parquet.read_table(episode_path).to_pylist()
        _validate_episode_metadata(episode_rows, self.source)
        tasks_table = parquet.read_table(self.source.root / "meta" / "tasks.parquet")
        tasks = {
            int(row["task_index"]): row["task"] for row in tasks_table.to_pylist()
        }
        for row in rows:
            task_index = row.get("task_index")
            if task_index not in tasks:
                raise RuntimeError(f"review frame refers to an unknown task: {self.source.root}")
            row["task"] = tasks[task_index]
        self._rows = rows
        self._stack = ExitStack()
        self._video_iterators = {}
        for key in self.source.video_keys:
            relative = _format_source_path(info, "video_path", video_key=key)
            container = self._stack.enter_context(av.open(str(self.source.root / relative)))
            self._video_iterators[key] = iter(container.decode(video=0))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stack.close()

    def __iter__(self):
        for index, raw in enumerate(self._rows):
            frame: dict[str, Any] = {}
            for key, feature in self.source.features.items():
                if key in self.source.video_keys:
                    try:
                        video_frame = next(self._video_iterators[key])
                    except StopIteration as exc:
                        raise RuntimeError(
                            f"review video ended before frame {index}: {self.source.root}: {key}"
                        ) from exc
                    image = video_frame.to_ndarray(format="rgb24")
                    expected_shape = tuple(feature.get("shape", ()))
                    if image.shape != expected_shape:
                        raise RuntimeError(
                            f"review video frame shape differs: {self.source.root}: "
                            f"{key}: {image.shape} != {expected_shape}"
                        )
                    frame[key] = image
                else:
                    if key not in raw:
                        raise RuntimeError(f"review Parquet feature is missing: {key}")
                    dtype = np.dtype(feature.get("dtype"))
                    value = np.asarray(raw[key], dtype=dtype)
                    expected_shape = tuple(feature.get("shape", ()))
                    if value.shape != expected_shape:
                        raise RuntimeError(
                            f"review feature shape differs: {self.source.root}: "
                            f"{key}: {value.shape} != {expected_shape}"
                        )
                    if not np.isfinite(value).all():
                        raise RuntimeError(
                            f"review feature is non-finite: {self.source.root}: {key}: {index}"
                        )
                    frame[key] = value
            frame["task"] = raw["task"]
            yield index, frame
        for key, iterator in self._video_iterators.items():
            try:
                next(iterator)
            except StopIteration:
                continue
            raise RuntimeError(f"review video has extra frames: {self.source.root}: {key}")


def _require_source_files(source: ReviewSource, info: dict[str, Any]) -> None:
    paths = [
        source.root / _format_source_path(info, "data_path"),
        source.root / "meta" / "tasks.parquet",
        *(
            source.root / _format_source_path(info, "video_path", video_key=key)
            for key in source.video_keys
        ),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"review export files are missing: {missing}")
    _episode_metadata_path(source)


def _episode_metadata_path(source: ReviewSource) -> Path:
    paths = sorted((source.root / "meta" / "episodes").rglob("*.parquet"))
    if len(paths) != 1:
        raise RuntimeError(
            f"review source must contain one episode metadata file: {source.root}"
        )
    return paths[0]


def _validate_generated_columns(rows: list[dict[str, Any]], source: ReviewSource) -> None:
    for index, row in enumerate(rows):
        expected = {
            "frame_index": index,
            "episode_index": 0,
            "index": index,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"review Parquet indices are not an identity timeline: "
                f"{source.root}: row={index}"
            )
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, int | float) or not np.isclose(
            timestamp, index / source.fps, atol=1e-4, rtol=0.0
        ):
            raise RuntimeError(
                f"review Parquet timestamp differs at {index}: {source.root}: {timestamp}"
            )
        task_index = row.get("task_index")
        if isinstance(task_index, bool) or not isinstance(task_index, int):
            raise RuntimeError(
                f"review Parquet task_index is invalid at {index}: {source.root}"
            )


def _validate_episode_metadata(
    rows: list[dict[str, Any]], source: ReviewSource
) -> None:
    if len(rows) != 1:
        raise RuntimeError(
            f"review episode metadata must contain one row: {source.root}"
        )
    row = rows[0]
    expected = {
        "episode_index": 0,
        "length": source.frame_count,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": 0,
        "dataset_to_index": source.frame_count,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            f"review episode metadata is not an identity mapping: {source.root}"
        )
    expected_end = source.frame_count / source.fps
    for key in source.video_keys:
        prefix = f"videos/{key}"
        video_expected = {
            f"{prefix}/chunk_index": 0,
            f"{prefix}/file_index": 0,
        }
        if any(row.get(field) != value for field, value in video_expected.items()):
            raise RuntimeError(
                f"review episode video metadata is invalid: {source.root}: {key}"
            )
        start = row.get(f"{prefix}/from_timestamp")
        end = row.get(f"{prefix}/to_timestamp")
        if (
            not isinstance(start, int | float)
            or not isinstance(end, int | float)
            or not np.isclose(start, 0.0, atol=1e-4, rtol=0.0)
            or not np.isclose(end, expected_end, atol=1e-4, rtol=0.0)
        ):
            raise RuntimeError(
                f"review episode video timestamps are invalid: {source.root}: {key}"
            )


def _format_source_path(
    info: dict[str, Any], field: str, *, video_key: str | None = None
) -> Path:
    template = info.get(field)
    if not isinstance(template, str) or not template:
        raise RuntimeError(f"review info has no {field}")
    values: dict[str, Any] = {"chunk_index": 0, "file_index": 0}
    if video_key is not None:
        values["video_key"] = video_key
    return Path(template.format(**values))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read review metadata: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"review metadata must be an object: {path}")
    return value


def _positive_integer(value: object, name: str, root: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"review {name} must be a positive integer: {root}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
