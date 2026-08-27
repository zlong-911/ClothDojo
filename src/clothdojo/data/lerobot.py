"""Shared helpers for ordinary LeRobot v2.1 datasets."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType
from typing import Any, Protocol

EXPORT_SCHEMA = "clothdojo_lerobot_export"
EXPORT_FORMAT_VERSION = 1
VIDEO_CODEC = "h264"
VIDEO_CRF = 23
VIDEO_ENCODER_WORKERS = 4


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


class _Writer(Protocol):
    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self) -> None: ...


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
