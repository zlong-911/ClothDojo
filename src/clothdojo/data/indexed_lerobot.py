"""Pinned-LeRobot reader for ClothDojo's immutable chunked-media index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def open_lerobot_dataset(repo_id: str, *, root: str | Path, **kwargs: Any) -> Any:
    """Open an ordinary v2.1 dataset or its ClothDojo indexed-media extension."""

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path(root).expanduser().resolve()
    # The pinned TorchCodec wheel is not ABI-compatible with the CUDA/PyTorch
    # environment used on the training workers. PyAV is already a pinned
    # project dependency and is the validated LeRobot backend here.
    kwargs.setdefault("video_backend", "pyav")
    sidecar_path = dataset_root / "meta" / "clothdojo.json"
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sidecar = {}
    migration = sidecar.get("migration") if isinstance(sidecar, dict) else None
    if not isinstance(migration, dict) or migration.get("kind") != (
        "immutable_chunked_media_index"
    ):
        return LeRobotDataset(repo_id, root=dataset_root, **kwargs)

    # v3 stores float32 frame timestamps plus a per-video float64 offset. Their
    # sum can differ from the decoded PTS by slightly more than LeRobot v2.1's
    # 100 µs default even when it selects the exact 50 Hz frame. Half a frame
    # is the strict nearest-frame bound and still rejects a one-frame slip.
    info = json.loads(
        (dataset_root / "meta" / "info.json").read_text(encoding="utf-8")
    )
    fps = int(info["fps"])
    if fps < 1:
        raise ValueError(f"indexed LeRobot FPS is invalid: {dataset_root}")
    kwargs.setdefault("tolerance_s", 0.5 / fps + 1e-6)

    index_name = migration.get("video_index")
    if not isinstance(index_name, str):
        raise ValueError(f"indexed LeRobot dataset has no video index: {dataset_root}")
    index_path = dataset_root / index_name
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    offsets: list[dict[str, float]] = []
    for expected, row in enumerate(rows):
        if row.get("episode_index") != expected:
            raise ValueError(f"indexed LeRobot episode order is invalid: {index_path}")
        current = row.get("timestamp_offsets")
        if not isinstance(current, dict):
            raise TypeError(f"indexed LeRobot timestamp offsets are invalid: {index_path}")
        offsets.append({str(key): float(value) for key, value in current.items()})

    class IndexedMediaLeRobotDataset(LeRobotDataset):
        def _query_videos(self, query_timestamps, ep_idx):
            try:
                episode_offsets = offsets[int(ep_idx)]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"indexed LeRobot episode is invalid: {ep_idx}") from exc
            shifted = {}
            for key, timestamps in query_timestamps.items():
                if key not in episode_offsets:
                    raise KeyError(f"indexed LeRobot camera is missing: {key}")
                offset = episode_offsets[key]
                shifted[key] = [float(timestamp) + offset for timestamp in timestamps]
            return super()._query_videos(shifted, ep_idx)

    return IndexedMediaLeRobotDataset(repo_id, root=dataset_root, **kwargs)


def replace_pair_dataset_base(pair_dataset: Any) -> None:
    """Give an upstream STEAM pair dataset the project-owned media reader."""

    source = pair_dataset.source
    root = Path(source.base.root)
    replacement = open_lerobot_dataset(
        root.name,
        root=root,
        download_videos=False,
    )
    from rlinf.data.datasets.recap.utils import decode_image_struct_batch

    replacement.hf_dataset.set_transform(decode_image_struct_batch)
    source.base = replacement
