"""OpenPI data preparation for the RLINF ALOHA baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_norm_stats(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    config_name: str = "pi05_aloha_robotwin",
    block_size: int = 4096,
) -> dict[str, Any]:
    """Compute exact state/action stats with the same transforms as training."""

    if isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be positive")
    dataset = Path(dataset_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    info_path = dataset / "meta/info.json"
    sidecar_path = dataset / "meta/clothdojo.json"
    if not info_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            "normalization requires a completed ClothDojo LeRobot export with "
            f"meta/info.json and meta/clothdojo.json: {dataset}"
        )
    if output.exists():
        raise FileExistsError(f"normalization output already exists: {output}")

    import openpi.shared.normalize as normalize
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    data_files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"LeRobot parquet files are missing: {dataset / 'data'}")
    table = pq.read_table(
        data_files,
        columns=["observation.state", "action", "episode_index"],
    )
    states = _float_matrix(table["observation.state"], pa)
    actions = _float_matrix(table["action"], pa)
    episode_indices = np.asarray(
        table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if states.shape != actions.shape or states.shape[1] != 14:
        raise ValueError(
            f"ALOHA state/action matrices must have matching 14D shape: "
            f"{states.shape} != {actions.shape}"
        )
    episode_ends, episode_count = _episode_ends(episode_indices)

    data_kwargs = {
        "adapt_to_pi": False,
        "extra_delta_transform": True,
        "default_prompt": "Flatten the shirt.",
    }
    config = get_openpi_config(
        config_name,
        repo_id=str(dataset),
        data_kwargs=data_kwargs,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    horizon = int(config.model.action_horizon)
    action_offsets = np.arange(horizon, dtype=np.int64)
    dummy_images = {"cam_high": np.zeros((3, 1, 1), dtype=np.uint8)}
    running = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    for start in range(0, len(states), block_size):
        end = min(start + block_size, len(states))
        action_indices = np.minimum(
            np.arange(start, end, dtype=np.int64)[:, None] + action_offsets,
            episode_ends[start:end, None] - 1,
        )
        transformed: dict[str, Any] = {
            "state": states[start:end],
            "actions": actions[action_indices],
            "images": dummy_images,
        }
        for transform in data_config.data_transforms.inputs:
            transformed = transform(transformed)
        for key in running:
            running[key].update(np.asarray(transformed[key]))

    stats = {key: value.get_statistics() for key, value in running.items()}
    output.mkdir(parents=True)
    normalize.save(output, stats)
    provenance = {
        "schema": "clothdojo_openpi_norm_stats",
        "format_version": 1,
        "dataset_root": str(dataset),
        "dataset_info_sha256": _sha256(info_path),
        "dataset_sidecar_sha256": _sha256(sidecar_path),
        "config_name": config_name,
        "data_kwargs": data_kwargs,
        "action_horizon": horizon,
        "frames": len(states),
        "episodes": episode_count,
    }
    (output / "clothdojo.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**provenance, "output_root": str(output)}


def _float_matrix(column: Any, pa: Any) -> np.ndarray:
    array = column.combine_chunks()
    if isinstance(array, pa.FixedSizeListArray):
        width = array.type.list_size
    elif isinstance(array, (pa.ListArray, pa.LargeListArray)):
        offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False))
        widths = np.diff(offsets)
        if not len(widths) or np.any(widths != widths[0]):
            raise ValueError("state/action list columns must have a fixed width")
        width = int(widths[0])
    else:
        raise TypeError(f"state/action column must be a list array, got {array.type}")
    values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return np.ascontiguousarray(values.reshape(len(array), width))


def _episode_ends(episode_indices: np.ndarray) -> tuple[np.ndarray, int]:
    if episode_indices.ndim != 1 or not len(episode_indices):
        raise ValueError("episode_index must be a non-empty vector")
    boundaries = np.flatnonzero(episode_indices[1:] != episode_indices[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_indices)]))
    labels = episode_indices[starts]
    if not np.array_equal(labels, np.arange(len(labels), dtype=labels.dtype)):
        raise ValueError("episode_index values must be contiguous and zero-based")
    return np.repeat(ends, ends - starts), len(labels)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
