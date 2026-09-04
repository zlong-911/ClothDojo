"""OpenPI data preparation for the RLINF ALOHA baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OPENPI_CONFIG = "pi05_aloha_robotwin"
ACTION_CHUNK_BOUNDARIES = frozenset({"repeat_last", "full_only"})


@dataclass(frozen=True)
class TrainingPrompt:
    """Apply the configured single-task text before OpenPI tokenization."""

    prompt: str

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("OpenPI training default_prompt must be a non-empty string")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "prompt": np.asarray(self.prompt)}


def compute_norm_stats(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    block_size: int = 4096,
    action_chunk_boundary: str = "repeat_last",
) -> dict[str, Any]:
    """Compute exact state/action stats with the same transforms as training."""

    if isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be positive")
    action_chunk_boundary = validate_action_chunk_boundary(action_chunk_boundary)
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
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.training.config import (
        AssetsConfig,
        DataConfig,
        LeRobotAlohaDataConfig,
    )

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

    prompt = _dataset_prompt(sidecar_path)
    data_contract = {
        "adapt_to_pi": False,
        "extra_delta_transform": True,
        "default_prompt": prompt,
    }
    model_config = pi0_config.Pi0Config(pi05=True, discrete_state_input=True)
    data_factory = LeRobotAlohaDataConfig(
        repo_id=str(dataset),
        assets=AssetsConfig(assets_dir=str(output.parent), asset_id=output.name),
        base_config=DataConfig(prompt_from_task=True),
        repack_transforms=transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
        default_prompt=prompt,
        adapt_to_pi=False,
        use_delta_joint_actions=True,
        action_sequence_keys=("action",),
    )
    data_config = data_factory.create(output.parent, model_config)
    horizon = int(model_config.action_horizon)
    action_offsets = np.arange(horizon, dtype=np.int64)
    if action_chunk_boundary == "full_only":
        anchor_indices = np.flatnonzero(
            np.arange(len(states), dtype=np.int64) + horizon <= episode_ends
        )
    else:
        anchor_indices = np.arange(len(states), dtype=np.int64)
    if not len(anchor_indices):
        raise ValueError(
            f"dataset has no {action_chunk_boundary} anchors for action horizon {horizon}"
        )
    dummy_images = {"cam_high": np.zeros((3, 1, 1), dtype=np.uint8)}
    running = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    for start in range(0, len(anchor_indices), block_size):
        anchors = anchor_indices[start : start + block_size]
        action_indices = anchors[:, None] + action_offsets
        if action_chunk_boundary == "repeat_last":
            action_indices = np.minimum(
                action_indices,
                episode_ends[anchors, None] - 1,
            )
        transformed: dict[str, Any] = {
            "state": states[anchors],
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
        "config_name": OPENPI_CONFIG,
        "data_kwargs": data_contract,
        "action_horizon": horizon,
        "action_chunk_boundary": action_chunk_boundary,
        "sampled_anchors": len(anchor_indices),
        "frames": len(states),
        "episodes": episode_count,
    }
    (output / "clothdojo.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**provenance, "output_root": str(output)}


def validate_action_chunk_boundary(value: Any) -> str:
    """Validate the supervision policy used at LeRobot episode boundaries."""

    if not isinstance(value, str) or value not in ACTION_CHUNK_BOUNDARIES:
        raise ValueError(
            "action_chunk_boundary must be one of "
            f"{sorted(ACTION_CHUNK_BOUNDARIES)}, got {value!r}"
        )
    return value


def _dataset_prompt(sidecar_path: Path) -> str:
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read ClothDojo dataset metadata: {sidecar_path}: {exc}"
        ) from exc
    scenario = sidecar.get("scenario") if isinstance(sidecar, dict) else None
    prompt = scenario.get("prompt") if isinstance(scenario, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"dataset sidecar has no task prompt: {sidecar_path}")
    return prompt.strip()


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
