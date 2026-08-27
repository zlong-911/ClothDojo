"""Owned offline STEAM advantage labeling for ClothDojo CFG datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.training.episode_selection import (
    read_episode_selection,
    validate_episode_selection,
)

_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class SteamAdvantageDataset:
    role: str
    dataset_path: Path
    positive_fraction: float
    episodes: tuple[int, ...]
    total_episodes: int
    total_frames: int
    output_path: Path

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("dataset_path", "output_path"):
            value[key] = str(value[key])
        value["episodes"] = list(self.episodes)
        return value


@dataclass(frozen=True, slots=True)
class SteamAdvantageSpec:
    config_path: Path
    checkpoint: Path
    tag: str
    prompt: str
    stride: int
    camera_keys: tuple[str, ...]
    batch_size: int
    num_workers: int
    num_processes: int
    master_port: int
    work_dir: Path
    datasets: tuple[SteamAdvantageDataset, ...]

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("config_path", "checkpoint", "work_dir"):
            value[key] = str(value[key])
        value["camera_keys"] = list(self.camera_keys)
        value["datasets"] = [dataset.to_json_dict() for dataset in self.datasets]
        return value


def compose_steam_advantage_spec(config_path: str | Path) -> SteamAdvantageSpec:
    """Resolve and validate one project-owned STEAM labeling config."""

    from omegaconf import OmegaConf

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STEAM advantage config does not exist: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("STEAM advantage config must be a mapping")
    if raw.get("schema") != "clothdojo_steam_advantages" or raw.get("format_version") != 1:
        raise ValueError("unsupported STEAM advantage config schema")

    checkpoint = Path(str(raw.get("checkpoint", ""))).expanduser().resolve()
    _checkpoint_weights(checkpoint)
    tag = str(raw.get("tag", ""))
    if not _TAG.fullmatch(tag):
        raise ValueError(f"invalid STEAM advantage tag: {tag!r}")
    stride = _positive_int(raw.get("stride"), "stride")
    camera_keys = tuple(str(value) for value in raw.get("camera_keys", ()))
    if not camera_keys or len(set(camera_keys)) != len(camera_keys):
        raise ValueError("STEAM camera_keys must be non-empty and unique")
    batch_size = _positive_int(raw.get("batch_size"), "batch_size")
    num_workers = _nonnegative_int(raw.get("num_workers"), "num_workers")
    num_processes = _positive_int(raw.get("num_processes"), "num_processes")
    master_port = _positive_int(raw.get("master_port"), "master_port")
    if master_port > 65535:
        raise ValueError("master_port must be <= 65535")
    work_dir = Path(str(raw.get("work_dir", ""))).expanduser().resolve()
    if not str(raw.get("work_dir", "")).strip():
        raise ValueError("STEAM work_dir must be explicit")

    entries = raw.get("datasets")
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("STEAM labeling requires exactly one sft and one rollout dataset")
    datasets = tuple(
        _compose_dataset(entry, tag=tag, camera_keys=camera_keys)
        for entry in entries
    )
    if sorted(dataset.role for dataset in datasets) != ["rollout", "sft"]:
        raise ValueError("STEAM labeling requires dataset roles sft and rollout")
    if len({dataset.dataset_path for dataset in datasets}) != len(datasets):
        raise ValueError("STEAM labeling datasets must be distinct")
    prompts = {_dataset_prompt(dataset.dataset_path) for dataset in datasets}
    if len(prompts) != 1:
        raise ValueError("STEAM labeling datasets must contain the same single task prompt")
    return SteamAdvantageSpec(
        config_path=path,
        checkpoint=checkpoint,
        tag=tag,
        prompt=prompts.pop(),
        stride=stride,
        camera_keys=camera_keys,
        batch_size=batch_size,
        num_workers=num_workers,
        num_processes=num_processes,
        master_port=master_port,
        work_dir=work_dir,
        datasets=datasets,
    )


def inspect_steam_advantages(spec: SteamAdvantageSpec) -> dict[str, Any]:
    """Return the resolved labeling contract without loading a model."""

    weights = _checkpoint_weights(spec.checkpoint)
    return {
        "compatible": True,
        "schema": "clothdojo_steam_advantage_plan",
        "format_version": 1,
        "checkpoint": str(spec.checkpoint),
        "checkpoint_weights": str(weights),
        "checkpoint_config_sha256": _sha256(spec.checkpoint / "config.json"),
        "checkpoint_weights_sha256": _sha256(weights),
        "tag": spec.tag,
        "prompt": spec.prompt,
        "stride": spec.stride,
        "camera_keys": list(spec.camera_keys),
        "batch_size": spec.batch_size,
        "num_processes": spec.num_processes,
        "datasets": [dataset.to_json_dict() for dataset in spec.datasets],
    }


def run_steam_advantages(spec: SteamAdvantageSpec) -> None:
    """Run one rank of distributed inference and publish labels on rank zero."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import torch.distributed as dist
    from rlinf.data.datasets.steam.pair_dataset import (
        BinaryPairDataCollator,
        BinaryPairInferenceDataset,
    )
    from rlinf.models.embodiment.value_model.steam.ensemble_modeling_critic import (
        coerce_to_ensemble,
    )
    from rlinf.models.embodiment.value_model.steam.modeling_critic import (
        SteamCriticModel,
    )
    from torch.utils.data import DataLoader, Subset

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != spec.num_processes:
        raise ValueError(
            f"STEAM config requires {spec.num_processes} processes, got {world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=torch.device(device))

    run_dir = spec.work_dir / f"steam_advantages_{spec.tag}.partial"
    if rank == 0:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
    if world_size > 1:
        dist.barrier()

    model = coerce_to_ensemble(
        SteamCriticModel.from_checkpoint(spec.checkpoint, device=device)
    )
    model.eval()
    _configure_processor_cameras(model, spec.camera_keys)
    weights_path = _checkpoint_weights(spec.checkpoint)
    weights_sha256 = _sha256(weights_path) if rank == 0 else None
    config_sha256 = _sha256(spec.checkpoint / "config.json") if rank == 0 else None
    collator = BinaryPairDataCollator(
        processor=model.members[0].processor,
        max_length=int(getattr(model.config, "max_token_len", 16)),
        train=False,
    )

    try:
        for dataset_index, entry in enumerate(spec.datasets):
            dataset = BinaryPairInferenceDataset(
                dataset_path=str(entry.dataset_path),
                camera_keys=list(spec.camera_keys),
                k=spec.stride,
                prompt=spec.prompt,
                dataset_type=entry.role,
            )
            _replace_pair_dataset_base(dataset)
            selected_indices = _selected_anchor_indices(dataset, entry.episodes)
            rank_indices = selected_indices[rank::world_size]
            loader = DataLoader(
                Subset(dataset, rank_indices),
                batch_size=spec.batch_size,
                shuffle=False,
                num_workers=spec.num_workers,
                collate_fn=collator,
                pin_memory=True,
                persistent_workers=spec.num_workers > 0,
                prefetch_factor=2 if spec.num_workers > 0 else None,
            )
            episodes: list[np.ndarray] = []
            frames: list[np.ndarray] = []
            scores: list[np.ndarray] = []
            for batch_index, batch in enumerate(loader):
                observation = _move_to_device(batch["observation"], device)
                with torch.inference_mode():
                    output = model.predict(observation)
                episodes.append(batch["episode"].numpy(force=True).astype(np.int64))
                frames.append(batch["frame_idx_t"].numpy(force=True).astype(np.int64))
                scores.append(
                    output.predicted_values.detach().float().cpu().numpy().astype(np.float32)
                )
                if rank == 0 and (batch_index == 0 or (batch_index + 1) % 100 == 0):
                    logging.info(
                        "STEAM %s: rank0 batch %d/%d",
                        entry.role,
                        batch_index + 1,
                        len(loader),
                    )
            shard = pa.table(
                {
                    "episode_index": pa.array(_concatenate(episodes, np.int64)),
                    "frame_index": pa.array(_concatenate(frames, np.int64)),
                    "advantage_continuous": pa.array(_concatenate(scores, np.float32)),
                }
            )
            shard_dir = run_dir / f"dataset_{dataset_index}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(shard, shard_dir / f"rank_{rank:03d}.parquet")
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                _publish_dataset_labels(
                    spec=spec,
                    entry=entry,
                    shard_dir=shard_dir,
                    weights_path=weights_path,
                    weights_sha256=weights_sha256,
                    config_sha256=config_sha256,
                )
            if world_size > 1:
                dist.barrier()
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0:
        shutil.rmtree(run_dir)


def smoke_steam_advantages(spec: SteamAdvantageSpec) -> dict[str, Any]:
    """Load the exact evaluator and score one real pair without publishing labels."""

    import torch
    from rlinf.data.datasets.steam.pair_dataset import (
        BinaryPairDataCollator,
        BinaryPairInferenceDataset,
    )
    from rlinf.models.embodiment.value_model.steam.ensemble_modeling_critic import (
        coerce_to_ensemble,
    )
    from rlinf.models.embodiment.value_model.steam.modeling_critic import (
        SteamCriticModel,
    )
    from torch.utils.data import DataLoader, Subset

    device = "cuda:0"
    started = time.monotonic()
    model = coerce_to_ensemble(
        SteamCriticModel.from_checkpoint(spec.checkpoint, device=device)
    )
    model.eval()
    _configure_processor_cameras(model, spec.camera_keys)
    loaded_seconds = time.monotonic() - started
    entry = spec.datasets[0]
    dataset = BinaryPairInferenceDataset(
        dataset_path=str(entry.dataset_path),
        camera_keys=list(spec.camera_keys),
        k=spec.stride,
        prompt=spec.prompt,
        dataset_type=entry.role,
    )
    _replace_pair_dataset_base(dataset)
    index = _selected_anchor_indices(dataset, (entry.episodes[0],))[0]
    collator = BinaryPairDataCollator(
        processor=model.members[0].processor,
        max_length=int(getattr(model.config, "max_token_len", 16)),
        train=False,
    )
    batch = next(
        iter(
            DataLoader(
                Subset(dataset, [index]),
                batch_size=1,
                num_workers=0,
                collate_fn=collator,
            )
        )
    )
    observation = _move_to_device(batch["observation"], device)
    forward_started = time.monotonic()
    with torch.inference_mode():
        output = model.predict(observation)
    values = output.predicted_values.detach().float().cpu().tolist()
    return {
        "compatible": True,
        "checkpoint_load_seconds": loaded_seconds,
        "first_forward_seconds": time.monotonic() - forward_started,
        "episode": int(batch["episode"][0]),
        "frame_index": int(batch["frame_idx_t"][0]),
        "predicted_values": values,
    }


def _compose_dataset(
    value: Any,
    *,
    tag: str,
    camera_keys: tuple[str, ...],
) -> SteamAdvantageDataset:
    if not isinstance(value, dict):
        raise TypeError("STEAM dataset entry must be a mapping")
    role = str(value.get("role", ""))
    if role not in {"sft", "rollout"}:
        raise ValueError(f"unsupported STEAM dataset role: {role!r}")
    dataset = Path(str(value.get("dataset_path", ""))).expanduser().resolve()
    info = _dataset_info(dataset)
    features = info.get("features", {})
    missing = [key for key in camera_keys if key not in features]
    if missing:
        raise ValueError(f"STEAM dataset is missing camera features {missing}: {dataset}")
    for key in camera_keys:
        if features[key].get("dtype") != "video":
            raise ValueError(f"STEAM camera feature must be video: {dataset}: {key}")
    positive_fraction = float(value.get("positive_fraction", 0.0))
    if not np.isfinite(positive_fraction) or not 0.0 < positive_fraction < 1.0:
        raise ValueError("STEAM positive_fraction must be finite and in (0, 1)")
    episodes_path = value.get("episodes_path")
    if value.get("episodes") is not None and episodes_path is not None:
        raise ValueError("set only one of episodes or episodes_path")
    total_episodes = _positive_int(info.get("total_episodes"), "total_episodes")
    if episodes_path is not None:
        episodes = read_episode_selection(episodes_path, total_episodes=total_episodes)
    elif value.get("episodes") is not None:
        episodes = validate_episode_selection(value["episodes"], total_episodes)
    else:
        episodes = tuple(range(total_episodes))
    return SteamAdvantageDataset(
        role=role,
        dataset_path=dataset,
        positive_fraction=positive_fraction,
        episodes=episodes,
        total_episodes=total_episodes,
        total_frames=_positive_int(info.get("total_frames"), "total_frames"),
        output_path=dataset / "meta" / f"advantages_{tag}.parquet",
    )


def _selected_anchor_indices(dataset: Any, episodes: tuple[int, ...]) -> list[int]:
    selected = set(episodes)
    indices: list[int] = []
    offset = 0
    for episode in dataset.eligible_episodes:
        length = int(dataset.source.episode_length(episode))
        count = length - 1
        if episode in selected:
            indices.extend(range(offset, offset + count))
        offset += count
    if selected != set(dataset.eligible_episodes) & selected:
        unavailable = sorted(selected - set(dataset.eligible_episodes))
        raise ValueError(f"selected STEAM episodes cannot form frame pairs: {unavailable}")
    return indices


def _replace_pair_dataset_base(pair_dataset: Any) -> None:
    """Use the validated PyAV backend for an ordinary LeRobot v2.1 dataset."""

    from rlinf.data.datasets.recap.utils import decode_image_struct_batch

    from clothdojo.data.lerobot import open_lerobot_dataset

    source = pair_dataset.source
    root = Path(source.base.root)
    replacement = open_lerobot_dataset(
        root.name,
        root=root,
        download_videos=False,
    )
    replacement.hf_dataset.set_transform(decode_image_struct_batch)
    source.base = replacement


def _publish_dataset_labels(
    *,
    spec: SteamAdvantageSpec,
    entry: SteamAdvantageDataset,
    shard_dir: Path,
    weights_path: Path,
    weights_sha256: str | None = None,
    config_sha256: str | None = None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(path) for path in sorted(shard_dir.glob("rank_*.parquet"))]
    if len(tables) != spec.num_processes:
        raise RuntimeError(f"missing STEAM rank shards in {shard_dir}")
    table = pa.concat_tables(tables)
    episode = np.asarray(
        table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    frame = np.asarray(
        table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    score = np.asarray(
        table["advantage_continuous"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )
    order = np.lexsort((frame, episode))
    episode, frame, score = episode[order], frame[order], score[order]

    lengths = _episode_lengths(entry.dataset_path)
    terminal_episode = np.asarray(entry.episodes, dtype=np.int64)
    terminal_frame = lengths[terminal_episode] - 1
    episode = np.concatenate((episode, terminal_episode))
    frame = np.concatenate((frame, terminal_frame))
    score = np.concatenate((score, np.full(len(terminal_episode), -1.0, np.float32)))
    order = np.lexsort((frame, episode))
    episode, frame, score = episode[order], frame[order], score[order]
    _require_exact_coverage(episode, frame, lengths, entry.episodes)
    if not np.isfinite(score).all() or np.any(score < -1.0001) or np.any(score > 1.0001):
        raise ValueError(f"STEAM scores are not finite values in [-1, 1]: {entry.role}")

    threshold = float(np.quantile(score, 1.0 - entry.positive_fraction))
    labels = score >= threshold
    output = pa.table(
        {
            "episode_index": pa.array(episode, type=pa.int64()),
            "frame_index": pa.array(frame, type=pa.int64()),
            "advantage": pa.array(labels, type=pa.bool_()),
            "advantage_continuous": pa.array(score, type=pa.float32()),
        }
    )
    entry.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = entry.output_path.with_name(f".{entry.output_path.name}.tmp")
    pq.write_table(output, temporary)
    os.replace(temporary, entry.output_path)

    manifest = {
        "schema": "clothdojo_steam_advantages",
        "format_version": 1,
        "tag": spec.tag,
        "role": entry.role,
        "prompt": spec.prompt,
        "checkpoint": str(spec.checkpoint),
        "checkpoint_config_sha256": config_sha256
        or _sha256(spec.checkpoint / "config.json"),
        "checkpoint_weights": str(weights_path),
        "checkpoint_weights_sha256": weights_sha256 or _sha256(weights_path),
        "dataset_path": str(entry.dataset_path),
        "dataset_info_sha256": _sha256(entry.dataset_path / "meta" / "info.json"),
        "episodes": list(entry.episodes),
        "stride": spec.stride,
        "camera_keys": list(spec.camera_keys),
        "positive_fraction_requested": entry.positive_fraction,
        "threshold": threshold,
        "rows": len(labels),
        "positive": int(labels.sum()),
        "negative": int((~labels).sum()),
    }
    manifest_path = entry.output_path.with_suffix(".json")
    _write_json_atomic(manifest_path, manifest)
    logging.info(
        "Published %s STEAM labels: %s (%d positive, %d negative, threshold %.6f)",
        entry.role,
        entry.output_path,
        manifest["positive"],
        manifest["negative"],
        threshold,
    )


def _require_exact_coverage(
    episode: np.ndarray,
    frame: np.ndarray,
    lengths: np.ndarray,
    selected: tuple[int, ...],
) -> None:
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    keys = offsets[episode] + frame
    expected = int(lengths[np.asarray(selected, dtype=np.int64)].sum())
    if len(keys) != expected or len(np.unique(keys)) != expected:
        raise ValueError(f"STEAM label coverage is not exact: {len(keys)} != {expected}")
    expected_keys = np.concatenate(
        [np.arange(offsets[ep], offsets[ep + 1], dtype=np.int64) for ep in selected]
    )
    if not np.array_equal(np.sort(keys), np.sort(expected_keys)):
        raise ValueError("STEAM label episode/frame keys differ from the selected dataset")


def _episode_rows(root: Path) -> list[tuple[int, int]]:
    path = root / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [(int(row["episode_index"]), int(row["length"])) for row in rows]


def _episode_lengths(root: Path) -> np.ndarray:
    rows = _episode_rows(root)
    if [episode for episode, _length in rows] != list(range(len(rows))):
        raise ValueError(f"episode indices are not contiguous: {root}")
    return np.asarray([length for _episode, length in rows], dtype=np.int64)


def _dataset_info(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"STEAM dataset does not exist: {root}")
    path = root / "meta" / "info.json"
    info = _read_object(path, "LeRobot info")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"STEAM requires standard LeRobot v2.1: {path}")
    return info


def _dataset_prompt(root: Path) -> str:
    path = root / "meta" / "tasks.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read LeRobot task metadata: {path}: {exc}") from exc
    prompts = {
        row.get("task", "").strip()
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("task"), str)
    }
    if len(prompts) != 1 or not next(iter(prompts), ""):
        raise ValueError(f"STEAM dataset must contain one non-empty task prompt: {path}")
    return prompts.pop()


def _checkpoint_weights(root: Path) -> Path:
    candidates = (
        root / "model_state_dict" / "full_weights.pt",
        root / "actor" / "model_state_dict" / "full_weights.pt",
        root / "full_weights.pt",
    )
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"STEAM checkpoint config.json does not exist: {root}")
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"STEAM checkpoint has no full weights: {root}")


def _move_to_device(value: Any, device: str) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _configure_processor_cameras(model: Any, camera_keys: tuple[str, ...]) -> None:
    """Map the evaluator's positional view slots onto this dataset's camera keys."""

    processor = model.members[0].processor.image_processor
    # This checkpoint predates processor_config.json, so from_checkpoint uses
    # the library's three-camera defaults. The STEAM backbone pools a dynamic
    # camera axis; the authoritative dual-view contract lives in this owned
    # labeling config and in the advantage manifest.
    processor.image_keys = list(camera_keys)


def _concatenate(values: list[np.ndarray], dtype: Any) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=dtype)
    return np.concatenate(values).astype(dtype, copy=False)


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object: {path}")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    spec = compose_steam_advantage_spec(args.config)
    if args.smoke:
        print(json.dumps(smoke_steam_advantages(spec), indent=2, sort_keys=True))
    else:
        run_steam_advantages(spec)


if __name__ == "__main__":
    main(sys.argv[1:])
