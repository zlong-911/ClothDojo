"""Project-owned launch gate and checkpoint assembly for STEAM critics."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

MEMBER_INDICES = (0, 1, 2)
EXPECTED_FPS = 50
EXPECTED_STATE_DIM = 14
EXPECTED_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
)
RUN_SEED_BASE = 20_260_826

_CONFIG_CONTRACT: tuple[tuple[str, Any], ...] = (
    ("cluster.num_nodes", 4),
    ("runner.task_type", "sft"),
    ("runner.max_epochs", -1),
    ("runner.max_steps", 30_000),
    ("runner.val_check_interval", -1),
    ("runner.save_interval", 1_000),
    ("data.balance_weights", True),
    ("data.seed", 42),
    ("data.dataset_type", "sft"),
    ("data.only_success", False),
    ("data.camera_keys", list(EXPECTED_CAMERAS)),
    ("data.k", 64),
    ("data.length_scale_enabled", True),
    ("data.length_scale_percentile", 90.0),
    ("data.min_episode_length", 2),
    ("data.train_num_workers", 8),
    ("data.eval_num_workers", 0),
    ("data.prefetch_factor", 2),
    ("data.persistent_workers", True),
    ("data.pin_memory", True),
    ("data.do_augment", False),
    ("actor.training_backend", "fsdp"),
    ("actor.micro_batch_size", 32),
    ("actor.global_batch_size", 512),
    ("actor.model.model_type", "steam_value_model"),
    ("actor.model.precision", "fp32"),
    ("actor.model.fusion_hidden_dim", 512),
    ("actor.model.dropout", 0.1),
    ("actor.model.label_smoothing", 0.05),
    ("actor.model.num_frames_per_pair", 2),
    ("actor.model.num_bins", 32),
    ("actor.model.ensemble_size", 1),
    ("actor.model.ensemble_head_seed_base", None),
    ("actor.model.freeze_vision_encoder", False),
    ("actor.model.freeze_language_model", False),
    ("actor.model.include_state", True),
    ("actor.model.state_dim", EXPECTED_STATE_DIM),
    ("actor.model.max_token_len", 16),
    ("actor.model.use_gradient_checkpointing", True),
    ("actor.optim.lr", 5.0e-5),
    ("actor.optim.value_lr", 5.0e-5),
    ("actor.optim.adam_beta1", 0.9),
    ("actor.optim.adam_beta2", 0.95),
    ("actor.optim.adam_eps", 1.0e-8),
    ("actor.optim.weight_decay", 1.0e-5),
    ("actor.optim.clip_grad", 10.0),
    ("actor.optim.lr_scheduler", "constant"),
    ("actor.optim.lr_warmup_steps", 500),
    ("actor.optim.total_training_steps", 30_000),
    ("actor.fsdp_config.strategy", "fsdp"),
    ("actor.fsdp_config.sharding_strategy", "no_shard"),
    ("actor.fsdp_config.use_orig_params", True),
    ("actor.fsdp_config.gradient_checkpointing", False),
    ("actor.fsdp_config.cpu_offload", False),
    ("actor.fsdp_config.reshard_after_forward", False),
    ("actor.fsdp_config.enable_gradient_accumulation", True),
    ("actor.fsdp_config.mixed_precision.param_dtype", "bf16"),
    ("actor.fsdp_config.mixed_precision.reduce_dtype", "fp32"),
    ("actor.fsdp_config.mixed_precision.buffer_dtype", "fp32"),
    ("actor.fsdp_config.amp_autocast.enabled", False),
    ("actor.fsdp_config.grad_scaler.enabled", False),
)


def configure_steam_member(cfg: Any, member_index: int) -> dict[str, int]:
    """Apply the only per-member differences to an already composed config."""

    index = _member_index(member_index)
    head_seed = 100 + index
    run_seed = RUN_SEED_BASE + index * 100_000
    from omegaconf import OmegaConf

    OmegaConf.update(cfg, "actor.group_name", f"SteamMember{index}")
    os.environ["CLOTHDOJO_STEAM_MEMBER_INDEX"] = str(index)
    os.environ["CLOTHDOJO_STEAM_HEAD_SEED"] = str(head_seed)
    os.environ["CLOTHDOJO_STEAM_RUN_SEED"] = str(run_seed)
    return {"member_index": index, "head_seed": head_seed, "run_seed": run_seed}


def inspect_steam_training(cfg: Any, member_index: int) -> dict[str, Any]:
    """Validate the standard-v2.1, FP32-master STEAM training contract."""

    member = configure_steam_member(cfg, member_index)
    resolved = _resolved_config(cfg)
    values: dict[str, Any] = {}
    for dotted_key, expected in _CONFIG_CONTRACT:
        actual = _select(resolved, dotted_key)
        if actual != expected:
            raise ValueError(
                f"STEAM config changed at {dotted_key}: {actual!r} != {expected!r}"
            )
        values[dotted_key] = actual

    entries = resolved["data"].get("train_data_paths")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("STEAM training requires exactly one expert dataset")
    entry = entries[0]
    if entry.get("type") != "sft" or not entry.get("dataset_path"):
        raise ValueError("STEAM train_data_paths[0] must be an explicit sft dataset")

    world_size = 8
    gradient_accumulation = (
        values["actor.global_batch_size"]
        // values["actor.micro_batch_size"]
        // world_size
    )
    if gradient_accumulation != 2:
        raise ValueError("STEAM batch contract must resolve to two accumulation steps")
    placement = resolved["cluster"].get("component_placement", {})
    if list(placement.values()) != [str(os.environ.get("CLOTHDOJO_STEAM_PLACEMENT"))]:
        raise ValueError("STEAM placement must pin the actor to one 8-GPU node")

    runtime = inspect_steam_runtime()
    dataset = inspect_steam_dataset(entry["dataset_path"], cfg=cfg)
    resolved_blob = json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "schema": "clothdojo_steam_launch_preflight",
        "format_version": 1,
        "compatible": True,
        "config_sha256": hashlib.sha256(resolved_blob).hexdigest(),
        "member": member,
        "training": {
            "world_size": world_size,
            "micro_batch_size": values["actor.micro_batch_size"],
            "global_batch_size": values["actor.global_batch_size"],
            "gradient_accumulation": gradient_accumulation,
            "master_precision": values["actor.model.precision"],
            "compute_precision": values[
                "actor.fsdp_config.mixed_precision.param_dtype"
            ],
            "full_model": True,
            "include_state": True,
            "steps": values["runner.max_steps"],
            "warmup_steps": values["actor.optim.lr_warmup_steps"],
            "learning_rate": values["actor.optim.lr"],
        },
        "runtime": runtime,
        "dataset": dataset,
        "resolved_config": resolved,
    }


def inspect_steam_runtime() -> dict[str, str]:
    """Validate the TorchCodec/FFmpeg runtime selected by the throughput gate."""

    backend = os.environ.get("RLINF_STEAM_VIDEO_BACKEND")
    if backend != "torchcodec":
        raise ValueError("RLINF_STEAM_VIDEO_BACKEND must be torchcodec")
    value = os.environ.get("CLOTHDOJO_STEAM_FFMPEG_DIR", "")
    ffmpeg_dir = Path(value).expanduser().resolve()
    if not value or not ffmpeg_dir.is_dir():
        raise FileNotFoundError(
            "CLOTHDOJO_STEAM_FFMPEG_DIR must name the FFmpeg 7 library directory"
        )
    if not any(ffmpeg_dir.glob("libavcodec.so*")):
        raise FileNotFoundError(f"FFmpeg libavcodec is absent: {ffmpeg_dir}")
    return {"video_backend": backend, "ffmpeg_library_dir": str(ffmpeg_dir)}


def inspect_steam_dataset(value: Any, *, cfg: Any) -> dict[str, Any]:
    """Validate the exact complete whole-trajectory Good STEAM dataset."""

    root = Path(str(value)).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    provenance_path = root / "meta" / "clothdojo.json"
    info = _read_object(info_path, "LeRobot info")
    provenance = _read_object(provenance_path, "ClothDojo provenance")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"STEAM requires pinned LeRobot v2.1: {info_path}")
    total_episodes = _positive_int(info.get("total_episodes"), "total_episodes")
    total_frames = _positive_int(info.get("total_frames"), "total_frames")
    if info.get("fps") != EXPECTED_FPS:
        raise ValueError(f"STEAM dataset must use {EXPECTED_FPS} FPS: {info_path}")

    if (
        provenance.get("schema") != "clothdojo_lerobot_export"
        or provenance.get("format_version") != 1
        or provenance.get("export_status") != "complete"
    ):
        raise ValueError(f"unsupported or incomplete dataset provenance: {provenance_path}")
    selection = provenance.get("selection", {})
    counts = provenance.get("counts", {})
    if (
        selection.get("rule") != "explicit_whole_trajectory_good"
        or selection.get("required_review_status") != "complete"
        or counts.get("whole_trajectory_good_only") is not True
        or counts.get("output_episodes") != total_episodes
        or counts.get("output_frames") != total_frames
    ):
        raise ValueError("STEAM dataset is not a complete whole-trajectory Good export")

    features = info.get("features", {})
    if features.get("observation.state", {}).get("shape") != [EXPECTED_STATE_DIM]:
        raise ValueError("STEAM observation.state must be 14-dimensional")
    for camera in EXPECTED_CAMERAS:
        if features.get(camera, {}).get("dtype") != "video":
            raise ValueError(f"STEAM camera is absent or not video: {camera}")
    tasks = _read_tasks(root)
    if len(tasks) != 1 or not tasks[0].strip():
        raise ValueError("STEAM dataset must contain exactly one non-empty prompt")
    prompt = tasks[0].strip()

    episodes = provenance.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != total_episodes:
        raise ValueError("STEAM provenance has an incomplete episode list")
    episodes = sorted(episodes, key=lambda row: row.get("episode_index", -1))
    if [row.get("episode_index") for row in episodes] != list(range(total_episodes)):
        raise ValueError("STEAM episode indices must be contiguous and zero-based")
    if any(
        row.get("quality") != "good"
        or row.get("review_complete") is not True
        or row.get("full_trajectory") is not True
        for row in episodes
    ):
        raise ValueError("STEAM provenance contains a non-Good or incomplete episode")
    lengths = np.asarray([int(row["length"]) for row in episodes], dtype=np.int64)
    if np.any(lengths < 2) or int(lengths.sum()) != total_frames:
        raise ValueError("STEAM episode lengths are invalid or do not sum to total_frames")

    episode_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    data_paths = sorted((root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"STEAM dataset contains no parquet data: {root}")
    for path in data_paths:
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "task_index", "observation.state"],
        )
        episode_parts.append(np.asarray(table["episode_index"].to_numpy(), dtype=np.int64))
        frame_parts.append(np.asarray(table["frame_index"].to_numpy(), dtype=np.int64))
        if np.any(np.asarray(table["task_index"].to_numpy(), dtype=np.int64) != 0):
            raise ValueError(f"STEAM data contains a nonzero task_index: {path}")
        state_parts.append(_as_list_column(table["observation.state"], EXPECTED_STATE_DIM))
    episode_index = np.concatenate(episode_parts)
    frame_index = np.concatenate(frame_parts)
    states = np.concatenate(state_parts).astype(np.float64, copy=False)
    if states.shape != (total_frames, EXPECTED_STATE_DIM) or not np.isfinite(states).all():
        raise ValueError("STEAM state rows are missing, malformed, or non-finite")
    if not np.array_equal(
        episode_index, np.repeat(np.arange(total_episodes), lengths)
    ):
        raise ValueError("STEAM episode_index is not contiguous")
    if not np.array_equal(
        frame_index, np.concatenate([np.arange(length) for length in lengths])
    ):
        raise ValueError("STEAM frame_index is not contiguous within each episode")

    observed_mean = states.mean(axis=0)
    observed_std = states.std(axis=0)
    configured_mean = np.asarray(cfg.actor.model.state_mean, dtype=np.float64)
    configured_std = np.asarray(cfg.actor.model.state_std, dtype=np.float64)
    if not np.allclose(observed_mean, configured_mean, rtol=0.0, atol=2e-7):
        raise ValueError(f"STEAM state_mean differs from data: {observed_mean.tolist()}")
    if not np.allclose(observed_std, configured_std, rtol=0.0, atol=2e-7):
        raise ValueError(f"STEAM state_std differs from data: {observed_std.tolist()}")

    return {
        "root": str(root),
        "info_sha256": _sha256(info_path),
        "provenance_sha256": _sha256(provenance_path),
        "episodes": total_episodes,
        "frames": total_frames,
        "pair_samples": 2 * (total_frames - total_episodes),
        "fps": EXPECTED_FPS,
        "prompt": prompt,
        "camera_keys": list(EXPECTED_CAMERAS),
        "state_dim": EXPECTED_STATE_DIM,
        "state_mean": observed_mean.tolist(),
        "state_std": observed_std.tolist(),
        "episode_length": {
            "min": int(lengths.min()),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
        },
        "per_member_dataset_passes": (
            int(cfg.runner.max_steps)
            * int(cfg.actor.global_batch_size)
            / (2 * (total_frames - total_episodes))
        ),
    }


def write_steam_launch_manifest(cfg: Any, manifest: dict[str, Any]) -> Path:
    """Write the checked launch manifest before Ray resources are allocated."""

    root = Path(str(cfg.runner.logger.log_path)).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "steam_launch_manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def merge_steam_members(
    members: list[str], output: str | Path, *, overwrite: bool = False
) -> Path:
    """Assemble independently trained members with RLINF's canonical merger."""

    if len(members) != 3:
        raise ValueError("ClothDojo STEAM requires exactly three member checkpoints")
    from rlinf.models.embodiment.value_model.steam.checkpoint_merge import (
        merge_ensemble_checkpoints,
    )

    return merge_ensemble_checkpoints(members, output, overwrite=overwrite)


def _member_index(value: int) -> int:
    index = int(value)
    if index not in MEMBER_INDICES:
        raise ValueError(f"STEAM member_index must be one of {MEMBER_INDICES}")
    return index


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"STEAM {name} must be a positive integer")
    return value


def _resolved_config(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved STEAM config must be a mapping")
    return value


def _select(value: dict[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"STEAM config is missing {dotted_key}")
        current = current[part]
    return current


def _read_tasks(root: Path) -> list[str]:
    parquet = root / "meta" / "tasks.parquet"
    if parquet.is_file():
        return [str(value) for value in pq.read_table(parquet)["task"].to_pylist()]
    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.is_file():
        return [
            str(json.loads(line)["task"])
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise FileNotFoundError(f"STEAM task metadata is absent: {root}")


def _as_list_column(column: Any, width: int) -> np.ndarray:
    array = column.combine_chunks()
    values = np.asarray(array.values.to_numpy(zero_copy_only=False))
    return values.reshape(-1, width)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
