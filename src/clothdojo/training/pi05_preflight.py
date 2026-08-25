"""Read-only launch preflight for the reviewed-good Pi0.5 SFT baseline."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_TASK = "flatten the shirt"
EXPECTED_BASE_COMMIT = "7de663972b7817d2c4cf2d84c821153dfea772e9"
MIN_SEGMENT_FRAMES = 65
TRAINING_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

# This table is the executable form of docs/pi05_sft.md's resolved experiment
# contract. Values are compared against the composed YAML before any artifact
# checks so command-line overrides cannot silently create a different baseline.
_PI05_CONFIG_CONTRACT: tuple[tuple[str, Any], ...] = (
    ("cluster.num_nodes", 1),
    ("cluster.component_placement.actor", "0-7"),
    ("runner.task_type", "sft"),
    ("runner.logger.project_name", "clothdojo"),
    ("runner.logger.experiment_name", "shirt_flatten_reviewed_good_pi05_aloha_40k_v2"),
    ("runner.logger.logger_backends", ["tensorboard"]),
    ("runner.max_epochs", -1),
    ("runner.max_steps", 40000),
    ("runner.val_check_interval", -1),
    ("runner.save_interval", 10000),
    ("actor.group_name", "ActorGroup"),
    ("actor.training_backend", "fsdp"),
    ("actor.micro_batch_size", 8),
    ("actor.global_batch_size", 64),
    ("actor.seed", 42),
    ("actor.model.model_type", "openpi"),
    ("actor.model.precision", None),
    ("actor.model.num_action_chunks", 50),
    ("actor.model.action_dim", 14),
    ("actor.model.is_lora", False),
    ("actor.model.lora_rank", 32),
    ("actor.model.use_proprio", True),
    ("actor.model.num_steps", 5),
    ("actor.model.add_value_head", False),
    ("actor.model.openpi.config_name", "pi05_aloha_robotwin"),
    ("actor.model.openpi.num_images_in_input", 3),
    ("actor.model.openpi.noise_level", 0.5),
    ("actor.model.openpi.action_chunk", 50),
    ("actor.model.openpi.num_steps", 5),
    ("actor.model.openpi.train_expert_only", False),
    ("actor.model.openpi.action_env_dim", 14),
    ("actor.model.openpi.noise_method", "flow_sde"),
    ("actor.model.openpi.add_value_head", False),
    ("actor.model.openpi.value_after_vlm", False),
    ("actor.model.openpi.value_vlm_mode", "mean_token"),
    ("actor.model.openpi.detach_critic_input", True),
    ("actor.model.openpi_data.adapt_to_pi", False),
    ("actor.model.openpi_data.extra_delta_transform", True),
    ("actor.model.openpi_data.default_prompt", "Flatten the shirt."),
    ("actor.optim.lr", 2.5e-5),
    ("actor.optim.adam_beta1", 0.9),
    ("actor.optim.adam_beta2", 0.95),
    ("actor.optim.adam_eps", 1.0e-8),
    ("actor.optim.weight_decay", 1.0e-10),
    ("actor.optim.clip_grad", 1.0),
    ("actor.optim.total_training_steps", 30000),
    ("actor.optim.lr_warmup_steps", 1000),
    ("actor.optim.lr_scheduler", "openpi_cosine"),
    ("actor.optim.min_lr", 2.5e-6),
    ("actor.fsdp_config.strategy", "fsdp"),
    ("actor.fsdp_config.sharding_strategy", "no_shard"),
    ("actor.fsdp_config.use_orig_params", False),
    ("actor.fsdp_config.gradient_checkpointing", False),
    ("actor.fsdp_config.cpu_offload", False),
    ("actor.fsdp_config.offload_pin_memory", False),
    ("actor.fsdp_config.reshard_after_forward", True),
    ("actor.fsdp_config.enable_gradient_accumulation", True),
    ("actor.fsdp_config.forward_prefetch", False),
    ("actor.fsdp_config.limit_all_gathers", False),
    ("actor.fsdp_config.backward_prefetch", None),
    ("actor.fsdp_config.use_liger_kernel", False),
    ("actor.fsdp_config.mixed_precision.param_dtype", None),
    ("actor.fsdp_config.mixed_precision.reduce_dtype", None),
    ("actor.fsdp_config.mixed_precision.buffer_dtype", None),
    ("actor.fsdp_config.amp_autocast.enabled", False),
    ("actor.fsdp_config.amp_autocast.precision", "bf16"),
    ("actor.fsdp_config.grad_scaler.enabled", False),
)


@dataclass(frozen=True, slots=True)
class _ExpectedModelLayout:
    shapes: dict[str, tuple[int, ...]]
    tied_groups: tuple[tuple[str, ...], ...]


def inspect_pi05_training(cfg: Any) -> dict[str, Any]:
    """Validate one composed Pi0.5 config without loading tensor payloads."""

    resolved = _resolved_config(cfg)
    contract = _validate_config_contract(resolved)
    actor = cfg.actor.model
    horizon = int(actor.num_action_chunks)

    dataset = inspect_pi05_dataset(cfg.data.train_data_paths)
    norm = inspect_pi05_norm_stats(
        actor.openpi_data.norm_stats_path,
        dataset=dataset,
        config_name=str(actor.openpi.config_name),
        data_kwargs=_plain_data_kwargs(actor.openpi_data),
        action_horizon=horizon,
    )
    checkpoint = _inspect_hf_base(actor.model_path, cfg)
    resolved_json = json.dumps(
        resolved,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "schema": "clothdojo_pi05_launch_preflight",
        "format_version": 1,
        "compatible": True,
        "config_sha256": hashlib.sha256(resolved_json).hexdigest(),
        "config_contract": contract,
        "training": {
            "run_steps": contract["values"]["runner.max_steps"],
            "save_interval": contract["values"]["runner.save_interval"],
            "world_size": contract["derived"]["world_size"],
            "local_batch_size": contract["values"]["actor.micro_batch_size"],
            "global_batch_size": contract["values"]["actor.global_batch_size"],
            "gradient_accumulation": contract["derived"]["gradient_accumulation"],
            "model_type": contract["values"]["actor.model.model_type"],
            "openpi_config_name": contract["values"]["actor.model.openpi.config_name"],
            "action_env_dim": contract["values"]["actor.model.action_dim"],
            "model_action_dim": checkpoint["model_action_dim"],
            "action_horizon": contract["values"]["actor.model.num_action_chunks"],
            "task": dataset["task"],
            "training_cameras": dataset["training_cameras"],
            "optimizer": contract["derived"]["optimizer"],
            "lr_schedule": contract["derived"]["lr_schedule"],
        },
        "dataset": dataset,
        "norm_stats": norm,
        "checkpoint": checkpoint,
        "resolved_config": resolved,
    }


def write_pi05_launch_manifest(cfg: Any, manifest: dict[str, Any]) -> Path:
    """Write the validated manifest once, before any Ray resources are created."""

    output = Path(str(cfg.runner.logger.log_path)).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(f"Pi0.5 output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"Pi0.5 output directory is not unique and empty: {output}")
    else:
        output.mkdir(parents=True)
    path = output / "pi05_launch_manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Pi0.5 output directory is not unique; launch manifest already exists: {path}"
        ) from exc
    return path


def inspect_pi05_dataset(value: Any) -> dict[str, Any]:
    """Inspect the reviewed-good LeRobot artifact shared by both SFT backends."""
    root = Path(str(value)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Pi0.5 dataset does not exist: {root}")
    info_path = root / "meta" / "info.json"
    sidecar_path = root / "meta" / "clothdojo.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    info = _read_object(info_path, "LeRobot info")
    sidecar = _read_object(sidecar_path, "ClothDojo dataset sidecar")

    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Pi0.5 requires pinned LeRobot v2.1: {info_path}")
    total_episodes = _positive_int(info.get("total_episodes"), "total_episodes")
    total_frames = _positive_int(info.get("total_frames"), "total_frames")
    if info.get("total_tasks") != 1:
        raise ValueError(f"Pi0.5 dataset must declare exactly one task: {info_path}")
    _validate_features(info.get("features"), root)

    if sidecar.get("schema") != "clothdojo_lerobot_export":
        raise ValueError(f"unsupported ClothDojo dataset sidecar: {sidecar_path}")
    if sidecar.get("format_version") != 1 or sidecar.get("export_status") != "complete":
        raise ValueError(f"ClothDojo dataset export is incomplete: {sidecar_path}")
    if sidecar.get("training_eligible", True) is not True:
        raise ValueError(f"ClothDojo dataset is ineligible for training: {sidecar_path}")
    selection = sidecar.get("selection")
    if not isinstance(selection, dict):
        raise TypeError(f"dataset selection provenance must be an object: {sidecar_path}")
    minimum = _positive_int(selection.get("min_segment_frames"), "min_segment_frames")
    if minimum < MIN_SEGMENT_FRAMES:
        raise ValueError(
            f"dataset min_segment_frames must be >= {MIN_SEGMENT_FRAMES}, got {minimum}"
        )

    tasks = _read_jsonl(tasks_path, "LeRobot tasks")
    task_values = {row.get("task") for row in tasks}
    if len(tasks) != 1 or task_values != {EXPECTED_TASK} or tasks[0].get("task_index") != 0:
        raise ValueError(
            f"Pi0.5 dataset task must be exactly {EXPECTED_TASK!r}: {tasks_path}"
        )
    episodes = _read_jsonl(episodes_path, "LeRobot episodes")
    if len(episodes) != total_episodes:
        raise ValueError("LeRobot episode count differs from info.json")
    episodes.sort(key=lambda row: row.get("episode_index", -1))
    if [row.get("episode_index") for row in episodes] != list(range(total_episodes)):
        raise ValueError("LeRobot episode indices must be contiguous and zero-based")
    lengths = [_positive_int(row.get("length"), "episode length") for row in episodes]
    if sum(lengths) != total_frames:
        raise ValueError("LeRobot episode lengths do not sum to total_frames")
    if min(lengths) < MIN_SEGMENT_FRAMES:
        raise ValueError(
            f"dataset contains an episode shorter than {MIN_SEGMENT_FRAMES} frames"
        )

    sidecar_episodes = sidecar.get("episodes")
    counts = sidecar.get("counts")
    if not isinstance(sidecar_episodes, list) or len(sidecar_episodes) != total_episodes:
        raise ValueError("dataset sidecar episode list is incomplete")
    if not isinstance(counts, dict):
        raise TypeError("dataset sidecar counts must be an object")
    if (
        counts.get("output_episodes") != total_episodes
        or counts.get("output_frames") != total_frames
    ):
        raise ValueError("dataset sidecar counts differ from info.json")
    if counts.get("min_segment_frames") != minimum:
        raise ValueError("dataset sidecar minimum-length provenance is inconsistent")
    for index, (record, length) in enumerate(zip(sidecar_episodes, lengths, strict=True)):
        if record.get("episode_index") != index or record.get("length") != length:
            raise ValueError("dataset sidecar episode extents differ from LeRobot metadata")

    return {
        "root": str(root),
        "info_path": str(info_path),
        "info_sha256": _sha256(info_path),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "tasks_path": str(tasks_path),
        "tasks_sha256": _sha256(tasks_path),
        "episodes_path": str(episodes_path),
        "episodes_sha256": _sha256(episodes_path),
        "episodes": total_episodes,
        "frames": total_frames,
        "fps": _positive_int(info.get("fps"), "fps"),
        "min_segment_frames": minimum,
        "minimum_episode_frames": min(lengths),
        "task": EXPECTED_TASK,
        "training_cameras": list(TRAINING_CAMERAS),
        "state_dim": 14,
        "action_dim": 14,
    }


def inspect_pi05_norm_stats(
    value: Any,
    *,
    dataset: dict[str, Any],
    config_name: str,
    data_kwargs: dict[str, Any],
    action_horizon: int,
) -> dict[str, Any]:
    """Require normalization statistics to match the exact reviewed dataset."""
    root = Path(str(value)).expanduser().resolve()
    stats_path = root / "norm_stats.json"
    provenance_path = root / "clothdojo.json"
    stats = _read_object(stats_path, "OpenPI normalization statistics")
    provenance = _read_object(provenance_path, "OpenPI norm-stat provenance")
    if provenance.get("schema") != "clothdojo_openpi_norm_stats":
        raise ValueError(f"unsupported norm-stat provenance: {provenance_path}")
    if provenance.get("format_version") != 1:
        raise ValueError(f"unsupported norm-stat format version: {provenance_path}")
    expected = {
        "dataset_info_sha256": dataset["info_sha256"],
        "dataset_sidecar_sha256": dataset["sidecar_sha256"],
        "config_name": config_name,
        "data_kwargs": data_kwargs,
        "action_horizon": action_horizon,
        "frames": dataset["frames"],
        "episodes": dataset["episodes"],
    }
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            raise ValueError(
                f"norm-stat provenance mismatch for {key}: "
                f"{provenance.get(key)!r} != {expected_value!r}"
            )
    values = stats.get("norm_stats")
    if not isinstance(values, dict) or set(values) != {"state", "actions"}:
        raise ValueError(f"norm_stats.json must contain state and actions: {stats_path}")
    for name in ("state", "actions"):
        record = values[name]
        if not isinstance(record, dict):
            raise TypeError(f"norm stats {name} must be an object")
        for statistic in ("mean", "std", "q01", "q99"):
            vector = record.get(statistic)
            if not isinstance(vector, list) or len(vector) != 14:
                raise ValueError(f"norm stats {name}.{statistic} must be 14D")
    return {
        "root": str(root),
        "stats_path": str(stats_path),
        "stats_sha256": _sha256(stats_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        **expected,
    }


def _inspect_hf_base(value: Any, cfg: Any) -> dict[str, Any]:
    root = Path(str(value)).expanduser().resolve()
    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    config = _read_object(config_path, "Hugging Face Pi0.5 config")
    if config.get("type") != "pi05":
        raise ValueError(f"HF base config must have type=pi05: {config_path}")
    if config.get("dtype") != "float32":
        raise ValueError(f"HF base config must declare dtype=float32: {config_path}")
    dimensions: dict[str, int] = {}
    for name, keys, expected, label in (
        ("model_action_dim", ("max_action_dim", "action_dim"), 32, "action dimension"),
        ("model_state_dim", ("max_state_dim", "state_dim"), 32, "state dimension"),
        ("action_horizon", ("chunk_size", "action_horizon"), 50, "action horizon"),
    ):
        actual = _first_present(config, keys)
        if actual != expected:
            raise ValueError(f"HF base {label} must be {expected}, got {actual!r}")
        dimensions[name] = actual
    if not weights_path.is_file():
        raise FileNotFoundError(f"HF base model.safetensors does not exist: {weights_path}")
    weights_blob = weights_path.resolve()
    is_blob = len(weights_blob.name) == 64 and all(
        character in "0123456789abcdef" for character in weights_blob.name
    )
    if not is_blob:
        raise ValueError(f"HF base weights must resolve to a pinned SHA-256 blob: {weights_blob}")

    from safetensors import safe_open

    actual_shapes: dict[str, tuple[int, ...]] = {}
    dtype_counts: Counter[str] = Counter()
    with safe_open(weights_path, framework="pt", device="cpu") as stream:
        for key in stream.keys():
            tensor = stream.get_slice(key)
            actual_shapes[key] = tuple(int(dim) for dim in tensor.get_shape())
            dtype_counts[str(tensor.get_dtype())] += 1
    expected = _expected_pi05_layout(cfg)
    missing = sorted(set(expected.shapes) - set(actual_shapes))
    unexpected = sorted(set(actual_shapes) - set(expected.shapes))
    if unexpected:
        raise ValueError(f"HF base contains unexpected model keys: {unexpected[:8]}")
    bad_shapes = {
        key: {"expected": expected.shapes[key], "actual": actual_shapes[key]}
        for key in sorted(actual_shapes)
        if actual_shapes[key] != expected.shapes[key]
    }
    if bad_shapes:
        raise ValueError(f"HF base tensor shape mismatch: {dict(list(bad_shapes.items())[:8])}")

    accepted_aliases = []
    remaining = set(missing)
    for group in expected.tied_groups:
        absent = sorted(set(group) & remaining)
        present = sorted(set(group) & set(actual_shapes))
        if len(group) == 2 and len(absent) == 1 and len(present) == 1:
            accepted_aliases.append(
                {"omitted_key": absent[0], "stored_key": present[0], "semantics": "same_parameter"}
            )
            remaining.remove(absent[0])
    if remaining or len(accepted_aliases) != 1:
        raise ValueError(
            "HF base key coverage differs from the meta-device model; "
            f"missing={sorted(remaining)}, tied_aliases={accepted_aliases}"
        )
    if set(dtype_counts) != {"F32"}:
        raise ValueError(f"HF base tensors must all be stored as F32, got {dict(dtype_counts)}")
    is_commit = len(root.name) == 40 and all(c in "0123456789abcdef" for c in root.name)
    if not is_commit:
        raise ValueError(f"HF base must be a pinned 40-hex snapshot directory: {root}")
    commit = root.name
    if commit != EXPECTED_BASE_COMMIT:
        raise ValueError(
            f"HF base snapshot differs from the Pi0.5 contract: "
            f"{commit} != {EXPECTED_BASE_COMMIT}"
        )
    return {
        "root": str(root),
        "snapshot_commit": commit,
        **dimensions,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "weights_path": str(weights_path),
        "weights_blob_path": str(weights_blob),
        "weights_blob_sha256": weights_blob.name,
        "weights_size_bytes": weights_path.stat().st_size,
        "tensor_count": len(actual_shapes),
        "tensor_elements": sum(math.prod(shape) for shape in actual_shapes.values()),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "expected_state_dict_keys": len(expected.shapes),
        "tied_aliases": accepted_aliases,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
    }


def _expected_pi05_layout(cfg: Any) -> _ExpectedModelLayout:
    """Build only meta tensors and retain duplicate names to prove tie semantics."""

    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    del cfg
    with torch.device("meta"):
        model = PI0Pytorch(
            Pi0Config(
                pi05=True,
                dtype="bfloat16",
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
                action_dim=32,
                action_horizon=50,
                max_token_len=200,
            )
        )
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    aliases: defaultdict[int, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].append(name)
    tied = tuple(tuple(names) for names in aliases.values() if len(names) > 1)
    del model
    return _ExpectedModelLayout(shapes=shapes, tied_groups=tied)


def _validate_config_contract(resolved: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path, expected in _PI05_CONFIG_CONTRACT:
        actual = _nested_value(resolved, path)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"Pi0.5 config contract drift at {path}: "
                f"expected {expected!r}, got {actual!r}"
            )
        values[path] = actual

    placement = values["cluster.component_placement.actor"]
    start, end = (int(part) for part in placement.split("-", 1))
    world_size = end - start + 1
    micro = values["actor.micro_batch_size"]
    global_batch = values["actor.global_batch_size"]
    if world_size != 8 or global_batch % (world_size * micro) != 0:
        raise ValueError("Pi0.5 batch contract does not resolve to eight local ranks")
    accumulation = global_batch // (world_size * micro)
    if accumulation != 1:
        raise ValueError("Pi0.5 gradient accumulation must be exactly one")
    return {
        "reference": "docs/pi05_sft.md#resolved-experiment-contract",
        "values": values,
        "derived": {
            "world_size": world_size,
            "gradient_accumulation": accumulation,
            "optimizer": {
                "type": "torch.optim.AdamW",
                "lr": values["actor.optim.lr"],
                "betas": [
                    values["actor.optim.adam_beta1"],
                    values["actor.optim.adam_beta2"],
                ],
                "eps": values["actor.optim.adam_eps"],
                "weight_decay": values["actor.optim.weight_decay"],
                "clip_grad": values["actor.optim.clip_grad"],
            },
            "lr_schedule": {
                "type": values["actor.optim.lr_scheduler"],
                "warmup_steps": values["actor.optim.lr_warmup_steps"],
                "schedule_steps": values["actor.optim.total_training_steps"],
                "min_lr": values["actor.optim.min_lr"],
            },
        },
    }


def _nested_value(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Pi0.5 config contract field is missing: {path}")
        current = current[part]
    return current


def _validate_features(value: Any, root: Path) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"LeRobot features must be an object: {root}")
    for key in ("observation.state", "action"):
        feature = value.get(key)
        if (
            not isinstance(feature, dict)
            or feature.get("dtype") != "float32"
            or feature.get("shape") != [14]
        ):
            raise ValueError(f"Pi0.5 feature {key} must be float32[14]: {root}")
    for key in TRAINING_CAMERAS:
        feature = value.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"Pi0.5 training camera must be a video feature: {key}")
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Pi0.5 training camera must have HWC RGB shape: {key}")


def _plain_data_kwargs(value: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    plain = OmegaConf.to_container(value, resolve=True)
    if not isinstance(plain, dict):
        raise TypeError("actor.model.openpi_data must be a mapping")
    plain.pop("norm_stats_path", None)
    return plain


def _resolved_config(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved Pi0.5 config must be a mapping")
    return value


def _first_present(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object: {path}")
    return value


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {context}: {path}: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{context} must contain JSON objects: {path}")
    return rows


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
