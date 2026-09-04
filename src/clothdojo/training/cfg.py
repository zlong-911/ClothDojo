"""Static preflight checks for ClothDojo's offline CFG mixture."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_FEATURES = (
    "observation.state",
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "action",
)
CFG_ROLES = ("sft", "rollout")


@dataclass(frozen=True, slots=True)
class CfgDatasetReport:
    role: str
    dataset_path: str
    advantage_tag: str | None
    prompt: str
    weight: float
    episodes: int
    frames: int
    fps: int
    advantage_path: str
    advantage_manifest_path: str
    positive_frames: int
    negative_frames: int
    steam_checkpoint_config_sha256: str
    steam_checkpoint_weights_sha256: str
    steam_stride: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_cfg_training(
    config_path: str | Path,
    overrides: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Validate every labeled dataset referenced by one CFG config."""

    from clothdojo.rollout.checkpoint import rlinf_openpi_weights_sha256
    from clothdojo.training.openpi import TrainingPrompt
    from clothdojo.training.rlinf import compose_config

    cfg = compose_config(config_path, overrides)
    if cfg.actor.model.model_type != "clothdojo_cfg_model":
        raise ValueError("CFG config must use model_type=clothdojo_cfg_model")
    openpi_data = cfg.actor.model.get("openpi_data", {})
    training_prompt = TrainingPrompt(openpi_data.get("default_prompt")).prompt
    if openpi_data.get("adapt_to_pi") is not False:
        raise ValueError("Piper CFG requires actor.model.openpi_data.adapt_to_pi=false")
    if openpi_data.get("extra_delta_transform") is not True:
        raise ValueError("Piper CFG requires extra_delta_transform=true")
    norm_root = Path(str(openpi_data.get("norm_stats_path", ""))).expanduser().resolve()
    if not (norm_root / "norm_stats.json").is_file():
        raise FileNotFoundError(f"CFG norm_stats.json does not exist: {norm_root}")
    norm_provenance = _read_object(norm_root / "clothdojo.json", "CFG norm-stat provenance")
    if norm_provenance.get("schema") != "clothdojo_openpi_norm_stats":
        raise ValueError(f"unsupported CFG norm-stat provenance: {norm_root}")
    if norm_provenance.get("config_name") != cfg.actor.model.openpi.config_name:
        raise ValueError("CFG norm stats use a different OpenPI config")
    model_path = _validate_init_checkpoint(cfg.actor.model.model_path)
    init_weights_sha256 = rlinf_openpi_weights_sha256(model_path)

    data_cfg = cfg.data
    balance_dataset_weights = bool(data_cfg.get("balance_dataset_weights", True))
    entries = list(data_cfg.get("train_data_paths", ()))
    roles = [str(entry.get("type", "")) for entry in entries]
    if roles.count("sft") != 1:
        raise ValueError(
            f"CFG requires exactly one sft dataset and optional rollout datasets, got {roles}"
        )
    unsupported = sorted(set(roles) - set(CFG_ROLES))
    if unsupported:
        raise ValueError(f"unsupported CFG dataset types: {unsupported}")

    reports = [
        inspect_cfg_dataset(
            entry["dataset_path"],
            role=str(entry["type"]),
            weight=float(entry.get("weight", 1.0)),
            advantage_tag=entry.get("advantage_tag"),
            episodes=entry.get("episodes"),
            episodes_path=entry.get("episodes_path"),
        )
        for entry in entries
    ]
    _require_compatible_datasets(reports)
    expert = next(report for report in reports if report.role == "sft")
    expert_info = Path(expert.dataset_path) / "meta" / "info.json"
    norm_binding = _validate_norm_binding(
        model_path, norm_root, norm_provenance, expert_info, cfg.actor.model
    )
    if sum(report.positive_frames for report in reports) == 0:
        raise ValueError("CFG mixture has no positive advantage labels")
    if sum(report.negative_frames for report in reports) == 0:
        raise ValueError("CFG mixture has no negative advantage labels")

    raw_weights = np.asarray([report.weight for report in reports], dtype=np.float64)
    if not np.isfinite(raw_weights).all() or np.any(raw_weights <= 0):
        raise ValueError(f"CFG dataset weights must be finite and positive: {raw_weights}")
    sampling_weights = raw_weights.copy()
    if balance_dataset_weights:
        sampling_weights *= np.asarray([report.frames for report in reports], dtype=np.float64)
    probabilities = sampling_weights / sampling_weights.sum()
    role_probabilities = {
        role: float(
            sum(
                probability
                for report, probability in zip(reports, probabilities, strict=True)
                if report.role == role
            )
        )
        for role in CFG_ROLES
    }
    model = cfg.actor.model
    norm_file = norm_root / "norm_stats.json"
    provenance_file = norm_root / "clothdojo.json"
    return {
        "compatible": True,
        "training_prompt": training_prompt,
        "image_augmentation": False,
        "config": str(Path(config_path).expanduser().resolve()),
        "model_type": str(cfg.actor.model.model_type),
        "model_path": str(model_path),
        "init_checkpoint_weights_sha256": init_weights_sha256,
        "norm_stats_path": str(norm_root),
        "norm_binding": norm_binding,
        "norm_stats": {
            "stats_sha256": _sha256(norm_file),
            "provenance_sha256": _sha256(provenance_file),
        },
        "inference_contract": {
            "openpi_config": str(model.openpi.config_name),
            "action_dim": int(model.action_dim),
            "action_horizon": int(model.num_action_chunks),
            "num_steps": int(model.openpi.num_steps),
            "guidance_type": str(model.openpi.guidance_type),
            "guidance_scale": float(model.openpi.cfgrl_guidance_scale),
            "adapt_to_pi": False,
            "extra_delta_transform": True,
            "train_expert_only": bool(model.openpi.train_expert_only),
            "positive_only_conditional": bool(model.openpi.positive_only_conditional),
            "unconditional_prob": float(model.openpi.unconditional_prob),
        },
        "balance_dataset_weights": balance_dataset_weights,
        "sampling_probabilities": role_probabilities,
        "dataset_sampling_probabilities": [
            {
                "role": report.role,
                "dataset_path": report.dataset_path,
                "probability": float(probability),
            }
            for report, probability in zip(reports, probabilities, strict=True)
        ],
        "datasets": [report.to_json_dict() for report in reports],
    }


def write_cfg_launch_manifest(cfg: Any, report: dict[str, Any]) -> Path:
    """Publish the checked training/inference contract before RLINF starts."""

    output = Path(str(cfg.runner.logger.log_path)).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(f"CFG output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"CFG output directory is not unique and empty: {output}")
    else:
        output.mkdir(parents=True)
    path = output / "cfg_launch_manifest.json"
    payload = {
        "schema": "clothdojo_cfg_launch",
        "format_version": 1,
        **report,
    }
    temporary = output / ".cfg_launch_manifest.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def inspect_cfg_dataset(
    dataset_path: str | Path,
    *,
    role: str,
    weight: float,
    advantage_tag: Any = None,
    episodes: Any = None,
    episodes_path: Any = None,
) -> CfgDatasetReport:
    """Validate one pinned-LeRobot dataset and exact advantage coverage."""

    if role not in CFG_ROLES:
        raise ValueError(f"unsupported CFG dataset role: {role!r}")
    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CFG {role} dataset does not exist: {root}")
    info = _read_object(root / "meta" / "info.json", "LeRobot info")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"CFG requires pinned LeRobot v2.1: {root}")
    total_episodes = _positive_int(info.get("total_episodes"), "total_episodes")
    total_frames = _positive_int(info.get("total_frames"), "total_frames")
    fps = _positive_int(info.get("fps"), "fps")
    _validate_features(info.get("features"), root)
    _validate_sidecar(root)
    from clothdojo.data.lerobot import load_single_task_prompt

    prompt = load_single_task_prompt(root)
    lengths = _episode_lengths(root, total_episodes, total_frames)
    selected = resolve_cfg_episode_selection(
        episodes=episodes,
        episodes_path=episodes_path,
        total_episodes=total_episodes,
    )

    tag = _optional_advantage_tag(advantage_tag)
    filename = f"advantages_{tag}.parquet" if tag else "advantages.parquet"
    advantage_path = root / "meta" / filename
    positive, negative, selected_frames = _validate_advantages(advantage_path, lengths, selected)
    manifest_path = advantage_path.with_suffix(".json")
    manifest = _validate_advantage_manifest(
        manifest_path,
        root=root,
        role=role,
        prompt=prompt,
        selected=selected,
        rows=selected_frames,
        positive=positive,
        negative=negative,
    )
    return CfgDatasetReport(
        role=role,
        dataset_path=str(root),
        advantage_tag=tag,
        prompt=prompt,
        weight=float(weight),
        episodes=len(selected),
        frames=selected_frames,
        fps=fps,
        advantage_path=str(advantage_path),
        advantage_manifest_path=str(manifest_path),
        positive_frames=positive,
        negative_frames=negative,
        steam_checkpoint_config_sha256=_require_sha256(
            manifest.get("checkpoint_config_sha256"), "STEAM checkpoint config"
        ),
        steam_checkpoint_weights_sha256=_require_sha256(
            manifest.get("checkpoint_weights_sha256"), "STEAM checkpoint weights"
        ),
        steam_stride=_positive_int(manifest.get("stride"), "STEAM stride"),
    )


def _optional_advantage_tag(value: Any) -> str | None:
    if value is None:
        return None
    from clothdojo.training.steam_advantages import _advantage_tag

    return _advantage_tag(value)


def resolve_cfg_episode_selection(
    *,
    episodes: Any,
    episodes_path: Any,
    total_episodes: int,
) -> np.ndarray:
    """Resolve an inline or file-backed CFG episode selection."""

    if episodes is not None and episodes_path is not None:
        raise ValueError("CFG dataset may set only one of episodes or episodes_path")
    if episodes_path is not None:
        from clothdojo.training.episode_selection import read_episode_selection

        return np.asarray(
            read_episode_selection(episodes_path, total_episodes=total_episodes),
            dtype=np.int64,
        )
    return _selected_episodes(episodes, total_episodes)


def _validate_advantage_manifest(
    path: Path,
    *,
    root: Path,
    role: str,
    prompt: str,
    selected: np.ndarray,
    rows: int,
    positive: int,
    negative: int,
) -> dict[str, Any]:
    manifest = _read_object(path, "STEAM advantage provenance")
    if manifest.get("schema") != "clothdojo_steam_advantages":
        raise ValueError(f"unsupported STEAM advantage provenance: {path}")
    if manifest.get("format_version") != 1:
        raise ValueError(f"unsupported STEAM advantage provenance: {path}")
    if manifest.get("dataset_info_sha256") != _sha256(root / "meta" / "info.json"):
        raise ValueError(f"STEAM advantages target a different dataset revision: {path}")
    expected_contract = {
        "role": role,
        "prompt": prompt,
        "state_key": "observation.state",
        "camera_keys": [
            "observation.images.cam_high",
            "observation.images.cam_low",
        ],
    }
    for field, expected in expected_contract.items():
        if manifest.get(field) != expected:
            raise ValueError(f"STEAM advantage {field} provenance differs: {path}")
    if tuple(manifest.get("episodes", ())) != tuple(int(value) for value in selected):
        raise ValueError(f"STEAM advantage episode selection differs from CFG: {path}")
    expected_counts = {"rows": rows, "positive": positive, "negative": negative}
    for key, expected in expected_counts.items():
        if manifest.get(key) != expected:
            raise ValueError(f"STEAM advantage {key} provenance differs: {path}")
    for field, lower, upper in (
        ("threshold", -1.0001, 1.0001),
        ("positive_fraction_requested", 0.0, 1.0),
    ):
        value = manifest.get(field)
        if type(value) not in (int, float) or not np.isfinite(value) or not lower < value < upper:
            raise ValueError(f"invalid STEAM advantage {field}: {path}")
    return manifest


def _validate_advantage_columns(table: Any, path: Path) -> None:
    import pyarrow as pa

    for name in ("episode_index", "frame_index", "advantage"):
        column = table[name]
        expected = pa.types.is_boolean if name == "advantage" else pa.types.is_integer
        if not expected(column.type):
            raise TypeError(f"CFG advantage {name} has an invalid column type: {path}")
        if column.null_count:
            raise ValueError(f"CFG advantage {name} contains null values: {path}")


def _validate_advantages(
    path: Path,
    episode_lengths: np.ndarray,
    selected: np.ndarray,
) -> tuple[int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"CFG advantage labels do not exist: {path}; run STEAM advantage labeling first"
        )
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["episode_index", "frame_index", "advantage"])
    _validate_advantage_columns(table, path)
    episode = np.asarray(
        table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    frame = np.asarray(
        table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    labels = np.asarray(
        table["advantage"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.bool_,
    )
    if not len(episode) or not (len(episode) == len(frame) == len(labels)):
        raise ValueError(f"CFG advantage parquet is empty or misaligned: {path}")
    if np.any(episode < 0) or np.any(episode >= len(episode_lengths)):
        raise ValueError(f"CFG advantage episode_index is outside the dataset: {path}")
    if np.any(frame < 0) or np.any(frame >= episode_lengths[episode]):
        raise ValueError(f"CFG advantage frame_index is outside its episode: {path}")

    offsets = np.concatenate(([0], np.cumsum(episode_lengths, dtype=np.int64)))
    keys = offsets[episode] + frame
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"CFG advantage parquet has duplicate episode/frame keys: {path}")
    selected_mask = np.isin(episode, selected)
    expected = int(episode_lengths[selected].sum())
    selected_keys = keys[selected_mask]
    if len(selected_keys) != expected or len(np.unique(selected_keys)) != expected:
        raise ValueError(
            f"CFG advantage coverage differs from selected dataset frames: "
            f"{len(selected_keys)} != {expected}: {path}"
        )
    selected_labels = labels[selected_mask]
    positive = int(selected_labels.sum())
    return positive, expected - positive, expected


def _episode_lengths(root: Path, total_episodes: int, total_frames: int) -> np.ndarray:
    path = root / "meta" / "episodes.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read LeRobot episode metadata: {path}: {exc}") from exc
    if len(rows) != total_episodes:
        raise ValueError(f"LeRobot episode metadata count differs from info.json: {path}")
    rows.sort(key=lambda row: row.get("episode_index", -1))
    if [row.get("episode_index") for row in rows] != list(range(total_episodes)):
        raise ValueError(f"LeRobot episode indices are not contiguous: {path}")
    lengths = np.asarray(
        [_positive_int(row.get("length"), "episode length") for row in rows],
        dtype=np.int64,
    )
    if int(lengths.sum()) != total_frames:
        raise ValueError(f"LeRobot episode lengths do not sum to total_frames: {path}")
    return lengths


def _selected_episodes(value: Any, total: int) -> np.ndarray:
    if value is None:
        return np.arange(total, dtype=np.int64)
    selected = np.asarray(list(value), dtype=np.int64)
    if not len(selected) or len(np.unique(selected)) != len(selected):
        raise ValueError("CFG selected episodes must be non-empty and unique")
    if np.any(selected < 0) or np.any(selected >= total):
        raise ValueError("CFG selected episode is outside the dataset")
    return selected


def _validate_features(value: Any, root: Path) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"LeRobot features must be an object: {root}")
    missing = [key for key in REQUIRED_FEATURES if key not in value]
    if missing:
        raise ValueError(f"CFG dataset is missing required features {missing}: {root}")
    for key in ("observation.state", "action"):
        feature = value[key]
        if feature.get("dtype") != "float32" or feature.get("shape") != [14]:
            raise ValueError(f"CFG feature {key} must be float32[14]: {root}")
    for key in REQUIRED_FEATURES[1:4]:
        if value[key].get("dtype") != "video":
            raise ValueError(f"CFG feature {key} must be video: {root}")


def _validate_sidecar(root: Path) -> None:
    path = root / "meta" / "clothdojo.json"
    sidecar = _read_object(path, "ClothDojo dataset provenance")
    if sidecar.get("schema") != "clothdojo_lerobot_export":
        raise ValueError(f"unsupported ClothDojo dataset provenance: {path}")
    if sidecar.get("format_version") != 1 or sidecar.get("export_status") != "complete":
        raise ValueError(f"ClothDojo dataset export is not complete: {path}")
    if sidecar.get("training_eligible", True) is not True:
        raise ValueError(f"ClothDojo dataset is explicitly ineligible for training: {path}")


def _require_compatible_datasets(reports: list[CfgDatasetReport]) -> None:
    if not reports:
        raise ValueError("CFG compatibility checks require at least one dataset")
    reference = reports[0]
    for report in reports[1:]:
        if reference.fps != report.fps:
            raise ValueError(f"CFG dataset FPS differs: {reference.fps} != {report.fps}")
        if reference.prompt != report.prompt:
            raise ValueError("CFG dataset task prompts differ")
    infos = [
        _read_object(Path(report.dataset_path) / "meta" / "info.json", "LeRobot info")
        for report in reports
    ]
    for info in infos[1:]:
        for key in ("robot_type",):
            if infos[0].get(key) != info.get(key):
                raise ValueError(f"CFG dataset {key} differs")
        for key in REQUIRED_FEATURES:
            reference_feature = infos[0]["features"][key]
            feature = info["features"][key]
            for field in ("dtype", "shape", "names"):
                if reference_feature.get(field) != feature.get(field):
                    raise ValueError(f"CFG dataset feature differs: {key}.{field}")
    identity_fields = (
        "steam_checkpoint_config_sha256",
        "steam_checkpoint_weights_sha256",
        "steam_stride",
    )
    for report in reports[1:]:
        for field in identity_fields:
            if getattr(reference, field) != getattr(report, field):
                raise ValueError(f"CFG datasets were labeled by different STEAM {field}")
    role_thresholds: dict[str, tuple[float, float]] = {}
    for report in reports:
        manifest = _read_object(Path(report.advantage_manifest_path), "STEAM advantage provenance")
        threshold = (manifest["threshold"], manifest["positive_fraction_requested"])
        if report.role in role_thresholds and role_thresholds[report.role] != threshold:
            raise ValueError(f"CFG {report.role} shards use different STEAM label thresholds")
        role_thresholds[report.role] = threshold


def _validate_norm_binding(
    checkpoint: Path,
    norm_root: Path,
    provenance: dict[str, Any],
    expert_info: Path,
    model: Any,
) -> dict[str, Any]:
    """Keep a continuation in its initialization checkpoint's action coordinates.

    Released/converted checkpoints without a project launch manifest retain the
    original dataset-bound gate. A project SFT continuation must use its exact
    saved normalization artifacts, even when training on a newly labeled subset.
    """
    from clothdojo.rollout.artifacts import find_training_manifest, validate_sft_norm_binding

    try:
        find_training_manifest(checkpoint)
    except FileNotFoundError:
        if provenance.get("dataset_info_sha256") != _sha256(expert_info):
            raise ValueError(
                "CFG norm stats match neither the expert dataset nor a recorded SFT checkpoint"
            ) from None
        return {"source": "expert_dataset"}
    binding = validate_sft_norm_binding(
        checkpoint,
        norm_root,
        openpi_config=str(model.openpi.config_name),
        action_horizon=int(model.num_action_chunks),
    )
    return {"source": "sft_initialization", **binding}


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object: {path}")
    return payload


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} SHA-256 must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} SHA-256 must contain 64 hexadecimal characters") from exc
    return value


def _validate_init_checkpoint(value: Any) -> Path:
    root = Path(str(value)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CFG initialization checkpoint is not a directory: {root}")
    candidates = (
        root / "actor" / "model_state_dict" / "full_weights.pt",
        root / "model_state_dict" / "full_weights.pt",
        root / "full_weights.pt",
        root / "model.safetensors",
    )
    if not any(path.is_file() and path.stat().st_size > 0 for path in candidates):
        raise FileNotFoundError(f"CFG initialization checkpoint has no model weights: {root}")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
