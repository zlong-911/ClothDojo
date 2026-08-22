"""Static preflight checks for ClothDojo's offline CFG mixture."""

from __future__ import annotations

import hashlib
import json
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
    weight: float
    episodes: int
    frames: int
    fps: int
    advantage_path: str
    positive_frames: int
    negative_frames: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_cfg_training(
    config_path: str | Path,
    overrides: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Validate both labeled datasets referenced by one CFG config."""

    from clothdojo.training.rlinf import compose_config

    cfg = compose_config(config_path, overrides)
    if cfg.actor.model.model_type != "clothdojo_cfg_model":
        raise ValueError("CFG config must use model_type=clothdojo_cfg_model")
    openpi_data = cfg.actor.model.get("openpi_data", {})
    if openpi_data.get("adapt_to_pi") is not False:
        raise ValueError("Piper CFG requires actor.model.openpi_data.adapt_to_pi=false")
    norm_root = Path(str(openpi_data.get("norm_stats_path", ""))).expanduser().resolve()
    if not (norm_root / "norm_stats.json").is_file():
        raise FileNotFoundError(f"CFG norm_stats.json does not exist: {norm_root}")
    norm_provenance = _read_object(norm_root / "clothdojo.json", "CFG norm-stat provenance")
    if norm_provenance.get("schema") != "clothdojo_openpi_norm_stats":
        raise ValueError(f"unsupported CFG norm-stat provenance: {norm_root}")
    if norm_provenance.get("config_name") != cfg.actor.model.openpi.config_name:
        raise ValueError("CFG norm stats use a different OpenPI config")
    model_path = _validate_init_checkpoint(cfg.actor.model.model_path)

    data_cfg = cfg.data
    if data_cfg.get("balance_dataset_weights", True):
        raise ValueError(
            "CFG requires balance_dataset_weights=false so configured weights are "
            "the actual source probabilities"
        )
    entries = list(data_cfg.get("train_data_paths", ()))
    if len(entries) != 2:
        raise ValueError("CFG requires exactly one sft and one rollout dataset")
    roles = [str(entry.get("type", "")) for entry in entries]
    if sorted(roles) != sorted(CFG_ROLES):
        raise ValueError(f"CFG dataset types must be {CFG_ROLES}, got {roles}")

    tag = data_cfg.get("advantage_tag")
    tag = None if tag is None else str(tag)
    reports = [
        inspect_cfg_dataset(
            entry["dataset_path"],
            role=str(entry["type"]),
            weight=float(entry.get("weight", 1.0)),
            advantage_tag=tag,
            episodes=entry.get("episodes"),
        )
        for entry in entries
    ]
    _require_compatible_datasets(reports)
    expert = next(report for report in reports if report.role == "sft")
    expert_info = Path(expert.dataset_path) / "meta" / "info.json"
    if norm_provenance.get("dataset_info_sha256") != _sha256(expert_info):
        raise ValueError("CFG norm stats were not computed from the expert dataset")
    if sum(report.positive_frames for report in reports) == 0:
        raise ValueError("CFG mixture has no positive advantage labels")
    if sum(report.negative_frames for report in reports) == 0:
        raise ValueError("CFG mixture has no negative advantage labels")

    weights = np.asarray([report.weight for report in reports], dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError(f"CFG dataset weights must be finite and positive: {weights}")
    probabilities = weights / weights.sum()
    return {
        "compatible": True,
        "config": str(Path(config_path).expanduser().resolve()),
        "model_type": str(cfg.actor.model.model_type),
        "model_path": str(model_path),
        "norm_stats_path": str(norm_root),
        "advantage_tag": tag,
        "balance_dataset_weights": False,
        "sampling_probabilities": {
            report.role: float(probability)
            for report, probability in zip(reports, probabilities, strict=True)
        },
        "datasets": [report.to_json_dict() for report in reports],
    }


def inspect_cfg_dataset(
    dataset_path: str | Path,
    *,
    role: str,
    weight: float,
    advantage_tag: str | None,
    episodes: Any = None,
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
    lengths = _episode_lengths(root, total_episodes, total_frames)
    selected = _selected_episodes(episodes, total_episodes)

    filename = f"advantages_{advantage_tag}.parquet" if advantage_tag else "advantages.parquet"
    advantage_path = root / "meta" / filename
    positive, negative, selected_frames = _validate_advantages(advantage_path, lengths, selected)
    return CfgDatasetReport(
        role=role,
        dataset_path=str(root),
        weight=float(weight),
        episodes=len(selected),
        frames=selected_frames,
        fps=fps,
        advantage_path=str(advantage_path),
        positive_frames=positive,
        negative_frames=negative,
    )


def _validate_advantages(
    path: Path,
    episode_lengths: np.ndarray,
    selected: np.ndarray,
) -> tuple[int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"CFG advantage labels do not exist: {path}; run STEAM advantage labeling first"
        )
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["episode_index", "frame_index", "advantage"])
    if not pa.types.is_boolean(table.schema.field("advantage").type):
        raise TypeError(f"CFG advantage column must be boolean: {path}")
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
    if reports[0].fps != reports[1].fps:
        raise ValueError(f"CFG dataset FPS differs: {reports[0].fps} != {reports[1].fps}")
    infos = [
        _read_object(Path(report.dataset_path) / "meta" / "info.json", "LeRobot info")
        for report in reports
    ]
    for key in ("robot_type",):
        if infos[0].get(key) != infos[1].get(key):
            raise ValueError(f"CFG dataset {key} differs")
    for key in REQUIRED_FEATURES:
        left = infos[0]["features"][key]
        right = infos[1]["features"][key]
        for field in ("dtype", "shape", "names"):
            if left.get(field) != right.get(field):
                raise ValueError(f"CFG dataset feature differs: {key}.{field}")


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


def _validate_init_checkpoint(value: Any) -> Path:
    root = Path(str(value)).expanduser().resolve()
    candidates = (
        root / "actor" / "model_state_dict" / "full_weights.pt",
        root / "model_state_dict" / "full_weights.pt",
        root / "full_weights.pt",
        root / "model.safetensors",
    )
    if root.is_file():
        candidates = (root,)
    if not any(path.is_file() and path.stat().st_size > 0 for path in candidates):
        raise FileNotFoundError(f"CFG initialization checkpoint has no model weights: {root}")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
