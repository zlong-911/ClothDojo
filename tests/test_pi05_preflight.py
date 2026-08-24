from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from clothdojo.training.pi05_preflight import (
    _PI05_CONFIG_CONTRACT,
    EXPECTED_BASE_COMMIT,
    _ExpectedModelLayout,
    inspect_pi05_training,
    write_pi05_launch_manifest,
)
from clothdojo.training.rlinf import compose_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch, *, minimum: int = 65):
    dataset = tmp_path / "dataset"
    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "total_episodes": 1,
        "total_frames": minimum,
        "total_tasks": 1,
        "fps": 50,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            "observation.images.cam_high": {"dtype": "video", "shape": [8, 8, 3]},
            "observation.images.cam_left_wrist": {
                "dtype": "video",
                "shape": [8, 8, 3],
            },
            "observation.images.cam_right_wrist": {
                "dtype": "video",
                "shape": [8, 8, 3],
            },
        },
    }
    sidecar = {
        "schema": "clothdojo_lerobot_export",
        "format_version": 1,
        "export_status": "complete",
        "selection": {"min_segment_frames": minimum},
        "counts": {
            "output_episodes": 1,
            "output_frames": minimum,
            "min_segment_frames": minimum,
        },
        "episodes": [{"episode_index": 0, "length": minimum}],
    }
    _write_json(dataset / "meta/info.json", info)
    _write_json(dataset / "meta/clothdojo.json", sidecar)
    (dataset / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "flatten the shirt"}) + "\n",
        encoding="utf-8",
    )
    (dataset / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": minimum}) + "\n",
        encoding="utf-8",
    )

    norm = tmp_path / "norm"
    vector = [0.0] * 14
    _write_json(
        norm / "norm_stats.json",
        {
            "norm_stats": {
                "state": dict.fromkeys(("mean", "std", "q01", "q99"), vector),
                "actions": dict.fromkeys(("mean", "std", "q01", "q99"), vector),
            }
        },
    )
    data_kwargs = {
        "adapt_to_pi": False,
        "extra_delta_transform": True,
        "default_prompt": "Flatten the shirt.",
    }
    _write_json(
        norm / "clothdojo.json",
        {
            "schema": "clothdojo_openpi_norm_stats",
            "format_version": 1,
            "dataset_info_sha256": _sha256(dataset / "meta/info.json"),
            "dataset_sidecar_sha256": _sha256(dataset / "meta/clothdojo.json"),
            "config_name": "pi05_aloha_robotwin",
            "data_kwargs": data_kwargs,
            "action_horizon": 50,
            "frames": minimum,
            "episodes": 1,
        },
    )

    checkpoint = tmp_path / "snapshots" / EXPECTED_BASE_COMMIT
    _write_json(
        checkpoint / "config.json",
        {
            "type": "pi05",
            "dtype": "float32",
            "max_action_dim": 32,
            "max_state_dim": 32,
            "chunk_size": 50,
        },
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "embedding.weight": torch.zeros(4, 2),
            "projection.weight": torch.zeros(2, 2),
        },
        checkpoint / "model.safetensors",
    )
    weights = checkpoint / "model.safetensors"
    blob = tmp_path / ("b" * 64)
    weights.rename(blob)
    weights.symlink_to(blob)
    monkeypatch.setattr(
        "clothdojo.training.pi05_preflight._expected_pi05_layout",
        lambda cfg: _ExpectedModelLayout(
            shapes={
                "embedding.weight": (4, 2),
                "lm_head.weight": (4, 2),
                "projection.weight": (2, 2),
            },
            tied_groups=(("embedding.weight", "lm_head.weight"),),
        ),
    )
    output = tmp_path / "output"
    for name, value in {
        "CLOTHDOJO_LEROBOT_DATASET": dataset,
        "CLOTHDOJO_PI05_CHECKPOINT": checkpoint,
        "CLOTHDOJO_PI05_NORM_STATS": norm,
        "CLOTHDOJO_TRAIN_OUTPUT": output,
    }.items():
        monkeypatch.setenv(name, str(value))
    config_path = Path(__file__).resolve().parents[1] / "configs/training/pi05_aloha.yaml"
    cfg = compose_config(config_path)
    return cfg, dataset, norm, checkpoint, output


def test_pi05_preflight_returns_json_manifest_without_loading_weights(
    tmp_path: Path, monkeypatch
) -> None:
    cfg, _, _, _, output = _fixture(tmp_path, monkeypatch)

    manifest = inspect_pi05_training(cfg)

    json.dumps(manifest)
    assert manifest["compatible"] is True
    assert manifest["dataset"]["min_segment_frames"] == 65
    assert manifest["dataset"]["task"] == "flatten the shirt"
    assert manifest["checkpoint"]["tensor_count"] == 2
    assert manifest["checkpoint"]["weights_blob_sha256"] == "b" * 64
    assert manifest["checkpoint"]["tied_aliases"] == [
        {
            "omitted_key": "lm_head.weight",
            "stored_key": "embedding.weight",
            "semantics": "same_parameter",
        }
    ]

    path = write_pi05_launch_manifest(cfg, manifest)
    assert path == output / "pi05_launch_manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError, match="not unique"):
        write_pi05_launch_manifest(cfg, manifest)


def test_pi05_preflight_rejects_short_export(tmp_path: Path, monkeypatch) -> None:
    cfg, _, _, _, _ = _fixture(tmp_path, monkeypatch, minimum=64)

    with pytest.raises(ValueError, match="min_segment_frames must be >= 65"):
        inspect_pi05_training(cfg)


def test_pi05_preflight_rejects_norm_dataset_digest_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    cfg, _, norm, _, _ = _fixture(tmp_path, monkeypatch)
    provenance = json.loads((norm / "clothdojo.json").read_text(encoding="utf-8"))
    provenance["dataset_info_sha256"] = "0" * 64
    _write_json(norm / "clothdojo.json", provenance)

    with pytest.raises(ValueError, match="dataset_info_sha256"):
        inspect_pi05_training(cfg)


def test_pi05_preflight_does_not_silence_non_alias_missing_key(
    tmp_path: Path, monkeypatch
) -> None:
    cfg, _, _, checkpoint, _ = _fixture(tmp_path, monkeypatch)
    save_file(
        {"embedding.weight": torch.zeros(4, 2)},
        (checkpoint / "model.safetensors").resolve(),
    )

    with pytest.raises(ValueError, match=r"missing=.*projection\.weight"):
        inspect_pi05_training(cfg)


@pytest.mark.parametrize(("path", "expected"), _PI05_CONFIG_CONTRACT)
def test_pi05_preflight_rejects_every_contract_field_drift(
    tmp_path: Path,
    monkeypatch,
    path: str,
    expected: object,
) -> None:
    cfg, _, _, _, _ = _fixture(tmp_path, monkeypatch)
    if expected is None:
        drifted: object = "enabled"
    elif isinstance(expected, bool):
        drifted = not expected
    elif isinstance(expected, int):
        drifted = expected + 1
    elif isinstance(expected, float):
        drifted = expected * 2
    else:
        drifted = f"{expected}-drifted"
    OmegaConf.update(cfg, path, drifted)

    with pytest.raises(ValueError, match=re.escape(f"contract drift at {path}")):
        inspect_pi05_training(cfg)
