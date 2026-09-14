"""Score the 890 original human demonstrations under the 60-second protocol."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from clothdojo.evaluation.metric import (
    factorized_cloth_distance,
    file_sha256,
    load_canonical_template,
)
from clothdojo.evaluation.protocol import load_evaluation_protocol


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "data/collections/shirt_flatten/trajectories"
AUGMENTATION = REPO / "data/collections/shirt_flatten_mirrored/augmentation.json"
TEMPLATE_COLLECTION = REPO / "data/benchmarks/large_support_v1_20260908/templates"
TEMPLATES = TEMPLATE_COLLECTION / "templates"
PROTOCOL = REPO / "configs/evaluation/shirt_flatten.yaml"
OUTPUT = Path(__file__).with_name("expert_reference_evidence.json")


def _score(episode_id: str) -> dict[str, object]:
    trajectory = SOURCE / episode_id
    manifest_path = trajectory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    if manifest["status"] != "complete" or metadata.get("source") != "vr_teleop":
        raise ValueError(f"not a complete human demonstration: {episode_id}")
    if "__mirror" in episode_id:
        raise ValueError(f"mirrored trajectory selected: {episode_id}")

    protocol = load_evaluation_protocol(PROTOCOL)
    asset_id = str(metadata["asset_id"])
    template_path = TEMPLATES / f"{asset_id}.npz"
    template = load_canonical_template(template_path)
    action_count = int(manifest["action_count"])
    if int(manifest["state_count"]) != action_count + 1:
        raise ValueError(f"state/action alignment differs: {episode_id}")
    if int(manifest["control_fps"]) != protocol.control_fps:
        raise ValueError(f"control rate differs: {episode_id}")

    # Score on the exact 10-Hz grid used by policy evaluation. Demonstrations
    # longer than 60 s are truncated; after a shorter demonstration ends, its
    # physical terminal state is held fixed through 60 s.
    sample_steps = np.arange(
        0, protocol.horizon_steps + 1, protocol.metric_stride, dtype=np.int32
    )
    source_steps = np.minimum(sample_steps, action_count)
    with np.load(trajectory / "states.npz", allow_pickle=False) as archive:
        positions = np.asarray(archive["particle_q"][source_steps], dtype=np.float32)

    curve = np.empty((len(sample_steps), 3), dtype=np.float32)
    for start in range(0, len(sample_steps), 32):
        stop = min(start + 32, len(sample_steps))
        q = positions[start:stop]
        values = factorized_cloth_distance(
            q,
            np.zeros_like(q),
            template,
            deformable_weight=protocol.deformable_weight,
            symmetry=protocol.symmetry,
            symmetry_center_xy=protocol.target.symmetry_center_xy,
        )
        curve[start:stop] = np.stack(
            (
                values.deformable_distance,
                values.rigid_distance,
                values.combined_distance,
            ),
            axis=1,
        )

    confirmed = protocol.continuous_success_mask(curve[:, 0], curve[:, 1])
    success = bool(confirmed.any())
    confirmed_step = int(np.argmax(confirmed)) * protocol.metric_stride if success else None
    ttc_seconds = (
        confirmed_step / protocol.control_fps if success else protocol.horizon_seconds
    )
    n_auc = float(
        np.trapz(curve[:, 2], dx=protocol.metric_stride) / protocol.horizon_steps
    )
    return {
        "episode_id": episode_id,
        "asset_id": asset_id,
        "manifest_sha256": file_sha256(manifest_path),
        "states_sha256": file_sha256(trajectory / "states.npz"),
        "template_sha256": file_sha256(template_path),
        "action_count": action_count,
        "duration_seconds": action_count / protocol.control_fps,
        "horizon_treatment": (
            "terminal_hold" if action_count < protocol.horizon_steps else
            "truncated" if action_count > protocol.horizon_steps else "exact"
        ),
        "success": success,
        "confirmed_step": confirmed_step,
        "ttc_seconds_failure_is_60": float(ttc_seconds),
        "n_auc_60s": n_auc,
    }


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()

    augmentation = json.loads(AUGMENTATION.read_text(encoding="utf-8"))
    ids = [str(pair["source_id"]) for pair in augmentation["trajectory_pairs"]]
    if len(ids) != 890 or len(set(ids)) != 890 or any("__mirror" in item for item in ids):
        raise ValueError("augmentation manifest does not identify 890 unique originals")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_score, ids, chunksize=1))
    rows.sort(key=lambda row: str(row["episode_id"]))

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["asset_id"])].append(row)
    if len(grouped) != 182:
        raise ValueError(f"expected 182 expert assets, got {len(grouped)}")

    macro_ttc = float(np.mean([_mean(group, "ttc_seconds_failure_is_60") for group in grouped.values()]))
    macro_auc = float(np.mean([_mean(group, "n_auc_60s") for group in grouped.values()]))
    protocol = load_evaluation_protocol(PROTOCOL)
    report = {
        "schema": "clothdojo_expert_reference_60s_v1",
        "status": "complete",
        "selection": {
            "definition": "all original complete human demonstrations; review annotations ignored",
            "trajectory_count": len(rows),
            "asset_count": len(grouped),
            "excluded_mirror": 890,
            "excluded_hil_corrections": "source collection contains none",
            "excluded_model_rollouts": "source collection contains none",
            "source": str(SOURCE),
            "augmentation_manifest": str(AUGMENTATION),
            "augmentation_sha256": file_sha256(AUGMENTATION),
        },
        "scoring": {
            "protocol": str(PROTOCOL),
            "protocol_fingerprint": protocol.fingerprint,
            "templates": str(TEMPLATES),
            "templates_meta_sha256": file_sha256(TEMPLATE_COLLECTION / "meta.json"),
            "horizon_seconds": protocol.horizon_seconds,
            "metric_hz": protocol.metric_hz,
            "continuous_success_seconds": protocol.success_window_seconds,
            "short_trajectory_treatment": "hold the physical terminal state to 60 s",
            "long_trajectory_treatment": "truncate at 60 s",
            "ttc_failure_value_seconds": protocol.horizon_seconds,
            "aggregation_for_paper": "asset-equal mean: trajectory mean within asset, then mean over assets",
        },
        "counts": {
            "success": sum(bool(row["success"]) for row in rows),
            "terminal_hold": sum(row["horizon_treatment"] == "terminal_hold" for row in rows),
            "truncated": sum(row["horizon_treatment"] == "truncated" for row in rows),
            "exact": sum(row["horizon_treatment"] == "exact" for row in rows),
        },
        "paper_metrics": {
            "mean_ttc_seconds": macro_ttc,
            "n_auc_60s": macro_auc,
        },
        "micro_diagnostics": {
            "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
            "mean_ttc_seconds": _mean(rows, "ttc_seconds_failure_is_60"),
            "n_auc_60s": _mean(rows, "n_auc_60s"),
        },
        "results": rows,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps({key: report[key] for key in ("selection", "counts", "paper_metrics", "micro_diagnostics")}, indent=2))


if __name__ == "__main__":
    main()
