"""Measure completion time and geometry at first confirmed success."""

from __future__ import annotations

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
from clothdojo.trajectory import TrajectoryReader


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "configs/evaluation/shirt_flatten.yaml"
TEMPLATES = REPO / "data/benchmarks/large_support_v1_20260908/templates/templates"
EXPERT_EVIDENCE = Path(__file__).with_name("expert_reference_evidence.json")
OUTPUT = Path(__file__).with_name("success_quality_evidence.json")
RUNS = {
    "White": REPO / "outputs/evaluations/white919_noaug_30k_s43_chunk_h10_balanced200_v2_large_support_60s_rescore_20260908",
    "NOCS": REPO / "outputs/evaluations/nocs919_noaug_30k_s43_chunk_h10_balanced200_v2_large_support_60s_rescore_20260908",
    "+ Mirror": REPO / "outputs/evaluations/nocs1838_mirrored_runtime_target_fullonly_40k_s43_chunk_h10_balanced200_v2_large_support_60s_rescore_20260908",
    "DAgger": REPO / "outputs/evaluations/dagger_hil800_aggregate_strict_40k_s43_chunk_h10_balanced200_v2_large_support_60s_rescore_20260908",
}


def _key(row: dict[str, object]) -> tuple[str, int, int]:
    return str(row["asset_id"]), int(row["initial_state_index"]), int(row["replicate"])


def _score(job: tuple[str, str, int]) -> tuple[float, float]:
    trajectory_path, asset_id, confirmed_step = job
    reader = TrajectoryReader(trajectory_path)
    q = np.asarray(
        reader.state_arrays(("particle_q",), mmap=True)["particle_q"][confirmed_step],
        dtype=np.float32,
    )[None]
    values = factorized_cloth_distance(
        q,
        np.zeros_like(q),
        load_canonical_template(TEMPLATES / f"{asset_id}.npz"),
        deformable_weight=load_evaluation_protocol(PROTOCOL).deformable_weight,
        symmetry=load_evaluation_protocol(PROTOCOL).symmetry,
        symmetry_center_xy=load_evaluation_protocol(PROTOCOL).target.symmetry_center_xy,
    )
    return float(values.deformable_distance[0]), float(values.rigid_distance[0])


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["asset_id"])].append(row)

    def mean(group: list[dict[str, object]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in group]))

    keys = ("ttc_seconds", "deformable_at_completion", "rigid_at_completion")
    return {
        "successful_trajectory_count": len(rows),
        "assets_with_success": len(grouped),
        "micro": {key: mean(rows, key) for key in keys},
        "asset_equal": {
            key: float(np.mean([mean(group, key) for group in grouped.values()]))
            for key in keys
        },
    }


def _policy_rows(name: str, run: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    meta_path = run / "meta.json"
    results_path = run / "results.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    results = [
        json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = Path(meta["rescore_source"]["path"])
    trajectory_by_key = {_key(row): source / str(row["path"]) for row in meta["trajectories"]}
    successful = [row for row in results if bool(row["success"])]
    jobs = [
        (str(trajectory_by_key[_key(row)]), str(row["asset_id"]), int(row["confirmed_step"]))
        for row in successful
    ]
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as executor:
        distances = list(executor.map(_score, jobs, chunksize=4))
    rows = [
        {
            "asset_id": str(row["asset_id"]),
            "initial_state_index": int(row["initial_state_index"]),
            "replicate": int(row["replicate"]),
            "confirmed_step": int(row["confirmed_step"]),
            "ttc_seconds": int(row["confirmed_step"]) / 50.0,
            "deformable_at_completion": distance[0],
            "rigid_at_completion": distance[1],
        }
        for row, distance in zip(successful, distances, strict=True)
    ]
    return rows, {
        "name": name,
        "run": str(run),
        "meta_sha256": file_sha256(meta_path),
        "results_sha256": file_sha256(results_path),
        "case_count": len(results),
    }


def _expert_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    evidence = json.loads(EXPERT_EVIDENCE.read_text(encoding="utf-8"))
    source = Path(evidence["selection"]["source"])
    successful = [row for row in evidence["results"] if bool(row["success"])]
    jobs = [
        (
            str(source / str(row["episode_id"])),
            str(row["asset_id"]),
            min(int(row["confirmed_step"]), int(row["action_count"])),
        )
        for row in successful
    ]
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as executor:
        distances = list(executor.map(_score, jobs, chunksize=4))
    rows = [
        {
            "asset_id": str(row["asset_id"]),
            "episode_id": str(row["episode_id"]),
            "confirmed_step": int(row["confirmed_step"]),
            "ttc_seconds": int(row["confirmed_step"]) / 50.0,
            "deformable_at_completion": distance[0],
            "rigid_at_completion": distance[1],
        }
        for row, distance in zip(successful, distances, strict=True)
    ]
    return rows, {
        "name": "Human Expert",
        "source": str(source),
        "expert_evidence": str(EXPERT_EVIDENCE),
        "expert_evidence_sha256": file_sha256(EXPERT_EVIDENCE),
        "original_trajectory_count": len(evidence["results"]),
    }


def main() -> None:
    protocol = load_evaluation_protocol(PROTOCOL)
    conditions: dict[str, dict[str, object]] = {}
    sources: list[dict[str, object]] = []
    for name, run in RUNS.items():
        rows, source = _policy_rows(name, run)
        sources.append(source)
        conditions[name] = {**_summarize(rows), "results": rows}
    rows, source = _expert_rows()
    sources.append(source)
    conditions["Human Expert"] = {**_summarize(rows), "results": rows}

    for name, condition in conditions.items():
        for row in condition["results"]:
            if float(row["deformable_at_completion"]) > protocol.deformable_distance_max:
                raise ValueError(f"{name} completion exceeds deformable threshold")
            if float(row["rigid_at_completion"]) > protocol.rigid_distance_max:
                raise ValueError(f"{name} completion exceeds rigid threshold")

    report = {
        "schema": "clothdojo_success_quality_v1",
        "status": "complete",
        "definition": (
            "successful trajectories only; TTC and distances are measured at the "
            "state that first confirms one continuous second inside both thresholds"
        ),
        "protocol": str(PROTOCOL),
        "protocol_fingerprint": protocol.fingerprint,
        "sources": sources,
        "conditions": conditions,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps({name: value["asset_equal"] for name, value in conditions.items()}, indent=2))


if __name__ == "__main__":
    main()
