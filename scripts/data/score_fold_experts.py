#!/usr/bin/env python3
"""Score canonical shirt-fold expert trajectories against leave-one-out consensus."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from clothdojo.evaluation.folding import aligned_fold_distance, fold_state_diagnostics
from clothdojo.trajectory import TrajectoryReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("data/collections/shirt_fold/trajectories"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluations/fold_expert_metrics_v1"),
    )
    parser.add_argument("--medoid-particles", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.medoid_particles < 16:
        parser.error("--medoid-particles must be at least 16")
    if args.workers < 1:
        parser.error("--workers must be positive")
    rows = score_collection(
        args.trajectories,
        medoid_particles=args.medoid_particles,
        workers=args.workers,
    )
    publish(rows, args.output, source=args.trajectories, medoid_particles=args.medoid_particles)


def score_collection(root: Path, *, medoid_particles: int, workers: int) -> list[dict]:
    annotation_root = root / "annotations"
    reviewed = {
        path.name.removesuffix(".segments.json")
        for path in annotation_root.glob("*.segments.json")
        if _reviewed_good(path)
    }
    records: list[dict] = []
    endpoints: dict[str, list[np.ndarray]] = defaultdict(list)
    indices: dict[str, list[int]] = defaultdict(list)

    manifests = sorted(root.glob("*/manifest.json"))
    jobs = [(path, path.parent.name in reviewed) for path in manifests]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        loaded = executor.map(_load_record, jobs, chunksize=1)
        for ordinal, (record, terminal) in enumerate(loaded, start=1):
            asset_id = record["asset_id"]
            records.append(record)
            endpoints[asset_id].append(terminal)
            indices[asset_id].append(len(records) - 1)
            if ordinal % 50 == 0 or ordinal == len(manifests):
                print(f"loaded {ordinal}/{len(manifests)} trajectories", flush=True)

    for ordinal, asset_id in enumerate(sorted(endpoints), start=1):
        states = endpoints[asset_id]
        matrix = _pairwise_approximate_distances(states, medoid_particles)
        canonical = _medoid(matrix)
        group_scale = float(
            np.median(
                [records[index]["normalization_scale_m"] for index in indices[asset_id]]
            )
        )
        for local_index, record_index in enumerate(indices[asset_id]):
            record = records[record_index]
            record["asset_expert_count"] = len(states)
            record["canonical_reference_episode_id"] = records[indices[asset_id][canonical]][
                "episode_id"
            ]
            record["canonical_fold_error"] = aligned_fold_distance(
                states[local_index], states[canonical], normalization_scale_m=group_scale
            )
            loo = _leave_one_out_medoid(matrix, local_index)
            if loo is None:
                record["loo_reference_episode_id"] = None
                record["loo_consensus_fold_error"] = None
                record["expert_consistency_percentile"] = None
            else:
                record["loo_reference_episode_id"] = records[indices[asset_id][loo]][
                    "episode_id"
                ]
                record["loo_consensus_fold_error"] = aligned_fold_distance(
                    states[local_index], states[loo], normalization_scale_m=group_scale
                )
        if ordinal % 25 == 0 or ordinal == len(endpoints):
            print(f"built consensus for {ordinal}/{len(endpoints)} assets", flush=True)

    eligible = [
        record for record in records if record["loo_consensus_fold_error"] is not None
    ]
    order = sorted(
        range(len(eligible)), key=lambda index: eligible[index]["loo_consensus_fold_error"]
    )
    denominator = max(1, len(order) - 1)
    for rank, index in enumerate(order):
        # 100 means most consistent with other demonstrations; this is a relative
        # expert percentile, not an independently calibrated success probability.
        eligible[index]["expert_consistency_percentile"] = 100.0 * (
            1.0 - rank / denominator
        )
    return records


def _load_record(job: tuple[Path, bool]) -> tuple[dict, np.ndarray]:
    manifest_path, reviewed_good = job
    trajectory = manifest_path.parent
    reader = TrajectoryReader(trajectory)
    arrays = reader.state_arrays(("particle_q", "particle_qd"), mmap=True)
    initial = np.asarray(arrays["particle_q"][0])
    terminal = np.asarray(arrays["particle_q"][-1]).copy()
    velocity = np.asarray(arrays["particle_qd"][-1])
    diagnostics = fold_state_diagnostics(initial, terminal, velocity)
    record = {
        "episode_id": trajectory.name,
        "asset_id": str(reader.manifest.metadata["asset_id"]),
        "trajectory_path": str(trajectory.resolve()),
        "action_count": reader.manifest.action_count,
        "duration_seconds": reader.manifest.action_count / reader.manifest.control_fps,
        "reviewed_good": reviewed_good,
        **asdict(diagnostics),
        "normalization_scale_m": float(np.sqrt(diagnostics.initial_footprint_area_m2)),
    }
    return record, terminal


def publish(
    rows: list[dict], output: Path, *, source: Path, medoid_particles: int
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row["asset_id"], row["episode_id"]))
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    metrics = (
        "canonical_fold_error",
        "loo_consensus_fold_error",
        "convex_footprint_ratio",
        "convex_rectangularity",
        "footprint_aspect_ratio",
        "normalized_thickness_p90",
        "speed_rms_m_s",
        "duration_seconds",
    )
    summary = {
        "schema": "clothdojo_fold_expert_metrics",
        "format_version": 1,
        "metric_status": "research_not_success_calibrated",
        "source": str(source.resolve()),
        "trajectory_count": len(rows),
        "asset_count": len({row["asset_id"] for row in rows}),
        "reviewed_good_count": sum(row["reviewed_good"] for row in rows),
        "loo_eligible_count": sum(
            row["loo_consensus_fold_error"] is not None for row in rows
        ),
        "singleton_asset_trajectory_count": sum(
            row["asset_expert_count"] == 1 for row in rows
        ),
        "medoid_selection_particle_count_max": medoid_particles,
        "distance_evaluation": "all_corresponding_particles_float64",
        "quantiles": {
            name: _quantiles([row[name] for row in rows if row[name] is not None])
            for name in metrics
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    by_asset = []
    for asset_id in sorted({row["asset_id"] for row in rows}):
        group = [row for row in rows if row["asset_id"] == asset_id]
        by_asset.append(
            {
                "asset_id": asset_id,
                "trajectory_count": len(group),
                "reviewed_good_count": sum(row["reviewed_good"] for row in group),
                "canonical_reference_episode_id": group[0][
                    "canonical_reference_episode_id"
                ],
                "loo_consensus_fold_error": _quantiles(
                    [
                        row["loo_consensus_fold_error"]
                        for row in group
                        if row["loo_consensus_fold_error"] is not None
                    ]
                ),
            }
        )
    (output / "per_asset.json").write_text(
        json.dumps(by_asset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"published {len(rows)} scores to {output}", flush=True)


def _reviewed_good(path: Path) -> bool:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("review_status") == "complete" and not value.get("segments")


def _pairwise_approximate_distances(
    states: list[np.ndarray], particle_limit: int
) -> np.ndarray:
    count = len(states)
    matrix = np.zeros((count, count), dtype=np.float64)
    if count < 2:
        return matrix
    particle_count = len(states[0])
    selection = np.linspace(
        0, particle_count - 1, min(particle_limit, particle_count), dtype=np.int64
    )
    sampled = [state[selection] for state in states]
    # Selection only chooses a medoid; exact reported errors use every particle.
    for left in range(count):
        for right in range(left + 1, count):
            value = aligned_fold_distance(
                sampled[left], sampled[right], normalization_scale_m=1.0
            )
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def _medoid(matrix: np.ndarray) -> int:
    if len(matrix) == 1:
        return 0
    # np.nanmedian is needed because the diagonal is deliberately excluded.
    scores = np.nanmedian(
        np.where(np.eye(len(matrix), dtype=bool), np.nan, matrix), axis=1
    )
    return int(np.argmin(scores))


def _leave_one_out_medoid(matrix: np.ndarray, held_out: int) -> int | None:
    count = len(matrix)
    if count == 1:
        return None
    candidates = [index for index in range(count) if index != held_out]
    if len(candidates) == 1:
        return candidates[0]
    candidate_matrix = matrix[np.ix_(candidates, candidates)]
    values = np.where(np.eye(len(candidates), dtype=bool), np.nan, candidate_matrix)
    scores = np.nanmedian(values, axis=1)
    return candidates[int(np.argmin(scores))]


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    result = np.quantile(
        np.asarray(values, dtype=np.float64),
        [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
    )
    names = ("min", "p05", "p25", "median", "p75", "p95", "max")
    return dict(zip(names, map(float, result), strict=True))


if __name__ == "__main__":
    main()
