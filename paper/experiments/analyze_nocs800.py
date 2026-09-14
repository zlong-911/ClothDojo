#!/usr/bin/env python3
"""Reproduce the paired White-919/NOCS-919 EVAL-800 evidence (CPU only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper/experiments/nocs800_evidence.json"
SEED, DRAWS = 20260905, 100_000
WHITE = REPO / "outputs/evaluations/white919_noaug_balanced200_20260902"
NOCS = REPO / "outputs/evaluations/nocs919_noaug_balanced200_20260902"
PAIR = REPO / "outputs/evaluations/eval919_balanced200_20260902"
FROZEN = PAIR / "runtime/frozen_runtime/src/clothdojo"
STATES_META = REPO / "data/benchmarks/balanced200_20260901/states/meta.json"
REPEATS = REPO / "outputs/evaluations/balanced200_repeatability_20260901"
OLD_WHITE = REPO / "outputs/evaluations/white919_noaug_20260831/runtime"
WHITE_PREFLIGHT = REPO / "outputs/training/pi05_white919_noaug_promptfix_b128_s43_20260831/preflight.json"
NOCS_PREFLIGHT = REPO / "outputs/training/pi05_nocs919_noaug_promptfix_b128_s43_20260831/preflight.json"

KEY = ("asset_id", "initial_state_index", "replicate")
METRICS = {
    "best_deformable_distance": ("best_observed", "deformable_distance"),
    "best_rigid_distance": ("best_observed", "rigid_distance"),
    "best_combined_distance": ("best_observed", "combined_distance"),
    "terminal_deformable_distance": ("terminal", "deformable_distance"),
    "terminal_rigid_distance": ("terminal", "rigid_distance"),
    "terminal_combined_distance": ("terminal", "combined_distance"),
}
PROTOCOL_FIELDS = (
    "airborne_metric", "benchmark", "collection_protocol", "execution",
    "initial_states_sha256", "observation", "parallelism", "protocol",
    "scenario", "scoring_protocol", "templates",
)
POLICY_FIELDS = (
    "action_dim", "action_horizon", "backend", "batched_inference",
    "camera_names", "checkpoint_validation", "image_preprocessing",
    "inference_precision", "num_steps", "openpi_config", "protocol_version",
    "rtc_guidance", "source_framework", "transport_image_size",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in KEY)


def indexed(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {row_key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate case identity")
    return result


def integrity(meta: dict[str, Any], rows: list[dict[str, Any]], split: dict[str, str]) -> dict[str, Any]:
    counts = Counter(row["asset_id"] for row in rows)
    result_index, trajectory_index = indexed(rows), indexed(meta["trajectories"])
    identities_match = set(result_index) == set(trajectory_index) and all(
        result_index[k]["initial_state_seed"] == trajectory_index[k]["seed"]
        and result_index[k]["policy_seed"] == trajectory_index[k]["policy_seed"]
        for k in result_index
    )
    checks = {
        "complete_800": meta["status"] == meta["phase"] == "complete" and len(rows) == meta["case_count"] == 800,
        "unique_200_assets_x4": len(counts) == 200 and set(counts.values()) == {4},
        "four_state_indices_each": all(
            {r["initial_state_index"] for r in rows if r["asset_id"] == asset} == {0, 1, 2, 3}
            for asset in counts
        ),
        "split_100_seen_100_unseen": Counter(split[a] for a in counts) == {"seen": 100, "unseen": 100},
        "manifest_case_and_seeds_match_results": identities_match,
    }
    return {"passed": all(checks.values()), "checks": checks}


def subset(rows: list[dict[str, Any]], split: dict[str, str], scope: str) -> list[dict[str, Any]]:
    return rows if scope == "all" else [r for r in rows if split[r["asset_id"]] == scope]


def summary(rows: list[dict[str, Any]], split: dict[str, str]) -> dict[str, Any]:
    output = {}
    for scope in ("all", "seen", "unseen"):
        selected = subset(rows, split, scope)
        end = sum(bool(r["success_at_end"]) for r in selected)
        ever = sum(bool(r["success"]) for r in selected)
        means = {
            name: statistics.fmean(float(r[group][field]) for r in selected)
            for name, (group, field) in METRICS.items()
        }
        means["airborne_seconds"] = statistics.fmean(float(r["airborne_seconds"]) for r in selected)
        output[scope] = {
            "assets": len({r["asset_id"] for r in selected}), "cases": len(selected),
            "terminal_success": [end, end / len(selected)], "ever_success": [ever, ever / len(selected)],
            "means": means,
        }
    return output


def deltas(white: list[dict[str, Any]], nocs: list[dict[str, Any]], split: dict[str, str]) -> dict[str, Any]:
    wi, ni = indexed(white), indexed(nocs)
    output = {}
    for scope in ("all", "seen", "unseen"):
        keys = [k for k in wi if scope == "all" or split[k[0]] == scope]
        avg = lambda fn: statistics.fmean(fn(ni[k]) - fn(wi[k]) for k in keys)
        output[scope] = {
            "terminal_success_rate": avg(lambda r: float(bool(r["success_at_end"]))),
            "ever_success_rate": avg(lambda r: float(bool(r["success"]))),
            "means": {
                **{name: avg(lambda r, g=g, f=f: float(r[g][f])) for name, (g, f) in METRICS.items()},
                "airborne_seconds": avg(lambda r: float(r["airborne_seconds"])),
            },
        }
    return output


def bootstrap(white: list[dict[str, Any]], nocs: list[dict[str, Any]], split: dict[str, str], draws: int, seed: int) -> dict[str, Any]:
    def asset_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
        assets = sorted({r["asset_id"] for r in rows})
        return {a: statistics.fmean(float(r["success_at_end"]) for r in rows if r["asset_id"] == a) for a in assets}

    wr, nr = asset_rates(white), asset_rates(nocs)
    diffs = {a: nr[a] - wr[a] for a in wr}
    values = {
        scope: np.asarray([diffs[a] for a in sorted(diffs) if scope == "all" or split[a] == scope])
        for scope in ("all", "seen", "unseen")
    }
    rng = np.random.default_rng(seed)
    seen, unseen = values["seen"], values["unseen"]
    sampled = {scope: [] for scope in values}
    remaining = draws
    while remaining:
        batch = min(10_000, remaining)
        ss = seen[rng.integers(0, len(seen), (batch, len(seen)))].mean(1)
        uu = unseen[rng.integers(0, len(unseen), (batch, len(unseen)))].mean(1)
        sampled["seen"].append(ss); sampled["unseen"].append(uu); sampled["all"].append((ss + uu) / 2)
        remaining -= batch
    result = {}
    for scope in sampled:
        samples = np.concatenate(sampled[scope])
        point = (seen.mean() + unseen.mean()) / 2 if scope == "all" else values[scope].mean()
        result[scope] = {"assets": 200 if scope == "all" else 100, "point_estimate": float(point), "ci95_percentile": np.quantile(samples, [0.025, 0.975]).tolist()}
    return {
        "status": "computed_strict_case_paired", "contrast": "nocs_minus_white",
        "unit": "asset", "states_retained_per_asset": 4, "seed": seed, "draws": draws,
        "method": "paired percentile bootstrap; assets resampled within seen/unseen, splits equally weighted",
        **result,
    }


def visual_audit() -> tuple[dict[str, Any], list[Path]]:
    paths = {
        "launcher": PAIR / "runtime/eval800.py", "local_plan": PAIR / "runtime/white919-local-plan.json",
        "remote_plan": PAIR / "runtime/white919-remote-plan.json",
        "collection": FROZEN / "rollout/trajectory_collection.py", "scene": FROZEN / "sim/scene.py",
        "rendering": FROZEN / "rendering.py", "old_scene": OLD_WHITE / "frozen_runtime/src/clothdojo/sim/scene.py",
        "old_rendering": OLD_WHITE / "frozen_runtime/src/clothdojo/rendering.py",
        "old_collection": OLD_WHITE / "frozen_runtime/src/clothdojo/rollout/trajectory_collection.py",
        "old_smoke_code": OLD_WHITE / "smoke.py", "old_smoke_result": OLD_WHITE / "smoke.json",
        "old_smoke_image": OLD_WHITE / "white_observation.png",
    }
    launcher, collection = paths["launcher"].read_text(), paths["collection"].read_text()
    scene, rendering = paths["scene"].read_text(), paths["rendering"].read_text()
    plans = [load(paths["local_plan"]), load(paths["remote_plan"])]
    manifests = [load(p) for p in sorted((WHITE / "trajectories").glob("*/manifest.json"))]
    visual_pairs = Counter((m["metadata"]["cloth_visual"], m["metadata"]["observation"]["cloth_visual"]) for m in manifests)
    media = [p for root in (WHITE, PAIR) for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".npy"}]
    checks = {
        "launcher_maps_white_and_copies_to_spec": '"white919": {' in launcher and '"cloth_visual": "white"' in launcher and 'cloth_visual=config["cloth_visual"]' in launcher,
        "both_persisted_worker_plans_white": all(p["spec"]["cloth_visual"] == "white" for p in plans),
        "plans_cover_200_assets_14_workers": len({j["asset_id"] for p in plans for j in p["jobs"]}) == 200 and sum(len(p["gpus"]) for p in plans) == 14,
        "worker_passes_spec_visual_to_scene": "cloth_visual=spec.cloth_visual" in collection,
        "scene_white_branch_sets_rgb255": 'if cloth_visual == "white":' in scene and 'texture[..., :3] = 255' in scene,
        "metadata_helper_hardcodes_nocs": 'def policy_observation_settings()' in rendering and '"cloth_visual": "nocs"' in rendering,
        "all_800_manifests_white_with_nocs_residue": visual_pairs == {("white", "nocs"): 800},
        "old_white_smoke_uses_identical_frozen_sources": all(sha(paths[n]) == sha(paths["old_" + n]) for n in ("scene", "rendering", "collection")),
        "old_white_smoke_passed_same_checkpoint": load(paths["old_smoke_result"])["passed"] and load(paths["old_smoke_result"])["policy"]["checkpoint_sha256"] == load(WHITE / "meta.json")["policy"]["checkpoint_sha256"],
    }
    return {
        "verdict": "white_scene_supported_by_frozen_execution_provenance_not_direct_eval800_pixel_audit",
        "checks": checks, "all_checks_passed": all(checks.values()),
        "eval800_retained_policy_input_pixels": [rel(p) for p in media],
        "metadata_conflict_explained": "policy_observation_settings() hard-codes nocs while scene appearance uses spec.cloth_visual",
        "corroborating_pixel_artifact": {
            "path": rel(paths["old_smoke_image"]), "sha256": sha(paths["old_smoke_image"]),
            "scope": "earlier White smoke using byte-identical frozen scene/rendering/collection sources; not an EVAL-800 frame",
            "manual_inspection": "three-camera image shows a white/gray garment rather than NOCS coordinate colors",
        },
        "claim_limit": "Report +34.125 pp as strict case-paired evidence with White execution provenance; not an unconditionally pixel-proven visual-only causal effect.",
    }, list(paths.values())


def repeat_audit(split: dict[str, str]) -> tuple[dict[str, Any], list[Path]]:
    paths, rows, metas = [], {}, {}
    for name in ("r1", "r2", "r3"):
        base = REPEATS / "conditions" / name
        paths += [base / "meta.json", base / "results.jsonl"]
        metas[name], rows[name] = load(paths[-2]), jsonl(paths[-1])
    summaries = {name: summary(rows[name], split) for name in rows}
    base = indexed(rows["r1"])
    identity = {name: set(base) == set(indexed(rows[name])) and all(base[k][f] == indexed(rows[name])[k][f] for k in base for f in ("initial_state_seed", "policy_seed")) for name in ("r2", "r3")}
    rates = [summaries[n]["all"]["terminal_success"][1] for n in rows]
    published = {r["condition_id"]: r for r in load(REPEATS / "analysis.json")["paired_summary"]["conditions"]}
    crosscheck = {n: summaries[n]["all"]["terminal_success"][0] == published[n]["success_at_end_count"] for n in rows}
    extra = [REPEATS / "analysis.json", REPEATS / "experiment.json", REPEATS / "results.md"]
    return {
        "by_round": summaries, "case_and_seed_identity": identity,
        "same_protocol": all(all(metas["r1"][f] == metas[n][f] for f in PROTOCOL_FIELDS) for n in ("r2", "r3")),
        "same_checkpoint": len({metas[n]["policy"]["checkpoint_sha256"] for n in metas}) == 1,
        "published_counts_crosscheck": crosscheck,
        "terminal_rate_mean": statistics.fmean(rates), "terminal_rate_range": max(rates) - min(rates),
        "terminal_rate_sample_sd": statistics.stdev(rates),
        "limit": "30k NOCS-1838 checkpoint; three hosts confound host and round and do not bound future-run variation.",
    }, paths + extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args()
    if args.draws <= 0:
        parser.error("--draws must be positive")
    split = {a["asset_id"]: a["split"] for a in load(STATES_META)["assets"]}
    wm, nm = load(WHITE / "meta.json"), load(NOCS / "meta.json")
    wr, nr = jsonl(WHITE / "results.jsonl"), jsonl(NOCS / "results.jsonl")
    wi, ni = indexed(wr), indexed(nr)
    pair_checks = {
        "exact_case_keys": set(wi) == set(ni),
        "exact_initial_and_policy_seeds": set(wi) == set(ni) and all(wi[k][f] == ni[k][f] for k in wi for f in ("initial_state_seed", "policy_seed")),
        "shared_evaluation_protocol": all(wm[f] == nm[f] for f in PROTOCOL_FIELDS),
        "shared_policy_runtime_fields": all(wm["policy"][f] == nm["policy"][f] for f in POLICY_FIELDS),
    }
    integrity_checks = {"white": integrity(wm, wr, split), "nocs": integrity(nm, nr, split)}
    pair_supported = all(pair_checks.values()) and all(item["passed"] for item in integrity_checks.values())
    paired_bootstrap = (
        bootstrap(wr, nr, split, args.draws, args.seed)
        if pair_supported
        else {"status": "not_computed", "reason": "case pairing or input integrity check failed"}
    )
    visual, visual_sources = visual_audit()
    repeats, repeat_sources = repeat_audit(split)
    wp, npf = load(WHITE_PREFLIGHT), load(NOCS_PREFLIGHT)
    training_counts_match = all(
        wp["dataset"][field] == npf["dataset"][field] for field in ("episodes", "frames")
    )
    training = {
        "white_episodes_frames": [wp["dataset"]["episodes"], wp["dataset"]["frames"]],
        "nocs_episodes_frames": [npf["dataset"]["episodes"], npf["dataset"]["frames"]],
        "episode_and_frame_counts_match": training_counts_match,
        "seed": wp["training_contract"]["seed"],
        "matched_recipe": training_counts_match and wp["training_contract"] == npf["training_contract"] and wp["action_offset_loss"] == npf["action_offset_loss"],
        "limit": "Preflights do not prove byte-identical state/action payloads; visual dataset manifests intentionally differ.",
    }
    base_sources = [Path(__file__), STATES_META, WHITE / "meta.json", WHITE / "results.jsonl", NOCS / "meta.json", NOCS / "results.jsonl", WHITE_PREFLIGHT, NOCS_PREFLIGHT, PAIR / "configs/white919.yaml", PAIR / "configs/nocs919.yaml"]
    source_paths = list(dict.fromkeys(base_sources + visual_sources + repeat_sources))
    report = {
        "schema": "clothdojo_nocs800_evidence_v2", "comparison": "NOCS-919 minus White-919",
        "integrity": integrity_checks,
        "case_pairing": {"supported": pair_supported, "checks": pair_checks},
        "visual_identity_audit": visual, "training_pairing": training,
        "condition_summaries": {"white": summary(wr, split), "nocs": summary(nr, split)},
        "paired_deltas_nocs_minus_white": deltas(wr, nr, split),
        "terminal_success_asset_bootstrap": paired_bootstrap,
        "repeatability_800_three_rounds": repeats,
        "limits": ["Single training seed and one EVAL-800 round per White/NOCS condition.", "Bootstrap conditions on observed runs and fixed seed identities; it excludes training-seed and future-run/host variation.", "Distance/airborne means are descriptive; multiple-comparison adjustment: none.", "This is a data-agent panel, not downstream RGB evaluation."],
        "source_files": [{"path": rel(p), "sha256": sha(p)} for p in source_paths],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
