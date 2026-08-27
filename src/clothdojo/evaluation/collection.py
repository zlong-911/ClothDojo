"""Trajectory-first multi-GPU evaluation over a fixed initial-state pool."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from clothdojo.assets import GarmentLibrary
from clothdojo.evaluation.metric import file_sha256, load_canonical_template
from clothdojo.evaluation.protocol import load_evaluation_protocol
from clothdojo.evaluation.results import load_evaluation_condition, write_evaluation_cases
from clothdojo.evaluation.scoring import score_evaluation_trajectory
from clothdojo.evaluation.templates import (
    garment_topology_sha256,
    resolve_canonical_templates,
    shirt_flatten_physics_sha256,
)
from clothdojo.rendering import policy_observation_settings
from clothdojo.rollout.initial_states import (
    InitialStatePool,
    load_initial_state_collection,
    validate_pool_compatibility,
)
from clothdojo.rollout.policy import RemotePolicy
from clothdojo.rollout.scenario import load_rollout_scenario
from clothdojo.rollout.scheduling import (
    policy_endpoints_for_workers,
    select_assets,
    validate_policy_identities,
)
from clothdojo.rollout.trajectory_collection import (
    TrajectoryAssetJob,
    TrajectoryCollectionSpec,
    atomic_write_json,
    collect_trajectories,
    collect_trajectory_index,
    execution_settings,
)
from clothdojo.sim.scene import WORLD_LAYOUT, WORLD_SPACING_M, ShirtFlattenSpec
from clothdojo.trajectory import TrajectoryReader

EVALUATION_CONDITION_SCHEMA = "clothdojo_evaluation_condition"
_SCORING_TEMPLATE_CACHE: dict[tuple[str, str], Any] = {}


def score_evaluation_condition(
    condition: str | Path,
    protocol_path: str | Path,
    templates_path: str | Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Recompute one condition from its trajectories without policy or Newton."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("evaluation scoring workers must be a positive integer")

    root = Path(condition).expanduser().resolve()
    meta_path = root / "meta.json"
    meta = _load_running_meta(meta_path)
    protocol = load_evaluation_protocol(protocol_path)
    if protocol.fingerprint != meta.get("protocol", {}).get("fingerprint"):
        raise ValueError("evaluation protocol differs from the recorded condition")
    trajectories = meta.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != meta.get("case_count"):
        raise RuntimeError("evaluation condition does not contain every trajectory")
    templates_root = Path(templates_path).expanduser().resolve()
    resolved_templates = resolve_canonical_templates(
        templates_root,
        meta.get("templates", {}),
        target_settings=protocol.target.to_json_dict(),
    )
    templates = {}
    for asset_id, expected_sha256 in meta.get("templates", {}).items():
        template_path, collection_sha256 = resolved_templates[asset_id]
        if collection_sha256 != expected_sha256:
            raise ValueError(f"canonical template changed for asset {asset_id}")
        template = load_canonical_template(template_path)
        templates[asset_id] = (template_path, expected_sha256)
        _SCORING_TEMPLATE_CACHE[(str(template_path), expected_sha256)] = template

    score_jobs = []
    expected_keys = []
    seen = set()
    for item in trajectories:
        key = (item["asset_id"], item["initial_state_index"], item["replicate"])
        if key in seen:
            raise ValueError("evaluation trajectory index contains duplicate cases")
        seen.add(key)
        relative = Path(item["path"])
        if relative.parent != Path("trajectories"):
            raise ValueError("evaluation trajectory path is not flat")
        reader = TrajectoryReader(root / relative)
        metadata = reader.manifest.metadata
        if (
            metadata.get("protocol_fingerprint") != protocol.fingerprint
            or metadata.get("template_sha256") != meta["templates"][item["asset_id"]]
        ):
            raise ValueError("evaluation trajectory scoring provenance differs")
        expected_keys.append(key)
        template_path, template_sha256 = templates[item["asset_id"]]
        score_jobs.append(
            (root / relative, template_path, template_sha256, protocol)
        )

    if workers == 1:
        cases = [_score_evaluation_job(job) for job in score_jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(score_jobs))) as executor:
            cases = list(executor.map(_score_evaluation_job, score_jobs, chunksize=1))
    for expected_key, result in zip(expected_keys, cases, strict=True):
        actual_key = (
            result.key.asset_id,
            result.key.initial_state_index,
            result.key.replicate,
        )
        if actual_key != expected_key:
            raise ValueError("evaluation trajectory identity differs from its index")

    write_evaluation_cases(root / "results.jsonl", cases)
    meta["status"] = "complete"
    meta["phase"] = "complete"
    meta.pop("error", None)
    atomic_write_json(meta_path, meta)
    return {
        "compatible": True,
        "output": str(root),
        "condition_id": meta["condition_id"],
        "case_count": len(cases),
        "training_eligible": False,
    }


def run_evaluation_condition(
    scenario_path: str | Path,
    protocol_path: str | Path,
    initial_states_path: str | Path,
    templates_path: str | Path,
    policy_endpoints: tuple[str, ...],
    output: str | Path,
    *,
    condition_id: str,
    expected_checkpoint_sha256: str | None = None,
    assets_root: str | Path = "assets",
    gpus: tuple[str, ...] = ("0",),
    limit_per_asset: int | None = None,
    seen_assets: int | None = None,
    unseen_assets: int | None = None,
    execution: str = "chunk",
    exec_horizon: int = 10,
    rtc_delay_steps: int = 0,
    batch_size: int = 4,
    replicates: int = 1,
    defer_scoring: bool = False,
    score_workers: int = 1,
) -> dict[str, Any]:
    """Collect canonical trajectories, then score one immutable condition offline."""

    if not condition_id.strip():
        raise ValueError("condition_id must not be empty")
    if not policy_endpoints or any(not endpoint.strip() for endpoint in policy_endpoints):
        raise ValueError("at least one policy endpoint is required")
    if not gpus or any(not gpu.strip() for gpu in gpus):
        raise ValueError("at least one simulation GPU is required")
    if execution not in {"chunk", "rtc"}:
        raise ValueError("execution must be 'chunk' or 'rtc'")
    for name, value in (
        ("batch_size", batch_size),
        ("replicates", replicates),
        ("score_workers", score_workers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(rtc_delay_steps, bool)
        or not isinstance(rtc_delay_steps, int)
        or rtc_delay_steps < 0
    ):
        raise ValueError("rtc_delay_steps must be a non-negative integer")

    scenario = load_rollout_scenario(scenario_path)
    protocol = load_evaluation_protocol(protocol_path)
    if scenario.task_name != protocol.task_name:
        raise ValueError("scenario task differs from evaluation protocol")
    if scenario.steps != protocol.horizon_steps:
        raise ValueError("scenario steps differ from the evaluation horizon")
    if not 1 <= exec_horizon <= scenario.action_horizon:
        raise ValueError("exec_horizon must be in [1, action_horizon]")
    logical_delay = rtc_delay_steps if execution == "rtc" else 0
    if execution == "rtc" and logical_delay >= exec_horizon:
        raise ValueError("RTC delay must be smaller than exec_horizon")
    if execution == "rtc" and exec_horizon + logical_delay > scenario.action_horizon:
        raise ValueError("RTC horizon plus delay exceeds the model action horizon")

    states_root = Path(initial_states_path).expanduser().resolve()
    states_meta = load_initial_state_collection(states_root)
    templates_root = Path(templates_path).expanduser().resolve()
    assets = Path(assets_root).expanduser().resolve()
    garments = GarmentLibrary(assets)
    destination = Path(output).expanduser().resolve()
    selected_assets = select_assets(
        states_meta["assets"], seen_assets=seen_assets, unseen_assets=unseen_assets
    )
    selected_assets.sort(key=lambda item: int(item["particle_count"]), reverse=True)
    resolved_templates = resolve_canonical_templates(
        templates_root,
        (item["asset_id"] for item in selected_assets),
        target_settings=protocol.target.to_json_dict(),
    )
    jobs: list[TrajectoryAssetJob] = []
    for item in selected_assets:
        pool_path = states_root / item["asset_id"] / "states.npz"
        pool = InitialStatePool.load(pool_path)
        count = pool.state_count if limit_per_asset is None else int(limit_per_asset)
        if count < 1 or count > pool.state_count:
            raise ValueError(f"limit_per_asset must be in [1, {pool.state_count}]")
        template_path, template_sha256 = resolved_templates[item["asset_id"]]
        template = load_canonical_template(template_path)
        if template.asset_id != item["asset_id"]:
            raise ValueError(f"canonical template asset mismatch: {template_path}")
        garment = garments.converted(item["asset_id"])
        validate_pool_compatibility(
            pool,
            garment_path=garment,
            particle_count=pool.particle_count,
        )
        if template.asset_sha256 != file_sha256(garment):
            raise ValueError("canonical template garment hash differs from the runtime asset")
        if template.topology_sha256 != garment_topology_sha256(garment):
            raise ValueError("canonical template topology differs from the runtime asset")
        physics_spec = ShirtFlattenSpec(garment_id=item["asset_id"])
        if template.scene_physics_sha256 != shirt_flatten_physics_sha256(physics_spec):
            raise ValueError("canonical template physics differs from the runtime scene")
        jobs.append(
            TrajectoryAssetJob(
                asset_id=item["asset_id"],
                pool=str(pool_path),
                pool_sha256=file_sha256(pool_path),
                state_count=count,
                replicates=tuple(range(replicates)),
                metadata={
                    "condition_id": condition_id,
                    "protocol_fingerprint": protocol.fingerprint,
                    "template_sha256": template_sha256,
                },
            )
        )

    worker_count = min(len(gpus), len(jobs))
    active_endpoints = policy_endpoints_for_workers(policy_endpoints, worker_count)
    identities = tuple(
        RemotePolicy(
            endpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            action_horizon=scenario.action_horizon,
        ).identity
        for endpoint in active_endpoints
    )
    policy_identity = validate_policy_identities(
        identities,
        execution=execution,
        worlds_per_batch=batch_size,
    )
    requested = {
        "condition_id": condition_id,
        "protocol": {
            "id": protocol.protocol_id,
            "stage": protocol.stage,
            "fingerprint": protocol.fingerprint,
            "horizon_steps": protocol.horizon_steps,
            "control_fps": protocol.control_fps,
            "metric_backend": "numpy_float64_reference",
        },
        "scenario": scenario.to_json_dict(),
        "cloth_visual": "nocs",
        "observation": policy_observation_settings(),
        "initial_states_sha256": file_sha256(states_root / "meta.json"),
        "initial_state_pools": {job.asset_id: job.pool_sha256 for job in jobs},
        "templates": {
            job.asset_id: job.metadata["template_sha256"] for job in jobs
        },
        "policy": policy_identity,
        "execution": execution_settings(
            execution,
            exec_horizon,
            rtc_delay_steps=logical_delay,
        ),
        "replicates": replicates,
        "parallelism": {
            "worlds_per_batch": batch_size,
            "simulation": {
                "engine": "newton-worlds",
                "world_layout": WORLD_LAYOUT,
                "spacing_m": WORLD_SPACING_M,
            },
            "rendering": "sensor-tiled-camera",
            "policy": "single-batch-request",
        },
        "trajectory_storage": {"npz_compressed": False},
        "case_count": sum(job.state_count for job in jobs) * replicates,
    }
    collection_spec = TrajectoryCollectionSpec(
        scenario_path=str(Path(scenario_path).expanduser().resolve()),
        assets_root=str(assets),
        trajectories_root=str(destination / "trajectories"),
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        policy_identity=policy_identity,
        execution=execution,
        exec_horizon=exec_horizon,
        rtc_delay_steps=logical_delay,
        horizon_steps=scenario.steps,
        batch_size=batch_size,
        producer="clothdojo_evaluation",
        training_eligible=False,
        compressed=False,
    )
    meta_path = destination / "meta.json"
    if meta_path.is_file():
        meta = _load_running_meta(meta_path)
        if {key: meta.get(key) for key in requested} != requested:
            raise ValueError("existing evaluation condition settings differ")
        if meta.get("status") == "complete":
            try:
                _validate_published_condition(destination, jobs, collection_spec)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError, EOFError):
                meta["status"] = "running"
                meta["phase"] = "scoring"
                atomic_write_json(meta_path, meta)
            else:
                return _summary(destination, meta, gpus)
    else:
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"evaluation output is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "trajectories").mkdir()
        meta = {
            "schema": EVALUATION_CONDITION_SCHEMA,
            "status": "running",
            "phase": "trajectories",
            "training_eligible": False,
            "results": "results.jsonl",
            "trajectories": [],
            **requested,
        }
        atomic_write_json(meta_path, meta)

    try:
        meta["status"] = "running"
        meta["phase"] = "trajectories"
        meta.pop("error", None)
        atomic_write_json(meta_path, meta)
        collect_trajectories(
            jobs,
            gpus=gpus,
            policy_endpoints=active_endpoints,
            spec=collection_spec,
        )
        trajectories = collect_trajectory_index(jobs, spec=collection_spec)
        meta["trajectories"] = trajectories
        meta["phase"] = "scoring"
        atomic_write_json(meta_path, meta)
        if defer_scoring:
            return _summary(destination, meta, gpus)
        score_evaluation_condition(
            destination,
            protocol_path,
            templates_path,
            workers=score_workers,
        )
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["trajectories"] = collect_trajectory_index(
            jobs, spec=collection_spec, partial=True
        )
        atomic_write_json(meta_path, meta)
        raise
    meta = _load_running_meta(meta_path)
    return _summary(destination, meta, gpus)


def _validate_published_condition(
    root: Path,
    jobs: list[TrajectoryAssetJob],
    spec: TrajectoryCollectionSpec,
) -> None:
    meta = _load_running_meta(root / "meta.json")
    trajectories = collect_trajectory_index(jobs, spec=spec)
    if trajectories != meta.get("trajectories"):
        raise ValueError("published evaluation trajectory index differs from the trajectories")
    published = load_evaluation_condition(root)
    expected = {
        (item["asset_id"], item["initial_state_index"], item["replicate"]): (
            item["seed"],
            item["policy_seed"],
        )
        for item in trajectories
    }
    actual = {
        (case.key.asset_id, case.key.initial_state_index, case.key.replicate): (
            case.initial_state_seed,
            case.policy_seed,
        )
        for case in published.cases
    }
    if actual != expected:
        raise ValueError("published evaluation case index differs from the trajectories")


def _load_running_meta(path: Path) -> dict[str, Any]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read evaluation metadata {path}: {exc}") from exc
    if meta.get("schema") != EVALUATION_CONDITION_SCHEMA:
        raise ValueError("unsupported evaluation condition schema")
    return meta


def _summary(root: Path, meta: dict[str, Any], gpus) -> dict[str, Any]:
    result = {
        "compatible": True,
        "output": str(root),
        "condition_id": meta["condition_id"],
        "status": meta["status"],
        "phase": meta["phase"],
        "protocol_stage": meta["protocol"]["stage"],
        "case_count": meta["case_count"],
        "trajectory_count": len(meta.get("trajectories", [])),
        "checkpoint_sha256": meta["policy"]["checkpoint_sha256"],
        "training_eligible": False,
        "gpus": list(gpus),
    }
    performance = _performance_summary(root, meta.get("trajectories", []))
    if performance is not None:
        result["performance"] = performance
    return result


def _score_evaluation_job(job) -> Any:
    trajectory, template_path, template_sha256, protocol = job
    key = (str(template_path), template_sha256)
    template = _SCORING_TEMPLATE_CACHE.get(key)
    if template is None:
        template = load_canonical_template(template_path)
        _SCORING_TEMPLATE_CACHE[key] = template
    return score_evaluation_trajectory(trajectory, template, protocol).result


def _performance_summary(root: Path, trajectories) -> dict[str, float | int] | None:
    batches = []
    seen = set()
    for item in trajectories:
        try:
            metadata = TrajectoryReader(root / item["path"]).manifest.metadata
            parallelism = metadata["parallelism"]
            key = (
                item["asset_id"],
                item["replicate"],
                parallelism["batch_start"],
                metadata["policy_seed"],
            )
            if key in seen:
                continue
            seen.add(key)
            performance = dict(metadata["execution_report"]["performance"])
            performance["active_world_count"] = parallelism["active_worlds"]
            performance["trajectory_publication"] = metadata.get(
                "trajectory_publication", {}
            )
            batches.append(performance)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, EOFError):
            continue
    if not batches:
        return None
    total_seconds = sum(float(item["wall_seconds"]) for item in batches)
    result: dict[str, float | int] = {
        "batch_count": len(batches),
        "episode_worker_seconds": total_seconds,
        "trajectory_finalize_seconds": sum(
            float(item["trajectory_publication"].get("finalize_seconds", 0.0))
            for item in batches
        ),
    }
    for phase in ("policy_session_reset", "observation", "policy_end_to_end", "execution"):
        phase_seconds = sum(float(item["phases"][phase]["total_seconds"]) for item in batches)
        result[f"{phase}_seconds"] = phase_seconds
        result[f"{phase}_fraction"] = phase_seconds / total_seconds
    unattributed_seconds = sum(
        float(item.get("unattributed_seconds", item.get("engine_overhead_seconds", 0.0)))
        for item in batches
    )
    overlapped_seconds = sum(
        float(item.get("overlapped_phase_seconds", 0.0)) for item in batches
    )
    result["unattributed_seconds"] = unattributed_seconds
    result["unattributed_fraction"] = unattributed_seconds / total_seconds
    result["overlapped_phase_seconds"] = overlapped_seconds
    result["overlapped_phase_fraction"] = overlapped_seconds / total_seconds
    return result
