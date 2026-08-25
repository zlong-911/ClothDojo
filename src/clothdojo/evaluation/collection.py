"""One-condition, multi-GPU benchmark execution over a fixed initial-state pool."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.evaluation.metric import file_sha256, load_canonical_template
from clothdojo.evaluation.protocol import load_evaluation_protocol
from clothdojo.evaluation.results import (
    EVALUATION_RESULT_FORMAT_VERSION,
    EvaluationCaseKey,
    EvaluationCaseResult,
    load_evaluation_condition,
    write_evaluation_cases,
)
from clothdojo.evaluation.templates import (
    garment_topology_sha256,
    shirt_flatten_physics_sha256,
)
from clothdojo.rollout.initial_states import (
    InitialStatePool,
    load_initial_state_collection,
    validate_pool_compatibility,
)
from clothdojo.rollout.policy import RemoteSFTPolicy
from clothdojo.rollout.scenario import load_rollout_scenario
from clothdojo.rollout.scheduling import (
    policy_endpoints_for_workers,
    run_policy_asset_workers,
    select_assets,
    validate_policy_identities,
)

EVALUATION_CONDITION_SCHEMA = "clothdojo_evaluation_condition"
EVALUATION_CONDITION_FORMAT_VERSION = EVALUATION_RESULT_FORMAT_VERSION


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
    batch_size: int = 4,
    replicates: int = 1,
) -> dict[str, Any]:
    """Evaluate one immutable checkpoint/execution condition."""

    if not condition_id.strip():
        raise ValueError("condition_id must not be empty")
    if not policy_endpoints or any(not endpoint.strip() for endpoint in policy_endpoints):
        raise ValueError("at least one policy endpoint is required")
    if not gpus or any(not gpu.strip() for gpu in gpus):
        raise ValueError("at least one simulation GPU is required")
    if execution not in {"chunk", "rtc"}:
        raise ValueError("execution must be 'chunk' or 'rtc'")
    for name, value in (("batch_size", batch_size), ("replicates", replicates)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    scenario = load_rollout_scenario(scenario_path)
    protocol = load_evaluation_protocol(protocol_path)
    if scenario.task_name != protocol.task_name:
        raise ValueError("scenario task differs from evaluation protocol")
    if scenario.steps != protocol.horizon_steps:
        raise ValueError("scenario steps differ from the evaluation horizon")
    if not 1 <= exec_horizon <= scenario.action_horizon:
        raise ValueError("exec_horizon must be in [1, action_horizon]")

    states_root = Path(initial_states_path).expanduser().resolve()
    states_meta = load_initial_state_collection(states_root)
    templates_root = Path(templates_path).expanduser().resolve()
    assets = Path(assets_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    selected_assets = select_assets(
        states_meta["assets"], seen_assets=seen_assets, unseen_assets=unseen_assets
    )
    jobs = []
    for item in selected_assets:
        pool_path = states_root / item["asset_id"] / "states.npz"
        pool = InitialStatePool.load(pool_path)
        count = pool.state_count if limit_per_asset is None else int(limit_per_asset)
        if count < 1 or count > pool.state_count:
            raise ValueError(f"limit_per_asset must be in [1, {pool.state_count}]")
        template_path = templates_root / f"{item['asset_id']}.npz"
        template = load_canonical_template(template_path)
        if template.asset_id != item["asset_id"]:
            raise ValueError(f"canonical template asset mismatch: {template_path}")
        jobs.append(
            {
                "asset_id": item["asset_id"],
                "pool": str(pool_path),
                "pool_sha256": file_sha256(pool_path),
                "template": str(template_path),
                "template_sha256": file_sha256(template_path),
                "state_count": count,
            }
        )

    worker_count = min(len(gpus), len(jobs))
    active_endpoints = policy_endpoints_for_workers(policy_endpoints, worker_count)
    identities = tuple(
        RemoteSFTPolicy(
            endpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            action_horizon=scenario.action_horizon,
        ).identity
        for endpoint in active_endpoints
    )
    policy_identity = validate_policy_identities(
        identities,
        execution=execution,
        exec_horizon=exec_horizon,
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
        },
        "scenario": scenario.to_json_dict(),
        "cloth_visual": "nocs",
        "initial_states_sha256": file_sha256(states_root / "meta.json"),
        "initial_state_pools": {
            job["asset_id"]: job["pool_sha256"] for job in jobs
        },
        "templates": {job["asset_id"]: job["template_sha256"] for job in jobs},
        "policy": policy_identity,
        "execution": {"mode": execution, "exec_horizon": exec_horizon},
        "replicates": replicates,
        "parallelism": {
            "worlds_per_batch": batch_size,
            "simulation": "newton-worlds",
            "rendering": "sensor-tiled-camera",
            "policy": "single-batch-request",
        },
        "case_count": sum(job["state_count"] for job in jobs) * replicates,
    }
    meta_path = destination / "meta.json"
    if meta_path.is_file():
        meta = _load_running_meta(meta_path)
        actual = {key: meta.get(key) for key in requested}
        if actual != requested:
            raise ValueError("existing evaluation condition settings differ")
        if meta.get("status") == "complete":
            try:
                published = load_evaluation_condition(destination)
                cases = _collect_cases(
                    destination,
                    jobs,
                    replicates=replicates,
                    sample_count=protocol.sample_count,
                )
                if {case.key: case for case in published.cases} != {
                    case.key: case for case in cases
                }:
                    raise ValueError("evaluation cases differ from the per-asset results")
            except (OSError, RuntimeError, TypeError, ValueError, KeyError, EOFError):
                meta["status"] = "running"
                meta["results"] = "results.jsonl"
                _atomic_json(meta_path, meta)
            else:
                return _summary(destination, meta, gpus)
    else:
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"evaluation output is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "assets").mkdir()
        meta = {
            "schema": EVALUATION_CONDITION_SCHEMA,
            "format_version": EVALUATION_CONDITION_FORMAT_VERSION,
            "status": "running",
            "training_eligible": False,
            "results": "results.jsonl",
            **requested,
        }
        _atomic_json(meta_path, meta)

    try:
        _run_workers(
            jobs,
            gpus=gpus,
            policy_endpoints=active_endpoints,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            scenario_path=Path(scenario_path).expanduser().resolve(),
            protocol_path=Path(protocol_path).expanduser().resolve(),
            assets_root=assets,
            output=destination,
            execution=execution,
            exec_horizon=exec_horizon,
            batch_size=batch_size,
            replicates=replicates,
            sample_count=protocol.sample_count,
        )
        cases = _collect_cases(
            destination,
            jobs,
            replicates=replicates,
            sample_count=protocol.sample_count,
        )
        write_evaluation_cases(destination / "results.jsonl", cases)
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(meta_path, meta)
        raise
    meta["status"] = "complete"
    meta.pop("error", None)
    _atomic_json(meta_path, meta)
    return _summary(destination, meta, gpus)


def _run_workers(
    jobs,
    *,
    gpus,
    policy_endpoints,
    expected_checkpoint_sha256,
    scenario_path,
    protocol_path,
    assets_root,
    output,
    execution,
    exec_horizon,
    batch_size,
    replicates,
    sample_count,
) -> None:
    meta = _load_running_meta(output / "meta.json")
    pending = [
        job
        for job in jobs
        if not _asset_result_is_valid(
            output,
            job,
            meta=meta,
            replicates=replicates,
            sample_count=sample_count,
        )
    ]
    if not pending:
        return
    worker_count = min(len(gpus), len(pending))
    run_policy_asset_workers(
        pending,
        gpus=gpus[:worker_count],
        policy_endpoints=policy_endpoints[:worker_count],
        worker_target=_worker,
        common_args=(
            expected_checkpoint_sha256,
            str(scenario_path),
            str(protocol_path),
            str(assets_root),
            str(output),
            execution,
            exec_horizon,
            batch_size,
            replicates,
        ),
        process_kind="evaluation",
    )


def _worker(
    job_queue,
    result_queue,
    gpu,
    endpoint,
    expected_checkpoint_sha256,
    scenario_path,
    protocol_path,
    assets_root,
    output,
    execution,
    exec_horizon,
    batch_size,
    replicates,
) -> None:
    if gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cpu" if gpu == "cpu" else "cuda:0"
    while True:
        job = job_queue.get()
        if job is None:
            return
        try:
            _run_asset(
                job,
                endpoint=endpoint,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                scenario_path=scenario_path,
                protocol_path=protocol_path,
                assets_root=assets_root,
                output=Path(output),
                device=device,
                execution=execution,
                exec_horizon=exec_horizon,
                batch_size=batch_size,
                replicates=replicates,
            )
        except Exception as exc:
            result_queue.put(
                {
                    "status": "error",
                    "asset_id": job["asset_id"],
                    "gpu": gpu,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return
        result_queue.put({"status": "complete", "asset_id": job["asset_id"], "gpu": gpu})


def _run_asset(
    job,
    *,
    endpoint,
    expected_checkpoint_sha256,
    scenario_path,
    protocol_path,
    assets_root,
    output,
    device,
    execution,
    exec_horizon,
    batch_size,
    replicates,
) -> None:
    from clothdojo.assets import GarmentLibrary
    from clothdojo.evaluation.newton import NewtonEvaluationEpisode
    from clothdojo.rollout.engine import run_episode
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import (
        ShirtFlattenSpec,
        build_shirt_flatten,
    )

    asset_id = job["asset_id"]
    pool = InitialStatePool.load(job["pool"])
    template = load_canonical_template(job["template"])
    protocol = load_evaluation_protocol(protocol_path)
    scenario = replace(load_rollout_scenario(scenario_path), garment_id=asset_id)
    assets = Path(assets_root).expanduser().resolve()
    garment = GarmentLibrary(assets).converted(asset_id)
    validate_pool_compatibility(pool, garment_path=garment, particle_count=pool.particle_count)
    if template.asset_sha256 != file_sha256(garment):
        raise ValueError("canonical template garment hash differs from the runtime asset")
    policy = RemoteSFTPolicy(
        endpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        action_horizon=scenario.action_horizon,
    )
    world_count = min(batch_size, job["state_count"])
    spec = ShirtFlattenSpec(garment_id=asset_id, world_count=world_count)
    if template.topology_sha256 != garment_topology_sha256(garment):
        raise ValueError("canonical template topology differs from the runtime asset")
    if template.scene_physics_sha256 != shirt_flatten_physics_sha256(spec):
        raise ValueError("canonical template physics differs from the runtime scene")
    scene = build_shirt_flatten(assets, spec=spec, device=device, visuals=True)
    runtime = ShirtFlattenRuntime(scene)
    validate_pool_compatibility(
        pool,
        garment_path=garment,
        particle_count=runtime.grasp.particle_ranges()[0][1]
        - runtime.grasp.particle_ranges()[0][0],
    )
    camera = ShirtFlattenCameraObservation(scene)
    shape = (replicates, job["state_count"])
    arrays = {
        "initial_state_seed": np.zeros(shape, dtype=np.int32),
        "policy_seed": np.zeros(shape, dtype=np.int64),
        "success": np.zeros(shape, dtype=bool),
        "success_at_end": np.zeros(shape, dtype=bool),
        "confirmed_step": np.full(shape, -1, dtype=np.int32),
        "stable_since_step": np.full(shape, -1, dtype=np.int32),
        "terminal_deformable_distance": np.zeros(shape, dtype=np.float32),
        "terminal_rigid_distance": np.zeros(shape, dtype=np.float32),
        "terminal_combined_distance": np.zeros(shape, dtype=np.float32),
        "terminal_speed_rms_m_s": np.zeros(shape, dtype=np.float32),
        "best_settled_step": np.full(shape, -1, dtype=np.int32),
        "best_settled_deformable_distance": np.full(shape, np.nan, dtype=np.float32),
        "best_settled_rigid_distance": np.full(shape, np.nan, dtype=np.float32),
        "best_settled_combined_distance": np.full(shape, np.nan, dtype=np.float32),
        "curve": np.full((*shape, protocol.sample_count, 4), np.nan, dtype=np.float32),
    }
    for replicate in range(replicates):
        for batch_start in range(0, job["state_count"], world_count):
            active = list(range(batch_start, min(batch_start + world_count, job["state_count"])))
            world_indices = active + [active[-1]] * (world_count - len(active))
            selected_seeds = np.asarray(pool.seeds[world_indices], dtype=np.int32)
            policy_seed = _policy_seed(selected_seeds, replicate)
            runtime.reset()
            runtime.restore_particles(
                np.ascontiguousarray(pool.particle_q[world_indices]),
                np.ascontiguousarray(pool.particle_qd[world_indices]),
            )
            episode = NewtonEvaluationEpisode(
                runtime,
                observation_provider=camera,
                template=template,
                protocol=protocol,
            )
            run_episode(
                episode,
                policy,
                scenario,
                mode=execution,
                exec_horizon=exec_horizon,
                policy_seed=policy_seed,
            )
            result = episode.result()
            count = len(active)
            selection = (replicate, slice(batch_start, batch_start + count))
            arrays["initial_state_seed"][selection] = selected_seeds[:count]
            arrays["policy_seed"][selection] = policy_seed
            for name in (
                "success",
                "success_at_end",
                "confirmed_step",
                "stable_since_step",
                "terminal_deformable_distance",
                "terminal_rigid_distance",
                "terminal_combined_distance",
                "terminal_speed_rms_m_s",
                "best_settled_step",
                "best_settled_deformable_distance",
                "best_settled_rigid_distance",
                "best_settled_combined_distance",
            ):
                arrays[name][selection] = getattr(result, name)[:count]
            arrays["curve"][selection] = np.transpose(result.curve[:, :count], (1, 0, 2))
    _write_asset_result(
        _asset_result_path(output, asset_id),
        asset_id=asset_id,
        pool_sha256=job["pool_sha256"],
        template_sha256=job["template_sha256"],
        protocol_fingerprint=protocol.fingerprint,
        checkpoint_sha256=policy.identity["checkpoint_sha256"],
        execution=execution,
        exec_horizon=exec_horizon,
        arrays=arrays,
    )


def _collect_cases(
    root: Path,
    jobs,
    *,
    replicates: int,
    sample_count: int,
) -> tuple[EvaluationCaseResult, ...]:
    meta = _load_running_meta(root / "meta.json")
    cases = []
    for job in jobs:
        path = _asset_result_path(root, job["asset_id"])
        cases.extend(
            _load_asset_cases(
                path,
                job,
                meta=meta,
                replicates=replicates,
                sample_count=sample_count,
            )
        )
    return tuple(cases)


def _load_asset_cases(
    path: Path,
    job,
    *,
    meta: dict[str, Any],
    replicates: int,
    sample_count: int,
) -> tuple[EvaluationCaseResult, ...]:
    cases = []
    shape = (replicates, job["state_count"])
    array_names = (
        "initial_state_seed",
        "policy_seed",
        "success",
        "success_at_end",
        "confirmed_step",
        "stable_since_step",
        "terminal_deformable_distance",
        "terminal_rigid_distance",
        "terminal_combined_distance",
        "terminal_speed_rms_m_s",
        "best_settled_step",
        "best_settled_deformable_distance",
        "best_settled_rigid_distance",
        "best_settled_combined_distance",
    )
    with np.load(path, allow_pickle=False) as data:
        if _scalar(data["asset_id"]) != job["asset_id"]:
            raise ValueError(f"evaluation asset result identity differs: {path}")
        if _scalar(data["pool_sha256"]) != job["pool_sha256"]:
            raise ValueError(f"evaluation asset result pool differs: {path}")
        if _scalar(data["template_sha256"]) != job["template_sha256"]:
            raise ValueError(f"evaluation asset result template differs: {path}")
        if _scalar(data["protocol_fingerprint"]) != meta["protocol"]["fingerprint"]:
            raise ValueError(f"evaluation asset result protocol differs: {path}")
        if _scalar(data["checkpoint_sha256"]) != meta["policy"]["checkpoint_sha256"]:
            raise ValueError(f"evaluation asset result checkpoint differs: {path}")
        if _scalar(data["execution"]) != meta["execution"]["mode"]:
            raise ValueError(f"evaluation asset result execution mode differs: {path}")
        if (
            np.asarray(data["exec_horizon"]).shape != ()
            or int(data["exec_horizon"].item()) != meta["execution"]["exec_horizon"]
        ):
            raise ValueError(f"evaluation asset result execution horizon differs: {path}")
        for name in array_names:
            if data[name].shape != shape:
                raise ValueError(f"evaluation asset result {name} shape differs: {path}")
        curve = np.asarray(data["curve"])
        if curve.shape != (*shape, sample_count, 4):
            raise ValueError(f"evaluation asset result curve shape differs: {path}")
        if curve.dtype.kind not in "fiu" or not np.isfinite(curve).all() or np.any(curve < 0):
            raise ValueError(f"evaluation asset result curve is invalid: {path}")
        for replicate in range(replicates):
            for index in range(job["state_count"]):
                best_step = int(data["best_settled_step"][replicate, index])
                confirmed = int(data["confirmed_step"][replicate, index])
                stable_since = int(data["stable_since_step"][replicate, index])
                cases.append(
                    EvaluationCaseResult(
                        key=EvaluationCaseKey(job["asset_id"], index, replicate),
                        initial_state_seed=int(data["initial_state_seed"][replicate, index]),
                        policy_seed=int(data["policy_seed"][replicate, index]),
                        success=bool(data["success"][replicate, index]),
                        success_at_end=bool(data["success_at_end"][replicate, index]),
                        confirmed_step=None if confirmed < 0 else confirmed,
                        stable_since_step=None if stable_since < 0 else stable_since,
                        terminal_deformable_distance=float(
                            data["terminal_deformable_distance"][replicate, index]
                        ),
                        terminal_rigid_distance=float(
                            data["terminal_rigid_distance"][replicate, index]
                        ),
                        terminal_combined_distance=float(
                            data["terminal_combined_distance"][replicate, index]
                        ),
                        terminal_speed_rms_m_s=float(
                            data["terminal_speed_rms_m_s"][replicate, index]
                        ),
                        best_settled_step=None if best_step < 0 else best_step,
                        best_settled_deformable_distance=(
                            None
                            if best_step < 0
                            else float(data["best_settled_deformable_distance"][replicate, index])
                        ),
                        best_settled_rigid_distance=(
                            None
                            if best_step < 0
                            else float(data["best_settled_rigid_distance"][replicate, index])
                        ),
                        best_settled_combined_distance=(
                            None
                            if best_step < 0
                            else float(data["best_settled_combined_distance"][replicate, index])
                        ),
                    )
                )
    return tuple(cases)


def _asset_result_is_valid(
    root: Path,
    job,
    *,
    meta: dict[str, Any],
    replicates: int,
    sample_count: int,
) -> bool:
    try:
        _load_asset_cases(
            _asset_result_path(root, job["asset_id"]),
            job,
            meta=meta,
            replicates=replicates,
            sample_count=sample_count,
        )
    except (OSError, TypeError, ValueError, KeyError, EOFError):
        return False
    return True


def _write_asset_result(path: Path, *, arrays, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            **{key: np.asarray(value) for key, value in metadata.items()},
            **arrays,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _policy_seed(seeds: np.ndarray, replicate: int) -> int:
    values = np.asarray(seeds, dtype=np.int32)
    if values.ndim != 1 or not len(values) or np.any(values < 0):
        raise ValueError("evaluation batch seeds must be non-negative int32")
    if replicate < 0:
        raise ValueError("replicate must be non-negative")
    payload = values.astype("<i4", copy=False).tobytes()
    payload += np.asarray(replicate, dtype="<i4").tobytes()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _asset_result_path(root: Path, asset_id: str) -> Path:
    return Path(root) / "assets" / f"{asset_id}.npz"


def _load_running_meta(path: Path) -> dict[str, Any]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read evaluation metadata {path}: {exc}") from exc
    if meta.get("schema") != EVALUATION_CONDITION_SCHEMA:
        raise ValueError("unsupported evaluation condition schema")
    if meta.get("format_version") != EVALUATION_CONDITION_FORMAT_VERSION:
        raise ValueError("unsupported evaluation condition version")
    return meta


def _summary(root: Path, meta: dict[str, Any], gpus) -> dict[str, Any]:
    return {
        "compatible": True,
        "output": str(root),
        "condition_id": meta["condition_id"],
        "protocol_stage": meta["protocol"]["stage"],
        "case_count": meta["case_count"],
        "checkpoint_sha256": meta["policy"]["checkpoint_sha256"],
        "training_eligible": False,
        "gpus": list(gpus),
    }


def _scalar(value) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("evaluation asset metadata must be scalar")
    return str(array.item())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
