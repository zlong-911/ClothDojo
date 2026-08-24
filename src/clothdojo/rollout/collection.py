"""Multi-asset policy rollouts with teleoperation-compatible trajectories."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from clothdojo.rollout.initial_states import (
    InitialStatePool,
    load_initial_state_collection,
    validate_pool_compatibility,
)
from clothdojo.rollout.policy import RemoteSFTPolicy
from clothdojo.rollout.render import CAMERA_NAMES
from clothdojo.rollout.scenario import load_rollout_scenario
from clothdojo.trajectory import TrajectoryReader

ROLLOUT_COLLECTION_SCHEMA = "clothdojo_rollout"
ROLLOUT_COLLECTION_FORMAT_VERSION = 1


def load_rollout_collection(source: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Validate one rollout experiment and its flat canonical trajectories."""

    root = Path(source).expanduser().resolve()
    meta_path = root / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read rollout metadata: {meta_path}: {exc}") from exc
    if meta.get("schema") != ROLLOUT_COLLECTION_SCHEMA:
        raise ValueError("unsupported rollout schema")
    if meta.get("format_version") != ROLLOUT_COLLECTION_FORMAT_VERSION:
        raise ValueError("unsupported rollout version")
    if require_complete and meta.get("status") != "complete":
        raise RuntimeError(f"rollout is not complete: {meta.get('status')!r}")
    trajectories = meta.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("rollout trajectory list is invalid")
    if require_complete and len(trajectories) != meta.get("expected_trajectories"):
        raise ValueError("rollout is missing expected trajectories")
    paths: set[str] = set()
    for item in trajectories:
        if not isinstance(item, dict):
            raise ValueError("rollout trajectory entry is invalid")
        relative = item.get("path")
        if not isinstance(relative, str) or Path(relative).parent != Path("trajectories"):
            raise ValueError("rollout trajectory path is not flat")
        if relative in paths:
            raise ValueError("rollout trajectory paths are not unique")
        paths.add(relative)
        if require_complete:
            reader = TrajectoryReader(root / relative)
            if reader.manifest.action_count != item.get("action_count"):
                raise ValueError("rollout trajectory action count is inconsistent")
            metadata = reader.manifest.metadata
            if metadata.get("asset_id") != item.get("asset_id"):
                raise ValueError("rollout trajectory asset identity is inconsistent")
            if metadata.get("initial_state", {}).get("index") != item.get("initial_state_index"):
                raise ValueError("rollout initial-state index is inconsistent")
    return meta


def run_rollout_collection(
    scenario_path: str | Path,
    initial_states_path: str | Path,
    policy_endpoints: tuple[str, ...],
    output: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    assets_root: str | Path = "assets",
    gpus: tuple[str, ...] = ("0",),
    limit_per_asset: int | None = None,
    seen_assets: int | None = None,
    unseen_assets: int | None = None,
    execution: str = "chunk",
    exec_horizon: int = 10,
) -> dict[str, Any]:
    """Run every selected asset on a dynamic local GPU worker queue."""

    if not policy_endpoints or any(not endpoint.strip() for endpoint in policy_endpoints):
        raise ValueError("at least one policy endpoint is required")
    if not gpus or any(not gpu.strip() for gpu in gpus):
        raise ValueError("at least one simulation GPU is required")
    if execution not in {"chunk", "rtc"}:
        raise ValueError("execution must be 'chunk' or 'rtc'")
    scenario = load_rollout_scenario(scenario_path)
    if not 1 <= exec_horizon <= scenario.action_horizon:
        raise ValueError("exec_horizon must be in [1, action_horizon]")
    states_root = Path(initial_states_path).expanduser().resolve()
    states_meta = load_initial_state_collection(states_root)
    assets = Path(assets_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    selected_assets = _select_assets(
        states_meta["assets"],
        seen_assets=seen_assets,
        unseen_assets=unseen_assets,
    )
    asset_jobs = []
    expected = 0
    for item in selected_assets:
        pool_path = states_root / item["asset_id"] / "states.npz"
        pool = InitialStatePool.load(pool_path)
        count = pool.state_count if limit_per_asset is None else int(limit_per_asset)
        if count < 1 or count > pool.state_count:
            raise ValueError(f"limit_per_asset must be in [1, {pool.state_count}]")
        asset_jobs.append(
            {
                "asset_id": item["asset_id"],
                "pool": str(pool_path),
                "pool_sha256": _file_sha256(pool_path),
                "state_count": count,
            }
        )
        expected += count

    worker_count = min(len(gpus), len(asset_jobs))
    active_endpoints = _policy_endpoints_for_workers(policy_endpoints, worker_count)
    identities = tuple(
        RemoteSFTPolicy(
            endpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            action_horizon=scenario.action_horizon,
        ).identity
        for endpoint in active_endpoints
    )
    first_identity = _validate_policy_identities(
        identities,
        execution=execution,
        exec_horizon=exec_horizon,
    )

    requested = {
        "scenario": scenario.to_json_dict(),
        "cloth_visual": "nocs",
        "initial_states": {
            "path": str(states_root),
            "meta_sha256": _file_sha256(states_root / "meta.json"),
        },
        "policy": first_identity,
        "policy_endpoints": list(active_endpoints),
        "limit_per_asset": limit_per_asset,
        "asset_selection": {
            "seen": sum(item["split"] == "seen" for item in selected_assets),
            "unseen": sum(item["split"] == "unseen" for item in selected_assets),
            "asset_ids": [item["asset_id"] for item in selected_assets],
        },
        "execution": _execution_settings(execution, exec_horizon),
        "expected_trajectories": expected,
    }
    meta_path = destination / "meta.json"
    if meta_path.is_file():
        meta = load_rollout_collection(destination, require_complete=False)
        actual = {key: meta.get(key) for key in requested}
        if actual != requested:
            raise ValueError(f"existing rollout settings differ: {actual} != {requested}")
        if meta.get("status") == "complete":
            complete = load_rollout_collection(destination)
            return _rollout_summary(destination, complete, gpus)
    else:
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"rollout output is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "trajectories").mkdir()
        meta = {
            "schema": ROLLOUT_COLLECTION_SCHEMA,
            "format_version": ROLLOUT_COLLECTION_FORMAT_VERSION,
            "status": "running",
            "camera_names": list(CAMERA_NAMES),
            "training_eligible": True,
            "trajectories": [],
            **requested,
        }
        _atomic_write_json(meta_path, meta)

    try:
        _run_rollout_workers(
            asset_jobs,
            gpus=gpus,
            policy_endpoints=active_endpoints,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            scenario_path=Path(scenario_path).expanduser().resolve(),
            assets_root=assets,
            output=destination,
            execution=execution,
            exec_horizon=exec_horizon,
        )
        trajectories = _collect_trajectories(destination, asset_jobs)
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["trajectories"] = _collect_trajectories(destination, asset_jobs, partial=True)
        _atomic_write_json(meta_path, meta)
        raise

    meta["status"] = "complete"
    meta.pop("error", None)
    meta["trajectories"] = trajectories
    _atomic_write_json(meta_path, meta)
    return _rollout_summary(destination, meta, gpus)


def _select_assets(
    assets: list[dict[str, Any]],
    *,
    seen_assets: int | None,
    unseen_assets: int | None,
) -> list[dict[str, Any]]:
    if (seen_assets is None) != (unseen_assets is None):
        raise ValueError("seen_assets and unseen_assets must be provided together")
    if seen_assets is None:
        return list(assets)

    counts = {"seen": seen_assets, "unseen": unseen_assets}
    for split, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{split}_assets must be a non-negative integer")
    if seen_assets + unseen_assets < 1:
        raise ValueError("at least one rollout asset must be selected")

    selected = []
    for split, count in counts.items():
        candidates = [item for item in assets if item.get("split") == split]
        if count > len(candidates):
            raise ValueError(
                f"requested {count} {split} assets, but the state pool contains {len(candidates)}"
            )
        selected.extend(candidates[:count])
    return selected


def _policy_endpoints_for_workers(
    policy_endpoints: tuple[str, ...], worker_count: int
) -> tuple[str, ...]:
    if worker_count < 1:
        raise ValueError("rollout requires at least one active worker")
    if len(policy_endpoints) < worker_count:
        raise ValueError(
            "reproducible multi-GPU rollout requires one unique policy endpoint "
            f"per active worker: workers={worker_count}, endpoints={len(policy_endpoints)}"
        )
    active = policy_endpoints[:worker_count]
    if len(set(active)) != len(active):
        raise ValueError("each active rollout worker requires a unique policy endpoint")
    return active


def _validate_policy_identities(
    identities: tuple[dict[str, Any], ...], *, execution: str, exec_horizon: int
) -> dict[str, Any]:
    if not identities:
        raise ValueError("rollout requires at least one policy identity")
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise ValueError("policy endpoints do not serve the same checkpoint and configuration")
    if execution == "rtc":
        if first.get("backend") != "rlinf-sft":
            raise ValueError("RTC rollout requires an RLINF SFT policy server")
        server_horizon = first.get("rtc_guidance", {}).get("execution_horizon")
        if server_horizon != exec_horizon:
            raise ValueError(
                "RTC execution horizon differs between rollout and policy server: "
                f"{exec_horizon} != {server_horizon}"
            )
    return first


def _run_rollout_workers(
    jobs: list[dict[str, Any]],
    *,
    gpus: tuple[str, ...],
    policy_endpoints: tuple[str, ...],
    expected_checkpoint_sha256: str | None,
    scenario_path: Path,
    assets_root: Path,
    output: Path,
    execution: str,
    exec_horizon: int,
) -> None:
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    for job in jobs:
        job_queue.put(job)
    worker_count = min(len(gpus), len(jobs))
    for _ in range(worker_count):
        job_queue.put(None)
    workers = []
    for index, gpu in enumerate(gpus[:worker_count]):
        endpoint = policy_endpoints[index]
        worker = context.Process(
            target=_rollout_worker,
            args=(
                job_queue,
                result_queue,
                gpu,
                endpoint,
                expected_checkpoint_sha256,
                str(scenario_path),
                str(assets_root),
                str(output),
                execution,
                exec_horizon,
            ),
            name=f"clothdojo-rollout-gpu-{gpu}",
        )
        workers.append(worker)
    for worker in workers:
        worker.start()
    completed = 0
    try:
        while completed < len(jobs):
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                failures = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if failures:
                    raise RuntimeError(
                        f"rollout worker exited with code {failures[0].exitcode}"
                    ) from None
                continue
            if result["status"] == "error":
                raise RuntimeError(
                    f"rollout failed for {result['asset_id']} on GPU {result['gpu']}: "
                    f"{result['error']}"
                )
            completed += 1
    finally:
        for worker in workers:
            if worker.is_alive() and completed < len(jobs):
                worker.terminate()
            worker.join()


def _rollout_worker(
    job_queue,
    result_queue,
    gpu: str,
    endpoint: str,
    expected_checkpoint_sha256: str | None,
    scenario_path: str,
    assets_root: str,
    output: str,
    execution: str,
    exec_horizon: int,
) -> None:
    if gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cpu" if gpu == "cpu" else "cuda:0"
    while True:
        job = job_queue.get()
        if job is None:
            return
        try:
            _run_asset_rollouts(
                job,
                scenario_path=scenario_path,
                endpoint=endpoint,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                assets_root=assets_root,
                trajectories_root=Path(output) / "trajectories",
                device=device,
                execution=execution,
                exec_horizon=exec_horizon,
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


def _run_asset_rollouts(
    job: dict[str, Any],
    *,
    scenario_path: str | Path,
    endpoint: str,
    expected_checkpoint_sha256: str | None,
    assets_root: str | Path,
    trajectories_root: Path,
    device: str,
    execution: str,
    exec_horizon: int,
) -> None:
    from clothdojo.assets import GarmentLibrary
    from clothdojo.rollout.engine import run_chunk_episode, run_rtc_episode
    from clothdojo.rollout.newton import NewtonTrajectoryEpisode
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import (
        ShirtFlattenSpec,
        build_shirt_flatten,
        build_shirt_flatten_render_scene,
    )

    asset_id = job["asset_id"]
    pool = InitialStatePool.load(job["pool"])
    scenario_template = load_rollout_scenario(scenario_path)
    scenario = replace(scenario_template, garment_id=asset_id)
    assets = Path(assets_root).expanduser().resolve()
    garment = GarmentLibrary(assets).converted(asset_id)
    validate_pool_compatibility(pool, garment_path=garment, particle_count=pool.particle_count)
    policy = RemoteSFTPolicy(
        endpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        action_horizon=scenario.action_horizon,
    )
    spec = ShirtFlattenSpec(garment_id=asset_id, world_count=1)
    scene = build_shirt_flatten(assets, spec=spec, device=device)
    runtime = ShirtFlattenRuntime(scene)
    validate_pool_compatibility(
        pool,
        garment_path=garment,
        particle_count=runtime.grasp.particle_ranges()[0][1]
        - runtime.grasp.particle_ranges()[0][0],
    )
    render_scene = build_shirt_flatten_render_scene(assets, spec=spec, device=device)
    camera = ShirtFlattenCameraObservation(scene, render_scene)
    for index in range(job["state_count"]):
        name = f"{asset_id}_{index:06d}"
        target = trajectories_root / name
        if target.is_dir():
            _validate_existing_trajectory(
                target,
                asset_id=asset_id,
                initial_state_index=index,
                pool_sha256=job["pool_sha256"],
                checkpoint_sha256=policy.identity["checkpoint_sha256"],
            )
            continue
        staging = trajectories_root / f".{name}.{uuid.uuid4().hex}.tmp"
        runtime.reset()
        runtime.restore_particles(
            pool.particle_q[index : index + 1],
            pool.particle_qd[index : index + 1],
        )
        episode_scenario = replace(scenario, seed=int(pool.seeds[index]))
        try:
            with NewtonTrajectoryEpisode(
                runtime,
                staging,
                episode_scenario,
                assets_root=assets,
                observation_provider=camera,
                trajectory_metadata={
                    "producer": "clothdojo_rollout",
                    "asset_id": asset_id,
                    "policy_kind": policy.identity["backend"],
                    "policy": policy.identity,
                    "cloth_visual": "nocs",
                    "execution": _execution_settings(execution, exec_horizon),
                    "initial_state": {
                        "pool_sha256": job["pool_sha256"],
                        "index": index,
                        "seed": int(pool.seeds[index]),
                    },
                    "training_eligible": True,
                },
            ) as episode:
                report = (
                    run_chunk_episode(
                        episode,
                        policy,
                        episode_scenario,
                        exec_horizon=exec_horizon,
                    )
                    if execution == "chunk"
                    else run_rtc_episode(
                        episode,
                        policy,
                        episode_scenario,
                        exec_horizon=exec_horizon,
                    )
                )
            manifest_path = staging / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"]["execution_report"] = report.to_json_dict()
            _atomic_write_json(manifest_path, manifest)
            os.replace(staging, target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def _validate_existing_trajectory(
    path: Path,
    *,
    asset_id: str,
    initial_state_index: int,
    pool_sha256: str,
    checkpoint_sha256: str,
) -> None:
    reader = TrajectoryReader(path)
    reader.validate_training_data()
    metadata = reader.manifest.metadata
    actual = {
        "asset_id": metadata.get("asset_id"),
        "initial_state_index": metadata.get("initial_state", {}).get("index"),
        "pool_sha256": metadata.get("initial_state", {}).get("pool_sha256"),
        "checkpoint_sha256": metadata.get("policy", {}).get("checkpoint_sha256"),
        "cloth_visual": metadata.get("cloth_visual"),
    }
    expected = {
        "asset_id": asset_id,
        "initial_state_index": initial_state_index,
        "pool_sha256": pool_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "cloth_visual": "nocs",
    }
    if actual != expected:
        raise ValueError(f"existing rollout trajectory has different provenance: {path}")


def _collect_trajectories(
    root: Path,
    jobs: list[dict[str, Any]],
    *,
    partial: bool = False,
) -> list[dict[str, Any]]:
    trajectories = []
    for job in jobs:
        for index in range(job["state_count"]):
            relative = Path("trajectories") / f"{job['asset_id']}_{index:06d}"
            path = root / relative
            if not path.is_dir():
                if partial:
                    continue
                raise RuntimeError(f"rollout trajectory is missing: {path}")
            reader = TrajectoryReader(path)
            metadata = reader.manifest.metadata
            trajectories.append(
                {
                    "asset_id": job["asset_id"],
                    "initial_state_index": index,
                    "seed": metadata["initial_state"]["seed"],
                    "path": relative.as_posix(),
                    "action_count": reader.manifest.action_count,
                    "execution": metadata.get("execution_report", {}),
                }
            )
    return trajectories


def _rollout_summary(root: Path, meta: dict[str, Any], gpus: tuple[str, ...]) -> dict[str, Any]:
    return {
        "compatible": True,
        "output": str(root),
        "trajectory_count": len(meta["trajectories"]),
        "action_count": sum(item["action_count"] for item in meta["trajectories"]),
        "checkpoint_sha256": meta["policy"]["checkpoint_sha256"],
        "training_eligible": True,
        "gpus": list(gpus),
    }


def _execution_settings(mode: str, exec_horizon: int) -> dict[str, int | str]:
    return {
        "mode": mode,
        "exec_horizon": int(exec_horizon),
        **({"inference_delay_steps": 0} if mode == "rtc" else {}),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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
