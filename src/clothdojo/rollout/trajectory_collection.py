"""One trajectory collection runner shared by rollout and evaluation plans."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.rendering import POLICY_RENDER_SIZE, policy_observation_settings
from clothdojo.rollout.initial_states import InitialStatePool, validate_pool_compatibility
from clothdojo.rollout.policy import RemotePolicy
from clothdojo.rollout.scenario import load_rollout_scenario
from clothdojo.rollout.scheduling import run_policy_asset_workers
from clothdojo.trajectory import TrajectoryReader

_BASE_METADATA_KEYS = frozenset(
    {
        "asset_id",
        "policy_kind",
        "policy",
        "policy_seed",
        "cloth_visual",
        "observation",
        "execution",
        "parallelism",
        "initial_state",
        "replicate",
    }
)


@dataclass(frozen=True, slots=True)
class TrajectoryAssetJob:
    """All cases for one asset and one immutable initial-state pool."""

    asset_id: str
    pool: str
    pool_sha256: str
    state_count: int
    replicates: tuple[int | None, ...] = (None,)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("trajectory asset id must not be empty")
        if (
            isinstance(self.state_count, bool)
            or not isinstance(self.state_count, int)
            or self.state_count < 1
        ):
            raise ValueError("trajectory state count must be positive")
        if not self.replicates:
            raise ValueError("trajectory job must contain at least one replicate")
        concrete = [value for value in self.replicates if value is not None]
        if len(concrete) != len(set(concrete)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in concrete
        ):
            raise ValueError("trajectory replicates must be unique non-negative integers")
        if None in self.replicates and self.replicates != (None,):
            raise ValueError("the unreplicated trajectory layout cannot mix with replicates")
        overlap = _BASE_METADATA_KEYS & set(self.metadata)
        if overlap:
            raise ValueError(f"trajectory job metadata contains reserved keys: {sorted(overlap)}")


@dataclass(frozen=True, slots=True)
class TrajectoryCollectionSpec:
    """Execution and publication settings common to every asset worker."""

    scenario_path: str
    assets_root: str
    trajectories_root: str
    expected_checkpoint_sha256: str | None
    policy_identity: dict[str, Any]
    execution: str
    exec_horizon: int
    rtc_delay_steps: int
    horizon_steps: int
    batch_size: int
    producer: str
    training_eligible: bool
    compressed: bool = True
    cloth_visual: str = "nocs"

    def __post_init__(self) -> None:
        if self.cloth_visual not in {"nocs", "white"}:
            raise ValueError("cloth_visual must be 'nocs' or 'white'")
        if self.execution not in {"chunk", "rtc"}:
            raise ValueError("trajectory execution must be 'chunk' or 'rtc'")
        for name, value in (
            ("exec_horizon", self.exec_horizon),
            ("horizon_steps", self.horizon_steps),
            ("batch_size", self.batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"trajectory {name} must be a positive integer")
        if (
            isinstance(self.rtc_delay_steps, bool)
            or not isinstance(self.rtc_delay_steps, int)
            or self.rtc_delay_steps < 0
        ):
            raise ValueError("trajectory rtc_delay_steps must be a non-negative integer")
        if self.execution == "chunk" and self.rtc_delay_steps:
            raise ValueError("chunk trajectories do not accept an RTC delay")
        if self.execution == "rtc" and self.rtc_delay_steps >= self.exec_horizon:
            raise ValueError("trajectory RTC delay must be smaller than exec_horizon")
        if not self.producer:
            raise ValueError("trajectory producer must not be empty")
        if not isinstance(self.training_eligible, bool):
            raise TypeError("trajectory training_eligible must be a bool")
        if not isinstance(self.compressed, bool):
            raise TypeError("trajectory compressed must be a bool")


def collect_trajectories(
    jobs: list[TrajectoryAssetJob],
    *,
    gpus: tuple[str, ...],
    policy_endpoints: tuple[str, ...],
    spec: TrajectoryCollectionSpec,
) -> None:
    """Collect every missing trajectory with one asset worker per simulation GPU."""

    pending = [job for job in jobs if asset_has_missing_trajectories(job, spec=spec)]
    if not pending:
        return
    worker_count = min(len(gpus), len(pending))
    run_policy_asset_workers(
        pending,
        gpus=gpus[:worker_count],
        policy_endpoints=policy_endpoints[:worker_count],
        worker_target=_trajectory_worker,
        common_args=(spec,),
        process_kind="trajectory collection",
    )


def collect_trajectory_index(
    jobs: list[TrajectoryAssetJob],
    *,
    spec: TrajectoryCollectionSpec,
    partial: bool = False,
) -> list[dict[str, Any]]:
    """Build the canonical flat trajectory index, validating provenance on the way."""

    root = Path(spec.trajectories_root)
    trajectories = []
    for job in jobs:
        for replicate in job.replicates:
            for index in range(job.state_count):
                relative = Path("trajectories") / trajectory_name(
                    job.asset_id, index, replicate
                )
                path = root / trajectory_name(job.asset_id, index, replicate)
                if not path.is_dir():
                    if partial:
                        continue
                    raise RuntimeError(f"trajectory is missing: {path}")
                try:
                    validate_existing_trajectory(
                        path,
                        job,
                        spec=spec,
                        initial_state_index=index,
                        replicate=replicate,
                        validate_arrays=False,
                    )
                    reader = TrajectoryReader(path)
                    metadata = reader.manifest.metadata
                except (OSError, RuntimeError, TypeError, ValueError, KeyError, EOFError):
                    if partial:
                        continue
                    raise
                item = {
                    "asset_id": job.asset_id,
                    "initial_state_index": index,
                    "seed": metadata["initial_state"]["seed"],
                    "policy_seed": metadata["policy_seed"],
                    "path": relative.as_posix(),
                    "action_count": reader.manifest.action_count,
                    "execution_report": metadata.get("execution_report", {}),
                }
                if replicate is not None:
                    item["replicate"] = replicate
                trajectories.append(item)
    return trajectories


def asset_has_missing_trajectories(
    job: TrajectoryAssetJob,
    *,
    spec: TrajectoryCollectionSpec,
) -> bool:
    root = Path(spec.trajectories_root)
    missing = False
    for replicate in job.replicates:
        for index in range(job.state_count):
            path = root / trajectory_name(job.asset_id, index, replicate)
            if not path.is_dir():
                missing = True
                continue
            validate_existing_trajectory(
                path,
                job,
                spec=spec,
                initial_state_index=index,
                replicate=replicate,
            )
    return missing


def validate_existing_trajectory(
    path: Path,
    job: TrajectoryAssetJob,
    *,
    spec: TrajectoryCollectionSpec,
    initial_state_index: int,
    replicate: int | None,
    validate_arrays: bool = True,
) -> None:
    reader = TrajectoryReader(path)
    if validate_arrays:
        reader.validate_training_data()
    metadata = reader.manifest.metadata
    expected = {
        "producer": spec.producer,
        "training_eligible": spec.training_eligible,
        "asset_id": job.asset_id,
        "initial_state_index": initial_state_index,
        "pool_sha256": job.pool_sha256,
        "policy": spec.policy_identity,
        "replicate": replicate,
        "execution": execution_settings(
            spec.execution,
            spec.exec_horizon,
            rtc_delay_steps=spec.rtc_delay_steps,
        ),
        "cloth_visual": spec.cloth_visual,
        "observation": policy_observation_settings(),
        **job.metadata,
    }
    actual = {
        "producer": metadata.get("producer"),
        "training_eligible": metadata.get("training_eligible"),
        "asset_id": metadata.get("asset_id"),
        "initial_state_index": metadata.get("initial_state", {}).get("index"),
        "pool_sha256": metadata.get("initial_state", {}).get("pool_sha256"),
        "policy": metadata.get("policy"),
        "replicate": metadata.get("replicate"),
        "execution": metadata.get("execution"),
        "cloth_visual": metadata.get("cloth_visual"),
        "observation": metadata.get("observation"),
        **{key: metadata.get(key) for key in job.metadata},
    }
    if actual != expected:
        raise ValueError(f"existing trajectory has different provenance: {path}")
    if reader.manifest.action_count != spec.horizon_steps:
        raise ValueError(f"existing trajectory has the wrong horizon: {path}")


def trajectory_name(asset_id: str, index: int, replicate: int | None) -> str:
    suffix = "" if replicate is None else f"_r{replicate:03d}"
    return f"{asset_id}_{index:06d}{suffix}"


def policy_seed_for_batch(seeds: np.ndarray, replicate: int | None = None) -> int:
    """Derive one stable Torch seed for an ordered world batch."""

    values = np.asarray(seeds, dtype=np.int32)
    if values.ndim != 1 or not len(values) or np.any(values < 0):
        raise ValueError("batch seeds must be a non-empty non-negative int32 vector")
    if replicate is not None and replicate < 0:
        raise ValueError("replicate must be non-negative")
    if replicate is None and len(values) == 1:
        return int(values[0])
    payload = values.astype("<i4", copy=False).tobytes()
    if replicate is not None:
        payload += np.asarray(replicate, dtype="<i4").tobytes()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def execution_settings(
    mode: str,
    exec_horizon: int,
    *,
    rtc_delay_steps: int = 0,
) -> dict[str, int | str]:
    return {
        "mode": mode,
        "exec_horizon": int(exec_horizon),
        **(
            {
                "scheduler": "fixed_logical_delay",
                "inference_delay_steps": int(rtc_delay_steps),
            }
            if mode == "rtc"
            else {}
        ),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _trajectory_worker(job_queue, result_queue, gpu, endpoint, spec) -> None:
    if gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cpu" if gpu == "cpu" else "cuda:0"
    while True:
        job = job_queue.get()
        if job is None:
            return
        try:
            _run_asset(job, endpoint=endpoint, device=device, spec=spec)
        except Exception as exc:
            result_queue.put(
                {
                    "status": "error",
                    "asset_id": job.asset_id,
                    "gpu": gpu,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return
        result_queue.put({"status": "complete", "asset_id": job.asset_id, "gpu": gpu})


def _run_asset(
    job: TrajectoryAssetJob,
    *,
    endpoint: str,
    device: str,
    spec: TrajectoryCollectionSpec,
) -> None:
    from clothdojo.assets import GarmentLibrary
    from clothdojo.rollout.engine import run_episode
    from clothdojo.rollout.newton import NewtonTrajectoryEpisode
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

    pool = InitialStatePool.load(job.pool)
    scenario = replace(load_rollout_scenario(spec.scenario_path), garment_id=job.asset_id)
    assets = Path(spec.assets_root)
    garment = GarmentLibrary(assets).converted(job.asset_id)
    validate_pool_compatibility(pool, garment_path=garment, particle_count=pool.particle_count)
    policy = RemotePolicy(
        endpoint,
        expected_checkpoint_sha256=spec.expected_checkpoint_sha256,
        expected_image_preprocessing=spec.policy_identity.get("image_preprocessing", "none"),
        action_horizon=scenario.action_horizon,
    )
    if policy.identity != spec.policy_identity:
        raise ValueError("trajectory worker policy identity differs from the collection plan")
    world_count = min(spec.batch_size, job.state_count)
    scene = build_scene(
        assets,
        spec=ShirtFlattenSpec(garment_id=job.asset_id, world_count=world_count),
        device=device,
        visuals=True,
        cloth_visual=spec.cloth_visual,
    )
    runtime = ShirtFlattenRuntime(scene)
    validate_pool_compatibility(
        pool,
        garment_path=garment,
        particle_count=runtime.grasp.particle_ranges()[0][1]
        - runtime.grasp.particle_ranges()[0][0],
    )
    camera = ShirtFlattenCameraObservation(
        scene,
        width=POLICY_RENDER_SIZE[0],
        height=POLICY_RENDER_SIZE[1],
    )
    root = Path(spec.trajectories_root)
    for replicate in job.replicates:
        for batch_start in range(0, job.state_count, world_count):
            active = list(range(batch_start, min(batch_start + world_count, job.state_count)))
            world_indices = active + [active[-1]] * (world_count - len(active))
            targets = [
                root / trajectory_name(job.asset_id, index, replicate) for index in active
            ]
            missing = []
            for index, target in zip(active, targets, strict=True):
                if target.is_dir():
                    validate_existing_trajectory(
                        target,
                        job,
                        spec=spec,
                        initial_state_index=index,
                        replicate=replicate,
                    )
                else:
                    missing.append(index)
            if not missing:
                continue
            staging_by_index = {
                index: root
                / f".{trajectory_name(job.asset_id, index, replicate)}.{uuid.uuid4().hex}.tmp"
                for index in missing
            }
            policy_seed = policy_seed_for_batch(
                np.asarray(pool.seeds[world_indices], dtype=np.int32), replicate
            )
            runtime.reset()
            runtime.restore_particles(
                np.ascontiguousarray(pool.particle_q[world_indices]),
                np.ascontiguousarray(pool.particle_qd[world_indices]),
            )
            scenarios = tuple(
                replace(scenario, seed=int(pool.seeds[index])) for index in world_indices
            )
            outputs = tuple(
                staging_by_index.get(index) if world < len(active) else None
                for world, index in enumerate(world_indices)
            )
            metadata = tuple(
                {
                    "asset_id": job.asset_id,
                    "policy_kind": policy.identity["backend"],
                    "policy": policy.identity,
                    "policy_seed": policy_seed,
                    "cloth_visual": spec.cloth_visual,
                    "observation": policy_observation_settings(),
                    "execution": execution_settings(
                        spec.execution,
                        spec.exec_horizon,
                        rtc_delay_steps=spec.rtc_delay_steps,
                    ),
                    "parallelism": {
                        "worlds": world_count,
                        "active_worlds": len(active),
                        "batch_start": batch_start,
                    },
                    "initial_state": {
                        "pool_sha256": job.pool_sha256,
                        "index": index,
                        "seed": int(pool.seeds[index]),
                    },
                    **({"replicate": replicate} if replicate is not None else {}),
                    **job.metadata,
                }
                for index in world_indices
            )
            try:
                batch_started = time.perf_counter()
                with NewtonTrajectoryEpisode(
                    runtime,
                    outputs,
                    scenarios,
                    assets_root=assets,
                    observation_provider=camera,
                    trajectory_metadata=metadata,
                    producer=spec.producer,
                    training_eligible=spec.training_eligible,
                    compressed=spec.compressed,
                ) as episode:
                    report = run_episode(
                        episode,
                        policy,
                        scenario,
                        mode=spec.execution,
                        exec_horizon=spec.exec_horizon,
                        policy_seed=policy_seed,
                        rtc_delay_steps=spec.rtc_delay_steps,
                    )
                batch_wall_seconds = time.perf_counter() - batch_started
                for index, target in zip(active, targets, strict=True):
                    staging = staging_by_index.get(index)
                    if staging is None:
                        continue
                    manifest_path = staging / "manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["metadata"]["execution_report"] = report.to_json_dict()
                    manifest["metadata"]["trajectory_publication"] = {
                        "batch_wall_seconds": batch_wall_seconds,
                        "finalize_seconds": max(
                            0.0, batch_wall_seconds - report.timings.wall_seconds
                        ),
                    }
                    atomic_write_json(manifest_path, manifest)
                    os.replace(staging, target)
            finally:
                for staging in staging_by_index.values():
                    shutil.rmtree(staging, ignore_errors=True)
