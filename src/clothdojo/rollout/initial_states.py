"""Validated cloth initial-state pools and their Newton generator."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

INITIAL_STATE_SCHEMA = "clothdojo_initial_state_pool"
INITIAL_STATE_FORMAT_VERSION = 1
INITIAL_STATE_COLLECTION_SCHEMA = "clothdojo_initial_states"
INITIAL_STATE_COLLECTION_FORMAT_VERSION = 1
INITIALIZATION_METHOD = "hang_drop"
INITIALIZATION_HOLD_SECONDS = 1.5
INITIALIZATION_ANCHOR_CLEARANCE_M = 0.05


@dataclass(frozen=True, slots=True)
class InitialStateValidation:
    valid: np.ndarray
    p99_height_m: np.ndarray
    elevated_fraction: np.ndarray
    near_table_fraction: np.ndarray
    speed_rms_m_s: np.ndarray
    table_contact_count: np.ndarray
    robot_contact_count: np.ndarray
    reasons: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class InitialStatePool:
    particle_q: np.ndarray
    particle_qd: np.ndarray
    seeds: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        q = np.asarray(self.particle_q)
        qd = np.asarray(self.particle_qd)
        seeds = np.asarray(self.seeds)
        if q.ndim != 3 or q.shape[-1] != 3:
            raise ValueError(f"particle_q must have shape (states, particles, 3), got {q.shape}")
        if qd.shape != q.shape:
            raise ValueError("particle_qd shape must match particle_q")
        if q.dtype != np.float32 or qd.dtype != np.float32:
            raise TypeError("initial particle states must use float32")
        if seeds.shape != (len(q),) or seeds.dtype != np.int32:
            raise ValueError("seeds must be int32 with one value per state")
        if len(q) < 1 or len(np.unique(seeds)) != len(seeds):
            raise ValueError("initial-state pool must contain unique non-empty seeds")
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("initial-state pool contains non-finite values")
        if not isinstance(self.metadata, dict):
            raise TypeError("initial-state metadata must be a dict")

    @property
    def state_count(self) -> int:
        return len(self.seeds)

    @property
    def particle_count(self) -> int:
        return int(self.particle_q.shape[1])

    def select(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("initial-state indices must be a vector")
        if np.any(values < 0) or np.any(values >= self.state_count):
            raise IndexError("initial-state index is outside the pool")
        return (
            np.ascontiguousarray(self.particle_q[values]),
            np.ascontiguousarray(self.particle_qd[values]),
            np.ascontiguousarray(self.seeds[values]),
        )

    def save(self, output: str | Path) -> Path:
        destination = Path(output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"initial-state pool already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.metadata,
            "schema": INITIAL_STATE_SCHEMA,
            "format_version": INITIAL_STATE_FORMAT_VERSION,
            "state_count": self.state_count,
            "particle_count": self.particle_count,
        }
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    particle_q=self.particle_q,
                    particle_qd=self.particle_qd,
                    seeds=self.seeds,
                    metadata_json=np.asarray(
                        json.dumps(payload, sort_keys=True, separators=(",", ":"))
                    ),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, source: str | Path) -> InitialStatePool:
        path = Path(source).expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            required = {"particle_q", "particle_qd", "seeds", "metadata_json"}
            if set(data.files) != required:
                raise ValueError(f"initial-state fields differ: {sorted(data.files)}")
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.pop("schema", None) != INITIAL_STATE_SCHEMA:
                raise ValueError("unsupported initial-state pool schema")
            if metadata.pop("format_version", None) != INITIAL_STATE_FORMAT_VERSION:
                raise ValueError("unsupported initial-state pool version")
            state_count = metadata.pop("state_count", None)
            particle_count = metadata.pop("particle_count", None)
            pool = cls(
                particle_q=np.asarray(data["particle_q"], dtype=np.float32),
                particle_qd=np.asarray(data["particle_qd"], dtype=np.float32),
                seeds=np.asarray(data["seeds"], dtype=np.int32),
                metadata=metadata,
            )
        if state_count != pool.state_count or particle_count != pool.particle_count:
            raise ValueError("initial-state metadata disagrees with its arrays")
        return pool


def collect_seen_garments(source: str | Path) -> set[str]:
    """Return assets named by complete canonical expert trajectories."""

    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"expert trajectory root is missing: {root}")
    seen: set[str] = set()
    manifests = set(root.glob("trajectories/*/manifest.json"))
    manifests.update(root.glob("*/trajectories/*/manifest.json"))
    for path in sorted(manifests):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read expert trajectory manifest: {path}: {exc}") from exc
        if payload.get("status") != "complete":
            continue
        metadata = payload.get("metadata")
        asset_id = metadata.get("asset_id") if isinstance(metadata, dict) else None
        if isinstance(asset_id, str) and asset_id:
            seen.add(asset_id)
    sidecar_path = root / "meta" / "clothdojo.json"
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot read expert dataset metadata: {sidecar_path}: {exc}"
            ) from exc
        assets = sidecar.get("assets")
        if not isinstance(assets, list) or any(
            not isinstance(asset_id, str) or not asset_id for asset_id in assets
        ):
            raise ValueError(f"expert dataset asset list is invalid: {sidecar_path}")
        seen.update(assets)
    return seen


def select_garments(
    collectable_ids: tuple[str, ...] | list[str],
    seen_ids: set[str],
    *,
    seen_count: int,
    unseen_count: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Deterministically sample accepted and converted garments without replacement."""

    if min(seen_count, unseen_count) < 0:
        raise ValueError("seen_assets and unseen_assets must be non-negative")
    if seen_count + unseen_count < 1:
        raise ValueError("at least one asset must be selected")
    if isinstance(seed, bool) or not 0 <= seed <= np.iinfo(np.int32).max:
        raise ValueError("seed must be a non-negative int32")
    collectable = set(collectable_ids)
    seen = sorted(collectable & set(seen_ids))
    unseen = sorted(collectable - set(seen_ids))
    if seen_count > len(seen) or unseen_count > len(unseen):
        raise ValueError(
            "not enough collectable assets: "
            f"requested seen={seen_count}/{len(seen)}, "
            f"unseen={unseen_count}/{len(unseen)}"
        )
    generator = np.random.default_rng(seed)
    selected_seen = (
        [seen[index] for index in generator.choice(len(seen), seen_count, replace=False)]
        if seen_count
        else []
    )
    selected_unseen = (
        [unseen[index] for index in generator.choice(len(unseen), unseen_count, replace=False)]
        if unseen_count
        else []
    )
    selected = [(asset_id, "seen") for asset_id in selected_seen]
    selected.extend((asset_id, "unseen") for asset_id in selected_unseen)
    generator_seeds = _unique_seeds(generator, len(selected), set())
    return tuple(
        {
            "asset_id": asset_id,
            "split": split,
            "generator_seed": int(asset_seed),
        }
        for (asset_id, split), asset_seed in zip(selected, generator_seeds, strict=True)
    )


def load_initial_state_collection(source: str | Path, *, require_complete: bool = True) -> dict:
    root = Path(source).expanduser().resolve()
    meta_path = root / "meta.json"
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read initial-state metadata: {meta_path}: {exc}") from exc
    if payload.get("schema") != INITIAL_STATE_COLLECTION_SCHEMA:
        raise ValueError("unsupported initial-state collection schema")
    if payload.get("format_version") != INITIAL_STATE_COLLECTION_FORMAT_VERSION:
        raise ValueError("unsupported initial-state collection version")
    if require_complete and payload.get("status") != "complete":
        raise RuntimeError(f"initial-state collection is not complete: {payload.get('status')!r}")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("initial-state collection has no assets")
    ids = [item.get("asset_id") for item in assets if isinstance(item, dict)]
    if len(ids) != len(assets) or len(ids) != len(set(ids)):
        raise ValueError("initial-state collection asset list is invalid")
    for item in assets:
        expected = f"{item['asset_id']}/states.npz"
        if item.get("path") not in {None, expected}:
            raise ValueError("initial-state asset path is invalid")
        if require_complete:
            pool = InitialStatePool.load(root / expected)
            if pool.metadata.get("garment_id") != item["asset_id"]:
                raise ValueError("initial-state pool asset identity differs from meta.json")
            if pool.state_count != payload.get("states_per_asset"):
                raise ValueError("initial-state pool count differs from meta.json")
    return payload


def generate_initial_state_collection(
    assets_root: str | Path,
    output: str | Path,
    *,
    expert_trajectories: str | Path | None,
    seen_assets: int,
    unseen_assets: int,
    states_per_asset: int = 32,
    gpus: tuple[str, ...] = ("0",),
    candidate_batch: int = 8,
    max_candidates: int = 512,
    seed: int = 260815,
    drop_seconds: float = 3.0,
) -> dict[str, Any]:
    """Select assets once, then generate one atomic state pool per asset."""

    from clothdojo.assets import GarmentLibrary

    if states_per_asset < 1:
        raise ValueError("states_per_asset must be positive")
    if not gpus or any(not isinstance(gpu, str) or not gpu.strip() for gpu in gpus):
        raise ValueError("gpus must contain at least one device id")
    assets = Path(assets_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    expert_root = (
        None if expert_trajectories is None else Path(expert_trajectories).expanduser().resolve()
    )
    requested = {
        "assets_root": str(assets),
        "expert_trajectories": None if expert_root is None else str(expert_root),
        "seen_assets": int(seen_assets),
        "unseen_assets": int(unseen_assets),
        "states_per_asset": int(states_per_asset),
        "candidate_batch": int(candidate_batch),
        "max_candidates": int(max_candidates),
        "seed": int(seed),
        "drop_seconds": float(drop_seconds),
        "initialization": INITIALIZATION_METHOD,
    }
    meta_path = destination / "meta.json"
    if meta_path.is_file():
        meta = load_initial_state_collection(destination, require_complete=False)
        actual = {key: meta.get(key) for key in requested}
        if actual != requested:
            raise ValueError(f"existing initial-state settings differ: {actual} != {requested}")
    else:
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"initial-state output is not empty: {destination}")
        if seen_assets and expert_root is None:
            raise ValueError("expert_trajectories is required when seen_assets is positive")
        seen = set() if expert_root is None else collect_seen_garments(expert_root)
        selected = select_garments(
            GarmentLibrary(assets).collectable_ids(),
            seen,
            seen_count=seen_assets,
            unseen_count=unseen_assets,
            seed=seed,
        )
        destination.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": INITIAL_STATE_COLLECTION_SCHEMA,
            "format_version": INITIAL_STATE_COLLECTION_FORMAT_VERSION,
            "status": "generating",
            **requested,
            "assets": list(selected),
        }
        _atomic_write_json(meta_path, meta)

    pending = []
    for item in meta["assets"]:
        asset_id = item["asset_id"]
        pool_path = destination / asset_id / "states.npz"
        if pool_path.is_file():
            pool = InitialStatePool.load(pool_path)
            if pool.metadata.get("garment_id") != asset_id or pool.state_count != states_per_asset:
                raise ValueError(f"existing initial-state pool is incompatible: {pool_path}")
            continue
        pending.append(dict(item))

    if pending:
        try:
            _run_state_workers(
                pending,
                gpus=gpus,
                assets_root=assets,
                output=destination,
                states_per_asset=states_per_asset,
                candidate_batch=candidate_batch,
                max_candidates=max_candidates,
                drop_seconds=drop_seconds,
            )
        except Exception as exc:
            meta["status"] = "failed"
            meta["error"] = f"{type(exc).__name__}: {exc}"
            _atomic_write_json(meta_path, meta)
            raise

    summaries = []
    for item in meta["assets"]:
        relative = Path(item["asset_id"]) / "states.npz"
        pool_path = destination / relative
        pool = InitialStatePool.load(pool_path)
        summaries.append(
            {
                **item,
                "path": relative.as_posix(),
                "state_count": pool.state_count,
                "particle_count": pool.particle_count,
                "sha256": _sha256(pool_path),
            }
        )
    meta["assets"] = summaries
    meta["status"] = "complete"
    meta.pop("error", None)
    _atomic_write_json(meta_path, meta)
    return {
        "compatible": True,
        "output": str(destination),
        "asset_count": len(summaries),
        "seen_assets": seen_assets,
        "unseen_assets": unseen_assets,
        "state_count": len(summaries) * states_per_asset,
        "gpus": list(gpus),
    }


def _run_state_workers(
    jobs: list[dict[str, Any]],
    *,
    gpus: tuple[str, ...],
    assets_root: Path,
    output: Path,
    states_per_asset: int,
    candidate_batch: int,
    max_candidates: int,
    drop_seconds: float,
) -> None:
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    for job in jobs:
        job_queue.put(job)
    worker_count = min(len(gpus), len(jobs))
    for _ in range(worker_count):
        job_queue.put(None)
    workers = [
        context.Process(
            target=_state_worker,
            args=(
                job_queue,
                result_queue,
                gpu,
                str(assets_root),
                str(output),
                states_per_asset,
                candidate_batch,
                max_candidates,
                drop_seconds,
            ),
            name=f"clothdojo-states-gpu-{gpu}",
        )
        for gpu in gpus[:worker_count]
    ]
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
                        f"initial-state worker exited with code {failures[0].exitcode}"
                    ) from None
                continue
            if result["status"] == "error":
                raise RuntimeError(
                    f"initial-state generation failed for {result['asset_id']} on "
                    f"GPU {result['gpu']}: {result['error']}"
                )
            completed += 1
    finally:
        for worker in workers:
            if worker.is_alive() and completed < len(jobs):
                worker.terminate()
            worker.join()


def _state_worker(
    job_queue,
    result_queue,
    gpu: str,
    assets_root: str,
    output: str,
    states_per_asset: int,
    candidate_batch: int,
    max_candidates: int,
    drop_seconds: float,
) -> None:
    if gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cpu" if gpu == "cpu" else "cuda:0"
    while True:
        job = job_queue.get()
        if job is None:
            return
        asset_id = job["asset_id"]
        try:
            generate_initial_states(
                assets_root,
                Path(output) / asset_id / "states.npz",
                garment_id=asset_id,
                state_count=states_per_asset,
                candidate_batch=candidate_batch,
                max_candidates=max_candidates,
                seed=job["generator_seed"],
                drop_seconds=drop_seconds,
                device=device,
            )
        except Exception as exc:
            result_queue.put(
                {
                    "status": "error",
                    "asset_id": asset_id,
                    "gpu": gpu,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return
        result_queue.put({"status": "complete", "asset_id": asset_id, "gpu": gpu})


def validate_initial_states(
    particle_q: np.ndarray,
    particle_qd: np.ndarray,
    *,
    table_top_m: float,
    maximum_p99_height_above_table_m: float = 0.25,
    elevated_height_above_table_m: float = 0.40,
    maximum_elevated_fraction: float = 0.002,
    contact_height_above_table_m: float = 0.03,
    minimum_near_table_fraction: float = 0.01,
    maximum_speed_rms_m_s: float = 0.20,
    table_contact_count: np.ndarray | None = None,
    robot_contact_count: np.ndarray | None = None,
) -> InitialStateValidation:
    """Reject numerical failures and cloth that remains suspended after settling."""

    q = np.asarray(particle_q)
    qd = np.asarray(particle_qd)
    if q.ndim != 3 or q.shape[-1] != 3 or qd.shape != q.shape:
        raise ValueError("particle_q/particle_qd must share shape (states, particles, 3)")
    if q.dtype != np.float32 or qd.dtype != np.float32:
        raise TypeError("validated particle states must use float32")
    values = {
        "table_top_m": table_top_m,
        "maximum_p99_height_above_table_m": maximum_p99_height_above_table_m,
        "elevated_height_above_table_m": elevated_height_above_table_m,
        "maximum_elevated_fraction": maximum_elevated_fraction,
        "contact_height_above_table_m": contact_height_above_table_m,
        "minimum_near_table_fraction": minimum_near_table_fraction,
        "maximum_speed_rms_m_s": maximum_speed_rms_m_s,
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("initial-state validation thresholds must be finite")
    if (
        maximum_p99_height_above_table_m <= 0
        or elevated_height_above_table_m <= 0
        or contact_height_above_table_m <= 0
    ):
        raise ValueError("initial-state height thresholds must be positive")
    if (
        not 0 <= maximum_elevated_fraction <= 1
        or not 0 <= minimum_near_table_fraction <= 1
        or maximum_speed_rms_m_s < 0
    ):
        raise ValueError("initial-state fraction/speed thresholds are invalid")

    finite = np.isfinite(q).all(axis=(1, 2)) & np.isfinite(qd).all(axis=(1, 2))
    safe_q = np.where(np.isfinite(q), q, np.float32(table_top_m))
    safe_qd = np.where(np.isfinite(qd), qd, np.float32(0.0))
    p99 = np.quantile(safe_q[..., 2], 0.99, axis=1)
    elevated = np.mean(
        safe_q[..., 2] > table_top_m + elevated_height_above_table_m,
        axis=1,
    )
    near_table = np.mean(
        safe_q[..., 2] <= table_top_m + contact_height_above_table_m,
        axis=1,
    )
    speed_rms = np.sqrt(np.mean(np.sum(safe_qd * safe_qd, axis=2), axis=1))
    table_contacts = _contact_count_vector(table_contact_count, len(q), "table_contact_count")
    robot_contacts = _contact_count_vector(robot_contact_count, len(q), "robot_contact_count")
    height_ok = p99 <= table_top_m + maximum_p99_height_above_table_m
    elevated_ok = elevated <= maximum_elevated_fraction
    contact_ok = near_table >= minimum_near_table_fraction
    speed_ok = speed_rms <= maximum_speed_rms_m_s
    table_contact_ok = np.ones(len(q), dtype=bool) if table_contacts is None else table_contacts > 0
    robot_contact_ok = (
        np.ones(len(q), dtype=bool) if robot_contacts is None else robot_contacts == 0
    )
    valid = (
        finite
        & height_ok
        & elevated_ok
        & contact_ok
        & speed_ok
        & table_contact_ok
        & robot_contact_ok
    )
    reasons = []
    for index in range(len(q)):
        current = []
        if not finite[index]:
            current.append("non_finite")
        if not height_ok[index]:
            current.append("p99_height")
        if not elevated_ok[index]:
            current.append("elevated_fraction")
        if not contact_ok[index]:
            current.append("no_table_contact")
        if not speed_ok[index]:
            current.append("unsettled_speed")
        if not table_contact_ok[index]:
            current.append("no_physical_table_contact")
        if not robot_contact_ok[index]:
            current.append("robot_contact")
        reasons.append(tuple(current))
    return InitialStateValidation(
        valid=np.asarray(valid, dtype=bool),
        p99_height_m=np.asarray(p99, dtype=np.float32),
        elevated_fraction=np.asarray(elevated, dtype=np.float32),
        near_table_fraction=np.asarray(near_table, dtype=np.float32),
        speed_rms_m_s=np.asarray(speed_rms, dtype=np.float32),
        table_contact_count=(
            np.full(len(q), -1, dtype=np.int32) if table_contacts is None else table_contacts
        ),
        robot_contact_count=(
            np.full(len(q), -1, dtype=np.int32) if robot_contacts is None else robot_contacts
        ),
        reasons=tuple(reasons),
    )


def _contact_count_vector(value, state_count: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value)
    if result.shape != (state_count,) or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must be an integer vector with one value per state")
    if np.any(result < 0):
        raise ValueError(f"{name} cannot contain negative values")
    return np.asarray(result, dtype=np.int32)


def validate_pool_compatibility(
    pool: InitialStatePool,
    *,
    garment_path: str | Path,
    particle_count: int,
) -> None:
    if pool.particle_count != int(particle_count):
        raise ValueError(
            "initial-state particle count differs from the garment: "
            f"{pool.particle_count} != {particle_count}"
        )
    expected = pool.metadata.get("garment_sha256")
    if not isinstance(expected, str) or _sha256(Path(garment_path)) != expected:
        raise ValueError("initial-state pool belongs to a different garment asset")


def generate_initial_states(
    assets_root: str | Path,
    output: str | Path,
    *,
    garment_id: str,
    state_count: int = 32,
    candidate_batch: int = 8,
    max_candidates: int = 512,
    seed: int = 260815,
    drop_seconds: float = 3.0,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Generate one validated, asset-specific hang-drop state pool."""

    from clothdojo.assets import GarmentLibrary
    from clothdojo.sim.grasp import StaticParticlePatch
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_shirt_flatten

    if min(state_count, candidate_batch, max_candidates) < 1:
        raise ValueError("state_count, candidate_batch and max_candidates must be positive")
    if max_candidates < state_count or max_candidates % candidate_batch:
        raise ValueError("max_candidates must cover state_count in complete candidate batches")
    if isinstance(seed, bool) or not 0 <= seed <= np.iinfo(np.int32).max:
        raise ValueError("seed must be a non-negative int32")
    if drop_seconds <= 0:
        raise ValueError("drop_seconds must be positive")

    assets = Path(assets_root).expanduser().resolve()
    garment = GarmentLibrary(assets).converted(garment_id)
    spec = ShirtFlattenSpec(garment_id=garment_id, world_count=candidate_batch)
    scene = build_shirt_flatten(assets, spec=spec, device=device)
    runtime = ShirtFlattenRuntime(scene)
    ranges = runtime.grasp.particle_ranges()
    particle_counts = {end - start for start, end in ranges}
    if len(particle_counts) != 1:
        raise RuntimeError("candidate worlds have different garment topology")
    generator = np.random.default_rng(seed)
    used_seeds: set[int] = set()
    accepted_q: list[np.ndarray] = []
    accepted_qd: list[np.ndarray] = []
    accepted_seeds: list[int] = []
    accepted_metrics: list[dict[str, float]] = []
    rejected: Counter[str] = Counter()
    candidates = 0
    table_top = spec.table_position[2] + spec.table_half_extents[2]
    cloth_step_graph = None
    static_patch = StaticParticlePatch(scene)

    while len(accepted_q) < state_count:
        if candidates >= max_candidates:
            raise RuntimeError(
                f"accepted {len(accepted_q)}/{state_count} states after {candidates} "
                f"candidates; rejected={dict(rejected)}"
            )
        seeds = _unique_seeds(generator, candidate_batch, used_seeds)
        used_seeds.update(map(int, seeds))
        runtime.reset()
        q, patches, attachments = _hang_candidates(
            runtime,
            ranges,
            seeds,
            table_top=table_top,
        )
        runtime.restore_particles(q, np.zeros_like(q))
        static_patch.configure(patches, runtime.state_0.particle_q)
        cloth_step_graph = _advance_cloth(
            runtime,
            round(INITIALIZATION_HOLD_SECONDS * spec.control_fps),
            cloth_step_graph,
            static_patch,
        )
        static_patch.disable()
        runtime.cloth.rebuild_bvh(runtime.state_0)
        cloth_step_graph = _advance_cloth(
            runtime,
            round(drop_seconds * spec.control_fps),
            cloth_step_graph,
            static_patch,
        )

        final_q, final_qd = _capture_particles(runtime, ranges)
        table_contacts, robot_contacts = _cloth_contact_counts(runtime)
        validation = validate_initial_states(
            final_q,
            final_qd,
            table_top_m=table_top,
            table_contact_count=table_contacts,
            robot_contact_count=robot_contacts,
        )
        candidates += candidate_batch
        for index, valid in enumerate(validation.valid):
            if not valid:
                rejected.update(validation.reasons[index])
                continue
            if len(accepted_q) >= state_count:
                break
            accepted_q.append(final_q[index].copy())
            accepted_qd.append(final_qd[index].copy())
            accepted_seeds.append(int(seeds[index]))
            accepted_metrics.append(
                {
                    "p99_height_m": float(validation.p99_height_m[index]),
                    "elevated_fraction": float(validation.elevated_fraction[index]),
                    "near_table_fraction": float(validation.near_table_fraction[index]),
                    "speed_rms_m_s": float(validation.speed_rms_m_s[index]),
                    "table_contact_count": int(validation.table_contact_count[index]),
                    "robot_contact_count": int(validation.robot_contact_count[index]),
                    **attachments[index],
                }
            )
        print(
            f"garment={garment_id} accepted={len(accepted_q)}/{state_count} "
            f"candidates={candidates} rejected={dict(rejected)} "
            f"speed_rms={validation.speed_rms_m_s.tolist()}",
            file=sys.stderr,
            flush=True,
        )

    metadata = {
        "garment_id": garment_id,
        "garment_path": str(garment),
        "garment_sha256": _sha256(garment),
        "scene_sha256": _json_sha256(asdict(spec) | {"world_count": 1}),
        "initialization": INITIALIZATION_METHOD,
        "generator_seed": seed,
        "candidate_count": candidates,
        "rejection_counts": dict(rejected),
        "hold_seconds": INITIALIZATION_HOLD_SECONDS,
        "drop_seconds": drop_seconds,
        "anchor_clearance_m": INITIALIZATION_ANCHOR_CLEARANCE_M,
        "accepted_metrics": accepted_metrics,
    }
    pool = InitialStatePool(
        particle_q=np.ascontiguousarray(np.stack(accepted_q), dtype=np.float32),
        particle_qd=np.ascontiguousarray(np.stack(accepted_qd), dtype=np.float32),
        seeds=np.asarray(accepted_seeds, dtype=np.int32),
        metadata=metadata,
    )
    saved = pool.save(output)
    return {
        "compatible": True,
        "output": str(saved),
        "garment_id": garment_id,
        "state_count": pool.state_count,
        "particle_count": pool.particle_count,
        "candidate_count": candidates,
        "rejection_counts": dict(rejected),
        "device": device,
    }


def _hang_candidates(runtime, ranges, seeds, *, table_top: float):
    base = runtime.state_0.particle_q.numpy()
    q = []
    patches = []
    attachments = []
    for (start, end), seed in zip(ranges, seeds, strict=True):
        points = np.asarray(base[start:end], dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        rotation = _haar_rotation(rng)
        rotated = (points - points.mean(axis=0)) @ rotation.T
        local_particle = int(rng.integers(0, end - start))
        particle = start + local_particle
        patch = runtime.grasp.connected_patch(particle)
        diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
        anchor = np.asarray(
            (
                points[:, 0].mean(),
                points[:, 1].mean(),
                table_top + diagonal + INITIALIZATION_ANCHOR_CLEARANCE_M,
            )
        )
        rotated += anchor - rotated[local_particle]
        q.append(rotated.astype(np.float32))
        patches.append(patch)
        attachments.append(
            {
                "hang_particle_index": local_particle,
                "hang_patch_size": len(patch),
                "anchor_position": [float(value) for value in anchor],
            }
        )
    return np.ascontiguousarray(np.stack(q), dtype=np.float32), patches, attachments


def _cloth_contact_counts(runtime) -> tuple[np.ndarray, np.ndarray]:
    runtime.collision.collide(runtime.state_0, runtime.contacts)
    count = int(runtime.contacts.soft_contact_count.numpy()[0])
    count = min(count, len(runtime.contacts.soft_contact_particle))
    particles = runtime.contacts.soft_contact_particle.numpy()[:count]
    shapes = runtime.contacts.soft_contact_shape.numpy()[:count]
    shape_bodies = runtime.model.shape_body.numpy()
    particle_world = runtime.model.particle_world.numpy()
    table = np.zeros(runtime.model.world_count, dtype=np.int32)
    robot = np.zeros(runtime.model.world_count, dtype=np.int32)
    for particle, shape in zip(particles, shapes, strict=True):
        particle = int(particle)
        shape = int(shape)
        if particle < 0 or shape < 0:
            continue
        world = int(particle_world[particle])
        target = table if int(shape_bodies[shape]) < 0 else robot
        target[world] += 1
    return table, robot


def _haar_rotation(generator: np.random.Generator) -> np.ndarray:
    quaternion = generator.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _capture_particles(runtime, ranges):
    q = runtime.state_0.particle_q.numpy()
    qd = runtime.state_0.particle_qd.numpy()
    return (
        np.ascontiguousarray(np.stack([q[start:end] for start, end in ranges]), dtype=np.float32),
        np.ascontiguousarray(np.stack([qd[start:end] for start, end in ranges]), dtype=np.float32),
    )


def _advance_cloth(runtime, steps: int, graph, static_patch):
    import warp as wp

    if steps < 1:
        return graph
    return _advance_with_graph(runtime, steps, graph, static_patch, wp)


def _advance_with_graph(runtime, steps: int, graph, static_patch, wp):
    if steps < 1:
        return graph
    if graph is None:
        # Compile every lazy kernel once before capture. The captured control
        # step has an even number of state swaps, so it retains the same bound
        # state arrays on every replay.
        runtime.step_cloth(static_patch=static_patch)
        steps -= 1
        if steps:
            with wp.ScopedCapture(device=runtime.model.device) as capture:
                runtime.step_cloth(static_patch=static_patch)
            graph = capture.graph
            steps -= 1
    for _ in range(steps):
        wp.capture_launch(graph)
    return graph


def _unique_seeds(generator, count: int, excluded: set[int]) -> np.ndarray:
    values = []
    used = set(excluded)
    while len(values) < count:
        seed = int(generator.integers(0, np.iinfo(np.int32).max))
        if seed not in used:
            values.append(seed)
            used.add(seed)
    return np.asarray(values, dtype=np.int32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
