"""Validated cloth initial-state pools and their Newton generator."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

INITIAL_STATE_SCHEMA = "clothdojo_initial_state_pool"
INITIAL_STATE_FORMAT_VERSION = 1
INITIALIZATION_PROTOCOL = "haar_hang_drop_v1"


@dataclass(frozen=True, slots=True)
class InitialStateValidation:
    valid: np.ndarray
    p99_height_m: np.ndarray
    elevated_fraction: np.ndarray
    speed_rms_m_s: np.ndarray
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


def validate_initial_states(
    particle_q: np.ndarray,
    particle_qd: np.ndarray,
    *,
    table_top_m: float,
    maximum_p99_height_above_table_m: float = 0.25,
    elevated_height_above_table_m: float = 0.40,
    maximum_elevated_fraction: float = 0.002,
    maximum_speed_rms_m_s: float = 0.20,
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
        "maximum_speed_rms_m_s": maximum_speed_rms_m_s,
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("initial-state validation thresholds must be finite")
    if maximum_p99_height_above_table_m <= 0 or elevated_height_above_table_m <= 0:
        raise ValueError("initial-state height thresholds must be positive")
    if not 0 <= maximum_elevated_fraction <= 1 or maximum_speed_rms_m_s < 0:
        raise ValueError("initial-state fraction/speed thresholds are invalid")

    finite = np.isfinite(q).all(axis=(1, 2)) & np.isfinite(qd).all(axis=(1, 2))
    safe_q = np.where(np.isfinite(q), q, np.float32(table_top_m))
    safe_qd = np.where(np.isfinite(qd), qd, np.float32(0.0))
    p99 = np.quantile(safe_q[..., 2], 0.99, axis=1)
    elevated = np.mean(
        safe_q[..., 2] > table_top_m + elevated_height_above_table_m,
        axis=1,
    )
    speed_rms = np.sqrt(np.mean(np.sum(safe_qd * safe_qd, axis=2), axis=1))
    height_ok = p99 <= table_top_m + maximum_p99_height_above_table_m
    elevated_ok = elevated <= maximum_elevated_fraction
    speed_ok = speed_rms <= maximum_speed_rms_m_s
    valid = finite & height_ok & elevated_ok & speed_ok
    reasons = []
    for index in range(len(q)):
        current = []
        if not finite[index]:
            current.append("non_finite")
        if not height_ok[index]:
            current.append("p99_height")
        if not elevated_ok[index]:
            current.append("elevated_fraction")
        if not speed_ok[index]:
            current.append("unsettled_speed")
        reasons.append(tuple(current))
    return InitialStateValidation(
        valid=np.asarray(valid, dtype=bool),
        p99_height_m=np.asarray(p99, dtype=np.float32),
        elevated_fraction=np.asarray(elevated, dtype=np.float32),
        speed_rms_m_s=np.asarray(speed_rms, dtype=np.float32),
        reasons=tuple(reasons),
    )


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
    hold_seconds: float = 1.5,
    settle_seconds: float = 2.5,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Generate one validated, asset-specific hang-drop state pool."""

    from clothdojo.assets import GarmentLibrary
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_shirt_flatten

    if min(state_count, candidate_batch, max_candidates) < 1:
        raise ValueError("state_count, candidate_batch and max_candidates must be positive")
    if max_candidates < state_count or max_candidates % candidate_batch:
        raise ValueError("max_candidates must cover state_count in complete candidate batches")
    if isinstance(seed, bool) or not 0 <= seed <= np.iinfo(np.int32).max:
        raise ValueError("seed must be a non-negative int32")
    if hold_seconds < 0 or settle_seconds <= 0:
        raise ValueError("hold_seconds must be non-negative and settle_seconds positive")

    assets = Path(assets_root).expanduser().resolve()
    garment = GarmentLibrary(assets).converted(garment_id)
    spec = ShirtFlattenSpec(garment_id=garment_id, world_count=candidate_batch)
    scene = build_shirt_flatten(assets, spec=spec, device=device)
    runtime = ShirtFlattenRuntime(scene)
    ranges = runtime.grasp.particle_ranges()
    particle_counts = {end - start for start, end in ranges}
    if len(particle_counts) != 1:
        raise RuntimeError("candidate worlds have different garment topology")
    neighbors = _local_particle_neighbors(runtime, ranges[0])
    default_flags = runtime.model.particle_flags.numpy().copy()
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

    while len(accepted_q) < state_count:
        if candidates >= max_candidates:
            raise RuntimeError(
                f"accepted {len(accepted_q)}/{state_count} states after {candidates} "
                f"candidates; rejected={dict(rejected)}"
            )
        seeds = _unique_seeds(generator, candidate_batch, used_seeds)
        used_seeds.update(map(int, seeds))
        runtime.reset()
        q, pinned = _hang_drop_candidates(runtime, ranges, seeds, table_top=table_top)
        runtime.restore_particles(q, np.zeros_like(q))
        flags = default_flags.copy()
        for world, local_particle in enumerate(pinned):
            start, _ = ranges[world]
            for particle in (int(local_particle), *neighbors[int(local_particle)]):
                flags[start + particle] &= ~1
        runtime.model.particle_flags.assign(flags)
        runtime.cloth.rebuild_bvh(runtime.state_0)
        cloth_step_graph = _advance_cloth(
            runtime,
            round(hold_seconds * spec.control_fps),
            cloth_step_graph,
        )
        runtime.model.particle_flags.assign(default_flags)
        runtime.cloth.rebuild_bvh(runtime.state_0)
        cloth_step_graph = _advance_cloth(
            runtime,
            round(settle_seconds * spec.control_fps),
            cloth_step_graph,
        )

        final_q, final_qd = _capture_particles(runtime, ranges)
        validation = validate_initial_states(
            final_q,
            final_qd,
            table_top_m=table_top,
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
                    "speed_rms_m_s": float(validation.speed_rms_m_s[index]),
                }
            )
        print(
            f"garment={garment_id} accepted={len(accepted_q)}/{state_count} "
            f"candidates={candidates} rejected={dict(rejected)}",
            file=sys.stderr,
            flush=True,
        )

    metadata = {
        "garment_id": garment_id,
        "garment_path": str(garment),
        "garment_sha256": _sha256(garment),
        "scene_sha256": _json_sha256(asdict(spec) | {"world_count": 1}),
        "initialization_protocol": INITIALIZATION_PROTOCOL,
        "generator_seed": seed,
        "candidate_count": candidates,
        "rejection_counts": dict(rejected),
        "hold_seconds": hold_seconds,
        "settle_seconds": settle_seconds,
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


def _hang_drop_candidates(runtime, ranges, seeds, *, table_top: float):
    base = runtime.state_0.particle_q.numpy()
    q = []
    pinned = []
    for world, ((start, end), seed) in enumerate(zip(ranges, seeds, strict=True)):
        del world
        points = np.asarray(base[start:end], dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        rotation = _haar_rotation(rng)
        rotated = (points - points.mean(axis=0)) @ rotation.T
        particle = int(rng.integers(0, len(points)))
        diagonal = float(np.linalg.norm(np.ptp(rotated, axis=0)))
        anchor = np.asarray(
            (
                runtime.spec.table_position[0] + rng.uniform(-0.08, 0.08),
                runtime.spec.table_position[1] + rng.uniform(-0.08, 0.08),
                table_top + diagonal + 0.05,
            )
        )
        rotated += anchor - rotated[particle]
        q.append(rotated.astype(np.float32))
        pinned.append(particle)
    return np.ascontiguousarray(np.stack(q), dtype=np.float32), tuple(pinned)


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


def _local_particle_neighbors(runtime, particle_range) -> tuple[tuple[int, ...], ...]:
    start, end = particle_range
    result = [set() for _ in range(end - start)]
    for triangle in runtime.model.tri_indices.numpy().reshape(-1, 3):
        values = [int(value) for value in triangle]
        if not all(start <= value < end for value in values):
            continue
        a, b, c = (value - start for value in values)
        result[a].update((b, c))
        result[b].update((a, c))
        result[c].update((a, b))
    return tuple(tuple(sorted(values)) for values in result)


def _capture_particles(runtime, ranges):
    q = runtime.state_0.particle_q.numpy()
    qd = runtime.state_0.particle_qd.numpy()
    return (
        np.ascontiguousarray(np.stack([q[start:end] for start, end in ranges]), dtype=np.float32),
        np.ascontiguousarray(np.stack([qd[start:end] for start, end in ranges]), dtype=np.float32),
    )


def _advance_cloth(runtime, steps: int, graph):
    import warp as wp

    if steps < 1:
        return graph
    if graph is None:
        # Compile every lazy kernel once before capture. The captured control
        # step has an even number of state swaps, so it retains the same bound
        # state arrays on every replay.
        runtime.step_cloth()
        steps -= 1
        with wp.ScopedCapture(device=runtime.model.device) as capture:
            runtime.step_cloth()
        graph = capture.graph
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
