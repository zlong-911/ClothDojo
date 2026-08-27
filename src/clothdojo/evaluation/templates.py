"""Deterministic free-settle targets for a fixed benchmark asset collection."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import uuid
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.evaluation.metric import (
    CanonicalTemplate,
    file_sha256,
    load_canonical_template,
    save_canonical_template,
)

TEMPLATE_COLLECTION_SCHEMA = "clothdojo_canonical_template_collection"
TEMPLATE_COLLECTION_FORMAT_VERSION = 1


def resolve_canonical_templates(
    collection: str | Path,
    asset_ids,
    *,
    target_settings: dict[str, Any],
) -> dict[str, tuple[Path, str]]:
    """Resolve selected templates through the collection manifest."""

    root = Path(collection).expanduser().resolve()
    meta = _load_collection_meta(root / "meta.json")
    if meta.get("status") != "complete":
        raise ValueError("canonical template collection is incomplete")
    if meta.get("protocol", {}).get("target") != target_settings:
        raise ValueError("canonical template collection target settings differ")
    entries = meta.get("assets")
    if not isinstance(entries, list) or len(entries) != meta.get("asset_count"):
        raise ValueError("canonical template asset count differs")
    index = {}
    for entry in entries:
        asset_id = entry.get("asset_id") if isinstance(entry, dict) else None
        if not isinstance(asset_id, str) or not asset_id or asset_id in index:
            raise ValueError("canonical template collection has invalid asset ids")
        index[asset_id] = entry

    resolved = {}
    for asset_id in asset_ids:
        entry = index.get(asset_id)
        if entry is None:
            raise ValueError(f"canonical template is missing asset {asset_id}")
        if entry.get("status") != "complete":
            raise ValueError(f"canonical template entry is incomplete: {asset_id}")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"canonical template path is invalid: {asset_id}")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"canonical template path is invalid: {asset_id}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"canonical template path escapes its collection: {asset_id}")
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_sha256, str) or file_sha256(path) != expected_sha256:
            raise ValueError(f"canonical template changed for asset {asset_id}")
        resolved[asset_id] = (path, expected_sha256)
    return resolved


def generate_template_collection(
    initial_states: str | Path,
    protocol_path: str | Path,
    output: str | Path,
    *,
    assets_root: str | Path = "assets",
    gpus: tuple[str, ...] = ("0",),
) -> dict[str, Any]:
    """Generate one robot-independent canonical target for every pool asset."""

    from clothdojo.evaluation.protocol import load_evaluation_protocol
    from clothdojo.rollout.initial_states import load_initial_state_collection

    if not gpus or any(not isinstance(gpu, str) or not gpu.strip() for gpu in gpus):
        raise ValueError("gpus must contain at least one device id")
    states_root = Path(initial_states).expanduser().resolve()
    states_meta = load_initial_state_collection(states_root)
    protocol_source = Path(protocol_path).expanduser().resolve()
    protocol = load_evaluation_protocol(protocol_source)
    if protocol.target.robot_particle_collision:
        raise ValueError("canonical targets must disable robot particle collision")
    assets = Path(assets_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    requested_assets = [
        {"asset_id": item["asset_id"], "split": item["split"]}
        for item in states_meta["assets"]
    ]
    requested = {
        "schema": TEMPLATE_COLLECTION_SCHEMA,
        "format_version": TEMPLATE_COLLECTION_FORMAT_VERSION,
        "initial_states": {
            "path": str(states_root),
            "meta_sha256": file_sha256(states_root / "meta.json"),
        },
        "protocol": {
            "path": str(protocol_source),
            "fingerprint": protocol.fingerprint,
            "target": protocol.target.to_json_dict(),
        },
        "assets_root": str(assets),
        "asset_count": len(requested_assets),
    }
    meta_path = destination / "meta.json"
    if meta_path.is_file():
        meta = _load_collection_meta(meta_path)
        actual = {key: meta.get(key) for key in requested}
        if actual != requested:
            raise ValueError("existing canonical template collection settings differ")
        if [
            {"asset_id": item["asset_id"], "split": item["split"]}
            for item in meta.get("assets", [])
        ] != requested_assets:
            raise ValueError("existing canonical template asset set differs")
        if meta.get("status") == "complete":
            _validate_completed_collection(destination, meta)
            return _collection_summary(destination, meta, gpus)
    else:
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"canonical template output is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "templates").mkdir()
        meta = {
            **requested,
            "status": "generating",
            "assets": [
                {
                    **item,
                    "status": "pending",
                    "path": f"templates/{item['asset_id']}.npz",
                }
                for item in requested_assets
            ],
        }
        _atomic_write_json(meta_path, meta)

    completed = {
        item["asset_id"]
        for item in meta["assets"]
        if item.get("status") == "complete"
        and _completed_entry_is_valid(destination, item, protocol.fingerprint)
    }
    jobs = [item for item in requested_assets if item["asset_id"] not in completed]
    if jobs:
        meta["status"] = "generating"
        meta.pop("error", None)
        _atomic_write_json(meta_path, meta)
        try:
            for result in _run_template_workers(
                jobs,
                gpus=gpus,
                assets_root=assets,
                protocol_path=protocol_source,
                output=destination,
            ):
                entry = next(
                    item for item in meta["assets"] if item["asset_id"] == result["asset_id"]
                )
                entry.update(result)
                entry["status"] = "complete"
                _atomic_write_json(meta_path, meta)
        except Exception as exc:
            meta["status"] = "failed"
            meta["error"] = f"{type(exc).__name__}: {exc}"
            _atomic_write_json(meta_path, meta)
            raise

    meta["status"] = "complete"
    meta.pop("error", None)
    _validate_completed_collection(destination, meta)
    _atomic_write_json(meta_path, meta)
    return _collection_summary(destination, meta, gpus)


def generate_asset_template(
    asset_id: str,
    protocol_path: str | Path,
    output: str | Path,
    *,
    assets_root: str | Path = "assets",
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Free-settle one asset from its exact default pose with ghosted robot contact."""

    from clothdojo.assets import GarmentLibrary
    from clothdojo.evaluation.protocol import load_evaluation_protocol
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_shirt_flatten

    if not asset_id:
        raise ValueError("asset_id must not be empty")
    protocol = load_evaluation_protocol(protocol_path)
    target_protocol = protocol.target
    if target_protocol.robot_particle_collision:
        raise ValueError("canonical targets must disable robot particle collision")
    assets = Path(assets_root).expanduser().resolve()
    garment = GarmentLibrary(assets).converted(asset_id)
    spec = ShirtFlattenSpec(garment_id=asset_id, world_count=1)
    if spec.control_fps != protocol.control_fps:
        raise ValueError("target protocol and scene control_fps differ")
    if not np.array_equal(
        np.asarray(target_protocol.symmetry_center_xy, dtype=np.float64),
        np.asarray(spec.table_position[:2], dtype=np.float64),
    ):
        raise ValueError("target symmetry center differs from the scene table center")

    scene = build_shirt_flatten(
        assets,
        spec=spec,
        device=device,
        visuals=False,
        robot_cloth_collision=False,
    )
    runtime = ShirtFlattenRuntime(scene, use_cuda_graph=False)
    ranges = runtime.grasp.particle_ranges()
    if len(ranges) != 1:
        raise RuntimeError("canonical target generation requires one Newton world")
    start, end = ranges[0]
    min_steps = protocol.target_min_settle_steps
    max_steps = protocol.target_max_settle_steps
    sample_stride = protocol.metric_stride
    window_samples = protocol.target_stability_samples
    window_steps = window_samples * sample_stride
    graph = runtime.advance_cloth(min_steps - window_steps)
    tail_speeds: deque[float] = deque(maxlen=window_samples)
    settled_steps = min_steps - window_steps
    while settled_steps < max_steps:
        graph = runtime.advance_cloth(sample_stride, graph=graph)
        qd = runtime.state_0.particle_qd.numpy()[start:end]
        tail_speeds.append(_speed_rms(qd))
        settled_steps += sample_stride
        if (
            settled_steps >= min_steps
            and len(tail_speeds) == window_samples
            and max(tail_speeds) <= target_protocol.speed_rms_max_m_s
        ):
            break

    particle_q = np.ascontiguousarray(
        runtime.state_0.particle_q.numpy()[start:end], dtype=np.float32
    )
    particle_qd = np.ascontiguousarray(
        runtime.state_0.particle_qd.numpy()[start:end], dtype=np.float32
    )
    if not np.isfinite(particle_q).all() or not np.isfinite(particle_qd).all():
        raise RuntimeError("canonical target contains non-finite particle state")
    maximum_tail_speed = float(max(tail_speeds))
    if maximum_tail_speed > target_protocol.speed_rms_max_m_s:
        raise RuntimeError(
            "canonical target did not settle before timeout: "
            f"tail_speed_max={maximum_tail_speed:.8f} > "
            f"{target_protocol.speed_rms_max_m_s:.8f} after "
            f"{settled_steps / spec.control_fps:.2f}s"
        )
    table_contacts, robot_contacts = runtime.cloth_contact_counts()
    if int(table_contacts[0]) < 1:
        raise RuntimeError("canonical target has no physical table contact")
    if int(robot_contacts[0]) != 0:
        raise RuntimeError("canonical target unexpectedly contacted the ghosted robot")
    extent = np.ptp(particle_q[:, :2].astype(np.float64), axis=0)
    scale = float(np.linalg.norm(extent))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("canonical target has a degenerate planar extent")

    asset_sha256 = file_sha256(garment)
    topology_sha256 = garment_topology_sha256(garment)
    physics_sha256 = shirt_flatten_physics_sha256(spec)
    source_sha256 = _target_source_sha256(
        asset_sha256=asset_sha256,
        topology_sha256=topology_sha256,
        physics_sha256=physics_sha256,
        protocol_fingerprint=protocol.fingerprint,
        particle_q=particle_q,
    )
    template = CanonicalTemplate(
        asset_id=asset_id,
        asset_sha256=asset_sha256,
        topology_sha256=topology_sha256,
        scene_physics_sha256=physics_sha256,
        source_kind="default_free_settle",
        source_sha256=source_sha256,
        normalization_scale_m=scale,
        particle_q=particle_q,
    )
    destination = Path(output).expanduser().resolve()
    save_canonical_template(destination, template)
    table_top = spec.table_position[2] + spec.table_half_extents[2]
    return {
        "asset_id": asset_id,
        "path": f"templates/{asset_id}.npz",
        "sha256": file_sha256(destination),
        "protocol_fingerprint": protocol.fingerprint,
        "particle_count": template.particle_count,
        "normalization_scale_m": template.normalization_scale_m,
        "tail_speed_rms_max_m_s": maximum_tail_speed,
        "settled_seconds": settled_steps / spec.control_fps,
        "terminal_speed_rms_m_s": _speed_rms(particle_qd),
        "table_contact_count": int(table_contacts[0]),
        "p99_height_above_table_m": float(
            np.quantile(particle_q[:, 2].astype(np.float64), 0.99) - table_top
        ),
        "source_kind": template.source_kind,
        "source_sha256": template.source_sha256,
    }


def garment_topology_sha256(path: str | Path) -> str:
    """Fingerprint the particle correspondence and triangle order."""

    from clothdojo.sim.cloth import load_garment_mesh

    vertices, indices, _, _ = load_garment_mesh(path)
    triangles = np.asarray(indices, dtype="<i4", order="C")
    payload = np.asarray([len(vertices), len(triangles)], dtype="<i8").tobytes()
    payload += triangles.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def shirt_flatten_physics_sha256(spec) -> str:
    """Fingerprint target-relevant physics, excluding batch and asset identity."""

    payload = asdict(spec)
    payload.pop("garment_id")
    payload.pop("world_count")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_template_workers(jobs, *, gpus, assets_root, protocol_path, output):
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
            target=_template_worker,
            args=(
                job_queue,
                result_queue,
                gpu,
                str(assets_root),
                str(protocol_path),
                str(output),
            ),
            name=f"clothdojo-template-gpu-{gpu}",
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
                failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(
                        f"template worker exited with code {failed[0].exitcode}"
                    ) from None
                continue
            if result["status"] == "error":
                raise RuntimeError(
                    f"template generation failed for {result['asset_id']} on GPU "
                    f"{result['gpu']}: {result['error']}"
                )
            completed += 1
            yield result["result"]
    finally:
        for worker in workers:
            if worker.is_alive() and completed < len(jobs):
                worker.terminate()
            worker.join()


def _template_worker(job_queue, result_queue, gpu, assets_root, protocol_path, output):
    if gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cpu" if gpu == "cpu" else "cuda:0"
    while True:
        job = job_queue.get()
        if job is None:
            return
        asset_id = job["asset_id"]
        try:
            result = generate_asset_template(
                asset_id,
                protocol_path,
                Path(output) / "templates" / f"{asset_id}.npz",
                assets_root=assets_root,
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
        result_queue.put({"status": "complete", "result": result})


def _target_source_sha256(
    *, asset_sha256, topology_sha256, physics_sha256, protocol_fingerprint, particle_q
):
    digest = hashlib.sha256()
    identity = {
        "asset_sha256": asset_sha256,
        "topology_sha256": topology_sha256,
        "physics_sha256": physics_sha256,
        "protocol_fingerprint": protocol_fingerprint,
    }
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    digest.update(np.asarray(particle_q, dtype="<f4", order="C").tobytes())
    return digest.hexdigest()


def _speed_rms(particle_qd) -> float:
    values = np.asarray(particle_qd, dtype=np.float64)
    return float(np.sqrt(np.square(values).sum(axis=1).mean()))


def _completed_entry_is_valid(root, entry, protocol_fingerprint) -> bool:
    path = root / entry.get("path", "")
    if not path.is_file() or entry.get("sha256") != file_sha256(path):
        return False
    try:
        template = load_canonical_template(path)
    except (OSError, ValueError):
        return False
    return (
        template.asset_id == entry.get("asset_id")
        and template.source_kind == "default_free_settle"
        and entry.get("protocol_fingerprint") == protocol_fingerprint
    )


def _validate_completed_collection(root, meta) -> None:
    if len(meta.get("assets", ())) != meta.get("asset_count"):
        raise ValueError("canonical template asset count differs")
    for entry in meta["assets"]:
        if entry.get("status") != "complete" or not _completed_entry_is_valid(
            root, entry, meta["protocol"]["fingerprint"]
        ):
            raise ValueError(f"canonical template entry is incomplete: {entry.get('asset_id')}")


def _load_collection_meta(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read canonical template metadata: {path}: {exc}") from exc
    if value.get("schema") != TEMPLATE_COLLECTION_SCHEMA:
        raise ValueError("unsupported canonical template collection schema")
    if value.get("format_version") != TEMPLATE_COLLECTION_FORMAT_VERSION:
        raise ValueError("unsupported canonical template collection version")
    return value


def _collection_summary(root, meta, gpus):
    return {
        "compatible": True,
        "output": str(root),
        "asset_count": meta["asset_count"],
        "protocol_fingerprint": meta["protocol"]["fingerprint"],
        "gpus": list(gpus),
    }


def _atomic_write_json(path, payload):
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
