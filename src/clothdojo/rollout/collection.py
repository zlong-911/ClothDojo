"""Multi-asset policy rollouts with teleoperation-compatible trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from clothdojo.rendering import policy_observation_settings
from clothdojo.rollout.initial_states import (
    InitialStatePool,
    load_initial_state_collection,
)
from clothdojo.rollout.policy import RemotePolicy
from clothdojo.rollout.render import CAMERA_NAMES
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
from clothdojo.sim.scene import WORLD_LAYOUT, WORLD_SPACING_M
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
    rtc_delay_steps: int = 0,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Run every selected asset on a dynamic local GPU worker queue."""

    if not policy_endpoints or any(not endpoint.strip() for endpoint in policy_endpoints):
        raise ValueError("at least one policy endpoint is required")
    if not gpus or any(not gpu.strip() for gpu in gpus):
        raise ValueError("at least one simulation GPU is required")
    if execution not in {"chunk", "rtc"}:
        raise ValueError("execution must be 'chunk' or 'rtc'")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(rtc_delay_steps, bool)
        or not isinstance(rtc_delay_steps, int)
        or rtc_delay_steps < 0
    ):
        raise ValueError("rtc_delay_steps must be a non-negative integer")
    scenario = load_rollout_scenario(scenario_path)
    if not 1 <= exec_horizon <= scenario.action_horizon:
        raise ValueError("exec_horizon must be in [1, action_horizon]")
    logical_delay = rtc_delay_steps if execution == "rtc" else 0
    if execution == "rtc" and logical_delay >= exec_horizon:
        raise ValueError("RTC delay must be smaller than exec_horizon")
    if execution == "rtc" and exec_horizon + logical_delay > scenario.action_horizon:
        raise ValueError("RTC horizon plus delay exceeds the model action horizon")
    states_root = Path(initial_states_path).expanduser().resolve()
    states_meta = load_initial_state_collection(states_root)
    assets = Path(assets_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    selected_assets = select_assets(
        states_meta["assets"],
        seen_assets=seen_assets,
        unseen_assets=unseen_assets,
    )
    asset_jobs: list[TrajectoryAssetJob] = []
    expected = 0
    for item in selected_assets:
        pool_path = states_root / item["asset_id"] / "states.npz"
        pool = InitialStatePool.load(pool_path)
        count = pool.state_count if limit_per_asset is None else int(limit_per_asset)
        if count < 1 or count > pool.state_count:
            raise ValueError(f"limit_per_asset must be in [1, {pool.state_count}]")
        asset_jobs.append(
            TrajectoryAssetJob(
                asset_id=item["asset_id"],
                pool=str(pool_path),
                pool_sha256=_file_sha256(pool_path),
                state_count=count,
            )
        )
        expected += count

    worker_count = min(len(gpus), len(asset_jobs))
    active_endpoints = policy_endpoints_for_workers(policy_endpoints, worker_count)
    identities = tuple(
        RemotePolicy(
            endpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            action_horizon=scenario.action_horizon,
        ).identity
        for endpoint in active_endpoints
    )
    first_identity = validate_policy_identities(
        identities,
        execution=execution,
        worlds_per_batch=batch_size,
    )

    requested = {
        "scenario": scenario.to_json_dict(),
        "cloth_visual": "nocs",
        "observation": policy_observation_settings(),
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
        "execution": execution_settings(
            execution,
            exec_horizon,
            rtc_delay_steps=logical_delay,
        ),
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
        "expected_trajectories": expected,
    }
    collection_spec = TrajectoryCollectionSpec(
        scenario_path=str(Path(scenario_path).expanduser().resolve()),
        assets_root=str(assets),
        trajectories_root=str(destination / "trajectories"),
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        policy_identity=first_identity,
        execution=execution,
        exec_horizon=exec_horizon,
        rtc_delay_steps=logical_delay,
        horizon_steps=scenario.steps,
        batch_size=batch_size,
        producer="clothdojo_rollout",
        training_eligible=True,
    )
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
        atomic_write_json(meta_path, meta)

    try:
        collect_trajectories(
            asset_jobs,
            gpus=gpus,
            policy_endpoints=active_endpoints,
            spec=collection_spec,
        )
        trajectories = collect_trajectory_index(asset_jobs, spec=collection_spec)
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["trajectories"] = collect_trajectory_index(
            asset_jobs, spec=collection_spec, partial=True
        )
        atomic_write_json(meta_path, meta)
        raise

    meta["status"] = "complete"
    meta.pop("error", None)
    meta["trajectories"] = trajectories
    atomic_write_json(meta_path, meta)
    return _rollout_summary(destination, meta, gpus)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
