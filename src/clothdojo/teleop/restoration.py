"""Strict, cached cold-restore plans for HIL and continuation teleoperation."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from clothdojo.collection.intervention_leases import assert_owned, claim_next, point_key
from clothdojo.teleop.session import SessionPlan
from clothdojo.trajectory import TrajectoryReader

INTERVENTION_PLAN_SCHEMA = "clothdojo_intervention_plan_v3"
SINGLE_ASSET_INTERVENTION_PLAN_SCHEMA = "clothdojo_intervention_plan_v2"
LEGACY_INTERVENTION_PLAN_SCHEMA = "clothdojo_intervention_plan_v1"
CONTINUATION_PLAN_SCHEMA = "clothdojo_continuation_plan_v1"


@dataclass(frozen=True, slots=True)
class RestorationPoint:
    kind: Literal["human_intervention", "task_continuation"]
    asset_id: str
    source_episode_id: str
    source_sample_index: int
    trajectory: Path
    manifest_sha256: str
    source_run_id: str | None = None
    source_rollout_root: Path | None = None
    source_rollout_meta_sha256: str | None = None
    source_profile_id: str | None = None

    def recording_metadata(self) -> dict[str, Any]:
        value = {
            "source": self.kind,
            "control_source": "vr_teleop",
            "source_episode_id": self.source_episode_id,
            "source_sample_index": self.source_sample_index,
            "source_manifest_sha256": self.manifest_sha256,
        }
        if self.source_run_id is not None:
            value["source_run_id"] = self.source_run_id
        if self.source_rollout_root is not None:
            value["source_rollout_root"] = str(self.source_rollout_root)
        if self.source_rollout_meta_sha256 is not None:
            value["source_rollout_meta_sha256"] = self.source_rollout_meta_sha256
        if self.source_profile_id is not None:
            value["source_profile_id"] = self.source_profile_id
        return value


class RestorationQueue:
    """Resolve immutable plans once and cache one reader per source trajectory."""

    def __init__(
        self,
        points: list[RestorationPoint],
        readers: dict[Path, TrajectoryReader],
        asset_runtime: dict[str, dict[str, Any]],
        *,
        active_asset_id: str,
    ) -> None:
        if not points:
            raise ValueError("a restore plan must contain at least one point")
        self._points = deque(points)
        self._readers = readers
        self._asset_runtime = asset_runtime
        self._active_asset_id = active_asset_id
        self.current: RestorationPoint | None = None
        self.current_gripper_open: np.ndarray | None = None
        self._allocation = None

    @classmethod
    def load(cls, plan: SessionPlan, runtime) -> RestorationQueue | None:
        if plan.intervention_plan is not None and plan.continuation_plan is not None:
            raise ValueError("intervention and continuation plans are mutually exclusive")
        path = plan.intervention_plan or plan.continuation_plan
        if path is None:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("restore plan must be an object")
        if plan.intervention_plan is not None:
            return cls._load_intervention(plan, runtime, value)
        return cls._load_continuation(plan, runtime, value)

    @classmethod
    def _load_intervention(cls, plan, runtime, value) -> RestorationQueue:
        allocation = value.get("allocation")
        if allocation is not None:
            value = {key: item for key, item in value.items() if key != "allocation"}
        if value.get("schema") == LEGACY_INTERVENTION_PLAN_SCHEMA:
            return cls._load_legacy_intervention(plan, runtime, value)
        if value.get("schema") == SINGLE_ASSET_INTERVENTION_PLAN_SCHEMA:
            asset_id = value.get("asset_id")
            value = {
                "schema": INTERVENTION_PLAN_SCHEMA,
                "profile_id": value.get("profile_id"),
                "asset_id": asset_id,
                "assets": [{"asset_id": asset_id, "runtime": value.get("runtime")}],
                "runs": value.get("runs"),
                "breakpoints": [
                    {"asset_id": asset_id, **raw} if isinstance(raw, dict) else raw
                    for raw in value.get("breakpoints", [])
                ],
            }
        _require_keys(
            value,
            {
                "schema",
                "profile_id",
                "asset_id",
                "assets",
                "runs",
                "breakpoints",
            },
            "intervention plan",
        )
        if value["schema"] != INTERVENTION_PLAN_SCHEMA:
            raise ValueError("unsupported intervention plan schema")
        if value["profile_id"] != plan.profile_id or value["asset_id"] != plan.asset_id:
            raise ValueError("intervention plan does not match the teleoperation session")
        raw_assets = value["assets"]
        if not isinstance(raw_assets, list) or not raw_assets:
            raise ValueError("intervention plan needs garment assets")
        asset_runtime: dict[str, dict[str, Any]] = {}
        for raw_asset in raw_assets:
            if not isinstance(raw_asset, dict):
                raise ValueError("intervention garment asset must be an object")
            _require_keys(raw_asset, {"asset_id", "runtime"}, "intervention garment asset")
            asset_id = str(raw_asset["asset_id"])
            if asset_id in asset_runtime:
                raise ValueError("intervention plan contains duplicate garment assets")
            if not isinstance(raw_asset["runtime"], dict):
                raise ValueError("intervention garment runtime must be an object")
            asset_runtime[asset_id] = raw_asset["runtime"]
        if plan.asset_id not in asset_runtime:
            raise ValueError("initial intervention asset is missing from its plan")
        _validate_runtime(plan, runtime, asset_runtime[plan.asset_id], source_task=plan.scene)
        raw_runs = value["runs"]
        if not isinstance(raw_runs, list) or not raw_runs:
            raise ValueError("intervention plan needs rollout provenance")
        run_provenance: dict[str, tuple[Path, str]] = {}
        for raw_run in raw_runs:
            if not isinstance(raw_run, dict):
                raise ValueError("intervention rollout provenance must be an object")
            _require_keys(
                raw_run,
                {"run_id", "rollout_root", "rollout_meta_sha256"},
                "intervention rollout provenance",
            )
            run_id = str(raw_run["run_id"])
            if run_id in run_provenance:
                raise ValueError("intervention plan contains duplicate rollout ids")
            rollout_root = Path(raw_run["rollout_root"]).resolve()
            meta_sha256 = str(raw_run["rollout_meta_sha256"])
            if _sha256(rollout_root / "meta.json") != meta_sha256:
                raise ValueError("intervention rollout metadata changed after planning")
            run_provenance[run_id] = (rollout_root, meta_sha256)
        raw_points = value["breakpoints"]
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError("intervention plan needs breakpoints")
        points = []
        for raw in raw_points:
            if not isinstance(raw, dict):
                raise ValueError("intervention breakpoint must be an object")
            _require_keys(
                raw,
                {
                    "asset_id",
                    "source_run_id",
                    "source_episode_id",
                    "source_sample_index",
                    "trajectory",
                    "manifest_sha256",
                },
                "intervention breakpoint",
            )
            run_id = str(raw["source_run_id"])
            provenance = run_provenance.get(run_id)
            if provenance is None:
                raise ValueError("intervention breakpoint references an unknown rollout")
            rollout_root, rollout_meta_sha256 = provenance
            point = _point(
                {
                    key: item
                    for key, item in raw.items()
                    if key not in {"source_run_id", "asset_id"}
                },
                kind="human_intervention",
                asset_id=str(raw.get("asset_id", "")),
                source_run_id=run_id,
                source_rollout_root=rollout_root,
                source_rollout_meta_sha256=rollout_meta_sha256,
            )
            if point.trajectory.parent != (rollout_root / "trajectories").resolve():
                raise ValueError("intervention trajectory is outside its rollout root")
            if point.asset_id not in asset_runtime:
                raise ValueError("intervention breakpoint references an unknown garment asset")
            points.append(point)
        queue = cls(
            points,
            _readers(points, runtime, source_task=plan.scene),
            asset_runtime,
            active_asset_id=plan.asset_id,
        )
        if allocation is not None:
            if allocation["session_id"] != plan.session_id:
                raise ValueError("HIL allocation belongs to another session")
            queue._allocation = allocation
            queue._pool = value["breakpoints"]
            queue._pool_points = {point_key(p.recording_metadata()): p for p in points}
            queue._profile_id = plan.profile_id
            queue._output_root = plan.output_root
            queue._visited_assets = {plan.asset_id}
            queue._points = deque(queue._pool_points[point_key(p)]
                                  for p in allocation["initial_points"])
        return queue

    @classmethod
    def _load_legacy_intervention(cls, plan, runtime, value) -> RestorationQueue:
        _require_keys(
            value,
            {
                "schema",
                "profile_id",
                "run_id",
                "rollout_root",
                "rollout_meta_sha256",
                "asset_id",
                "runtime",
                "breakpoints",
            },
            "legacy intervention plan",
        )
        if value["profile_id"] != plan.profile_id or value["asset_id"] != plan.asset_id:
            raise ValueError("intervention plan does not match the teleoperation session")
        rollout_root = Path(value["rollout_root"]).resolve()
        if _sha256(rollout_root / "meta.json") != value["rollout_meta_sha256"]:
            raise ValueError("intervention rollout metadata changed after planning")
        _validate_runtime(plan, runtime, value["runtime"], source_task=plan.scene)
        raw_points = value["breakpoints"]
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError("intervention plan needs breakpoints")
        points = [
            _point(
                raw,
                kind="human_intervention",
                asset_id=str(value["asset_id"]),
                source_run_id=str(value["run_id"]),
                source_rollout_root=rollout_root,
                source_rollout_meta_sha256=str(value["rollout_meta_sha256"]),
            )
            for raw in raw_points
        ]
        trajectory_root = (rollout_root / "trajectories").resolve()
        if any(point.trajectory.parent != trajectory_root for point in points):
            raise ValueError("intervention trajectory is outside its rollout root")
        return cls(
            points,
            _readers(points, runtime, source_task=plan.scene),
            {str(value["asset_id"]): value["runtime"]},
            active_asset_id=plan.asset_id,
        )

    @classmethod
    def _load_continuation(cls, plan, runtime, value) -> RestorationQueue:
        _require_keys(
            value,
            {
                "schema",
                "profile_id",
                "source_profile_id",
                "asset_id",
                "runtime",
                "sources",
            },
            "continuation plan",
        )
        if value["schema"] != CONTINUATION_PLAN_SCHEMA:
            raise ValueError("unsupported continuation plan schema")
        if value["profile_id"] != plan.profile_id or value["asset_id"] != plan.asset_id:
            raise ValueError("continuation plan does not match the teleoperation session")
        source_profile = str(value["source_profile_id"])
        runtime_identity = value["runtime"]
        if not isinstance(runtime_identity, dict):
            raise ValueError("continuation runtime identity must be an object")
        source_task = str(runtime_identity.get("source_task_name", ""))
        _validate_runtime(plan, runtime, runtime_identity, source_task=source_task)
        raw_points = value["sources"]
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError("continuation plan needs terminal sources")
        points = [
            _point(
                raw,
                kind="task_continuation",
                asset_id=str(value["asset_id"]),
                source_profile_id=source_profile,
            )
            for raw in raw_points
        ]
        readers = _readers(points, runtime, source_task=source_task, require_terminal=True)
        return cls(
            points,
            readers,
            {str(value["asset_id"]): runtime_identity},
            active_asset_id=plan.asset_id,
        )

    @property
    def kind(self) -> str:
        first = self.current or (self._points[0] if self._points else None)
        return "none" if first is None else first.kind

    @property
    def remaining(self) -> int:
        return len(self._points)

    def prepare_next(self, *, requeue_current: bool = False) -> str | None:
        """Return the next asset without consuming its restoration point."""

        if requeue_current and self.current is not None and self.current not in self._points:
            self._points.append(self.current)
        if not self._points and self._allocation is not None:
            claimed = claim_next(
                Path(self._allocation["database_path"]), profile_id=self._profile_id,
                session_id=self._allocation["session_id"], points=self._pool,
                output_root=self._output_root, visited=self._visited_assets,
            )
            if claimed:
                self._visited_assets.add(claimed[0]["asset_id"])
                self._points.extend(self._pool_points[point_key(p)] for p in claimed)
        return None if not self._points else self._points[0].asset_id

    def validate_runtime(self, plan: SessionPlan, runtime) -> None:
        identity = self._asset_runtime.get(plan.asset_id)
        if identity is None:
            raise ValueError(f"restore queue has no runtime identity for {plan.asset_id}")
        _validate_runtime(plan, runtime, identity, source_task=plan.scene)

    def activate(self, plan: SessionPlan, runtime) -> None:
        self.validate_runtime(plan, runtime)
        self._active_asset_id = plan.asset_id

    def prepare_runtime_for_next(self, plan: SessionPlan, runtime) -> RestorationPoint:
        """Restore the queued state on a candidate runtime without consuming it."""

        self.validate_runtime(plan, runtime)
        if not self._points:
            raise RuntimeError("restore queue has no next point")
        point = self._points[0]
        if point.asset_id != plan.asset_id:
            raise RuntimeError(
                f"next restore point needs asset {point.asset_id}, candidate is {plan.asset_id}"
            )
        state = self._readers[point.trajectory].state_at(point.source_sample_index)
        runtime.restore_state(state)
        return point

    def restore_next(
        self,
        runtime,
        *,
        requeue_current: bool = False,
    ) -> RestorationPoint | None:
        if self.prepare_next(requeue_current=requeue_current) is None:
            self.current = None
            self.current_gripper_open = None
            return None
        point = self._points[0]
        if point.asset_id != self._active_asset_id:
            raise RuntimeError(
                f"restore point needs asset {point.asset_id}, active asset is "
                f"{self._active_asset_id}"
            )
        self._points.popleft()
        return self._restore(runtime, point)

    def restore_current(self, runtime) -> RestorationPoint | None:
        """Restore the active queue item again without changing queue order."""

        if self.current is None:
            return None
        return self._restore(runtime, self.current)

    def _restore(self, runtime, point: RestorationPoint) -> RestorationPoint:
        self.assert_ownership(point.asset_id)
        state = self._readers[point.trajectory].state_at(point.source_sample_index)
        runtime.restore_state(state)
        self.current = point
        self.current_gripper_open = np.asarray(state.gripper_open_state, dtype=np.float32).copy()
        return point

    def assert_ownership(self, asset_id: str | None = None) -> None:
        if self._allocation is not None:
            assert_owned(Path(self._allocation["database_path"]), self._profile_id,
                         self._allocation["session_id"], asset_id or self._active_asset_id)


def _point(
    raw: object,
    *,
    kind: Literal["human_intervention", "task_continuation"],
    asset_id: str,
    source_run_id: str | None = None,
    source_rollout_root: Path | None = None,
    source_rollout_meta_sha256: str | None = None,
    source_profile_id: str | None = None,
) -> RestorationPoint:
    expected = {
        "source_episode_id",
        "source_sample_index",
        "trajectory",
        "manifest_sha256",
    }
    if not isinstance(raw, dict):
        raise ValueError("restore point must be an object")
    _require_keys(raw, expected, "restore point")
    sample = raw["source_sample_index"]
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
        raise ValueError("restore sample index must be a non-negative integer")
    path = Path(raw["trajectory"])
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("restore trajectory must be an absolute non-symlink path")
    return RestorationPoint(
        kind=kind,
        asset_id=asset_id,
        source_episode_id=str(raw["source_episode_id"]),
        source_sample_index=sample,
        trajectory=path.resolve(),
        manifest_sha256=str(raw["manifest_sha256"]),
        source_run_id=source_run_id,
        source_rollout_root=source_rollout_root,
        source_rollout_meta_sha256=source_rollout_meta_sha256,
        source_profile_id=source_profile_id,
    )


def _readers(
    points: list[RestorationPoint],
    runtime,
    *,
    source_task: str,
    require_terminal: bool = False,
) -> dict[Path, TrajectoryReader]:
    readers: dict[Path, TrajectoryReader] = {}
    for point in points:
        manifest_path = point.trajectory / "manifest.json"
        if _sha256(manifest_path) != point.manifest_sha256:
            raise ValueError(f"restore source manifest changed: {point.source_episode_id}")
        reader = readers.setdefault(point.trajectory, TrajectoryReader(point.trajectory))
        manifest = reader.manifest
        if point.trajectory.name != point.source_episode_id:
            raise ValueError("restore source id differs from its trajectory directory")
        if manifest.status != "complete":
            raise ValueError("restore source trajectory is not complete")
        if manifest.scene.task_name != source_task:
            raise ValueError("restore source task does not match its plan")
        if manifest.scene.robot_id != "piper":
            raise ValueError("restore source robot is not Piper")
        if manifest.metadata.get("asset_id") != point.asset_id:
            raise ValueError("restore source garment asset does not match its plan")
        if not np.isclose(manifest.control_fps, runtime.spec.control_fps, rtol=0.0, atol=1e-9):
            raise ValueError("restore source control FPS differs from the runtime")
        if manifest.sim_substeps != runtime.spec.sim_substeps:
            raise ValueError("restore source sim substeps differ from the runtime")
        if point.source_sample_index >= manifest.state_count:
            raise ValueError("restore point is outside the source state archive")
        if require_terminal and point.source_sample_index != manifest.state_count - 1:
            raise ValueError("continuation sources must select terminal states")
    return readers


def _validate_runtime(plan, runtime, value: object, *, source_task: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("restore runtime identity must be an object")
    task = value.get("task_name")
    if task not in {plan.scene, source_task}:
        raise ValueError("restore runtime task identity is incompatible")
    if value.get("robot_id") != "piper":
        raise ValueError("restore runtime robot identity is incompatible")
    if not np.isclose(
        float(value.get("control_fps", -1)),
        runtime.spec.control_fps,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("restore runtime control FPS is incompatible")
    if int(value.get("sim_substeps", -1)) != runtime.spec.sim_substeps:
        raise ValueError("restore runtime sim substeps are incompatible")
    fingerprints = value.get("asset_fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("restore runtime asset fingerprints are missing")
    expected = {
        "garment_usdc": _sha256(plan.asset_path),
        "piper_entry_usda": _sha256(plan.assets_root / "robots/piper/piper.usda"),
    }
    if any(fingerprints.get(name) != digest for name, digest in expected.items()):
        raise ValueError("restore runtime assets differ from the current session")


def _require_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
