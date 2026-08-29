"""Strict, cached cold-restore plans for HIL and continuation teleoperation."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from clothdojo.teleop.session import SessionPlan
from clothdojo.trajectory import TrajectoryReader

INTERVENTION_PLAN_SCHEMA = "clothdojo_intervention_plan_v1"
CONTINUATION_PLAN_SCHEMA = "clothdojo_continuation_plan_v1"


@dataclass(frozen=True, slots=True)
class RestorationPoint:
    kind: Literal["human_intervention", "task_continuation"]
    source_episode_id: str
    source_sample_index: int
    trajectory: Path
    manifest_sha256: str
    source_run_id: str | None = None
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
        if self.source_profile_id is not None:
            value["source_profile_id"] = self.source_profile_id
        return value


class RestorationQueue:
    """Resolve immutable plans once and cache one reader per source trajectory."""

    def __init__(
        self,
        points: list[RestorationPoint],
        readers: dict[Path, TrajectoryReader],
    ) -> None:
        if not points:
            raise ValueError("a restore plan must contain at least one point")
        self._points = deque(points)
        self._readers = readers
        self.current: RestorationPoint | None = None
        self.current_gripper_open: np.ndarray | None = None

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
            "intervention plan",
        )
        if value["schema"] != INTERVENTION_PLAN_SCHEMA:
            raise ValueError("unsupported intervention plan schema")
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
                source_run_id=str(value["run_id"]),
            )
            for raw in raw_points
        ]
        trajectory_root = (rollout_root / "trajectories").resolve()
        if any(point.trajectory.parent != trajectory_root for point in points):
            raise ValueError("intervention trajectory is outside its rollout root")
        return cls(points, _readers(points, runtime, source_task=plan.scene))

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
            _point(raw, kind="task_continuation", source_profile_id=source_profile)
            for raw in raw_points
        ]
        readers = _readers(points, runtime, source_task=source_task, require_terminal=True)
        return cls(points, readers)

    @property
    def kind(self) -> str:
        first = self.current or (self._points[0] if self._points else None)
        return "none" if first is None else first.kind

    @property
    def remaining(self) -> int:
        return len(self._points)

    def restore_next(
        self,
        runtime,
        *,
        requeue_current: bool = False,
    ) -> RestorationPoint | None:
        if requeue_current and self.current is not None:
            self._points.append(self.current)
        if not self._points:
            self.current = None
            self.current_gripper_open = None
            return None
        point = self._points.popleft()
        state = self._readers[point.trajectory].state_at(point.source_sample_index)
        runtime.restore_state(state)
        self.current = point
        self.current_gripper_open = np.asarray(state.gripper_open_state, dtype=np.float32).copy()
        return point


def _point(
    raw: object,
    *,
    kind: Literal["human_intervention", "task_continuation"],
    source_run_id: str | None = None,
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
        source_episode_id=str(raw["source_episode_id"]),
        source_sample_index=sample,
        trajectory=path.resolve(),
        manifest_sha256=str(raw["manifest_sha256"]),
        source_run_id=source_run_id,
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
