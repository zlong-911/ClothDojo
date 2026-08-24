"""Canonical Newton 1.5 trajectory recording for teleoperation."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from clothdojo.teleop.session import SessionPlan
from clothdojo.trajectory import SceneMetadata, TrajectoryWriter
from clothdojo.trajectory.newton import capture_actions, capture_states


class TeleopRecorder:
    def __init__(self, runtime, plan: SessionPlan) -> None:
        self.runtime = runtime
        self.plan = plan
        self.writer: TrajectoryWriter | None = None
        self.path: Path | None = None
        self.sample_index = 0

    @property
    def is_recording(self) -> bool:
        return self.writer is not None

    @property
    def frame_count(self) -> int:
        return self.sample_index

    def start(self, *, origin: Mapping[str, object] | None = None) -> Path:
        if self.writer is not None:
            raise RuntimeError("teleoperation recording is already active")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        episode_id = f"teleop_{stamp}_{uuid.uuid4().hex[:8]}"
        output = self.plan.output_root / episode_id
        robot = self.plan.assets_root / "robots/piper/piper.usda"
        metadata: dict[str, object] = {
            "producer": "clothdojo_isaac_teleop",
            "session_id": self.plan.session_id,
            "profile_id": self.plan.profile_id,
            "asset_id": self.plan.asset_id,
            "mode": self.plan.mode,
            "training_eligible": self.plan.mode == "collection",
        }
        if origin is not None:
            protected = set(metadata) & set(origin)
            if protected:
                raise ValueError(
                    f"restore metadata cannot replace session fields: {sorted(protected)}"
                )
            metadata.update(origin)
        self.writer = TrajectoryWriter(
            output,
            control_fps=self.runtime.spec.control_fps,
            sim_substeps=self.runtime.spec.sim_substeps,
            max_actions=self.plan.max_actions,
            scene=SceneMetadata(
                task_name=self.plan.scene,
                robot_name="piper",
                asset_fingerprints={
                    "garment_usdc": _sha256(self.plan.asset_path),
                    "piper_entry_usda": _sha256(robot),
                },
                config_fingerprints={},
            ),
            metadata=metadata,
        )
        try:
            states = capture_states(self.runtime, 0)
            if len(states) != 1:
                raise RuntimeError("teleoperation requires exactly one Newton world")
            self.writer.write_initial_state(states[0])
        except Exception:
            self.writer.abort()
            self.writer = None
            raise
        self.path = output
        self.sample_index = 0
        return output

    def step(self, arm_command, gripper_open) -> bool:
        """Record exactly the command passed to Newton and its resulting state."""
        if self.writer is None:
            self.runtime.step(arm_command, gripper_open)
            return False
        actions = capture_actions(
            self.runtime,
            self.sample_index,
            arm_command,
            gripper_open,
        )
        self.runtime.step(arm_command, gripper_open)
        self.sample_index += 1
        states = capture_states(self.runtime, self.sample_index)
        self.writer.append_transition(actions[0], states[0])
        if self.sample_index >= self.plan.max_actions:
            self.stop(reason="max_actions_reached")
            return True
        return False

    def stop(self, *, reason: str | None = None) -> Path | None:
        if self.writer is None:
            return None
        writer, path = self.writer, self.path
        self.writer = None
        self.path = None
        if self.sample_index == 0:
            writer.abort()
            return None
        if reason is not None:
            writer.manifest.metadata["stop_reason"] = reason
        writer.close()
        return path

    def abort(self, reason: str) -> Path | None:
        if self.writer is None:
            return None
        writer, path = self.writer, self.path
        self.writer = None
        self.path = None
        writer.manifest.metadata["abort_reason"] = reason
        writer.abort()
        return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
