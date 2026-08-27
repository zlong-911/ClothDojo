"""Canonical Newton 1.5 trajectory recording for teleoperation."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread

from clothdojo.teleop.session import SessionPlan
from clothdojo.trajectory import (
    SceneMetadata,
    TrajectoryAction,
    TrajectoryState,
    TrajectoryWriter,
)
from clothdojo.trajectory.newton import capture_actions, capture_states


@dataclass(frozen=True, slots=True)
class _AppendTransition:
    action: TrajectoryAction
    next_state: TrajectoryState


@dataclass(frozen=True, slots=True)
class _CloseTrajectory:
    reason: str | None


@dataclass(frozen=True, slots=True)
class _AbortTrajectory:
    reason: str | None


_WriterCommand = _AppendTransition | _CloseTrajectory | _AbortTrajectory


class _TrajectoryWriterWorker:
    """Serialize one writer on a bounded background queue."""

    def __init__(self, writer: TrajectoryWriter, *, queue_size: int = 64) -> None:
        if queue_size < 1:
            raise ValueError("writer queue size must be positive")
        self._writer = writer
        self._queue: Queue[_WriterCommand] = Queue(maxsize=queue_size)
        self._done = Event()
        self._failure: BaseException | None = None
        self._terminal_queued = False
        self._thread = Thread(
            target=self._run,
            name="clothdojo-teleop-writer",
            daemon=False,
        )
        self._thread.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def append(self, action: TrajectoryAction, next_state: TrajectoryState) -> None:
        if self._terminal_queued:
            raise RuntimeError("cannot append after trajectory finalization has started")
        self._enqueue(_AppendTransition(action, next_state))

    def close(self, reason: str | None) -> None:
        if self._terminal_queued:
            raise RuntimeError("trajectory finalization has already started")
        self._terminal_queued = True
        self._enqueue(_CloseTrajectory(reason))

    def abort(self, reason: str | None) -> None:
        if self._terminal_queued:
            raise RuntimeError("trajectory finalization has already started")
        self._terminal_queued = True
        self._enqueue(_AbortTrajectory(reason))

    def wait(self) -> None:
        self._thread.join()
        if self._failure is not None:
            raise self._failure

    def _enqueue(self, command: _WriterCommand) -> None:
        accepted = False
        while not self._done.is_set():
            try:
                self._queue.put(command, timeout=0.05)
                accepted = True
                break
            except Full:
                continue
        if not accepted:
            self.wait()
            raise RuntimeError("trajectory writer stopped before accepting a command")

    def _run(self) -> None:
        command: _WriterCommand | None = None
        try:
            while True:
                command = self._queue.get()
                if isinstance(command, _AppendTransition):
                    self._writer.append_transition(command.action, command.next_state)
                    continue
                if isinstance(command, _CloseTrajectory):
                    if command.reason is not None:
                        self._writer.manifest.metadata["stop_reason"] = str(command.reason)
                    self._writer.close()
                    return
                if command.reason is not None:
                    self._writer.manifest.metadata["abort_reason"] = str(command.reason)
                self._writer.abort()
                return
        except BaseException as error:
            if not isinstance(command, _AbortTrajectory):
                try:
                    reason = (
                        "append_failed"
                        if isinstance(command, _AppendTransition)
                        else "finalization_failed"
                    )
                    self._writer.manifest.metadata["abort_reason"] = reason
                    self._writer.abort()
                except BaseException as abort_error:
                    error.add_note(f"trajectory cleanup also failed: {abort_error!r}")
            self._failure = error
        finally:
            self._done.set()


class TeleopRecorder:
    """Record simulation frames while keeping filesystem work off the XR loop."""

    def __init__(self, runtime, plan: SessionPlan) -> None:
        self.runtime = runtime
        self.plan = plan
        self._worker: _TrajectoryWriterWorker | None = None
        self._path: Path | None = None
        self._sample_index = 0
        self._pending_worker: _TrajectoryWriterWorker | None = None
        self._pending_path: Path | None = None
        self._pending_complete = False
        self._background_error: BaseException | None = None

    @property
    def is_recording(self) -> bool:
        return self._worker is not None

    @property
    def is_finalizing(self) -> bool:
        return self._pending_worker is not None

    @property
    def frame_count(self) -> int:
        return self._sample_index if self._worker is not None else 0

    def poll_finalization(self) -> Path | None:
        """Return a newly committed trajectory without blocking the XR loop."""

        self._raise_background_error()
        active = self._worker
        if active is not None and active.done:
            self._detach_worker(active, complete=False)
            return self.wait_for_finalization()
        worker = self._pending_worker
        if worker is None or not worker.done:
            return None
        return self.wait_for_finalization()

    def wait_for_finalization(self) -> Path | None:
        """Wait for pending disk work and propagate its first failure."""

        self._raise_background_error()
        worker = self._pending_worker
        if worker is None:
            return None
        path = self._pending_path
        complete = self._pending_complete
        try:
            worker.wait()
        except BaseException as error:
            self._clear_pending()
            self._background_error = error
            raise
        self._clear_pending()
        return path if complete else None

    def start(self, *, origin: Mapping[str, object] | None = None) -> Path:
        self.poll_finalization()
        if self._worker is not None:
            raise RuntimeError("teleoperation recording is already active")
        if self._pending_worker is not None:
            raise RuntimeError("the previous trajectory is still finalizing")
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
        writer = TrajectoryWriter(
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
            writer.write_initial_state(states[0])
        except Exception:
            writer.abort()
            raise
        self._worker = _TrajectoryWriterWorker(writer)
        self._path = output
        self._sample_index = 0
        return output

    def step(self, arm_command, gripper_open) -> bool:
        """Record exactly the command passed to Newton and its resulting state."""

        worker = self._worker
        if worker is None:
            self.runtime.step(arm_command, gripper_open)
            return False
        sample_index = self._sample_index
        try:
            actions = capture_actions(
                self.runtime,
                sample_index,
                arm_command,
                gripper_open,
            )
            self.runtime.step(arm_command, gripper_open)
            states = capture_states(self.runtime, sample_index + 1)
            if len(actions) != 1 or len(states) != 1:
                raise RuntimeError("teleoperation requires exactly one Newton world")
            worker.append(actions[0], states[0])
        except Exception:
            if self._worker is worker:
                if worker.done:
                    self._detach_worker(worker, complete=False)
                    self.wait_for_finalization()
                else:
                    self.abort("append_failed")
            raise
        self._sample_index += 1
        if self._sample_index >= self.plan.max_actions:
            self.stop(reason="max_actions_reached")
            return True
        return False

    def stop(self, *, reason: str | None = None) -> Path | None:
        worker = self._worker
        if worker is None:
            return None
        path = self._path
        if self._sample_index == 0:
            worker.abort(None)
            self._detach_worker(worker, complete=False)
            self.wait_for_finalization()
            if path is not None:
                shutil.rmtree(path)
            return None
        try:
            worker.close(reason)
        except BaseException:
            self._detach_worker(worker, complete=False)
            self.wait_for_finalization()
            raise
        self._detach_worker(worker, complete=True)
        return path

    def abort(self, reason: str) -> Path | None:
        worker = self._worker
        if worker is None:
            return None
        path = self._path
        try:
            worker.abort(reason)
        except BaseException:
            self._detach_worker(worker, complete=False)
            self.wait_for_finalization()
            raise
        self._detach_worker(worker, complete=False)
        return path

    def close(self) -> Path | None:
        if self._worker is not None:
            self.stop()
        return self.wait_for_finalization()

    def _detach_worker(
        self,
        worker: _TrajectoryWriterWorker,
        *,
        complete: bool,
    ) -> None:
        if self._worker is not worker:
            raise RuntimeError("teleop writer lifecycle is inconsistent")
        if self._pending_worker is not None:
            raise RuntimeError("another trajectory is already finalizing")
        self._pending_worker = worker
        self._pending_path = self._path
        self._pending_complete = bool(complete)
        self._worker = None
        self._path = None
        self._sample_index = 0

    def _clear_pending(self) -> None:
        self._pending_worker = None
        self._pending_path = None
        self._pending_complete = False

    def _raise_background_error(self) -> None:
        if self._background_error is not None:
            raise self._background_error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
