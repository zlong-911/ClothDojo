from __future__ import annotations

import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import clothdojo.teleop.recording as recording
from clothdojo.trajectory import (
    TrajectoryAction,
    TrajectoryReader,
    TrajectoryState,
    TrajectoryWriter,
)


def _state(index: int) -> TrajectoryState:
    return TrajectoryState(
        sample_index=index,
        joint_q=np.full(4, index, dtype=np.float32),
        joint_qd=np.zeros(4, dtype=np.float32),
        body_q=np.zeros((2, 7), dtype=np.float32),
        body_qd=np.zeros((2, 6), dtype=np.float32),
        particle_q=np.full((3, 3), index, dtype=np.float32),
        particle_qd=np.zeros((3, 3), dtype=np.float32),
        gripper_open_state=np.ones(2, dtype=np.float32),
        grasp_attached=np.zeros(2, dtype=np.int32),
        grasp_previous_open=np.ones(2, dtype=np.float32),
        grasp_pin_count=np.zeros(2, dtype=np.int32),
        grasp_pins=np.full((2, 2), -1, dtype=np.int32),
        grasp_pin_local=np.zeros((2, 2, 3), dtype=np.float32),
    )


def _action(index: int) -> TrajectoryAction:
    return TrajectoryAction(
        sample_index=index,
        joint_target=np.full(12, index, dtype=np.float32),
        gripper_open_command=np.ones(2, dtype=np.float32),
    )


class _Runtime:
    spec = SimpleNamespace(control_fps=50.0, sim_substeps=6)

    def __init__(self) -> None:
        self.steps = 0

    def step(self, _arm_command, _gripper_open) -> None:
        self.steps += 1


def _plan(tmp_path):
    assets = tmp_path / "assets"
    garment = assets / "garments/converted/shirt/shirt.usdc"
    robot = assets / "robots/piper/piper.usda"
    garment.parent.mkdir(parents=True)
    robot.parent.mkdir(parents=True)
    garment.write_bytes(b"garment")
    robot.write_bytes(b"robot")
    return SimpleNamespace(
        output_root=tmp_path / "trajectories",
        assets_root=assets,
        asset_path=garment,
        session_id="session",
        profile_id="shirt_flatten",
        asset_id="shirt",
        mode="collection",
        scene="shirt_flatten",
        max_actions=20,
    )


@pytest.fixture
def deterministic_capture(monkeypatch):
    monkeypatch.setattr(
        recording,
        "capture_states",
        lambda _runtime, sample_index: (_state(sample_index),),
    )
    monkeypatch.setattr(
        recording,
        "capture_actions",
        lambda _runtime, sample_index, _arm, _gripper: (_action(sample_index),),
    )


def test_teleop_finalization_does_not_block_the_control_thread(
    tmp_path, monkeypatch, deterministic_capture
) -> None:
    close_started = Event()
    allow_close = Event()

    class BlockingWriter(TrajectoryWriter):
        def close(self) -> None:
            close_started.set()
            if not allow_close.wait(timeout=5.0):
                raise TimeoutError("test did not release trajectory finalization")
            super().close()

    monkeypatch.setattr(recording, "TrajectoryWriter", BlockingWriter)
    recorder = recording.TeleopRecorder(_Runtime(), _plan(tmp_path))
    path = recorder.start()
    recorder.step(object(), object())

    assert recorder.stop(reason="operator_stop") == path
    assert close_started.wait(timeout=2.0)
    assert recorder.is_finalizing
    assert recorder.poll_finalization() is None
    with pytest.raises(RuntimeError, match="still finalizing"):
        recorder.start()

    allow_close.set()
    assert recorder.wait_for_finalization() == path
    assert not recorder.is_finalizing
    reader = TrajectoryReader(path)
    assert reader.manifest.action_count == 1
    assert reader.manifest.metadata["stop_reason"] == "operator_stop"


def test_teleop_finalization_failure_is_persistent_and_aborts_output(
    tmp_path, monkeypatch, deterministic_capture
) -> None:
    class FailingWriter(TrajectoryWriter):
        def close(self) -> None:
            raise RuntimeError("disk failed")

    monkeypatch.setattr(recording, "TrajectoryWriter", FailingWriter)
    recorder = recording.TeleopRecorder(_Runtime(), _plan(tmp_path))
    path = recorder.start()
    recorder.step(object(), object())
    recorder.stop()

    with pytest.raises(RuntimeError, match="disk failed"):
        recorder.wait_for_finalization()
    assert not recorder.is_finalizing
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
    assert manifest["metadata"]["abort_reason"] == "finalization_failed"
    with pytest.raises(RuntimeError, match="disk failed"):
        recorder.poll_finalization()


def test_teleop_background_writer_preserves_every_queued_transition(
    tmp_path, deterministic_capture
) -> None:
    runtime = _Runtime()
    recorder = recording.TeleopRecorder(runtime, _plan(tmp_path))
    path = recorder.start(origin={"source": "task_continuation"})
    for _ in range(7):
        recorder.step(object(), object())
    recorder.stop(reason="continuation_advance")

    assert recorder.close() == path
    reader = TrajectoryReader(path)
    assert runtime.steps == 7
    assert reader.manifest.action_count == 7
    assert reader.manifest.state_count == 8
    assert reader.manifest.metadata["source"] == "task_continuation"
    assert reader.manifest.metadata["stop_reason"] == "continuation_advance"
    assert reader.terminal_state().sample_index == 7


def test_empty_teleop_recording_is_removed(tmp_path, deterministic_capture) -> None:
    recorder = recording.TeleopRecorder(_Runtime(), _plan(tmp_path))
    path = recorder.start()

    assert recorder.stop() is None
    assert not path.exists()
    assert not recorder.is_finalizing
