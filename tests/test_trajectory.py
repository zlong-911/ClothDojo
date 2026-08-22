from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.trajectory import (
    TrajectoryAction,
    TrajectoryReader,
    TrajectoryState,
    TrajectoryWriter,
)


def _state(index: int) -> TrajectoryState:
    return TrajectoryState(
        sample_index=index,
        joint_q=np.full((3,), index, dtype=np.float64),
        joint_qd=np.full((3,), index + 0.1, dtype=np.float64),
        body_q=np.full((2, 7), index, dtype=np.float64),
        body_qd=np.full((2, 6), index + 0.2, dtype=np.float64),
        particle_q=np.full((4, 3), index, dtype=np.float64),
        particle_qd=np.full((4, 3), index + 0.3, dtype=np.float64),
        gripper_open_state=np.full((2,), index, dtype=np.float64),
        grasp_attached=np.asarray([index % 2, 0], dtype=np.int64),
        grasp_previous_open=np.full((2,), index + 0.4, dtype=np.float64),
        grasp_pin_count=np.asarray([1, 0], dtype=np.int64),
        grasp_pins=np.asarray([[2, -1], [-1, -1]], dtype=np.int64),
        grasp_pin_local=np.full((2, 2, 3), index + 0.5, dtype=np.float64),
    )


def _action(index: int) -> TrajectoryAction:
    return TrajectoryAction(
        sample_index=index,
        joint_target=np.full((3,), index + 0.25, dtype=np.float64),
        gripper_open_command=np.full((2,), index + 0.5, dtype=np.float64),
    )


def _write_trajectory(path: Path, *, action_count: int = 3) -> Path:
    writer = TrajectoryWriter(
        path,
        control_fps=30.0,
        sim_substeps=20,
        max_actions=max(1, action_count),
        chunk_size=2,
    )
    writer.write_initial_state(_state(0))
    for index in range(action_count):
        writer.append_transition(_action(index), _state(index + 1))
    writer.close()
    return path


def test_writer_preserves_n_plus_one_alignment_and_reader_views(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory"
    writer = TrajectoryWriter(
        trajectory,
        control_fps=30.0,
        sim_substeps=20,
        chunk_size=2,
    )
    writer.write_initial_state(_state(0))
    for index in range(3):
        writer.append_transition(_action(index), _state(index + 1))

    assert (trajectory / ".chunks").is_dir()
    assert not (trajectory / "states.npz").exists()
    writer.close()

    reader = TrajectoryReader(trajectory)
    samples = list(reader.iter_samples())
    assert reader.manifest.state_count == 4
    assert reader.manifest.action_count == 3
    assert [sample.state.sample_index for sample in samples] == [0, 1, 2]
    assert [sample.action.sample_index for sample in samples] == [0, 1, 2]
    assert reader.terminal_state().sample_index == 3
    assert samples[0].state.particle_q.dtype == np.float32
    assert samples[0].state.grasp_pins.dtype == np.int32
    assert samples[0].action.joint_target.dtype == np.float32
    assert samples[0].state.particle_q.flags.writeable is False
    assert not (trajectory / ".chunks").exists()
    manifest = json.loads((trajectory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert "schema_version" not in manifest
    assert "state_profile" not in manifest


def test_reader_rejects_manifest_without_coordinate_frame(
    tmp_path: Path,
) -> None:
    trajectory = _write_trajectory(tmp_path / "trajectory")
    manifest_path = trajectory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("coordinate_frame")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported trajectory coordinate frame"):
        TrajectoryReader(trajectory)


def test_training_validation_excludes_unpaired_terminal_state(tmp_path: Path) -> None:
    trajectory = _write_trajectory(tmp_path / "trajectory")
    states_path = trajectory / "states.npz"
    with np.load(states_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["particle_q"][-1, 0, 0] = np.nan
    with states_path.open("wb") as file:
        np.savez_compressed(file, **payload)

    reader = TrajectoryReader(trajectory)
    reader.validate_training_data()
    assert np.isnan(reader.terminal_state().particle_q[0, 0])


def test_writer_rejects_shifted_action_index(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path / "trajectory", control_fps=30.0, sim_substeps=20)
    writer.write_initial_state(_state(0))

    with pytest.raises(ValueError, match="does not match source state"):
        writer.append_transition(_action(1), _state(1))

    writer.abort()
    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
