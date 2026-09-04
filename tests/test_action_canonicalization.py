from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.data.action_canonicalization import (
    _apply_limit_overrides,
    _resolved_bounds,
    canonicalize_action_collection,
    canonicalize_bounded_actions,
)
from clothdojo.robots import PIPER, robot_profile
from clothdojo.trajectory import (
    SceneMetadata,
    TrajectoryAction,
    TrajectoryReader,
    TrajectoryState,
    TrajectoryWriter,
)


def _state(index: int) -> TrajectoryState:
    return TrajectoryState(
        sample_index=index,
        joint_q=np.full(16, index, dtype=np.float32),
        joint_qd=np.zeros(16, dtype=np.float32),
        body_q=np.zeros((21, 7), dtype=np.float32),
        body_qd=np.zeros((21, 6), dtype=np.float32),
        particle_q=np.full((3, 3), index, dtype=np.float32),
        particle_qd=np.zeros((3, 3), dtype=np.float32),
        gripper_open_state=np.ones(2, dtype=np.float32),
        grasp_attached=np.zeros(2, dtype=np.int32),
        grasp_previous_open=np.ones(2, dtype=np.float32),
        grasp_pin_count=np.zeros(2, dtype=np.int32),
        grasp_pins=np.full((2, 2), -1, dtype=np.int32),
        grasp_pin_local=np.zeros((2, 2, 3), dtype=np.float32),
    )


def _collection(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    trajectory = source / "trajectories" / "episode"
    profile = robot_profile(PIPER)
    usd = Path(__file__).parents[1] / "assets" / profile.asset
    joint_names = list(profile.arm_joint_names)
    writer = TrajectoryWriter(
        trajectory,
        control_fps=50,
        sim_substeps=10,
        scene=SceneMetadata(task_name="shirt_flatten", robot_id="piper"),
        metadata={
            "joint_names": joint_names,
            "scene_config": {
                "robot": {
                    "usd_path": str(usd),
                    "arm_joint_position_limit_overrides_rad": [],
                }
            },
        },
    )
    writer.write_initial_state(_state(0))
    raw = np.zeros(12, dtype=np.float32)
    raw[[0, 6]] = (2.0, -2.0)
    raw[[5, 11]] = (4.0, -4.0)
    writer.append_transition(
        TrajectoryAction(
            sample_index=0,
            joint_target=raw,
            gripper_open_command=np.asarray((-0.2, 1.2), dtype=np.float32),
        ),
        _state(1),
    )
    writer.close()
    annotations = source / "trajectories" / "annotations"
    annotations.mkdir()
    (annotations / "episode.segments.json").write_bytes(
        b'{"schema":"sim1_segments_v1","segments":[],"review_status":"complete"}\n'
    )
    return source


def test_canonicalize_bounded_actions_requires_float32_and_clips_both_fields() -> None:
    lower = np.asarray((-1.0, -2.0), dtype=np.float32)
    upper = np.asarray((1.0, 2.0), dtype=np.float32)

    arm, gripper = canonicalize_bounded_actions(
        np.asarray(((3.0, -3.0),), dtype=np.float32),
        np.asarray(((-0.1, 1.1),), dtype=np.float32),
        lower=lower,
        upper=upper,
    )

    np.testing.assert_array_equal(arm, [[1.0, -2.0]])
    np.testing.assert_array_equal(gripper, [[0.0, 1.0]])
    with pytest.raises(TypeError, match="float32"):
        canonicalize_bounded_actions(
            np.asarray(((0.0, 0.0),), dtype=np.float64),
            gripper,
            lower=lower,
            upper=upper,
        )


def test_action_snapshot_copies_states_and_publishes_bounded_actions(tmp_path) -> None:
    source = _collection(tmp_path)
    output = tmp_path / "canonical"

    result = canonicalize_action_collection(source, output)

    assert result["input_trajectories"] == 1
    assert result["changed_frames"] == 1
    snapshot = output / "trajectories" / "episode"
    source_trajectory = source / "trajectories" / "episode"
    assert (snapshot / "states.npz").read_bytes() == (
        source_trajectory / "states.npz"
    ).read_bytes()
    assert not (snapshot / "states.npz").samefile(source_trajectory / "states.npz")
    assert (output / "trajectories/annotations/episode.segments.json").read_bytes() == (
        source / "trajectories/annotations/episode.segments.json"
    ).read_bytes()
    base = TrajectoryReader(source_trajectory)
    reader = TrajectoryReader(snapshot)
    np.testing.assert_array_equal(reader.state_at(0).particle_q, base.state_at(0).particle_q)
    actions = reader.action_arrays()
    np.testing.assert_allclose(
        actions["joint_target"][[0], [0, 6]],
        [0.5 * np.pi, -0.5 * np.pi],
    )
    np.testing.assert_array_equal(actions["gripper_open_command"], [[0.0, 1.0]])
    assert reader.manifest.metadata["action_semantics"] == "runtime_target"
    assert "storage" not in reader.manifest.metadata
    assert json.loads((output / "action_canonicalization.json").read_text())[
        "source_is_runtime_dependency"
    ] is False


def test_action_snapshot_is_independent_from_the_source(tmp_path) -> None:
    source = _collection(tmp_path)
    output = tmp_path / "canonical"
    source_trajectory = source / "trajectories/episode"
    source_manifest = (source_trajectory / "manifest.json").read_bytes()
    source_actions = (source_trajectory / "actions.npz").read_bytes()
    source_states = (source_trajectory / "states.npz").read_bytes()
    canonicalize_action_collection(source, output)
    with (output / "trajectories/episode/states.npz").open("ab") as file:
        file.write(b"tampered")

    assert (source_trajectory / "manifest.json").read_bytes() == source_manifest
    assert (source_trajectory / "actions.npz").read_bytes() == source_actions
    assert (source_trajectory / "states.npz").read_bytes() == source_states


def test_action_snapshot_creates_a_missing_output_parent(tmp_path) -> None:
    source = _collection(tmp_path)
    output = tmp_path / "new" / "nested" / "canonical"

    canonicalize_action_collection(source, output)

    assert (output / "action_canonicalization.json").is_file()


def test_limit_override_cannot_widen_or_duplicate_asset_range() -> None:
    with pytest.raises(RuntimeError, match="widens"):
        _apply_limit_overrides(
            {"left/joint1": -1.0},
            {"left/joint1": 1.0},
            [["left/joint1", -2.0, 0.5]],
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        _apply_limit_overrides(
            {"left/joint1": -1.0},
            {"left/joint1": 1.0},
            [["left/joint1", -0.5, 0.5], ["left/joint1", -0.25, 0.25]],
        )


def test_resolved_bounds_equal_newton_runtime_model_limits(tmp_path) -> None:
    import newton
    import warp as wp

    from clothdojo.sim.scene import (
        _add_robot,
        _joint_dofs,
        _register_solver_attributes,
        _select_piper_robot,
    )

    source = _collection(tmp_path)
    reader = TrajectoryReader(source / "trajectories/episode")
    bounds = _resolved_bounds(reader.manifest.to_json_dict(), {})
    profile = robot_profile(PIPER)
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(
        newton,
        wp,
        builder,
        Path(__file__).parents[1] / "assets" / profile.asset,
        profile,
        load_visual_shapes=False,
    )
    arm_joints, _gripper_joints, arm_dofs = _select_piper_robot(builder)
    assert arm_dofs == _joint_dofs(builder, arm_joints)
    np.testing.assert_array_equal(
        bounds.lower,
        np.asarray(builder.joint_limit_lower, dtype=np.float32)[arm_dofs],
    )
    np.testing.assert_array_equal(
        bounds.upper,
        np.asarray(builder.joint_limit_upper, dtype=np.float32)[arm_dofs],
    )


def test_action_canonicalization_plan_does_not_create_output(tmp_path) -> None:
    source = _collection(tmp_path)
    output = tmp_path / "canonical"

    report = canonicalize_action_collection(source, output, plan_only=True)

    assert report["changed_frames"] == 1
    assert not output.exists()


def test_action_canonicalization_plan_validates_the_output_target(tmp_path) -> None:
    source = _collection(tmp_path)
    output = tmp_path / "canonical"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        canonicalize_action_collection(source, output, plan_only=True)


def test_action_canonicalization_rejects_a_runtime_target_source(tmp_path) -> None:
    source = _collection(tmp_path)
    output = tmp_path / "canonical"
    canonicalize_action_collection(source, output)

    with pytest.raises(RuntimeError, match="action_semantics must be absent"):
        canonicalize_action_collection(output, tmp_path / "canonical-twice", plan_only=True)


def test_action_canonicalization_rejects_unknown_source_action_semantics(tmp_path) -> None:
    source = _collection(tmp_path)
    manifest_path = source / "trajectories/episode/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["action_semantics"] = "delta_joint_target"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="action_semantics must be absent"):
        canonicalize_action_collection(source, plan_only=True)
