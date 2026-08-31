from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo import cli
from clothdojo.data.annotations import build_good_segment_plan
from clothdojo.data.mirror import (
    _PIPER_HOME_BODY_QUATERNIONS,
    PIPER_ARM_SIGNS,
    augment_mirror,
    mirror_action_arrays,
    mirror_state_arrays,
)
from clothdojo.trajectory import (
    SceneMetadata,
    TrajectoryAction,
    TrajectoryReader,
    TrajectoryState,
    TrajectoryWriter,
)


def _state(index: int) -> TrajectoryState:
    body_q = np.arange(21 * 7, dtype=np.float32).reshape(21, 7) / 100.0
    body_q[:, :3] += index
    body_q[:, 3:7] /= np.linalg.norm(body_q[:, 3:7], axis=1, keepdims=True)
    body_qd = np.arange(21 * 6, dtype=np.float32).reshape(21, 6) / 50.0
    particle_q = np.asarray(
        ((0.5, 0.2, 0.8), (0.6, -0.3, 0.9), (0.7, 0.0, 1.0)),
        dtype=np.float32,
    )
    particle_q[:, 0] += index
    pin_local = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3) / 1000.0
    return TrajectoryState(
        sample_index=index,
        joint_q=np.arange(16, dtype=np.float32) + index,
        joint_qd=np.arange(16, dtype=np.float32) / 10.0 + index,
        body_q=body_q,
        body_qd=body_qd,
        particle_q=particle_q,
        particle_qd=particle_q / 10.0,
        gripper_open_state=np.asarray((0.2, 0.8), dtype=np.float32),
        grasp_attached=np.asarray((1, 0), dtype=np.int32),
        grasp_previous_open=np.asarray((0.25, 0.75), dtype=np.float32),
        grasp_pin_count=np.asarray((2, 1), dtype=np.int32),
        grasp_pins=np.asarray(((2, 1, -1), (0, -1, -1)), dtype=np.int32),
        grasp_pin_local=pin_local,
    )


def _action(index: int) -> TrajectoryAction:
    return TrajectoryAction(
        sample_index=index,
        joint_target=np.arange(12, dtype=np.float32) + 0.25 + index,
        gripper_open_command=np.asarray((0.1, 0.9), dtype=np.float32),
    )


def _collection(tmp_path: Path, *, robot_name: str = "piper") -> tuple[Path, str]:
    root = tmp_path / "source"
    trajectory_id = "episode_000"
    trajectory = root / "trajectories" / trajectory_id
    with TrajectoryWriter(
        trajectory,
        control_fps=50.0,
        sim_substeps=6,
        max_actions=2,
        scene=SceneMetadata(task_name="shirt_fold", robot_id=robot_name),
        metadata={"asset_id": "test_shirt", "task_prompt": "fold the shirt"},
    ) as writer:
        writer.write_initial_state(_state(0))
        writer.append_transition(_action(0), _state(1))
        writer.append_transition(_action(1), _state(2))
    annotation = root / "trajectories" / "annotations" / f"{trajectory_id}.segments.json"
    annotation.parent.mkdir()
    annotation.write_text(
        json.dumps(
            {
                "schema": "sim1_segments_v1",
                "review_status": "complete",
                "segments": [{"start_frame": 1, "end_frame": 1, "label": "bad"}],
                "updated_at": "2026-08-28T00:00:00Z",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, trajectory_id


def test_array_mirror_is_involutive_and_uses_piper_semantics(tmp_path: Path) -> None:
    source, trajectory_id = _collection(tmp_path)
    reader = TrajectoryReader(source / "trajectories" / trajectory_id)
    states = reader.state_arrays()
    actions = reader.action_arrays()

    mirrored_states = mirror_state_arrays(states)
    mirrored_actions = mirror_action_arrays(actions)

    for name, values in states.items():
        np.testing.assert_allclose(
            mirror_state_arrays(mirrored_states)[name],
            values,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
    for name, values in actions.items():
        np.testing.assert_array_equal(mirror_action_arrays(mirrored_actions)[name], values)

    np.testing.assert_array_equal(
        mirrored_states["joint_q"][:, :6], states["joint_q"][:, 8:14] * PIPER_ARM_SIGNS
    )
    np.testing.assert_array_equal(mirrored_states["joint_q"][:, 6:8], states["joint_q"][:, 14:16])
    np.testing.assert_array_equal(
        mirrored_states["particle_q"][..., 0], states["particle_q"][..., 0]
    )
    np.testing.assert_array_equal(
        mirrored_states["particle_q"][..., 1], -states["particle_q"][..., 1]
    )
    np.testing.assert_array_equal(
        mirrored_states["body_q"][:, 1, :3], states["body_q"][:, 11, :3] * (1, -1, 1)
    )
    np.testing.assert_array_equal(
        mirrored_states["body_q"][:, 1, 3:7],
        states["body_q"][:, 11, 3:7] * (-1, 1, -1, 1),
    )
    np.testing.assert_array_equal(
        mirrored_states["body_qd"][:, 1],
        states["body_qd"][:, 11] * (1, -1, 1, -1, 1, -1),
    )
    np.testing.assert_array_equal(
        mirrored_states["body_q"][:, 9, :3], states["body_q"][:, 19, :3] * (1, -1, 1)
    )
    np.testing.assert_array_equal(
        mirrored_states["body_q"][:, 10, :3], states["body_q"][:, 20, :3] * (1, -1, 1)
    )
    np.testing.assert_array_equal(
        mirrored_states["grasp_pins"][:, 0], states["grasp_pins"][:, 1]
    )
    np.testing.assert_array_equal(
        mirrored_actions["joint_target"][:, :6],
        actions["joint_target"][:, 6:12] * PIPER_ARM_SIGNS,
    )


def test_body_mirror_preserves_piper_home_frame_and_finger_identity(tmp_path: Path) -> None:
    source, trajectory_id = _collection(tmp_path)
    states = TrajectoryReader(source / "trajectories" / trajectory_id).state_arrays()
    states["body_q"] = states["body_q"].copy()
    states["body_q"][..., 3:7] = _PIPER_HOME_BODY_QUATERNIONS

    mirrored = mirror_state_arrays(states)

    np.testing.assert_allclose(
        mirrored["body_q"][..., 3:7],
        np.broadcast_to(_PIPER_HOME_BODY_QUATERNIONS, mirrored["body_q"][..., 3:7].shape),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        mirrored["body_q"][:, 9, :3], states["body_q"][:, 19, :3] * (1, -1, 1)
    )
    np.testing.assert_array_equal(
        mirrored["body_q"][:, 10, :3], states["body_q"][:, 20, :3] * (1, -1, 1)
    )


def test_mirrored_grasp_pins_reconstruct_reflected_world_positions(tmp_path: Path) -> None:
    from scipy.spatial.transform import Rotation

    source, trajectory_id = _collection(tmp_path)
    states = TrajectoryReader(source / "trajectories" / trajectory_id).state_arrays()
    mirrored = mirror_state_arrays(states)

    def pin_world_positions(arrays):
        # Piper attaches particles in link6 coordinates, offset along local z.
        bodies = arrays["body_q"][:, (7, 17)]
        rotations = Rotation.from_quat(bodies[..., 3:].reshape(-1, 4)).as_matrix()
        rotations = rotations.reshape(*bodies.shape[:2], 3, 3)
        local = arrays["grasp_pin_local"] + np.asarray((0.0, 0.0, 0.13503))
        return bodies[..., None, :3] + np.einsum("nsij,nspj->nspi", rotations, local)

    expected = pin_world_positions(states)[:, ::-1] * (1, -1, 1)
    np.testing.assert_allclose(pin_world_positions(mirrored), expected, atol=5e-7)


def test_augmentation_publishes_independent_original_and_mirror_collection(
    tmp_path: Path,
) -> None:
    source, trajectory_id = _collection(tmp_path)
    source_trajectory = source / "trajectories" / trajectory_id
    source_hashes = {
        path.name: _sha256(path)
        for path in source_trajectory.iterdir()
        if path.is_file()
    }
    output = tmp_path / "augmented"

    result = augment_mirror(source, output)

    assert result == {
        "output": str(output),
        "input_trajectories": 1,
        "output_trajectories": 2,
        "skipped_incomplete_trajectories": 0,
        "annotation_files": 2,
        "normalized_legacy_manifests": 0,
    }
    copied = output / "trajectories" / trajectory_id
    assert {
        path.name: _sha256(path) for path in copied.iterdir() if path.is_file()
    } == source_hashes
    mirror_id = f"{trajectory_id}__mirror"
    mirror = TrajectoryReader(output / "trajectories" / mirror_id)
    assert mirror.manifest.scene.task_name == "shirt_fold"
    augmentation = mirror.manifest.metadata["augmentation"]
    assert augmentation["kind"] == "bilateral_mirror"
    assert augmentation["source_trajectory_id"] == trajectory_id
    assert augmentation["kinematics_used"] is False
    assert augmentation["physics_used"] is False
    assert augmentation["arm_joint_signs"] == [-1, 1, 1, -1, 1, -1]
    assert augmentation["body_orientation_frame"] == "piper_home_body"
    assert augmentation["surface_winding_reversed"] is True
    original_annotation = (
        source / "trajectories" / "annotations" / f"{trajectory_id}.segments.json"
    ).read_bytes()
    assert (
        output / "trajectories" / "annotations" / f"{trajectory_id}.segments.json"
    ).read_bytes() == original_annotation
    assert (
        output / "trajectories" / "annotations" / f"{mirror_id}.segments.json"
    ).read_bytes() == original_annotation
    collection = json.loads((output / "augmentation.json").read_text(encoding="utf-8"))
    assert collection["status"] == "complete"
    assert collection["source_is_runtime_dependency"] is False
    assert collection["output_trajectories"] == 2
    assert collection["surface_winding_reversed"] is True
    assert collection["trajectory_pairs"][0]["mirror_id"] == mirror_id
    plan = build_good_segment_plan(output, min_segment_frames=1)
    assert plan.source_trajectories == 2
    assert [segment.source_trajectory for segment in plan.segments] == [
        trajectory_id,
        mirror_id,
    ]
    assert all(segment.review_status == "complete" for segment in plan.segments)

    source.rename(tmp_path / "source-removed")
    TrajectoryReader(copied).validate_training_data()
    mirror.validate_training_data()


def test_augmentation_normalizes_legacy_manifest_in_independent_copy(
    tmp_path: Path,
) -> None:
    source, trajectory_id = _collection(tmp_path)
    manifest_path = source / "trajectories" / trajectory_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("format_version")
    manifest["schema_version"] = "newton_cloth_episode"
    manifest["state_profile"] = "full"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    legacy_hash = _sha256(manifest_path)
    output = tmp_path / "augmented"

    result = augment_mirror(source, output)

    assert result["normalized_legacy_manifests"] == 1
    copied = TrajectoryReader(output / "trajectories" / trajectory_id)
    copied.validate_training_data()
    mirror = TrajectoryReader(output / "trajectories" / f"{trajectory_id}__mirror")
    augmentation = mirror.manifest.metadata["augmentation"]
    assert augmentation["legacy_source_manifest_sha256"] == legacy_hash
    pair = json.loads((output / "augmentation.json").read_text(encoding="utf-8"))[
        "trajectory_pairs"
    ][0]
    assert pair["legacy_source_manifest_sha256"] == legacy_hash


def test_augmentation_rejects_non_piper_without_publishing(tmp_path: Path) -> None:
    source, _trajectory_id = _collection(tmp_path, robot_name="aloha2")
    output = tmp_path / "augmented"

    with pytest.raises(ValueError, match="requires Piper"):
        augment_mirror(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".augmented.tmp-*"))


def test_mirror_cli_forwards_compact_collection_options(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.data.mirror.augment_mirror",
        lambda *args: call.update(args=args) or {"ok": True},
    )

    cli.main(
        [
            "data",
            "augment-mirror",
            "--source",
            "source",
            "--output",
            "output",
        ]
    )

    assert call == {"args": ("source", "output")}
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_review_cli_forwards_selected_mirror_ids(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.data.rollout_review.render_review",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs) or {"ok": True},
    )

    cli.main(
        [
            "data",
            "render-review",
            "--source",
            "augmented",
            "--trajectory",
            "episode_a__mirror",
            "--trajectory",
            "episode_b__mirror",
        ]
    )

    assert call["args"] == ("augmented",)
    assert call["kwargs"]["trajectory_ids"] == [
        "episode_a__mirror",
        "episode_b__mirror",
    ]
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
