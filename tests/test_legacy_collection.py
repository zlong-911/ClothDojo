from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clothdojo.data.legacy_collection import (
    IMPORT_SCHEMA,
    import_legacy_collection,
    inspect_legacy_collection,
)
from clothdojo.trajectory import TrajectoryReader


def test_legacy_collection_import_normalizes_manifest_and_annotations(tmp_path: Path) -> None:
    source = tmp_path / "old"
    episode = source / "trajectories/episode_a"
    episode.mkdir(parents=True)
    state_shapes = {
        "joint_q": [1],
        "joint_qd": [1],
        "body_q": [1, 7],
        "body_qd": [1, 6],
        "particle_q": [1, 3],
        "particle_qd": [1, 3],
        "gripper_open_state": [2],
        "grasp_attached": [2],
        "grasp_previous_open": [2],
        "grasp_pin_count": [2],
        "grasp_pins": [2, 1],
        "grasp_pin_local": [2, 1, 3],
    }
    manifest = {
        "schema_version": "newton_cloth_episode",
        "state_profile": "full",
        "alignment": "states_n_plus_1_actions_n",
        "status": "complete",
        "control_fps": 50.0,
        "sim_substeps": 6,
        "units": "m",
        "coordinate_convention": "right_handed_z_up",
        "coordinate_frame": "world_local",
        "max_actions": 10,
        "state_count": 3,
        "action_count": 2,
        "state_shapes": state_shapes,
        "action_shapes": {"joint_target": [1], "gripper_open_command": [2]},
        "scene": {
            "task_name": "shirt_flatten",
            "robot_name": "piper",
            "asset_fingerprints": {},
            "config_fingerprints": {},
        },
        "metadata": {"asset_id": "shirt_a"},
    }
    (episode / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    states = {"sample_index": np.arange(3, dtype=np.int64)}
    float_fields = {
        "joint_q",
        "joint_qd",
        "body_q",
        "body_qd",
        "particle_q",
        "particle_qd",
        "gripper_open_state",
        "grasp_previous_open",
        "grasp_pin_local",
    }
    for name, shape in state_shapes.items():
        dtype = np.float32 if name in float_fields else np.int32
        states[name] = np.zeros((3, *shape), dtype=dtype)
    np.savez(episode / "states.npz", **states)
    np.savez(
        episode / "actions.npz",
        sample_index=np.arange(2, dtype=np.int64),
        joint_target=np.zeros((2, 1), dtype=np.float32),
        gripper_open_command=np.ones((2, 2), dtype=np.float32),
    )
    annotations = source / "trajectories/annotations"
    annotations.mkdir()
    (annotations / "episode_a.segments.json").write_text(
        json.dumps(
            {
                "schema": "sim1_segments_v1",
                "review_status": "complete",
                "segments": [],
                "updated_at": "2026-08-23T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    plan = inspect_legacy_collection(source, include_reviews=False)
    assert plan.complete_trajectories == 1
    output = tmp_path / "new"
    imported = import_legacy_collection(source, output, include_reviews=False, workers=2)

    assert imported.source_snapshot_sha256 == plan.source_snapshot_sha256
    reader = TrajectoryReader(output / "trajectories/episode_a")
    assert reader.manifest.format_version == 1
    canonical = json.loads((output / "trajectories/episode_a/manifest.json").read_text())
    assert "schema_version" not in canonical
    assert "state_profile" not in canonical
    assert json.loads((output / "migration.json").read_text())["schema"] == IMPORT_SCHEMA
    assert (output / "trajectories/annotations/episode_a.segments.json").is_file()
