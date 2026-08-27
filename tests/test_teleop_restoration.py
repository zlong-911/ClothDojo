from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.collection.config import (
    CollectionConfig,
    CollectionProfile,
    TeleoperationConfig,
)
from clothdojo.collection.continuation import build_continuation_plan
from clothdojo.collection.hil import HILCatalog
from clothdojo.teleop.restoration import RestorationQueue
from clothdojo.teleop.session import SessionPlan
from clothdojo.trajectory import (
    SceneMetadata,
    TrajectoryAction,
    TrajectoryState,
    TrajectoryWriter,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(index: int) -> TrajectoryState:
    return TrajectoryState(
        sample_index=index,
        joint_q=np.full(4, index, dtype=np.float32),
        joint_qd=np.zeros(4, dtype=np.float32),
        body_q=np.zeros((2, 7), dtype=np.float32),
        body_qd=np.zeros((2, 6), dtype=np.float32),
        particle_q=np.full((3, 3), index, dtype=np.float32),
        particle_qd=np.zeros((3, 3), dtype=np.float32),
        gripper_open_state=np.asarray((0.25, 0.75), dtype=np.float32),
        grasp_attached=np.zeros(2, dtype=np.int32),
        grasp_previous_open=np.ones(2, dtype=np.float32),
        grasp_pin_count=np.zeros(2, dtype=np.int32),
        grasp_pins=np.full((2, 32), -1, dtype=np.int32),
        grasp_pin_local=np.zeros((2, 32, 3), dtype=np.float32),
    )


def _write_trajectory(path, *, task: str, fingerprints: dict[str, str]) -> None:
    writer = TrajectoryWriter(
        path,
        control_fps=20.0,
        sim_substeps=6,
        max_actions=2,
        scene=SceneMetadata(
            task_name=task,
            robot_name="piper",
            asset_fingerprints=fingerprints,
        ),
        metadata={"asset_id": "shirt_a"},
    )
    writer.write_initial_state(_state(0))
    writer.append_transition(
        TrajectoryAction(
            sample_index=0,
            joint_target=np.zeros(12, dtype=np.float32),
            gripper_open_command=np.ones(2, dtype=np.float32),
        ),
        _state(1),
    )
    writer.close()


def _context(tmp_path):
    assets = tmp_path / "assets"
    garment = assets / "garments/converted/shirt_a/shirt_a.usdc"
    robot = assets / "robots/piper/piper.usda"
    garment.parent.mkdir(parents=True)
    robot.parent.mkdir(parents=True)
    garment.write_bytes(b"garment")
    robot.write_bytes(b"robot")
    fingerprints = {
        "garment_usdc": _sha256(garment),
        "piper_entry_usda": _sha256(robot),
    }
    return assets, garment, fingerprints


def _session(tmp_path, assets, garment, **updates) -> SessionPlan:
    values = {
        "session_id": "session_test",
        "mode": "collection",
        "profile_id": "shirt_flatten",
        "scene": "shirt_flatten",
        "asset_id": "shirt_a",
        "asset_path": garment,
        "assets_root": assets,
        "output_root": tmp_path / "output",
        "session_root": tmp_path / "session",
        "status_path": tmp_path / "session/status.json",
        "gpu_id": 0,
        "endpoint": "127.0.0.1:44401",
        "signaling_port": 44388,
        "media_port": 33434,
        "proxy_port": 44401,
        "max_actions": 20,
    }
    values.update(updates)
    return SessionPlan(**values)


class _Runtime:
    spec = SimpleNamespace(control_fps=20.0, sim_substeps=6)

    def __init__(self) -> None:
        self.restored = None

    def restore_state(self, state) -> None:
        self.restored = state


def test_hil_plan_cold_restores_hashed_breakpoint(tmp_path) -> None:
    assets, garment, fingerprints = _context(tmp_path)
    profile = CollectionProfile(
        id="shirt_flatten",
        display_name="flatten",
        scene="shirt_flatten",
        dataset_root=tmp_path / "dataset",
        rollout_root=tmp_path / "rollouts",
        max_actions=20,
    )
    run = profile.rollout_root / "run_a"
    trajectory = run / "trajectories/episode_a"
    _write_trajectory(trajectory, task=profile.scene, fingerprints=fingerprints)
    (run / "meta.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_rollout",
                "format_version": 1,
                "status": "complete",
                "camera_names": [],
                "trajectories": [
                    {
                        "asset_id": "shirt_a",
                        "path": "trajectories/episode_a",
                        "action_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan_value = HILCatalog(profile).intervention_plan(
        "run_a", [{"episode_id": "episode_a", "sample_index": 0}]
    )
    plan_path = tmp_path / "intervention.json"
    plan_path.write_text(json.dumps(plan_value), encoding="utf-8")
    session = _session(
        tmp_path,
        assets,
        garment,
        mode="intervention",
        intervention_plan=plan_path,
    )
    runtime = _Runtime()

    queue = RestorationQueue.load(session, runtime)
    assert queue is not None
    point = queue.restore_next(runtime)
    assert point is not None and point.source_episode_id == "episode_a"
    assert runtime.restored.sample_index == 0
    np.testing.assert_allclose(queue.current_gripper_open, (0.25, 0.75))
    assert point.recording_metadata()["source"] == "human_intervention"

    manifest = trajectory / "manifest.json"
    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        RestorationQueue.load(session, runtime)


def test_continuation_plan_uses_only_terminal_unconsumed_sources(tmp_path) -> None:
    assets, garment, fingerprints = _context(tmp_path)
    source = CollectionProfile(
        id="shirt_flatten",
        display_name="flatten",
        scene="shirt_flatten",
        dataset_root=tmp_path / "flatten",
        rollout_root=tmp_path / "rollouts",
        max_actions=20,
    )
    destination = CollectionProfile(
        id="shirt_fold",
        display_name="fold",
        scene="shirt_fold",
        dataset_root=tmp_path / "fold",
        rollout_root=tmp_path / "fold-rollouts",
        max_actions=20,
        continuation_source_profile=source.id,
    )
    _write_trajectory(
        source.trajectory_root / "episode_a",
        task=source.scene,
        fingerprints=fingerprints,
    )
    config = CollectionConfig(
        project_root=tmp_path,
        assets_root=assets,
        state_dir=tmp_path / "state",
        host="127.0.0.1",
        port=8000,
        teleoperation=TeleoperationConfig(client_host="auto", gpu_ids=(0,)),
        profiles={source.id: source, destination.id: destination},
    )

    value = build_continuation_plan(config, destination, asset_id="shirt_a")
    assert value["sources"][0]["source_sample_index"] == 1
    path = tmp_path / "continuation.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    session = _session(
        tmp_path,
        assets,
        garment,
        profile_id=destination.id,
        scene=destination.scene,
        continuation_plan=path,
    )
    runtime = _Runtime()
    queue = RestorationQueue.load(session, runtime)
    assert queue is not None
    point = queue.restore_next(runtime)
    assert point is not None and point.source_sample_index == 1
    assert runtime.restored.sample_index == 1
    retried = queue.restore_next(runtime, requeue_current=True)
    assert retried is not None and retried.source_episode_id == "episode_a"


def test_continuation_inventory_refreshes_when_assets_change(tmp_path) -> None:
    assets, garment, fingerprints = _context(tmp_path)
    source = CollectionProfile(
        id="shirt_flatten",
        display_name="flatten",
        scene="shirt_flatten",
        dataset_root=tmp_path / "flatten",
        rollout_root=tmp_path / "rollouts",
        max_actions=20,
    )
    destination = CollectionProfile(
        id="shirt_fold",
        display_name="fold",
        scene="shirt_fold",
        dataset_root=tmp_path / "fold",
        rollout_root=tmp_path / "fold-rollouts",
        max_actions=20,
        continuation_source_profile=source.id,
    )
    _write_trajectory(
        source.trajectory_root / "episode_a",
        task=source.scene,
        fingerprints=fingerprints,
    )
    config = CollectionConfig(
        project_root=tmp_path,
        assets_root=assets,
        state_dir=tmp_path / "state",
        host="127.0.0.1",
        port=8000,
        teleoperation=TeleoperationConfig(client_host="auto", gpu_ids=(0,)),
        profiles={source.id: source, destination.id: destination},
    )

    assert build_continuation_plan(config, destination, asset_id="shirt_a")

    garment.write_bytes(b"updated garment")

    with pytest.raises(ValueError, match="no uncollected continuation"):
        build_continuation_plan(config, destination, asset_id="shirt_a")
