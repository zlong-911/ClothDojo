from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.rendering import policy_observation_settings
from clothdojo.rollout.collection import load_rollout_collection, run_rollout_collection
from clothdojo.rollout.scheduling import (
    policy_endpoints_for_workers,
    select_assets,
    validate_policy_identities,
)
from clothdojo.rollout.trajectory_collection import (
    TrajectoryAssetJob,
    TrajectoryCollectionSpec,
    collect_trajectory_index,
    execution_settings,
    policy_seed_for_batch,
    trajectory_name,
)
from clothdojo.trajectory import TrajectoryAction, TrajectoryState, TrajectoryWriter


def _state(index: int) -> TrajectoryState:
    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    return TrajectoryState(
        sample_index=index,
        joint_q=zeros((12,)),
        joint_qd=zeros((12,)),
        body_q=zeros((2, 7)),
        body_qd=zeros((2, 6)),
        particle_q=zeros((4, 3)),
        particle_qd=zeros((4, 3)),
        gripper_open_state=zeros((2,)),
        grasp_attached=np.zeros((2,), dtype=np.int32),
        grasp_previous_open=zeros((2,)),
        grasp_pin_count=np.zeros((2,), dtype=np.int32),
        grasp_pins=np.zeros((2, 2), dtype=np.int32),
        grasp_pin_local=zeros((2, 2, 3)),
    )


def _trajectory(path: Path) -> None:
    writer = TrajectoryWriter(
        path,
        control_fps=50,
        sim_substeps=6,
        max_actions=1,
        metadata={
            "asset_id": "shirt",
            "initial_state": {"index": 7, "seed": 17},
        },
    )
    writer.write_initial_state(_state(0))
    writer.append_transition(
        TrajectoryAction(
            sample_index=0,
            joint_target=np.zeros((12,), dtype=np.float32),
            gripper_open_command=np.zeros((2,), dtype=np.float32),
        ),
        _state(1),
    )
    writer.close()


def test_rollout_uses_flat_teleoperation_trajectory_layout(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectories" / "shirt_000007"
    _trajectory(trajectory)
    meta = {
        "schema": "clothdojo_rollout",
        "format_version": 1,
        "status": "complete",
        "expected_trajectories": 1,
        "trajectories": [
            {
                "asset_id": "shirt",
                "initial_state_index": 7,
                "path": "trajectories/shirt_000007",
                "action_count": 1,
            }
        ],
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    loaded = load_rollout_collection(tmp_path)

    assert loaded["trajectories"][0]["path"] == "trajectories/shirt_000007"
    assert not (tmp_path / "episodes").exists()


def test_incomplete_rollout_is_not_loadable(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_rollout",
                "format_version": 1,
                "status": "failed",
                "expected_trajectories": 1,
                "trajectories": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not complete"):
        load_rollout_collection(tmp_path)


def test_failed_rollout_resume_reports_running_and_keeps_completed_index(
    tmp_path: Path, monkeypatch
) -> None:
    states = tmp_path / "states"
    pool = states / "shirt/states.npz"
    pool.parent.mkdir(parents=True)
    pool.write_bytes(b"pool")
    (states / "meta.json").write_text("{}\n", encoding="utf-8")
    scenario = SimpleNamespace(
        action_horizon=1,
        steps=1,
        to_json_dict=lambda: {"scenario_id": "test", "steps": 1},
    )
    identity = {
        "backend": "openpi",
        "checkpoint_sha256": "a" * 64,
        "batched_inference": True,
    }
    monkeypatch.setattr(
        "clothdojo.rollout.collection.load_initial_state_collection",
        lambda _root: {"assets": [{"asset_id": "shirt", "split": "unseen"}]},
    )
    monkeypatch.setattr(
        "clothdojo.rollout.collection.InitialStatePool.load",
        lambda _path: SimpleNamespace(state_count=1),
    )
    monkeypatch.setattr(
        "clothdojo.rollout.collection.load_rollout_scenario",
        lambda _path: scenario,
    )
    monkeypatch.setattr(
        "clothdojo.rollout.collection.RemotePolicy",
        lambda *_args, **_kwargs: SimpleNamespace(identity=identity),
    )
    indexed = [
        {
            "asset_id": "shirt",
            "initial_state_index": 0,
            "path": "trajectories/shirt_000000",
            "action_count": 1,
        }
    ]
    failed = True

    def collect_index(_jobs, *, spec, partial=False):
        del spec
        return [] if partial and failed else indexed

    def fail_collection(*_args, **_kwargs):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(
        "clothdojo.rollout.collection.collect_trajectory_index",
        collect_index,
    )
    monkeypatch.setattr(
        "clothdojo.rollout.collection.collect_trajectories",
        fail_collection,
    )
    output = tmp_path / "rollout"
    arguments = (
        tmp_path / "scenario.yaml",
        states,
        ("ws://policy:8080",),
        output,
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        run_rollout_collection(
            *arguments,
            gpus=("0",),
            batch_size=1,
            exec_horizon=1,
        )
    assert json.loads((output / "meta.json").read_text())["status"] == "failed"

    failed = False

    def resume_collection(*_args, **_kwargs):
        meta = json.loads((output / "meta.json").read_text())
        assert meta["status"] == "running"
        assert "error" not in meta
        assert meta["trajectories"] == indexed

    monkeypatch.setattr(
        "clothdojo.rollout.collection.collect_trajectories",
        resume_collection,
    )
    result = run_rollout_collection(
        *arguments,
        gpus=("0",),
        batch_size=1,
        exec_horizon=1,
    )

    assert result["trajectory_count"] == 1
    assert json.loads((output / "meta.json").read_text())["status"] == "complete"


def test_rollout_selects_a_fixed_seen_unseen_subset() -> None:
    assets = [
        {"asset_id": "s0", "split": "seen"},
        {"asset_id": "s1", "split": "seen"},
        {"asset_id": "u0", "split": "unseen"},
        {"asset_id": "u1", "split": "unseen"},
    ]

    selected = select_assets(assets, seen_assets=1, unseen_assets=1)

    assert [(item["asset_id"], item["split"]) for item in selected] == [
        ("s0", "seen"),
        ("u0", "unseen"),
    ]


def test_multi_gpu_rollout_requires_one_unique_endpoint_per_worker() -> None:
    endpoints = ("ws://policy-a:8080", "ws://policy-b:8080", "ws://unused:8080")

    assert policy_endpoints_for_workers(endpoints, 2) == endpoints[:2]
    with pytest.raises(ValueError, match="one unique policy endpoint"):
        policy_endpoints_for_workers((endpoints[0],), 2)
    with pytest.raises(ValueError, match="requires a unique policy endpoint"):
        policy_endpoints_for_workers((endpoints[0], endpoints[0]), 2)


def test_rtc_rollout_accepts_independent_server_guidance_horizon() -> None:
    identity = {
        "backend": "rlinf-sft",
        "rtc_guidance": {"guidance_horizon": 5},
    }

    assert validate_policy_identities((identity,), execution="rtc") is identity


def test_rtc_rollout_accepts_capable_openpi_server() -> None:
    identity = {
        "backend": "openpi",
        "rtc_guidance": {"guidance_horizon": 10},
    }

    assert validate_policy_identities((identity,), execution="rtc") is identity


def test_rtc_rollout_rejects_openpi_without_rtc_capability() -> None:
    with pytest.raises(ValueError, match="RTC-capable policy server"):
        validate_policy_identities(
            ({"backend": "openpi"},),
            execution="rtc",
        )


def test_parallel_rollout_requires_server_batch_capability() -> None:
    identity = {"backend": "openpi", "batched_inference": True}

    assert (
        validate_policy_identities(
            (identity,),
            execution="chunk",
            worlds_per_batch=8,
        )
        is identity
    )
    with pytest.raises(ValueError, match="batched-inference"):
        validate_policy_identities(
            ({"backend": "openpi"},),
            execution="chunk",
            worlds_per_batch=2,
        )


def test_policy_batch_seed_is_stable_and_preserves_single_world_seed() -> None:
    assert policy_seed_for_batch(np.asarray([17], dtype=np.int32)) == 17
    first = policy_seed_for_batch(np.asarray([17, 23, 41], dtype=np.int32))
    assert first == policy_seed_for_batch(np.asarray([17, 23, 41], dtype=np.int32))
    assert first != policy_seed_for_batch(np.asarray([41, 23, 17], dtype=np.int32))


def test_shared_runner_uses_one_layout_for_rollout_and_evaluation_cases() -> None:
    assert trajectory_name("shirt", 7, None) == "shirt_000007"
    assert trajectory_name("shirt", 7, 2) == "shirt_000007_r002"
    seeds = np.asarray([17, 23], dtype=np.int32)
    assert policy_seed_for_batch(seeds, 0) != policy_seed_for_batch(seeds, 1)

    with pytest.raises(ValueError, match="reserved keys"):
        TrajectoryAssetJob(
            asset_id="shirt",
            pool="states.npz",
            pool_sha256="a" * 64,
            state_count=1,
            metadata={"policy": {}},
        )


def test_shared_runner_indexes_evaluation_provenance(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        """schema: clothdojo_rollout_scenario
format_version: 1
scenario_id: test
task_name: shirt_flatten
seed: 0
prompt: flatten the shirt
steps: 1
action_horizon: 1
""",
        encoding="utf-8",
    )
    identity = {
        "backend": "openpi",
        "checkpoint_sha256": "b" * 64,
        "batched_inference": True,
    }
    extra = {
        "condition_id": "condition",
        "protocol_fingerprint": "c" * 64,
        "template_sha256": "d" * 64,
    }
    root = tmp_path / "trajectories"
    output = root / "shirt_000000_r000"
    writer = TrajectoryWriter(
        output,
        control_fps=50,
        sim_substeps=6,
        max_actions=1,
        metadata={
            "producer": "clothdojo_evaluation",
            "training_eligible": False,
            "asset_id": "shirt",
            "policy_kind": "openpi",
            "policy": identity,
            "policy_seed": 123,
            "cloth_visual": "nocs",
            "observation": policy_observation_settings(),
            "execution": execution_settings("chunk", 1),
            "parallelism": {"worlds": 1, "active_worlds": 1, "batch_start": 0},
            "initial_state": {
                "pool_sha256": "a" * 64,
                "index": 0,
                "seed": 17,
            },
            "replicate": 0,
            **extra,
        },
    )
    writer.write_initial_state(_state(0))
    writer.append_transition(
        TrajectoryAction(
            sample_index=0,
            joint_target=np.zeros((12,), dtype=np.float32),
            gripper_open_command=np.zeros((2,), dtype=np.float32),
        ),
        _state(1),
    )
    writer.close()
    job = TrajectoryAssetJob(
        asset_id="shirt",
        pool="states.npz",
        pool_sha256="a" * 64,
        state_count=1,
        replicates=(0,),
        metadata=extra,
    )
    spec = TrajectoryCollectionSpec(
        scenario_path=str(scenario),
        assets_root=str(tmp_path / "assets"),
        trajectories_root=str(root),
        expected_checkpoint_sha256=None,
        policy_identity=identity,
        execution="chunk",
        exec_horizon=1,
        rtc_delay_steps=0,
        horizon_steps=1,
        batch_size=1,
        producer="clothdojo_evaluation",
        training_eligible=False,
    )

    index = collect_trajectory_index([job], spec=spec)

    assert index == [
        {
            "asset_id": "shirt",
            "initial_state_index": 0,
            "seed": 17,
            "policy_seed": 123,
            "path": "trajectories/shirt_000000_r000",
            "action_count": 1,
            "execution_report": {},
            "replicate": 0,
        }
    ]
