from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.rollout.collection import (
    _policy_endpoints_for_workers,
    _select_assets,
    _validate_policy_identities,
    load_rollout_collection,
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


def test_rollout_selects_a_fixed_seen_unseen_subset() -> None:
    assets = [
        {"asset_id": "s0", "split": "seen"},
        {"asset_id": "s1", "split": "seen"},
        {"asset_id": "u0", "split": "unseen"},
        {"asset_id": "u1", "split": "unseen"},
    ]

    selected = _select_assets(assets, seen_assets=1, unseen_assets=1)

    assert [(item["asset_id"], item["split"]) for item in selected] == [
        ("s0", "seen"),
        ("u0", "unseen"),
    ]


def test_multi_gpu_rollout_requires_one_unique_endpoint_per_worker() -> None:
    endpoints = ("ws://policy-a:8080", "ws://policy-b:8080", "ws://unused:8080")

    assert _policy_endpoints_for_workers(endpoints, 2) == endpoints[:2]
    with pytest.raises(ValueError, match="one unique policy endpoint"):
        _policy_endpoints_for_workers((endpoints[0],), 2)
    with pytest.raises(ValueError, match="requires a unique policy endpoint"):
        _policy_endpoints_for_workers((endpoints[0], endpoints[0]), 2)


def test_rtc_rollout_requires_matching_server_execution_horizon() -> None:
    identity = {
        "backend": "rlinf-sft",
        "rtc_guidance": {"execution_horizon": 10},
    }

    assert _validate_policy_identities((identity,), execution="rtc", exec_horizon=10) is identity
    with pytest.raises(ValueError, match="execution horizon differs"):
        _validate_policy_identities((identity,), execution="rtc", exec_horizon=5)
