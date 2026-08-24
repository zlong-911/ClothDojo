from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.rollout.initial_states import (
    INITIALIZATION_METHOD,
    InitialStatePool,
    collect_seen_garments,
    generate_initial_state_collection,
    load_initial_state_collection,
    select_garments,
    validate_initial_states,
    validate_pool_compatibility,
)


def _pool() -> InitialStatePool:
    return InitialStatePool(
        particle_q=np.zeros((2, 4, 3), dtype=np.float32),
        particle_qd=np.zeros((2, 4, 3), dtype=np.float32),
        seeds=np.asarray([11, 22], dtype=np.int32),
        metadata={
            "garment_id": "shirt",
            "garment_sha256": "placeholder",
            "initialization": INITIALIZATION_METHOD,
        },
    )


def test_initial_state_pool_round_trip_is_strict_and_atomic(tmp_path: Path) -> None:
    output = tmp_path / "shirt_states.npz"
    pool = _pool()

    pool.save(output)
    loaded = InitialStatePool.load(output)

    np.testing.assert_array_equal(loaded.particle_q, pool.particle_q)
    np.testing.assert_array_equal(loaded.particle_qd, pool.particle_qd)
    np.testing.assert_array_equal(loaded.seeds, pool.seeds)
    assert loaded.metadata == pool.metadata
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        pool.save(output)


def test_initial_state_selection_preserves_requested_order() -> None:
    pool = _pool()
    q, qd, seeds = pool.select(np.asarray([1, 0]))

    assert q.shape == qd.shape == (2, 4, 3)
    np.testing.assert_array_equal(seeds, [22, 11])


def test_initial_state_validation_rejects_unsettled_elevated_and_suspended_cloth() -> None:
    q = np.zeros((4, 10, 3), dtype=np.float32)
    q[..., 2] = 0.8
    q[1, :, 2] = 1.2
    q[3, :, 2] = 0.9
    qd = np.zeros_like(q)
    qd[2, :, 0] = 1.0

    result = validate_initial_states(q, qd, table_top_m=0.8)

    np.testing.assert_array_equal(result.valid, [True, False, False, False])
    assert "p99_height" in result.reasons[1]
    assert "unsettled_speed" in result.reasons[2]
    assert "no_table_contact" in result.reasons[3]
    assert result.near_table_fraction[0] == 1.0


def test_initial_state_validation_requires_table_contact_and_rejects_robot_contact() -> None:
    q = np.zeros((3, 10, 3), dtype=np.float32)
    q[..., 2] = 0.8
    qd = np.zeros_like(q)

    result = validate_initial_states(
        q,
        qd,
        table_top_m=0.8,
        table_contact_count=np.asarray([5, 0, 5], dtype=np.int32),
        robot_contact_count=np.asarray([0, 0, 1], dtype=np.int32),
    )

    np.testing.assert_array_equal(result.valid, [True, False, False])
    assert "no_physical_table_contact" in result.reasons[1]
    assert "robot_contact" in result.reasons[2]


def test_initial_state_pool_checks_asset_identity(tmp_path: Path) -> None:
    garment = tmp_path / "shirt.usdc"
    garment.write_bytes(b"garment")
    pool = _pool()
    metadata = dict(pool.metadata)
    metadata["garment_sha256"] = hashlib.sha256(garment.read_bytes()).hexdigest()
    compatible = InitialStatePool(pool.particle_q, pool.particle_qd, pool.seeds, metadata)

    validate_pool_compatibility(compatible, garment_path=garment, particle_count=4)
    garment.write_bytes(b"changed")
    with pytest.raises(ValueError, match="different garment"):
        validate_pool_compatibility(compatible, garment_path=garment, particle_count=4)


def test_initial_state_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    output = tmp_path / "bad.npz"
    metadata = {
        "schema": "other",
        "format_version": 1,
        "state_count": 1,
        "particle_count": 1,
    }
    np.savez(
        output,
        particle_q=np.zeros((1, 1, 3), dtype=np.float32),
        particle_qd=np.zeros((1, 1, 3), dtype=np.float32),
        seeds=np.zeros(1, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    with pytest.raises(ValueError, match="schema"):
        InitialStatePool.load(output)


def test_seen_assets_require_complete_trajectory_with_explicit_asset_id(
    tmp_path: Path,
) -> None:
    for name, status, asset_id in (
        ("good", "complete", "shirt_seen"),
        ("recording", "recording", "shirt_recording"),
        ("legacy", "complete", None),
    ):
        path = tmp_path / "trajectories" / name
        path.mkdir(parents=True)
        metadata = {} if asset_id is None else {"asset_id": asset_id}
        (path / "manifest.json").write_text(
            json.dumps({"status": status, "metadata": metadata}), encoding="utf-8"
        )

    assert collect_seen_garments(tmp_path) == {"shirt_seen"}


def test_seen_assets_can_come_from_project_lerobot_metadata(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "clothdojo.json").write_text(
        json.dumps({"assets": ["shirt_a", "shirt_b"]}), encoding="utf-8"
    )

    assert collect_seen_garments(tmp_path) == {"shirt_a", "shirt_b"}


def test_asset_selection_is_seeded_and_separates_seen_from_unseen() -> None:
    collectable = ["seen_a", "seen_b", "new_a", "new_b"]

    first = select_garments(
        collectable,
        {"seen_a", "seen_b"},
        seen_count=1,
        unseen_count=1,
        seed=7,
    )
    second = select_garments(
        collectable,
        {"seen_a", "seen_b"},
        seen_count=1,
        unseen_count=1,
        seed=7,
    )

    assert first == second
    assert [item["split"] for item in first] == ["seen", "unseen"]
    assert len({item["generator_seed"] for item in first}) == 2


def test_initial_state_collection_is_one_pool_per_asset(tmp_path: Path) -> None:
    root = tmp_path / "states"
    pool_path = root / "shirt" / "states.npz"
    pool = _pool()
    metadata = dict(pool.metadata)
    metadata["garment_id"] = "shirt"
    InitialStatePool(pool.particle_q, pool.particle_qd, pool.seeds, metadata).save(pool_path)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_initial_states",
                "format_version": 1,
                "status": "complete",
                "states_per_asset": 2,
                "assets": [{"asset_id": "shirt", "path": "shirt/states.npz"}],
            }
        ),
        encoding="utf-8",
    )

    meta = load_initial_state_collection(root)

    assert meta["assets"][0]["path"] == "shirt/states.npz"


def test_initial_state_collection_freezes_selection_and_resumes(
    monkeypatch, tmp_path: Path
) -> None:
    expert = tmp_path / "expert" / "trajectories" / "episode"
    expert.mkdir(parents=True)
    (expert / "manifest.json").write_text(
        json.dumps(
            {"status": "complete", "metadata": {"asset_id": "shirt_seen"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "clothdojo.assets.GarmentLibrary.collectable_ids",
        lambda _: ("shirt_seen", "shirt_unseen"),
    )
    calls = []

    def generate(jobs, **kwargs):
        calls.append(tuple(item["asset_id"] for item in jobs))
        for item in jobs:
            pool = _pool()
            metadata = dict(pool.metadata)
            metadata["garment_id"] = item["asset_id"]
            InitialStatePool(pool.particle_q, pool.particle_qd, pool.seeds, metadata).save(
                kwargs["output"] / item["asset_id"] / "states.npz"
            )

    monkeypatch.setattr("clothdojo.rollout.initial_states._run_state_workers", generate)
    output = tmp_path / "states"
    arguments = {
        "expert_trajectories": tmp_path / "expert",
        "seen_assets": 1,
        "unseen_assets": 1,
        "states_per_asset": 2,
        "gpus": ("0", "1"),
        "candidate_batch": 2,
        "max_candidates": 4,
        "seed": 7,
        "drop_seconds": 1.0,
    }

    first = generate_initial_state_collection(tmp_path / "assets", output, **arguments)
    second = generate_initial_state_collection(tmp_path / "assets", output, **arguments)

    assert calls == [("shirt_seen", "shirt_unseen")]
    assert first["state_count"] == second["state_count"] == 4
    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "complete"
    assert {item["split"] for item in meta["assets"]} == {"seen", "unseen"}
