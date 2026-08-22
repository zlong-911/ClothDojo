from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.rollout.initial_states import (
    INITIALIZATION_PROTOCOL,
    InitialStatePool,
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
            "initialization_protocol": INITIALIZATION_PROTOCOL,
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


def test_initial_state_validation_rejects_unsettled_and_elevated_cloth() -> None:
    q = np.zeros((3, 10, 3), dtype=np.float32)
    q[..., 2] = 0.8
    q[1, :, 2] = 1.2
    qd = np.zeros_like(q)
    qd[2, :, 0] = 1.0

    result = validate_initial_states(q, qd, table_top_m=0.8)

    np.testing.assert_array_equal(result.valid, [True, False, False])
    assert "p99_height" in result.reasons[1]
    assert "unsettled_speed" in result.reasons[2]


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
