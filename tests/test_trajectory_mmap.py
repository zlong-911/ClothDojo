from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from test_trajectory import _write_trajectory

from clothdojo.trajectory import TrajectoryReader
from clothdojo.trajectory.reader import _stored_array_view


@pytest.mark.parametrize("compressed", [False, True])
def test_sampled_state_mapping_matches_regular_reader(tmp_path: Path, compressed: bool):
    path = _write_trajectory(tmp_path / "trajectory")
    archive = path / "states.npz"
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    writer = np.savez_compressed if compressed else np.savez
    writer(archive, **arrays)
    reader = TrajectoryReader(path)
    regular = reader.state_arrays(("particle_q", "particle_qd"))
    mapped = reader.state_arrays(("particle_q", "particle_qd"), mmap=True)
    for key in regular:
        np.testing.assert_array_equal(mapped[key], regular[key])
        np.testing.assert_array_equal(mapped[key][[0, 2, 3]], regular[key][[0, 2, 3]])
        assert not mapped[key].flags.writeable
        assert isinstance(mapped[key], np.memmap) is (not compressed)


@pytest.mark.parametrize("fortran", [False, True])
def test_mapping_handles_zip64_and_memory_order(tmp_path: Path, fortran: bool):
    path = tmp_path / "arrays.npz"
    values = np.arange(36, dtype=np.float32).reshape(4, 3, 3)
    if fortran:
        values = np.asfortranarray(values)
    np.savez(path, particle_q=values, empty=np.empty((4, 0), dtype=np.float32))
    with zipfile.ZipFile(path) as archive:
        np.testing.assert_array_equal(_stored_array_view(path, archive, "particle_q"), values)
        assert _stored_array_view(path, archive, "empty").shape == (4, 0)


def test_mapping_rejects_object_arrays(tmp_path: Path):
    path = tmp_path / "arrays.npz"
    np.savez(path, particle_q=np.asarray([object()], dtype=object))
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="object"):
        _stored_array_view(path, archive, "particle_q")


def test_mapping_retains_manifest_validation(tmp_path: Path):
    path = _write_trajectory(tmp_path / "trajectory")
    archive = path / "states.npz"
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays["particle_q"] = arrays["particle_q"].astype(np.float64)
    np.savez(archive, **arrays)
    with pytest.raises(RuntimeError, match="invalid dtypes"):
        TrajectoryReader(path).state_arrays(("particle_q",), mmap=True)


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("damage", ["shape", "count", "indices", "missing", "extra"])
def test_mapping_retains_all_selected_array_contracts(tmp_path: Path, mapped: bool, damage: str):
    path = _write_trajectory(tmp_path / "trajectory")
    archive = path / "states.npz"
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    if damage == "shape":
        arrays["particle_q"] = arrays["particle_q"][:, :, :2]
    elif damage == "count":
        arrays["particle_q"] = arrays["particle_q"][:-1]
    elif damage == "indices":
        arrays["sample_index"][1] = 0
    elif damage == "missing":
        arrays.pop("particle_qd")
    else:
        arrays["unexpected"] = np.zeros(1)
    np.savez(archive, **arrays)
    with pytest.raises(RuntimeError):
        TrajectoryReader(path).state_arrays(("particle_q",), mmap=mapped)


def test_mapping_does_not_materialize_stored_fields(tmp_path: Path, monkeypatch):
    path = _write_trajectory(tmp_path / "trajectory")
    archive = path / "states.npz"
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
        archive_type = type(stored)
    np.savez(archive, **arrays)

    def reject_materialization(self, key):
        raise AssertionError(f"uncompressed {key} must not be materialized")

    monkeypatch.setattr(archive_type, "__getitem__", reject_materialization)
    selected = TrajectoryReader(path).state_arrays(("particle_q",), mmap=True)
    assert set(selected) == {"sample_index", "particle_q"}
    np.testing.assert_array_equal(selected["particle_q"][2], arrays["particle_q"][2])


def test_mapping_supports_mixed_compression(tmp_path: Path):
    path = _write_trajectory(tmp_path / "trajectory")
    archive = path / "states.npz"
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    with zipfile.ZipFile(archive, "w") as target:
        for key, value in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            compression = zipfile.ZIP_DEFLATED if key == "particle_qd" else zipfile.ZIP_STORED
            target.writestr(f"{key}.npy", buffer.getvalue(), compress_type=compression)
    selected = TrajectoryReader(path).state_arrays(("particle_q", "particle_qd"), mmap=True)
    assert isinstance(selected["particle_q"], np.memmap)
    assert not isinstance(selected["particle_qd"], np.memmap)
    for key, value in selected.items():
        np.testing.assert_array_equal(value, arrays[key])
        assert not value.flags.writeable


def test_mapping_rejects_short_npy_payload(tmp_path: Path):
    path = tmp_path / "arrays.npz"
    buffer = io.BytesIO()
    np.save(buffer, np.arange(36, dtype=np.float32).reshape(4, 3, 3), allow_pickle=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("particle_q.npy", buffer.getvalue()[:-4])
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="payload size"):
        _stored_array_view(path, archive, "particle_q")


def test_mapping_rejects_inconsistent_stored_member_sizes(tmp_path: Path):
    path = tmp_path / "arrays.npz"
    # Exceed ZipExtFile's 4 KiB read-ahead so header parsing does not implicitly
    # consume the payload and catch this only by CRC at EOF.
    np.savez(path, particle_q=np.arange(3000, dtype=np.float32).reshape(1000, 3))
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("particle_q.npy")
        info.compress_size -= 4
        with pytest.raises(ValueError, match=r"size|stored"):
            _stored_array_view(path, archive, "particle_q")


def test_offline_scoring_is_identical_for_stored_and_compressed_states(tmp_path: Path):
    from test_evaluation import _offline_evaluation

    from clothdojo.evaluation.metric import load_canonical_template
    from clothdojo.evaluation.protocol import load_evaluation_protocol
    from clothdojo.evaluation.scoring import score_evaluation_trajectory

    condition, protocol_path, templates, compressed_score = _offline_evaluation(
        tmp_path, quarter_turn=True
    )
    trajectory = condition / "trajectories/shirt_000003_r002"
    path = trajectory / "states.npz"
    with zipfile.ZipFile(path) as archive:
        assert archive.getinfo("particle_q.npy").compress_type == zipfile.ZIP_DEFLATED
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    np.savez(path, **arrays)
    mapped_score = score_evaluation_trajectory(
        trajectory,
        load_canonical_template(templates / "templates/shirt.npz"),
        load_evaluation_protocol(protocol_path),
    )
    np.testing.assert_array_equal(mapped_score.sample_steps, compressed_score.sample_steps)
    np.testing.assert_array_equal(mapped_score.curve, compressed_score.curve)
    assert mapped_score.result == compressed_score.result
