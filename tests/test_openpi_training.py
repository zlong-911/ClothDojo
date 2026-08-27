from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from clothdojo.training.openpi import _dataset_prompt, _episode_ends, _float_matrix


def test_float_matrix_keeps_fixed_size_list_layout() -> None:
    values = pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float32())
    column = pa.chunked_array([pa.FixedSizeListArray.from_arrays(values, 2)])

    matrix = _float_matrix(column, pa)

    np.testing.assert_array_equal(matrix, [[1.0, 2.0], [3.0, 4.0]])
    assert matrix.dtype == np.float32
    assert matrix.flags.c_contiguous


def test_episode_ends_are_frame_aligned() -> None:
    ends, count = _episode_ends(np.array([0, 0, 0, 1, 1, 2], dtype=np.int64))

    np.testing.assert_array_equal(ends, [3, 3, 3, 5, 5, 6])
    assert count == 3


def test_episode_ends_reject_non_contiguous_labels() -> None:
    with pytest.raises(ValueError, match="contiguous and zero-based"):
        _episode_ends(np.array([0, 0, 2], dtype=np.int64))


def test_norm_stats_reads_task_prompt_from_dataset_sidecar(tmp_path) -> None:
    sidecar = tmp_path / "clothdojo.json"
    sidecar.write_text('{"scenario":{"prompt":"Fold the shirt."}}', encoding="utf-8")

    assert _dataset_prompt(sidecar) == "Fold the shirt."

    sidecar.write_text('{"scenario":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="no task prompt"):
        _dataset_prompt(sidecar)
