import json
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.evaluation.metric import CanonicalTemplate
from clothdojo.training.rlinf_steam_state import _state_from_keys
from clothdojo.training.steam_advantages import _evaluator_state_keys
from clothdojo.training.steam_geometry import frame_geometry


def test_mirror_geometry_uses_corresponding_target():
    positions = np.array(
        [[0.1, -0.4, 1.1], [0.5, 0.1, 1.1], [0.7, 0.3, 1.2], [0.2, 0.2, 1.1]], dtype=np.float32
    )
    template = CanonicalTemplate(
        "asset", *(["a" * 64] * 3), "default_free_settle", "b" * 64, 1.0, positions
    )
    current = positions[None].copy()
    current[0, 1, 2] += 0.1
    mirrored = current.copy()
    mirrored[..., 1] *= -1
    kwargs = {"symmetry": "planar_90", "center_xy": [0.65, 0]}
    reference = frame_geometry(current, template, **kwargs)
    actual = frame_geometry(
        mirrored,
        template,
        augmentation={"kind": "bilateral_mirror", "mirror_plane": {"axis": "y", "offset_m": 0.0}},
        **kwargs,
    )
    np.testing.assert_array_equal(actual, reference)
    assert frame_geometry(mirrored, template, **kwargs)[0, 0] > reference[0, 0]


def test_state_features_are_ordered_and_missing_geometry_fails():
    state = {"observation.state": np.arange(14), "observation.cloth_metrics": [0.4, 0.2]}
    keys = ["observation.state", "observation.cloth_metrics"]
    result = _state_from_keys(state, keys)
    assert result.shape == (16,)
    np.testing.assert_allclose(result[-2:], [0.4, 0.2])
    with pytest.raises(KeyError):
        _state_from_keys({"observation.state": state[keys[0]]}, keys)


def test_geometry_checkpoint_rejects_different_inference_semantics(tmp_path):
    keys = ["observation.state", "observation.cloth_metrics"]
    contract = {"version": 1, "symmetry": "planar_90"}
    model = SimpleNamespace(
        config=SimpleNamespace(state_keys=keys, geometry_contract=json.dumps(contract))
    )
    (tmp_path / "meta").mkdir()
    sidecar = tmp_path / "meta/clothdojo.json"
    sidecar.write_text(json.dumps({"geometry": {"contract": contract}}))
    assert _evaluator_state_keys(model, tmp_path) == keys
    sidecar.write_text(json.dumps({"geometry": {"contract": {**contract, "symmetry": "none"}}}))
    with pytest.raises(ValueError, match="geometry features"):
        _evaluator_state_keys(model, tmp_path)


def test_old_state_only_checkpoint_keeps_its_input_contract(tmp_path):
    model = SimpleNamespace(config=SimpleNamespace())
    assert _evaluator_state_keys(model, tmp_path) == ["observation.state"]


def test_valid_parquet_never_enters_tar_probe(tmp_path, monkeypatch):
    import tarfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets.utils.extract import Extractor

    from clothdojo.training.steam_dataloader import install_parquet_archive_guard

    path = tmp_path / "episode.parquet"
    pq.write_table(pa.table({"frame_index": [0, 1]}), path)
    install_parquet_archive_guard()
    monkeypatch.setattr(
        tarfile, "is_tarfile", lambda path: pytest.fail("Parquet reached TAR probe")
    )
    assert Extractor.infer_extractor_format(path) is None
    assert pq.read_table(path).num_rows == 2
