from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.data.retarget import _set_gripper_positions, _validate_fr3_asset
from clothdojo.data.retarget_dynamics import AlohaDynamicsRetargeter
from clothdojo.robots import ALOHA2, FR3_DUO, UR5E_2F85, X5


@pytest.fixture
def fr3_assets(tmp_path, monkeypatch):
    from pxr import Usd

    root = tmp_path / "assets"
    directory = root / "robots/fr3_duo"
    (directory / "payloads").mkdir(parents=True)
    (directory / "provenance").mkdir()
    payload = Usd.Stage.CreateNew(str(directory / "payloads/workstation.usdc"))
    payload.DefinePrim("/Robot", "Xform")
    payload.GetRootLayer().Save()
    entry = Usd.Stage.CreateNew(str(directory / "fr3_duo.usda"))
    entry.GetRootLayer().subLayerPaths = ["./payloads/workstation.usdc"]
    entry.GetRootLayer().Save()
    provenance = directory / "provenance/dynamics.json"
    provenance.write_text(json.dumps({
        "franka_revision": "source-franka-revision",
        "robotiq_revision": "source-robotiq-revision",
    }))

    calls = []

    def read_profile(robot_id, *, assets_root):
        calls.append((robot_id, assets_root))
        return SimpleNamespace(robot_id=robot_id)

    monkeypatch.setattr(
        importlib.import_module("clothdojo.data.retarget"), "robot_profile", read_profile
    )
    return root, directory, calls


def test_fr3_records_usd_and_real_provenance_without_mjcf_or_pins(fr3_assets):
    root, _directory, calls = fr3_assets
    metadata = _validate_fr3_asset(root)
    assert metadata["format"] == "usd"
    assert metadata["asset"] == "robots/fr3_duo/fr3_duo.usda"
    assert metadata["payload"] == "robots/fr3_duo/payloads/workstation.usdc"
    assert metadata["sources"]["arm_dynamics"]["revision"] == "source-franka-revision"
    assert metadata["sources"]["gripper_dynamics"]["revision"] == "source-robotiq-revision"
    assert "mjcf" not in json.dumps(metadata)
    assert "sha256" not in json.dumps(metadata)
    assert calls == [(FR3_DUO, root.resolve())]


@pytest.mark.parametrize("missing", ["payloads/workstation.usdc", "provenance/dynamics.json"])
def test_fr3_requires_payload_and_provenance(fr3_assets, missing):
    root, directory, _calls = fr3_assets
    (directory / missing).unlink()
    with pytest.raises(FileNotFoundError):
        _validate_fr3_asset(root)


def test_fr3_checks_transitive_usd_dependencies(fr3_assets):
    from pxr import Sdf

    root, directory, _calls = fr3_assets
    layer = Sdf.Layer.FindOrOpen(str(directory / "payloads/workstation.usdc"))
    layer.subLayerPaths = ["./missing-component.usdc"]
    layer.Save()
    with pytest.raises(FileNotFoundError, match="missing-component"):
        _validate_fr3_asset(root)


def test_fr3_reuses_runtime_semantic_validation(fr3_assets, monkeypatch):
    root, _directory, _calls = fr3_assets

    def reject_profile(*_args, **_kwargs):
        raise RuntimeError("missing workstation home semantics")

    monkeypatch.setattr(
        importlib.import_module("clothdojo.data.retarget"), "robot_profile", reject_profile
    )
    with pytest.raises(RuntimeError, match="home semantics"):
        _validate_fr3_asset(root)


@pytest.mark.parametrize("sources", [[], {}, {"franka_revision": "f", "robotiq_revision": ""}])
def test_fr3_rejects_unusable_dynamics_provenance(fr3_assets, sources):
    root, directory, _calls = fr3_assets
    (directory / "provenance/dynamics.json").write_text(json.dumps(sources))
    with pytest.raises(ValueError, match="provenance"):
        _validate_fr3_asset(root)


def test_fr3_gripper_commands_only_write_two_controlled_coordinates():
    coordinates = np.arange(3 * 26, dtype=np.float32).reshape(3, 26)
    before = coordinates.copy()
    indices = np.asarray((7, 20))
    openness = np.asarray(((0.0, 1.0), (0.25, 0.75), (1.0, 0.0)))
    closed = np.asarray((0.8,))
    opened = np.asarray((0.0,))
    _set_gripper_positions(coordinates, openness, indices, closed, opened)
    np.testing.assert_allclose(coordinates[:, indices], 0.8 * (1.0 - openness))
    untouched = np.setdiff1d(np.arange(26), indices)
    np.testing.assert_array_equal(coordinates[:, untouched], before[:, untouched])

    dynamics = AlohaDynamicsRetargeter.__new__(AlohaDynamicsRetargeter)
    dynamics.profile = SimpleNamespace(gripper_coords_per_side=1)
    dynamics.gripper = indices
    dynamics.gripper_leaders = indices
    dynamics.gripper_span = -0.8
    dynamics.closed = closed
    dynamics.opened = opened
    actual = np.stack([dynamics._openness(q) for q in coordinates])
    np.testing.assert_allclose(actual, openness, atol=1e-7)


def test_fr3_authored_layout_matches_retarget_controlled_coordinates():
    from pxr import Usd

    asset = Path(__file__).resolve().parents[1] / "assets/robots/fr3_duo/fr3_duo.usda"
    stage = Usd.Stage.Open(str(asset))
    root = stage.GetDefaultPrim()
    assert root.GetAttribute("clothdojo:robotId").Get() == FR3_DUO
    assert len(root.GetAttribute("clothdojo:homeJointQ").Get()) == 26
    assert len(root.GetAttribute("clothdojo:armCoordIndices").Get()) == 14
    assert tuple(root.GetAttribute("clothdojo:gripperCoordIndices").Get()) == (7, 20)
    assert tuple(root.GetAttribute("clothdojo:gripperDofIndices").Get()) == (7, 20)
    assert len(root.GetAttribute("clothdojo:gripperClosedPositions").Get()) == 1
    assert len(root.GetAttribute("clothdojo:gripperOpenPositions").Get()) == 1


def test_only_legacy_mjcf_targets_keep_mjcf_pins():
    module = importlib.import_module("clothdojo.data.retarget")
    assert set(module._TARGET_ASSET_PINS) == {ALOHA2, UR5E_2F85, X5}
    assert set(module._TARGET_SOURCE_ASSETS) == {ALOHA2, UR5E_2F85, X5}
