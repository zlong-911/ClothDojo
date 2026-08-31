from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.data.retarget import _set_gripper_positions, _validate_fr3_asset
from clothdojo.data.retarget_dynamics import RobotDynamicsRetargeter
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
    assert metadata["sources"]["gripper_dynamics"]["source"] == (
        "google-deepmind/mujoco_menagerie/robotiq_2f85"
    )
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


@pytest.mark.parametrize("coordinate_count,indices", [
    (26, (7, 20)),
    (46, (7, 11, 22, 26)),
])
def test_fr3_gripper_commands_only_write_profile_driver_coordinates(coordinate_count, indices):
    coordinates = np.arange(3 * coordinate_count, dtype=np.float32).reshape(3, coordinate_count)
    before = coordinates.copy()
    indices = np.asarray(indices)
    per_side = len(indices) // 2
    openness = np.asarray(((0.0, 1.0), (0.25, 0.75), (1.0, 0.0)))
    closed = np.full(per_side, 0.8)
    opened = np.zeros(per_side)
    _set_gripper_positions(coordinates, openness, indices, closed, opened)
    np.testing.assert_allclose(
        coordinates[:, indices], 0.8 * (1.0 - openness.repeat(per_side, axis=1)),
    )
    untouched = np.setdiff1d(np.arange(coordinate_count), indices)
    np.testing.assert_array_equal(coordinates[:, untouched], before[:, untouched])

    dynamics = RobotDynamicsRetargeter.__new__(RobotDynamicsRetargeter)
    dynamics.profile = SimpleNamespace(gripper_coords_per_side=per_side)
    dynamics.gripper = indices
    dynamics.gripper_leaders = indices[::per_side]
    dynamics.gripper_span = -0.8
    dynamics.closed = closed
    dynamics.opened = opened
    actual = np.stack([dynamics._openness(q) for q in coordinates])
    np.testing.assert_allclose(actual, openness, atol=1e-7)


@pytest.mark.parametrize("robot_id", [ALOHA2, X5, UR5E_2F85])
def test_retarget_validates_runtime_usd_without_legacy_xml(tmp_path, monkeypatch, robot_id):
    from pxr import Usd

    module = importlib.import_module("clothdojo.data.retarget")
    directory = tmp_path / "robots" / robot_id
    directory.mkdir(parents=True)
    entry = directory / f"{robot_id}.usda"
    stage = Usd.Stage.CreateNew(str(entry))
    stage.DefinePrim("/Robot", "Xform")
    stage.GetRootLayer().Save()
    calls = []
    monkeypatch.setattr(module, "robot_profile", lambda robot, **kw: calls.append((robot, kw)))
    metadata = module._validate_target_asset(tmp_path, robot_id)
    assert metadata["format"] == "usd"
    assert metadata["asset"] == f"robots/{robot_id}/{robot_id}.usda"
    assert calls == [(robot_id, {"assets_root": tmp_path.resolve()})]
    assert "mjcf" not in json.dumps(metadata)
    stage.GetRootLayer().subLayerPaths = ["missing.usdc"]
    stage.GetRootLayer().Save()
    with pytest.raises(FileNotFoundError, match=r"missing\.usdc"):
        module._validate_target_asset(tmp_path, robot_id)
