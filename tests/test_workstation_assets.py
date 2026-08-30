import pytest

from clothdojo.assets.workstations import (
    WORKSTATION_IDS,
    inspect_workstation_assets,
    prepare_workstation_assets,
)


@pytest.fixture
def workstation_assets(tmp_path):
    pytest.importorskip("pxr.UsdUtils")
    root = tmp_path / "source"
    common = root / "robots/common"
    common.mkdir(parents=True)
    (common / "camera.usda").write_text('#usda 1.0\ndef Xform "Camera" {}\n')
    for robot_id in WORKSTATION_IDS:
        directory = root / "robots" / robot_id
        directory.mkdir()
        (directory / f"{robot_id}.usda").write_text(
            "#usda 1.0\n(\nsubLayers = [@../common/camera.usda@]\n)\n"
            'def Xform "Robot" {\n'
            "asset texture = @./texture.png@\n"
            "asset shader = @OmniPBR.mdl@\n}\n"
        )
        (directory / "texture.png").write_bytes(b"source texture")
        (directory / "source.stl").write_bytes(b"source mesh")
    return root


def test_install_roundtrip_and_idempotence(workstation_assets, tmp_path):
    root = workstation_assets
    cache = root / "robots/x5/__pycache__"
    cache.mkdir()
    (cache / "build.pyc").write_bytes(b"not an asset")
    old = root / "robots/panda"
    old.mkdir()
    (old / "panda.usda").write_text("deprecated")
    target = tmp_path / "destination"
    report = prepare_workstation_assets(root, target)
    assert report["copied_files"] == 16
    assert report["runtime_files"] == 11
    assert set(report["workstations"]) == set(WORKSTATION_IDS)
    assert report["workstations"]["x5"]["renderer_modules"] == ["OmniPBR.mdl"]
    assert (target / "robots/x5/source.stl").read_bytes() == b"source mesh"
    assert not (target / "robots/x5/__pycache__").exists()
    assert not (target / "robots/panda").exists()
    assert prepare_workstation_assets(root, target)["copied_files"] == 0
    assert prepare_workstation_assets(root, target)["reused_files"] == 16
    assert inspect_workstation_assets(target)["runtime_files"] == 11


def test_conflicting_tracked_entry_does_not_copy_anything(workstation_assets, tmp_path):
    target = tmp_path / "destination"
    directory = target / "robots/x5"
    directory.mkdir(parents=True)
    entry = directory / "x5.usda"
    entry.write_text("existing checkout entry")
    with pytest.raises(FileExistsError, match="matching this checkout"):
        prepare_workstation_assets(workstation_assets, target)
    assert entry.read_text() == "existing checkout entry"
    assert not (target / "robots/piper").exists()


@pytest.mark.parametrize(
    "reference", ["/tmp/external.usda", "https://host/robot.usda", "../../../outside.usda"]
)
def test_rejects_nonportable_dependencies(workstation_assets, reference):
    entry = workstation_assets / "robots/x5/x5.usda"
    entry.write_text(f"#usda 1.0\n(\nsubLayers = [@{reference}@]\n)\n")
    with pytest.raises(ValueError, match=r"portable|escapes"):
        inspect_workstation_assets(workstation_assets)


def test_missing_texture_is_not_mistaken_for_a_renderer_module(workstation_assets):
    (workstation_assets / "robots/ur5e_2f85/texture.png").unlink()
    with pytest.raises(FileNotFoundError, match=r"texture\.png"):
        inspect_workstation_assets(workstation_assets)


def test_unknown_mdl_is_not_whitelisted(workstation_assets):
    entry = workstation_assets / "robots/x5/x5.usda"
    entry.write_text('#usda 1.0\ndef Shader "M" {\nasset source = @custom.mdl@\n}\n')
    with pytest.raises(FileNotFoundError, match=r"custom\.mdl"):
        inspect_workstation_assets(workstation_assets)


def test_destination_symlink_escape_is_rejected(workstation_assets, tmp_path):
    target = tmp_path / "destination"
    (target / "robots").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (target / "robots/x5").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="destination escapes"):
        prepare_workstation_assets(workstation_assets, target)
    assert list(external.iterdir()) == []
