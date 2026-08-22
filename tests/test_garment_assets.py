from __future__ import annotations

import json
from pathlib import Path

import pytest

from clothdojo.assets import GarmentLibrary


def _candidate(assets: Path, asset_id: str = "rand_ZZCWITH5HY") -> GarmentLibrary:
    library = GarmentLibrary(assets)
    preview = library.previews_root / asset_id
    preview.mkdir(parents=True)
    (preview / "front.png").write_bytes(b"front")
    (preview / "back.png").write_bytes(b"back")
    return library


def test_selection_and_conversion_define_collectable_assets(tmp_path: Path) -> None:
    library = _candidate(tmp_path / "assets")

    [garment] = library.garments()
    assert garment.decision == "pending"
    assert garment.converted is False
    assert library.preview(garment.id, "front") == garment.front
    assert library.preview(garment.id, "back") == garment.back

    library.decide(garment.id, "accepted")
    source = tmp_path / "conversion.usdc"
    source.write_bytes(b"usd")
    installed = library.install_converted(garment.id, source)

    assert installed == library.converted(garment.id)
    assert installed.name == "rand_ZZCWITH5HY.usdc"
    assert library.collectable_ids() == ("rand_ZZCWITH5HY",)
    assert json.loads(library.selection_path.read_text(encoding="utf-8")) == {
        "accepted": ["rand_ZZCWITH5HY"],
        "rejected": [],
    }

    library.decide(garment.id, "rejected")
    assert library.collectable_ids() == ()
    assert installed.is_file()


def test_install_does_not_overwrite_converted_asset(tmp_path: Path) -> None:
    library = _candidate(tmp_path / "assets")
    source = tmp_path / "conversion.usdc"
    source.write_bytes(b"first")
    installed = library.install_converted("rand_ZZCWITH5HY", source)
    source.write_bytes(b"second")

    with pytest.raises(FileExistsError, match="already exists"):
        library.install_converted("rand_ZZCWITH5HY", source)

    assert installed.read_bytes() == b"first"


def test_selection_rejects_overlap(tmp_path: Path) -> None:
    library = _candidate(tmp_path / "assets")
    library.selection_path.parent.mkdir(parents=True, exist_ok=True)
    library.selection_path.write_text(
        json.dumps(
            {
                "accepted": ["rand_ZZCWITH5HY"],
                "rejected": ["rand_ZZCWITH5HY"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="accepted and rejected"):
        library.garments()


def test_web_facing_paths_reject_unknown_and_unsafe_ids(tmp_path: Path) -> None:
    library = _candidate(tmp_path / "assets")

    with pytest.raises(KeyError, match="unknown garment"):
        library.decide("rand_UNKNOWN", "accepted")
    with pytest.raises(ValueError, match="invalid garment id"):
        library.preview("../secret", "front")
