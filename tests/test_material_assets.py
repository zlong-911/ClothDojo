from __future__ import annotations

import json
from pathlib import Path

import pytest

from clothdojo.assets.materials import (
    DEFAULT_MATERIAL_CATALOG,
    load_material_catalog,
    prepare_polyhaven_assets,
    require_polyhaven_assets,
)


def test_catalog_contains_only_runtime_materials() -> None:
    catalog = load_material_catalog()

    assert len(catalog.materials) == 22
    assert len(catalog.pool("cloth")) == 14
    assert sum(len(item.color_channels) for item in catalog.pool("cloth")) == 19
    assert len(catalog.pool("table")) == 4
    assert len(catalog.pool("ground")) == 4
    assert len(catalog.lighting) == 4
    calibrated = next(
        item for item in catalog.lighting if item.id == "studio_small_03"
    )
    assert calibrated.exposure_offset == -1.4


def test_catalog_preserves_physical_texture_scale_and_colorways() -> None:
    catalog = load_material_catalog()
    book = catalog.material("book_pattern")
    table = catalog.material("wood_table_001")

    assert book.base_color_name("col1") == "base_color_col1.jpg"
    assert book.uv_scale() == pytest.approx((10.0 / 300.0, 10.0 / 300.0))
    assert table.base_color_name("Diffuse") == "base_color.jpg"
    assert table.uv_scale() == pytest.approx((1000.0 / 1500.0, 1000.0 / 1500.0))


def test_runtime_asset_contract_requires_only_rendered_channels(tmp_path: Path) -> None:
    catalog = load_material_catalog()
    for item in catalog.materials:
        directory = item.directory(tmp_path)
        directory.mkdir(parents=True)
        for channel in item.color_channels:
            (directory / item.base_color_name(channel)).write_bytes(b"color")
        (directory / "normal_gl.jpg").write_bytes(b"normal")
        (directory / "roughness.jpg").write_bytes(b"roughness")
    for item in catalog.lighting:
        path = tmp_path / "lighting" / f"{item.id}.hdr"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"hdr")

    assert require_polyhaven_assets(tmp_path, catalog) == tmp_path.resolve()

    (catalog.materials[0].directory(tmp_path) / "normal_gl.jpg").unlink()
    with pytest.raises(FileNotFoundError, match=r"normal_gl\.jpg"):
        require_polyhaven_assets(tmp_path, catalog)


def test_complete_material_set_is_reused_without_network(
    monkeypatch, tmp_path: Path
) -> None:
    catalog = load_material_catalog()
    for item in catalog.materials:
        directory = item.directory(tmp_path)
        directory.mkdir(parents=True)
        for channel in item.color_channels:
            (directory / item.base_color_name(channel)).write_bytes(b"color")
        (directory / "normal_gl.jpg").write_bytes(b"normal")
        (directory / "roughness.jpg").write_bytes(b"roughness")
    for item in catalog.lighting:
        path = tmp_path / "lighting" / f"{item.id}.hdr"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"hdr")
    monkeypatch.setattr(
        "clothdojo.assets.materials._api_json",
        lambda _url: pytest.fail("complete local assets must not access the network"),
    )

    result = prepare_polyhaven_assets(tmp_path, catalog)

    assert result["downloaded_files"] == 0
    assert result["reused_files"] == 75
    assert result["bytes"] > 0


def test_catalog_rejects_unpinned_assets(tmp_path: Path) -> None:
    value = json.loads(DEFAULT_MATERIAL_CATALOG.read_text(encoding="utf-8"))
    value["source_files_hashes"].pop(next(iter(value["source_files_hashes"])))
    path = tmp_path / "materials.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="source hashes"):
        load_material_catalog(path)
