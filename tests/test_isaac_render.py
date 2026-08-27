from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from clothdojo.assets.materials import load_material_catalog
from clothdojo.data.isaac_render import (
    _ROOMS,
    ROBODOJO_REVISION,
    _matches_semantic_body,
    _physical_gpu,
    _require_robodojo_rooms,
    _require_visual_assets,
    _split_tiled_rgb,
    export_isaac_rollout_collection,
    parse_visual_randomization,
    prepare_isaac_assets,
    sample_visual_layout,
)


def test_visual_layout_is_stable_by_source_episode() -> None:
    first = sample_visual_layout(7, 0, source_episode_id="shirt_000003")
    reordered = sample_visual_layout(7, 99, source_episode_id="shirt_000003")
    changed = sample_visual_layout(8, 0, source_episode_id="shirt_000003")

    assert asdict(first) == {**asdict(reordered), "episode_index": 0}
    assert first.seed != changed.seed
    assert first.room is None
    assert 0.0 <= first.lighting.rotation_z <= 360.0


def test_visual_layout_applies_hdri_exposure_calibration() -> None:
    layout = sample_visual_layout(
        17,
        0,
        source_episode_id="existing_rollout_excerpt",
    )

    assert layout.lighting.environment == "studio_small_03"
    assert layout.lighting.exposure == pytest.approx(-1.780174)


def test_visual_layout_room_is_explicit() -> None:
    neutral = sample_visual_layout(3, 2, include_room=False)
    room = sample_visual_layout(3, 2, include_room=True)

    assert neutral.table == room.table
    assert neutral.ground == room.ground
    assert neutral.cloth == room.cloth
    assert neutral.lighting == room.lighting
    assert neutral.room is None
    assert room.room is not None


def test_visual_layout_uses_semantic_independent_pools() -> None:
    catalog = load_material_catalog()
    layouts = [sample_visual_layout(11, index) for index in range(200)]
    grounds_by_table = {}
    for layout in layouts:
        grounds_by_table.setdefault(layout.table.material, set()).add(layout.ground.material)

    assert {layout.cloth.material for layout in layouts} <= {
        item.id for item in catalog.pool("cloth")
    }
    assert max(map(len, grounds_by_table.values())) >= 3
    assert {layout.table.material for layout in layouts} <= {
        item.id for item in catalog.pool("table")
    }
    assert {layout.ground.material for layout in layouts} <= {
        item.id for item in catalog.pool("ground")
    }


def test_visual_layout_colorways_belong_to_sampled_materials() -> None:
    catalog = load_material_catalog()
    layouts = [sample_visual_layout(5, index) for index in range(100)]
    appearances = [
        appearance
        for layout in layouts
        for appearance in (layout.cloth, layout.table, layout.ground)
    ]

    assert all(
        appearance.color_channel in catalog.material(appearance.material).color_channels
        for appearance in appearances
    )


def test_visual_layout_randomizes_only_selected_components() -> None:
    layouts = [sample_visual_layout(9, index, randomize=("cloth",)) for index in range(50)]

    assert len({layout.cloth for layout in layouts}) > 1
    assert len({layout.table for layout in layouts}) == 1
    assert len({layout.ground for layout in layouts}) == 1
    assert len({layout.lighting for layout in layouts}) == 1


@pytest.mark.parametrize(
    "value, expected",
    [
        ("none", ()),
        ("cloth", ("cloth",)),
        ("lighting,cloth", ("cloth", "lighting")),
        ("all", ("cloth", "table", "ground", "lighting")),
    ],
)
def test_visual_randomization_selector_is_concise_and_canonical(value, expected) -> None:
    assert parse_visual_randomization(value) == expected


def test_visual_randomization_selector_rejects_unknown_components() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_visual_randomization("cloth,camera")


def test_visual_layout_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        sample_visual_layout(-1, 0)


def test_prepare_isaac_assets_pins_revision_and_minimal_subset(monkeypatch, tmp_path) -> None:
    call = []

    def prepare(output, catalog):
        call.append((output, catalog))
        return {"output": str(output), "downloaded_files": 0, "reused_files": 1, "bytes": 1}

    monkeypatch.setattr("clothdojo.data.isaac_render.prepare_polyhaven_assets", prepare)
    monkeypatch.setattr(
        "clothdojo.data.isaac_render._require_visual_assets",
        lambda *_args, **_kwargs: None,
    )

    result = prepare_isaac_assets(tmp_path)

    assert call[0][0] == tmp_path / "polyhaven"
    assert result["polyhaven"]["reused_files"] == 1
    assert result["robodojo_revision"] is None
    assert result["rooms"] == "excluded"


def test_tiled_rgb_preserves_dataset_camera_order() -> None:
    tiled = np.zeros((960, 1280, 4), dtype=np.uint8)
    tiled[:480, :640, :3] = 1
    tiled[:480, 640:, :3] = 2
    tiled[480:, :640, :3] = 3
    tiled[480:, 640:, :3] = 4

    images = _split_tiled_rgb(tiled, camera_count=4)

    assert [int(image[0, 0, 0]) for image in images] == [1, 2, 3, 4]
    assert all(image.shape == (480, 640, 3) for image in images)


def test_semantic_camera_mount_matches_full_newton_body_path() -> None:
    left = "/piper_dual/Geometry/world_base/left/base_link/link1/link6/camera"

    assert _matches_semantic_body(left, "left/camera")
    assert not _matches_semantic_body(left, "right/camera")
    assert not _matches_semantic_body(left, "left/link6")


def test_robodojo_rooms_require_pinned_revision(tmp_path) -> None:
    (tmp_path / "clothdojo_revision.txt").write_text(f"{ROBODOJO_REVISION}\n")
    for room in _ROOMS:
        path = tmp_path / "Assets/Room" / room / "base0.usd"
        path.parent.mkdir(parents=True)
        path.write_text("usd")

    _require_robodojo_rooms(tmp_path)


def test_missing_visual_assets_point_to_the_single_preparation_command(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare-isaac-assets"):
        _require_visual_assets(tmp_path, load_material_catalog(), include_room=False)


@pytest.mark.parametrize("value, expected", [("cuda:0", 0), ("cuda:12", 12)])
def test_isaac_device_is_an_explicit_physical_gpu(value, expected) -> None:
    assert _physical_gpu(value) == expected


def test_isaac_device_rejects_cpu_and_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="physical CUDA index"):
        _physical_gpu("cpu")


def test_isaac_export_uses_the_canonical_collection_loop(monkeypatch, tmp_path) -> None:
    calls = {}

    class App:
        closed = False

        def is_running(self):
            return True

        def close(self, *, wait_for_replicator):
            calls["wait_for_replicator"] = wait_for_replicator
            self.closed = True

    app = App()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("clothdojo.data.isaac_render._require_isaac5_runtime", lambda: None)
    monkeypatch.setattr(
        "clothdojo.data.isaac_render._require_visual_assets",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr("clothdojo.data.isaac_render._preload_python_libstdcpp", lambda: None)
    monkeypatch.setattr("clothdojo.data.isaac_render._start_isaac_app", lambda *_a: app)
    monkeypatch.setattr(
        "clothdojo.teleop.isaac_compat.configure_replicator_warp",
        lambda: None,
    )
    monkeypatch.setattr(
        "clothdojo.data.isaac_render.IsaacRandomizedCamera",
        lambda *args, **kwargs: calls.update(camera_args=args, camera_kwargs=kwargs) or object(),
    )

    def export(source, output, **kwargs):
        calls.update(source=source, output=output, export_kwargs=kwargs)
        kwargs["camera_factory"]("shirt_a", object())
        return {"renderer": kwargs["rendering"]["backend"]}

    monkeypatch.setattr("clothdojo.data.rollout._export_canonical_collection", export)

    result = export_isaac_rollout_collection(
        tmp_path / "source",
        tmp_path / "output",
        assets_root=tmp_path / "assets",
        visual_assets_root=tmp_path / "visuals",
        device="cuda:3",
        seed=17,
        include_room=False,
        randomize=("cloth", "lighting"),
        repo_id=None,
        image_writer_threads=4,
    )

    assert result == {"renderer": "isaac_rtx"}
    assert (
        calls["camera_kwargs"]["asset_path"]
        == (tmp_path / "assets/garments/converted/shirt_a/shirt_a.usdc").resolve()
    )
    assert calls["export_kwargs"]["rendering"]["randomized_components"] == [
        "cloth",
        "lighting",
    ]
    assert calls["wait_for_replicator"] is False
    assert app.closed
