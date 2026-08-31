from __future__ import annotations

import json
from pathlib import Path

import pytest

from clothdojo.data.asset_sampling import (
    asset_tempered_frame_weights,
    build_asset_sampling_index,
    load_asset_sampling_index,
    make_asset_tempered_sampler,
    validate_asset_sampling_index,
    write_asset_sampling_index,
)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    dataset = root / "dataset"
    collection = root / "collection"
    (dataset / "meta").mkdir(parents=True)
    episodes = [
        {"episode_index": 0, "tasks": ["flatten"], "length": 2},
        {"episode_index": 1, "tasks": ["flatten"], "length": 3},
    ]
    (dataset / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(episode, separators=(",", ":")) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    (dataset / "meta/info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 5}), encoding="utf-8"
    )
    sidecar_episodes = []
    for episode, trajectory, asset in zip(
        episodes, ("trajectory-a", "trajectory-b"), ("asset-a", "asset-b"), strict=True
    ):
        trajectory_root = collection / "trajectories" / trajectory
        trajectory_root.mkdir(parents=True)
        (trajectory_root / "manifest.json").write_text(
            json.dumps(
                {"action_count": episode["length"], "metadata": {"asset_id": asset}},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        sidecar_episodes.append(
            {
                "episode_index": episode["episode_index"],
                "length": episode["length"],
                "action_start": 0,
                "action_end_exclusive": episode["length"],
                "source_action_count": episode["length"],
                "source_trajectory": trajectory,
            }
        )
    (dataset / "meta/clothdojo.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_lerobot_export",
                "format_version": 1,
                "source": {"dataset_root": str(collection)},
                "episodes": sidecar_episodes,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset, collection


def test_index_round_trip_is_content_addressed_and_maps_episode_assets(tmp_path: Path) -> None:
    dataset, collection = _write_fixture(tmp_path)

    index = build_asset_sampling_index(dataset, collection_root=collection)
    output = write_asset_sampling_index(index, tmp_path / "asset_sampling_index.json")
    loaded = load_asset_sampling_index(output)

    assert loaded == index
    assert loaded.sha256 == index.sha256
    assert loaded.episode_to_asset == {0: "asset-a", 1: "asset-b"}
    assert loaded.asset_frame_counts == {"asset-a": 2, "asset-b": 3}
    assert [(entry.frame_start, entry.frame_end_exclusive) for entry in loaded.episodes] == [
        (0, 2),
        (2, 5),
    ]
    validate_asset_sampling_index(
        loaded,
        dataset,
        collection_root=collection,
        validate_source_manifests=True,
    )


def test_loader_rejects_tampered_index(tmp_path: Path) -> None:
    dataset, collection = _write_fixture(tmp_path)
    index = build_asset_sampling_index(dataset, collection_root=collection)
    output = write_asset_sampling_index(index, tmp_path / "asset_sampling_index.json")
    document = json.loads(output.read_text(encoding="utf-8"))
    document["episodes"][0]["asset_id"] = "tampered"
    output.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_asset_sampling_index(output)


def test_index_validation_detects_dataset_metadata_drift(tmp_path: Path) -> None:
    dataset, collection = _write_fixture(tmp_path)
    index = build_asset_sampling_index(dataset, collection_root=collection)
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["unrelated_but_provenance_relevant_field"] = True
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset hash mismatch"):
        validate_asset_sampling_index(index, dataset, collection_root=collection)


def test_builder_rejects_inconsistent_source_extent(tmp_path: Path) -> None:
    dataset, collection = _write_fixture(tmp_path)
    sidecar_path = dataset / "meta/clothdojo.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["episodes"][1]["action_end_exclusive"] = 2
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent extents"):
        build_asset_sampling_index(dataset, collection_root=collection)


def test_optional_source_validation_detects_manifest_drift(tmp_path: Path) -> None:
    dataset, collection = _write_fixture(tmp_path)
    index = build_asset_sampling_index(dataset, collection_root=collection)
    manifest_path = collection / "trajectories/trajectory-a/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["asset_id"] = "changed-asset"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validate_asset_sampling_index(index, dataset)
    with pytest.raises(ValueError, match="source collection manifests"):
        validate_asset_sampling_index(
            index,
            dataset,
            collection_root=collection,
            validate_source_manifests=True,
        )


def test_asset_tempering_weights_and_sampling_are_deterministic(tmp_path: Path) -> None:
    dataset, collection = _write_fixture(tmp_path)
    index = build_asset_sampling_index(dataset, collection_root=collection)

    weights = asset_tempered_frame_weights(index, alpha=0.0)
    assert weights.tolist() == pytest.approx([0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])
    assert weights[:2].sum().item() == pytest.approx(weights[2:].sum().item())
    first = list(make_asset_tempered_sampler(index, alpha=0.5, seed=43, num_samples=30))
    second = list(make_asset_tempered_sampler(index, alpha=0.5, seed=43, num_samples=30))
    third = list(make_asset_tempered_sampler(index, alpha=0.5, seed=44, num_samples=30))
    assert first == second
    assert first != third


def test_asset_tempering_rejects_invalid_parameters(tmp_path: Path) -> None:
    dataset, collection = _write_fixture(tmp_path)
    index = build_asset_sampling_index(dataset, collection_root=collection)

    with pytest.raises(ValueError, match="alpha"):
        asset_tempered_frame_weights(index, alpha=1.1)
    with pytest.raises(ValueError, match="seed"):
        make_asset_tempered_sampler(index, alpha=0.5, seed=-1)
