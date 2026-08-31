"""Current-frame geometry features for STEAM, shared by experts and rollouts.

These are simulation-privileged observations, not temporal-pair labels. Robot
state remains untouched so the same LeRobot data can still feed a 14-D policy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from clothdojo.evaluation.metric import factorized_cloth_distance, load_canonical_template
from clothdojo.evaluation.protocol import load_evaluation_protocol
from clothdojo.evaluation.templates import resolve_canonical_templates
from clothdojo.trajectory.reader import TrajectoryReader

GEOMETRY_KEY = "observation.cloth_metrics"
GEOMETRY_NAMES = ["deformable_distance", "rigid_distance"]
STATE_KEYS = ["observation.state", GEOMETRY_KEY]


def geometry_contract(protocol) -> dict:
    return {
        "version": 1,
        "feature_names": GEOMETRY_NAMES,
        "symmetry": protocol.symmetry,
        "target": protocol.target.to_json_dict(),
        "mirror_handling": "inverse_reflect_current_y",
        "alignment": "current_state_at_action_index",
    }


def frame_geometry(positions, template, *, symmetry, center_xy, augmentation=None):
    """Score current positions; undo a recorded bilateral mirror before scoring."""
    positions = np.asarray(positions, dtype=np.float32)
    if augmentation is not None:
        if augmentation.get("kind") != "bilateral_mirror" or augmentation.get("mirror_plane") != {
            "axis": "y",
            "offset_m": 0.0,
        }:
            raise ValueError("unsupported geometry augmentation")
        positions = positions.copy()
        positions[..., 1] *= -1
    scores = factorized_cloth_distance(
        positions,
        np.zeros_like(positions),
        template,
        symmetry=symmetry,
        symmetry_center_xy=center_xy,
    )
    return np.stack((scores.deformable_distance, scores.rigid_distance), axis=-1)


def _score_trajectory(job):
    source, template_path, cache_path, contract = job
    reader = TrajectoryReader(source)
    template = load_canonical_template(template_path)
    if reader.manifest.metadata["asset_id"] != template.asset_id:
        raise ValueError("geometry target asset does not match trajectory")
    cache = Path(cache_path)
    if cache.exists():
        result = np.load(cache, allow_pickle=False)
        if result.shape != (reader.manifest.action_count, 2) or not np.isfinite(result).all():
            raise ValueError(f"invalid cached geometry: {cache}")
        return str(cache)
    arrays = reader.state_arrays(("particle_q",), mmap=True)
    positions = arrays["particle_q"]
    output = np.empty((reader.manifest.action_count, 2), dtype=np.float32)
    for start in range(0, len(output), 32):
        end = min(start + 32, len(output))
        output[start:end] = frame_geometry(
            positions[start:end],
            template,
            symmetry=contract["symmetry"],
            center_xy=contract["target"]["symmetry_center_xy"],
            augmentation=reader.manifest.metadata.get("augmentation"),
        )
    temporary = cache.with_suffix(".tmp.npy")
    np.save(temporary, output, allow_pickle=False)
    temporary.replace(cache)
    return str(cache)


def prepare_geometry_dataset(
    *, dataset, trajectories, templates, protocol, selection, output, workers=8
):
    """Select explicitly approved whole trajectories, reuse video, add geometry.

    Selection maps original IDs to their approval evidence. Mirrors inherit
    that evidence, not a fabricated annotation/review status. Output is only
    published after all features and mirror-consistency checks pass.
    """
    source = Path(dataset).resolve()
    trajectory_root = Path(trajectories).resolve()
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    approved = json.loads(Path(selection).read_text())
    if not approved or not all(isinstance(v, str) and v for v in approved.values()):
        raise ValueError("selection must map original IDs to non-empty approval evidence")
    info = json.loads((source / "meta/info.json").read_text())
    provenance = json.loads((source / "meta/clothdojo.json").read_text())
    if provenance.get("export_status") != "complete":
        raise ValueError("source export is incomplete")
    if provenance.get("rendering", {}).get("cloth_visual", "nocs") != "nocs":
        raise ValueError("this STEAM recipe requires NOCS rendering")
    selected = [
        row
        for row in provenance["episodes"]
        if row["source_trajectory"].removesuffix("__mirror") in approved
        and row.get("full_trajectory") is True
    ]
    originals = {
        row["source_trajectory"]
        for row in selected
        if not row["source_trajectory"].endswith("__mirror")
    }
    if originals != set(approved) or len(selected) != 2 * len(approved):
        raise ValueError("every approved original must have exactly one complete mirror")
    identifiers = [row["source_trajectory"] for row in selected]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate whole-trajectory episodes")
    manifests = {
        name: json.loads((trajectory_root / name / "manifest.json").read_text())
        for name in identifiers
    }
    assets = {m["metadata"]["asset_id"] for m in manifests.values()}
    recipe = load_evaluation_protocol(protocol)
    contract = geometry_contract(recipe)
    targets = resolve_canonical_templates(
        templates, assets, target_settings=recipe.target.to_json_dict()
    )
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cache = destination.parent / f".{destination.name}.geometry-cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs, scores = [], {}
    for row in selected:
        name = row["source_trajectory"]
        manifest = manifests[name]
        if row["length"] != manifest["action_count"] or row["action_start"] != 0:
            raise ValueError(f"whole-trajectory alignment differs: {name}")
        target, target_hash = targets[manifest["metadata"]["asset_id"]]
        identity = json.dumps([manifest, target_hash, contract_hash], sort_keys=True).encode()
        path = cache / f"{name}-{hashlib.sha256(identity).hexdigest()[:20]}.npy"
        scores[name] = path
        jobs.append((str(trajectory_root / name), str(target), str(path), contract))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_score_trajectory, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 20 == 0 or count == len(jobs):
                print(f"geometry {count}/{len(jobs)}", flush=True)
    for name in approved:
        a, b = (np.load(scores[key]) for key in (name, name + "__mirror"))
        if not np.allclose(a, b, atol=2e-6, rtol=0):
            raise ValueError(f"mirror geometry disagrees: {name}")

    staging = destination.parent / f".{destination.name}.publishing"
    staging.mkdir()  # Do not overwrite a previous incomplete publication.
    (staging / "meta").mkdir()
    episode_stats = {
        r["episode_index"]: r for r in _jsonlines(source / "meta/episodes_stats.jsonl")
    }
    episode_meta = {r["episode_index"]: r for r in _jsonlines(source / "meta/episodes.jsonl")}
    new_rows, new_stats, new_episodes, all_states = [], [], [], []
    offset = 0
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    for index, row in enumerate(selected):
        old_index, length = row["episode_index"], row["length"]
        old_fields = {"episode_index": old_index, "episode_chunk": old_index // info["chunks_size"]}
        fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
        table = pq.read_table(source / info["data_path"].format(**old_fields))
        if len(table) != length:
            raise ValueError("parquet length differs from episode provenance")
        for key, values in (
            ("episode_index", np.full(length, index)),
            ("index", np.arange(offset, offset + length)),
        ):
            table = table.set_column(
                table.schema.get_field_index(key), key, pa.array(values, type=pa.int64())
            )
        values = np.load(scores[row["source_trajectory"]])
        table = table.append_column(
            GEOMETRY_KEY, pa.array(values.tolist(), type=pa.list_(pa.float32(), 2))
        )
        path = staging / info["data_path"].format(**fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        for key in cameras:
            original = source / info["video_path"].format(**old_fields, video_key=key)
            if not original.is_file():
                raise FileNotFoundError(original)
            target = staging / info["video_path"].format(**fields, video_key=key)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(original, target.parent))
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        all_states.append(np.concatenate((state, values.astype(np.float64)), axis=1))
        stats = copy.deepcopy(episode_stats[old_index])
        stats["episode_index"] = index
        from clothdojo.data.lerobot import _shift_episode_stats

        _shift_episode_stats(stats, episode_index=index, frame_offset=offset)
        stats["stats"][GEOMETRY_KEY] = _stats(values)
        new_stats.append(stats)
        new_episodes.append({**episode_meta[old_index], "episode_index": index})
        new_rows.append(
            {
                **row,
                "episode_index": index,
                "quality_basis": approved[row["source_trajectory"].removesuffix("__mirror")],
            }
        )
        offset += length
    combined = np.concatenate(all_states)
    normalization = {
        "state_keys": STATE_KEYS,
        "state_dim": 16,
        "state_mean": combined.mean(axis=0).tolist(),
        "state_std": combined.std(axis=0).tolist(),
        "geometry_contract": contract,
    }
    info.update(
        total_episodes=len(selected),
        total_frames=offset,
        total_videos=len(selected) * len(cameras),
        total_chunks=(len(selected) + info["chunks_size"] - 1) // info["chunks_size"],
        splits={"train": f"0:{len(selected)}"},
    )
    info["features"][GEOMETRY_KEY] = {"dtype": "float32", "shape": [2], "names": GEOMETRY_NAMES}
    provenance.update(
        episodes=new_rows,
        selection={"rule": "approved_whole_trajectory_good", "approved_originals": approved},
        counts={
            "whole_trajectory_good_only": True,
            "output_episodes": len(selected),
            "output_frames": offset,
        },
        geometry={"contract": contract, "templates": {k: v[1] for k, v in targets.items()}},
        parent_dataset={
            "root": str(source),
            "provenance_sha256": hashlib.sha256(
                (source / "meta/clothdojo.json").read_bytes()
            ).hexdigest(),
        },
    )
    provenance.pop("episode_sets", None)
    for name, value in (
        ("info.json", info),
        ("clothdojo.json", provenance),
        ("steam_state.json", normalization),
    ):
        (staging / "meta" / name).write_text(json.dumps(value, indent=2) + "\n")
    for name, rows in (("episodes.jsonl", new_episodes), ("episodes_stats.jsonl", new_stats)):
        (staging / "meta" / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copyfile(source / "meta/tasks.jsonl", staging / "meta/tasks.jsonl")
    staging.rename(destination)
    print(
        json.dumps({"output": str(destination), "episodes": len(selected), "frames": offset}),
        flush=True,
    )


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dataset", "trajectories", "templates", "protocol", "selection", "output"):
        parser.add_argument(f"--{argument}", required=True)
    parser.add_argument("--workers", type=int, default=8)
    prepare_geometry_dataset(**vars(parser.parse_args()))
