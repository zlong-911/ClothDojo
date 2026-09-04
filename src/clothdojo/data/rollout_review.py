"""Incremental lightweight review videos for canonical trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

from clothdojo.rendering import DATASET_CAMERA_NAMES
from clothdojo.robots import PIPER

REVIEW_CAMERA = "cam_high"


@dataclass(frozen=True, slots=True)
class _ReviewTrajectory:
    id: str
    path: Path
    asset_id: str
    manifest_sha256: str


def render_review(
    source: str | Path,
    *,
    assets_root: str | Path = "assets",
    device: str = "cuda:0",
    fps: int = 10,
    trajectory_ids: Sequence[str] | None = None,
    camera_name: str = REVIEW_CAMERA,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Incrementally render one native camera used by trajectory review."""

    from clothdojo.data.rollout import (
        _restore_render_state,
        _set_camera_cloth_surface_winding_reversed,
        _trajectory_surface_winding_reversed,
        _TrajectoryRenderState,
    )
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene
    from clothdojo.trajectory import TrajectoryReader

    root = Path(source).expanduser().resolve()
    if not isinstance(camera_name, str) or not camera_name:
        raise ValueError("review camera name must be a non-empty string")
    all_trajectories = _review_trajectories(root)
    trajectories = _select_review_trajectories(all_trajectories, trajectory_ids)
    destination = (
        Path(output).expanduser().resolve() if output is not None else root / "review"
    )
    videos = destination / "videos" / camera_name
    videos.mkdir(parents=True, exist_ok=True)
    previous = _review_records(destination / "meta.json")

    active_scene = None
    camera = None
    runtime = None
    selected_ids = {trajectory.id for trajectory in trajectories}
    current_ids = {trajectory.id for trajectory in all_trajectories}
    records_by_id = {
        trajectory_id: record
        for trajectory_id, record in previous.items()
        if trajectory_id in current_ids and trajectory_id not in selected_ids
    }
    rendered = 0
    reused = 0
    try:
        for trajectory in trajectories:
            reader = TrajectoryReader(trajectory.path)
            source_fps = float(reader.manifest.control_fps)
            stride = _review_stride(source_fps, fps)
            action_count = int(reader.manifest.action_count)
            surface_winding_reversed = _trajectory_surface_winding_reversed(reader)
            video = videos / f"{trajectory.id}.mp4"
            identity = {
                "trajectory_id": trajectory.id,
                "manifest_sha256": trajectory.manifest_sha256,
                "source_frames": action_count,
                "source_fps": source_fps,
                "video_fps": fps,
                "sample_stride": stride,
                "surface_winding_reversed": surface_winding_reversed,
            }
            old = previous.get(trajectory.id)
            if (
                old is not None
                and all(old.get(key) == value for key, value in identity.items())
                and video.is_file()
                and not video.is_symlink()
                and isinstance(old.get("video_frames"), int)
            ):
                records_by_id[trajectory.id] = old
                reused += 1
                continue

            robot_id = getattr(
                getattr(reader.manifest, "scene", None),
                "robot_id",
                PIPER,
            ) or PIPER
            scene_identity = (trajectory.asset_id, robot_id)
            if scene_identity != active_scene:
                _close_camera(camera)
                scene = build_scene(
                    assets_root,
                    spec=ShirtFlattenSpec(garment_id=trajectory.asset_id, world_count=1),
                    device=device,
                    visuals=True,
                    robot_id=robot_id,
                )
                runtime = _TrajectoryRenderState(scene)
                render_camera_names = (
                    (REVIEW_CAMERA,)
                    if camera_name == REVIEW_CAMERA
                    else DATASET_CAMERA_NAMES
                )
                camera = ShirtFlattenCameraObservation(
                    scene,
                    camera_names=render_camera_names,
                )
                active_scene = scene_identity

            _set_camera_cloth_surface_winding_reversed(
                camera,
                surface_winding_reversed,
            )

            encoder = _RawVideoEncoder(
                video,
                width=camera.width,
                height=camera.height,
                fps=fps,
            )
            video_frames = 0
            try:
                for sample_index in range(0, action_count, stride):
                    _restore_render_state(runtime, reader.state_at(sample_index))
                    image = camera(runtime)[f"observation.images.{camera_name}"][0]
                    encoder.write(image)
                    video_frames += 1
                encoder.close()
            except BaseException:
                encoder.abort()
                raise
            records_by_id[trajectory.id] = {**identity, "video_frames": video_frames}
            rendered += 1
    finally:
        _close_camera(camera)

    records = _merge_review_metadata(
        destination / "meta.json",
        source=root,
        camera=camera_name,
        trajectories=all_trajectories,
        selected_ids=selected_ids,
        records=records_by_id,
    )
    return {
        "source": str(root),
        "output": str(destination),
        "episodes": len(records),
        "rendered": rendered,
        "reused": reused,
        "source_frames": sum(int(item["source_frames"]) for item in records),
        "video_frames": sum(int(item["video_frames"]) for item in records),
        "video_fps": fps,
        "camera_names": [camera_name],
    }


def _review_trajectories(root: Path) -> list[_ReviewTrajectory]:
    meta_path = root / "meta.json"
    if meta_path.is_file():
        try:
            schema = json.loads(meta_path.read_text(encoding="utf-8")).get("schema")
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read source metadata: {meta_path}: {exc}") from exc
        if schema == "clothdojo_rollout":
            from clothdojo.rollout.collection import load_rollout_collection

            manifest = load_rollout_collection(root)
            return [
                _review_trajectory(root / item["path"], asset_id=str(item["asset_id"]))
                for item in manifest["trajectories"]
            ]
        if schema == "clothdojo_retarget_collection":
            from clothdojo.data.retarget import load_retarget_collection

            manifest = load_retarget_collection(root)
            return [
                _review_trajectory(root / item["path"], asset_id=str(item["asset_id"]))
                for item in manifest["trajectories"]
            ]
        raise ValueError(f"unsupported review source schema: {schema!r}")

    trajectories_root = root / "trajectories"
    if not trajectories_root.is_dir():
        raise NotADirectoryError(f"trajectory directory is missing: {trajectories_root}")
    result = []
    for manifest_path in sorted(trajectories_root.glob("*/manifest.json")):
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or value.get("status") != "complete":
            continue
        metadata = value.get("metadata")
        asset_id = metadata.get("asset_id") if isinstance(metadata, dict) else None
        if not isinstance(asset_id, str) or not asset_id:
            raise RuntimeError(f"trajectory asset_id is missing: {manifest_path}")
        result.append(_review_trajectory(manifest_path.parent, asset_id=asset_id))
    if not result:
        raise RuntimeError(f"no complete trajectories found in {trajectories_root}")
    return result


def _select_review_trajectories(
    trajectories: list[_ReviewTrajectory],
    trajectory_ids: Sequence[str] | None,
) -> list[_ReviewTrajectory]:
    if trajectory_ids is None:
        return trajectories
    requested = tuple(trajectory_ids)
    if not requested or any(not isinstance(value, str) or not value for value in requested):
        raise ValueError("review trajectory ids must be non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("review trajectory ids must be unique")
    by_id = {trajectory.id: trajectory for trajectory in trajectories}
    missing = [trajectory_id for trajectory_id in requested if trajectory_id not in by_id]
    if missing:
        raise ValueError(f"unknown review trajectory ids: {missing}")
    return [by_id[trajectory_id] for trajectory_id in requested]


def _review_trajectory(path: Path, *, asset_id: str) -> _ReviewTrajectory:
    manifest = path / "manifest.json"
    return _ReviewTrajectory(
        id=path.name,
        path=path,
        asset_id=asset_id,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )


def _review_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    episodes = value.get("episodes") if isinstance(value, dict) else None
    if not isinstance(episodes, list):
        return {}
    return {
        str(item["trajectory_id"]): item
        for item in episodes
        if isinstance(item, dict) and isinstance(item.get("trajectory_id"), str)
    }


def _merge_review_metadata(
    path: Path,
    *,
    source: Path,
    camera: str,
    trajectories: Sequence[_ReviewTrajectory],
    selected_ids: set[str],
    records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge one render shard without losing metadata written by another GPU."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(path.with_name(f".{path.name}.lock")):
        latest = _review_records(path)
        merged = {
            trajectory.id: latest[trajectory.id]
            for trajectory in trajectories
            if trajectory.id not in selected_ids and trajectory.id in latest
        }
        merged.update(
            {
                trajectory_id: record
                for trajectory_id, record in records.items()
                if trajectory_id in selected_ids
            }
        )
        ordered = [
            merged[trajectory.id]
            for trajectory in trajectories
            if trajectory.id in merged
        ]
        _atomic_json(
            path,
            {
                "source": str(source),
                "camera": camera,
                "episodes": ordered,
            },
        )
    return ordered


@contextmanager
def _exclusive_lock(path: Path):
    import fcntl

    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _close_camera(camera: Any | None) -> None:
    if camera is not None and hasattr(camera, "close"):
        camera.close()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _review_stride(source_fps: float, video_fps: int) -> int:
    if isinstance(video_fps, bool) or not isinstance(video_fps, int) or video_fps < 1:
        raise ValueError("review fps must be a positive integer")
    ratio = source_fps / video_fps
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > 1.0e-9:
        raise ValueError("review fps must divide the trajectory control rate exactly")
    return stride


class _RawVideoEncoder:
    def __init__(
        self,
        output: Path,
        *,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.output = output
        self.temporary = output.with_name(f".{output.stem}.tmp.mp4")
        self._shape = (height, width, 3)
        self._closed = False
        self._container = av.open(str(self.temporary), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"preset": "veryfast", "crf": "23"}

    def write(self, image: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("review video encoder is closed")
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.shape != self._shape:
            raise ValueError(
                f"review image must be uint8{self._shape}, got {array.dtype}{array.shape}"
            )
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
        except Exception:
            self.temporary.unlink(missing_ok=True)
            raise
        os.replace(self.temporary, self.output)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._container.close()
        self.temporary.unlink(missing_ok=True)
