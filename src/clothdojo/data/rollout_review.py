"""Fast, directly encoded review videos for canonical rollout trajectories."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import av
import numpy as np

REVIEW_SCHEMA = "clothdojo_rollout_review"
REVIEW_FORMAT_VERSION = 1
REVIEW_CAMERA = "cam_high"


def render_rollout_review(
    source: str | Path,
    *,
    output: str | Path | None = None,
    assets_root: str | Path = "assets",
    device: str = "cuda:0",
    fps: int = 10,
) -> dict[str, Any]:
    """Render one lightweight HIL review stream without building a solver."""

    from clothdojo.data.rollout import (
        _load_canonical_collection,
        _restore_render_state,
        _TrajectoryRenderState,
    )
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.sim.scene import ShirtFlattenSpec, build_shirt_flatten
    from clothdojo.trajectory import TrajectoryReader

    root = Path(source).expanduser().resolve()
    destination = (
        Path(output).expanduser().resolve() if output is not None else root / "review"
    )
    if destination.exists():
        raise FileExistsError(f"rollout review output already exists: {destination}")
    staging = destination.parent / f".{destination.name}.tmp"
    if staging.exists():
        raise FileExistsError(f"unfinished rollout review output exists: {staging}")

    manifest, source_kind = _load_canonical_collection(root)
    trajectories = manifest["trajectories"]
    first_reader = TrajectoryReader(root / trajectories[0]["path"])
    source_fps = float(first_reader.manifest.control_fps)
    stride = _review_stride(source_fps, fps)
    source_meta_sha256 = hashlib.sha256((root / "meta.json").read_bytes()).hexdigest()

    videos = staging / "videos" / REVIEW_CAMERA
    videos.mkdir(parents=True)
    active_asset = None
    scene = None
    runtime = None
    camera = None
    episode_records = []
    try:
        for episode_index, episode in enumerate(trajectories):
            asset_id = str(episode["asset_id"])
            if asset_id != active_asset:
                active_asset = asset_id
                scene = build_shirt_flatten(
                    assets_root,
                    spec=ShirtFlattenSpec(garment_id=asset_id, world_count=1),
                    device=device,
                    visuals=True,
                )
                runtime = _TrajectoryRenderState(scene)
                camera = ShirtFlattenCameraObservation(
                    scene,
                    camera_names=(REVIEW_CAMERA,),
                )
            reader = TrajectoryReader(root / episode["path"])
            if float(reader.manifest.control_fps) != source_fps:
                raise ValueError("rollout trajectories use different control rates")
            action_count = int(reader.manifest.action_count)
            if action_count != int(episode["action_count"]):
                raise ValueError("rollout trajectory count differs from collection metadata")
            encoder = _RawVideoEncoder(
                videos / f"episode_{episode_index:06d}.mp4",
                width=camera.width,
                height=camera.height,
                fps=fps,
            )
            frame_count = 0
            try:
                for sample_index in range(0, action_count, stride):
                    _restore_render_state(runtime, reader.state_at(sample_index))
                    image = camera(runtime)[f"observation.images.{REVIEW_CAMERA}"][0]
                    encoder.write(image)
                    frame_count += 1
                encoder.close()
            except Exception:
                encoder.abort()
                raise
            episode_records.append(
                {
                    "episode_index": episode_index,
                    "trajectory": episode["path"],
                    "source_frames": action_count,
                    "video_frames": frame_count,
                }
            )
        review_meta = {
            "schema": REVIEW_SCHEMA,
            "format_version": REVIEW_FORMAT_VERSION,
            "status": "complete",
            "source": str(root),
            "source_kind": source_kind,
            "source_meta_sha256": source_meta_sha256,
            "source_fps": source_fps,
            "video_fps": fps,
            "sample_stride": stride,
            "camera_names": [REVIEW_CAMERA],
            "episodes": episode_records,
        }
        (staging / "meta.json").write_text(
            json.dumps(review_meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
    except Exception:
        raise

    return {
        "source": str(root),
        "output": str(destination),
        "episodes": len(episode_records),
        "source_frames": sum(item["source_frames"] for item in episode_records),
        "video_frames": sum(item["video_frames"] for item in episode_records),
        "source_fps": source_fps,
        "video_fps": fps,
        "sample_stride": stride,
        "camera_names": [REVIEW_CAMERA],
    }


def _review_stride(source_fps: float, video_fps: int) -> int:
    if isinstance(video_fps, bool) or not isinstance(video_fps, int) or video_fps < 1:
        raise ValueError("review fps must be a positive integer")
    ratio = source_fps / video_fps
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > 1.0e-9:
        raise ValueError("review fps must divide the rollout control rate exactly")
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
