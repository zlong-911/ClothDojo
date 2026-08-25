"""Transactional LeRobot publication and canonical rollout replay."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.data.lerobot import (
    EXPORT_FORMAT_VERSION,
    EXPORT_SCHEMA,
    _create_writer,
    _validate_video,
)

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ARM_JOINTS = tuple(f"{side}/joint{joint}" for side in ("left", "right") for joint in range(1, 7))


class RolloutLeRobotWriter:
    """Write complete rollout episodes and atomically publish their root."""

    def __init__(
        self,
        output: str | Path,
        *,
        fps: int,
        prompt: str,
        scenario: dict[str, Any],
        trajectory: str,
        policy: dict[str, Any],
        training_eligible: bool,
        source_kind: str = "rtc_rollout",
        image_height: int = 480,
        image_width: int = 640,
        repo_id: str | None = None,
        image_writer_threads: int = 6,
        _writer_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.output = Path(output).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"rollout LeRobot output already exists: {self.output}")
        self.staging = self.output.parent / f".{self.output.name}.tmp"
        if self.staging.exists():
            raise FileExistsError(f"unfinished rollout staging exists: {self.staging}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.prompt = str(prompt)
        self.scenario = dict(scenario)
        self.trajectory = str(trajectory)
        self.policy = dict(policy)
        self.training_eligible = bool(training_eligible)
        self.source_kind = str(source_kind)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.repo_id = repo_id or f"local/{self.output.name}"
        self._injected = _writer_factory is not None
        if self._injected:
            self.staging.mkdir()
        factory = _writer_factory or _create_writer
        self.writer = factory(
            repo_id=self.repo_id,
            root=self.staging,
            fps=self.fps,
            robot_type="piper",
            features=_features(self.image_height, self.image_width),
            image_writer_threads=image_writer_threads,
        )
        self.frames = 0
        self._episode_frames = 0
        self._episode_lengths: list[int] = []
        self._closed = False

    def add_frame(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot add a frame to a closed rollout dataset")
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.shape != (1, 14) or action.shape != (14,):
            raise ValueError(
                f"rollout state/action must be (1, 14)/(14,), got {state.shape}/{action.shape}"
            )
        frame: dict[str, Any] = {
            "observation.state": state[0].copy(),
            "action": action.copy(),
            "task": self.prompt,
        }
        for camera in CAMERA_NAMES:
            key = f"observation.images.{camera}"
            image = np.asarray(observation[key])
            expected = (1, self.image_height, self.image_width, 3)
            if image.dtype != np.uint8 or image.shape != expected:
                raise ValueError(
                    f"rollout image {key} must be uint8{expected}, got {image.dtype}{image.shape}"
                )
            frame[key] = image[0].copy()
        self.writer.add_frame(frame)
        self.frames += 1
        self._episode_frames += 1

    def save_episode(self) -> None:
        if self._closed:
            raise RuntimeError("cannot save an episode in a closed rollout dataset")
        if self._episode_frames < 1:
            raise RuntimeError("cannot save an empty rollout episode")
        self.writer.save_episode()
        self._episode_lengths.append(self._episode_frames)
        self._episode_frames = 0

    def close(self) -> None:
        if self._closed:
            return
        if self.frames < 1:
            raise RuntimeError("cannot publish an empty rollout dataset")
        if self._episode_frames:
            self.save_episode()
        if not self._injected:
            _validate_output(
                self.staging,
                repo_id=self.repo_id,
                episode_lengths=self._episode_lengths,
            )
        _write_sidecar(
            self.staging,
            frames=self.frames,
            scenario=self.scenario,
            trajectory=self.trajectory,
            policy=self.policy,
            training_eligible=self.training_eligible,
            source_kind=self.source_kind,
            episode_lengths=self._episode_lengths,
        )
        self.staging.rename(self.output)
        self._closed = True

    def abort(self) -> None:
        self._closed = True


def inspect_rollout_collection_export(source: str | Path) -> dict[str, Any]:
    """Validate a trajectory collection without building Newton or decoding video."""

    from clothdojo.rollout.collection import load_rollout_collection

    root = Path(source).expanduser().resolve()
    manifest = load_rollout_collection(root)
    frames = sum(int(item["action_count"]) for item in manifest["trajectories"])
    return {
        "source_kind": "rollout_collection",
        "source": str(root),
        "output_format": "LeRobot v2.1",
        "output_episodes": len(manifest["trajectories"]),
        "output_frames": frames,
        "camera_names": list(CAMERA_NAMES),
        "training_eligible": manifest.get("training_eligible") is True,
    }


def export_rollout_collection(
    source: str | Path,
    output: str | Path,
    *,
    assets_root: str | Path = "assets",
    device: str = "cuda:0",
    repo_id: str | None = None,
    image_writer_threads: int = 6,
) -> dict[str, Any]:
    """Replay canonical rollout states and publish one multi-episode LeRobot dataset."""

    from clothdojo.rollout.collection import load_rollout_collection
    from clothdojo.rollout.newton import pack_policy_state
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_shirt_flatten
    from clothdojo.trajectory import TrajectoryReader

    root = Path(source).expanduser().resolve()
    manifest = load_rollout_collection(root)
    scenario = manifest["scenario"]
    trajectories = manifest["trajectories"]
    first_reader = TrajectoryReader(root / trajectories[0]["path"])
    sink = RolloutLeRobotWriter(
        output,
        fps=int(first_reader.manifest.control_fps),
        prompt=scenario["prompt"],
        scenario=scenario,
        trajectory=str(root),
        policy=manifest["policy"],
        training_eligible=manifest.get("training_eligible") is True,
        source_kind="rollout_collection",
        repo_id=repo_id,
        image_writer_threads=image_writer_threads,
    )
    try:
        active_asset = None
        scene = None
        runtime = None
        camera = None
        for episode in trajectories:
            if episode["asset_id"] != active_asset:
                active_asset = episode["asset_id"]
                spec = ShirtFlattenSpec(garment_id=active_asset, world_count=1)
                scene = build_shirt_flatten(
                    assets_root,
                    spec=spec,
                    device=device,
                    visuals=True,
                )
                runtime = ShirtFlattenRuntime(scene)
                camera = ShirtFlattenCameraObservation(scene)
            reader = TrajectoryReader(root / episode["path"])
            if reader.manifest.control_fps != first_reader.manifest.control_fps:
                raise ValueError("rollout trajectories use different control rates")
            for sample in reader.iter_samples():
                _restore_render_state(runtime, sample.state)
                arm_q = scene.arms.get_attribute("joint_q", runtime.state_0).numpy()[:, 0]
                observation = {
                    "observation.state": pack_policy_state(
                        arm_q,
                        sample.state.gripper_open_state.reshape(1, 2),
                    ),
                    **camera(runtime),
                }
                policy_action = pack_policy_state(
                    sample.action.joint_target.reshape(1, 12),
                    sample.action.gripper_open_command.reshape(1, 2),
                )[0]
                sink.add_frame(observation, policy_action)
            sink.save_episode()
        sink.close()
    except Exception:
        sink.abort()
        raise
    summary = inspect_rollout_collection_export(root)
    summary.update({"compatible": True, "output": str(Path(output).expanduser().resolve())})
    return summary


def _restore_render_state(runtime, state) -> None:
    robot = runtime.scene.robot
    for name, values in (
        ("joint_q", state.joint_q),
        ("joint_qd", state.joint_qd),
        ("body_q", state.body_q),
        ("body_qd", state.body_qd),
    ):
        target = robot.get_attribute(name, runtime.state_0)
        target.assign(np.asarray(values).reshape(target.numpy().shape))
        robot.set_attribute(name, runtime.state_0, target)
    runtime.state_0.particle_q.assign(state.particle_q)
    runtime.state_0.particle_qd.assign(state.particle_qd)


def _features(height: int, width: int) -> dict[str, dict[str, Any]]:
    state_names = (
        *ARM_JOINTS[:6],
        "left_gripper_open_state",
        *ARM_JOINTS[6:],
        "right_gripper_open_state",
    )
    action_names = tuple(
        name if name.endswith("_cmd") else f"{name}_target"
        for name in (
            *ARM_JOINTS[:6],
            "left_gripper_open_cmd",
            *ARM_JOINTS[6:],
            "right_gripper_open_cmd",
        )
    )
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": state_names,
        },
        "action": {"dtype": "float32", "shape": (14,), "names": action_names},
    }
    features.update(
        {
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ("height", "width", "channels"),
            }
            for camera in CAMERA_NAMES
        }
    )
    return features


def _validate_output(root: Path, *, repo_id: str, episode_lengths: list[int]) -> None:
    frames = sum(episode_lengths)
    episodes = len(episode_lengths)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    actual = {
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
    }
    expected = {
        "codebase_version": "v2.1",
        "total_episodes": episodes,
        "total_frames": frames,
    }
    if actual != expected:
        raise RuntimeError(f"rollout LeRobot counts differ: {actual} != {expected}")
    for camera in CAMERA_NAMES:
        key = f"observation.images.{camera}"
        files = sorted((root / "videos").rglob(f"{key}/episode_*.mp4"))
        if len(files) != episodes:
            raise RuntimeError(f"rollout camera video is missing: {camera}")
        for path, expected_frames in zip(files, episode_lengths, strict=True):
            _validate_video(path, expected_frames)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if len(LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)) != frames:
        raise RuntimeError("pinned LeRobot could not reload the rollout dataset")


def _write_sidecar(
    root: Path,
    *,
    frames: int,
    scenario: dict[str, Any],
    trajectory: str,
    policy: dict[str, Any],
    training_eligible: bool,
    source_kind: str,
    episode_lengths: list[int],
) -> None:
    episodes = len(episode_lengths)
    payload = {
        "schema": EXPORT_SCHEMA,
        "format_version": EXPORT_FORMAT_VERSION,
        "export_status": "complete",
        "training_eligible": training_eligible,
        "selection": {"rule": "full_rollout"},
        "source": {
            "kind": source_kind,
            "trajectory": trajectory,
            "output_format": "LeRobot v2.1",
        },
        "scenario": scenario,
        "policy": policy,
        "counts": {"output_episodes": episodes, "output_frames": frames},
        "episode_sets": {
            "good": list(range(episodes)),
            "full_trajectory": list(range(episodes)),
            "partial_trajectory": [],
            "annotation_present": [],
            "review_complete": [],
            "default_good": list(range(episodes)),
        },
        "episodes": [
            {
                "episode_index": index,
                "length": length,
                "source_episode_complete": True,
                "full_trajectory": True,
            }
            for index, length in enumerate(episode_lengths)
        ],
    }
    path = root / "meta" / "clothdojo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
