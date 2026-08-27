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
from clothdojo.rendering import DATASET_CAMERA_NAMES

CAMERA_NAMES = DATASET_CAMERA_NAMES
ARM_JOINTS = tuple(f"{side}/joint{joint}" for side in ("left", "right") for joint in range(1, 7))
_ARM_COORD_INDICES = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13)


class _TrajectoryRenderState:
    """Minimal state owner for trajectory replay; no solver or contacts are built."""

    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.state_0 = scene.model.state()


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
        rendering: dict[str, Any] | None = None,
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
        self.rendering = dict(rendering or {"backend": "newton"})
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
            rendering=self.rendering,
        )
        self.staging.rename(self.output)
        self._closed = True

    def abort(self) -> None:
        self._closed = True


def inspect_rollout_collection_export(source: str | Path) -> dict[str, Any]:
    """Inspect canonical collection metadata without rendering or writing data."""

    root = Path(source).expanduser().resolve()
    manifest, source_kind = _load_canonical_collection(root)
    frames = sum(int(item["action_count"]) for item in manifest["trajectories"])
    return {
        "source_kind": source_kind,
        "source": str(root),
        "output_format": "LeRobot v2.1",
        "output_episodes": len(manifest["trajectories"]),
        "output_frames": frames,
        "camera_names": list(CAMERA_NAMES),
        "renderer": "newton",
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

    def camera_factory(asset_id: str, _initial_state: Any):
        return _NewtonTrajectoryCamera(
            asset_id,
            assets_root=assets_root,
            device=device,
        )

    return _export_canonical_collection(
        source,
        output,
        camera_factory=camera_factory,
        rendering={"backend": "newton"},
        repo_id=repo_id,
        image_writer_threads=image_writer_threads,
    )


class _NewtonTrajectoryCamera:
    """Render recorded states through the project's Newton camera backend."""

    def __init__(
        self,
        asset_id: str,
        *,
        assets_root: str | Path,
        device: str,
    ) -> None:
        from clothdojo.rollout.render import ShirtFlattenCameraObservation
        from clothdojo.sim.scene import ShirtFlattenSpec, build_shirt_flatten

        self._scene = build_shirt_flatten(
            assets_root,
            spec=ShirtFlattenSpec(garment_id=asset_id, world_count=1),
            device=device,
            visuals=True,
        )
        self._runtime = _TrajectoryRenderState(self._scene)
        self._camera = ShirtFlattenCameraObservation(
            self._scene,
            camera_names=DATASET_CAMERA_NAMES,
        )

    def __call__(self, state) -> dict[str, np.ndarray]:
        _restore_render_state(self._runtime, state)
        return self._camera(self._runtime)

    def close(self) -> None:
        if hasattr(self._camera, "close"):
            self._camera.close()


def _export_canonical_collection(
    source: str | Path,
    output: str | Path,
    *,
    camera_factory: Callable[[str, Any], Any],
    rendering: dict[str, Any],
    repo_id: str | None,
    image_writer_threads: int,
) -> dict[str, Any]:
    """Publish one canonical collection using a backend-specific camera."""

    from clothdojo.rollout.newton import pack_policy_state
    from clothdojo.trajectory import TrajectoryReader

    root = Path(source).expanduser().resolve()
    manifest, source_kind = _load_canonical_collection(root)
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
        source_kind=source_kind,
        rendering=rendering,
        repo_id=repo_id,
        image_writer_threads=image_writer_threads,
    )
    try:
        active_asset = None
        camera = None
        for episode_index, episode in enumerate(trajectories):
            reader = TrajectoryReader(root / episode["path"])
            if reader.manifest.control_fps != first_reader.manifest.control_fps:
                raise ValueError("rollout trajectories use different control rates")
            if episode["asset_id"] != active_asset:
                _close_camera(camera)
                camera = None
                active_asset = episode["asset_id"]
                camera = camera_factory(active_asset, reader.state_at(0))
            if hasattr(camera, "begin_episode"):
                camera.begin_episode(
                    episode_index,
                    source_episode_id=Path(episode["path"]).name,
                )
            for sample in reader.iter_samples():
                arm_q = np.asarray(sample.state.joint_q, dtype=np.float32)[
                    np.asarray(_ARM_COORD_INDICES)
                ][None, :]
                observation = {
                    "observation.state": pack_policy_state(
                        arm_q,
                        sample.state.gripper_open_state.reshape(1, 2),
                    ),
                    **camera(sample.state),
                }
                policy_action = pack_policy_state(
                    sample.action.joint_target.reshape(1, 12),
                    sample.action.gripper_open_command.reshape(1, 2),
                )[0]
                sink.add_frame(observation, policy_action)
            sink.save_episode()
        _close_camera(camera)
        camera = None
        sink.close()
    except BaseException:
        _close_camera(camera)
        sink.abort()
        raise
    summary = inspect_rollout_collection_export(root)
    summary.update(
        {
            "compatible": True,
            "output": str(Path(output).expanduser().resolve()),
            "renderer": rendering["backend"],
        }
    )
    return summary


def _close_camera(camera: Any | None) -> None:
    if camera is not None and hasattr(camera, "close"):
        camera.close()


def _load_canonical_collection(root: Path) -> tuple[dict[str, Any], str]:
    try:
        schema = json.loads((root / "meta.json").read_text(encoding="utf-8")).get("schema")
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read trajectory collection metadata: {root}: {exc}") from exc
    if schema == "clothdojo_rollout":
        from clothdojo.rollout.collection import load_rollout_collection

        return load_rollout_collection(root), "rollout_collection"
    if schema == "clothdojo_evaluation_condition":
        from clothdojo.evaluation.results import load_evaluation_condition

        condition = load_evaluation_condition(root)
        if condition.meta.get("training_eligible") is not False:
            raise ValueError("evaluation condition must be marked training_eligible=false")
        trajectories = condition.meta.get("trajectories")
        if not isinstance(trajectories, list) or len(trajectories) != len(condition.cases):
            raise ValueError("evaluation condition trajectory index is incomplete")
        return condition.meta, "evaluation_condition"
    raise ValueError(f"unsupported canonical trajectory collection schema: {schema!r}")


def _restore_render_state(runtime, state) -> None:
    if int(runtime.scene.model.world_count) != 1:
        raise ValueError("single-state replay requires one Newton world")
    robot = runtime.scene.robot
    world_offsets = np.asarray(runtime.scene.world_offsets, dtype=np.float32)
    if world_offsets.shape != (1, 3):
        raise RuntimeError("render world offsets do not match the Newton world count")
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        target = robot.get_attribute(name, runtime.state_0)
        values = np.asarray(getattr(state, name), dtype=np.float32)[None, None]
        if name == "body_q":
            values = values.copy()
            values[0, 0, :, :3] += world_offsets[0]
        _assign_render_array(target, values, name=name)
        robot.set_attribute(name, runtime.state_0, target)
    particle_q = (
        np.asarray(state.particle_q, dtype=np.float32) + world_offsets[0]
    )
    particle_qd = np.asarray(state.particle_qd, dtype=np.float32)
    _assign_render_array(runtime.state_0.particle_q, particle_q, name="particle_q")
    _assign_render_array(runtime.state_0.particle_qd, particle_qd, name="particle_qd")


def _assign_render_array(target, values: np.ndarray, *, name: str) -> None:
    """Assign NumPy scalar/vector views to a Warp array without device readback."""

    logical_shape = tuple(target.shape)
    actual_shape = tuple(values.shape)
    if actual_shape != logical_shape and not (
        len(actual_shape) == len(logical_shape) + 1
        and actual_shape[:-1] == logical_shape
    ):
        raise ValueError(
            f"render replay {name} shape {actual_shape} is incompatible with "
            f"Warp shape {logical_shape}"
        )
    target.assign(np.ascontiguousarray(values))


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
    rendering: dict[str, Any],
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
        "rendering": rendering,
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
