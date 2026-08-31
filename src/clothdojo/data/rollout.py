"""Transactional LeRobot publication and canonical rollout replay."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.data.annotations import ExportPlan, build_good_segment_plan
from clothdojo.data.lerobot import (
    EXPORT_FORMAT_VERSION,
    EXPORT_SCHEMA,
    _create_writer,
)
from clothdojo.rendering import DATASET_CAMERA_NAMES
from clothdojo.robots import RobotProfile, robot_profile
from clothdojo.sim.scene import ClothVisual

CAMERA_NAMES = DATASET_CAMERA_NAMES
ARM_JOINTS = tuple(f"{side}/joint{joint}" for side in ("left", "right") for joint in range(1, 7))


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
        selection_plan: ExportPlan | None = None,
        robot_type: str = "piper",
        arm_joint_names: tuple[str, ...] = ARM_JOINTS,
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
        self.action_dim = len(arm_joint_names) + 2
        self.selection_plan = selection_plan
        self._injected = _writer_factory is not None
        if self._injected:
            self.staging.mkdir()
        factory = _writer_factory or _create_writer
        self.writer = factory(
            repo_id=self.repo_id,
            root=self.staging,
            fps=self.fps,
            robot_type=robot_type,
            features=_features(
                self.image_height,
                self.image_width,
                arm_joint_names=arm_joint_names,
            ),
            image_writer_threads=image_writer_threads,
        )
        self._episode_lengths: list[int] = []
        self.frames = 0
        self._episode_frames = 0
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
        if state.shape != (1, self.action_dim) or action.shape != (self.action_dim,):
            raise ValueError(
                "rollout state/action must match the robot profile, got "
                f"{state.shape}/{action.shape}"
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
            selection_plan=self.selection_plan,
        )
        self.staging.rename(self.output)
        self._closed = True

    def abort(self) -> None:
        self._closed = True


def inspect_rollout_collection_export(
    source: str | Path,
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
    cloth_visual: ClothVisual = "nocs",
) -> dict[str, Any]:
    """Inspect canonical collection metadata without rendering or writing data."""

    root = Path(source).expanduser().resolve()
    manifest, source_kind = _load_canonical_collection(root)
    manifest = _select_trajectory_shard(
        manifest,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    frames = sum(int(item["action_count"]) for item in manifest["trajectories"])
    result = {
        "source_kind": source_kind,
        "source": str(root),
        "output_format": "LeRobot v2.1",
        "output_episodes": len(manifest["trajectories"]),
        "output_frames": frames,
        "camera_names": list(CAMERA_NAMES),
        "renderer": "newton",
        "cloth_visual": cloth_visual,
        "training_eligible": manifest.get("training_eligible") is True,
    }
    if shard_count is not None:
        result["trajectory_shard"] = {
            "index": shard_index,
            "count": shard_count,
        }
    return result


def export_rollout_collection(
    source: str | Path,
    output: str | Path,
    *,
    assets_root: str | Path = "assets",
    device: str = "cuda:0",
    repo_id: str | None = None,
    image_writer_threads: int = 6,
    shard_index: int | None = None,
    shard_count: int | None = None,
    cloth_visual: ClothVisual = "nocs",
) -> dict[str, Any]:
    """Replay canonical rollout states and publish one multi-episode LeRobot dataset."""

    def camera_factory(
        asset_id: str,
        _initial_state: Any,
        profile: RobotProfile,
    ):
        return _NewtonTrajectoryCamera(
            asset_id,
            assets_root=assets_root,
            device=device,
            profile=profile,
            cloth_visual=cloth_visual,
        )

    return _export_canonical_collection(
        source,
        output,
        camera_factory=camera_factory,
        rendering={"backend": "newton", "cloth_visual": cloth_visual},
        repo_id=repo_id,
        image_writer_threads=image_writer_threads,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def inspect_trajectory_dataset_export(
    source: str | Path,
    *,
    min_segment_frames: int,
    whole_trajectory_good_only: bool,
    shard_index: int | None = None,
    shard_count: int | None = None,
    cloth_visual: ClothVisual = "nocs",
) -> dict[str, Any]:
    """Inspect annotation selection directly on canonical trajectories."""

    root = Path(source).expanduser().resolve()
    plan = _trajectory_export_plan(
        root,
        min_segment_frames=min_segment_frames,
        whole_trajectory_good_only=whole_trajectory_good_only,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    fps, task_name, _assets = _trajectory_export_context(root, plan)
    result = plan.to_json_dict()
    result.pop("episodes")
    result.pop("source_action_counts")
    result.update(
        {
            "source_kind": "trajectory_collection",
            "source": str(root),
            "task": task_name,
            "fps": fps,
            "output_format": "LeRobot v2.1",
            "camera_names": list(CAMERA_NAMES),
            "renderer": "newton",
            "cloth_visual": cloth_visual,
        }
    )
    if shard_count is not None:
        result["trajectory_shard"] = {
            "index": shard_index,
            "count": shard_count,
        }
    return result


def export_trajectory_dataset(
    source: str | Path,
    output: str | Path,
    *,
    assets_root: str | Path = "assets",
    device: str = "cuda:0",
    repo_id: str | None = None,
    image_writer_threads: int = 6,
    min_segment_frames: int,
    whole_trajectory_good_only: bool,
    shard_index: int | None = None,
    shard_count: int | None = None,
    cloth_visual: ClothVisual = "nocs",
) -> dict[str, Any]:
    """Replay selected canonical trajectory ranges into standard LeRobot v2.1."""

    root = Path(source).expanduser().resolve()
    plan = _trajectory_export_plan(
        root,
        min_segment_frames=min_segment_frames,
        whole_trajectory_good_only=whole_trajectory_good_only,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    fps, task_name, assets = _trajectory_export_context(root, plan)
    if shard_count is not None:
        plan = replace(
            plan,
            segments=tuple(
                sorted(
                    plan.segments,
                    key=lambda segment: (
                        assets[segment.source_trajectory],
                        segment.source_trajectory,
                        segment.action_start,
                    ),
                )
            ),
        )
    prompt = _task_prompt(task_name)
    trajectories = [
        {
            "path": f"trajectories/{segment.source_trajectory}",
            "asset_id": assets[segment.source_trajectory],
            "action_count": segment.length,
            "action_start": segment.action_start,
            "action_end_exclusive": segment.action_end_exclusive,
            "source_episode_id": segment.source_trajectory,
        }
        for segment in plan.segments
    ]
    manifest = {
        "scenario": {"scenario_id": task_name, "prompt": prompt},
        "policy": {},
        "training_eligible": True,
        "trajectories": trajectories,
    }

    def camera_factory(
        asset_id: str,
        _initial_state: Any,
        profile: RobotProfile,
    ):
        return _NewtonTrajectoryCamera(
            asset_id,
            assets_root=assets_root,
            device=device,
            profile=profile,
            cloth_visual=cloth_visual,
        )

    _export_collection_manifest(
        root,
        output,
        manifest=manifest,
        source_kind="trajectory_collection",
        camera_factory=camera_factory,
        rendering={
            "backend": "newton",
            "cloth_visual": cloth_visual,
            **(
                {"trajectory_shard": {"index": shard_index, "count": shard_count}}
                if shard_count is not None
                else {}
            ),
        },
        repo_id=repo_id,
        image_writer_threads=image_writer_threads,
        selection_plan=plan,
        fps=fps,
    )
    result = inspect_trajectory_dataset_export(
        root,
        min_segment_frames=min_segment_frames,
        whole_trajectory_good_only=whole_trajectory_good_only,
        shard_index=shard_index,
        shard_count=shard_count,
        cloth_visual=cloth_visual,
    )
    result.update({"compatible": True, "output": str(Path(output).expanduser().resolve())})
    return result


def _trajectory_export_plan(
    root: Path,
    *,
    min_segment_frames: int,
    whole_trajectory_good_only: bool,
    shard_index: int | None,
    shard_count: int | None,
) -> ExportPlan:
    selected_source_ids = _trajectory_shard_source_ids(
        root,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return build_good_segment_plan(
        root,
        min_segment_frames=min_segment_frames,
        whole_trajectory_good_only=whole_trajectory_good_only,
        source_ids=selected_source_ids,
    )


def _trajectory_shard_source_ids(
    root: Path,
    *,
    shard_index: int | None,
    shard_count: int | None,
) -> frozenset[str] | None:
    if shard_index is None and shard_count is None:
        return None
    if shard_index is None or shard_count is None:
        raise ValueError("shard_index and shard_count must be provided together")
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")

    source_records: list[tuple[str, str, str, int]] = []
    pair_assets: dict[str, str] = {}
    for manifest_path in sorted((root / "trajectories").glob("*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        metadata = payload.get("metadata")
        asset_id = metadata.get("asset_id") if isinstance(metadata, dict) else None
        if not isinstance(asset_id, str) or not asset_id:
            raise RuntimeError(f"trajectory asset_id is missing: {manifest_path.parent.name}")
        action_count = payload.get("action_count", 1)
        if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 1:
            raise RuntimeError(f"trajectory action_count is invalid: {manifest_path.parent.name}")
        source_id = manifest_path.parent.name
        pair_id = source_id.removesuffix("__mirror")
        previous_asset = pair_assets.setdefault(pair_id, asset_id)
        if previous_asset != asset_id:
            raise RuntimeError(f"mirror pair asset_id differs: {pair_id}")
        source_records.append((source_id, pair_id, asset_id, action_count))
    selected_pairs = _balanced_group_shards(
        [
            (pair_id, action_count)
            for _source_id, pair_id, _asset_id, action_count in source_records
        ],
        shard_count,
    )[shard_index]
    selected = frozenset(
        source_id
        for source_id, pair_id, _asset_id, _action_count in source_records
        if pair_id in selected_pairs
    )
    if not selected:
        raise ValueError(f"trajectory shard {shard_index}/{shard_count} is empty")
    return selected


class _NewtonTrajectoryCamera:
    """Render recorded states through the project's Newton camera backend."""

    def __init__(
        self,
        asset_id: str,
        *,
        assets_root: str | Path,
        device: str,
        profile: RobotProfile,
        cloth_visual: ClothVisual,
    ) -> None:
        from clothdojo.rollout.render import ShirtFlattenCameraObservation
        from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

        self._scene = build_scene(
            assets_root,
            spec=ShirtFlattenSpec(garment_id=asset_id, world_count=1),
            device=device,
            visuals=True,
            cloth_visual=cloth_visual,
            robot_id=profile.robot_id,
        )
        self._runtime = _TrajectoryRenderState(self._scene)
        self._camera = ShirtFlattenCameraObservation(
            self._scene,
            camera_names=DATASET_CAMERA_NAMES,
        )

    def __call__(self, state) -> dict[str, np.ndarray]:
        _restore_render_state(self._runtime, state)
        return self._camera(self._runtime)

    def set_cloth_surface_winding_reversed(self, reversed_: bool) -> None:
        self._camera.set_cloth_surface_winding_reversed(reversed_)

    def close(self) -> None:
        if hasattr(self._camera, "close"):
            self._camera.close()


def _export_canonical_collection(
    source: str | Path,
    output: str | Path,
    *,
    camera_factory: Callable[[str, Any, RobotProfile], Any],
    rendering: dict[str, Any],
    repo_id: str | None,
    image_writer_threads: int,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict[str, Any]:
    """Publish one canonical collection using a backend-specific camera."""

    root = Path(source).expanduser().resolve()
    manifest, source_kind = _load_canonical_collection(root)
    manifest = _select_trajectory_shard(
        manifest,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    rendering = dict(rendering)
    if shard_count is not None:
        rendering["trajectory_shard"] = {
            "index": shard_index,
            "count": shard_count,
        }
    _export_collection_manifest(
        root,
        output,
        manifest=manifest,
        source_kind=source_kind,
        camera_factory=camera_factory,
        rendering=rendering,
        repo_id=repo_id,
        image_writer_threads=image_writer_threads,
    )
    summary = inspect_rollout_collection_export(
        root,
        shard_index=shard_index,
        shard_count=shard_count,
        cloth_visual=rendering.get("cloth_visual", "nocs"),
    )
    summary.update(
        {
            "compatible": True,
            "output": str(Path(output).expanduser().resolve()),
            "renderer": rendering["backend"],
        }
    )
    return summary


def _select_trajectory_shard(
    manifest: dict[str, Any],
    *,
    shard_index: int | None,
    shard_count: int | None,
) -> dict[str, Any]:
    """Select a deterministic shard without splitting one asset across workers."""

    if shard_index is None and shard_count is None:
        return manifest
    if shard_index is None or shard_count is None:
        raise ValueError("shard_index and shard_count must be provided together")
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    trajectories = manifest["trajectories"]
    weighted_assets = []
    for episode in trajectories:
        start = int(episode.get("action_start", 0))
        end = int(episode.get("action_end_exclusive", episode.get("action_count", start + 1)))
        weighted_assets.append((episode["asset_id"], max(1, end - start)))
    selected_assets = _balanced_group_shards(weighted_assets, shard_count)[shard_index]
    trajectories = [
        episode for episode in trajectories if episode["asset_id"] in selected_assets
    ]
    if not trajectories:
        raise ValueError(
            f"trajectory shard {shard_index}/{shard_count} is empty"
        )
    selected = dict(manifest)
    selected["trajectories"] = trajectories
    return selected


def _balanced_group_shards(
    weighted_groups: list[tuple[str, int]], shard_count: int
) -> list[set[str]]:
    """Assign whole groups by descending work to the currently lightest shard."""

    group_weights: dict[str, int] = {}
    group_order: dict[str, int] = {}
    for group_id, weight in weighted_groups:
        group_order.setdefault(group_id, len(group_order))
        group_weights[group_id] = group_weights.get(group_id, 0) + weight
    ordered_groups = sorted(
        group_weights,
        key=lambda group_id: (-group_weights[group_id], group_order[group_id]),
    )
    shards = [set() for _ in range(shard_count)]
    loads = [0] * shard_count
    for group_id in ordered_groups:
        index = min(range(shard_count), key=lambda value: (loads[value], value))
        shards[index].add(group_id)
        loads[index] += group_weights[group_id]
    return shards


def _export_collection_manifest(
    root: Path,
    output: str | Path,
    *,
    manifest: dict[str, Any],
    source_kind: str,
    camera_factory: Callable[[str, Any, RobotProfile], Any],
    rendering: dict[str, Any],
    repo_id: str | None,
    image_writer_threads: int,
    selection_plan: ExportPlan | None = None,
    fps: float | None = None,
) -> None:
    """Replay the ranges in one in-memory canonical collection manifest."""

    from clothdojo.rollout.newton import pack_policy_state
    from clothdojo.trajectory import TrajectoryReader

    scenario = manifest["scenario"]
    trajectories = manifest["trajectories"]
    first_reader = TrajectoryReader(root / trajectories[0]["path"])
    collection_robot = _trajectory_robot(first_reader)
    missing_cameras = set(CAMERA_NAMES) - {camera.name for camera in collection_robot.cameras}
    if missing_cameras:
        raise ValueError(
            f"robot {collection_robot.robot_id} cannot export the four-camera dataset: "
            f"missing {', '.join(sorted(missing_cameras))}; native cameras are not duplicated"
        )
    export_fps = float(first_reader.manifest.control_fps) if fps is None else float(fps)
    sink = RolloutLeRobotWriter(
        output,
        fps=int(export_fps),
        prompt=scenario["prompt"],
        scenario=scenario,
        trajectory=str(root),
        policy=manifest["policy"],
        training_eligible=manifest.get("training_eligible") is True,
        source_kind=source_kind,
        rendering=rendering,
        repo_id=repo_id,
        image_writer_threads=image_writer_threads,
        selection_plan=selection_plan,
        robot_type=collection_robot.robot_id,
        arm_joint_names=collection_robot.arm_joint_names,
    )
    try:
        active_asset = None
        camera = None
        for episode_index, episode in enumerate(trajectories):
            reader = TrajectoryReader(root / episode["path"])
            if float(reader.manifest.control_fps) != export_fps:
                raise ValueError("trajectories use different control rates")
            profile = _trajectory_robot(reader)
            if profile.robot_id != collection_robot.robot_id:
                raise ValueError("one export cannot mix robot profiles")
            action_start = int(episode.get("action_start", 0))
            action_end = int(
                episode.get("action_end_exclusive", reader.manifest.action_count)
            )
            if episode["asset_id"] != active_asset:
                _close_camera(camera)
                camera = None
                active_asset = episode["asset_id"]
                camera = camera_factory(
                    active_asset,
                    reader.state_at(action_start),
                    profile,
                )
            _set_camera_cloth_surface_winding_reversed(
                camera,
                _trajectory_surface_winding_reversed(reader),
            )
            if hasattr(camera, "begin_episode"):
                camera.begin_episode(
                    episode_index,
                    source_episode_id=episode.get(
                        "source_episode_id", Path(episode["path"]).name
                    ),
                )
            for sample in reader.iter_samples(
                action_start=action_start,
                action_end_exclusive=action_end,
            ):
                arm_q = np.asarray(sample.state.joint_q, dtype=np.float32)[
                    np.asarray(profile.arm_coord_indices)
                ][None, :]
                observation = {
                    "observation.state": pack_policy_state(
                        arm_q,
                        sample.state.gripper_open_state.reshape(1, 2),
                    ),
                    **camera(sample.state),
                }
                policy_action = pack_policy_state(
                    sample.action.joint_target.reshape(1, -1),
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

def _close_camera(camera: Any | None) -> None:
    if camera is not None and hasattr(camera, "close"):
        camera.close()


def _trajectory_surface_winding_reversed(reader: Any) -> bool:
    metadata = getattr(reader.manifest, "metadata", {})
    augmentation = metadata.get("augmentation") if isinstance(metadata, dict) else None
    if not isinstance(augmentation, dict):
        return False
    reversed_ = augmentation.get("surface_winding_reversed", False)
    if not isinstance(reversed_, bool):
        raise ValueError("trajectory surface_winding_reversed must be bool")
    return reversed_


def _set_camera_cloth_surface_winding_reversed(camera: Any, reversed_: bool) -> None:
    setter = getattr(camera, "set_cloth_surface_winding_reversed", None)
    if setter is not None:
        setter(reversed_)
    elif reversed_:
        raise RuntimeError("trajectory renderer cannot reverse cloth surface winding")


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
    if schema == "clothdojo_retarget_collection":
        from clothdojo.data.retarget import load_retarget_collection

        return load_retarget_collection(root), "retarget_collection"
    raise ValueError(f"unsupported canonical trajectory collection schema: {schema!r}")


def _trajectory_robot(reader) -> RobotProfile:
    return robot_profile(reader.manifest.scene.robot_id)


def _trajectory_export_context(
    root: Path,
    plan: ExportPlan,
) -> tuple[float, str, dict[str, str]]:
    from clothdojo.trajectory import TrajectoryReader

    fps: float | None = None
    task_name: str | None = None
    assets: dict[str, str] = {}
    for source_id, _action_count in plan.source_action_counts:
        reader = TrajectoryReader(root / "trajectories" / source_id)
        current_fps = float(reader.manifest.control_fps)
        current_task = reader.manifest.scene.task_name
        asset_id = reader.manifest.metadata.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise RuntimeError(f"trajectory asset_id is missing: {source_id}")
        if fps is None:
            fps = current_fps
            task_name = current_task
        elif current_fps != fps or current_task != task_name:
            raise ValueError("one export cannot mix trajectory tasks or control rates")
        assets[source_id] = asset_id
    assert fps is not None and task_name is not None
    return fps, task_name, assets


def _task_prompt(task_name: str) -> str:
    prompts = {
        "shirt_flatten": "Flatten the shirt.",
        "shirt_fold": "Fold the shirt.",
    }
    try:
        return prompts[task_name]
    except KeyError as exc:
        raise ValueError(f"no training prompt is defined for task: {task_name}") from exc


def _restore_render_state(runtime, state) -> None:
    import newton

    if int(runtime.scene.model.world_count) != 1:
        raise ValueError("single-state replay requires one Newton world")
    world_offsets = np.asarray(runtime.scene.world_offsets, dtype=np.float32)
    if world_offsets.shape != (1, 3):
        raise RuntimeError("render world offset does not match one Newton world")
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        target = getattr(runtime.state_0, name)
        values = np.asarray(getattr(state, name), dtype=np.float32)
        if name == "body_q":
            values = values.copy()
            values[:, :3] += world_offsets[0]
        _assign_render_array(target, values, name=name)
    particle_q = np.asarray(state.particle_q, dtype=np.float32) + world_offsets[0]
    particle_qd = np.asarray(state.particle_qd, dtype=np.float32)
    _assign_render_array(runtime.state_0.particle_q, particle_q, name="particle_q")
    _assign_render_array(runtime.state_0.particle_qd, particle_qd, name="particle_qd")
    # Match the legacy EpisodeV1 exporter: recorded generalized coordinates
    # own articulated rendering, while body_q remains authoritative for every
    # non-articulated body.  Persisted body arrays predate a guaranteed
    # importer-stable articulation layout.
    newton.eval_fk(
        runtime.scene.model,
        runtime.state_0.joint_q,
        runtime.state_0.joint_qd,
        runtime.state_0,
    )


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


def _features(
    height: int,
    width: int,
    *,
    arm_joint_names: tuple[str, ...] = ARM_JOINTS,
) -> dict[str, dict[str, Any]]:
    if not arm_joint_names or len(arm_joint_names) % 2:
        raise ValueError("LeRobot features require equal non-empty left/right arms")
    side = len(arm_joint_names) // 2
    state_names = (
        *arm_joint_names[:side],
        "left_gripper_open_state",
        *arm_joint_names[side:],
        "right_gripper_open_state",
    )
    action_names = tuple(
        name if name.endswith("_cmd") else f"{name}_target"
        for name in (
            *arm_joint_names[:side],
            "left_gripper_open_cmd",
            *arm_joint_names[side:],
            "right_gripper_open_cmd",
        )
    )
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(arm_joint_names) + 2,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(arm_joint_names) + 2,),
            "names": action_names,
        },
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
        if any(path.stat().st_size == 0 for path in files):
            raise RuntimeError(f"rollout camera video is empty: {camera}")
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
    selection_plan: ExportPlan | None,
) -> None:
    episodes = len(episode_lengths)
    if selection_plan is not None:
        if episode_lengths != [segment.length for segment in selection_plan.segments]:
            raise RuntimeError("exported episode lengths differ from the annotation plan")
        _write_selected_sidecar(
            root,
            source_root=Path(trajectory),
            plan=selection_plan,
            scenario=scenario,
            rendering=rendering,
        )
        return
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


def _write_selected_sidecar(
    root: Path,
    *,
    source_root: Path,
    plan: ExportPlan,
    scenario: dict[str, Any],
    rendering: dict[str, Any],
) -> None:
    episodes = []
    full = []
    partial = []
    annotation_present = []
    review_complete = []
    default_good = []
    for index, segment in enumerate(plan.segments):
        record = {
            **segment.to_json_dict(episode_index=index),
            "quality": "good",
            "source_episode_complete": True,
            "contains_source_terminal": (
                segment.action_end_exclusive == segment.source_action_count
            ),
        }
        episodes.append(record)
        (full if segment.full_trajectory else partial).append(index)
        (annotation_present if segment.annotation_present else default_good).append(index)
        if segment.review_status == "complete":
            review_complete.append(index)
    payload = {
        "schema": EXPORT_SCHEMA,
        "format_version": EXPORT_FORMAT_VERSION,
        "export_status": "complete",
        "training_eligible": True,
        "selection": {
            "rule": (
                "explicit_whole_trajectory_good"
                if plan.whole_trajectory_good_only
                else "bad_complement_with_minimum_length"
            ),
            "min_segment_frames": plan.min_segment_frames,
        },
        "source": {
            "kind": "trajectory_collection",
            "trajectory": str(source_root),
            "output_format": "LeRobot v2.1",
        },
        "scenario": scenario,
        "rendering": rendering,
        "counts": {
            key: value
            for key, value in plan.to_json_dict().items()
            if key
            not in {
                "episodes",
                "source_action_counts",
                "dropped_short_source_ids",
                "dropped_short_ranges",
            }
        },
        "episode_sets": {
            "good": list(range(plan.output_episodes)),
            "full_trajectory": full,
            "partial_trajectory": partial,
            "annotation_present": annotation_present,
            "review_complete": review_complete,
            "default_good": default_good,
        },
        "episodes": episodes,
    }
    path = root / "meta" / "clothdojo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
