"""Data model for ClothDojo state/action trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

TRAJECTORY_FORMAT_VERSION = 1
TRAJECTORY_ALIGNMENT = "states_n_plus_1_actions_n"
WORLD_LOCAL_FRAME = "world_local"
RUNTIME_ACTION_SEMANTICS = "runtime_target"
# joint_target is radians after runtime joint-limit clipping; gripper_open_command
# remains normalized openness after runtime clipping to [0, 1].
TrajectoryStatus = Literal["recording", "complete", "aborted"]

SAMPLE_INDEX_DTYPE = np.dtype(np.int64)
STATE_ARRAY_DTYPES = {
    "joint_q": np.dtype(np.float32),
    "joint_qd": np.dtype(np.float32),
    "body_q": np.dtype(np.float32),
    "body_qd": np.dtype(np.float32),
    "particle_q": np.dtype(np.float32),
    "particle_qd": np.dtype(np.float32),
    "gripper_open_state": np.dtype(np.float32),
    "grasp_attached": np.dtype(np.int32),
    "grasp_previous_open": np.dtype(np.float32),
    "grasp_pin_count": np.dtype(np.int32),
    "grasp_pins": np.dtype(np.int32),
    "grasp_pin_local": np.dtype(np.float32),
}
ACTION_ARRAY_DTYPES = {
    "joint_target": np.dtype(np.float32),
    "gripper_open_command": np.dtype(np.float32),
}


@dataclass(frozen=True, slots=True)
class TrajectoryState:
    """One Newton state captured at a control boundary."""

    sample_index: int
    joint_q: np.ndarray
    joint_qd: np.ndarray
    body_q: np.ndarray
    body_qd: np.ndarray
    particle_q: np.ndarray
    particle_qd: np.ndarray
    gripper_open_state: np.ndarray
    grasp_attached: np.ndarray
    grasp_previous_open: np.ndarray
    grasp_pin_count: np.ndarray
    grasp_pins: np.ndarray
    grasp_pin_local: np.ndarray


@dataclass(frozen=True, slots=True)
class TrajectoryAction:
    """The controller command applied to the state with the same index."""

    sample_index: int
    joint_target: np.ndarray
    gripper_open_command: np.ndarray


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    state: TrajectoryState
    action: TrajectoryAction


@dataclass(slots=True)
class SceneMetadata:
    task_name: str = "minimal_box"
    robot_id: str = "none"
    asset_fingerprints: dict[str, str] = field(default_factory=dict)
    config_fingerprints: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> SceneMetadata:
        """Read current metadata and the former task-coupled environment IDs."""

        values = dict(data)
        robot_id = str(values.pop("robot_id", ""))
        robot_name = str(values.pop("robot_name", ""))
        task_name = str(values.pop("task_name", ""))
        environment_id = str(values.pop("environment_id", ""))
        values.pop("environment_fingerprint", None)
        legacy_scenes = {
            "piper_shirt_flatten_v1": ("piper", "shirt_flatten"),
            "aloha2_shirt_flatten_v1": ("aloha2", "shirt_flatten"),
            "x5_shirt_flatten_v1": ("x5", "shirt_flatten"),
            "panda_shirt_flatten_v1": ("panda", "shirt_flatten"),
            "ur5e_2f85_shirt_flatten_v1": ("ur5e_2f85", "shirt_flatten"),
        }
        if environment_id and environment_id not in legacy_scenes:
            raise ValueError(f"unknown legacy trajectory environment: {environment_id!r}")
        legacy_robot, legacy_task = legacy_scenes.get(environment_id, ("", ""))
        candidates = {value for value in (robot_id, robot_name, legacy_robot) if value}
        if len(candidates) > 1:
            raise ValueError("trajectory scene contains conflicting robot identities")
        task_candidates = {value for value in (task_name, legacy_task) if value}
        if len(task_candidates) > 1:
            raise ValueError("trajectory scene contains conflicting task identities")
        values["robot_id"] = next(iter(candidates), "none")
        values["task_name"] = next(iter(task_candidates), "minimal_box")
        return cls(**values)


@dataclass(slots=True)
class TrajectoryManifest:
    format_version: int = TRAJECTORY_FORMAT_VERSION
    alignment: str = TRAJECTORY_ALIGNMENT
    status: TrajectoryStatus = "recording"
    control_fps: float = 30.0
    sim_substeps: int = 20
    units: str = "m"
    coordinate_convention: str = "right_handed_z_up"
    coordinate_frame: str = WORLD_LOCAL_FRAME
    max_actions: int = 3600
    state_count: int = 0
    action_count: int = 0
    state_shapes: dict[str, list[int]] = field(default_factory=dict)
    action_shapes: dict[str, list[int]] = field(default_factory=dict)
    scene: SceneMetadata = field(default_factory=SceneMetadata)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> TrajectoryManifest:
        """Load one manifest in the current trajectory format."""

        return cls(
            format_version=int(data.get("format_version", 0)),
            alignment=str(data.get("alignment", "")),
            status=str(data.get("status", "recording")),
            control_fps=float(data["control_fps"]),
            sim_substeps=int(data["sim_substeps"]),
            units=str(data["units"]),
            coordinate_convention=str(data["coordinate_convention"]),
            coordinate_frame=str(data.get("coordinate_frame", "")),
            max_actions=int(data["max_actions"]),
            state_count=int(data.get("state_count", 0)),
            action_count=int(data.get("action_count", 0)),
            state_shapes={
                key: [int(value) for value in shape]
                for key, shape in data.get("state_shapes", {}).items()
            },
            action_shapes={
                key: [int(value) for value in shape]
                for key, shape in data.get("action_shapes", {}).items()
            },
            scene=SceneMetadata.from_json_dict(data.get("scene", {})),
            metadata=dict(data.get("metadata", {})),
        )
