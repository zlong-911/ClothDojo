"""One-way Newton integration for policy rollouts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.rollout.engine import PolicyOutput, RTCContext
from clothdojo.rollout.scenario import RolloutScenario
from clothdojo.trajectory import SceneMetadata, TrajectoryWriter
from clothdojo.trajectory.newton import capture_actions, capture_states, gripper_openness

POLICY_ACTION_DIM = 14


def pack_policy_state(arm_q: np.ndarray, gripper_open: np.ndarray) -> np.ndarray:
    """Pack ``left arm, grip, right arm, grip`` in the training-data order."""

    arms = np.asarray(arm_q, dtype=np.float32)
    grippers = np.asarray(gripper_open, dtype=np.float32)
    if arms.ndim != 2 or arms.shape[1] != 12:
        raise ValueError(f"arm state must have shape (batch, 12), got {arms.shape}")
    if grippers.shape != (arms.shape[0], 2):
        raise ValueError(
            f"gripper state must have shape {(arms.shape[0], 2)}, got {grippers.shape}"
        )
    return np.concatenate(
        (arms[:, :6], grippers[:, :1], arms[:, 6:], grippers[:, 1:]),
        axis=1,
        dtype=np.float32,
    )


def unpack_policy_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (POLICY_ACTION_DIM,) or not np.isfinite(values).all():
        raise ValueError(f"policy action must be finite float32[14], got {values.shape}")
    arms = np.concatenate((values[:6], values[7:13])).reshape(1, 12)
    grippers = values[[6, 13]].reshape(1, 2)
    return np.ascontiguousarray(arms), np.ascontiguousarray(grippers)


class HoldChunkPolicy:
    """Deterministic acceptance policy; never a training-data policy."""

    def __init__(self, action_horizon: int):
        self.action_horizon = int(action_horizon)
        self.seed: int | None = None

    def reset(self, seed: int) -> None:
        self.seed = int(seed)

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        *,
        rtc: RTCContext | None = None,
    ) -> PolicyOutput:
        del prompt, rtc
        if self.seed is None:
            raise RuntimeError("hold policy must be reset before inference")
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (1, POLICY_ACTION_DIM):
            raise ValueError(f"hold policy state must have shape (1, 14), got {state.shape}")
        actions = np.repeat(state[:, None, :], self.action_horizon, axis=1)
        return PolicyOutput(actions=actions, model_actions=actions.copy())


class NewtonTrajectoryEpisode:
    """Expose one Newton world while recording exact transitions."""

    action_dim = POLICY_ACTION_DIM

    def __init__(
        self,
        runtime: Any,
        output: str | Path,
        scenario: RolloutScenario,
        *,
        assets_root: str | Path,
        observation_provider=None,
        frame_sink=None,
        trajectory_metadata: dict[str, Any] | None = None,
    ) -> None:
        if int(runtime.model.world_count) != 1:
            raise ValueError("the first RTC trajectory slice requires exactly one world")
        self.runtime = runtime
        self.scenario = scenario
        self.observation_provider = observation_provider
        self.frame_sink = frame_sink
        self._cached_observation: dict[str, np.ndarray] | None = None
        self._arm_lower = np.asarray(
            runtime.scene.arms.get_attribute("joint_limit_lower", runtime.model).numpy()[:, 0],
            dtype=np.float32,
        )
        self._arm_upper = np.asarray(
            runtime.scene.arms.get_attribute("joint_limit_upper", runtime.model).numpy()[:, 0],
            dtype=np.float32,
        )
        assets = Path(assets_root).expanduser().resolve()
        garment = (
            assets / "garments" / "converted" / scenario.garment_id / f"{scenario.garment_id}.usdc"
        )
        robot = assets / "robots" / "piper" / "piper.usda"
        metadata = {
            "producer": "clothdojo_rollout",
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "prompt": scenario.prompt,
            "training_eligible": True,
        }
        if trajectory_metadata is not None:
            metadata.update(trajectory_metadata)
        self.writer = TrajectoryWriter(
            output,
            control_fps=runtime.spec.control_fps,
            sim_substeps=runtime.spec.sim_substeps,
            max_actions=scenario.steps,
            scene=SceneMetadata(
                task_name=scenario.task_name,
                robot_name="piper",
                asset_fingerprints={
                    "garment_usdc": _sha256(garment),
                    "piper_entry_usda": _sha256(robot),
                },
                config_fingerprints={"rtc_scenario": scenario.fingerprint},
            ),
            metadata=metadata,
        )
        try:
            initial = capture_states(runtime, 0)
            if len(initial) != 1:
                raise RuntimeError("Newton state capture did not return one world")
            self.writer.write_initial_state(initial[0])
        except Exception:
            self.writer.abort()
            raise
        self._sample_index = 0

    def observe(self) -> dict[str, np.ndarray]:
        if self._cached_observation is not None:
            return {key: value.copy() for key, value in self._cached_observation.items()}
        scene = self.runtime.scene
        state = self.runtime.state_0
        arms = scene.arms.get_attribute("joint_q", state).numpy()[:, 0]
        gripper_q = scene.grippers.get_attribute("joint_q", state).numpy()[:, 0]
        lower = scene.grippers.get_attribute("joint_limit_lower", self.runtime.model).numpy()[:, 0]
        upper = scene.grippers.get_attribute("joint_limit_upper", self.runtime.model).numpy()[:, 0]
        observation = {
            "observation.state": pack_policy_state(arms, gripper_openness(gripper_q, lower, upper))
        }
        if self.observation_provider is not None:
            images = self.observation_provider(self.runtime)
            overlap = set(observation) & set(images)
            if overlap:
                raise ValueError(f"observation provider replaced reserved keys: {overlap}")
            observation.update(images)
        self._cached_observation = observation
        return {key: value.copy() for key, value in observation.items()}

    def step(self, policy_action: np.ndarray) -> None:
        import warp as wp

        observation = self.observe()
        if self.frame_sink is not None:
            self.frame_sink.add_frame(observation, policy_action)
        arms, grippers = unpack_policy_action(policy_action)
        # Record the command that Newton actually executes.  The runtime clips
        # the same physical limits on device; doing it here keeps trajectory
        # actions and physical transitions exactly aligned.
        arms = np.ascontiguousarray(np.clip(arms, self._arm_lower, self._arm_upper))
        grippers = np.ascontiguousarray(np.clip(grippers, 0.0, 1.0))
        arm_device = wp.array(arms, dtype=wp.float32, device=self.runtime.model.device)
        gripper_device = wp.array(grippers, dtype=wp.float32, device=self.runtime.model.device)
        actions = capture_actions(
            self.runtime,
            self._sample_index,
            arm_device,
            gripper_device,
        )
        self.runtime.step(arm_device, gripper_device)
        self._sample_index += 1
        self._cached_observation = None
        states = capture_states(self.runtime, self._sample_index)
        self.writer.append_transition(actions[0], states[0])

    def close(self) -> None:
        # Publish the trajectory first: the LeRobot sidecar points at it and
        # must never claim a complete paired dataset while that source is
        # still in the recording state.
        self.writer.close()
        if self.frame_sink is not None:
            self.frame_sink.close()

    def abort(self) -> None:
        if self.frame_sink is not None:
            self.frame_sink.abort()
        self.writer.abort()

    def __enter__(self) -> NewtonTrajectoryEpisode:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
