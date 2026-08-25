"""One-way Newton integration for policy rollouts."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

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


def unpack_policy_actions(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a policy batch into Newton's arm and gripper control arrays."""

    values = np.asarray(actions, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != POLICY_ACTION_DIM
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"policy actions must be finite float32[batch, 14], got {values.shape}"
        )
    arms = np.concatenate((values[:, :6], values[:, 7:13]), axis=1)
    grippers = values[:, [6, 13]]
    return np.ascontiguousarray(arms), np.ascontiguousarray(grippers)


class NewtonBatchEnvironment:
    """The only policy-facing Newton environment; one world is a batch of one."""

    action_dim = POLICY_ACTION_DIM

    def __init__(self, runtime: Any, *, observation_provider) -> None:
        import warp as wp

        self.runtime = runtime
        self.observation_provider = observation_provider
        self._cached_observation: dict[str, np.ndarray] | None = None
        self._sample_index = 0
        model = runtime.model
        scene = runtime.scene
        self._arm_lower = np.asarray(
            scene.arms.get_attribute("joint_limit_lower", model).numpy()[:, 0],
            dtype=np.float32,
        )
        self._arm_upper = np.asarray(
            scene.arms.get_attribute("joint_limit_upper", model).numpy()[:, 0],
            dtype=np.float32,
        )
        self._gripper_lower = np.asarray(
            scene.grippers.get_attribute("joint_limit_lower", model).numpy()[:, 0],
            dtype=np.float32,
        )
        self._gripper_upper = np.asarray(
            scene.grippers.get_attribute("joint_limit_upper", model).numpy()[:, 0],
            dtype=np.float32,
        )
        self._arm_device = wp.empty(
            (self.world_count, 12), dtype=wp.float32, device=model.device
        )
        self._gripper_device = wp.empty(
            (self.world_count, 2), dtype=wp.float32, device=model.device
        )

    @property
    def world_count(self) -> int:
        return int(self.runtime.model.world_count)

    @property
    def sample_index(self) -> int:
        return self._sample_index

    def observe(self) -> dict[str, np.ndarray]:
        if self._cached_observation is not None:
            return self._cached_observation
        scene = self.runtime.scene
        state = self.runtime.state_0
        arms = scene.arms.get_attribute("joint_q", state).numpy()[:, 0]
        gripper_q = scene.grippers.get_attribute("joint_q", state).numpy()[:, 0]
        observation = {
            "observation.state": pack_policy_state(
                arms,
                gripper_openness(
                    gripper_q,
                    self._gripper_lower,
                    self._gripper_upper,
                ),
            )
        }
        images = self.observation_provider(self.runtime)
        overlap = set(observation) & set(images)
        if overlap:
            raise ValueError(f"observation provider replaced reserved keys: {overlap}")
        observation.update(images)
        if any(value.shape[0] != self.world_count for value in observation.values()):
            raise RuntimeError("policy observation does not contain one row per Newton world")
        self._cached_observation = observation
        return observation

    def step(self, policy_actions: np.ndarray):
        expected = (self.world_count, POLICY_ACTION_DIM)
        if np.asarray(policy_actions).shape != expected:
            raise ValueError(f"batched policy action shape must be {expected}")
        arms, grippers = unpack_policy_actions(policy_actions)
        arms = np.ascontiguousarray(np.clip(arms, self._arm_lower, self._arm_upper))
        grippers = np.ascontiguousarray(np.clip(grippers, 0.0, 1.0))
        self._arm_device.assign(arms)
        self._gripper_device.assign(grippers)
        self.runtime.step(self._arm_device, self._gripper_device)
        self._sample_index += 1
        self._cached_observation = None
        return self._arm_device, self._gripper_device


class NewtonTrajectoryEpisode(NewtonBatchEnvironment):
    """Step one homogeneous Newton world batch and publish per-world trajectories."""

    action_dim = POLICY_ACTION_DIM

    def __init__(
        self,
        runtime: Any,
        outputs: tuple[str | Path | None, ...],
        scenarios: tuple[RolloutScenario, ...],
        *,
        assets_root: str | Path,
        observation_provider,
        trajectory_metadata: tuple[dict[str, Any], ...],
    ) -> None:
        super().__init__(runtime, observation_provider=observation_provider)
        self.scenarios = scenarios
        if len(outputs) != self.world_count or len(scenarios) != self.world_count:
            raise ValueError("outputs and scenarios must contain one entry per Newton world")
        if len(trajectory_metadata) != self.world_count:
            raise ValueError("trajectory_metadata must contain one entry per Newton world")
        if len({scenario.steps for scenario in scenarios}) != 1:
            raise ValueError("batched rollout scenarios must have the same step count")

        assets = Path(assets_root).expanduser().resolve()
        robot = assets / "robots" / "piper" / "piper.usda"
        robot_sha256 = _sha256(robot)
        garment_sha256 = {
            garment_id: _sha256(
                assets / "garments" / "converted" / garment_id / f"{garment_id}.usdc"
            )
            for garment_id in {str(scenario.garment_id) for scenario in scenarios}
        }
        initial_states = capture_states(runtime, 0)
        if len(initial_states) != self.world_count:
            raise RuntimeError("Newton state capture returned the wrong world count")

        writers: list[TrajectoryWriter | None] = []
        try:
            for world, (output, scenario, extra) in enumerate(
                zip(outputs, scenarios, trajectory_metadata, strict=True)
            ):
                if output is None:
                    writers.append(None)
                    continue
                garment_id = str(scenario.garment_id)
                metadata = {
                    "producer": "clothdojo_rollout",
                    "scenario_id": scenario.scenario_id,
                    "seed": scenario.seed,
                    "prompt": scenario.prompt,
                    "training_eligible": True,
                    **extra,
                }
                writer = TrajectoryWriter(
                    output,
                    control_fps=runtime.spec.control_fps,
                    sim_substeps=runtime.spec.sim_substeps,
                    max_actions=scenario.steps,
                    scene=SceneMetadata(
                        task_name=scenario.task_name,
                        robot_name="piper",
                        asset_fingerprints={
                            "garment_usdc": garment_sha256[garment_id],
                            "piper_entry_usda": robot_sha256,
                        },
                        config_fingerprints={"rtc_scenario": scenario.fingerprint},
                    ),
                    metadata=metadata,
                )
                writers.append(writer)
                writer.write_initial_state(initial_states[world])
        except Exception:
            for writer in writers:
                if writer is not None:
                    writer.abort()
            raise
        self.writers = tuple(writers)

    def step(self, policy_actions: np.ndarray) -> None:
        sample_index = self.sample_index
        arm_device, gripper_device = super().step(policy_actions)
        actions = capture_actions(
            self.runtime,
            sample_index,
            arm_device,
            gripper_device,
        )
        states = capture_states(self.runtime, self.sample_index)
        for writer, action, state in zip(self.writers, actions, states, strict=True):
            if writer is not None:
                writer.append_transition(action, state)

    def close(self) -> None:
        active = tuple(writer for writer in self.writers if writer is not None)
        if not active:
            return
        # NPZ compression is independent per canonical trajectory and can use
        # host cores while the GPU worker prepares no further simulation work.
        with ThreadPoolExecutor(max_workers=min(4, len(active))) as executor:
            futures = [executor.submit(writer.close) for writer in active]
            for future in futures:
                future.result()

    def abort(self) -> None:
        for writer in self.writers:
            if writer is not None:
                writer.abort()

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
