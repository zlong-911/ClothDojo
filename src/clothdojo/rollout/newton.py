"""One-way Newton integration for policy rollouts."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.rollout.scenario import RolloutScenario
from clothdojo.trajectory import SceneMetadata, TrajectoryWriter
from clothdojo.trajectory.newton import NewtonTrajectoryRecorder, gripper_openness

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
    if values.ndim != 2 or values.shape[1] != POLICY_ACTION_DIM or not np.isfinite(values).all():
        raise ValueError(f"policy actions must be finite float32[batch, 14], got {values.shape}")
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
        self._arm_device = wp.empty((self.world_count, 12), dtype=wp.float32, device=model.device)
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
        return arms, grippers

    def synchronize(self) -> None:
        """Finish queued device work before end-to-end throughput is reported."""

        import warp as wp

        wp.synchronize_device(self.runtime.model.device)


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
        producer: str,
        training_eligible: bool,
        compressed: bool = True,
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
        writers: list[TrajectoryWriter | None] = []
        try:
            for output, scenario, extra in zip(
                outputs, scenarios, trajectory_metadata, strict=True
            ):
                if output is None:
                    writers.append(None)
                    continue
                garment_id = str(scenario.garment_id)
                overlap = {"producer", "training_eligible"} & set(extra)
                if overlap:
                    raise ValueError(
                        f"trajectory metadata contains reserved fields: {sorted(overlap)}"
                    )
                metadata = {
                    "producer": str(producer),
                    "scenario_id": scenario.scenario_id,
                    "seed": scenario.seed,
                    "prompt": scenario.prompt,
                    "training_eligible": bool(training_eligible),
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
                    compressed=compressed,
                )
                writers.append(writer)
        except Exception:
            for writer in writers:
                if writer is not None:
                    writer.abort()
            raise
        self.writers = tuple(writers)
        self._recorder = NewtonTrajectoryRecorder(
            runtime,
            capacity=max(scenario.action_horizon for scenario in scenarios) + 1,
        )
        self._recorder.capture_state(0)

    def step(self, policy_actions: np.ndarray) -> None:
        sample_index = self.sample_index
        arms, grippers = super().step(policy_actions)
        self._recorder.capture_action(sample_index, arms, grippers)
        self._recorder.capture_state(self.sample_index)

    def synchronize(self) -> None:
        self._recorder.enqueue_readback()
        super().synchronize()
        self._recorder.drain(self.writers)

    def close(self) -> None:
        self.synchronize()
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
        # Pinned buffers cannot be released while CUDA copies are in flight.
        super().synchronize()
        self._recorder.discard()
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
