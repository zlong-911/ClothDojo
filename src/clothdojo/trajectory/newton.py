"""Convert public Newton runtime state into the trajectory dataset model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from clothdojo.trajectory.schema import TrajectoryAction, TrajectoryState

if TYPE_CHECKING:
    from clothdojo.sim.runtime import ShirtFlattenRuntime


def capture_states(
    runtime: ShirtFlattenRuntime,
    sample_index: int,
) -> tuple[TrajectoryState, ...]:
    """Copy one world-local trajectory state for every parallel world."""

    scene = runtime.scene
    state = runtime.state_0
    joint_q = scene.robot.get_attribute("joint_q", state).numpy()[:, 0]
    joint_qd = scene.robot.get_attribute("joint_qd", state).numpy()[:, 0]
    body_q = scene.robot.get_attribute("body_q", state).numpy()[:, 0]
    body_qd = scene.robot.get_attribute("body_qd", state).numpy()[:, 0]
    particle_q = state.particle_q.numpy()
    particle_qd = state.particle_qd.numpy()
    gripper_q = scene.grippers.get_attribute("joint_q", state).numpy()[:, 0]
    gripper_lower = scene.grippers.get_attribute("joint_limit_lower", runtime.model).numpy()[:, 0]
    gripper_upper = scene.grippers.get_attribute("joint_limit_upper", runtime.model).numpy()[:, 0]
    gripper_open = gripper_openness(gripper_q, gripper_lower, gripper_upper)
    grasp = runtime.grasp.capture_all_states()
    ranges = runtime.grasp.particle_ranges()
    world_offsets = _world_offsets(runtime)
    return tuple(
        TrajectoryState(
            sample_index=int(sample_index),
            joint_q=joint_q[world].copy(),
            joint_qd=joint_qd[world].copy(),
            body_q=_local_body_q(body_q[world], world_offsets[world]),
            body_qd=body_qd[world].copy(),
            particle_q=particle_q[start:end].copy() - world_offsets[world],
            particle_qd=particle_qd[start:end].copy(),
            gripper_open_state=gripper_open[world].copy(),
            **grasp[world],
        )
        for world, (start, end) in enumerate(ranges)
    )


def capture_actions(
    runtime: ShirtFlattenRuntime,
    sample_index: int,
    arm_target,
    gripper_open,
) -> tuple[TrajectoryAction, ...]:
    """Copy commands paired with the current trajectory state."""

    arm_values = np.asarray(arm_target.numpy(), dtype=np.float32)
    gripper_values = np.asarray(gripper_open.numpy(), dtype=np.float32)
    expected_arm = (runtime.model.world_count, 12)
    expected_gripper = (runtime.model.world_count, 2)
    if arm_values.shape != expected_arm:
        raise ValueError(f"arm_target shape {arm_values.shape} != {expected_arm}")
    if gripper_values.shape != expected_gripper:
        raise ValueError(f"gripper_open shape {gripper_values.shape} != {expected_gripper}")
    return tuple(
        TrajectoryAction(
            sample_index=int(sample_index),
            joint_target=arm_values[world].copy(),
            gripper_open_command=gripper_values[world].copy(),
        )
        for world in range(runtime.model.world_count)
    )


def capture_canonical_actions(
    runtime: ShirtFlattenRuntime,
    sample_index: int,
) -> tuple[TrajectoryAction, ...]:
    """Copy the bounded action actually submitted to Newton's controller."""

    return capture_actions(
        runtime,
        sample_index,
        runtime.canonical_arm_target,
        runtime.canonical_gripper_open,
    )


def gripper_openness(q, lower, upper) -> np.ndarray:
    """Return normalized left/right openness from each leader finger joint."""

    controlled = (0, 2)
    return np.clip(
        (q[:, controlled] - lower[:, controlled]) / (upper[:, controlled] - lower[:, controlled]),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)


class NewtonTrajectoryRecorder:
    """Queue Newton state readback and publish it at an existing sync boundary.

    Rollout execution already synchronizes once per policy execution window.  A
    recorder must not add another synchronization at every control step: CUDA
    copies are queued into pinned host buffers here and consumed only after the
    caller has synchronized the runtime.
    """

    def __init__(self, runtime: ShirtFlattenRuntime, *, capacity: int) -> None:
        import warp as wp

        if capacity < 1:
            raise ValueError("trajectory recorder capacity must be positive")
        self.runtime = runtime
        self.capacity = int(capacity)
        self.world_count = int(runtime.model.world_count)
        self._ranges = runtime.grasp.particle_ranges()
        self._gripper_lower = np.asarray(
            runtime.scene.grippers.get_attribute(
                "joint_limit_lower", runtime.model
            ).numpy()[:, 0],
            dtype=np.float32,
        )
        self._gripper_upper = np.asarray(
            runtime.scene.grippers.get_attribute(
                "joint_limit_upper", runtime.model
            ).numpy()[:, 0],
            dtype=np.float32,
        )
        sources = self._state_sources()
        self._buffers = {
            name: _PinnedWarpBuffer(source, capacity=self.capacity)
            for name, source in sources.items()
        }
        self._state_indices: list[int] = []
        self._action_indices: list[int] = []
        expected_actions = {
            "joint_target": (self.world_count, 12),
            "gripper_open_command": (self.world_count, 2),
        }
        canonical_sources = {
            "joint_target": runtime.canonical_arm_target,
            "gripper_open_command": runtime.canonical_gripper_open,
        }
        action_device = canonical_sources["joint_target"].device
        model_device = getattr(runtime.model, "device", None)
        if model_device is not None and action_device != wp.get_device(model_device):
            raise RuntimeError("runtime canonical actions are on the wrong model device")
        for name, expected_shape in expected_actions.items():
            source = canonical_sources[name]
            if tuple(source.shape) != expected_shape or source.dtype != wp.float32:
                raise RuntimeError(
                    f"runtime canonical {name} must be float32 with shape {expected_shape}"
                )
            if source.device != action_device:
                raise RuntimeError(f"runtime canonical {name} is on the wrong device")
        self._action_buffers = {
            "joint_target": _PinnedWarpBuffer(
                canonical_sources["joint_target"],
                capacity=self.capacity,
            ),
            "gripper_open_command": _PinnedWarpBuffer(
                canonical_sources["gripper_open_command"],
                capacity=self.capacity,
            ),
        }
        # Keep temporary articulation-view arrays alive until their queued
        # device-to-host copies have completed.
        self._source_references: list[tuple[Any, ...]] = []

    def capture_state(self, sample_index: int) -> None:
        slot = len(self._state_indices)
        if slot >= self.capacity:
            raise RuntimeError(
                "trajectory recorder filled before the rollout synchronization boundary"
            )
        sources = self._state_sources()
        for name, source in sources.items():
            self._buffers[name].copy_from(source, slot=slot)
        self._source_references.append(tuple(sources.values()))
        self._state_indices.append(int(sample_index))

    def capture_action(
        self,
        sample_index: int,
    ) -> None:
        """Queue the latest runtime-canonical action without synchronizing."""

        slot = len(self._action_indices)
        if slot >= self.capacity:
            raise RuntimeError(
                "trajectory recorder filled before the rollout synchronization boundary"
            )
        self._action_buffers["joint_target"].copy_from(
            self.runtime.canonical_arm_target,
            slot=slot,
        )
        self._action_buffers["gripper_open_command"].copy_from(
            self.runtime.canonical_gripper_open,
            slot=slot,
        )
        self._action_indices.append(int(sample_index))

    def drain(self, writers) -> None:
        """Consume copies after the caller has synchronized Newton's device."""

        action_slots = {
            sample_index: slot for slot, sample_index in enumerate(self._action_indices)
        }
        if len(action_slots) != len(self._action_indices):
            raise RuntimeError("trajectory recorder received duplicate action indices")
        for state_slot, sample_index in enumerate(self._state_indices):
            states = self._states_at(state_slot, sample_index)
            if sample_index == 0:
                for writer, state in zip(writers, states, strict=True):
                    if writer is not None:
                        writer.write_initial_state(state)
                continue
            action_slot = action_slots.pop(sample_index - 1, None)
            if action_slot is None:
                raise RuntimeError(
                    f"trajectory recorder is missing action {sample_index - 1}"
                )
            for world, (writer, state) in enumerate(zip(writers, states, strict=True)):
                if writer is None:
                    continue
                writer.append_transition(
                    TrajectoryAction(
                        sample_index=sample_index - 1,
                        joint_target=self._action_buffers["joint_target"].numpy[
                            action_slot, world
                        ],
                        gripper_open_command=self._action_buffers[
                            "gripper_open_command"
                        ].numpy[action_slot, world],
                    ),
                    state,
                )
        if action_slots:
            raise RuntimeError(
                f"trajectory recorder has actions without next states: {sorted(action_slots)}"
            )
        self._state_indices.clear()
        self._action_indices.clear()
        self._source_references.clear()

    def enqueue_readback(self) -> None:
        """Queue one contiguous readback for the pending execution window."""

        count = len(self._state_indices)
        for buffer in self._buffers.values():
            buffer.enqueue_readback(count=count)
        action_count = len(self._action_indices)
        for buffer in self._action_buffers.values():
            buffer.enqueue_readback(count=action_count)

    def discard(self) -> None:
        self._state_indices.clear()
        self._action_indices.clear()
        self._source_references.clear()

    def _state_sources(self) -> dict[str, Any]:
        state = self.runtime.state_0
        scene = self.runtime.scene
        grasp = self.runtime.grasp
        return {
            "joint_q": scene.robot.get_attribute("joint_q", state),
            "joint_qd": scene.robot.get_attribute("joint_qd", state),
            "body_q": scene.robot.get_attribute("body_q", state),
            "body_qd": scene.robot.get_attribute("body_qd", state),
            "particle_q": state.particle_q,
            "particle_qd": state.particle_qd,
            "gripper_q": scene.grippers.get_attribute("joint_q", state),
            "grasp_attached": grasp.attached,
            "grasp_previous_open": grasp.previous_open,
            "grasp_pin_count": grasp.pin_count,
            "grasp_pins": grasp.pins,
            "grasp_pin_local": grasp.pin_local,
        }

    def _states_at(
        self, slot: int, sample_index: int
    ) -> tuple[TrajectoryState, ...]:
        values = {name: buffer.numpy[slot] for name, buffer in self._buffers.items()}
        world_offsets = _world_offsets(self.runtime)
        openness = gripper_openness(
            values["gripper_q"][:, 0],
            self._gripper_lower,
            self._gripper_upper,
        )
        states = []
        for world, (start, end) in enumerate(self._ranges):
            counts = values["grasp_pin_count"][world]
            pins = np.full(values["grasp_pins"][world].shape, -1, dtype=np.int32)
            for side, raw_count in enumerate(counts):
                count = int(raw_count)
                pins[side, :count] = (
                    values["grasp_pins"][world, side, :count] - int(start)
                )
            states.append(
                TrajectoryState(
                    sample_index=sample_index,
                    joint_q=values["joint_q"][world, 0],
                    joint_qd=values["joint_qd"][world, 0],
                    body_q=_local_body_q(
                        values["body_q"][world, 0], world_offsets[world]
                    ),
                    body_qd=values["body_qd"][world, 0],
                    particle_q=(
                        values["particle_q"][start:end].copy() - world_offsets[world]
                    ),
                    particle_qd=values["particle_qd"][start:end],
                    gripper_open_state=openness[world],
                    grasp_attached=values["grasp_attached"][world],
                    grasp_previous_open=values["grasp_previous_open"][world],
                    grasp_pin_count=counts,
                    grasp_pins=pins,
                    grasp_pin_local=values["grasp_pin_local"][world],
                )
            )
        return tuple(states)


def _world_offsets(runtime) -> np.ndarray:
    values = getattr(runtime.scene, "world_offsets", None)
    if values is None:
        return np.zeros((runtime.model.world_count, 3), dtype=np.float32)
    offsets = np.asarray(values, dtype=np.float32)
    expected = (runtime.model.world_count, 3)
    if offsets.shape != expected:
        raise RuntimeError(f"scene world offsets must have shape {expected}, got {offsets.shape}")
    return offsets


def _local_body_q(body_q: np.ndarray, offset: np.ndarray) -> np.ndarray:
    values = np.asarray(body_q, dtype=np.float32).copy()
    values[..., :3] -= offset
    return values


class _PinnedWarpBuffer:
    def __init__(self, source, *, capacity: int) -> None:
        import warp as wp

        self._shape = tuple(source.shape)
        self._dtype = source.dtype
        self._device = wp.empty(
            (capacity, *self._shape),
            dtype=source.dtype,
            device=source.device,
        )
        self._array = wp.empty(
            (capacity, *self._shape),
            dtype=source.dtype,
            device="cpu",
            pinned=bool(source.device.is_cuda),
        )
        self.numpy = self._array.numpy()

    def copy_from(self, source, *, slot: int) -> None:
        if tuple(source.shape) != self._shape or source.dtype != self._dtype:
            raise RuntimeError("Newton trajectory source shape or dtype changed")
        self._wp_copy(source, slot=slot)

    def _wp_copy(self, source, *, slot: int) -> None:
        import warp as wp

        wp.copy(
            self._device,
            source,
            dest_offset=slot * int(source.size),
            count=int(source.size),
        )

    def enqueue_readback(self, *, count: int) -> None:
        import warp as wp

        if count:
            wp.copy(
                self._array,
                self._device,
                count=count * int(np.prod(self._shape, dtype=np.int64)),
            )
