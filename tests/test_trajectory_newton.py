from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.trajectory.newton import (
    NewtonTrajectoryRecorder,
    capture_actions,
    capture_canonical_actions,
    capture_states,
)


class FakeArray:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def numpy(self) -> np.ndarray:
        return self.values


class FakeView:
    def __init__(self, **attributes) -> None:
        self.attributes = {name: FakeArray(value) for name, value in attributes.items()}

    def get_attribute(self, name, source) -> FakeArray:
        return self.attributes[name]


class FakeGrasp:
    def particle_ranges(self) -> tuple[tuple[int, int], ...]:
        return ((0, 3), (3, 6))

    def capture_all_states(self) -> tuple[dict[str, np.ndarray], ...]:
        return tuple(
            {
                "grasp_attached": np.asarray([world, 0], dtype=np.int32),
                "grasp_previous_open": np.asarray([0.25, 1.0], dtype=np.float32),
                "grasp_pin_count": np.asarray([1, 0], dtype=np.int32),
                "grasp_pins": np.asarray([[world, -1], [-1, -1]], dtype=np.int32),
                "grasp_pin_local": np.full((2, 2, 3), world, dtype=np.float32),
            }
            for world in range(2)
        )


def _runtime():
    robot = FakeView(
        joint_q=np.arange(12, dtype=np.float32).reshape(2, 1, 6),
        joint_qd=np.arange(12, dtype=np.float32).reshape(2, 1, 6) + 0.5,
        body_q=np.arange(28, dtype=np.float32).reshape(2, 1, 2, 7),
        body_qd=np.arange(24, dtype=np.float32).reshape(2, 1, 2, 6),
    )
    grippers = FakeView(
        joint_q=np.asarray(
            [[[0.02, -0.01, 0.04, -0.03]], [[-0.01, 0.0, 0.08, 0.0]]],
            dtype=np.float32,
        ),
        joint_limit_lower=np.asarray(
            [[[0.0, -0.04, 0.0, -0.04]], [[0.0, -0.04, 0.0, -0.04]]],
            dtype=np.float32,
        ),
        joint_limit_upper=np.asarray(
            [[[0.04, 0.0, 0.04, 0.0]], [[0.04, 0.0, 0.04, 0.0]]],
            dtype=np.float32,
        ),
    )
    state = SimpleNamespace(
        particle_q=FakeArray(np.arange(18, dtype=np.float32).reshape(6, 3)),
        particle_qd=FakeArray(np.arange(18, dtype=np.float32).reshape(6, 3) + 0.25),
    )
    runtime = SimpleNamespace(
        model=SimpleNamespace(world_count=2),
        scene=SimpleNamespace(robot=robot, grippers=grippers),
        state_0=state,
        grasp=FakeGrasp(),
    )
    runtime.canonical_arm_target = FakeArray(
        np.arange(24, dtype=np.float32).reshape(2, 12)
    )
    runtime.canonical_gripper_open = FakeArray(
        np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    )
    return runtime


def test_capture_states_splits_worlds_and_copies_simulator_arrays() -> None:
    runtime = _runtime()

    states = capture_states(runtime, 7)

    assert [state.sample_index for state in states] == [7, 7]
    np.testing.assert_array_equal(states[0].particle_q, np.arange(9).reshape(3, 3))
    np.testing.assert_array_equal(states[1].particle_q, np.arange(9, 18).reshape(3, 3))
    np.testing.assert_allclose(states[0].gripper_open_state, [0.5, 1.0])
    np.testing.assert_allclose(states[1].gripper_open_state, [0.0, 1.0])
    assert states[1].grasp_pins[0, 0] == 1

    runtime.state_0.particle_q.values.fill(-1)
    assert np.all(states[0].particle_q >= 0)


def test_capture_states_converts_physical_worlds_to_local_coordinates() -> None:
    runtime = _runtime()
    runtime.scene.world_offsets = ((0.0, 0.0, 0.0), (10.0, 20.0, 30.0))
    particle_physical = runtime.state_0.particle_q.values
    particle_physical[3:] += np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    body_physical = runtime.scene.robot.attributes["body_q"].values
    body_physical[1, 0, :, :3] += np.asarray([10.0, 20.0, 30.0], dtype=np.float32)

    states = capture_states(runtime, 0)

    np.testing.assert_array_equal(states[1].particle_q, np.arange(9, 18).reshape(3, 3))
    np.testing.assert_array_equal(
        states[1].body_q[:, :3],
        np.arange(14, 28, dtype=np.float32).reshape(2, 7)[:, :3],
    )


def test_capture_actions_preserves_worlds_and_rejects_wrong_shapes() -> None:
    runtime = _runtime()
    arm = FakeArray(np.arange(24, dtype=np.float32).reshape(2, 12))
    gripper = FakeArray(np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32))

    actions = capture_actions(runtime, 3, arm, gripper)

    assert [action.sample_index for action in actions] == [3, 3]
    np.testing.assert_array_equal(actions[1].joint_target, np.arange(12, 24))
    np.testing.assert_array_equal(actions[1].gripper_open_command, [0.5, 0.25])

    with pytest.raises(ValueError, match="arm_target shape"):
        capture_actions(runtime, 3, FakeArray(np.zeros((2, 11))), gripper)


def test_capture_canonical_actions_reads_runtime_owned_buffers() -> None:
    runtime = _runtime()

    actions = capture_canonical_actions(runtime, 4)

    assert [action.sample_index for action in actions] == [4, 4]
    np.testing.assert_array_equal(actions[1].joint_target, np.arange(12, 24))
    np.testing.assert_array_equal(actions[1].gripper_open_command, [0.5, 0.25])


def test_batched_recorder_copies_each_runtime_canonical_action_before_overwrite() -> None:
    import warp as wp

    worlds = 2

    class WarpView:
        def __init__(self, **values) -> None:
            self.values = values

        def get_attribute(self, name, _source):
            return self.values[name]

    class Grasp:
        def __init__(self) -> None:
            self.attached = wp.zeros((worlds, 2), dtype=wp.int32, device="cpu")
            self.previous_open = wp.ones((worlds, 2), dtype=wp.float32, device="cpu")
            self.pin_count = wp.zeros((worlds, 2), dtype=wp.int32, device="cpu")
            self.pins = wp.full((worlds, 2, 2), -1, dtype=wp.int32, device="cpu")
            self.pin_local = wp.zeros((worlds, 2, 2), dtype=wp.vec3, device="cpu")

        @staticmethod
        def particle_ranges():
            return ((0, 2), (2, 4))

    joint_q = wp.zeros((worlds, 1, 16), dtype=wp.float32, device="cpu")
    joint_qd = wp.zeros((worlds, 1, 16), dtype=wp.float32, device="cpu")
    robot = WarpView(
        joint_q=joint_q,
        joint_qd=joint_qd,
        body_q=wp.zeros((worlds, 1, 21), dtype=wp.transform, device="cpu"),
        body_qd=wp.zeros((worlds, 1, 21), dtype=wp.spatial_vector, device="cpu"),
    )
    grippers = WarpView(
        joint_q=wp.zeros((worlds, 1, 4), dtype=wp.float32, device="cpu"),
        joint_limit_lower=wp.zeros((worlds, 1, 4), dtype=wp.float32, device="cpu"),
        joint_limit_upper=wp.ones((worlds, 1, 4), dtype=wp.float32, device="cpu"),
    )
    state = SimpleNamespace(
        joint_q=joint_q,
        joint_qd=joint_qd,
        particle_q=wp.zeros(4, dtype=wp.vec3, device="cpu"),
        particle_qd=wp.zeros(4, dtype=wp.vec3, device="cpu"),
    )
    runtime = SimpleNamespace(
        model=SimpleNamespace(world_count=worlds),
        scene=SimpleNamespace(
            robot=robot,
            grippers=grippers,
            world_offsets=np.zeros((worlds, 3), dtype=np.float32),
        ),
        state_0=state,
        grasp=Grasp(),
        canonical_arm_target=wp.zeros((worlds, 12), dtype=wp.float32, device="cpu"),
        canonical_gripper_open=wp.zeros((worlds, 2), dtype=wp.float32, device="cpu"),
    )

    class Writer:
        def __init__(self) -> None:
            self.initial = None
            self.transitions = []

        def write_initial_state(self, value) -> None:
            self.initial = value

        def append_transition(self, action, next_state) -> None:
            self.transitions.append((action, next_state))

    writers = [Writer(), Writer()]
    recorder = NewtonTrajectoryRecorder(runtime, capacity=3)
    recorder.capture_state(0)
    for index in range(2):
        runtime.canonical_arm_target.assign(
            np.full((worlds, 12), 10 * (index + 1), dtype=np.float32)
            + np.arange(worlds, dtype=np.float32)[:, None]
        )
        runtime.canonical_gripper_open.assign(
            np.full((worlds, 2), 0.25 * (index + 1), dtype=np.float32)
        )
        recorder.capture_action(index)
        recorder.capture_state(index + 1)
    recorder.enqueue_readback()
    wp.synchronize_device("cpu")
    recorder.drain(writers)

    assert writers[0].initial.sample_index == 0
    assert [item[0].sample_index for item in writers[1].transitions] == [0, 1]
    np.testing.assert_array_equal(writers[1].transitions[0][0].joint_target, 11.0)
    np.testing.assert_array_equal(writers[1].transitions[1][0].joint_target, 21.0)
    np.testing.assert_array_equal(
        writers[0].transitions[1][0].gripper_open_command,
        [0.5, 0.5],
    )
