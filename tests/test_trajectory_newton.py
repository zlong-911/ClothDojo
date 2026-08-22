from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.trajectory.newton import capture_actions, capture_states


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
    return SimpleNamespace(
        model=SimpleNamespace(world_count=2),
        scene=SimpleNamespace(robot=robot, grippers=grippers),
        state_0=state,
        grasp=FakeGrasp(),
    )


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
