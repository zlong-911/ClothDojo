from __future__ import annotations

import numpy as np

from clothdojo.teleop.control import (
    BimanualTeleopController,
    TeleopSafetyConfig,
    TeleopState,
    TimestampedXRFrame,
    XRHandFrame,
)
from clothdojo.teleop.ik import DualArmIKController
from clothdojo.teleop.isaac_app import _targets_changed


def _pose(x: float, y: float, z: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = (x, y, z)
    return value


def test_teleop_ik_enables_cuda_graph_by_default() -> None:
    assert DualArmIKController.__dataclass_fields__["use_cuda_graph"].default is True


def test_teleop_only_updates_ik_when_cartesian_targets_change() -> None:
    left = _pose(0.5, 0.25, 1.0)
    right = _pose(0.5, -0.25, 1.0)

    assert not _targets_changed(left.copy(), right.copy(), left, right)
    moved = left.copy()
    moved[0, 3] += 0.01
    assert _targets_changed(moved, right, left, right)


def test_bimanual_controller_preserves_clutch_and_stale_input_safety() -> None:
    left_home = _pose(0.5, 0.25, 1.0)
    right_home = _pose(0.5, -0.25, 1.0)
    controller = BimanualTeleopController(
        left_home,
        right_home,
        TeleopSafetyConfig(stale_after_s=0.15, clutch_threshold=0.5),
    )

    released = TimestampedXRFrame(
        sequence=1,
        receive_time_s=1.0,
        left=XRHandFrame(pose=_pose(0.0, 0.2, 1.2), trigger=0.25, valid=True),
        right=XRHandFrame(pose=_pose(0.0, -0.2, 1.2), trigger=0.75, valid=True),
    )
    ready = controller.update(released, lambda: (left_home, right_home), now_s=1.0)
    assert ready.state == TeleopState.READY
    np.testing.assert_allclose(ready.gripper_open, (0.75, 0.25))

    anchored = TimestampedXRFrame(
        sequence=2,
        receive_time_s=1.01,
        left=XRHandFrame(pose=_pose(0.0, 0.2, 1.2), squeeze=1.0, valid=True),
        right=released.right,
    )
    first = controller.update(anchored, lambda: (left_home, right_home), now_s=1.01)
    assert first.left_engaged is True
    np.testing.assert_allclose(first.left_pose, left_home)

    moved = TimestampedXRFrame(
        sequence=3,
        receive_time_s=1.02,
        left=XRHandFrame(pose=_pose(0.1, 0.2, 1.2), squeeze=1.0, valid=True),
        right=released.right,
    )
    engaged = controller.update(moved, lambda: (left_home, right_home), now_s=1.02)
    assert engaged.state == TeleopState.ENGAGED
    np.testing.assert_allclose(engaged.left_pose[:3, 3], (0.6, 0.25, 1.0))

    stale = controller.update(moved, lambda: (left_home, right_home), now_s=2.0)
    assert stale.state == TeleopState.HOLD
    assert stale.left_engaged is False
