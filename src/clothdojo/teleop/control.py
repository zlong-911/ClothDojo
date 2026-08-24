from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

_GRIPPER_TAKEOVER_TOLERANCE = 0.1


class TeleopState(StrEnum):
    DISCONNECTED = "disconnected"
    READY = "ready"
    ENGAGED = "engaged"
    HOLD = "hold"
    FAULT = "fault"


@dataclass(slots=True)
class XRHandFrame:
    pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    squeeze: float = 0.0
    trigger: float = 0.0
    buttons: dict[str, float | bool] = field(default_factory=dict)
    valid: bool = False

    def __post_init__(self) -> None:
        self.pose = np.asarray(self.pose, dtype=np.float64).reshape(4, 4)
        self.squeeze = float(self.squeeze)
        self.trigger = float(self.trigger)
        controls_finite = math.isfinite(self.squeeze) and math.isfinite(self.trigger)
        self.valid = bool(self.valid and controls_finite)
        if not controls_finite:
            self.squeeze = 0.0
            self.trigger = 0.0


@dataclass(slots=True)
class TimestampedXRFrame:
    sequence: int
    receive_time_s: float
    left: XRHandFrame = field(default_factory=XRHandFrame)
    right: XRHandFrame = field(default_factory=XRHandFrame)
    device_time_s: float | None = None
    connection_generation: int = 0


@dataclass(slots=True)
class TeleopSafetyConfig:
    ee_min_x_m: float = 0.0
    ee_min_z_m: float = 0.82
    stale_after_s: float = 0.15
    clutch_threshold: float = 0.5
    translation_scale: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.stale_after_s,
            self.clutch_threshold,
            self.translation_scale,
            self.ee_min_x_m,
            self.ee_min_z_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("teleop safety scalars must be finite")
        if self.stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be positive")
        if not 0.0 <= self.clutch_threshold <= 1.0:
            raise ValueError("clutch_threshold must be in [0, 1]")
        if self.translation_scale <= 0.0:
            raise ValueError("translation_scale must be positive")


@dataclass(slots=True)
class TeleopTargets:
    left_pose: np.ndarray
    right_pose: np.ndarray
    gripper_open: np.ndarray
    state: TeleopState
    left_engaged: bool
    right_engaged: bool
    diagnostics: dict[str, float | int | str | bool]


def load_teleop_safety(data: dict) -> TeleopSafetyConfig:
    if not isinstance(data, dict):
        raise ValueError("runtime config must be a mapping")

    return TeleopSafetyConfig(
        ee_min_x_m=float(data["ee_min_x_m"]),
        ee_min_z_m=float(data["ee_min_z_m"]),
        stale_after_s=float(data["stale_after_s"]),
        clutch_threshold=float(data["clutch_threshold"]),
        translation_scale=float(data["translation_scale"]),
    )


class _HandClutch:
    def __init__(self, initial_target: np.ndarray):
        self.target = _valid_pose(initial_target).copy()
        self.engaged = False
        self._controller_origin: np.ndarray | None = None
        self._target_origin: np.ndarray | None = None

    def release(self) -> None:
        self.engaged = False
        self._controller_origin = None
        self._target_origin = None

    def reset_target(self, pose: np.ndarray) -> None:
        self.target = _valid_pose(pose).copy()
        self.release()

    def update(
        self,
        controller_pose: np.ndarray,
        measured_ee_pose: np.ndarray,
        *,
        engage: bool,
        ee_min_x_m: float,
        ee_min_z_m: float,
        translation_scale: float,
    ) -> tuple[np.ndarray, bool, bool]:
        controller_pose = _valid_pose(controller_pose)
        measured_ee_pose = _valid_pose(measured_ee_pose)
        if not engage:
            self.release()
            return self.target.copy(), False, False
        if not self.engaged:
            self.engaged = True
            self._controller_origin = controller_pose.copy()
            self._target_origin = measured_ee_pose.copy()
            self.target = measured_ee_pose.copy()
            return self.target.copy(), True, False

        assert self._controller_origin is not None
        assert self._target_origin is not None
        desired = self._target_origin.copy()
        controller_delta = controller_pose[:3, 3] - self._controller_origin[:3, 3]
        desired[:3, 3] = self._target_origin[:3, 3] + float(translation_scale) * controller_delta
        desired[:3, :3] = _orthogonalize(
            controller_pose[:3, :3]
            @ self._controller_origin[:3, :3].T
            @ self._target_origin[:3, :3]
        )

        limited = False
        safe_x = max(float(desired[0, 3]), ee_min_x_m)
        safe_z = max(float(desired[2, 3]), ee_min_z_m)
        if safe_x != desired[0, 3] or safe_z != desired[2, 3]:
            limited = True
            desired[0, 3] = safe_x
            desired[2, 3] = safe_z
        self.target = desired
        return self.target.copy(), True, limited


class BimanualTeleopController:
    """Backend-independent clutch, freshness watchdog and Cartesian safety layer."""

    def __init__(
        self,
        left_initial_pose: np.ndarray,
        right_initial_pose: np.ndarray,
        config: TeleopSafetyConfig,
    ) -> None:
        self.config = config
        self._left = _HandClutch(left_initial_pose)
        self._right = _HandClutch(right_initial_pose)
        self._connection_generation: int | None = None
        self._last_gripper_open = np.ones(2, dtype=np.float32)
        self._gripper_takeover_pending = np.zeros(2, dtype=bool)

    def reset_targets(
        self,
        *,
        left_pose: np.ndarray | None = None,
        right_pose: np.ndarray | None = None,
    ) -> None:
        if left_pose is not None:
            self._left.reset_target(left_pose)
        if right_pose is not None:
            self._right.reset_target(right_pose)

    def reset(self, left_pose: np.ndarray, right_pose: np.ndarray) -> None:
        """Reset clutch/watchdog state without replacing the controller."""
        self._left.reset_target(left_pose)
        self._right.reset_target(right_pose)
        self._connection_generation = None
        self._last_gripper_open.fill(1.0)
        self._gripper_takeover_pending.fill(False)

    def reset_from_state(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        gripper_open: np.ndarray,
    ) -> None:
        """Reset to restored targets and wait for each trigger to match its gripper."""
        restored_gripper = np.asarray(gripper_open, dtype=np.float32).reshape(-1)
        if restored_gripper.shape != (2,):
            raise ValueError("gripper_open must contain two values")
        if not np.all(np.isfinite(restored_gripper)) or np.any(
            (restored_gripper < 0.0) | (restored_gripper > 1.0)
        ):
            raise ValueError("gripper_open values must be finite and in [0, 1]")

        self._left.reset_target(left_pose)
        self._right.reset_target(right_pose)
        self._connection_generation = None
        self._last_gripper_open[:] = restored_gripper
        self._gripper_takeover_pending.fill(True)

    def ready_for_restored_takeover(
        self,
        frame: TimestampedXRFrame,
        *,
        now_s: float | None = None,
    ) -> bool:
        """Return whether both released hands safely match a restored state."""
        now_s = time.monotonic() if now_s is None else float(now_s)
        age_s = max(0.0, now_s - float(frame.receive_time_s))
        if age_s > float(self.config.stale_after_s):
            return False
        hands = (frame.left, frame.right)
        if any(
            not hand.valid
            or not _pose_is_finite(hand.pose)
            or hand.squeeze >= float(self.config.clutch_threshold)
            for hand in hands
        ):
            return False
        requested_open = np.asarray(
            [1.0 - float(np.clip(hand.trigger, 0.0, 1.0)) for hand in hands],
            dtype=np.float32,
        )
        return bool(
            np.all(np.abs(requested_open - self._last_gripper_open) <= _GRIPPER_TAKEOVER_TOLERANCE)
        )

    def update(
        self,
        frame: TimestampedXRFrame,
        measured_poses: Callable[[], tuple[np.ndarray, np.ndarray]],
        *,
        now_s: float | None = None,
    ) -> TeleopTargets:
        now_s = time.monotonic() if now_s is None else float(now_s)
        age_s = max(0.0, now_s - float(frame.receive_time_s))
        generation_changed = (
            self._connection_generation is not None
            and frame.connection_generation != self._connection_generation
        )
        self._connection_generation = frame.connection_generation
        stale = age_s > float(self.config.stale_after_s)
        if stale or generation_changed:
            self._left.release()
            self._right.release()

        left_valid = bool(frame.left.valid and not stale and _pose_is_finite(frame.left.pose))
        right_valid = bool(frame.right.valid and not stale and _pose_is_finite(frame.right.pose))
        needs_reanchor = (
            left_valid
            and frame.left.squeeze >= float(self.config.clutch_threshold)
            and not self._left.engaged
        ) or (
            right_valid
            and frame.right.squeeze >= float(self.config.clutch_threshold)
            and not self._right.engaged
        )
        if needs_reanchor:
            left_measured_pose, right_measured_pose = measured_poses()
        else:
            left_measured_pose = self._left.target
            right_measured_pose = self._right.target
        left_pose, left_engaged, left_limited = self._update_hand(
            self._left,
            frame.left,
            left_measured_pose,
            valid=left_valid,
        )
        right_pose, right_engaged, right_limited = self._update_hand(
            self._right,
            frame.right,
            right_measured_pose,
            valid=right_valid,
        )
        self._update_gripper(0, frame.left.trigger, valid=left_valid)
        self._update_gripper(1, frame.right.trigger, valid=right_valid)
        if not left_valid and not right_valid:
            state = TeleopState.DISCONNECTED if not stale else TeleopState.HOLD
        elif left_engaged or right_engaged:
            state = TeleopState.ENGAGED
        else:
            state = TeleopState.READY
        return TeleopTargets(
            left_pose=left_pose,
            right_pose=right_pose,
            gripper_open=self._last_gripper_open.copy(),
            state=state,
            left_engaged=left_engaged,
            right_engaged=right_engaged,
            diagnostics={
                "sequence": int(frame.sequence),
                "input_age_ms": age_s * 1000.0,
                "left_valid": left_valid,
                "right_valid": right_valid,
                "left_limited": left_limited,
                "right_limited": right_limited,
                "left_gripper_takeover_pending": bool(self._gripper_takeover_pending[0]),
                "right_gripper_takeover_pending": bool(self._gripper_takeover_pending[1]),
                "connection_generation": int(frame.connection_generation),
            },
        )

    def _update_gripper(self, index: int, trigger: float, *, valid: bool) -> None:
        if not valid:
            return
        requested_open = 1.0 - float(np.clip(trigger, 0.0, 1.0))
        if self._gripper_takeover_pending[index]:
            if (
                abs(requested_open - float(self._last_gripper_open[index]))
                > _GRIPPER_TAKEOVER_TOLERANCE
            ):
                return
            self._gripper_takeover_pending[index] = False
        self._last_gripper_open[index] = requested_open

    def _update_hand(
        self,
        clutch: _HandClutch,
        hand: XRHandFrame,
        measured_pose: np.ndarray,
        *,
        valid: bool,
    ) -> tuple[np.ndarray, bool, bool]:
        if not valid:
            clutch.release()
            return clutch.target.copy(), False, False
        return clutch.update(
            hand.pose,
            measured_pose,
            engage=hand.squeeze >= float(self.config.clutch_threshold),
            ee_min_x_m=float(self.config.ee_min_x_m),
            ee_min_z_m=float(self.config.ee_min_z_m),
            translation_scale=float(self.config.translation_scale),
        )


def _pose_is_finite(pose: np.ndarray) -> bool:
    try:
        matrix = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(matrix)))


def _valid_pose(pose: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("pose must be finite")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _orthogonalize(matrix[:3, :3])
    result[:3, 3] = matrix[:3, 3]
    return result


def _orthogonalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result
