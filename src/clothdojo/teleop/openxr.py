from __future__ import annotations

import time

import numpy as np

from .control import TimestampedXRFrame, XRHandFrame


class IsaacOpenXRBackend:
    """Read standard OpenXR controller state through Isaac Kit.

    Kit returns grip poses in the USD virtual world, including the configured
    stage anchor. The render stage and Newton both use meters and Z-up, so the
    pose is passed through without another coordinate conversion.
    """

    def __init__(self, xr_core=None) -> None:
        if xr_core is None:
            from omni.kit.xr.core import XRCore

            xr_core = XRCore.get_singleton()
        self._xr_core = xr_core
        self._sequence = 0
        self._generation = 0
        self._connected = False
        self._devices = {"left": None, "right": None}
        self._haptic_queue: list[tuple[float, float]] = []
        self._haptic_segment_end = 0.0
        self._haptic_amplitude = 0.0
        self._grasp_haptic_amplitude = {"left": 0.0, "right": 0.0}
        self._haptic_output_names: dict[str, str | None] = {
            "left": None,
            "right": None,
        }

    def poll(self) -> TimestampedXRFrame:
        self._advance_haptics()
        left = self._read_hand("/user/hand/left")
        right = self._read_hand("/user/hand/right")
        connected = left.valid or right.valid
        if connected != self._connected:
            self._generation += 1
            self._connected = connected
        self._sequence += 1
        return TimestampedXRFrame(
            sequence=self._sequence,
            receive_time_s=time.monotonic(),
            left=left,
            right=right,
            connection_generation=self._generation,
        )

    def close(self) -> None:
        self.stop_haptics()

    def pulse_connected(self) -> None:
        self._queue_haptics([(0.08, 0.45)])

    def pulse_recording_start(self) -> None:
        self._queue_haptics([(0.07, 0.80), (0.08, 0.0), (0.07, 0.80)])

    def pulse_recording_stop(self) -> None:
        self._queue_haptics([(0.16, 0.35)])

    def pulse_reset_complete(self) -> None:
        self._queue_haptics([(0.05, 0.55), (0.05, 0.0), (0.05, 0.55), (0.05, 0.0), (0.05, 0.55)])

    def set_grasp_haptics(self, left: bool, right: bool) -> None:
        """Sustain fixed per-hand feedback while cloth is in an attach region."""
        amplitude = 0.25
        self._grasp_haptic_amplitude["left"] = amplitude if left else 0.0
        self._grasp_haptic_amplitude["right"] = amplitude if right else 0.0
        self._apply_haptics()

    def stop_haptics(self) -> None:
        self._haptic_queue.clear()
        self._haptic_segment_end = 0.0
        self._haptic_amplitude = 0.0
        self._grasp_haptic_amplitude["left"] = 0.0
        self._grasp_haptic_amplitude["right"] = 0.0
        self._apply_haptics()

    def _read_hand(self, path: str) -> XRHandFrame:
        device = self._xr_core.get_input_device(path)
        side = "left" if path.endswith("left") else "right"
        self._devices[side] = device
        pose = _read_pose(device)
        stick_x, stick_y = _read_thumbstick(device)
        face_primary = "x" if side == "left" else "a"
        face_secondary = "y" if side == "left" else "b"
        return XRHandFrame(
            pose=np.eye(4, dtype=np.float64) if pose is None else pose,
            squeeze=_read_input(device, "squeeze"),
            trigger=_read_input(device, "trigger"),
            buttons={
                face_primary: _read_click(device, face_primary),
                face_secondary: _read_click(device, face_secondary),
                "thumbstick_click": _read_click(device, "thumbstick"),
                "thumbstick_x": stick_x,
                "thumbstick_y": stick_y,
            },
            valid=pose is not None,
        )

    def _queue_haptics(self, pattern: list[tuple[float, float]]) -> None:
        self._haptic_queue[:] = pattern
        self._haptic_segment_end = 0.0
        self._haptic_amplitude = 0.0

    def _advance_haptics(self) -> None:
        now = time.monotonic()
        if self._haptic_segment_end > now:
            return
        if self._haptic_queue:
            duration, amplitude = self._haptic_queue.pop(0)
            self._haptic_amplitude = float(np.clip(amplitude, 0.0, 1.0))
            self._haptic_segment_end = now + max(0.0, float(duration))
            self._apply_haptics()
        elif self._haptic_amplitude > 0.0:
            self._haptic_amplitude = 0.0
            self._apply_haptics()

    def _apply_haptics(self) -> None:
        for side, device in self._devices.items():
            _set_haptic_output(
                device,
                max(self._haptic_amplitude, self._grasp_haptic_amplitude[side]),
                cached_name=self._haptic_output_names.get(side),
                cache=lambda name, side=side: self._haptic_output_names.__setitem__(side, name),
            )


def _read_pose(device) -> np.ndarray | None:
    if device is None:
        return None
    for name in ("grip", "aim"):
        try:
            if not device.has_pose(name):
                continue
            # Gf.Matrix4d iteration is row-major while numpy consumers in the
            # controller use column-vector transforms.
            pose = np.asarray(device.get_virtual_world_pose(name), dtype=np.float64).reshape(4, 4).T
        except Exception:
            continue
        if np.all(np.isfinite(pose)):
            return pose
    return None


def _read_input(device, name: str) -> float:
    if device is None:
        return 0.0
    try:
        if not device.has_input(name):
            return 0.0
        value = float(device.get_input_gesture_value(name, "value"))
    except Exception:
        return 0.0
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def _read_click(device, name: str) -> bool:
    return _read_gesture(device, name, "click") > 0.5


def _read_thumbstick(device) -> tuple[float, float]:
    direct = (
        _read_axis_gesture(device, "thumbstick_x", "value"),
        _read_axis_gesture(device, "thumbstick_y", "value"),
    )
    if direct != (0.0, 0.0):
        return direct
    for name in ("thumbstick", "joystick"):
        axes = (
            _read_axis_gesture(device, name, "x"),
            _read_axis_gesture(device, name, "y"),
        )
        if axes != (0.0, 0.0):
            return axes
    return 0.0, 0.0


def _read_gesture(device, name: str, gesture: str) -> float:
    if device is None:
        return 0.0
    try:
        if not device.has_input(name):
            return 0.0
        value = float(device.get_input_gesture_value(name, gesture))
    except Exception:
        return 0.0
    return value if np.isfinite(value) else 0.0


def _read_axis_gesture(device, name: str, gesture: str) -> float:
    return float(np.clip(_read_gesture(device, name, gesture), -1.0, 1.0))


def _set_haptic_output(device, amplitude: float, *, cached_name, cache) -> bool:
    if device is None:
        return False
    names = [cached_name] if cached_name else []
    if hasattr(device, "get_output_names"):
        try:
            names.extend(str(name) for name in device.get_output_names() if name not in names)
        except Exception:
            pass
    names.extend(name for name in ("haptic", "vibration", "rumble", "vibrate") if name not in names)
    for name in names:
        try:
            if not device.has_output(name):
                device.ensure_output(name)
            if device.has_output(name):
                device.set_output_value(name, float(np.clip(amplitude, 0.0, 1.0)))
                cache(name)
                return True
        except Exception:
            continue
    return False
