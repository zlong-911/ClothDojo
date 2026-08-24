from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class XRLocomotion:
    """Map thumbsticks to camera-relative XR motion for one frame."""

    move_speed_mps: float = 0.5
    turn_speed_degps: float = 60.0
    deadzone: float = 0.18

    def __post_init__(self) -> None:
        values = (self.move_speed_mps, self.turn_speed_degps, self.deadzone)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("locomotion parameters must be finite")
        if self.move_speed_mps < 0.0 or self.turn_speed_degps < 0.0:
            raise ValueError("locomotion speeds must be non-negative")
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("locomotion deadzone must be in [0, 1)")

    def update(
        self,
        left_stick: tuple[float, float],
        right_stick: tuple[float, float],
        dt: float,
        *,
        enabled: bool,
    ) -> tuple[np.ndarray, float] | None:
        if not enabled or not math.isfinite(dt) or dt <= 0.0:
            return None
        left = self._deadzone(left_stick)
        right = self._deadzone(right_stick)
        translation = np.asarray(
            (left[0], right[1], -left[1]),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(translation))
        if norm > 1.0:
            translation /= norm
        translation *= float(self.move_speed_mps) * float(dt)
        yaw = float(right[0]) * math.radians(self.turn_speed_degps) * float(dt)
        if not np.any(translation) and yaw == 0.0:
            return None
        return translation, yaw

    def _deadzone(self, value: tuple[float, float]) -> np.ndarray:
        xy = np.clip(np.asarray(value, dtype=np.float64).reshape(2), -1.0, 1.0)
        norm = float(np.linalg.norm(xy))
        if norm > 1.0:
            xy /= norm
            norm = 1.0
        deadzone = float(self.deadzone)
        if norm <= deadzone:
            return np.zeros(2, dtype=np.float64)
        return xy / norm * ((norm - deadzone) / max(1.0e-6, 1.0 - deadzone))


@dataclass(slots=True)
class ButtonEdges:
    """Rising-edge detector for the five discrete teleop actions."""

    _previous: dict[str, bool] = field(default_factory=dict)

    def rising(self, name: str, pressed: bool) -> bool:
        pressed = bool(pressed)
        edge = pressed and not self._previous.get(name, False)
        self._previous[name] = pressed
        return edge

    def prime(self, states: dict[str, bool]) -> None:
        """Adopt current button levels without emitting connection-time actions."""
        self._previous.update({name: bool(pressed) for name, pressed in states.items()})

    def clear(self) -> None:
        self._previous.clear()


@dataclass(slots=True)
class HomeReturn:
    """Require a release-then-squeeze gesture to interrupt a Home return."""

    active: bool = False
    _resume_armed: bool = False

    def start(self) -> None:
        self.active = True
        self._resume_armed = False

    def update(self, squeeze: float, threshold: float) -> bool:
        if not self.active:
            return False
        if float(squeeze) < float(threshold):
            self._resume_armed = True
            return False
        if not self._resume_armed:
            return False
        self.active = False
        self._resume_armed = False
        return True

    def clear(self) -> None:
        self.active = False
        self._resume_armed = False
