"""Strict benchmark timing and success semantics.

The protocol is deliberately independent of Newton, rollout storage and policy
implementation.  A benchmark result is meaningful only together with this
fingerprinted contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

EVALUATION_PROTOCOL_SCHEMA = "clothdojo_evaluation_protocol"
EVALUATION_PROTOCOL_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class CanonicalTargetProtocol:
    method: Literal["default_free_settle"] = "default_free_settle"
    min_settle_seconds: float = 8.0
    max_settle_seconds: float = 30.0
    stability_window_seconds: float = 1.0
    speed_rms_max_m_s: float = 0.005
    robot_particle_collision: bool = False
    symmetry_center_xy: tuple[float, float] = (0.651, 0.0)

    def __post_init__(self) -> None:
        if self.method != "default_free_settle":
            raise ValueError("target method must be 'default_free_settle'")
        if not np.isfinite(self.min_settle_seconds) or self.min_settle_seconds <= 0.0:
            raise ValueError("target min_settle_seconds must be finite and positive")
        if (
            not np.isfinite(self.max_settle_seconds)
            or self.max_settle_seconds < self.min_settle_seconds
        ):
            raise ValueError(
                "target max_settle_seconds must be finite and no shorter than min_settle_seconds"
            )
        if (
            not np.isfinite(self.stability_window_seconds)
            or self.stability_window_seconds <= 0.0
            or self.stability_window_seconds > self.min_settle_seconds
        ):
            raise ValueError(
                "target stability_window_seconds must be finite, positive and no "
                "longer than min_settle_seconds"
            )
        if not np.isfinite(self.speed_rms_max_m_s) or self.speed_rms_max_m_s < 0.0:
            raise ValueError("target speed_rms_max_m_s must be finite and non-negative")
        if not isinstance(self.robot_particle_collision, bool):
            raise ValueError("target robot_particle_collision must be a bool")
        if self.robot_particle_collision:
            raise ValueError("canonical target must disable robot_particle_collision")
        center = np.asarray(self.symmetry_center_xy, dtype=np.float64)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("target symmetry_center_xy must be a finite planar point")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "min_settle_seconds": self.min_settle_seconds,
            "max_settle_seconds": self.max_settle_seconds,
            "stability_window_seconds": self.stability_window_seconds,
            "speed_rms_max_m_s": self.speed_rms_max_m_s,
            "robot_particle_collision": self.robot_particle_collision,
            "symmetry_center_xy": list(self.symmetry_center_xy),
        }


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    protocol_id: str
    stage: Literal["calibration", "frozen"]
    task_name: str
    control_fps: int
    horizon_seconds: float
    metric_hz: int
    stable_seconds: float
    deformable_weight: float
    symmetry: Literal["none", "planar_180"]
    deformable_distance_max: float
    rigid_distance_max: float | None
    particle_speed_rms_max_m_s: float
    require_released: bool
    target: CanonicalTargetProtocol = field(default_factory=CanonicalTargetProtocol)

    def __post_init__(self) -> None:
        if not isinstance(self.target, CanonicalTargetProtocol):
            raise TypeError("target must be a CanonicalTargetProtocol")
        for name in ("protocol_id", "task_name"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.stage not in {"calibration", "frozen"}:
            raise ValueError("stage must be 'calibration' or 'frozen'")
        if isinstance(self.control_fps, bool) or self.control_fps < 1:
            raise ValueError("control_fps must be a positive integer")
        if isinstance(self.metric_hz, bool) or self.metric_hz < 1:
            raise ValueError("metric_hz must be a positive integer")
        if self.control_fps % self.metric_hz:
            raise ValueError("metric_hz must divide control_fps exactly")
        if self.horizon_seconds <= 0.0 or self.stable_seconds <= 0.0:
            raise ValueError("horizon_seconds and stable_seconds must be positive")
        if not 0.0 <= self.deformable_weight <= 1.0:
            raise ValueError("deformable_weight must be in [0, 1]")
        if self.symmetry not in {"none", "planar_180"}:
            raise ValueError("unsupported symmetry")
        if self.deformable_distance_max <= 0.0:
            raise ValueError("deformable_distance_max must be positive")
        if self.rigid_distance_max is not None and self.rigid_distance_max <= 0.0:
            raise ValueError("rigid_distance_max must be null or positive")
        if self.particle_speed_rms_max_m_s < 0.0:
            raise ValueError("particle_speed_rms_max_m_s must be non-negative")
        horizon_steps = _exact_count(self.horizon_seconds, self.control_fps, "horizon_seconds")
        if horizon_steps % (self.control_fps // self.metric_hz):
            raise ValueError("horizon_seconds must end on the metric sampling grid")
        _exact_count(self.stable_seconds, self.metric_hz, "stable_seconds")
        _exact_count(
            self.target.min_settle_seconds,
            self.control_fps,
            "target min_settle_seconds",
        )
        _exact_count(
            self.target.max_settle_seconds,
            self.control_fps,
            "target max_settle_seconds",
        )
        _exact_count(
            self.target.stability_window_seconds,
            self.control_fps,
            "target stability_window_seconds",
        )
        _exact_count(
            self.target.stability_window_seconds,
            self.metric_hz,
            "target stability_window_seconds at metric_hz",
        )

    @property
    def horizon_steps(self) -> int:
        return _exact_count(self.horizon_seconds, self.control_fps, "horizon_seconds")

    @property
    def metric_stride(self) -> int:
        return self.control_fps // self.metric_hz

    @property
    def stable_samples(self) -> int:
        return _exact_count(self.stable_seconds, self.metric_hz, "stable_seconds")

    @property
    def sample_count(self) -> int:
        return self.horizon_steps // self.metric_stride + 1

    @property
    def target_min_settle_steps(self) -> int:
        return _exact_count(
            self.target.min_settle_seconds,
            self.control_fps,
            "target min_settle_seconds",
        )

    @property
    def target_max_settle_steps(self) -> int:
        return _exact_count(
            self.target.max_settle_seconds,
            self.control_fps,
            "target max_settle_seconds",
        )

    @property
    def target_stability_window_steps(self) -> int:
        return _exact_count(
            self.target.stability_window_seconds,
            self.control_fps,
            "target stability_window_seconds",
        )

    @property
    def target_stability_samples(self) -> int:
        return _exact_count(
            self.target.stability_window_seconds,
            self.metric_hz,
            "target stability_window_seconds at metric_hz",
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_PROTOCOL_SCHEMA,
            "format_version": EVALUATION_PROTOCOL_FORMAT_VERSION,
            "protocol_id": self.protocol_id,
            "stage": self.stage,
            "task": self.task_name,
            "timing": {
                "control_fps": self.control_fps,
                "horizon_seconds": self.horizon_seconds,
                "metric_hz": self.metric_hz,
                "stable_seconds": self.stable_seconds,
            },
            "metric": {
                "deformable_weight": self.deformable_weight,
                "symmetry": self.symmetry,
            },
            "success": {
                "deformable_distance_max": self.deformable_distance_max,
                "rigid_distance_max": self.rigid_distance_max,
                "particle_speed_rms_max_m_s": self.particle_speed_rms_max_m_s,
                "require_released": self.require_released,
            },
            "target": self.target.to_json_dict(),
        }


def load_evaluation_protocol(path: str | Path) -> EvaluationProtocol:
    source = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read evaluation protocol {source}: {exc}") from exc
    root = _mapping(payload, "protocol")
    _exact_keys(
        root,
        {
            "schema",
            "format_version",
            "protocol_id",
            "stage",
            "task",
            "timing",
            "metric",
            "success",
            "target",
        },
        "protocol",
    )
    if root["schema"] != EVALUATION_PROTOCOL_SCHEMA:
        raise ValueError("unsupported evaluation protocol schema")
    if root["format_version"] != EVALUATION_PROTOCOL_FORMAT_VERSION:
        raise ValueError("unsupported evaluation protocol version")
    timing = _mapping(root["timing"], "timing")
    metric = _mapping(root["metric"], "metric")
    success = _mapping(root["success"], "success")
    target = _mapping(root["target"], "target")
    _exact_keys(
        timing,
        {"control_fps", "horizon_seconds", "metric_hz", "stable_seconds"},
        "timing",
    )
    _exact_keys(metric, {"deformable_weight", "symmetry"}, "metric")
    _exact_keys(
        success,
        {
            "deformable_distance_max",
            "rigid_distance_max",
            "particle_speed_rms_max_m_s",
            "require_released",
        },
        "success",
    )
    _exact_keys(
        target,
        {
            "method",
            "min_settle_seconds",
            "max_settle_seconds",
            "stability_window_seconds",
            "speed_rms_max_m_s",
            "robot_particle_collision",
            "symmetry_center_xy",
        },
        "target",
    )
    return EvaluationProtocol(
        protocol_id=_string(root["protocol_id"], "protocol_id"),
        stage=_string(root["stage"], "stage"),
        task_name=_string(root["task"], "task"),
        control_fps=_integer(timing["control_fps"], "control_fps"),
        horizon_seconds=_number(timing["horizon_seconds"], "horizon_seconds"),
        metric_hz=_integer(timing["metric_hz"], "metric_hz"),
        stable_seconds=_number(timing["stable_seconds"], "stable_seconds"),
        deformable_weight=_number(metric["deformable_weight"], "deformable_weight"),
        symmetry=_string(metric["symmetry"], "symmetry"),
        deformable_distance_max=_number(
            success["deformable_distance_max"], "deformable_distance_max"
        ),
        rigid_distance_max=(
            None
            if success["rigid_distance_max"] is None
            else _number(success["rigid_distance_max"], "rigid_distance_max")
        ),
        particle_speed_rms_max_m_s=_number(
            success["particle_speed_rms_max_m_s"],
            "particle_speed_rms_max_m_s",
        ),
        require_released=_boolean(success["require_released"], "require_released"),
        target=CanonicalTargetProtocol(
            method=_string(target["method"], "target method"),
            min_settle_seconds=_number(target["min_settle_seconds"], "target min_settle_seconds"),
            max_settle_seconds=_number(target["max_settle_seconds"], "target max_settle_seconds"),
            stability_window_seconds=_number(
                target["stability_window_seconds"],
                "target stability_window_seconds",
            ),
            speed_rms_max_m_s=_number(
                target["speed_rms_max_m_s"],
                "target speed_rms_max_m_s",
            ),
            robot_particle_collision=_boolean(
                target["robot_particle_collision"],
                "target robot_particle_collision",
            ),
            symmetry_center_xy=_planar_point(
                target["symmetry_center_xy"],
                "target symmetry_center_xy",
            ),
        ),
    )


@dataclass(slots=True)
class StableSuccessTracker:
    """Small NumPy reference for the benchmark's absorbing success state."""

    protocol: EvaluationProtocol
    batch_size: int
    success: np.ndarray = field(init=False)
    streak: np.ndarray = field(init=False)
    confirmed_step: np.ndarray = field(init=False)
    stable_since_step: np.ndarray = field(init=False)
    _last_step: int = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.success = np.zeros(self.batch_size, dtype=bool)
        self.streak = np.zeros(self.batch_size, dtype=np.int32)
        self.confirmed_step = np.full(self.batch_size, -1, dtype=np.int32)
        self.stable_since_step = np.full(self.batch_size, -1, dtype=np.int32)
        self._last_step = -self.protocol.metric_stride

    def update(
        self,
        *,
        step: int,
        deformable_distance: np.ndarray,
        rigid_distance: np.ndarray,
        speed_rms_m_s: np.ndarray,
        released: np.ndarray,
    ) -> np.ndarray:
        if step != self._last_step + self.protocol.metric_stride:
            raise ValueError("success samples must be consecutive protocol sample steps")
        self._last_step = step
        deformable = self._vector(deformable_distance, "deformable_distance", np.float64)
        rigid = self._vector(rigid_distance, "rigid_distance", np.float64)
        speed = self._vector(speed_rms_m_s, "speed_rms_m_s", np.float64)
        free = self._vector(released, "released", bool)
        candidate = deformable <= self.protocol.deformable_distance_max
        if self.protocol.rigid_distance_max is not None:
            candidate &= rigid <= self.protocol.rigid_distance_max
        candidate &= speed <= self.protocol.particle_speed_rms_max_m_s
        if self.protocol.require_released:
            candidate &= free
        self.streak = np.where(candidate, self.streak + 1, 0).astype(np.int32)
        newly_confirmed = (~self.success) & (self.streak >= self.protocol.stable_samples)
        self.success |= newly_confirmed
        self.confirmed_step[newly_confirmed] = step
        self.stable_since_step[newly_confirmed] = (
            step - (self.protocol.stable_samples - 1) * self.protocol.metric_stride
        )
        return newly_confirmed

    def _vector(self, value, name: str, dtype) -> np.ndarray:
        array = np.asarray(value, dtype=dtype)
        if array.shape != (self.batch_size,):
            raise ValueError(f"{name} must have shape ({self.batch_size},)")
        if array.dtype != np.dtype(bool) and not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"invalid {name} fields: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _planar_point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    return (_number(value[0], name), _number(value[1], name))


def _exact_count(seconds: float, frequency: int, name: str) -> int:
    raw = float(seconds) * int(frequency)
    count = round(raw)
    if count < 1 or not np.isclose(raw, count, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{name} must resolve to an exact positive sample count")
    return int(count)
