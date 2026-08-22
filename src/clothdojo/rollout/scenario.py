"""Strict single-scenario contract for the first RTC rollout slice."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

RTC_SCENARIO_SCHEMA = "clothdojo_rtc_scenario"
RTC_SCENARIO_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class RTCSettings:
    min_exec_horizon: int
    initial_delay_steps: int
    delay_buffer_size: int


@dataclass(frozen=True, slots=True)
class RTCScenario:
    scenario_id: str
    task_name: str
    garment_id: str
    seed: int
    prompt: str
    steps: int
    action_horizon: int
    rtc: RTCSettings

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": RTC_SCENARIO_SCHEMA,
            "format_version": RTC_SCENARIO_FORMAT_VERSION,
            **asdict(self),
            "fingerprint": self.fingerprint,
        }


def load_rtc_scenario(path: str | Path) -> RTCScenario:
    source = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read RTC scenario: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"RTC scenario must be an object: {source}")
    required = {
        "schema",
        "format_version",
        "scenario_id",
        "task_name",
        "garment_id",
        "seed",
        "prompt",
        "steps",
        "action_horizon",
        "rtc",
    }
    if set(payload) != required:
        raise ValueError(
            f"RTC scenario fields differ: missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}"
        )
    if payload["schema"] != RTC_SCENARIO_SCHEMA:
        raise ValueError(f"unsupported RTC scenario schema: {payload['schema']!r}")
    if payload["format_version"] != RTC_SCENARIO_FORMAT_VERSION:
        raise ValueError(f"unsupported RTC scenario version: {payload['format_version']!r}")
    for field in ("scenario_id", "garment_id", "prompt"):
        _nonempty_string(payload[field], field)
    if payload["task_name"] != "shirt_flatten":
        raise ValueError("the first RTC slice supports task_name=shirt_flatten only")
    seed = _nonnegative_int(payload["seed"], "seed")
    steps = _positive_int(payload["steps"], "steps")
    action_horizon = _positive_int(payload["action_horizon"], "action_horizon")
    rtc_payload = payload["rtc"]
    rtc_fields = {"min_exec_horizon", "initial_delay_steps", "delay_buffer_size"}
    if not isinstance(rtc_payload, dict) or set(rtc_payload) != rtc_fields:
        raise ValueError(
            "RTC settings must contain exactly min_exec_horizon, "
            "initial_delay_steps, and delay_buffer_size"
        )
    min_exec = _positive_int(rtc_payload["min_exec_horizon"], "min_exec_horizon")
    if min_exec >= action_horizon:
        raise ValueError("min_exec_horizon must be smaller than action_horizon")
    rtc = RTCSettings(
        min_exec_horizon=min_exec,
        initial_delay_steps=_nonnegative_int(
            rtc_payload["initial_delay_steps"], "initial_delay_steps"
        ),
        delay_buffer_size=_positive_int(rtc_payload["delay_buffer_size"], "delay_buffer_size"),
    )
    return RTCScenario(
        scenario_id=str(payload["scenario_id"]),
        task_name="shirt_flatten",
        garment_id=str(payload["garment_id"]),
        seed=seed,
        prompt=str(payload["prompt"]),
        steps=steps,
        action_horizon=action_horizon,
        rtc=rtc,
    )


def _nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value
