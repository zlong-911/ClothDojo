"""Strict task scenario contract shared by chunk and RTC rollouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROLLOUT_SCENARIO_SCHEMA = "clothdojo_rollout_scenario"
ROLLOUT_SCENARIO_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class RolloutScenario:
    scenario_id: str
    task_name: str
    garment_id: str | None
    seed: int
    prompt: str
    steps: int
    action_horizon: int

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": ROLLOUT_SCENARIO_SCHEMA,
            "format_version": ROLLOUT_SCENARIO_FORMAT_VERSION,
            **asdict(self),
            "fingerprint": self.fingerprint,
        }


def load_rollout_scenario(path: str | Path) -> RolloutScenario:
    source = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read rollout scenario: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"rollout scenario must be an object: {source}")

    required = {
        "schema",
        "format_version",
        "scenario_id",
        "task_name",
        "seed",
        "prompt",
        "steps",
        "action_horizon",
    }
    allowed = required | {"garment_id"}
    if not required <= set(payload) or not set(payload) <= allowed:
        raise ValueError(
            f"rollout scenario fields differ: missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - allowed)}"
        )
    if payload["schema"] != ROLLOUT_SCENARIO_SCHEMA:
        raise ValueError(f"unsupported rollout scenario schema: {payload['schema']!r}")
    if payload["format_version"] != ROLLOUT_SCENARIO_FORMAT_VERSION:
        raise ValueError(
            f"unsupported rollout scenario version: {payload['format_version']!r}"
        )
    for field in ("scenario_id", "prompt"):
        _nonempty_string(payload[field], field)
    garment_id = payload.get("garment_id")
    if garment_id is not None:
        _nonempty_string(garment_id, "garment_id")
    if payload["task_name"] != "shirt_flatten":
        raise ValueError("the first rollout slice supports task_name=shirt_flatten only")

    return RolloutScenario(
        scenario_id=str(payload["scenario_id"]),
        task_name="shirt_flatten",
        garment_id=None if garment_id is None else str(garment_id),
        seed=_nonnegative_int(payload["seed"], "seed"),
        prompt=str(payload["prompt"]),
        steps=_positive_int(payload["steps"], "steps"),
        action_horizon=_positive_int(payload["action_horizon"], "action_horizon"),
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
