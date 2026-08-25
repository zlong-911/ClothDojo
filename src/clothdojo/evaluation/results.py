"""Atomic condition results and paired benchmark summaries."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EVALUATION_RESULT_SCHEMA = "clothdojo_evaluation_condition"
EVALUATION_RESULT_FORMAT_VERSION = 2


@dataclass(frozen=True, order=True, slots=True)
class EvaluationCaseKey:
    asset_id: str
    initial_state_index: int
    replicate: int = 0


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    key: EvaluationCaseKey
    initial_state_seed: int
    policy_seed: int
    success: bool
    success_at_end: bool
    confirmed_step: int | None
    stable_since_step: int | None
    terminal_deformable_distance: float
    terminal_rigid_distance: float
    terminal_combined_distance: float
    terminal_speed_rms_m_s: float
    best_settled_step: int | None
    best_settled_deformable_distance: float | None
    best_settled_rigid_distance: float | None
    best_settled_combined_distance: float | None

    def __post_init__(self) -> None:
        if not self.key.asset_id:
            raise ValueError("evaluation asset_id must not be empty")
        for name in ("initial_state_index", "replicate", "initial_state_seed", "policy_seed"):
            value = getattr(self.key, name) if hasattr(self.key, name) else getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.success, bool) or not isinstance(self.success_at_end, bool):
            raise ValueError("success fields must be booleans")
        if self.success != (self.confirmed_step is not None):
            raise ValueError("success and confirmed_step disagree")
        if (self.confirmed_step is None) != (self.stable_since_step is None):
            raise ValueError("confirmed_step and stable_since_step must appear together")
        if self.confirmed_step is not None:
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (self.confirmed_step, self.stable_since_step)
            ):
                raise ValueError("success steps must be non-negative integers")
            if self.stable_since_step > self.confirmed_step:
                raise ValueError("stable_since_step cannot follow confirmed_step")
        best_values = (
            self.best_settled_step,
            self.best_settled_deformable_distance,
            self.best_settled_rigid_distance,
            self.best_settled_combined_distance,
        )
        if any(value is None for value in best_values) != all(
            value is None for value in best_values
        ):
            raise ValueError("best-settled fields must be all null or all present")
        if self.best_settled_step is not None and (
            isinstance(self.best_settled_step, bool)
            or not isinstance(self.best_settled_step, int)
            or self.best_settled_step < 0
        ):
            raise ValueError("best_settled_step must be a non-negative integer")
        numeric = (
            self.terminal_deformable_distance,
            self.terminal_rigid_distance,
            self.terminal_combined_distance,
            self.terminal_speed_rms_m_s,
            *(value for value in best_values[1:] if value is not None),
        )
        if not np.isfinite(numeric).all() or any(value < 0.0 for value in numeric):
            raise ValueError("evaluation metrics must be finite and non-negative")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.key.asset_id,
            "initial_state_index": self.key.initial_state_index,
            "replicate": self.key.replicate,
            "initial_state_seed": self.initial_state_seed,
            "policy_seed": self.policy_seed,
            "success": self.success,
            "success_at_end": self.success_at_end,
            "confirmed_step": self.confirmed_step,
            "stable_since_step": self.stable_since_step,
            "terminal": {
                "deformable_distance": self.terminal_deformable_distance,
                "rigid_distance": self.terminal_rigid_distance,
                "combined_distance": self.terminal_combined_distance,
                "speed_rms_m_s": self.terminal_speed_rms_m_s,
            },
            "best_settled": (
                None
                if self.best_settled_step is None
                else {
                    "step": self.best_settled_step,
                    "deformable_distance": self.best_settled_deformable_distance,
                    "rigid_distance": self.best_settled_rigid_distance,
                    "combined_distance": self.best_settled_combined_distance,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class EvaluationConditionResults:
    root: Path
    meta: dict[str, Any]
    cases: tuple[EvaluationCaseResult, ...]

    @property
    def condition_id(self) -> str:
        return str(self.meta["condition_id"])

    @property
    def horizon_steps(self) -> int:
        return int(self.meta["protocol"]["horizon_steps"])

    @property
    def control_fps(self) -> int:
        return int(self.meta["protocol"]["control_fps"])


def write_evaluation_cases(
    path: str | Path, cases: Iterable[EvaluationCaseResult]
) -> None:
    """Atomically write the canonical, deterministically ordered case JSONL."""

    destination = Path(path).expanduser().resolve()
    ordered = tuple(sorted(cases, key=lambda item: item.key))
    if not ordered or len({item.key for item in ordered}) != len(ordered):
        raise ValueError("evaluation cases must be non-empty and uniquely keyed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            for result in ordered:
                stream.write(json.dumps(result.to_json_dict(), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_evaluation_condition(source: str | Path) -> EvaluationConditionResults:
    root = Path(source).expanduser().resolve()
    try:
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read evaluation condition {root}: {exc}") from exc
    if meta.get("schema") != EVALUATION_RESULT_SCHEMA:
        raise ValueError("unsupported evaluation result schema")
    if meta.get("format_version") != EVALUATION_RESULT_FORMAT_VERSION:
        raise ValueError("unsupported evaluation result version")
    if meta.get("status") != "complete":
        raise RuntimeError("evaluation condition is not complete")
    relative = meta.get("results")
    if relative != "results.jsonl":
        raise ValueError("evaluation result path must be results.jsonl")
    cases = []
    try:
        with (root / relative).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank evaluation result line {line_number}")
                cases.append(_case_from_json(json.loads(line)))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read evaluation cases from {root}: {exc}") from exc
    if len(cases) != meta.get("case_count"):
        raise ValueError("evaluation case count differs from metadata")
    if len({item.key for item in cases}) != len(cases):
        raise ValueError("evaluation cases contain duplicate keys")
    return EvaluationConditionResults(root, meta, tuple(cases))


def summarize_paired_conditions(sources: Iterable[str | Path]) -> dict[str, Any]:
    """Macro/micro summaries over an exactly paired case set."""

    conditions = tuple(load_evaluation_condition(source) for source in sources)
    if len(conditions) < 1:
        raise ValueError("at least one evaluation condition is required")
    ids = [condition.condition_id for condition in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation condition ids must be unique")
    reference = conditions[0]
    identity = (
        reference.meta["protocol"]["fingerprint"],
        reference.meta["scenario"],
        reference.meta["initial_states_sha256"],
        reference.meta["initial_state_pools"],
        reference.meta["templates"],
        reference.horizon_steps,
        reference.control_fps,
    )
    keys = {item.key for item in reference.cases}
    ordered_keys = sorted(keys)
    maps = []
    for condition in conditions:
        actual_identity = (
            condition.meta["protocol"]["fingerprint"],
            condition.meta["scenario"],
            condition.meta["initial_states_sha256"],
            condition.meta["initial_state_pools"],
            condition.meta["templates"],
            condition.horizon_steps,
            condition.control_fps,
        )
        if actual_identity != identity:
            raise ValueError(
                "conditions do not share one protocol, scenario and initial-state pool"
            )
        mapping = {item.key: item for item in condition.cases}
        if set(mapping) != keys:
            raise ValueError("conditions do not contain exactly the same paired cases")
        maps.append(mapping)
    baseline = maps[0]
    for mapping in maps[1:]:
        for key in ordered_keys:
            if (
                mapping[key].initial_state_seed != baseline[key].initial_state_seed
                or mapping[key].policy_seed != baseline[key].policy_seed
            ):
                raise ValueError("paired cases do not use identical state and policy seeds")
    summaries = [
        _condition_summary(condition, mapping, ordered_keys)
        for condition, mapping in zip(conditions, maps, strict=True)
    ]
    baseline = maps[0]
    comparisons = []
    for condition, mapping in zip(conditions[1:], maps[1:], strict=True):
        wins = losses = ties = 0
        for key in ordered_keys:
            left = baseline[key].success
            right = mapping[key].success
            wins += int(right and not left)
            losses += int(left and not right)
            ties += int(left == right)
        comparisons.append(
            {
                "baseline": reference.condition_id,
                "condition": condition.condition_id,
                "success_wins": wins,
                "success_losses": losses,
                "success_ties": ties,
                "paired_success_rate_delta": (
                    summaries[len(comparisons) + 1]["success_rate_micro"]
                    - summaries[0]["success_rate_micro"]
                ),
            }
        )
    return {
        "schema": "clothdojo_evaluation_summary",
        "format_version": 2,
        "protocol": reference.meta["protocol"],
        "initial_states_sha256": reference.meta["initial_states_sha256"],
        "templates": reference.meta["templates"],
        "case_count": len(ordered_keys),
        "asset_count": len({key.asset_id for key in ordered_keys}),
        "conditions": summaries,
        "paired_comparisons": comparisons,
    }


def _condition_summary(condition, mapping, ordered_keys):
    successes = np.asarray([mapping[key].success for key in ordered_keys], dtype=bool)
    successes_at_end = np.asarray(
        [mapping[key].success_at_end for key in ordered_keys], dtype=bool
    )
    horizon_seconds = condition.horizon_steps / condition.control_fps
    completion = np.asarray(
        [
            (
                mapping[key].confirmed_step / condition.control_fps
                if mapping[key].confirmed_step is not None
                else horizon_seconds
            )
            for key in ordered_keys
        ],
        dtype=np.float64,
    )
    assets = sorted({key.asset_id for key in ordered_keys})
    asset_rates = {
        asset: float(
            np.mean([mapping[key].success for key in ordered_keys if key.asset_id == asset])
        )
        for asset in assets
    }
    successful_times = completion[successes]
    return {
        "condition_id": condition.condition_id,
        "case_count": len(ordered_keys),
        "success_count": int(successes.sum()),
        "success_rate_micro": float(successes.mean()),
        "success_at_end_rate_micro": float(successes_at_end.mean()),
        "success_rate_asset_macro": float(np.mean(list(asset_rates.values()))),
        "asset_success_rates": asset_rates,
        "restricted_mean_completion_seconds": float(completion.mean()),
        "median_success_time_seconds": (
            None if not len(successful_times) else float(np.median(successful_times))
        ),
        "terminal_combined_distance_mean": float(
            np.mean([mapping[key].terminal_combined_distance for key in ordered_keys])
        ),
        "terminal_speed_rms_m_s_mean": float(
            np.mean([mapping[key].terminal_speed_rms_m_s for key in ordered_keys])
        ),
    }


def _case_from_json(value: Any) -> EvaluationCaseResult:
    if not isinstance(value, dict):
        raise ValueError("evaluation case must be a JSON object")
    terminal = value.get("terminal")
    if not isinstance(terminal, dict):
        raise ValueError("evaluation case terminal metric is invalid")
    best = value.get("best_settled")
    if best is not None and not isinstance(best, dict):
        raise ValueError("evaluation case best_settled metric is invalid")
    return EvaluationCaseResult(
        key=EvaluationCaseKey(
            asset_id=str(value["asset_id"]),
            initial_state_index=int(value["initial_state_index"]),
            replicate=int(value["replicate"]),
        ),
        initial_state_seed=int(value["initial_state_seed"]),
        policy_seed=int(value["policy_seed"]),
        success=value["success"],
        success_at_end=value["success_at_end"],
        confirmed_step=value.get("confirmed_step"),
        stable_since_step=value.get("stable_since_step"),
        terminal_deformable_distance=float(terminal["deformable_distance"]),
        terminal_rigid_distance=float(terminal["rigid_distance"]),
        terminal_combined_distance=float(terminal["combined_distance"]),
        terminal_speed_rms_m_s=float(terminal["speed_rms_m_s"]),
        best_settled_step=None if best is None else int(best["step"]),
        best_settled_deformable_distance=(
            None if best is None else float(best["deformable_distance"])
        ),
        best_settled_rigid_distance=(
            None if best is None else float(best["rigid_distance"])
        ),
        best_settled_combined_distance=(
            None if best is None else float(best["combined_distance"])
        ),
    )
