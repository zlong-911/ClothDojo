"""Atomic condition results and paired benchmark summaries."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EVALUATION_RESULT_SCHEMA = "clothdojo_evaluation_condition"
SUCCESS_CHECKPOINT_SECONDS = (30, 45, 60, 75, 90)
FORMAL_BENCHMARK_PANEL_ID = "balanced200_20260901"
FORMAL_BENCHMARK_ASSET_COUNT = 200
FORMAL_BENCHMARK_SEEN_ASSETS = 100
FORMAL_BENCHMARK_UNSEEN_ASSETS = 100
FORMAL_BENCHMARK_STATES_PER_ASSET = 4
FORMAL_BENCHMARK_CASE_COUNT = 800
FORMAL_BENCHMARK_INITIAL_STATES_SHA256 = (
    "5f2ad355ccdded1301ce774c487cf5b590c3b810a00d24f46c9e63780cd23361"
)
FORMAL_BENCHMARK_SELECTION_SHA256 = (
    "205813ee1f8c7aacc99f6eea61aeb3d48e03dcdd1da8200cc5363550b45822b1"
)
FORMAL_BENCHMARK_ASSET_SPLITS_SHA256 = (
    "3b18d289327d2da58030d517197744b8f5cacc1c4299a597c81bb25c772012cf"
)
FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT = (
    "ce1bd2b5c15a42a4130b07c14ca24e179d8acdc8a12c4463dd39f2afc945ea6e"
)
FORMAL_BENCHMARK_SCENARIO_FINGERPRINT = (
    "3bd7245bac1594d0f098638351af5b1c1621cc807b35fca3f902a2bde7b68b5e"
)
# Both exact task spellings occur in training datasets. Keep the actual prompt
# and fingerprint in results; accepting the panel does not make the prompts
# interchangeable in paired comparisons.
FORMAL_BENCHMARK_SCENARIO_FINGERPRINTS = frozenset(
    {
        FORMAL_BENCHMARK_SCENARIO_FINGERPRINT,
        "c5d4f890018c1aaf22f868045558ee45670d7bc18869c1d3954448a1dbdec992",
    }
)


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
    terminal_deformable_distance: float
    terminal_rigid_distance: float
    terminal_combined_distance: float
    terminal_speed_rms_m_s: float
    best_observed_step: int
    best_observed_deformable_distance: float
    best_observed_rigid_distance: float
    best_observed_combined_distance: float
    time_average_combined_distance: float | None = None
    airborne_seconds: float | None = None

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
        if self.confirmed_step is not None:
            if (
                isinstance(self.confirmed_step, bool)
                or not isinstance(self.confirmed_step, int)
                or self.confirmed_step < 0
            ):
                raise ValueError("confirmed_step must be a non-negative integer")
        if (
            isinstance(self.best_observed_step, bool)
            or not isinstance(self.best_observed_step, int)
            or self.best_observed_step < 0
        ):
            raise ValueError("best_observed_step must be a non-negative integer")
        numeric = (
            self.terminal_deformable_distance,
            self.terminal_rigid_distance,
            self.terminal_combined_distance,
            self.terminal_speed_rms_m_s,
            self.best_observed_deformable_distance,
            self.best_observed_rigid_distance,
            self.best_observed_combined_distance,
        )
        if not np.isfinite(numeric).all() or any(value < 0.0 for value in numeric):
            raise ValueError("evaluation metrics must be finite and non-negative")
        if self.time_average_combined_distance is not None and (
            not np.isfinite(self.time_average_combined_distance)
            or self.time_average_combined_distance < 0.0
        ):
            raise ValueError(
                "time_average_combined_distance must be finite and non-negative"
            )
        if self.airborne_seconds is not None and (
            isinstance(self.airborne_seconds, bool)
            or not np.isfinite(self.airborne_seconds)
            or self.airborne_seconds < 0.0
        ):
            raise ValueError("airborne_seconds must be finite and non-negative")

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
            "terminal": {
                "deformable_distance": self.terminal_deformable_distance,
                "rigid_distance": self.terminal_rigid_distance,
                "combined_distance": self.terminal_combined_distance,
                "speed_rms_m_s": self.terminal_speed_rms_m_s,
            },
            "best_observed": {
                "step": self.best_observed_step,
                "deformable_distance": self.best_observed_deformable_distance,
                "rigid_distance": self.best_observed_rigid_distance,
                "combined_distance": self.best_observed_combined_distance,
            },
            "time_average_combined_distance": self.time_average_combined_distance,
            "airborne_seconds": self.airborne_seconds,
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
    meta_path = root / "meta.json"
    try:
        original_metadata = meta_path.read_text(encoding="utf-8")
        meta = json.loads(original_metadata)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read evaluation condition {root}: {exc}") from exc
    if meta.get("schema") != EVALUATION_RESULT_SCHEMA:
        raise ValueError("unsupported evaluation result schema")
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
    # Publishing replaces results and metadata separately. A reader that saw
    # the previous complete metadata must not return a newly published score
    # file under that previous protocol, even if both publications completed.
    try:
        if meta_path.read_text(encoding="utf-8") != original_metadata:
            raise RuntimeError("evaluation condition changed while reading results")
    except OSError as exc:
        raise RuntimeError(f"cannot reread evaluation condition {root}: {exc}") from exc
    if len(cases) != meta.get("case_count"):
        raise ValueError("evaluation case count differs from metadata")
    if len({item.key for item in cases}) != len(cases):
        raise ValueError("evaluation cases contain duplicate keys")
    return EvaluationConditionResults(root, meta, tuple(cases))


def summarize_paired_conditions(
    sources: Iterable[str | Path],
    *,
    require_formal: bool = False,
) -> dict[str, Any]:
    """Macro/micro summaries over an exactly paired case set."""

    conditions = tuple(load_evaluation_condition(source) for source in sources)
    if len(conditions) < 1:
        raise ValueError("at least one evaluation condition is required")
    if require_formal:
        for condition in conditions:
            validate_formal_evaluation_metadata(condition.meta)
    ids = [condition.condition_id for condition in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation condition ids must be unique")
    reference = conditions[0]
    identity = (
        reference.meta["protocol"]["fingerprint"],
        reference.meta["scenario"],
        reference.meta["initial_state_pools"],
        reference.meta["templates"],
        reference.meta.get("benchmark"),
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
            condition.meta["initial_state_pools"],
            condition.meta["templates"],
            condition.meta.get("benchmark"),
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
        terminal_wins = terminal_losses = terminal_ties = 0
        for key in ordered_keys:
            left = baseline[key].success
            right = mapping[key].success
            wins += int(right and not left)
            losses += int(left and not right)
            ties += int(left == right)
            left_terminal = baseline[key].success_at_end
            right_terminal = mapping[key].success_at_end
            terminal_wins += int(right_terminal and not left_terminal)
            terminal_losses += int(left_terminal and not right_terminal)
            terminal_ties += int(left_terminal == right_terminal)
        comparisons.append(
            {
                "baseline": reference.condition_id,
                "condition": condition.condition_id,
                "success_at_end_wins": terminal_wins,
                "success_at_end_losses": terminal_losses,
                "success_at_end_ties": terminal_ties,
                "paired_success_at_end_rate_delta": (
                    summaries[len(comparisons) + 1].get(
                        "success_at_end_rate_balanced",
                        summaries[len(comparisons) + 1]["success_at_end_rate_micro"],
                    )
                    - summaries[0].get(
                        "success_at_end_rate_balanced",
                        summaries[0]["success_at_end_rate_micro"],
                    )
                ),
                "success_wins": wins,
                "success_losses": losses,
                "success_ties": ties,
                "paired_success_rate_delta": (
                    summaries[len(comparisons) + 1]["success_rate_micro"]
                    - summaries[0]["success_rate_micro"]
                ),
            }
        )
    summary = {
        "schema": "clothdojo_evaluation_summary",
        "protocol": reference.meta["protocol"],
        "initial_states_sha256": reference.meta["initial_states_sha256"],
        "templates": reference.meta["templates"],
        "case_count": len(ordered_keys),
        "asset_count": len({key.asset_id for key in ordered_keys}),
        "conditions": summaries,
        "paired_comparisons": comparisons,
    }
    benchmark = reference.meta.get("benchmark")
    if isinstance(benchmark, dict):
        summary["benchmark"] = {
            key: benchmark.get(key)
            for key in (
                "panel_id",
                "selection_sha256",
                "asset_splits_sha256",
                "seen_assets",
                "unseen_assets",
                "states_per_asset",
            )
        }
    return summary


def benchmark_asset_splits_sha256(asset_splits: dict[str, str]) -> str:
    """Fingerprint the frozen asset-to-split assignment independent of JSON layout."""

    payload = json.dumps(asset_splits, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_formal_benchmark_identity(
    benchmark: Any,
    *,
    require_asset_splits: bool,
) -> dict[str, str] | None:
    if not isinstance(benchmark, dict):
        raise ValueError("condition does not identify the formal benchmark panel")
    expected = {
        "panel_id": FORMAL_BENCHMARK_PANEL_ID,
        "selection_sha256": FORMAL_BENCHMARK_SELECTION_SHA256,
        "asset_splits_sha256": FORMAL_BENCHMARK_ASSET_SPLITS_SHA256,
        "seen_assets": FORMAL_BENCHMARK_SEEN_ASSETS,
        "unseen_assets": FORMAL_BENCHMARK_UNSEEN_ASSETS,
        "states_per_asset": FORMAL_BENCHMARK_STATES_PER_ASSET,
    }
    if {key: benchmark.get(key) for key in expected} != expected:
        raise ValueError("condition does not use the frozen balanced-200 benchmark panel")
    asset_splits = benchmark.get("asset_splits")
    if asset_splits is None and not require_asset_splits:
        return None
    if not isinstance(asset_splits, dict) or any(
        not isinstance(asset_id, str) or split not in {"seen", "unseen"}
        for asset_id, split in asset_splits.items()
    ):
        raise ValueError("formal benchmark asset split mapping is invalid")
    if (
        len(asset_splits) != FORMAL_BENCHMARK_ASSET_COUNT
        or sum(split == "seen" for split in asset_splits.values()) != FORMAL_BENCHMARK_SEEN_ASSETS
        or sum(split == "unseen" for split in asset_splits.values())
        != FORMAL_BENCHMARK_UNSEEN_ASSETS
        or benchmark_asset_splits_sha256(asset_splits) != FORMAL_BENCHMARK_ASSET_SPLITS_SHA256
    ):
        raise ValueError("formal benchmark asset split mapping changed")
    return asset_splits


def validate_formal_evaluation_metadata(meta: dict[str, Any]) -> None:
    """Reject conditions outside the frozen balanced-200 benchmark panel."""

    expected = {
        "case_count": FORMAL_BENCHMARK_CASE_COUNT,
        "initial_states_sha256": FORMAL_BENCHMARK_INITIAL_STATES_SHA256,
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(
                f"formal benchmark requires {field}={value!r}, got {meta.get(field)!r}"
            )
    protocol = meta.get("protocol")
    scenario = meta.get("scenario")
    if not isinstance(protocol, dict) or (
        protocol.get("fingerprint") != FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT
    ):
        raise ValueError("condition does not use the formal benchmark protocol")
    if not isinstance(scenario, dict) or (
        scenario.get("fingerprint") not in FORMAL_BENCHMARK_SCENARIO_FINGERPRINTS
    ):
        raise ValueError("condition does not use the formal benchmark scenario")

    trajectories = meta.get("trajectories")
    asset_splits = _validate_formal_benchmark_identity(
        meta.get("benchmark"), require_asset_splits=trajectories is not None
    )
    if trajectories is None:
        return
    if not isinstance(trajectories, list) or len(trajectories) != FORMAL_BENCHMARK_CASE_COUNT:
        raise ValueError("formal benchmark requires exactly 800 indexed trajectories")
    per_asset: dict[str, set[tuple[int, int]]] = {}
    for item in trajectories:
        if not isinstance(item, dict):
            raise ValueError("formal benchmark trajectory index is invalid")
        asset_id = item.get("asset_id")
        index = item.get("initial_state_index")
        replicate = item.get("replicate")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("formal benchmark trajectory asset_id is invalid")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("formal benchmark initial_state_index is invalid")
        if isinstance(replicate, bool) or not isinstance(replicate, int):
            raise ValueError("formal benchmark replicate is invalid")
        per_asset.setdefault(asset_id, set()).add((index, replicate))
    expected_cases = {(index, 0) for index in range(FORMAL_BENCHMARK_STATES_PER_ASSET)}
    if len(per_asset) != FORMAL_BENCHMARK_ASSET_COUNT or any(
        cases != expected_cases for cases in per_asset.values()
    ):
        raise ValueError(
            "formal benchmark requires 200 assets with state indices 0..3 and replicate 0"
        )
    if asset_splits is None or set(per_asset) != set(asset_splits):
        raise ValueError("formal benchmark trajectories differ from the frozen asset panel")


def validate_formal_evaluation_summary(summary: dict[str, Any]) -> None:
    """Validate a published comparison before it is reported as benchmark evidence."""

    if summary.get("case_count") != FORMAL_BENCHMARK_CASE_COUNT:
        raise ValueError("formal benchmark summary requires exactly 800 paired cases")
    if summary.get("asset_count") != FORMAL_BENCHMARK_ASSET_COUNT:
        raise ValueError("formal benchmark summary requires exactly 200 assets")
    if summary.get("initial_states_sha256") != FORMAL_BENCHMARK_INITIAL_STATES_SHA256:
        raise ValueError("formal benchmark summary does not use the frozen state panel")
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict) or (
        protocol.get("fingerprint") != FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT
    ):
        raise ValueError("formal benchmark summary does not use the frozen protocol")
    _validate_formal_benchmark_identity(summary.get("benchmark"), require_asset_splits=False)
    conditions = summary.get("conditions")
    if (
        not isinstance(conditions, list)
        or not conditions
        or any(
            not isinstance(item, dict)
            or item.get("case_count") != FORMAL_BENCHMARK_CASE_COUNT
            or not isinstance(item.get("success_at_end_rate_balanced"), float)
            for item in conditions
        )
    ):
        raise ValueError("formal benchmark summary contains an incomplete condition")


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
    asset_terminal_rates = {
        asset: float(
            np.mean(
                [mapping[key].success_at_end for key in ordered_keys if key.asset_id == asset]
            )
        )
        for asset in assets
    }
    successful_times = completion[successes]
    time_averages = [
        mapping[key].time_average_combined_distance for key in ordered_keys
    ]
    airborne_times = [mapping[key].airborne_seconds for key in ordered_keys]
    success_rates_by_seconds = {
        str(seconds): float(
            np.mean(
                [
                    mapping[key].confirmed_step is not None
                    and mapping[key].confirmed_step
                    <= seconds * condition.control_fps
                    for key in ordered_keys
                ]
            )
        )
        for seconds in SUCCESS_CHECKPOINT_SECONDS
        if seconds <= horizon_seconds
    }
    summary = {
        "condition_id": condition.condition_id,
        "case_count": len(ordered_keys),
        "airborne_metric": condition.meta.get("airborne_metric"),
        "airborne_seconds_mean": (
            None if any(value is None for value in airborne_times)
            else float(np.mean(airborne_times))
        ),
        "success_at_end_count": int(successes_at_end.sum()),
        "success_at_end_rate_micro": float(successes_at_end.mean()),
        "success_at_end_rate_asset_macro": float(np.mean(list(asset_terminal_rates.values()))),
        "asset_success_at_end_rates": asset_terminal_rates,
        "success_count": int(successes.sum()),
        "success_rate_micro": float(successes.mean()),
        "success_rate_asset_macro": float(np.mean(list(asset_rates.values()))),
        "success_rate_by_seconds": success_rates_by_seconds,
        "asset_success_rates": asset_rates,
        "restricted_mean_completion_seconds": float(completion.mean()),
        "median_success_time_seconds": (
            None if not len(successful_times) else float(np.median(successful_times))
        ),
        "terminal_deformable_distance_mean": float(
            np.mean([mapping[key].terminal_deformable_distance for key in ordered_keys])
        ),
        "terminal_rigid_distance_mean": float(
            np.mean([mapping[key].terminal_rigid_distance for key in ordered_keys])
        ),
        "terminal_combined_distance_mean": float(
            np.mean([mapping[key].terminal_combined_distance for key in ordered_keys])
        ),
        # All best-observed components refer to the same minimum-combined frame.
        "best_observed_deformable_distance_mean": float(
            np.mean([mapping[key].best_observed_deformable_distance for key in ordered_keys])
        ),
        "best_observed_rigid_distance_mean": float(
            np.mean([mapping[key].best_observed_rigid_distance for key in ordered_keys])
        ),
        "best_observed_combined_distance_mean": float(
            np.mean([mapping[key].best_observed_combined_distance for key in ordered_keys])
        ),
        "terminal_minus_best_combined_distance_mean": float(
            np.mean(
                [
                    mapping[key].terminal_combined_distance
                    - mapping[key].best_observed_combined_distance
                    for key in ordered_keys
                ]
            )
        ),
        "time_average_combined_distance_mean": (
            None
            if any(value is None for value in time_averages)
            else float(np.mean(time_averages))
        ),
        "terminal_speed_rms_m_s_mean": float(
            np.mean([mapping[key].terminal_speed_rms_m_s for key in ordered_keys])
        ),
    }
    benchmark = condition.meta.get("benchmark")
    asset_splits = benchmark.get("asset_splits") if isinstance(benchmark, dict) else None
    if isinstance(asset_splits, dict):
        split_rates = {}
        for split in ("seen", "unseen"):
            split_assets = [asset for asset in assets if asset_splits.get(asset) == split]
            split_keys = [key for key in ordered_keys if key.asset_id in split_assets]
            split_rates[split] = float(
                np.mean([asset_terminal_rates[asset] for asset in split_assets])
            )
            summary[f"success_at_end_{split}_count"] = int(
                sum(mapping[key].success_at_end for key in split_keys)
            )
            summary[f"success_at_end_{split}_case_count"] = len(split_keys)
            summary[f"success_at_end_rate_{split}_asset_macro"] = split_rates[split]
        summary["success_at_end_rate_balanced"] = float(
            0.5 * split_rates["seen"] + 0.5 * split_rates["unseen"]
        )
    return summary


def _case_from_json(value: Any) -> EvaluationCaseResult:
    if not isinstance(value, dict):
        raise ValueError("evaluation case must be a JSON object")
    terminal = value.get("terminal")
    if not isinstance(terminal, dict):
        raise ValueError("evaluation case terminal metric is invalid")
    best = value.get("best_observed")
    if not isinstance(best, dict):
        raise ValueError("evaluation case best_observed metric is invalid")
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
        terminal_deformable_distance=float(terminal["deformable_distance"]),
        terminal_rigid_distance=float(terminal["rigid_distance"]),
        terminal_combined_distance=float(terminal["combined_distance"]),
        terminal_speed_rms_m_s=float(terminal["speed_rms_m_s"]),
        best_observed_step=int(best["step"]),
        best_observed_deformable_distance=float(best["deformable_distance"]),
        best_observed_rigid_distance=float(best["rigid_distance"]),
        best_observed_combined_distance=float(best["combined_distance"]),
        time_average_combined_distance=(
            None
            if value.get("time_average_combined_distance") is None
            else float(value["time_average_combined_distance"])
        ),
        airborne_seconds=value.get("airborne_seconds"),
    )
