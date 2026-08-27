"""Offline evaluation derived solely from canonical trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clothdojo.evaluation.metric import CanonicalTemplate, factorized_cloth_distance
from clothdojo.evaluation.protocol import EvaluationProtocol
from clothdojo.evaluation.results import (
    EvaluationCaseKey,
    EvaluationCaseResult,
)
from clothdojo.trajectory import TrajectoryReader

METRIC_NAMES = (
    "deformable_distance",
    "rigid_distance",
    "combined_distance",
    "speed_rms_m_s",
)


@dataclass(frozen=True, slots=True)
class ScoredTrajectory:
    result: EvaluationCaseResult
    sample_steps: np.ndarray
    curve: np.ndarray


def score_evaluation_trajectory(
    trajectory: str | Path,
    template: CanonicalTemplate,
    protocol: EvaluationProtocol,
    *,
    metric_batch_size: int = 32,
) -> ScoredTrajectory:
    """Score one complete trajectory without Newton, a policy or a GPU."""

    if metric_batch_size < 1:
        raise ValueError("metric_batch_size must be positive")
    reader = TrajectoryReader(trajectory)
    manifest = reader.manifest
    if manifest.control_fps != protocol.control_fps:
        raise ValueError("evaluation trajectory and protocol control_fps differ")
    if manifest.action_count != protocol.horizon_steps:
        raise ValueError("evaluation trajectory does not reach the protocol horizon")
    if manifest.state_count != protocol.horizon_steps + 1:
        raise ValueError("evaluation trajectory state count differs from the protocol")
    metadata = manifest.metadata
    if metadata.get("training_eligible") is not False:
        raise ValueError("evaluation trajectory must be marked training_eligible=false")
    if metadata.get("asset_id") != template.asset_id:
        raise ValueError("evaluation trajectory and canonical template asset differ")

    arrays = reader.state_arrays(("particle_q", "particle_qd"))
    sample_steps = np.arange(
        0,
        protocol.horizon_steps + 1,
        protocol.metric_stride,
        dtype=np.int32,
    )
    curve = np.empty((protocol.sample_count, len(METRIC_NAMES)), dtype=np.float32)
    for start in range(0, protocol.sample_count, metric_batch_size):
        stop = min(start + metric_batch_size, protocol.sample_count)
        selection = sample_steps[start:stop]
        values = factorized_cloth_distance(
            arrays["particle_q"][selection],
            arrays["particle_qd"][selection],
            template,
            deformable_weight=protocol.deformable_weight,
            symmetry=protocol.symmetry,
            symmetry_center_xy=protocol.target.symmetry_center_xy,
        )
        curve[start:stop] = np.stack(
            (
                values.deformable_distance,
                values.rigid_distance,
                values.combined_distance,
                values.speed_rms_m_s,
            ),
            axis=1,
        )

    key = EvaluationCaseKey(
        str(metadata["asset_id"]),
        _non_negative_integer(metadata["initial_state"]["index"], "initial_state index"),
        _non_negative_integer(metadata["replicate"], "replicate"),
    )
    result = result_from_curve(
        curve,
        protocol,
        key=key,
        initial_state_seed=_non_negative_integer(
            metadata["initial_state"]["seed"], "initial_state seed"
        ),
        policy_seed=_non_negative_integer(metadata["policy_seed"], "policy seed"),
    )
    sample_steps.setflags(write=False)
    curve.setflags(write=False)
    return ScoredTrajectory(result=result, sample_steps=sample_steps, curve=curve)


def result_from_curve(
    curve: np.ndarray,
    protocol: EvaluationProtocol,
    *,
    key: EvaluationCaseKey,
    initial_state_seed: int,
    policy_seed: int,
) -> EvaluationCaseResult:
    """Derive terminal, best-observed and absorbing success from one curve."""

    values = np.asarray(curve, dtype=np.float32)
    expected = (protocol.sample_count, len(METRIC_NAMES))
    if values.shape != expected or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"evaluation curve must be finite and non-negative with shape {expected}")
    deformable = values[:, 0]
    rigid = values[:, 1]
    combined = values[:, 2]
    candidate = deformable <= protocol.deformable_distance_max
    if protocol.rigid_distance_max is not None:
        candidate &= rigid <= protocol.rigid_distance_max
    success = bool(candidate.any())
    confirmed_step = (
        int(candidate.argmax()) * protocol.metric_stride if success else None
    )
    best = int(combined.argmin())
    return EvaluationCaseResult(
        key=key,
        initial_state_seed=int(initial_state_seed),
        policy_seed=int(policy_seed),
        success=success,
        success_at_end=bool(candidate[-1]),
        confirmed_step=confirmed_step,
        terminal_deformable_distance=float(deformable[-1]),
        terminal_rigid_distance=float(rigid[-1]),
        terminal_combined_distance=float(combined[-1]),
        terminal_speed_rms_m_s=float(values[-1, 3]),
        best_observed_step=best * protocol.metric_stride,
        best_observed_deformable_distance=float(deformable[best]),
        best_observed_rigid_distance=float(rigid[best]),
        best_observed_combined_distance=float(combined[best]),
        time_average_combined_distance=float(
            np.trapz(combined, dx=protocol.metric_stride)
            / protocol.horizon_steps
        ),
    )


def _non_negative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
