from __future__ import annotations

import json

import numpy as np

from clothdojo.rollout.parity import (
    ConversionParityLimits,
    _error_metrics,
    _parity_violations,
)


def test_error_metrics_report_distribution_and_direction() -> None:
    reference = np.ones((1, 2, 2), dtype=np.float32)
    actual = reference + np.asarray([[[0.0, 0.01], [0.02, 0.03]]], dtype=np.float32)

    metrics = _error_metrics(reference, actual)

    assert metrics["shape"] == [1, 2, 2]
    assert np.isclose(metrics["mean_abs"], 0.015)
    assert np.isclose(metrics["max_abs"], 0.03)
    assert metrics["max_abs_index"] == [0, 1, 1]
    assert metrics["cosine"] > 0.9999
    json.dumps(metrics)


def test_parity_rejects_executed_prefix_outlier() -> None:
    passing = {
        "mean_abs": 0.0,
        "p99_abs": 0.0,
        "max_abs": 0.0,
        "cosine": 1.0,
    }
    metrics = {
        "model_1_step_full": dict(passing),
        "model_1_step_executed": dict(passing),
        "model_10_step_full": dict(passing),
        "model_10_step_executed": {**passing, "max_abs": 0.021},
        "model_10_step_rtc_guidance": dict(passing),
        "final_action_executed": dict(passing),
    }

    violations = _parity_violations(metrics, ConversionParityLimits())

    assert violations == ["model_10_step_executed.max_abs=0.021>0.02"]


def test_parity_rejects_rtc_guidance_outlier() -> None:
    passing = {
        "mean_abs": 0.0,
        "p99_abs": 0.0,
        "max_abs": 0.0,
        "cosine": 1.0,
    }
    metrics = {
        "model_1_step_full": dict(passing),
        "model_1_step_executed": dict(passing),
        "model_10_step_full": dict(passing),
        "model_10_step_executed": dict(passing),
        "model_10_step_rtc_guidance": {**passing, "max_abs": 0.021},
        "final_action_executed": dict(passing),
    }

    violations = _parity_violations(metrics, ConversionParityLimits())

    assert violations == ["model_10_step_rtc_guidance.max_abs=0.021>0.02"]


def test_parity_treats_full_tensor_max_as_diagnostic() -> None:
    passing = {
        "mean_abs": 0.0,
        "p99_abs": 0.0,
        "max_abs": 0.0,
        "cosine": 1.0,
    }
    metrics = {
        "model_1_step_full": {**passing, "max_abs": 1.0},
        "model_1_step_executed": dict(passing),
        "model_10_step_full": {**passing, "max_abs": 1.0},
        "model_10_step_executed": dict(passing),
        "model_10_step_rtc_guidance": dict(passing),
        "final_action_executed": dict(passing),
    }

    assert _parity_violations(metrics, ConversionParityLimits()) == []
