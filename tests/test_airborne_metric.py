from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_evaluation import (
    _case,
    _offline_evaluation,
    _protocol,
    _publish,
    _template,
    _trajectory_state,
)

from clothdojo.evaluation.collection import score_evaluation_condition
from clothdojo.evaluation.metric import AIRBORNE_METRIC, cloth_airborne_mask
from clothdojo.evaluation.results import load_evaluation_condition, summarize_paired_conditions
from clothdojo.evaluation.scoring import score_evaluation_trajectory
from clothdojo.trajectory import TrajectoryAction, TrajectoryReader, TrajectoryWriter


def test_flat_and_small_wrinkles_are_not_airborne():
    template = _template()
    positions = np.repeat(template.particle_q[None], 3, axis=0)
    positions[1, :, 2] += 0.01
    positions[2, :, 2] += [0.0, 0.005, 0.01, 0.02]
    assert cloth_airborne_mask(positions, template).tolist() == [False, False, False]


def test_airborne_requires_half_vertices_strictly_above_height_margin():
    template = _template()
    positions = np.repeat(template.particle_q[None], 4, axis=0).astype(np.float64)
    threshold = float(np.median(template.particle_q[:, 2])) + AIRBORNE_METRIC["height_threshold_m"]
    positions[0, :, 2] = threshold
    positions[1, :1, 2] = threshold + 0.01
    positions[2, :2, 2] = threshold + 0.01
    positions[3, :, 2] = threshold + 0.01
    assert cloth_airborne_mask(positions, template).tolist() == [False, False, True, True]


def test_airborne_uses_template_median_height_and_ignores_xy_and_asset_scale():
    original = _template()
    points = original.particle_q.copy()
    points[:, 2] = [0.7, 0.7, 0.7, 0.72]
    template = replace(original, particle_q=points)
    positions = points[None].copy()
    positions[:, :, 2] = 0.8
    assert cloth_airborne_mask(positions, template).item()
    positions[:, :, :2] += [13.0, -20.0]
    scaled_template = replace(template, normalization_scale_m=100.0)
    assert cloth_airborne_mask(positions, scaled_template).item()
    positions[:, :, 2] = 0.72
    assert not cloth_airborne_mask(positions, scaled_template).item()


@pytest.mark.parametrize("bad_height", [np.nan, np.inf, -np.inf])
def test_airborne_rejects_non_finite_heights(bad_height):
    template = _template()
    positions = template.particle_q[None].copy()
    positions[0, 0, 2] = bad_height
    with pytest.raises(ValueError, match="finite"):
        cloth_airborne_mask(positions, template)


@pytest.mark.parametrize("shape", [(4, 3), (1, 3, 3), (1, 4, 2)])
def test_airborne_rejects_wrong_particle_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        cloth_airborne_mask(np.zeros(shape), _template())


def _fake_trajectory(monkeypatch, indicators, *, horizon_seconds=1.0):
    from clothdojo.evaluation import scoring

    protocol = _protocol(horizon_seconds=horizon_seconds)
    template = _template()
    positions = np.repeat(template.particle_q[None], protocol.horizon_steps + 1, axis=0)
    sample_steps = np.arange(0, protocol.horizon_steps + 1, protocol.metric_stride)
    assert len(indicators) == len(sample_steps)
    positions[sample_steps[np.asarray(indicators, dtype=bool)], :, 2] += 0.2
    arrays = {"particle_q": positions, "particle_qd": np.zeros_like(positions)}
    manifest = SimpleNamespace(
        control_fps=protocol.control_fps,
        action_count=protocol.horizon_steps,
        state_count=protocol.horizon_steps + 1,
        metadata={
            "training_eligible": False,
            "asset_id": template.asset_id,
            "initial_state": {"index": 0, "seed": 1},
            "replicate": 0,
            "policy_seed": 2,
        },
    )

    def state_arrays(names, *, mmap):
        assert names == ("particle_q", "particle_qd") and mmap is True
        return arrays

    reader = SimpleNamespace(manifest=manifest, state_arrays=state_arrays)
    monkeypatch.setattr(scoring, "TrajectoryReader", lambda _: reader)
    return template, protocol


@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize(
    ("indicators", "seconds"),
    [
        ([True] * 6, 1.0),
        ([False] * 6, 0.0),
        ([True, False, False, False, False, False], 0.1),
        ([False, False, False, False, False, True], 0.1),
        ([False, False, True, False, False, False], 0.2),
        ([False, False, True, True, True, False], 0.6),
    ],
)
def test_airborne_time_integrates_float_indicators_without_double_counting_endpoints(
    monkeypatch, batch_size, indicators, seconds
):
    template, protocol = _fake_trajectory(monkeypatch, indicators)
    scored = score_evaluation_trajectory("fake", template, protocol, metric_batch_size=batch_size)
    assert scored.result.airborne_seconds == pytest.approx(seconds)
    assert scored.curve.shape == (protocol.sample_count, 4)


def test_airborne_for_entire_ninety_seconds_is_full_horizon_not_half(monkeypatch):
    sample_count = _protocol(horizon_seconds=90.0).sample_count
    template, protocol = _fake_trajectory(monkeypatch, [True] * sample_count, horizon_seconds=90.0)
    scored = score_evaluation_trajectory("fake", template, protocol, metric_batch_size=7)
    assert scored.result.airborne_seconds == pytest.approx(90.0)


@pytest.mark.parametrize("lifted", [False, True])
def test_real_trajectory_score_and_publication_include_airborne_time(tmp_path, lifted):
    condition, protocol_path, templates, original = _offline_evaluation(tmp_path)
    assert original.result.airborne_seconds == 0.0
    if lifted:
        metadata_path = condition / "meta.json"
        metadata = json.loads(metadata_path.read_text())
        source = condition / metadata["trajectories"][0]["path"]
        manifest = TrajectoryReader(source).manifest
        protocol = _protocol()
        points = _template().particle_q.copy()
        points[:, 2] += 0.2
        writer = TrajectoryWriter(
            condition / "trajectories/lifted",
            control_fps=protocol.control_fps,
            sim_substeps=1,
            max_actions=protocol.horizon_steps,
            metadata=manifest.metadata,
        )
        writer.write_initial_state(_trajectory_state(0, points))
        for index in range(protocol.horizon_steps):
            writer.append_transition(
                TrajectoryAction(
                    sample_index=index,
                    joint_target=np.zeros(12, dtype=np.float32),
                    gripper_open_command=np.zeros(2, dtype=np.float32),
                ),
                _trajectory_state(index + 1, points),
            )
        writer.close()
        metadata["trajectories"][0]["path"] = "trajectories/lifted"
        metadata_path.write_text(json.dumps(metadata))

    score_evaluation_condition(condition, protocol_path, templates)
    published = load_evaluation_condition(condition)
    assert published.cases[0].airborne_seconds == pytest.approx(float(lifted))
    assert published.meta["airborne_metric"] == {**AIRBORNE_METRIC, "sample_hz": 5}
    serialized = json.loads((condition / "results.jsonl").read_text())
    assert serialized["airborne_seconds"] == pytest.approx(float(lifted))


@pytest.mark.parametrize("missing", [False, True])
def test_airborne_json_roundtrip_and_summary_do_not_treat_missing_as_zero(tmp_path, missing):
    condition = tmp_path / "condition"
    cases = [
        replace(_case("a", 0, success=True), airborne_seconds=0.0),
        replace(_case("a", 1, success=False), airborne_seconds=0.4),
        replace(_case("b", 0, success=False), airborne_seconds=0.8),
    ]
    _publish(condition, "airborne", cases)
    if missing:
        path = condition / "results.jsonl"
        serialized = [json.loads(line) for line in path.read_text().splitlines()]
        serialized[-1].pop("airborne_seconds")  # Historical JSONL has no field.
        path.write_text("\n".join(json.dumps(row) for row in serialized) + "\n")
    actual = load_evaluation_condition(condition)
    assert actual.cases[0].airborne_seconds == 0.0
    assert actual.cases[-1].airborne_seconds == (None if missing else 0.8)
    summary = summarize_paired_conditions([condition])["conditions"][0]
    assert summary["case_count"] == 3
    if missing:
        assert summary["airborne_seconds_mean"] is None
    else:
        assert actual.cases == tuple(cases)
        assert summary["airborne_seconds_mean"] == pytest.approx(0.4)


@pytest.mark.parametrize("seconds", [-0.1, np.nan, np.inf, True])
def test_case_result_rejects_invalid_airborne_seconds(seconds):
    with pytest.raises(ValueError, match="airborne_seconds"):
        replace(_case("a", 0, success=False), airborne_seconds=seconds)
