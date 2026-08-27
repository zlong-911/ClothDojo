from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clothdojo.data.rollout import inspect_rollout_collection_export
from clothdojo.evaluation.collection import score_evaluation_condition
from clothdojo.evaluation.metric import (
    CanonicalTemplate,
    factorized_cloth_distance,
    file_sha256,
    load_canonical_template,
    save_canonical_template,
)
from clothdojo.evaluation.protocol import (
    EvaluationProtocol,
    SuccessTracker,
    load_evaluation_protocol,
)
from clothdojo.evaluation.results import (
    EVALUATION_RESULT_SCHEMA,
    EvaluationCaseKey,
    EvaluationCaseResult,
    load_evaluation_condition,
    summarize_paired_conditions,
    write_evaluation_cases,
)
from clothdojo.evaluation.scoring import result_from_curve, score_evaluation_trajectory
from clothdojo.sim.scene import ShirtFlattenSpec
from clothdojo.trajectory import TrajectoryAction, TrajectoryState, TrajectoryWriter

TABLE_CENTER_XY = ShirtFlattenSpec().table_position[:2]


def _template() -> CanonicalTemplate:
    points = np.asarray(
        [
            [-1.0, -0.5, 0.0],
            [1.0, -0.5, 0.0],
            [0.7, 0.5, 0.0],
            [-0.7, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    return CanonicalTemplate(
        asset_id="shirt",
        asset_sha256="a" * 64,
        topology_sha256="b" * 64,
        scene_physics_sha256="c" * 64,
        source_kind="test",
        source_sha256="d" * 64,
        normalization_scale_m=float(np.linalg.norm(np.ptp(points[:, :2], axis=0))),
        particle_q=points,
    )


def _protocol(**overrides) -> EvaluationProtocol:
    values = {
        "protocol_id": "test",
        "stage": "calibration",
        "task_name": "shirt_flatten",
        "control_fps": 10,
        "horizon_seconds": 1.0,
        "metric_hz": 5,
        "deformable_weight": 0.65,
        "symmetry": "planar_180",
        "deformable_distance_max": 0.1,
        "rigid_distance_max": None,
    }
    values.update(overrides)
    return EvaluationProtocol(**values)


def test_factorized_metric_separates_shape_from_planar_pose() -> None:
    template = _template()
    angle = np.deg2rad(35.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    transformed = template.particle_q.copy()
    transformed[:, :2] = transformed[:, :2] @ rotation.T + [0.4, -0.3]
    velocity = np.zeros_like(transformed)

    values = factorized_cloth_distance(transformed[None], velocity[None], template, symmetry="none")

    assert values.deformable_distance[0] == pytest.approx(0.0, abs=1.0e-6)
    assert values.rigid_distance[0] > 0.1
    assert values.combined_distance[0] == pytest.approx(0.35 * values.rigid_distance[0], rel=1.0e-5)

    translated = template.particle_q.copy()
    translated[:, :2] += [0.4, -0.3]
    translation_only = factorized_cloth_distance(
        translated[None], velocity[None], template, symmetry="none"
    )
    assert translation_only.deformable_distance[0] == pytest.approx(0.0, abs=1.0e-6)
    assert translation_only.rigid_distance[0] == pytest.approx(0.0, abs=1.0e-6)
    assert translation_only.combined_distance[0] == pytest.approx(0.0, abs=1.0e-6)


def test_factorized_metric_detects_deformation_and_180_symmetry() -> None:
    template = _template()
    deformed = template.particle_q.copy()
    deformed[0, 0] -= 0.5
    zero = np.zeros_like(deformed)
    distance = factorized_cloth_distance(
        deformed[None],
        zero[None],
        template,
        symmetry="planar_180",
        symmetry_center_xy=TABLE_CENTER_XY,
    )
    assert distance.deformable_distance[0] > 0.0

    rotated = template.particle_q.copy()
    rotated[:, :2] = 2.0 * np.asarray(TABLE_CENTER_XY) - rotated[:, :2]
    symmetric = factorized_cloth_distance(
        rotated[None],
        zero[None],
        template,
        symmetry="planar_180",
        symmetry_center_xy=TABLE_CENTER_XY,
    )
    assert symmetric.combined_distance[0] == pytest.approx(0.0, abs=1.0e-6)
    assert symmetric.symmetry_index[0] == 1

    centroid_rotated = template.particle_q.copy()
    centroid = centroid_rotated[:, :2].mean(axis=0)
    centroid_rotated[:, :2] = 2.0 * centroid - centroid_rotated[:, :2]
    translated_symmetric = factorized_cloth_distance(
        centroid_rotated[None],
        zero[None],
        template,
        symmetry="planar_180",
        symmetry_center_xy=TABLE_CENTER_XY,
    )
    assert translated_symmetric.combined_distance[0] == pytest.approx(0.0, abs=1.0e-6)
    assert translated_symmetric.symmetry_index[0] == 1


def test_planar_180_requires_an_explicit_finite_center() -> None:
    template = _template()
    state = template.particle_q[None]
    velocity = np.zeros_like(state)

    with pytest.raises(ValueError, match="symmetry_center_xy"):
        factorized_cloth_distance(state, velocity, template, symmetry="planar_180")
    with pytest.raises(ValueError, match="symmetry_center_xy"):
        factorized_cloth_distance(
            state,
            velocity,
            template,
            symmetry="planar_180",
            symmetry_center_xy=(np.nan, 0.0),
        )


def test_success_only_uses_distance_thresholds_and_is_absorbing() -> None:
    protocol = _protocol(rigid_distance_max=0.1)
    tracker = SuccessTracker(protocol, batch_size=2)
    good = np.asarray([0.05, 0.05])
    rigid = np.asarray([0.05, 0.2])

    new = tracker.update(
        step=0,
        deformable_distance=good,
        rigid_distance=rigid,
    )
    np.testing.assert_array_equal(new, [True, False])
    assert tracker.confirmed_step.tolist() == [0, -1]

    new = tracker.update(
        step=2,
        deformable_distance=good,
        rigid_distance=np.asarray([0.2, 0.05]),
    )
    np.testing.assert_array_equal(new, [False, True])
    assert tracker.confirmed_step.tolist() == [0, 2]
    assert tracker.success.tolist() == [True, True]

    tracker.update(
        step=4,
        deformable_distance=np.asarray([1.0, 1.0]),
        rigid_distance=rigid,
    )
    assert tracker.success.tolist() == [True, True]


def test_offline_curve_is_the_single_source_for_success_and_best_results() -> None:
    protocol = _protocol(rigid_distance_max=0.1)
    curve = np.asarray(
        [
            [0.2, 0.05, 0.2, 0.01],
            [0.1, 0.1, 0.1, 0.03],
            [0.3, 0.3, 0.1, 0.05],
            [0.4, 0.4, 0.4, 0.07],
            [0.5, 0.5, 0.5, 0.09],
            [0.6, 0.6, 0.6, 0.11],
        ],
        dtype=np.float32,
    )

    result = result_from_curve(
        curve,
        protocol,
        key=EvaluationCaseKey("shirt", 3, 1),
        initial_state_seed=7,
        policy_seed=8,
    )

    assert result.success is True
    assert result.success_at_end is False
    assert result.confirmed_step == 2
    assert result.best_observed_step == 2
    assert result.best_observed_combined_distance == pytest.approx(0.1)
    assert result.time_average_combined_distance == pytest.approx(0.3)


def test_protocol_and_template_round_trip(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    protocol = load_evaluation_protocol(project / "configs" / "evaluation" / "shirt_flatten.yaml")
    assert protocol.stage == "calibration"
    assert protocol.horizon_steps == 4500
    assert protocol.metric_stride == 5
    assert protocol.target.symmetry_center_xy == TABLE_CENTER_XY
    assert protocol.target_min_settle_steps == 400
    assert protocol.target_max_settle_steps == 1500
    assert protocol.target_stability_window_steps == 50
    assert protocol.target_stability_samples == 10
    assert len(protocol.fingerprint) == 64

    target = tmp_path / "template.npz"
    save_canonical_template(target, _template())
    loaded = load_canonical_template(target)
    assert loaded.asset_id == "shirt"
    np.testing.assert_array_equal(loaded.particle_q, _template().particle_q)


def test_protocol_rejects_non_integral_sampling_grid() -> None:
    with pytest.raises(ValueError, match="divide"):
        _protocol(metric_hz=6)

    with pytest.raises(ValueError, match="metric sampling grid"):
        _protocol(horizon_seconds=0.3)


def test_evaluation_is_scored_from_the_standard_complete_trajectory(tmp_path: Path) -> None:
    protocol = _protocol()
    template = _template()
    condition = tmp_path / "condition"
    output = condition / "trajectories" / "shirt_000003_r002"
    templates = tmp_path / "templates"
    (templates / "templates").mkdir(parents=True)
    template_path = templates / "templates" / "shirt.npz"
    save_canonical_template(template_path, template)
    (templates / "meta.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_canonical_template_collection",
                "format_version": 1,
                "status": "complete",
                "asset_count": 1,
                "protocol": {
                    "fingerprint": protocol.fingerprint,
                    "target": protocol.target.to_json_dict(),
                },
                "assets": [
                    {
                        "asset_id": "shirt",
                        "status": "complete",
                        "path": "templates/shirt.npz",
                        "sha256": file_sha256(template_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    writer = TrajectoryWriter(
        output,
        control_fps=protocol.control_fps,
        sim_substeps=1,
        max_actions=protocol.horizon_steps,
        metadata={
            "producer": "clothdojo_evaluation",
            "training_eligible": False,
            "asset_id": "shirt",
            "replicate": 2,
            "policy_seed": 8,
            "initial_state": {"index": 3, "seed": 7},
            "protocol_fingerprint": protocol.fingerprint,
            "template_sha256": file_sha256(template_path),
        },
    )
    writer.write_initial_state(_trajectory_state(0, template.particle_q))
    deformed = template.particle_q.copy()
    deformed[0, 0] -= 1.0
    for index in range(protocol.horizon_steps):
        writer.append_transition(
            TrajectoryAction(
                sample_index=index,
                joint_target=np.zeros(12, dtype=np.float32),
                gripper_open_command=np.zeros(2, dtype=np.float32),
            ),
            _trajectory_state(index + 1, deformed),
        )
    writer.close()

    scored = score_evaluation_trajectory(output, template, protocol)

    assert scored.result.key == EvaluationCaseKey("shirt", 3, 2)
    assert scored.result.success is True
    assert scored.result.success_at_end is False
    assert scored.result.confirmed_step == 0
    assert scored.curve.shape == (protocol.sample_count, 4)

    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        """schema: clothdojo_evaluation_protocol
protocol_id: test
stage: calibration
task: shirt_flatten
timing: {control_fps: 10, horizon_seconds: 1.0, metric_hz: 5}
metric: {deformable_weight: 0.65, symmetry: planar_180}
success: {deformable_distance_max: 0.1, rigid_distance_max: null}
target:
  method: default_free_settle
  min_settle_seconds: 8.0
  max_settle_seconds: 30.0
  stability_window_seconds: 1.0
  speed_rms_max_m_s: 0.005
  robot_particle_collision: false
  symmetry_center_xy: [0.651, 0.0]
""",
        encoding="utf-8",
    )
    condition_meta = {
        "schema": EVALUATION_RESULT_SCHEMA,
        "status": "failed",
        "condition_id": "offline",
        "case_count": 1,
        "results": "results.jsonl",
        "training_eligible": False,
        "scenario": {"prompt": "flatten the shirt"},
        "policy": {"checkpoint_sha256": "f" * 64},
        "protocol": {"fingerprint": protocol.fingerprint},
        "templates": {"shirt": file_sha256(template_path)},
        "trajectories": [
            {
                "asset_id": "shirt",
                "initial_state_index": 3,
                "replicate": 2,
                "path": "trajectories/shirt_000003_r002",
                "action_count": protocol.horizon_steps,
            }
        ],
    }
    (condition / "meta.json").write_text(
        json.dumps(condition_meta), encoding="utf-8"
    )

    summary = score_evaluation_condition(condition, protocol_path, templates, workers=2)

    assert summary["case_count"] == 1
    assert load_evaluation_condition(condition).cases == (scored.result,)
    export = inspect_rollout_collection_export(condition)
    assert export["source_kind"] == "evaluation_condition"
    assert export["training_eligible"] is False


def _trajectory_state(index: int, particles: np.ndarray) -> TrajectoryState:
    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    return TrajectoryState(
        sample_index=index,
        joint_q=zeros(12),
        joint_qd=zeros(12),
        body_q=zeros((2, 7)),
        body_qd=zeros((2, 6)),
        particle_q=np.asarray(particles, dtype=np.float32),
        particle_qd=zeros(particles.shape),
        gripper_open_state=zeros(2),
        grasp_attached=np.zeros(2, dtype=np.int32),
        grasp_previous_open=zeros(2),
        grasp_pin_count=np.zeros(2, dtype=np.int32),
        grasp_pins=np.full((2, 2), -1, dtype=np.int32),
        grasp_pin_local=zeros((2, 2, 3)),
    )


def _case(asset: str, index: int, *, success: bool) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        key=EvaluationCaseKey(asset, index),
        initial_state_seed=100 + index,
        policy_seed=200 + index,
        success=success,
        success_at_end=success,
        confirmed_step=4 if success else None,
        terminal_deformable_distance=0.05 if success else 0.3,
        terminal_rigid_distance=0.1,
        terminal_combined_distance=0.07 if success else 0.23,
        terminal_speed_rms_m_s=0.01 if success else 0.2,
        best_observed_step=4,
        best_observed_deformable_distance=0.05 if success else 0.2,
        best_observed_rigid_distance=0.1,
        best_observed_combined_distance=0.07 if success else 0.17,
    )


def _publish(
    path: Path,
    condition: str,
    cases,
    *,
    scenario: dict | None = None,
    initial_state_pools: dict | None = None,
    horizon_steps: int = 10,
    control_fps: int = 10,
) -> None:
    path.mkdir()
    ordered = tuple(cases)
    assets = {case.key.asset_id for case in ordered}
    write_evaluation_cases(path / "results.jsonl", ordered)
    meta = {
        "schema": EVALUATION_RESULT_SCHEMA,
        "status": "complete",
        "condition_id": condition,
        "protocol": {
            "id": "test",
            "stage": "frozen",
            "fingerprint": "d" * 64,
            "horizon_steps": horizon_steps,
            "control_fps": control_fps,
        },
        "scenario": scenario or {"fingerprint": "2" * 64},
        "initial_states_sha256": "e" * 64,
        "initial_state_pools": initial_state_pools or dict.fromkeys(assets, "0" * 64),
        "templates": dict.fromkeys(assets, "1" * 64),
        "policy": {"checkpoint_sha256": "f" * 64},
        "execution": {"mode": "chunk", "exec_horizon": 2},
        "case_count": len(ordered),
        "results": "results.jsonl",
    }
    (path / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_paired_summary_uses_macro_assets_and_failure_censoring(tmp_path: Path) -> None:
    keys = [("a", 0), ("a", 1), ("b", 0)]
    _publish(
        tmp_path / "base",
        "base",
        [_case(asset, index, success=False) for asset, index in keys],
    )
    _publish(
        tmp_path / "rtc",
        "rtc",
        [
            _case("a", 0, success=True),
            _case("a", 1, success=False),
            _case("b", 0, success=True),
        ],
    )

    summary = summarize_paired_conditions([tmp_path / "base", tmp_path / "rtc"])

    rtc = summary["conditions"][1]
    assert rtc["success_rate_micro"] == pytest.approx(2 / 3)
    assert rtc["success_rate_asset_macro"] == pytest.approx(0.75)
    assert rtc["restricted_mean_completion_seconds"] == pytest.approx(0.6)
    assert rtc["terminal_speed_rms_m_s_mean"] == pytest.approx(0.0733333333)
    assert rtc["best_observed_combined_distance_mean"] == pytest.approx(0.1033333333)
    assert rtc["terminal_minus_best_combined_distance_mean"] == pytest.approx(0.02)
    assert rtc["time_average_combined_distance_mean"] is None
    assert summary["paired_comparisons"][0]["success_wins"] == 2
    assert summary["paired_comparisons"][0]["success_losses"] == 0


def test_summary_reports_preregistered_success_checkpoints(tmp_path: Path) -> None:
    fast = replace(
        _case("a", 0, success=True),
        confirmed_step=250,
    )
    slow = replace(
        _case("a", 1, success=True),
        confirmed_step=700,
    )
    _publish(
        tmp_path / "condition",
        "condition",
        [fast, slow],
        horizon_steps=900,
        control_fps=10,
    )

    summary = summarize_paired_conditions([tmp_path / "condition"])

    assert summary["conditions"][0]["success_rate_by_seconds"] == {
        "30": 0.5,
        "45": 0.5,
        "60": 0.5,
        "75": 1.0,
        "90": 1.0,
    }


def test_paired_summary_rejects_missing_cases(tmp_path: Path) -> None:
    _publish(tmp_path / "base", "base", [_case("a", 0, success=False)])
    _publish(tmp_path / "other", "other", [_case("a", 1, success=False)])

    with pytest.raises(ValueError, match="same paired cases"):
        summarize_paired_conditions([tmp_path / "base", tmp_path / "other"])


def test_paired_summary_rejects_different_policy_seeds(tmp_path: Path) -> None:
    _publish(tmp_path / "base", "base", [_case("a", 0, success=False)])
    changed = _case("a", 0, success=False)
    changed = replace(changed, policy_seed=changed.policy_seed + 1)
    _publish(tmp_path / "other", "other", [changed])

    with pytest.raises(ValueError, match="identical state and policy seeds"):
        summarize_paired_conditions([tmp_path / "base", tmp_path / "other"])


def test_paired_summary_rejects_different_scenarios(tmp_path: Path) -> None:
    case = _case("a", 0, success=False)
    _publish(tmp_path / "base", "base", [case])
    _publish(
        tmp_path / "other",
        "other",
        [case],
        scenario={"fingerprint": "3" * 64},
    )

    with pytest.raises(ValueError, match="scenario"):
        summarize_paired_conditions([tmp_path / "base", tmp_path / "other"])


def test_paired_summary_rejects_different_initial_state_pool_files(
    tmp_path: Path,
) -> None:
    case = _case("a", 0, success=False)
    _publish(tmp_path / "base", "base", [case])
    _publish(
        tmp_path / "other",
        "other",
        [case],
        initial_state_pools={"a": "4" * 64},
    )

    with pytest.raises(ValueError, match="initial-state pool"):
        summarize_paired_conditions([tmp_path / "base", tmp_path / "other"])


def test_paired_summary_allows_extended_state_collection_metadata(tmp_path: Path) -> None:
    case = _case("a", 0, success=False)
    _publish(tmp_path / "base", "base", [case])
    _publish(tmp_path / "other", "other", [case])
    meta_path = tmp_path / "other" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["initial_states_sha256"] = "9" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    summary = summarize_paired_conditions([tmp_path / "base", tmp_path / "other"])

    assert summary["case_count"] == 1
