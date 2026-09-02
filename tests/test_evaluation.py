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
    FORMAL_BENCHMARK_ASSET_SPLITS_SHA256,
    FORMAL_BENCHMARK_CASE_COUNT,
    FORMAL_BENCHMARK_INITIAL_STATES_SHA256,
    FORMAL_BENCHMARK_PANEL_ID,
    FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT,
    FORMAL_BENCHMARK_SEEN_ASSETS,
    FORMAL_BENCHMARK_SELECTION_SHA256,
    FORMAL_BENCHMARK_STATES_PER_ASSET,
    FORMAL_BENCHMARK_UNSEEN_ASSETS,
    EvaluationCaseKey,
    EvaluationCaseResult,
    load_evaluation_condition,
    summarize_paired_conditions,
    validate_formal_evaluation_summary,
    write_evaluation_cases,
)
from clothdojo.evaluation.scoring import result_from_curve, score_evaluation_trajectory
from clothdojo.rendering import policy_observation_settings
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


@pytest.mark.parametrize("symmetry", ["planar_180", "planar_90"])
def test_planar_symmetry_requires_an_explicit_finite_center(symmetry: str) -> None:
    template = _template()
    state = template.particle_q[None]
    velocity = np.zeros_like(state)

    with pytest.raises(ValueError, match="symmetry_center_xy"):
        factorized_cloth_distance(state, velocity, template, symmetry=symmetry)
    with pytest.raises(ValueError, match="symmetry_center_xy"):
        factorized_cloth_distance(
            state,
            velocity,
            template,
            symmetry=symmetry,
            symmetry_center_xy=(np.nan, 0.0),
        )


def _rotate_target(points: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    result = points.astype(np.float64).copy()
    center = np.asarray(TABLE_CENTER_XY)
    result[:, :2] = (result[:, :2] - center) @ rotation.T + center
    return result


def test_quarter_turns_are_equivalent_but_diagonal_headings_are_not() -> None:
    template = _template()
    states = np.stack([_rotate_target(template.particle_q, angle)
                       for angle in (0, 90, 180, 270, 45, 135)])
    zero = np.zeros_like(states)
    values = factorized_cloth_distance(
        states, zero, template, symmetry_center_xy=TABLE_CENTER_XY
    )
    np.testing.assert_allclose(values.deformable_distance, 0.0, atol=1e-6)
    np.testing.assert_allclose(values.rigid_distance[:4], 0.0, atol=1e-6)
    np.testing.assert_array_equal(values.symmetry_index[:4], [0, 1, 2, 3])
    assert np.all(values.rigid_distance[4:] > 0.056)
    legacy = factorized_cloth_distance(
        states, zero, template, symmetry="planar_180", symmetry_center_xy=TABLE_CENTER_XY
    )
    assert np.all(legacy.rigid_distance[[1, 3]] > 0.056)


@pytest.mark.parametrize("weight", [0.0, 0.65, 1.0])
def test_quarter_turn_metric_matches_explicit_rotated_vertex_targets(weight: float) -> None:
    template = _template()
    rng = np.random.default_rng(173)
    positions = np.stack([_rotate_target(template.particle_q, angle)
                          for angle in (4, 48, 90, 178, 223, 271)])
    positions += rng.normal(0, 0.13, positions.shape)
    velocities = rng.normal(0, 0.04, positions.shape)
    actual = factorized_cloth_distance(
        positions, velocities, template, deformable_weight=weight,
        symmetry="planar_90", symmetry_center_xy=TABLE_CENTER_XY,
    )
    expected_deformable, expected_rigid, expected_indices = [], [], []
    for current in positions:
        candidates = []
        xy_mean = current[:, :2].mean(0)
        xy = current[:, :2] - xy_mean
        for heading in (0, 90, 180, 270):
            goal = _rotate_target(template.particle_q, heading)
            centered = goal[:, :2] - goal[:, :2].mean(0)
            theta = np.arctan2(
                np.sum(centered[:, 0] * xy[:, 1] - centered[:, 1] * xy[:, 0]),
                np.sum(centered * xy),
            )
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
            aligned_xy = centered @ rotation.T
            aligned = np.column_stack((aligned_xy + xy_mean, goal[:, 2]))
            candidates.append((np.linalg.norm(current - aligned, axis=1).mean(),
                               np.linalg.norm(aligned_xy - centered, axis=1).mean()))
        index = int(np.argmin([r for _, r in candidates]))
        deformable, rigid = candidates[index]
        expected_deformable.append(deformable / template.normalization_scale_m)
        expected_rigid.append(rigid / template.normalization_scale_m)
        expected_indices.append(index)
    np.testing.assert_allclose(actual.deformable_distance, expected_deformable, atol=1e-7)
    np.testing.assert_allclose(actual.rigid_distance, expected_rigid, atol=1e-7)
    np.testing.assert_array_equal(actual.symmetry_index, expected_indices)
    np.testing.assert_allclose(actual.combined_distance,
        weight * np.array(expected_deformable) + (1 - weight) * np.array(expected_rigid), atol=1e-7)
    for symmetry in ("none", "planar_180"):
        reference = factorized_cloth_distance(
            positions, velocities, template, deformable_weight=weight,
            symmetry=symmetry, symmetry_center_xy=TABLE_CENTER_XY,
        )
        np.testing.assert_array_equal(actual.deformable_distance, reference.deformable_distance)
        np.testing.assert_array_equal(actual.speed_rms_m_s, reference.speed_rms_m_s)
        assert np.all(actual.rigid_distance <= reference.rigid_distance + 1e-7)


@pytest.mark.parametrize("heading", [0, 90, 180, 270])
def test_quarter_turn_symmetry_preserves_flip_height_and_fold_penalties(heading: int) -> None:
    template = _template()
    flipped = template.particle_q.copy()
    flipped[:, 1:] *= -1  # A physical upside-down rotation around x, not a z rotation.
    lifted = template.particle_q.copy()
    lifted[:, 2] += 0.5
    folded = template.particle_q.copy()
    folded[0, :2] = folded[2, :2]
    states = np.stack([_rotate_target(q, heading) for q in (flipped, lifted, folded)])
    values = factorized_cloth_distance(
        states, np.zeros_like(states), template, symmetry_center_xy=TABLE_CENTER_XY
    )
    assert np.all(values.deformable_distance > 0.093)


def test_quarter_turn_metric_ignores_xy_translation_and_remains_asset_normalized() -> None:
    template = _template()
    states = _rotate_target(template.particle_q, 123)[None]
    states[0, 0, 2] += 0.2
    velocities = np.zeros_like(states)
    reference = factorized_cloth_distance(
        states, velocities, template, symmetry_center_xy=TABLE_CENTER_XY
    )
    translated = states.copy()
    translated[:, :, :2] += [11.0, -7.0]
    shifted = factorized_cloth_distance(
        translated, velocities, template, symmetry_center_xy=(-3.0, 8.0)
    )
    scaled = factorized_cloth_distance(
        states * 2, velocities, replace(template, particle_q=template.particle_q * 2,
            normalization_scale_m=template.normalization_scale_m * 2),
        symmetry_center_xy=TABLE_CENTER_XY,
    )
    for metric in ("deformable_distance", "rigid_distance", "combined_distance"):
        np.testing.assert_allclose(getattr(shifted, metric), getattr(reference, metric), atol=1e-7)
        np.testing.assert_allclose(getattr(scaled, metric), getattr(reference, metric), atol=1e-7)


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
    assert protocol.symmetry == "planar_90"
    assert protocol.fingerprint == FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT
    assert replace(protocol, symmetry="planar_180").fingerprint != protocol.fingerprint
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


def _offline_evaluation(tmp_path: Path, *, quarter_turn: bool = False):
    protocol = _protocol()
    template = _template()
    collection_identity = {
        "condition_id": "offline",
        "policy": {"checkpoint_sha256": "f" * 64},
        "execution": {"mode": "chunk", "exec_horizon": 2},
        "cloth_visual": "nocs",
        "observation": policy_observation_settings(),
    }
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
            **collection_identity,
            "producer": "clothdojo_evaluation",
            "training_eligible": False,
            "asset_id": "shirt",
            "replicate": 2,
            "policy_seed": 8,
            "initial_state": {"index": 3, "seed": 7, "pool_sha256": "e" * 64},
            "protocol_fingerprint": protocol.fingerprint,
            "template_sha256": file_sha256(template_path),
        },
    )
    writer.write_initial_state(_trajectory_state(0, template.particle_q))
    deformed = template.particle_q.copy()
    deformed[0, 0] -= 1.0
    if quarter_turn:
        deformed[:, :2] = deformed[:, :2] @ np.asarray([[0.0, 1.0], [-1.0, 0.0]])
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
        **collection_identity,
        "schema": EVALUATION_RESULT_SCHEMA,
        "status": "failed",
        "case_count": 1,
        "results": "results.jsonl",
        "training_eligible": False,
        "scenario": {"prompt": "flatten the shirt"},
        "protocol": {"fingerprint": protocol.fingerprint},
        "initial_state_pools": {"shirt": "e" * 64},
        "templates": {"shirt": file_sha256(template_path)},
        "trajectories": [
            {
                "asset_id": "shirt",
                "initial_state_index": 3,
                "replicate": 2,
                "seed": 7,
                "policy_seed": 8,
                "path": "trajectories/shirt_000003_r002",
                "action_count": protocol.horizon_steps,
            }
        ],
    }
    (condition / "meta.json").write_text(
        json.dumps(condition_meta), encoding="utf-8"
    )
    return condition, protocol_path, templates, scored


def test_evaluation_is_scored_from_the_standard_complete_trajectory(tmp_path: Path) -> None:
    condition, protocol_path, templates, scored = _offline_evaluation(tmp_path)
    assert scored.result.key == EvaluationCaseKey("shirt", 3, 2)
    assert scored.result.success is True
    assert scored.result.success_at_end is False
    assert scored.result.confirmed_step == 0
    assert scored.curve.shape == (_protocol().sample_count, 4)

    summary = score_evaluation_condition(condition, protocol_path, templates, workers=2)

    assert summary["case_count"] == 1
    assert load_evaluation_condition(condition).cases == (scored.result,)
    export = inspect_rollout_collection_export(condition)
    assert export["source_kind"] == "evaluation_condition"
    assert export["training_eligible"] is False


def test_rescore_requires_original_protocol_and_preserves_collection(tmp_path: Path) -> None:
    condition, original_path, templates, old_scores = _offline_evaluation(
        tmp_path, quarter_turn=True
    )
    original = load_evaluation_protocol(original_path)
    updated = replace(original, symmetry="planar_90", rigid_distance_max=0.05)
    updated_path = tmp_path / "updated.yaml"
    updated_path.write_text(json.dumps(updated.to_json_dict()), encoding="utf-8")
    original_meta = json.loads((condition / "meta.json").read_text())
    trajectory_manifest = condition / "trajectories/shirt_000003_r002/manifest.json"
    trajectory_bytes = trajectory_manifest.read_bytes()

    with pytest.raises(ValueError, match="differs from the recorded"):
        score_evaluation_condition(condition, updated_path, templates)
    with pytest.raises(ValueError, match="source protocol differs"):
        score_evaluation_condition(
            condition, updated_path, templates, rescore_from=updated_path
        )
    assert json.loads((condition / "meta.json").read_text()) == original_meta

    score_evaluation_condition(condition, updated_path, templates, rescore_from=original_path)
    actual = load_evaluation_condition(condition)
    assert actual.meta["protocol"]["fingerprint"] == updated.fingerprint
    assert actual.meta["scoring_protocol"] == updated.to_json_dict()
    assert actual.meta["collection_protocol"] == {
        **original_meta["protocol"], "definition": original.to_json_dict()
    }
    assert actual.cases[0].terminal_rigid_distance < old_scores.result.terminal_rigid_distance
    assert trajectory_manifest.read_bytes() == trajectory_bytes

    # Repeating the current score still validates immutable, old trajectory fingerprints.
    score_evaluation_condition(condition, updated_path, templates)
    repeated = load_evaluation_condition(condition)
    assert repeated == actual
    score_evaluation_condition(condition, updated_path, templates, rescore_from=original_path)
    assert load_evaluation_condition(condition) == actual


@pytest.mark.parametrize(
    "changes",
    [
        {"task_name": "other_task"},
        {"metric_hz": 10},
        {"horizon_seconds": 2.0},
        {"target": replace(_protocol().target, symmetry_center_xy=(0.0, 0.0))},
        {"protocol_id": "other_protocol"},
    ],
)
def test_rescore_rejects_non_scoring_changes(tmp_path: Path, changes) -> None:
    condition, original_path, templates, _ = _offline_evaluation(tmp_path)
    updated = replace(_protocol(), symmetry="planar_90", **changes)
    updated_path = tmp_path / "updated.yaml"
    updated_path.write_text(json.dumps(updated.to_json_dict()), encoding="utf-8")
    before = (condition / "meta.json").read_bytes()
    with pytest.raises(ValueError, match="may change only metric and success"):
        score_evaluation_condition(condition, updated_path, templates, rescore_from=original_path)
    assert (condition / "meta.json").read_bytes() == before


def test_rescore_formal_validation_uses_current_scoring_protocol(tmp_path, monkeypatch) -> None:
    condition, original_path, templates, _ = _offline_evaluation(tmp_path)
    updated = replace(_protocol(), symmetry="planar_90")
    updated_path = tmp_path / "updated.yaml"
    updated_path.write_text(json.dumps(updated.to_json_dict()), encoding="utf-8")
    checked = []
    monkeypatch.setattr(
        "clothdojo.evaluation.collection.validate_formal_evaluation_metadata",
        lambda meta: checked.append(meta["protocol"]["fingerprint"]),
    )
    score_evaluation_condition(
        condition, updated_path, templates, rescore_from=original_path, require_formal=True
    )
    assert checked == [updated.fingerprint]


def test_formal_panel_checks_pool_payload_against_frozen_metadata(tmp_path, monkeypatch):
    from clothdojo.evaluation import collection

    states = tmp_path / "states"
    templates = tmp_path / "templates"
    states.mkdir()
    templates.mkdir()
    assets = []
    for index in range(24):
        asset_id = f"shirt_{index}"
        path = states / asset_id / "states.npz"
        path.parent.mkdir()
        path.write_bytes(f"frozen pool {index}".encode())
        assets.append(
            {
                "asset_id": asset_id,
                "split": "seen" if index < 12 else "unseen",
                "state_count": 4,
                "sha256": file_sha256(path),
            }
        )
    meta = {
        "selection_provenance": {
            "panel_id": "test_panel",
            "selection_sha256": "test_selection",
        },
        "seen_assets": 12,
        "unseen_assets": 12,
        "assets": assets,
    }
    (states / "meta.json").write_text(json.dumps(meta))
    (templates / "meta.json").write_text("{}")
    frozen_meta_hash = file_sha256(states / "meta.json")
    monkeypatch.setattr(collection, "FORMAL_BENCHMARK_INITIAL_STATES_SHA256", frozen_meta_hash)
    monkeypatch.setattr(collection, "FORMAL_BENCHMARK_PANEL_ID", "test_panel")
    monkeypatch.setattr(collection, "FORMAL_BENCHMARK_SELECTION_SHA256", "test_selection")
    monkeypatch.setattr(collection, "FORMAL_BENCHMARK_SEEN_ASSETS", 12)
    monkeypatch.setattr(collection, "FORMAL_BENCHMARK_UNSEEN_ASSETS", 12)
    monkeypatch.setattr(
        collection,
        "FORMAL_BENCHMARK_ASSET_SPLITS_SHA256",
        collection.benchmark_asset_splits_sha256(
            {asset["asset_id"]: asset["split"] for asset in assets}
        ),
    )
    monkeypatch.setattr(
        collection, "_FORMAL_TEMPLATES_META_SHA256", file_sha256(templates / "meta.json")
    )
    checked = []

    def count_hash(path):
        checked.append(path)
        return file_sha256(path)

    monkeypatch.setattr(collection, "file_sha256", count_hash)
    assert collection._validate_formal_inputs(states, meta, templates) == {
        asset["asset_id"]: asset["sha256"] for asset in assets
    }
    assert all(checked.count(states / asset["asset_id"] / "states.npz") == 1 for asset in assets)
    (states / assets[0]["asset_id"] / "states.npz").write_bytes(b"replacement pool")
    assert file_sha256(states / "meta.json") == frozen_meta_hash
    with pytest.raises(ValueError, match="initial-state pool changed"):
        collection._validate_formal_inputs(states, meta, templates)


@pytest.mark.parametrize("field", ["protocol_fingerprint", "template_sha256"])
def test_rescore_still_validates_original_trajectory_provenance(tmp_path, field) -> None:
    condition, original_path, templates, _ = _offline_evaluation(tmp_path)
    updated = replace(_protocol(), symmetry="planar_90")
    updated_path = tmp_path / "updated.yaml"
    updated_path.write_text(json.dumps(updated.to_json_dict()), encoding="utf-8")
    manifest_path = condition / "trajectories/shirt_000003_r002/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"][field] = updated.fingerprint
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = (condition / "meta.json").read_bytes()
    with pytest.raises(ValueError, match="trajectory scoring provenance differs"):
        score_evaluation_condition(condition, updated_path, templates, rescore_from=original_path)
    assert (condition / "meta.json").read_bytes() == before


@pytest.mark.parametrize("field, value", [
    ("condition_id", "another-run"),
    ("policy", {"checkpoint_sha256": "0" * 64}),
    ("execution", {"mode": "chunk", "exec_horizon": 5}),
    ("cloth_visual", "white"),
    ("observation", {"renderer": "another-renderer"}),
    ("policy_seed", 9),
    ("initial_state", {"index": 3, "seed": 9, "pool_sha256": "e" * 64}),
    ("initial_state", {"index": 3, "seed": 7, "pool_sha256": "0" * 64}),
])
def test_offline_score_rejects_trajectory_from_another_collection(tmp_path, field, value):
    condition, protocol, templates, _ = _offline_evaluation(tmp_path)
    score_evaluation_condition(condition, protocol, templates)
    before_meta = (condition / "meta.json").read_bytes()
    before_scores = (condition / "results.jsonl").read_bytes()
    manifest_path = condition / "trajectories/shirt_000003_r002/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"][field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"trajectory .* differs"):
        score_evaluation_condition(condition, protocol, templates)

    assert (condition / "meta.json").read_bytes() == before_meta
    assert (condition / "results.jsonl").read_bytes() == before_scores


def test_score_publication_failure_never_exposes_mismatched_results(tmp_path, monkeypatch) -> None:
    condition, original_path, templates, _ = _offline_evaluation(tmp_path)
    score_evaluation_condition(condition, original_path, templates)
    updated = replace(_protocol(), symmetry="planar_90")
    updated_path = tmp_path / "updated.yaml"
    updated_path.write_text(json.dumps(updated.to_json_dict()), encoding="utf-8")
    from clothdojo.evaluation import collection

    real_write = collection.atomic_write_json

    def fail_completion(path, meta):
        if meta["status"] == "complete":
            raise OSError("simulated publication failure")
        real_write(path, meta)

    monkeypatch.setattr(collection, "atomic_write_json", fail_completion)
    with pytest.raises(OSError, match="simulated publication failure"):
        score_evaluation_condition(condition, updated_path, templates, rescore_from=original_path)
    with pytest.raises(RuntimeError, match="not complete"):
        load_evaluation_condition(condition)
    monkeypatch.setattr(collection, "atomic_write_json", real_write)
    score_evaluation_condition(condition, updated_path, templates, rescore_from=original_path)
    actual = load_evaluation_condition(condition).meta["protocol"]["fingerprint"]
    assert actual == updated.fingerprint


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

    with pytest.raises(ValueError, match="formal benchmark"):
        summarize_paired_conditions(
            [tmp_path / "base", tmp_path / "rtc"], require_formal=True
        )


def test_terminal_summary_distinguishes_lost_success_and_weights_assets(tmp_path: Path) -> None:
    # Every case reaches success at some point. Only persistence to the end
    # differs, and the assets have unequal case counts.
    reached = [_case("a", 0, success=True), _case("a", 1, success=True),
               _case("b", 0, success=True)]
    for name, terminal in (("base", (True, False, True)), ("other", (False, True, False))):
        _publish(tmp_path / name, name, [
            replace(case, success_at_end=at_end)
            for case, at_end in zip(reached, terminal, strict=True)
        ])
        meta_path = tmp_path / name / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["benchmark"] = {"asset_splits": {"a": "seen", "b": "unseen"}}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    summary = summarize_paired_conditions([tmp_path / "base", tmp_path / "other"])
    other = summary["conditions"][1]
    assert other["success_count"] == 3
    assert other["success_at_end_count"] == 1
    assert other["success_at_end_rate_micro"] == pytest.approx(1 / 3)
    assert other["success_at_end_rate_asset_macro"] == pytest.approx(0.25)
    assert other["success_at_end_rate_seen_asset_macro"] == pytest.approx(0.5)
    assert other["success_at_end_rate_unseen_asset_macro"] == 0.0
    assert other["success_at_end_rate_balanced"] == pytest.approx(0.25)
    assert other["asset_success_at_end_rates"] == {"a": 0.5, "b": 0.0}
    paired = summary["paired_comparisons"][0]
    assert paired["paired_success_rate_delta"] == 0
    assert paired["success_ties"] == 3
    assert paired["success_at_end_wins"] == 1
    assert paired["success_at_end_losses"] == 2
    assert paired["success_at_end_ties"] == 0
    assert paired["paired_success_at_end_rate_delta"] == pytest.approx(-0.5)


def test_formal_summary_requires_the_frozen_800_case_panel() -> None:
    summary = {
        "case_count": FORMAL_BENCHMARK_CASE_COUNT,
        "asset_count": FORMAL_BENCHMARK_SEEN_ASSETS + FORMAL_BENCHMARK_UNSEEN_ASSETS,
        "initial_states_sha256": FORMAL_BENCHMARK_INITIAL_STATES_SHA256,
        "protocol": {"fingerprint": FORMAL_BENCHMARK_PROTOCOL_FINGERPRINT},
        "benchmark": {
            "panel_id": FORMAL_BENCHMARK_PANEL_ID,
            "selection_sha256": FORMAL_BENCHMARK_SELECTION_SHA256,
            "asset_splits_sha256": FORMAL_BENCHMARK_ASSET_SPLITS_SHA256,
            "seen_assets": FORMAL_BENCHMARK_SEEN_ASSETS,
            "unseen_assets": FORMAL_BENCHMARK_UNSEEN_ASSETS,
            "states_per_asset": FORMAL_BENCHMARK_STATES_PER_ASSET,
        },
        "conditions": [
            {
                "condition_id": "formal",
                "case_count": FORMAL_BENCHMARK_CASE_COUNT,
                "success_at_end_rate_balanced": 0.5,
            }
        ],
    }

    validate_formal_evaluation_summary(summary)

    summary["case_count"] = 32
    with pytest.raises(ValueError, match="800 paired cases"):
        validate_formal_evaluation_summary(summary)


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
