from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import clothdojo.evaluation.collection as evaluation_collection
from clothdojo.evaluation.collection import run_evaluation_condition
from clothdojo.evaluation.metric import (
    CanonicalTemplate,
    TorchFactorizedClothMetric,
    factorized_cloth_distance,
    load_canonical_template,
    save_canonical_template,
)
from clothdojo.evaluation.protocol import (
    EvaluationProtocol,
    StableSuccessTracker,
    load_evaluation_protocol,
)
from clothdojo.evaluation.results import (
    EVALUATION_RESULT_FORMAT_VERSION,
    EVALUATION_RESULT_SCHEMA,
    EvaluationCaseKey,
    EvaluationCaseResult,
    load_evaluation_condition,
    summarize_paired_conditions,
    write_evaluation_cases,
)
from clothdojo.sim.scene import ShirtFlattenSpec

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
        "stable_seconds": 0.4,
        "deformable_weight": 0.65,
        "symmetry": "planar_180",
        "deformable_distance_max": 0.1,
        "rigid_distance_max": None,
        "particle_speed_rms_max_m_s": 0.02,
        "require_released": True,
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
    wrong_pivot = factorized_cloth_distance(
        centroid_rotated[None],
        zero[None],
        template,
        symmetry="planar_180",
        symmetry_center_xy=TABLE_CENTER_XY,
    )
    assert wrong_pivot.rigid_distance[0] > 0.1


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


def test_torch_metric_matches_numpy_reference() -> None:
    torch = pytest.importorskip("torch")
    template = _template()
    generator = np.random.default_rng(7)
    q = np.repeat(template.particle_q[None], 3, axis=0)
    q += generator.normal(0.0, 0.03, q.shape).astype(np.float32)
    qd = generator.normal(0.0, 0.01, q.shape).astype(np.float32)
    expected = factorized_cloth_distance(
        q,
        qd,
        template,
        symmetry_center_xy=TABLE_CENTER_XY,
    )

    metric = TorchFactorizedClothMetric(
        template,
        device="cpu",
        symmetry_center_xy=TABLE_CENTER_XY,
    )
    actual = metric(torch.from_numpy(q), torch.from_numpy(qd))

    for field in (
        "deformable_distance",
        "rigid_distance",
        "combined_distance",
        "speed_rms_m_s",
    ):
        np.testing.assert_allclose(
            getattr(actual, field).numpy(), getattr(expected, field), atol=1.0e-6
        )
    np.testing.assert_array_equal(actual.symmetry_index.numpy(), expected.symmetry_index)


def test_success_is_absorbing_only_after_release_speed_and_dwell() -> None:
    protocol = _protocol()
    tracker = StableSuccessTracker(protocol, batch_size=2)
    good = np.asarray([0.05, 0.05])
    rigid = np.asarray([0.2, 0.2])
    speed = np.asarray([0.0, 0.0])

    tracker.update(
        step=0,
        deformable_distance=good,
        rigid_distance=rigid,
        speed_rms_m_s=speed,
        released=np.asarray([False, True]),
    )
    new = tracker.update(
        step=2,
        deformable_distance=good,
        rigid_distance=rigid,
        speed_rms_m_s=speed,
        released=np.asarray([True, True]),
    )
    np.testing.assert_array_equal(new, [False, True])
    assert tracker.confirmed_step[1] == 2
    assert tracker.stable_since_step[1] == 0

    new = tracker.update(
        step=4,
        deformable_distance=good,
        rigid_distance=rigid,
        speed_rms_m_s=speed,
        released=np.asarray([True, False]),
    )
    np.testing.assert_array_equal(new, [True, False])
    new = tracker.update(
        step=6,
        deformable_distance=good,
        rigid_distance=rigid,
        speed_rms_m_s=speed,
        released=np.asarray([True, False]),
    )
    np.testing.assert_array_equal(new, [False, False])
    assert tracker.success.tolist() == [True, True]

    tracker.update(
        step=8,
        deformable_distance=np.asarray([1.0, 1.0]),
        rigid_distance=rigid,
        speed_rms_m_s=np.asarray([1.0, 1.0]),
        released=np.asarray([False, False]),
    )
    assert tracker.success.tolist() == [True, True]


def test_protocol_and_template_round_trip(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    protocol = load_evaluation_protocol(project / "configs" / "evaluation" / "shirt_flatten.yaml")
    assert protocol.stage == "calibration"
    assert protocol.horizon_steps == 1800
    assert protocol.metric_stride == 5
    assert protocol.stable_samples == 10
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


def test_evaluation_repairs_complete_results_and_invalid_asset_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    scenario = SimpleNamespace(
        task_name="shirt_flatten",
        steps=protocol.horizon_steps,
        action_horizon=10,
        to_json_dict=lambda: {"task": "shirt_flatten", "steps": 10},
    )
    states = tmp_path / "states"
    templates = tmp_path / "templates"
    (states / "shirt").mkdir(parents=True)
    templates.mkdir()
    (states / "meta.json").write_text("{}", encoding="utf-8")
    (states / "shirt" / "states.npz").write_bytes(b"states")
    (templates / "shirt.npz").write_bytes(b"template")

    monkeypatch.setattr(evaluation_collection, "load_rollout_scenario", lambda _: scenario)
    monkeypatch.setattr(evaluation_collection, "load_evaluation_protocol", lambda _: protocol)
    monkeypatch.setattr(
        evaluation_collection,
        "load_initial_state_collection",
        lambda _: {"assets": [{"asset_id": "shirt"}]},
    )
    monkeypatch.setattr(
        evaluation_collection.InitialStatePool,
        "load",
        lambda _: SimpleNamespace(state_count=1),
    )
    monkeypatch.setattr(
        evaluation_collection,
        "load_canonical_template",
        lambda _: SimpleNamespace(asset_id="shirt"),
    )
    policy_identity = {
        "checkpoint_sha256": "f" * 64,
        "batched_inference": True,
    }
    monkeypatch.setattr(
        evaluation_collection,
        "RemoteSFTPolicy",
        lambda *args, **kwargs: SimpleNamespace(identity=policy_identity),
    )

    worker_calls = []

    def run_workers(jobs, **kwargs) -> None:
        worker_calls.append([job["asset_id"] for job in jobs])
        output = Path(kwargs["common_args"][4])
        for job in jobs:
            shape = (1, job["state_count"])
            arrays = {
                "initial_state_seed": np.full(shape, 7, dtype=np.int32),
                "policy_seed": np.full(shape, 8, dtype=np.int64),
                "success": np.zeros(shape, dtype=bool),
                "success_at_end": np.zeros(shape, dtype=bool),
                "confirmed_step": np.full(shape, -1, dtype=np.int32),
                "stable_since_step": np.full(shape, -1, dtype=np.int32),
                "terminal_deformable_distance": np.full(shape, 0.2, dtype=np.float32),
                "terminal_rigid_distance": np.full(shape, 0.1, dtype=np.float32),
                "terminal_combined_distance": np.full(shape, 0.15, dtype=np.float32),
                "terminal_speed_rms_m_s": np.full(shape, 0.01, dtype=np.float32),
                "best_settled_step": np.full(shape, -1, dtype=np.int32),
                "best_settled_deformable_distance": np.full(shape, np.nan, dtype=np.float32),
                "best_settled_rigid_distance": np.full(shape, np.nan, dtype=np.float32),
                "best_settled_combined_distance": np.full(shape, np.nan, dtype=np.float32),
                "curve": np.zeros((*shape, protocol.sample_count, 4), dtype=np.float32),
            }
            evaluation_collection._write_asset_result(
                evaluation_collection._asset_result_path(output, job["asset_id"]),
                asset_id=job["asset_id"],
                pool_sha256=job["pool_sha256"],
                template_sha256=job["template_sha256"],
                protocol_fingerprint=protocol.fingerprint,
                checkpoint_sha256=policy_identity["checkpoint_sha256"],
                execution="chunk",
                exec_horizon=10,
                arrays=arrays,
            )

    monkeypatch.setattr(evaluation_collection, "run_policy_asset_workers", run_workers)
    output = tmp_path / "evaluation"
    arguments = {
        "scenario_path": tmp_path / "scenario.yaml",
        "protocol_path": tmp_path / "protocol.yaml",
        "initial_states_path": states,
        "templates_path": templates,
        "policy_endpoints": ("http://policy",),
        "output": output,
        "condition_id": "condition",
        "gpus": ("cpu",),
    }

    run_evaluation_condition(**arguments)
    assert worker_calls == [["shirt"]]

    (output / "results.jsonl").unlink()
    run_evaluation_condition(**arguments)
    assert worker_calls == [["shirt"]]
    assert len(load_evaluation_condition(output).cases) == 1

    (output / "assets" / "shirt.npz").write_bytes(b"broken")
    run_evaluation_condition(**arguments)
    assert worker_calls == [["shirt"], ["shirt"]]
    assert len(load_evaluation_condition(output).cases) == 1

    asset_result = output / "assets" / "shirt.npz"
    with np.load(asset_result, allow_pickle=False) as stored:
        payload = {name: stored[name] for name in stored.files}
    payload["curve"] = payload["curve"].copy()
    payload["curve"][0, 0, 0, 0] = np.nan
    np.savez_compressed(asset_result, **payload)
    run_evaluation_condition(**arguments)
    assert worker_calls == [["shirt"], ["shirt"], ["shirt"]]

    with np.load(asset_result, allow_pickle=False) as stored:
        payload = {name: stored[name] for name in stored.files}
    payload["curve"] = payload["curve"].copy()
    payload["curve"][0, 0, 0, 0] = -1.0
    np.savez_compressed(asset_result, **payload)
    run_evaluation_condition(**arguments)
    assert worker_calls == [["shirt"], ["shirt"], ["shirt"], ["shirt"]]


def _case(asset: str, index: int, *, success: bool) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        key=EvaluationCaseKey(asset, index),
        initial_state_seed=100 + index,
        policy_seed=200 + index,
        success=success,
        success_at_end=success,
        confirmed_step=4 if success else None,
        stable_since_step=2 if success else None,
        terminal_deformable_distance=0.05 if success else 0.3,
        terminal_rigid_distance=0.1,
        terminal_combined_distance=0.07 if success else 0.23,
        terminal_speed_rms_m_s=0.01 if success else 0.2,
        best_settled_step=4,
        best_settled_deformable_distance=0.05 if success else 0.2,
        best_settled_rigid_distance=0.1,
        best_settled_combined_distance=0.07 if success else 0.17,
    )


def _publish(
    path: Path,
    condition: str,
    cases,
    *,
    scenario: dict | None = None,
    initial_state_pools: dict | None = None,
) -> None:
    path.mkdir()
    ordered = tuple(cases)
    assets = {case.key.asset_id for case in ordered}
    write_evaluation_cases(path / "results.jsonl", ordered)
    meta = {
        "schema": EVALUATION_RESULT_SCHEMA,
        "format_version": EVALUATION_RESULT_FORMAT_VERSION,
        "status": "complete",
        "condition_id": condition,
        "protocol": {
            "id": "test",
            "stage": "frozen",
            "fingerprint": "d" * 64,
            "horizon_steps": 10,
            "control_fps": 10,
        },
        "scenario": scenario or {"fingerprint": "2" * 64},
        "initial_states_sha256": "e" * 64,
        "initial_state_pools": initial_state_pools
        or dict.fromkeys(assets, "0" * 64),
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
    assert summary["paired_comparisons"][0]["success_wins"] == 2
    assert summary["paired_comparisons"][0]["success_losses"] == 0


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
