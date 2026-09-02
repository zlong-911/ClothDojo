from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clothdojo.evaluation import managed
from clothdojo.evaluation.run_config import load_benchmark_run_config
from clothdojo.rollout.trajectory_collection import (
    TrajectoryAssetJob,
    TrajectoryCollectionSpec,
)


def _write_configs(root: Path, *, remote: bool = False, mismatched: bool = False) -> Path:
    (root / "configs/evaluation").mkdir(parents=True)
    (root / "configs/rollout").mkdir(parents=True)
    (root / "data/checkpoint").mkdir(parents=True)
    (root / "data/norm").mkdir(parents=True)
    benchmark = root / "configs/evaluation/panel.yaml"
    benchmark.write_text(
        """schema: clothdojo_frozen_benchmark
panel_id: balanced200_20260901
scenario: configs/rollout/scenario.yaml
protocol: configs/evaluation/protocol.yaml
states: data/states
templates: data/templates
assets: assets
""",
        encoding="utf-8",
    )
    host = "worker" if remote else "local"
    policy_remote_fields = (
        "\n      connect_host: 10.0.0.2\n      project_root: /remote/project"
        if remote
        else ""
    )
    simulation_remote_fields = (
        "\n      project_root: /remote/project" if remote else ""
    )
    deployment = root / "configs/evaluation/deployment.yaml"
    deployment.write_text(
        f"""schema: clothdojo_evaluation_deployment
policy:
  nodes:
    - host: {host}{policy_remote_fields}
      gpus: [0, 1]
      port_start: 18080
simulation:
  nodes:
    - host: {host}{simulation_remote_fields}
      gpus: {[2] if mismatched else [2, 3]}
score_workers: 3
startup_timeout_seconds: 30
""",
        encoding="utf-8",
    )
    run = root / "configs/evaluation/run.yaml"
    run.write_text(
        """schema: clothdojo_benchmark_run
condition_id: formal
benchmark: configs/evaluation/panel.yaml
deployment: configs/evaluation/deployment.yaml
policy:
  kind: cfg
  checkpoint: data/checkpoint
  norm_stats: data/norm
  num_steps: 10
  guidance_scale: 2.5
  guidance_type: positive
  positive_only_conditional: true
execution:
  mode: chunk
  exec_horizon: 10
  image_preprocessing: none
""",
        encoding="utf-8",
    )
    return run


def test_managed_run_config_owns_frozen_inputs_and_deployment(tmp_path: Path) -> None:
    run = _write_configs(tmp_path)

    config = load_benchmark_run_config(run, project_root=tmp_path)

    assert config.condition_id == "formal"
    assert config.output == tmp_path / "outputs/evaluations/formal"
    assert config.benchmark.states == tmp_path / "data/states"
    assert config.policy.kind == "cfg"
    assert config.policy.options["guidance_scale"] == 2.5
    assert config.execution.exec_horizon == 10
    assert config.deployment.worker_count == 2
    assert config.deployment.score_workers == 3


def test_remote_simulation_node_does_not_require_unused_connect_host(
    tmp_path: Path,
) -> None:
    run = _write_configs(tmp_path, remote=True)

    config = load_benchmark_run_config(run, project_root=tmp_path)

    assert config.deployment.simulation_nodes[0].connect_host is None


def test_deployment_requires_one_policy_per_simulation_worker(tmp_path: Path) -> None:
    run = _write_configs(tmp_path, mismatched=True)

    with pytest.raises(ValueError, match="one policy service per simulation worker"):
        load_benchmark_run_config(run, project_root=tmp_path)


def test_policy_and_execution_preprocessing_must_match(tmp_path: Path) -> None:
    run = _write_configs(tmp_path)
    text = run.read_text(encoding="utf-8").replace(
        "  num_steps: 10", "  num_steps: 10\n  image_preprocessing: training_fixed_crop"
    )
    run.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="image_preprocessing must match"):
        load_benchmark_run_config(run, project_root=tmp_path)


def test_run_config_rejects_non_string_paths_and_non_finite_numbers(
    tmp_path: Path,
) -> None:
    run = _write_configs(tmp_path)
    text = run.read_text(encoding="utf-8").replace(
        "benchmark: configs/evaluation/panel.yaml",
        "benchmark: 42",
    )
    run.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark config must be a non-empty string"):
        load_benchmark_run_config(run, project_root=tmp_path)

    run = _write_configs(tmp_path / "finite")
    text = run.read_text(encoding="utf-8").replace(
        "  guidance_scale: 2.5",
        "  guidance_scale: .nan",
    )
    run.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="guidance_scale must be a non-negative number"):
        load_benchmark_run_config(run, project_root=tmp_path / "finite")


def test_remote_paths_and_policy_command_are_mapped_by_project_root(tmp_path: Path) -> None:
    run = _write_configs(tmp_path, remote=True)
    config = load_benchmark_run_config(run, project_root=tmp_path)
    slot = managed._policy_slots(config)[0]

    command = managed._policy_command(config, slot)

    assert command[:5] == [
        "/remote/project/.venv/bin/python",
        "-m",
        "clothdojo",
        "policy",
        "serve-cfg",
    ]
    assert "/remote/project/data/checkpoint" in command
    assert "/remote/project/data/norm" in command
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "18080"]


def test_node_plans_partition_assets_without_shared_metadata_writers(
    tmp_path: Path, monkeypatch
) -> None:
    run = _write_configs(tmp_path)
    config = load_benchmark_run_config(run, project_root=tmp_path)
    config.output.mkdir(parents=True)
    logs = config.output / "runtime/logs"
    logs.mkdir(parents=True)
    jobs = [
        TrajectoryAssetJob(
            asset_id=f"asset-{index}",
            pool=str(tmp_path / f"data/states/asset-{index}/states.npz"),
            pool_sha256=str(index) * 64,
            state_count=4,
            replicates=(0,),
        )
        for index in range(6)
    ]
    spec = TrajectoryCollectionSpec(
        scenario_path=str(tmp_path / "configs/rollout/scenario.yaml"),
        assets_root=str(tmp_path / "assets"),
        trajectories_root=str(config.output / "trajectories"),
        expected_checkpoint_sha256=None,
        policy_identity={"checkpoint_sha256": "a" * 64},
        execution="chunk",
        exec_horizon=10,
        rtc_delay_steps=0,
        horizon_steps=4500,
        batch_size=4,
        producer="clothdojo_evaluation",
        training_eligible=False,
    )
    launched = []

    def launch(node, command, **kwargs):
        launched.append((node, command, kwargs))
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(managed, "_launch", launch)

    children = managed._launch_simulation_nodes(
        config,
        jobs,
        spec,
        ("ws://p0:1", "ws://p1:1"),
        logs,
        runtime_sha256="c" * 64,
    )

    assert len(children) == len(launched) == 1
    plan = json.loads(
        (config.output / "runtime/nodes/node-000.json").read_text(encoding="utf-8")
    )
    assert plan["schema"] == "clothdojo_evaluation_node_plan"
    assert plan["runtime_sha256"] == "c" * 64
    assert {item["asset_id"] for item in plan["jobs"]} == {
        f"asset-{index}" for index in range(6)
    }
    assert len(plan["jobs"]) == len({item["asset_id"] for item in plan["jobs"]})
    assert "meta" not in plan
    assert plan["policy_endpoints"] == ["ws://p0:1", "ws://p1:1"]


def test_worker_reconstructs_json_replicates(monkeypatch, tmp_path: Path) -> None:
    plan = tmp_path / "node.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "clothdojo_evaluation_node_plan",
                "runtime_sha256": "c" * 64,
                "gpus": ["3"],
                "policy_endpoints": ["ws://policy:8080"],
                "jobs": [
                    {
                        "asset_id": "shirt",
                        "pool": "states.npz",
                        "pool_sha256": "a" * 64,
                        "state_count": 4,
                        "replicates": [0],
                        "metadata": {},
                    }
                ],
                "spec": {
                    "scenario_path": "scenario.yaml",
                    "assets_root": "assets",
                    "trajectories_root": "trajectories",
                    "expected_checkpoint_sha256": None,
                    "policy_identity": {"checkpoint_sha256": "b" * 64},
                    "execution": "chunk",
                    "exec_horizon": 10,
                    "rtc_delay_steps": 0,
                    "horizon_steps": 4500,
                    "batch_size": 4,
                    "producer": "clothdojo_evaluation",
                    "training_eligible": False,
                    "compressed": False,
                    "cloth_visual": "nocs",
                },
            }
        ),
        encoding="utf-8",
    )
    call = {}
    monkeypatch.setattr(
        "clothdojo.evaluation.worker.collect_trajectories",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs),
    )
    monkeypatch.setattr(
        "clothdojo.evaluation.worker.runtime_source_sha256", lambda: "c" * 64
    )
    monkeypatch.setattr("sys.argv", ["worker", "--plan", str(plan)])

    from clothdojo.evaluation.worker import main

    main()

    assert call["args"][0][0].replicates == (0,)
    assert call["kwargs"]["gpus"] == ("3",)


def test_managed_entry_owns_policy_collection_and_scoring(
    monkeypatch, tmp_path: Path
) -> None:
    run = _write_configs(tmp_path)
    config = load_benchmark_run_config(run, project_root=tmp_path)
    launches = []

    class Process:
        pid = 123
        returncode = 0

        def __init__(self, running):
            self.running = running

        def poll(self):
            return None if self.running else 0

    class Log:
        def close(self):
            pass

    def launch(node, command, **kwargs):
        child = managed._Child(
            label=kwargs["label"],
            process=Process(running=kwargs["label"].startswith("policy")),
            log=Log(),
        )
        launches.append((node, command, kwargs, child))
        return child

    monkeypatch.setattr(managed, "load_benchmark_run_config", lambda _path: config)
    monkeypatch.setattr(managed, "_launch", launch)
    monkeypatch.setattr(managed, "_wait_for_endpoints", lambda *args, **kwargs: None)
    monkeypatch.setattr(managed, "_stop_process", lambda process: None)
    monkeypatch.setattr(managed, "runtime_source_sha256", lambda _path: "c" * 64)

    def run_condition(*args, **kwargs):
        jobs = [
            TrajectoryAssetJob(
                asset_id="shirt",
                pool=str(tmp_path / "data/states/shirt/states.npz"),
                pool_sha256="a" * 64,
                state_count=4,
                replicates=(0,),
            )
        ]
        spec = TrajectoryCollectionSpec(
            scenario_path=str(tmp_path / "configs/rollout/scenario.yaml"),
            assets_root=str(tmp_path / "assets"),
            trajectories_root=str(config.output / "trajectories"),
            expected_checkpoint_sha256=None,
            policy_identity={"checkpoint_sha256": "b" * 64},
            execution="chunk",
            exec_horizon=10,
            rtc_delay_steps=0,
            horizon_steps=4500,
            batch_size=4,
            producer="clothdojo_evaluation",
            training_eligible=False,
        )
        kwargs["trajectory_collector"](
            jobs,
            spec,
            ("ws://127.0.0.1:18080", "ws://127.0.0.1:18081"),
        )
        return {"compatible": True, "condition_id": kwargs["condition_id"]}

    monkeypatch.setattr(
        "clothdojo.evaluation.collection.run_evaluation_condition", run_condition
    )

    result = managed.run_managed_evaluation(run)

    assert result == {"compatible": True, "condition_id": "formal"}
    assert [entry[2]["label"] for entry in launches] == [
        "policy service 0",
        "policy service 1",
        "simulation node 0",
    ]
    assert (config.output / "runtime/managed-run.json").is_file()


def test_only_one_coordinator_can_own_an_output(tmp_path: Path) -> None:
    path = tmp_path / "coordinator.lock"
    first = managed._acquire_coordinator_lock(path)
    try:
        with pytest.raises(RuntimeError, match="another benchmark coordinator"):
            managed._acquire_coordinator_lock(path)
    finally:
        first.close()

    managed._acquire_coordinator_lock(path).close()
