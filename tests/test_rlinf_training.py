from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from clothdojo.training import rlinf


def test_run_config_delegates_to_rlinf_runner(monkeypatch) -> None:
    events = []

    class Multiprocessing:
        @staticmethod
        def set_start_method(method, *, force):
            events.append(("start_method", method, force))

    class OmegaConf:
        @staticmethod
        def to_container(cfg, *, resolve):
            assert resolve is True
            return {"resolved": True}

    class Cluster:
        def __init__(self, *, cluster_cfg):
            events.append(("cluster", cluster_cfg))

    class Placement:
        def __init__(self, cfg, cluster):
            events.append(("placement", cfg, cluster))

        def get_strategy(self, component):
            events.append(("strategy", component))
            return "actor-placement"

    class Worker:
        @classmethod
        def create_group(cls, cfg):
            events.append(("create_group", cfg))
            return cls()

        def launch(self, cluster, *, name, placement_strategy):
            events.append(("launch", cluster, name, placement_strategy))
            return "actor-group"

    class Runner:
        def __init__(self, *, cfg, actor):
            events.append(("runner", cfg, actor))

        def init_workers(self):
            events.append(("init_workers",))

        def run(self):
            events.append(("run",))

    runtime = rlinf._Runtime(
        validate_cfg=lambda cfg: cfg,
        cluster_type=Cluster,
        placement_type=Placement,
        runner_type=Runner,
        worker_type=Worker,
        multiprocessing=Multiprocessing,
        omega_conf=OmegaConf,
    )
    monkeypatch.setattr(rlinf, "_load_runtime", lambda kind: runtime)
    cfg = SimpleNamespace(
        cluster="cluster-config",
        actor=SimpleNamespace(group_name="ActorGroup"),
        get=lambda key, default: default,
    )

    rlinf.run_config(kind="pi05", cfg=cfg)

    assert events[0] == ("start_method", "spawn", True)
    assert ("strategy", "actor") in events
    assert ("runner", cfg, "actor-group") in events
    assert events[-2:] == [("init_workers",), ("run",)]


def test_pi05_aloha_config_preserves_training_contract(monkeypatch, tmp_path) -> None:
    for name, value in {
        "CLOTHDOJO_LEROBOT_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_PI05_CHECKPOINT": tmp_path / "checkpoint",
        "CLOTHDOJO_PI05_NORM_STATS": tmp_path / "norm-stats",
        "CLOTHDOJO_TRAIN_OUTPUT": tmp_path / "outputs",
    }.items():
        monkeypatch.setenv(name, str(value))

    project_root = Path(__file__).resolve().parents[1]
    cfg = rlinf.compose_config(project_root / "configs/training/pi05_aloha.yaml")

    assert cfg.runner.max_steps == 40000
    assert cfg.runner.max_epochs == -1
    assert cfg.runner.save_interval == 10000
    assert cfg.cluster.num_nodes == 1
    assert cfg.cluster.component_placement.actor == "0-7"
    assert cfg.data.train_data_paths == str(tmp_path / "dataset")
    assert cfg.actor.global_batch_size == 64
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.seed == 42
    assert cfg.actor.optim.total_training_steps == 40000
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.actor.model.openpi.config_name == "pi05_aloha_robotwin"
    assert cfg.actor.model.openpi_data.adapt_to_pi is False
    assert cfg.actor.model.openpi_data.extra_delta_transform is True
    assert cfg.actor.model.openpi_data.norm_stats_path == str(tmp_path / "norm-stats")
