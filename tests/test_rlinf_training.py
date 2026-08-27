from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from clothdojo.training import rlinf
from clothdojo.training.steam import configure_steam_member


def test_run_config_delegates_to_rlinf_runner(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        rlinf,
        "inspect_openpi_rlinf_sft",
        lambda cfg: events.append(("preflight", cfg)) or {"compatible": True},
    )

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

    rlinf.run_config(kind="openpi-rlinf", cfg=cfg)

    assert events[0] == ("preflight", cfg)
    assert events[1] == ("start_method", "spawn", True)
    assert ("strategy", "actor") in events
    assert ("runner", cfg, "actor-group") in events
    assert events[-2:] == [("init_workers",), ("run",)]


def test_openpi_rlinf_runtime_uses_current_worker() -> None:
    from clothdojo.training.openpi_rlinf_sft import OpenPiRlinfSftWorker

    assert rlinf._load_runtime("openpi-rlinf").worker_type is OpenPiRlinfSftWorker


def test_openpi_rlinf_check_does_not_import_scheduler_runtime(monkeypatch) -> None:
    cfg = object()
    monkeypatch.setattr(rlinf, "compose_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        rlinf,
        "_load_runtime",
        lambda kind: pytest.fail("openpi-rlinf --check must not initialize Ray"),
    )
    monkeypatch.setattr(
        rlinf,
        "inspect_openpi_rlinf_sft",
        lambda value: {"compatible": value is cfg},
    )

    assert rlinf.check(kind="openpi-rlinf", config_path="unused.yaml") == {
        "compatible": True
    }


def test_steam_check_uses_project_preflight_without_rlinf_runtime(monkeypatch) -> None:
    cfg = SimpleNamespace(actor=SimpleNamespace(group_name="SteamMember"))
    monkeypatch.setattr(rlinf, "compose_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        rlinf,
        "_load_runtime",
        lambda kind: pytest.fail("steam --check must not initialize Ray"),
    )
    monkeypatch.setattr(
        "clothdojo.training.steam.inspect_steam_training",
        lambda value, member_index: {
            "compatible": value is cfg,
            "member_index": member_index,
        },
    )

    assert rlinf.check(
        kind="steam",
        config_path="unused.yaml",
        steam_member_index=2,
    ) == {"compatible": True, "member_index": 2}


def test_configure_steam_member_owns_names_and_seeds(monkeypatch) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"actor": {"group_name": "SteamMember"}})
    result = configure_steam_member(cfg, 1)

    assert result == {
        "member_index": 1,
        "head_seed": 101,
        "run_seed": 20360826,
    }
    assert cfg.actor.group_name == "SteamMember1"
    assert os.environ["CLOTHDOJO_STEAM_MEMBER_INDEX"] == "1"
    assert os.environ["CLOTHDOJO_STEAM_HEAD_SEED"] == "101"
    assert os.environ["CLOTHDOJO_STEAM_RUN_SEED"] == "20360826"


def test_steam_shirt_flatten_config_is_self_contained(monkeypatch, tmp_path) -> None:
    for name, value in {
        "CLOTHDOJO_STEAM_PLACEMENT": "0-7",
        "CLOTHDOJO_STEAM_OUTPUT": tmp_path / "output",
        "CLOTHDOJO_STEAM_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_STEAM_SIGLIP_CHECKPOINT": tmp_path / "siglip",
        "CLOTHDOJO_STEAM_GEMMA3_CHECKPOINT": tmp_path / "gemma",
    }.items():
        monkeypatch.setenv(name, str(value))

    project_root = Path(__file__).resolve().parents[1]
    cfg = rlinf.compose_config(
        project_root / "configs/training/steam_value_shirt_flatten.yaml"
    )

    assert cfg.cluster.num_nodes == 4
    assert cfg.cluster.component_placement["actor,env,rollout"] == "0-7"
    assert cfg.runner.max_steps == 30000
    assert cfg.data.train_data_paths[0].dataset_path == str(tmp_path / "dataset")
    assert cfg.data.train_num_workers == 8
    assert cfg.actor.micro_batch_size == 32
    assert cfg.actor.global_batch_size == 512
    assert cfg.actor.model.model_type == "steam_value_model"
    assert cfg.actor.model.precision == "fp32"
    assert cfg.actor.model.include_state is True
    assert cfg.actor.model.freeze_vision_encoder is False
    assert cfg.actor.model.freeze_language_model is False
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype == "bf16"


def test_openpi_rlinf_config_preserves_training_contract(monkeypatch, tmp_path) -> None:
    for name, value in {
        "CLOTHDOJO_LEROBOT_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_PI05_CHECKPOINT": tmp_path / "checkpoint",
        "CLOTHDOJO_PI05_NORM_STATS": tmp_path / "norm-stats",
        "CLOTHDOJO_TRAIN_OUTPUT": tmp_path / "outputs",
    }.items():
        monkeypatch.setenv(name, str(value))

    project_root = Path(__file__).resolve().parents[1]
    cfg = rlinf.compose_config(
        project_root / "configs/training/pi05_aloha_openpi_rlinf_lr2x_60k.yaml"
    )

    assert cfg.runner.max_steps == 60000
    assert cfg.runner.max_epochs == -1
    assert cfg.runner.save_interval == 10000
    assert cfg.cluster.num_nodes == 1
    assert cfg.cluster.component_placement.actor == "0-7"
    assert cfg.data.train_data_paths == str(tmp_path / "dataset")
    assert cfg.actor.global_batch_size == 64
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.seed == 42
    assert cfg.actor.optim.total_training_steps == 60000
    assert cfg.actor.optim.lr_scheduler == "openpi_cosine"
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.min_lr == 5.0e-6
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.actor.model.model_type == "openpi_rlinf"
    assert cfg.actor.model.openpi.config_name == "pi05_aloha_robotwin"
    assert cfg.actor.model.openpi_data.adapt_to_pi is False
    assert cfg.actor.model.openpi_data.extra_delta_transform is True
    assert cfg.actor.model.openpi_data.norm_stats_path == str(tmp_path / "norm-stats")
