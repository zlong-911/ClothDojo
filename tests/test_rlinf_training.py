from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from clothdojo.training import rlinf
from clothdojo.training.openpi_rlinf_sft import (
    _adapt_official_openpi_loader,
    _EpochAdvancingSampler,
    _initialize_ema_state,
    _swap_parameter_data_with_ema,
    _update_ema_state,
    _validate_fp32_training_state,
)
from clothdojo.training.steam import configure_steam_member


def test_run_config_delegates_to_rlinf_runner(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        rlinf,
        "inspect_openpi_rlinf_sft",
        lambda cfg, **_kwargs: events.append(("preflight", cfg))
        or {"compatible": True, "openpi_data_home": "/openpi-cache"},
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
    assert os.environ["OPENPI_DATA_HOME"] == "/openpi-cache"
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
        lambda value, **_kwargs: {"compatible": value is cfg},
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
        "CLOTHDOJO_OPENPI_CACHE": tmp_path / "openpi-cache",
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
    assert cfg.actor.seed == 43
    assert cfg.actor.optim.total_training_steps == 60000
    assert cfg.actor.optim.lr_scheduler == "openpi_cosine"
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.min_lr == 5.0e-6
    assert cfg.actor.optim.ema_decay == 0.99
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.actor.fsdp_config.save_full_model_weights is False
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.actor.model.model_type == "openpi_rlinf"
    assert cfg.actor.model.openpi.config_name == "pi05_aloha_robotwin"
    assert cfg.actor.model.openpi_data.adapt_to_pi is False
    assert cfg.actor.model.openpi_data.extra_delta_transform is True
    assert cfg.actor.model.openpi_data.norm_stats_path == str(tmp_path / "norm-stats")
    assert cfg.clothdojo.jax_reference.trial == "decay60k_lr2x_seed43"


def test_openpi_rlinf_contract_is_derived_from_the_named_jax_trial(
    monkeypatch, tmp_path
) -> None:
    checkpoint = tmp_path / "pytorch-base"
    for name, value in {
        "CLOTHDOJO_LEROBOT_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_PI05_CHECKPOINT": checkpoint,
        "CLOTHDOJO_PI05_NORM_STATS": tmp_path / "norm-stats",
        "CLOTHDOJO_TRAIN_OUTPUT": tmp_path / "outputs",
        "CLOTHDOJO_OPENPI_CACHE": tmp_path / "openpi-cache",
        "CLOTHDOJO_PI05_JAX_PARAMS": tmp_path / "jax-params",
        "CLOTHDOJO_PI05_PYTORCH_BASE": checkpoint,
        "CLOTHDOJO_JAX_CHECKPOINT_BASE": tmp_path / "jax-checkpoints",
        "CLOTHDOJO_JAX_GATE_DIR": tmp_path / "jax-gates",
        "CLOTHDOJO_JAX_MATRIX_DIR": tmp_path / "jax-matrix",
    }.items():
        monkeypatch.setenv(name, str(value))

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs/training/pi05_aloha_openpi_rlinf_lr2x_60k.yaml"
    cfg = rlinf.compose_config(config_path)
    contract = {
        "seed": 43,
        "global_batch_size": 64,
        "steps": 60_000,
        "peak_lr": 5.0e-5,
        "warmup_steps": 1_000,
        "decay_steps": 60_000,
        "floor_lr": 5.0e-6,
        "adam_b1": 0.9,
        "adam_b2": 0.95,
        "adam_eps": 1.0e-8,
        "weight_decay": 1.0e-10,
        "clip_gradient_norm": 1.0,
        "ema_decay": 0.99,
    }

    reference = rlinf._inspect_jax_reference(cfg, config_path, contract)

    assert reference["trial"] == "decay60k_lr2x_seed43"
    assert reference["training_contract"] == contract
    with pytest.raises(ValueError, match="differs from its JAX reference"):
        rlinf._inspect_jax_reference(cfg, config_path, {**contract, "seed": 42})


def test_openpi_rlinf_preflight_reuses_shared_artifact_contract(
    monkeypatch, tmp_path
) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "runner": {"max_steps": 60000, "save_interval": 10000},
            "data": {
                "train_data_paths": "/dataset",
                "openpi_data_home": str(tmp_path / "openpi-cache"),
            },
            "actor": {
                "seed": 43,
                "global_batch_size": 64,
                "micro_batch_size": 8,
                "optim": {
                    "lr": 5.0e-5,
                    "lr_warmup_steps": 1000,
                    "total_training_steps": 60000,
                    "min_lr": 5.0e-6,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.95,
                    "adam_eps": 1.0e-8,
                    "weight_decay": 1.0e-10,
                    "clip_grad": 1.0,
                    "ema_decay": 0.99,
                    "lr_scheduler": "openpi_cosine",
                    "num_cycles": 0.5,
                },
                "model": {
                    "model_type": "openpi_rlinf",
                    "precision": "fp32",
                    "num_action_chunks": 50,
                    "openpi": {"config_name": "pi05_aloha_robotwin"},
                    "openpi_data": {
                        "adapt_to_pi": False,
                        "extra_delta_transform": True,
                        "default_prompt": "Flatten the shirt.",
                        "norm_stats_path": "/norm",
                    },
                },
                "fsdp_config": {
                    "save_full_model_weights": False,
                    "mixed_precision": {
                        "param_dtype": "bf16",
                        "reduce_dtype": "fp32",
                        "buffer_dtype": "fp32",
                    }
                },
            },
        }
    )
    tokenizer = tmp_path / "openpi-cache/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer")
    calls = []
    monkeypatch.setattr(
        "clothdojo.training.openpi_artifacts.inspect_openpi_dataset",
        lambda path: calls.append(("dataset", path)) or {"root": path},
    )
    monkeypatch.setattr(
        "clothdojo.training.openpi_artifacts.inspect_openpi_norm_stats",
        lambda path, **kwargs: calls.append(("norm", path, kwargs)) or {"root": path},
    )

    result = rlinf.inspect_openpi_rlinf_sft(cfg)

    assert result["compatible"] is True
    assert calls[0] == ("dataset", "/dataset")
    assert calls[1][0:2] == ("norm", "/norm")
    assert calls[1][2]["data_kwargs"] == {
        "adapt_to_pi": False,
        "extra_delta_transform": True,
        "default_prompt": "Flatten the shirt.",
    }


def test_openpi_loader_adapter_uses_actor_seed_and_advances_epoch() -> None:
    dataset = TensorDataset(torch.arange(24))
    sampler = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    torch_loader = DataLoader(dataset, batch_size=3, sampler=sampler)
    data_loader = SimpleNamespace(
        _data_loader=SimpleNamespace(_data_loader=torch_loader)
    )

    _adapt_official_openpi_loader(data_loader, seed=42, rank=1)

    adapted = torch_loader.batch_sampler.sampler
    assert isinstance(adapted, _EpochAdvancingSampler)
    assert sampler.seed == 42
    epoch_zero = [int(value) for batch in torch_loader for value in batch[0]]
    epoch_one = [int(value) for batch in torch_loader for value in batch[0]]
    reference_zero = DistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=42, drop_last=True
    )
    reference_one = DistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=42, drop_last=True
    )
    reference_one.set_epoch(1)
    assert epoch_zero == list(reference_zero)
    assert epoch_one == list(reference_one)
    assert epoch_zero != epoch_one


def test_openpi_rlinf_precision_gate_requires_fp32_masters() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()

    _validate_fp32_training_state(model, optimizer)

    with pytest.raises(RuntimeError, match="FP32 masters"):
        _validate_fp32_training_state(model.to(torch.bfloat16), optimizer)


def test_openpi_rlinf_ema_matches_jax_update_and_swaps_reversibly() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 3.0]]))
    ema = _initialize_ema_state(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[5.0, 7.0]]))

    _update_ema_state(model, ema, 0.75)

    torch.testing.assert_close(ema["weight"], torch.tensor([[2.0, 4.0]]))
    _swap_parameter_data_with_ema(model, ema, chunk_elements=1)
    torch.testing.assert_close(model.weight, torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(ema["weight"], torch.tensor([[5.0, 7.0]]))
    _swap_parameter_data_with_ema(model, ema, chunk_elements=1)
    torch.testing.assert_close(model.weight, torch.tensor([[5.0, 7.0]]))
    torch.testing.assert_close(ema["weight"], torch.tensor([[2.0, 4.0]]))
