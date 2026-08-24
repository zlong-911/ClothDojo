from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from clothdojo.training import rlinf
from clothdojo.training.rlinf_pi05 import (
    ClothDojoPi05SftWorker,
    _adapt_official_openpi_loader,
    _EpochAdvancingSampler,
    _seed_rank,
)


def test_run_config_delegates_to_rlinf_runner(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        "clothdojo.training.pi05_preflight.inspect_pi05_training",
        lambda cfg: events.append(("preflight", cfg)) or {"compatible": True},
    )
    monkeypatch.setattr(
        "clothdojo.training.pi05_preflight.write_pi05_launch_manifest",
        lambda cfg, manifest: events.append(("manifest", cfg, manifest))
        or Path("/output/pi05_launch_manifest.json"),
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

    rlinf.run_config(kind="pi05", cfg=cfg)

    assert events[0] == ("preflight", cfg)
    assert events[1][0] == "manifest"
    assert events[2] == ("start_method", "spawn", True)
    assert ("strategy", "actor") in events
    assert ("runner", cfg, "actor-group") in events
    assert events[-2:] == [("init_workers",), ("run",)]


def test_pi05_runtime_uses_project_worker_adapter() -> None:
    assert rlinf._load_runtime("pi05").worker_type is ClothDojoPi05SftWorker


def test_pi05_check_does_not_import_rlinf_runtime(monkeypatch) -> None:
    cfg = object()
    monkeypatch.setattr(rlinf, "compose_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        rlinf,
        "_load_runtime",
        lambda kind: pytest.fail("pi05 --check must not import rlinf.config or initialize Ray"),
    )
    monkeypatch.setattr(
        "clothdojo.training.pi05_preflight.inspect_pi05_training",
        lambda value: {"compatible": value is cfg},
    )

    assert rlinf.check(kind="pi05", config_path="unused.yaml") == {"compatible": True}


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
    assert cfg.actor.optim.total_training_steps == 30000
    assert cfg.actor.optim.lr_scheduler == "openpi_cosine"
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.min_lr == 2.5e-6
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.actor.model.openpi.config_name == "pi05_aloha_robotwin"
    assert cfg.actor.model.openpi_data.adapt_to_pi is False
    assert cfg.actor.model.openpi_data.extra_delta_transform is True
    assert cfg.actor.model.openpi_data.norm_stats_path == str(tmp_path / "norm-stats")


def test_seed_rank_seeds_python_numpy_torch_and_cuda(monkeypatch) -> None:
    cuda_seeds = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: cuda_seeds.append(("one", seed)))
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: cuda_seeds.append(("all", seed)),
    )

    assert _seed_rank(42, 3) == 45
    python_value = random.random()
    numpy_value = np.random.random()
    torch_value = torch.rand(1)

    _seed_rank(42, 3)
    assert random.random() == python_value
    assert np.random.random() == numpy_value
    assert torch.equal(torch.rand(1), torch_value)
    assert cuda_seeds.count(("one", 45)) == 2
    assert cuda_seeds.count(("all", 45)) >= 2


def test_openpi_loader_adapter_uses_uniform_seed_and_advances_epoch() -> None:
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

    class OpenPiTorchLoader:
        def __init__(self) -> None:
            self.torch_loader = torch_loader

    class OpenPiDataLoader:
        def __init__(self) -> None:
            self._data_loader = OpenPiTorchLoader()

    _adapt_official_openpi_loader(OpenPiDataLoader(), seed=42, rank=1)

    adapted = torch_loader.batch_sampler.sampler
    assert isinstance(adapted, _EpochAdvancingSampler)
    assert sampler.seed == 42

    epoch_zero = [int(value) for batch in torch_loader for value in batch[0]]
    epoch_one = [int(value) for batch in torch_loader for value in batch[0]]

    reference_zero = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=42,
        drop_last=True,
    )
    reference_one = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=42,
        drop_last=True,
    )
    reference_one.set_epoch(1)
    assert epoch_zero == list(reference_zero)
    assert epoch_one == list(reference_one)
    assert epoch_zero != epoch_one
