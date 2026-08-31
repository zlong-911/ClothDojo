from __future__ import annotations

import copy
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from rlinf.data.datasets.recap.common import BaseDataLoaderImpl, ReCapMixtureDataset

from clothdojo.training.cfg_data_runtime import (
    make_cfg_torch_loader,
    validate_runtime_state,
)


class Source:
    def __init__(self, count=23):
        self.count = count
        self.reads = []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.reads.append(index)
        return index


def make_loader(*, workers=0, rank=0, world=1, count=23):
    mixture = ReCapMixtureDataset([(Source(count), 1.0)], seed=42)
    loader = make_cfg_torch_loader(
        mixture, batch_size=2, rank=rank, world_size=world, seed=42, num_workers=workers
    )
    loader.pin_memory = False  # These regressions must remain CPU-only on GPU hosts.
    return loader


def stop_loader(loader):
    if loader._iterator is not None:
        loader._iterator._shutdown_workers()


def test_spawn_persistent_workers_observe_new_epoch_without_restart():
    loader = make_loader(workers=2, count=97)
    try:
        first = torch.cat(list(loader))
        pids = [worker.pid for worker in loader._iterator._workers]
        loader.sampler.set_epoch(1)
        second = torch.cat(list(loader))
        expected = torch.tensor([loader.dataset[key] for key in loader.sampler])[:96]
        assert torch.equal(second, expected)
        assert not torch.equal(first.sort().values, second.sort().values)
        assert pids == [worker.pid for worker in loader._iterator._workers]
    finally:
        stop_loader(loader)


def test_epoch_tag_survives_prefetch_and_parent_epoch_change():
    loader = make_loader()
    iterator = iter(loader.sampler)
    first = next(iterator)
    loader.sampler.set_epoch(5)
    assert first[0] == 0
    assert all(epoch == 0 for epoch, _ in iterator)
    assert all(epoch == 5 for epoch, _ in loader.sampler)
    mixture = loader.dataset.mixture
    old_value = loader.dataset[first]
    loader.dataset[(5, first[1])]
    assert loader.dataset[first] == old_value
    assert mixture._epoch == 0


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_resume_skips_indices_without_decoding_prior_batches(rank):
    loader = make_loader(rank=rank, world=3, count=97)
    loader.sampler.set_epoch(2)
    all_batches = list(loader)
    source = loader.dataset.mixture.datasets[0]
    source.reads.clear()
    loader.sampler.restore(2, 7)
    resumed = list(loader)
    assert all(torch.equal(a, b) for a, b in zip(resumed, all_batches[7:], strict=True))
    assert len(source.reads) == len(resumed) * 2
    assert len(loader) == 16  # Stable epoch length, not remaining-length arithmetic.


def test_distributed_sampling_shards_indices_before_mixture_sampling():
    shards = [list(make_loader(rank=rank, world=3, count=97).sampler) for rank in range(3)]
    assert all(len(shard) == 32 for shard in shards)
    assert len(set().union(*map(set, shards))) == 96


def test_loader_iteration_never_consumes_training_torch_rng():
    loader = make_loader()
    state = torch.get_rng_state().clone()
    list(loader)
    loader.sampler.restore(0, 5)
    list(loader)
    loader.sampler.set_epoch(1)
    list(loader)
    assert torch.equal(torch.get_rng_state(), state)


class ToyLoader(BaseDataLoaderImpl):
    def __iter__(self):
        for batch in self._data_loader:
            values = batch.float().unsqueeze(1) / 23
            yield values, values * 2, torch.ones(len(batch), dtype=torch.bool)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.samples = []

    def forward(self, data):
        observation = data["observation"]
        self.samples.append(observation.clone())
        noise = torch.randn_like(observation) + random.random() + np.random.random()
        loss = (self.linear(observation + noise * 0.01) - data["actions"]).square().mean()
        return loss, None


def cpu_rng():
    return {
        "cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }


def restore_cpu_rng(state):
    torch.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])


def make_worker(monkeypatch, *, rank=0, world=1, workers=0):
    import rlinf.utils.utils as utils
    import rlinf.workers.sft.fsdp_cfg_worker as upstream

    from clothdojo.training.rlinf_cfg import ClothDojoCfgWorker

    monkeypatch.setattr(utils, "get_rng_state", cpu_rng)
    monkeypatch.setattr(utils, "set_rng_state", restore_cpu_rng)
    monkeypatch.setattr(upstream, "all_reduce_dict", lambda metrics, op: metrics)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def save_weights(worker, path, step):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": worker.model.state_dict(),
                "optimizer": worker.optimizer.state_dict(),
                "scheduler": worker.lr_scheduler.state_dict(),
            },
            Path(path) / "toy_weights.pt",
        )

    def load_weights(worker, path):
        saved = torch.load(Path(path) / "toy_weights.pt", weights_only=False)
        worker.model.load_state_dict(saved["model"])
        worker.optimizer.load_state_dict(saved["optimizer"])
        worker.lr_scheduler.load_state_dict(saved["scheduler"])
        # Model initialization/upstream DCP may change/replace the RNG state.
        torch.manual_seed(987)
        random.seed(987)
        np.random.seed(987)

    monkeypatch.setattr(upstream.FSDPCfgWorker, "save_checkpoint", save_weights)
    monkeypatch.setattr(upstream.FSDPCfgWorker, "load_checkpoint", load_weights)
    worker = object.__new__(ClothDojoCfgWorker)
    worker._rank, worker._world_size = rank, world
    worker.cfg = OmegaConf.create(
        {"actor": {"global_batch_size": 4 * world, "micro_batch_size": 2}}
    )
    worker.gradient_accumulation = 2
    worker.model = ToyModel()
    worker.optimizer = torch.optim.AdamW(worker.model.parameters(), lr=0.01)
    worker.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(worker.optimizer, gamma=0.99)
    worker.data_loader = ToyLoader(
        None, make_loader(rank=rank, world=world, workers=workers, count=23 * world)
    )
    worker.data_iter = iter(worker.data_loader)
    worker._data_epoch, worker._data_iter_offset = 0, 0
    worker._cfg_runtime_identity = {
        "rank": rank,
        "world_size": world,
        "gradient_accumulation": 2,
        "fingerprint": "test",
    }
    worker.device = torch.device("cpu")
    worker.amp_context = nullcontext()
    worker.worker_timer = nullcontext
    worker.before_micro_batch = lambda *args, **kwargs: nullcontext()
    worker.grad_scaler = SimpleNamespace(scale=lambda value: value)

    def optimizer_step():
        worker.optimizer.step()
        return 0.0, [worker.optimizer.param_groups[0]["lr"]]

    worker.optimizer_step = optimizer_step
    return worker


def train(worker, start, stop):
    for step in range(start, stop):
        worker.set_global_step(step)
        worker.run_training()


@pytest.mark.parametrize("rank,world", [(0, 1), (0, 2), (1, 2)])
def test_actual_worker_resume_matches_continuous_adamw_training(monkeypatch, tmp_path, rank, world):
    torch.manual_seed(42 + rank)
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    continuous = make_worker(monkeypatch, rank=rank, world=world)
    train(continuous, 0, 3)
    continuous.save_checkpoint(str(tmp_path), 3)
    train(continuous, 3, 10)
    expected = copy.deepcopy(continuous.model.state_dict())
    expected_optimizer = copy.deepcopy(continuous.optimizer.state_dict())
    expected_samples = continuous.model.samples[6:]
    resumed = make_worker(monkeypatch, rank=rank, world=world)
    resumed.load_checkpoint(str(tmp_path))
    source = resumed.data_loader._data_loader.dataset.mixture.datasets[0]
    assert not source.reads
    train(resumed, 3, 10)
    for key, value in expected.items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    assert all(
        torch.equal(a, b) for a, b in zip(expected_samples, resumed.model.samples, strict=True)
    )
    for parameter, state in expected_optimizer["state"].items():
        for key, value in state.items():
            assert torch.equal(value, resumed.optimizer.state_dict()["state"][parameter][key])
    # 20 micro-batches cross the 11-batch epoch inside one optimizer step.
    assert (resumed._data_epoch, resumed._data_iter_offset) == (1, 9)


def test_resume_with_spawn_workers_matches_future_batches(monkeypatch, tmp_path):
    worker = make_worker(monkeypatch, workers=2)
    restored = None
    try:
        train(worker, 0, 3)
        worker.save_checkpoint(str(tmp_path), 3)
        train(worker, 3, 8)
        expected = copy.deepcopy(worker.model.state_dict())
        restored = make_worker(monkeypatch, workers=2)
        restored.load_checkpoint(str(tmp_path))
        train(restored, 3, 8)
        assert all(
            torch.equal(value, restored.model.state_dict()[key]) for key, value in expected.items()
        )
    finally:
        stop_loader(worker.data_loader._data_loader)
        if restored is not None:
            stop_loader(restored.data_loader._data_loader)


def test_old_checkpoint_is_rejected_for_exact_resume(monkeypatch, tmp_path):
    worker = make_worker(monkeypatch)
    with pytest.raises(FileNotFoundError, match="Older checkpoints remain usable"):
        worker.load_checkpoint(str(tmp_path))


def test_resume_rejects_changed_data_topology_cursor_and_runner_step(monkeypatch, tmp_path):
    worker = make_worker(monkeypatch)
    train(worker, 0, 2)
    worker.save_checkpoint(str(tmp_path), 2)
    state = torch.load(tmp_path / "cfg_runtime_rank_0.pt", weights_only=False)
    for field in ("rank", "world_size", "fingerprint"):
        identity = {**worker._cfg_runtime_identity, field: "changed"}
        with pytest.raises(ValueError, match="does not match"):
            validate_runtime_state(state, identity=identity, batches_per_epoch=11)
    with pytest.raises(ValueError, match="disagrees with optimizer step"):
        validate_runtime_state(
            {**state, "consumed_batches": 0},
            identity=worker._cfg_runtime_identity,
            batches_per_epoch=11,
        )
    worker.load_checkpoint(str(tmp_path))
    with pytest.raises(ValueError, match="runner step disagrees"):
        worker.set_global_step(3)


def test_empty_local_batch_is_rejected():
    with pytest.raises(ValueError, match="at least one full batch"):
        make_loader(world=8, count=8)


def test_actual_worker_applies_actor_seed_plus_rank_before_data_loading(monkeypatch):
    from clothdojo.training.rlinf_cfg import ClothDojoCfgWorker

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def draw(rank):
        worker = object.__new__(ClothDojoCfgWorker)
        worker._rank = rank
        worker.cfg = OmegaConf.create({"actor": {"seed": 42}, "data": {}})
        with pytest.raises(ValueError, match="must contain a CFG dataset"):
            worker.build_dataloader()
        return torch.rand(4), np.random.random(4), random.random()

    first, second, different_rank = draw(0), draw(0), draw(1)
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert not torch.equal(first[0], different_rank[0])
    assert not np.array_equal(first[1], different_rank[1])
    assert first[2] != different_rank[2]
