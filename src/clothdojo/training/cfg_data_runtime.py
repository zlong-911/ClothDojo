"""CFG's epoch-tagged sampling and consumed-batch checkpoint state.

Sampling stays in RLinf. Epoch tags travel with each queued index, so persistent
workers never depend on a mutable parent-process epoch or a shared epoch race.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

import torch
from torch.utils.data import DistributedSampler


class EpochTaggedMixture:
    def __init__(self, mixture: Any):
        self.mixture = mixture

    def __len__(self) -> int:
        return len(self.mixture)

    def __getitem__(self, key: tuple[int, int]) -> Any:
        epoch, index = key
        self.mixture.set_epoch(epoch)
        return self.mixture[index]


class CfgEpochSampler(DistributedSampler):
    """Shard/shuffle normally, but resume by skipping indices, not decoding data."""

    def __init__(self, dataset: Any, *, batch_size: int, **kwargs: Any):
        super().__init__(dataset, **kwargs)
        self.batch_size = batch_size
        self.start_batch = 0
        if batch_size <= 0 or self.batches_per_epoch == 0:
            raise ValueError("CFG dataset must provide at least one full batch per rank")

    @property
    def batches_per_epoch(self) -> int:
        return len(self) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.start_batch = 0

    def restore(self, epoch: int, consumed_batches: int) -> None:
        if epoch < 0 or not 0 <= consumed_batches <= self.batches_per_epoch:
            raise ValueError("Invalid CFG epoch/batch checkpoint cursor")
        self.set_epoch(epoch)
        self.start_batch = consumed_batches

    def __iter__(self):
        # Capture before yielding: already-prefetched indices keep their epoch.
        epoch = self.epoch
        indices = itertools.islice(super().__iter__(), self.start_batch * self.batch_size, None)
        return ((epoch, index) for index in indices)


def make_cfg_torch_loader(
    mixture: Any,
    *,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    dataset = EpochTaggedMixture(mixture)
    sampler = CfgEpochSampler(
        dataset,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=True,
        drop_last=True,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers else None,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers else None,
        # Iterator construction must not consume the training/flow-noise RNG.
        generator=torch.Generator().manual_seed(seed + rank),
    )


def resume_fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_runtime_state(
    state: dict[str, Any], *, identity: dict[str, Any], batches_per_epoch: int
) -> None:
    if state.get("identity") != identity:
        raise ValueError("CFG resume data/configuration or rank topology does not match")
    for name in ("step", "epoch", "consumed_batches"):
        if type(state.get(name)) is not int or state[name] < 0:
            raise ValueError(f"Invalid CFG runtime checkpoint {name}")
    consumed = state["epoch"] * batches_per_epoch + state["consumed_batches"]
    if (
        state["consumed_batches"] > batches_per_epoch
        or consumed != state["step"] * identity["gradient_accumulation"]
    ):
        raise ValueError("CFG checkpoint batch cursor disagrees with optimizer step")
    if not {"cpu", "numpy", "random"}.issubset(state.get("rng", {})):
        raise ValueError("CFG checkpoint is missing per-rank RNG state")
