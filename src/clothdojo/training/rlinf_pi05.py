"""Project-local launch fixes for RLINF's official OpenPI SFT path.

The upstream OpenPI loader intentionally presents an infinite iterator.  Its
inner ``torch.utils.data.DataLoader`` is recreated at dataset boundaries, but
the distributed sampler is otherwise left at epoch zero.  This module adapts
that narrow boundary without changing the installed RLINF or OpenPI packages.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from torch.utils.data.distributed import DistributedSampler


class _EpochAdvancingSampler:
    """Advance one distributed-sampler epoch for every new inner iterator."""

    def __init__(self, sampler: DistributedSampler[Any]) -> None:
        self._sampler = sampler
        self._next_epoch = 0

    def __iter__(self) -> Iterator[int]:
        self._sampler.set_epoch(self._next_epoch)
        self._next_epoch += 1
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._sampler)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch that will be used by the next inner iterator."""

        self._next_epoch = int(epoch)


def _seed_rank(seed: int, rank: int) -> int:
    """Seed process-global RNGs with the historical OpenPI rank convention."""

    rank_seed = int(seed) + int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)
    return rank_seed


def _adapt_official_openpi_loader(data_loader: Any, *, seed: int, rank: int) -> None:
    """Install deterministic rank seeding and epoch advancement on one loader.

    OpenPI exposes the inner torch loader through ``TorchDataLoader.torch_loader``
    but does not expose that wrapper from ``DataLoaderImpl``.  The single private
    lookup below is therefore guarded by explicit layout checks.
    """

    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "torch_loader", None)
    if torch_loader is None:
        raise TypeError(
            "OpenPI SFT loader layout changed: expected "
            "DataLoaderImpl._data_loader.torch_loader"
        )

    batch_sampler = getattr(torch_loader, "batch_sampler", None)
    sampler = getattr(batch_sampler, "sampler", None)
    if not isinstance(sampler, DistributedSampler):
        raise TypeError(
            "OpenPI SFT requires a torch DistributedSampler after process-group "
            f"initialization, got {type(sampler).__name__}"
        )

    # Every rank must construct the same global permutation; DistributedSampler
    # itself selects the disjoint rank-specific slice.
    sampler.seed = int(seed)
    batch_sampler.sampler = _EpochAdvancingSampler(sampler)

    # OpenPI uses this generator for worker initialization. Keep those streams
    # deterministic and rank-local as well.
    generator = getattr(torch_loader, "generator", None)
    if generator is not None:
        generator.manual_seed(int(seed) + int(rank))


def _worker_rank(worker: Any) -> int:
    value = getattr(worker, "_rank", os.environ.get("RANK", 0))
    rank = int(value)
    if rank < 0:
        raise ValueError(f"RLINF worker rank must be non-negative, got {rank}")
    return rank


def _actor_seed(cfg: Any) -> int:
    return int(cfg.actor.seed)


class ClothDojoPi05SftWorker(FSDPVlaSftWorker):
    """RLINF Pi0.5 worker with deterministic OpenPI data iteration."""

    def __init__(self, cfg: Any) -> None:
        rank = _worker_rank(self)
        _seed_rank(_actor_seed(cfg), rank)
        super().__init__(cfg)

    def build_dataloader(
        self,
        data_paths: Any,
        eval_dataset: bool = False,
    ) -> tuple[Any, Any]:
        data_loader, data_config = super().build_dataloader(
            data_paths,
            eval_dataset=eval_dataset,
        )
        if not eval_dataset:
            _adapt_official_openpi_loader(
                data_loader,
                seed=_actor_seed(self.cfg),
                rank=_worker_rank(self),
            )
        return data_loader, data_config
