"""Importable DataLoader hooks for ClothDojo's STEAM forkserver."""

from __future__ import annotations

import numpy as np
import torch


def seed_pair_dataset_worker(worker_id: int, *, rank_seed: int) -> None:
    """Seed one persistent PairDataset decoder worker deterministically."""

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    worker_seed = int(rank_seed) + int(worker_id)
    dataset = worker_info.dataset
    if hasattr(dataset, "_rng"):
        dataset._rng = np.random.default_rng(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)
