"""Importable DataLoader hooks for ClothDojo's STEAM forkserver."""

from __future__ import annotations

import numpy as np
import torch

_PARQUET_GUARD_INSTALLED = False


def install_parquet_archive_guard() -> None:
    """Keep explicit local Parquet out of HF's unnecessary TAR probing.

    With the pinned datasets/Python runtime, concurrent TAR sniffing raised
    EFAULT on otherwise readable Parquet on the shared filesystem. Check both
    Parquet magic markers; real data validation still belongs to PyArrow.
    Other file formats retain the original datasets behavior.
    """
    global _PARQUET_GUARD_INSTALLED
    if _PARQUET_GUARD_INSTALLED:
        return
    from pathlib import Path

    from datasets.utils.extract import Extractor

    original = Extractor.infer_extractor_format

    def infer_format(cls, path):
        if Path(path).suffix == ".parquet":
            with open(path, "rb") as stream:
                header = stream.read(4)
                stream.seek(-4, 2)
                footer = stream.read(4)
            if header != b"PAR1" or footer != b"PAR1":
                raise ValueError(f"invalid local Parquet magic: {path}")
            return None
        return original(path)

    Extractor.infer_extractor_format = classmethod(infer_format)
    _PARQUET_GUARD_INSTALLED = True


def seed_pair_dataset_worker(worker_id: int, *, rank_seed: int) -> None:
    """Seed one persistent PairDataset decoder worker deterministically."""

    from clothdojo.training.rlinf_steam_state import enable_state_aware_steam

    enable_state_aware_steam()
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    worker_seed = int(rank_seed) + int(worker_id)
    dataset = worker_info.dataset
    if hasattr(dataset, "_rng"):
        dataset._rng = np.random.default_rng(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)
