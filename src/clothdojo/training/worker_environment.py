"""Explicit transport settings applied at the RLINF worker boundary."""

from __future__ import annotations

import os
from typing import Any

_KEYS = frozenset(
    {
        "GLOO_SOCKET_IFNAME",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_HCA",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    }
)


def worker_environment_from_config(cfg: Any) -> dict[str, str | None]:
    from omegaconf import OmegaConf

    project = cfg.get("clothdojo")
    configured = None if project is None else project.get("worker_environment")
    if configured is None:
        return {}
    resolved = OmegaConf.to_container(configured, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("clothdojo.worker_environment must be a mapping")
    unknown = set(resolved) - _KEYS
    if unknown:
        raise ValueError(f"unsupported distributed environment keys: {sorted(unknown)}")
    return {str(name): None if value is None else str(value) for name, value in resolved.items()}


def apply_worker_environment(cfg: Any) -> None:
    """Override container defaults inside the actor, before FSDP starts."""
    for name, value in worker_environment_from_config(cfg).items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
