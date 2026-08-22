"""Minimal NCCL collective smoke for a provisioned training node."""

from __future__ import annotations

import json
import os


def main() -> None:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    value = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"NCCL all-reduce returned {value.item()}, expected {expected}")
    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "backend": dist.get_backend(),
                    "compatible": True,
                    "world_size": world_size,
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
