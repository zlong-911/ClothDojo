"""Run a bounded matrix workload on one GPU until asked to stop."""

from __future__ import annotations

import argparse
import signal
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-device", required=True)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--duty-cycle", type=float, default=1.0)
    args = parser.parse_args()
    if args.matrix_size < 1:
        raise ValueError("matrix-size must be positive")
    if not 0.0 < args.duty_cycle <= 1.0:
        raise ValueError("duty-cycle must be in (0, 1]")

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    device = torch.device("cuda:0")
    left = torch.randn(
        (args.matrix_size, args.matrix_size), device=device, dtype=torch.bfloat16
    )
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    torch.cuda.synchronize()
    print(f"GPU {args.host_device} keepalive ready", flush=True)

    while running:
        started = time.monotonic()
        torch.mm(left, right, out=output)
        torch.cuda.synchronize()
        active_seconds = time.monotonic() - started
        if args.duty_cycle < 1.0:
            time.sleep(active_seconds * (1.0 / args.duty_cycle - 1.0))


if __name__ == "__main__":
    main()
