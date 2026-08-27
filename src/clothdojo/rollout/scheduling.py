"""Shared asset selection and local policy-worker scheduling."""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any


def select_assets(
    assets: list[dict[str, Any]],
    *,
    seen_assets: int | None,
    unseen_assets: int | None,
) -> list[dict[str, Any]]:
    if (seen_assets is None) != (unseen_assets is None):
        raise ValueError("seen_assets and unseen_assets must be provided together")
    if seen_assets is None:
        return list(assets)

    counts = {"seen": seen_assets, "unseen": unseen_assets}
    for split, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{split}_assets must be a non-negative integer")
    if seen_assets + unseen_assets < 1:
        raise ValueError("at least one rollout asset must be selected")

    selected = []
    for split, count in counts.items():
        candidates = [item for item in assets if item.get("split") == split]
        if count > len(candidates):
            raise ValueError(
                f"requested {count} {split} assets, but the state pool contains {len(candidates)}"
            )
        selected.extend(candidates[:count])
    return selected


def policy_endpoints_for_workers(
    policy_endpoints: tuple[str, ...], worker_count: int
) -> tuple[str, ...]:
    if worker_count < 1:
        raise ValueError("rollout requires at least one active worker")
    if len(policy_endpoints) < worker_count:
        raise ValueError(
            "reproducible multi-GPU rollout requires one unique policy endpoint "
            f"per active worker: workers={worker_count}, endpoints={len(policy_endpoints)}"
        )
    active = policy_endpoints[:worker_count]
    if len(set(active)) != len(active):
        raise ValueError("each active rollout worker requires a unique policy endpoint")
    return active


def validate_policy_identities(
    identities: tuple[dict[str, Any], ...],
    *,
    execution: str,
    worlds_per_batch: int = 1,
) -> dict[str, Any]:
    if not identities:
        raise ValueError("rollout requires at least one policy identity")
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise ValueError("policy endpoints do not serve the same checkpoint and configuration")
    if worlds_per_batch > 1 and first.get("batched_inference") is not True:
        raise ValueError("parallel rollout requires a batched-inference policy server")
    if execution == "rtc":
        if not isinstance(first.get("rtc_guidance"), dict):
            raise ValueError("RTC rollout requires an RTC-capable policy server")
    return first


def run_policy_asset_workers(
    jobs: list[dict[str, Any]],
    *,
    gpus: tuple[str, ...],
    policy_endpoints: tuple[str, ...],
    worker_target,
    common_args: tuple[Any, ...],
    process_kind: str,
) -> None:
    """Run one asset at a time per GPU with one exclusive policy endpoint."""

    if not jobs:
        return
    worker_count = min(len(gpus), len(jobs))
    if len(policy_endpoints) != worker_count:
        raise ValueError("policy endpoint count must match active worker count")
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in range(worker_count):
        job_queue.put(None)
    workers = [
        context.Process(
            target=worker_target,
            args=(
                job_queue,
                result_queue,
                gpu,
                policy_endpoints[index],
                *common_args,
            ),
            name=f"clothdojo-{process_kind}-gpu-{gpu}",
        )
        for index, gpu in enumerate(gpus[:worker_count])
    ]
    for worker in workers:
        worker.start()
    completed = 0
    try:
        while completed < len(jobs):
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(
                        f"{process_kind} worker exited with code {failed[0].exitcode}"
                    ) from None
                continue
            if result["status"] == "error":
                raise RuntimeError(
                    f"{process_kind} failed for {result['asset_id']} on GPU "
                    f"{result['gpu']}: {result['error']}"
                )
            completed += 1
    finally:
        for worker in workers:
            if worker.is_alive() and completed < len(jobs):
                worker.terminate()
            worker.join()
