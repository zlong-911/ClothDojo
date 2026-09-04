"""Background multi-GPU review export for every configured trajectory source."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clothdojo.collection.config import CollectionConfig

ACTIVE_EXPORT_STATES = {"starting", "running"}
ProcessLauncher = Callable[[list[str], Path], subprocess.Popen]
GpuDetector = Callable[[], list[int]]


@dataclass(frozen=True, slots=True)
class _ReviewWork:
    source: Path
    trajectory_id: str
    action_count: int


class ReviewExportBusyError(RuntimeError):
    """Raised when a server-wide review export is already active."""


class ReviewExportManager:
    """Export all configured collections and rollouts on non-teleop GPUs."""

    def __init__(
        self,
        config: CollectionConfig,
        *,
        gpu_detector: GpuDetector | None = None,
        launcher: ProcessLauncher | None = None,
    ) -> None:
        self.config = config
        self._gpu_detector = gpu_detector or detect_nvidia_gpu_ids
        self._launcher = launcher or self._launch
        self._job: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self.root = config.state_dir / "review-exports"
        self.root.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None:
                return {
                    "status": "idle",
                    "excluded_gpu_ids": list(self.config.teleoperation.reserved_gpu_ids),
                }
            return _public_job(self._job)

    def start(self, *, fps: int = 10) -> dict[str, Any]:
        work = _all_review_work(self.config)
        if not work:
            raise ValueError("no completed trajectories are configured for review export")
        excluded = set(self.config.teleoperation.reserved_gpu_ids)
        gpu_ids = [gpu_id for gpu_id in self._gpu_detector() if gpu_id not in excluded]
        if not gpu_ids:
            raise RuntimeError("no non-teleop NVIDIA GPU is available for review export")
        shards = _balanced_shards(work, gpu_ids)
        job_id = f"review_{uuid.uuid4().hex[:12]}"
        job_root = self.root / job_id
        workers = [
            {
                "gpu_id": gpu_id,
                "work": items,
                "trajectory_count": len(items),
                "completed_trajectory_count": 0,
                "source_count": len({item.source for item in items}),
                "status": "pending",
                "pid": None,
                "exit_code": None,
                "error": None,
                "log_path": str(job_root / f"gpu-{gpu_id}.log"),
            }
            for gpu_id, items in shards
        ]
        job = {
            "id": job_id,
            "status": "starting",
            "fps": fps,
            "source_count": len({item.source for item in work}),
            "trajectory_count": len(work),
            "completed_trajectory_count": 0,
            "gpu_ids": [worker["gpu_id"] for worker in workers],
            "excluded_gpu_ids": list(self.config.teleoperation.reserved_gpu_ids),
            "workers": workers,
            "error": None,
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
        }
        with self._lock:
            if self._job is not None and self._job["status"] in ACTIVE_EXPORT_STATES:
                raise ReviewExportBusyError("review export is already running")
            job_root.mkdir(parents=True, exist_ok=False)
            self._job = job
        threading.Thread(
            target=self._run,
            args=(job,),
            name=f"review-{job_id}",
            daemon=True,
        ).start()
        return _public_job(job)

    def _run(self, job: dict[str, Any]) -> None:
        with self._lock:
            job["status"] = "running"
            job["started_at"] = _utc_now()
        threads = [
            threading.Thread(
                target=self._run_worker,
                args=(job, worker),
                name=f"review-{job['id']}-gpu-{worker['gpu_id']}",
                daemon=True,
            )
            for worker in job["workers"]
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        with self._lock:
            failures = [
                worker for worker in job["workers"] if worker["status"] == "failed"
            ]
            job["status"] = "failed" if failures else "complete"
            if failures:
                job["error"] = "; ".join(
                    f"GPU {worker['gpu_id']}: {worker['error']}" for worker in failures
                )
            job["finished_at"] = _utc_now()

    def _run_worker(self, job: dict[str, Any], worker: dict[str, Any]) -> None:
        failures = []
        with self._lock:
            worker["status"] = "running"
        for source, items in _work_by_source(worker["work"]):
            command = [
                sys.executable,
                "-m",
                "clothdojo",
                "data",
                "render-review",
                "--source",
                str(source),
                "--assets",
                str(self.config.assets_root),
                "--device",
                f"cuda:{worker['gpu_id']}",
                "--fps",
                str(job["fps"]),
            ]
            for item in items:
                command.extend(("--trajectory", item.trajectory_id))
            try:
                process = self._launcher(command, Path(worker["log_path"]))
                with self._lock:
                    worker["pid"] = process.pid
                while (exit_code := process.poll()) is None:
                    time.sleep(0.25)
            except Exception as exc:
                failures.append(f"{source}: {type(exc).__name__}: {exc}")
                continue
            with self._lock:
                worker["exit_code"] = exit_code
                if exit_code == 0:
                    count = len(items)
                    worker["completed_trajectory_count"] += count
                    job["completed_trajectory_count"] += count
                else:
                    failures.append(f"{source}: process exited with code {exit_code}")
        with self._lock:
            worker["pid"] = None
            worker["status"] = "failed" if failures else "complete"
            worker["error"] = "; ".join(failures) if failures else None

    def _launch(self, command: list[str], log_path: Path) -> subprocess.Popen:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log:
            return subprocess.Popen(
                command,
                cwd=self.config.project_root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )


def detect_nvidia_gpu_ids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise RuntimeError("nvidia-smi returned duplicate GPU indices")
    return gpu_ids


def _all_review_work(config: CollectionConfig) -> list[_ReviewWork]:
    sources = []
    for profile in config.profiles.values():
        sources.extend((profile.dataset_root, profile.hil_dataset_root))
        if (profile.rollout_root / "meta.json").is_file():
            sources.append(profile.rollout_root)
        if profile.rollout_root.is_dir():
            sources.extend(path.parent for path in profile.rollout_root.glob("*/meta.json"))
    result = []
    seen = set()
    for source in sources:
        source = source.resolve()
        if source in seen:
            continue
        seen.add(source)
        result.extend(_source_review_work(source))
    return result


def _source_review_work(source: Path) -> list[_ReviewWork]:
    meta_path = source / "meta.json"
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") != "clothdojo_rollout"
            or metadata.get("status") != "complete"
        ):
            return []
        trajectories = metadata.get("trajectories")
        if not isinstance(trajectories, list):
            return []
        return [
            _ReviewWork(source, Path(item["path"]).name, int(item["action_count"]))
            for item in trajectories
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("action_count"), int)
            and not isinstance(item.get("action_count"), bool)
        ]

    result = []
    for manifest_path in sorted((source / "trajectories").glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        action_count = manifest.get("action_count") if isinstance(manifest, dict) else None
        if (
            not isinstance(manifest, dict)
            or manifest.get("status") != "complete"
            or isinstance(action_count, bool)
            or not isinstance(action_count, int)
        ):
            continue
        result.append(_ReviewWork(source, manifest_path.parent.name, action_count))
    return result


def _balanced_shards(
    work: list[_ReviewWork], gpu_ids: list[int]
) -> list[tuple[int, list[_ReviewWork]]]:
    bins = [(gpu_id, [], 0) for gpu_id in gpu_ids[: len(work)]]
    for item in sorted(
        work,
        key=lambda value: (-value.action_count, str(value.source), value.trajectory_id),
    ):
        index = min(range(len(bins)), key=lambda value: (bins[value][2], bins[value][0]))
        gpu_id, items, frames = bins[index]
        items.append(item)
        bins[index] = (gpu_id, items, frames + item.action_count)
    return [(gpu_id, items) for gpu_id, items, _ in bins]


def _work_by_source(work: list[_ReviewWork]) -> list[tuple[Path, list[_ReviewWork]]]:
    grouped: dict[Path, list[_ReviewWork]] = {}
    for item in work:
        grouped.setdefault(item.source, []).append(item)
    return sorted(grouped.items(), key=lambda item: str(item[0]))


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key != "workers"} | {
        "workers": [
            {key: value for key, value in worker.items() if key != "work"}
            for worker in job["workers"]
        ]
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
