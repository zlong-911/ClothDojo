"""One coordinator for policy services, multi-host rollout, and formal scoring."""

from __future__ import annotations

import fcntl
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from clothdojo.evaluation.results import (
    FORMAL_BENCHMARK_ASSET_COUNT,
    FORMAL_BENCHMARK_PANEL_ID,
)
from clothdojo.evaluation.run_config import (
    BenchmarkRunConfig,
    DeploymentNode,
    load_benchmark_run_config,
)
from clothdojo.rollout.trajectory_collection import (
    TrajectoryAssetJob,
    TrajectoryCollectionSpec,
    atomic_write_json,
)
from clothdojo.runtime_identity import runtime_source_sha256


@dataclass(frozen=True, slots=True)
class _PolicySlot:
    node: DeploymentNode
    gpu: int
    port: int

    @property
    def endpoint(self) -> str:
        host = self.node.connect_host
        if host is None:
            raise ValueError("policy deployment node has no connect_host")
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"ws://{host}:{self.port}"


@dataclass(slots=True)
class _Child:
    label: str
    process: subprocess.Popen
    log: Any
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        _stop_process(self.process)
        self.log.close()


def run_managed_evaluation(path: str | Path) -> dict[str, Any]:
    """Run one formal condition from a single strict configuration file."""

    config = load_benchmark_run_config(path)
    if config.benchmark.panel_id != FORMAL_BENCHMARK_PANEL_ID:
        raise ValueError(
            f"benchmark run requires panel {FORMAL_BENCHMARK_PANEL_ID}, "
            f"got {config.benchmark.panel_id}"
        )
    if config.deployment.worker_count > FORMAL_BENCHMARK_ASSET_COUNT:
        raise ValueError(
            "formal evaluation cannot use more workers than its "
            f"{FORMAL_BENCHMARK_ASSET_COUNT} asset jobs"
        )
    _require_project_path(config.output, config.project_root, "evaluation output")
    policy_slots = _policy_slots(config)
    runtime_sha256 = runtime_source_sha256(config.project_root / "src/clothdojo")
    runtime = config.output / "runtime"
    logs = runtime / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    coordinator_lock = _acquire_coordinator_lock(runtime / "coordinator.lock")
    try:
        atomic_write_json(
            runtime / "managed-run.json",
            {
                "schema": "clothdojo_managed_evaluation_run",
                "condition_id": config.condition_id,
                "config": str(config.source),
                "benchmark": config.benchmark.panel_id,
                "worker_count": config.deployment.worker_count,
                "policy_endpoints": [slot.endpoint for slot in policy_slots],
                "runtime_sha256": runtime_sha256,
            },
        )
    except BaseException:
        coordinator_lock.close()
        raise
    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _raise_termination)
    children: list[_Child] = []
    try:
        for index, slot in enumerate(policy_slots):
            command = _policy_command(config, slot)
            children.append(
                _launch(
                    slot.node,
                    command,
                    environment={
                        "CUDA_VISIBLE_DEVICES": str(slot.gpu),
                        "PYTHONPATH": str(slot.node.project_root / "src"),
                    },
                    log_path=logs / f"policy-{index:03d}.log",
                    label=f"policy service {index}",
                )
            )
        _wait_for_endpoints(
            policy_slots,
            children,
            timeout_seconds=config.deployment.startup_timeout_seconds,
        )
        policy_children = list(children)

        from clothdojo.evaluation.collection import run_evaluation_condition

        simulation_children: list[_Child] = []

        def collect_distributed(
            jobs: list[TrajectoryAssetJob],
            spec: TrajectoryCollectionSpec,
            endpoints: tuple[str, ...],
        ) -> None:
            simulation_children.extend(
                _launch_simulation_nodes(
                    config,
                    jobs,
                    spec,
                    endpoints,
                    logs,
                    runtime_sha256=runtime_sha256,
                )
            )
            children.extend(simulation_children)
            _wait_for_children(
                simulation_children,
                "trajectory collection",
                dependencies=tuple(policy_children),
            )
            for child in policy_children:
                child.close()

        logical_workers = tuple(
            f"worker-{index:03d}" for index in range(config.deployment.worker_count)
        )
        return run_evaluation_condition(
            config.benchmark.scenario,
            config.benchmark.protocol,
            config.benchmark.states,
            config.benchmark.templates,
            tuple(slot.endpoint for slot in policy_slots),
            config.output,
            condition_id=config.condition_id,
            cloth_visual=config.execution.cloth_visual,
            expected_checkpoint_sha256=config.policy.expected_checkpoint_sha256,
            image_preprocessing=config.execution.image_preprocessing,
            assets_root=config.benchmark.assets,
            gpus=logical_workers,
            execution=config.execution.mode,
            exec_horizon=config.execution.exec_horizon,
            rtc_delay_steps=config.execution.rtc_delay_steps,
            batch_size=4,
            defer_scoring=False,
            score_workers=config.deployment.score_workers,
            trajectory_collector=collect_distributed,
            expected_policy_runtime_sha256=runtime_sha256,
            runtime_sha256=runtime_sha256,
        )
    finally:
        try:
            for child in reversed(children):
                child.close()
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)
            coordinator_lock.close()


def _policy_slots(config: BenchmarkRunConfig) -> tuple[_PolicySlot, ...]:
    slots = []
    endpoints = set()
    for node in config.deployment.policy_nodes:
        if node.port_start is None:
            raise ValueError("policy deployment node has no port_start")
        for offset, gpu in enumerate(node.gpus):
            slot = _PolicySlot(node=node, gpu=gpu, port=node.port_start + offset)
            if slot.endpoint in endpoints:
                raise ValueError(f"duplicate policy endpoint: {slot.endpoint}")
            endpoints.add(slot.endpoint)
            slots.append(slot)
    return tuple(slots)


def _policy_command(config: BenchmarkRunConfig, slot: _PolicySlot) -> list[str]:
    policy = config.policy
    checkpoint = _map_project_path(policy.checkpoint, config.project_root, slot.node)
    norm_stats = _map_project_path(policy.norm_stats, config.project_root, slot.node)
    command = [
        str(slot.node.python),
        "-m",
        "clothdojo",
        "policy",
        f"serve-{policy.kind}",
        "--checkpoint",
        str(checkpoint),
        "--norm-stats",
        str(norm_stats),
    ]
    options = dict(policy.options)
    configured_preprocessing = options.pop("image_preprocessing", None)
    if (
        configured_preprocessing is not None
        and configured_preprocessing != config.execution.image_preprocessing
    ):
        raise ValueError(
            "policy and execution image_preprocessing must match: "
            f"{configured_preprocessing!r} != {config.execution.image_preprocessing!r}"
        )
    if policy.kind == "jax":
        training = options.pop("training_config")
        training_path = _map_project_path(Path(training), config.project_root, slot.node)
        command.extend(["--training-config", str(training_path)])
    command.extend(["--image-preprocessing", config.execution.image_preprocessing])
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            command.extend([flag, str(value)])
        else:
            raise TypeError(f"unsupported policy option {key}: {value!r}")
    if policy.kind != "jax":
        command.extend(["--device", "cuda:0"])
    command.extend(["--host", "0.0.0.0", "--port", str(slot.port)])
    return command


def _launch_simulation_nodes(
    config: BenchmarkRunConfig,
    jobs: list[TrajectoryAssetJob],
    spec: TrajectoryCollectionSpec,
    endpoints: tuple[str, ...],
    logs: Path,
    *,
    runtime_sha256: str,
) -> list[_Child]:
    worker_count = config.deployment.worker_count
    if len(endpoints) != worker_count:
        raise ValueError("managed collector endpoint count differs from deployment")
    assignments = [jobs[index::worker_count] for index in range(worker_count)]
    runtime = config.output / "runtime" / "nodes"
    runtime.mkdir(parents=True, exist_ok=True)
    children = []
    cursor = 0
    try:
        for node_index, node in enumerate(config.deployment.simulation_nodes):
            count = len(node.gpus)
            node_jobs = [job for group in assignments[cursor : cursor + count] for job in group]
            node_endpoints = endpoints[cursor : cursor + count]
            cursor += count
            payload = {
                "schema": "clothdojo_evaluation_node_plan",
                "node_index": node_index,
                "gpus": [str(gpu) for gpu in node.gpus],
                "policy_endpoints": list(node_endpoints),
                "runtime_sha256": runtime_sha256,
                "jobs": [
                    {
                        **asdict(job),
                        "pool": str(
                            _map_project_path(Path(job.pool), config.project_root, node)
                        ),
                    }
                    for job in node_jobs
                ],
                "spec": _mapped_spec(spec, config.project_root, node),
            }
            local_plan = runtime / f"node-{node_index:03d}.json"
            atomic_write_json(local_plan, payload)
            remote_plan = _map_project_path(local_plan, config.project_root, node)
            children.append(
                _launch(
                    node,
                    [
                        str(node.python),
                        "-m",
                        "clothdojo.evaluation.worker",
                        "--plan",
                        str(remote_plan),
                    ],
                    environment={"PYTHONPATH": str(node.project_root / "src")},
                    log_path=logs / f"simulation-node-{node_index:03d}.log",
                    label=f"simulation node {node_index}",
                )
            )
    except Exception:
        for child in reversed(children):
            child.close()
        raise
    return children


def _mapped_spec(
    spec: TrajectoryCollectionSpec,
    local_root: Path,
    node: DeploymentNode,
) -> dict[str, Any]:
    result = asdict(spec)
    for field in ("scenario_path", "assets_root", "trajectories_root"):
        result[field] = str(
            _map_project_path(Path(result[field]), local_root, node)
        )
    return result


def _launch(
    node: DeploymentNode,
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
    label: str,
) -> _Child:
    log = log_path.open("ab", buffering=0)
    try:
        if node.local:
            process = subprocess.Popen(
                command,
                cwd=node.project_root,
                env={**os.environ, **environment},
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            remote = ["env", *[f"{key}={value}" for key, value in environment.items()], *command]
            remote_command = (
                f"cd {shlex.quote(str(node.project_root))} && "
                "child=; "
                "trap 'if [ -n \"$child\" ]; then "
                "kill -TERM -- -\"$child\" 2>/dev/null; "
                "wait \"$child\" 2>/dev/null; fi' HUP INT TERM EXIT; "
                f"setsid {shlex.join(remote)} & child=$!; "
                "wait \"$child\"; status=$?; trap - EXIT; exit \"$status\""
            )
            process = subprocess.Popen(
                [
                    "ssh",
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=15",
                    "-o",
                    "ServerAliveInterval=15",
                    "-o",
                    "ServerAliveCountMax=3",
                    node.host,
                    remote_command,
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception:
        log.close()
        raise
    return _Child(label=label, process=process, log=log)


def _wait_for_endpoints(
    slots: tuple[_PolicySlot, ...],
    children: list[_Child],
    *,
    timeout_seconds: int,
) -> None:
    pending = {slot.endpoint for slot in slots}
    deadline = time.monotonic() + timeout_seconds
    while pending:
        failed = [child for child in children if child.process.poll() is not None]
        if failed:
            raise RuntimeError(
                f"{failed[0].label} exited before becoming ready with code "
                f"{failed[0].process.returncode}"
            )
        for endpoint in tuple(pending):
            parsed = urlparse(endpoint)
            try:
                with socket.create_connection(
                    (parsed.hostname or "", parsed.port or 0), timeout=1.0
                ):
                    pending.remove(endpoint)
            except OSError:
                pass
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "policy service startup timed out: " + ", ".join(sorted(pending))
            )
        time.sleep(1.0)


def _wait_for_children(
    children: list[_Child], label: str, *, dependencies: tuple[_Child, ...] = ()
) -> None:
    pending = set(range(len(children)))
    while pending:
        failed_dependency = next(
            (child for child in dependencies if child.process.poll() is not None), None
        )
        if failed_dependency is not None:
            for index in pending:
                _stop_process(children[index].process)
            raise RuntimeError(
                f"{failed_dependency.label} exited during {label} with code "
                f"{failed_dependency.process.returncode}"
            )
        for index in tuple(pending):
            code = children[index].process.poll()
            if code is None:
                continue
            pending.remove(index)
            if code != 0:
                for other in pending:
                    _stop_process(children[other].process)
                raise RuntimeError(
                    f"{label} failed in {children[index].label} with exit code {code}"
                )
        if pending:
            time.sleep(1.0)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def _raise_termination(signum, _frame) -> None:
    raise KeyboardInterrupt(f"managed evaluation received signal {signum}")


def _acquire_coordinator_lock(path: Path):
    lock = path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(f"another benchmark coordinator owns {path.parent}") from exc
    return lock


def _map_project_path(path: Path, local_root: Path, node: DeploymentNode) -> Path:
    relative = _require_project_path(path, local_root, "managed evaluation path")
    return node.project_root / relative


def _require_project_path(path: Path, root: Path, label: str) -> Path:
    try:
        return path.expanduser().resolve().relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the project root: {path}") from exc
