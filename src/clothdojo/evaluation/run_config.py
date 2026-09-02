"""Strict, small configuration surface for one managed formal evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RUN_CONFIG_SCHEMA = "clothdojo_benchmark_run"
DEPLOYMENT_CONFIG_SCHEMA = "clothdojo_evaluation_deployment"
BENCHMARK_CONFIG_SCHEMA = "clothdojo_frozen_benchmark"
DEFAULT_BENCHMARK_CONFIG = Path("configs/evaluation/balanced200_20260901.yaml")


@dataclass(frozen=True, slots=True)
class BenchmarkPaths:
    panel_id: str
    scenario: Path
    protocol: Path
    states: Path
    templates: Path
    assets: Path


@dataclass(frozen=True, slots=True)
class PolicyLaunchConfig:
    kind: str
    checkpoint: Path
    norm_stats: Path
    options: dict[str, Any]
    expected_checkpoint_sha256: str | None


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    mode: str
    exec_horizon: int
    rtc_delay_steps: int
    cloth_visual: str
    image_preprocessing: str


@dataclass(frozen=True, slots=True)
class DeploymentNode:
    host: str
    connect_host: str | None
    project_root: Path
    python: Path
    gpus: tuple[int, ...]
    port_start: int | None = None

    @property
    def local(self) -> bool:
        return self.host == "local"


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    policy_nodes: tuple[DeploymentNode, ...]
    simulation_nodes: tuple[DeploymentNode, ...]
    score_workers: int
    startup_timeout_seconds: int

    @property
    def worker_count(self) -> int:
        return sum(len(node.gpus) for node in self.simulation_nodes)


@dataclass(frozen=True, slots=True)
class BenchmarkRunConfig:
    source: Path
    project_root: Path
    condition_id: str
    output: Path
    benchmark: BenchmarkPaths
    policy: PolicyLaunchConfig
    execution: ExecutionConfig
    deployment: DeploymentConfig


def load_benchmark_run_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> BenchmarkRunConfig:
    """Load the only user-facing formal evaluation configuration."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    source = _project_path(path, root, "benchmark run config")
    value = _yaml_mapping(source, "benchmark run config")
    _keys(
        value,
        {"schema", "condition_id", "output", "benchmark", "policy", "execution", "deployment"},
        "benchmark run config",
        required={"schema", "condition_id", "policy", "deployment"},
    )
    if value["schema"] != RUN_CONFIG_SCHEMA:
        raise ValueError(f"unsupported benchmark run schema: {value['schema']!r}")
    condition_id = _identifier(value["condition_id"], "condition_id")
    output_value = value.get("output", f"outputs/evaluations/{condition_id}")
    output = _project_path(output_value, root, "evaluation output")
    benchmark_path = _project_path(
        value.get("benchmark", DEFAULT_BENCHMARK_CONFIG), root, "benchmark config"
    )
    deployment_path = _project_path(value["deployment"], root, "deployment config")
    policy = _load_policy(value["policy"], root)
    execution = _load_execution(value.get("execution", {}))
    if policy.kind == "jax" and execution.image_preprocessing != "none":
        raise ValueError("JAX formal evaluation requires image_preprocessing='none'")
    configured_preprocessing = policy.options.get("image_preprocessing")
    if (
        configured_preprocessing is not None
        and configured_preprocessing != execution.image_preprocessing
    ):
        raise ValueError("policy and execution image_preprocessing must match")
    return BenchmarkRunConfig(
        source=source,
        project_root=root,
        condition_id=condition_id,
        output=output,
        benchmark=load_benchmark_paths(benchmark_path, project_root=root),
        policy=policy,
        execution=execution,
        deployment=load_deployment_config(deployment_path, project_root=root),
    )


def load_benchmark_paths(
    path: str | Path,
    *,
    project_root: str | Path,
) -> BenchmarkPaths:
    root = Path(project_root).expanduser().resolve()
    source = _project_path(path, root, "benchmark config")
    value = _yaml_mapping(source, "benchmark config")
    required = {"schema", "panel_id", "scenario", "protocol", "states", "templates", "assets"}
    _keys(value, required, "benchmark config", required=required)
    if value["schema"] != BENCHMARK_CONFIG_SCHEMA:
        raise ValueError(f"unsupported benchmark config schema: {value['schema']!r}")
    return BenchmarkPaths(
        panel_id=_text(value["panel_id"], "benchmark panel_id"),
        scenario=_project_path(value["scenario"], root, "benchmark scenario"),
        protocol=_project_path(value["protocol"], root, "benchmark protocol"),
        states=_project_path(value["states"], root, "benchmark states"),
        templates=_project_path(value["templates"], root, "benchmark templates"),
        assets=_project_path(value["assets"], root, "benchmark assets"),
    )


def load_deployment_config(
    path: str | Path,
    *,
    project_root: str | Path,
) -> DeploymentConfig:
    root = Path(project_root).expanduser().resolve()
    source = _project_path(path, root, "deployment config")
    value = _yaml_mapping(source, "deployment config")
    _keys(
        value,
        {"schema", "policy", "simulation", "score_workers", "startup_timeout_seconds"},
        "deployment config",
        required={"schema", "policy", "simulation"},
    )
    if value["schema"] != DEPLOYMENT_CONFIG_SCHEMA:
        raise ValueError(f"unsupported deployment schema: {value['schema']!r}")
    policy = _nodes(value["policy"], root, "policy", require_ports=True)
    simulation = _nodes(value["simulation"], root, "simulation", require_ports=False)
    policy_workers = sum(len(node.gpus) for node in policy)
    simulation_workers = sum(len(node.gpus) for node in simulation)
    if policy_workers != simulation_workers:
        raise ValueError(
            "deployment requires one policy service per simulation worker: "
            f"policy={policy_workers}, simulation={simulation_workers}"
        )
    if simulation_workers < 1:
        raise ValueError("deployment requires at least one evaluation worker")
    _validate_unique_gpus(policy, "policy")
    _validate_unique_gpus(simulation, "simulation")
    return DeploymentConfig(
        policy_nodes=policy,
        simulation_nodes=simulation,
        score_workers=_positive_int(value.get("score_workers", 1), "score_workers"),
        startup_timeout_seconds=_positive_int(
            value.get("startup_timeout_seconds", 1200), "startup_timeout_seconds"
        ),
    )


def _load_policy(value: Any, root: Path) -> PolicyLaunchConfig:
    policy = _mapping(value, "policy")
    common = {
        "kind",
        "checkpoint",
        "norm_stats",
        "expected_checkpoint_sha256",
        "num_steps",
        "image_preprocessing",
    }
    rtc = {
        "rtc_guidance_horizon",
        "rtc_max_guidance_weight",
        "rtc_prefix_schedule",
    }
    cfg_only = {
        "guidance_scale",
        "guidance_type",
        "positive_only_conditional",
        "no_autocast",
    }
    jax_only = {"training_config"}
    kind = _text(policy.get("kind"), "policy kind")
    if kind not in {"sft", "cfg", "jax"}:
        raise ValueError("policy kind must be 'sft', 'cfg' or 'jax'")
    allowed = (
        common
        | (rtc if kind in {"sft", "cfg"} else set())
        | (cfg_only if kind == "cfg" else set())
        | (jax_only if kind == "jax" else set())
    )
    required = {"kind", "checkpoint", "norm_stats"} | (
        {"training_config"} if kind == "jax" else set()
    )
    _keys(policy, allowed, "policy", required=required)
    expected = policy.get("expected_checkpoint_sha256")
    if expected is not None:
        expected = _sha256(expected, "expected_checkpoint_sha256")
    identity_fields = required | {"expected_checkpoint_sha256"}
    options = {key: value for key, value in policy.items() if key not in identity_fields}
    if "training_config" in policy:
        options["training_config"] = str(
            _project_path(policy["training_config"], root, "policy training_config")
        )
    _validate_policy_options(kind, options)
    return PolicyLaunchConfig(
        kind=kind,
        checkpoint=_project_path(policy["checkpoint"], root, "policy checkpoint"),
        norm_stats=_project_path(policy["norm_stats"], root, "policy norm_stats"),
        options=options,
        expected_checkpoint_sha256=expected,
    )


def _load_execution(value: Any) -> ExecutionConfig:
    execution = _mapping(value, "execution")
    _keys(
        execution,
        {"mode", "exec_horizon", "rtc_delay_steps", "cloth_visual", "image_preprocessing"},
        "execution",
    )
    mode = execution.get("mode", "chunk")
    if mode not in {"chunk", "rtc"}:
        raise ValueError("execution mode must be 'chunk' or 'rtc'")
    cloth_visual = execution.get("cloth_visual", "nocs")
    if cloth_visual not in {"nocs", "white"}:
        raise ValueError("cloth_visual must be 'nocs' or 'white'")
    preprocessing = execution.get("image_preprocessing", "none")
    if preprocessing not in {"none", "training_fixed_crop"}:
        raise ValueError("unsupported evaluation image_preprocessing")
    delay = _non_negative_int(execution.get("rtc_delay_steps", 0), "rtc_delay_steps")
    if mode == "chunk" and delay:
        raise ValueError("chunk execution does not accept rtc_delay_steps")
    return ExecutionConfig(
        mode=mode,
        exec_horizon=_positive_int(execution.get("exec_horizon", 10), "exec_horizon"),
        rtc_delay_steps=delay,
        cloth_visual=cloth_visual,
        image_preprocessing=preprocessing,
    )


def _nodes(
    value: Any, root: Path, label: str, *, require_ports: bool
) -> tuple[DeploymentNode, ...]:
    section = _mapping(value, f"deployment {label}")
    _keys(section, {"nodes"}, f"deployment {label}", required={"nodes"})
    raw_nodes = section["nodes"]
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError(f"deployment {label} nodes must be a non-empty list")
    nodes = []
    for index, raw in enumerate(raw_nodes):
        name = f"deployment {label} node {index}"
        node = _mapping(raw, name)
        allowed = {"host", "connect_host", "project_root", "python", "gpus"} | (
            {"port_start"} if require_ports else set()
        )
        required = {"host", "gpus"} | ({"port_start"} if require_ports else set())
        _keys(node, allowed, name, required=required)
        host = _host(node["host"], f"{name} host")
        local = host == "local"
        if not local and "project_root" not in node:
            raise ValueError(f"{name} requires project_root for remote path mapping")
        project = (
            root
            if local and "project_root" not in node
            else Path(_text(node["project_root"], f"{name} project_root")).expanduser()
        )
        if not project.is_absolute():
            if not local:
                raise ValueError(f"{name} project_root must be absolute")
            project = (root / project).resolve()
        python_value = node.get("python", ".venv/bin/python")
        python = Path(_text(python_value, f"{name} python")).expanduser()
        if not python.is_absolute():
            python = project / python
        gpus_value = node["gpus"]
        if not isinstance(gpus_value, list) or not gpus_value:
            raise ValueError(f"{name} gpus must be a non-empty list")
        gpus = tuple(_non_negative_int(gpu, f"{name} gpu") for gpu in gpus_value)
        if len(gpus) != len(set(gpus)):
            raise ValueError(f"{name} contains duplicate GPUs")
        connect_host = node.get("connect_host", "127.0.0.1" if local else None)
        if connect_host is None and require_ports:
            raise ValueError(f"{name} requires connect_host")
        if connect_host is not None:
            connect_host = _connect_host(connect_host, f"{name} connect_host")
        port = None
        if require_ports:
            port = _positive_int(node["port_start"], f"{name} port_start")
            if port + len(gpus) - 1 > 65535:
                raise ValueError(f"{name} policy port range exceeds 65535")
        nodes.append(
            DeploymentNode(
                host=host,
                connect_host=connect_host,
                project_root=project,
                python=python,
                gpus=gpus,
                port_start=port,
            )
        )
    return tuple(nodes)


def _yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"cannot read {label} {path}: {exc}") from exc
    return _mapping(value, label)


def _validate_policy_options(kind: str, options: dict[str, Any]) -> None:
    if "num_steps" in options:
        _positive_int(options["num_steps"], "policy num_steps")
    preprocessing = options.get("image_preprocessing")
    if preprocessing is not None and preprocessing not in {"none", "training_fixed_crop"}:
        raise ValueError("unsupported policy image_preprocessing")
    if "rtc_guidance_horizon" in options:
        _positive_int(options["rtc_guidance_horizon"], "policy rtc_guidance_horizon")
    if "rtc_max_guidance_weight" in options:
        _non_negative_number(
            options["rtc_max_guidance_weight"], "policy rtc_max_guidance_weight"
        )
    if "rtc_prefix_schedule" in options and options["rtc_prefix_schedule"] not in {
        "linear",
        "exp",
        "ones",
        "zeros",
    }:
        raise ValueError("unsupported policy rtc_prefix_schedule")
    if kind == "cfg":
        if "guidance_scale" in options:
            _non_negative_number(options["guidance_scale"], "policy guidance_scale")
        if "guidance_type" in options and options["guidance_type"] not in {
            "positive",
            "negative",
            "no_guide",
        }:
            raise ValueError("unsupported policy guidance_type")
        for name in ("positive_only_conditional", "no_autocast"):
            if name in options and not isinstance(options[name], bool):
                raise TypeError(f"policy {name} must be a bool")


def _validate_unique_gpus(nodes: tuple[DeploymentNode, ...], label: str) -> None:
    seen = set()
    for node in nodes:
        for gpu in node.gpus:
            key = (node.host, gpu)
            if key in seen:
                raise ValueError(f"deployment {label} reuses GPU {gpu} on {node.host}")
            seen.add(key)


def _project_path(value: Any, root: Path, label: str) -> Path:
    text = str(value) if isinstance(value, Path) else _text(value, label)
    path = Path(text).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _keys(
    value: dict[str, Any],
    allowed: set[str],
    label: str,
    *,
    required: set[str] = frozenset(),
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text) is None:
        raise ValueError(f"{label} must contain only letters, digits, '.', '_' or '-'")
    return text


def _host(value: Any, label: str) -> str:
    text = _text(value, label)
    if text == "local":
        return text
    if re.fullmatch(r"[A-Za-z0-9_.@:-]+", text) is None or text.startswith("-"):
        raise ValueError(f"{label} is not a safe SSH host or alias")
    return text


def _connect_host(value: Any, label: str) -> str:
    text = _text(value, label)
    if re.fullmatch(r"[A-Za-z0-9_.:-]+", text) is None or text.startswith("-"):
        raise ValueError(f"{label} is not a hostname or IP address")
    return text


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _non_negative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a non-negative number")
    return float(value)


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text
