"""Durable sequential runner for reviewed OpenPI/JAX SFT trials."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clothdojo.training.jax_sft import (
    Gate,
    compose_jax_sft_spec,
    gate_manifest_path,
    inspect_jax_sft,
)


@dataclass(frozen=True, slots=True)
class MatrixTrial:
    name: str
    overrides: tuple[str, ...]
    stages: tuple[Gate, ...]


@dataclass(frozen=True, slots=True)
class JaxSftMatrix:
    name: str
    config_path: Path
    base_config: Path
    status_dir: Path
    continue_on_error: bool
    trials: tuple[MatrixTrial, ...]

    @property
    def status_path(self) -> Path:
        return self.status_dir / self.name / "status.json"


def compose_jax_sft_matrix(config_path: str | Path) -> JaxSftMatrix:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JAX SFT matrix config does not exist: {path}")

    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("JAX SFT matrix config must resolve to a mapping")
    if raw.get("schema") != "clothdojo_pi05_jax_matrix" or raw.get("format_version") != 1:
        raise ValueError("unsupported JAX SFT matrix schema")

    base = Path(str(raw["base_config"])).expanduser()
    if not base.is_absolute():
        base = (path.parent / base).resolve()
    status_dir = Path(str(raw["status_dir"])).expanduser().resolve()
    stages = _stages(raw.get("stages", ("initialize", "overfit", "pilot", "parity", "full")))
    entries = raw.get("trials")
    if not isinstance(entries, list) or not entries:
        raise ValueError("JAX SFT matrix requires at least one trial")

    trials: list[MatrixTrial] = []
    names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("each JAX SFT matrix trial must be a mapping")
        name = str(entry["name"])
        if name in names:
            raise ValueError(f"duplicate JAX SFT matrix trial name: {name}")
        names.add(name)
        overrides = entry.get("overrides", ())
        if not isinstance(overrides, Sequence) or isinstance(overrides, str):
            raise TypeError(f"matrix trial {name} overrides must be a list")
        trial_stages = _stages(entry.get("stages", stages))
        trials.append(
            MatrixTrial(
                name=name,
                overrides=tuple(str(value) for value in overrides),
                stages=trial_stages,
            )
        )

    return JaxSftMatrix(
        name=str(raw["name"]),
        config_path=path,
        base_config=base,
        status_dir=status_dir,
        continue_on_error=bool(raw.get("continue_on_error", True)),
        trials=tuple(trials),
    )


def inspect_jax_sft_matrix(matrix: JaxSftMatrix, *, inspect_artifacts: bool) -> dict[str, Any]:
    trials = []
    for trial in matrix.trials:
        spec = compose_jax_sft_spec(matrix.base_config, trial.overrides)
        item: dict[str, Any] = {
            "name": trial.name,
            "profile": spec.tuning_profile,
            "run_name": spec.run_name,
            "stages": list(trial.stages),
            "optimizer": {
                "warmup_steps": spec.warmup_steps,
                "peak_lr": spec.peak_lr,
                "decay_steps": spec.decay_steps,
                "decay_lr": spec.decay_lr,
            },
        }
        if inspect_artifacts:
            item["preflight"] = inspect_jax_sft(spec)
        trials.append(item)
    return {
        "schema": "clothdojo_pi05_jax_matrix_plan",
        "format_version": 1,
        "name": matrix.name,
        "host": platform.node(),
        "base_config": str(matrix.base_config),
        "status_path": str(matrix.status_path),
        "continue_on_error": matrix.continue_on_error,
        "trials": trials,
    }


def run_jax_sft_matrix(matrix: JaxSftMatrix) -> dict[str, Any]:
    report = {
        "schema": "clothdojo_pi05_jax_matrix_plan",
        "format_version": 1,
        "name": matrix.name,
        "host": platform.node(),
        "base_config": str(matrix.base_config),
        "status_path": str(matrix.status_path),
        "continue_on_error": matrix.continue_on_error,
        "trials": [
            {"name": trial.name, "stages": list(trial.stages)}
            for trial in matrix.trials
        ],
    }
    report["started_at"] = _now()
    report["state"] = "running"
    _write_status(matrix.status_path, report)

    for item, trial in zip(report["trials"], matrix.trials, strict=True):
        item["state"] = "running"
        item["started_at"] = _now()
        item["gates"] = []
        _write_status(matrix.status_path, report)
        try:
            spec = compose_jax_sft_spec(matrix.base_config, trial.overrides)
            item.update(
                {
                    "profile": spec.tuning_profile,
                    "run_name": spec.run_name,
                    "optimizer": {
                        "warmup_steps": spec.warmup_steps,
                        "peak_lr": spec.peak_lr,
                        "decay_steps": spec.decay_steps,
                        "decay_lr": spec.decay_lr,
                    },
                }
            )
            preflight = _inspect_in_subprocess(matrix.base_config, trial.overrides)
            identity = preflight["artifact_sha256"]
            item["preflight_sha256"] = identity
        except Exception as exc:
            item["state"] = "failed"
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["finished_at"] = _now()
            _write_status(matrix.status_path, report)
            if not matrix.continue_on_error:
                report["state"] = "failed"
                report["finished_at"] = _now()
                _write_status(matrix.status_path, report)
                raise
            continue

        for stage in trial.stages:
            existing = _compatible_gate(spec, stage, identity)
            if existing is not None:
                item["gates"].append({"gate": stage, "state": "skipped", "manifest": existing})
                _write_status(matrix.status_path, report)
                continue
            try:
                _run_gate_in_subprocess(matrix.base_config, trial.overrides, stage)
                result = _read_gate(spec, stage)
            except Exception as exc:
                item["gates"].append(
                    {"gate": stage, "state": "failed", "error": f"{type(exc).__name__}: {exc}"}
                )
                item["state"] = "failed"
                item["finished_at"] = _now()
                _write_status(matrix.status_path, report)
                if not matrix.continue_on_error:
                    report["state"] = "failed"
                    report["finished_at"] = _now()
                    _write_status(matrix.status_path, report)
                    raise
                break
            item["gates"].append(
                {
                    "gate": stage,
                    "state": "completed",
                    "compatible": result.get("compatible"),
                    "manifest": str(gate_manifest_path(spec, stage)),
                }
            )
            _write_status(matrix.status_path, report)
        else:
            item["state"] = "completed"
            item["finished_at"] = _now()
            _write_status(matrix.status_path, report)

    report["state"] = (
        "failed" if any(item.get("state") == "failed" for item in report["trials"]) else "completed"
    )
    report["finished_at"] = _now()
    _write_status(matrix.status_path, report)
    return report


def _stages(values: Any) -> tuple[Gate, ...]:
    allowed = {"initialize", "overfit", "pilot", "parity", "full"}
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError("matrix stages must be a list")
    result = tuple(str(value) for value in values)
    invalid = set(result) - allowed
    if invalid:
        raise ValueError(f"unknown matrix stages: {sorted(invalid)}")
    if not result:
        raise ValueError("matrix stages cannot be empty")
    return result  # type: ignore[return-value]


def _compatible_gate(spec: Any, gate: Gate, identity: str) -> str | None:
    path = gate_manifest_path(spec, gate)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if payload.get("compatible") is True and payload.get("preflight_sha256") == identity:
        return str(path)
    return None


def _inspect_in_subprocess(base_config: Path, overrides: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(
        _command(base_config, overrides, check=True),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or payload.get("compatible") is not True:
        raise RuntimeError("JAX SFT subprocess preflight did not return a compatible manifest")
    return payload


def _run_gate_in_subprocess(base_config: Path, overrides: Sequence[str], gate: Gate) -> None:
    subprocess.run(_command(base_config, overrides, gate=gate), check=True)


def _command(
    base_config: Path,
    overrides: Sequence[str],
    *,
    check: bool = False,
    gate: Gate | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-c",
        "from clothdojo.cli import main; main()",
        "train",
        "openpi-jax",
        "--config",
        str(base_config),
    ]
    if check:
        command.append("--check")
    if gate is not None:
        command.extend(("--gate", gate))
    command.extend(overrides)
    return command


def _read_gate(spec: Any, gate: Gate) -> dict[str, Any]:
    path = gate_manifest_path(spec, gate)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"gate subprocess did not produce a valid manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("compatible") is not True:
        raise RuntimeError(f"gate subprocess produced an incompatible manifest: {path}")
    return payload


def _write_status(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat()
