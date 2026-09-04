"""Project-owned STEAM-labeling to offline-CFG training pipeline."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clothdojo.training.steam_advantages import (
    SteamAdvantageSpec,
    compose_steam_advantage_spec,
    inspect_steam_advantages,
)


def inspect_cfg_pipeline(
    *,
    advantage_config: str | Path,
    cfg_config: str | Path,
    cfg_overrides: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Validate artifact ownership and cross-config identity before launch."""

    from clothdojo.training.cfg import resolve_cfg_episode_selection
    from clothdojo.training.rlinf import compose_config

    project_root = _project_root()
    spec = compose_steam_advantage_spec(advantage_config)
    advantage = inspect_steam_advantages(spec)
    cfg = compose_config(cfg_config, cfg_overrides)
    checkpoint_roots = (
        project_root / "data" / "checkpoints",
        project_root / "outputs" / "training",
    )
    _require_within_any(spec.checkpoint, checkpoint_roots, "STEAM checkpoint")
    _require_within(spec.work_dir, project_root / "outputs", "STEAM work directory")
    _require_within_any(
        Path(str(cfg.actor.model.model_path)).expanduser().resolve(),
        checkpoint_roots,
        "CFG initialization checkpoint",
    )
    cfg_entries = list(cfg.data.train_data_paths)
    cfg_roles = [str(entry.type) for entry in cfg_entries]
    if cfg_roles.count("sft") != 1:
        raise ValueError(
            "CFG pipeline requires exactly one sft and optional rollout entries"
        )
    spec_roles = [dataset.role for dataset in spec.datasets]
    if spec_roles.count("sft") != 1:
        raise ValueError(
            "CFG pipeline requires exactly one STEAM sft and optional rollout datasets"
        )
    cfg_by_identity = {
        (
            str(entry.type),
            Path(str(entry.dataset_path)).expanduser().resolve(),
        ): entry
        for entry in cfg_entries
    }
    if len(cfg_by_identity) != len(cfg_entries):
        raise ValueError("CFG pipeline dataset entries must be unique by role and path")
    steam_by_identity = {
        (dataset.role, dataset.dataset_path): dataset for dataset in spec.datasets
    }
    if len(steam_by_identity) != len(spec.datasets):
        raise ValueError("STEAM pipeline datasets must be unique by role and path")
    if set(cfg_by_identity) != set(steam_by_identity):
        raise ValueError("STEAM and CFG dataset role/path identities differ")
    for dataset in spec.datasets:
        _require_within(
            dataset.dataset_path,
            project_root / "data" / "datasets",
            "CFG dataset",
        )
        cfg_entry = cfg_by_identity[(dataset.role, dataset.dataset_path)]
        if cfg_entry.get("advantage_tag") != dataset.advantage_tag:
            raise ValueError(
                f"STEAM and CFG {dataset.role} advantage tags differ: "
                f"{dataset.advantage_tag!r} != {cfg_entry.get('advantage_tag')!r}"
            )
        selected = resolve_cfg_episode_selection(
            episodes=cfg_entry.get("episodes"),
            episodes_path=cfg_entry.get("episodes_path"),
            total_episodes=dataset.total_episodes,
        )
        if tuple(int(value) for value in selected) != dataset.episodes:
            raise ValueError(
                f"STEAM and CFG {dataset.role} episode selections differ"
            )

    outputs = [dataset.output_path for dataset in spec.datasets]
    manifests = [path.with_suffix(".json") for path in outputs]
    present = [path.is_file() for path in (*outputs, *manifests)]
    if any(present) and not all(present):
        raise ValueError("CFG pipeline found a partial set of STEAM label artifacts")
    labels_complete = all(present)
    cfg_gate = None
    if labels_complete:
        from clothdojo.training.cfg import inspect_cfg_training

        _validate_existing_label_manifests(spec, advantage)
        cfg_gate = inspect_cfg_training(cfg_config, cfg_overrides)
    return {
        "compatible": True,
        "schema": "clothdojo_cfg_pipeline_plan",
        "format_version": 1,
        "project_root": str(project_root),
        "advantage": advantage,
        "cfg_config": str(Path(cfg_config).expanduser().resolve()),
        "labels_complete": labels_complete,
        "cfg_data_gate": cfg_gate,
        "runtime_dependency": "pinned rlinf package",
    }


def _validate_existing_label_manifests(
    spec: SteamAdvantageSpec,
    advantage: dict[str, Any],
) -> None:
    common = {
        "prompt": spec.prompt,
        "checkpoint_config_sha256": advantage["checkpoint_config_sha256"],
        "checkpoint_weights_sha256": advantage["checkpoint_weights_sha256"],
        "stride": spec.stride,
        "state_key": spec.state_key,
        "camera_keys": list(spec.camera_keys),
    }
    for dataset in spec.datasets:
        path = dataset.output_path.with_suffix(".json")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read STEAM label provenance: {path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise TypeError(f"STEAM label provenance must be an object: {path}")
        expected = {
            **common,
            "role": dataset.role,
            "positive_fraction_requested": dataset.positive_fraction,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(
                    f"existing STEAM labels differ from the current {field}: {path}"
                )


def run_cfg_pipeline(
    *,
    advantage_config: str | Path,
    cfg_config: str | Path,
    cfg_overrides: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Label the configured datasets, run the exact data gate, then launch CFG."""

    from clothdojo.training.cfg import inspect_cfg_training
    from clothdojo.training.rlinf import compose_config, run_config

    plan = inspect_cfg_pipeline(
        advantage_config=advantage_config,
        cfg_config=cfg_config,
        cfg_overrides=cfg_overrides,
    )
    spec = compose_steam_advantage_spec(advantage_config)
    status_path = spec.work_dir / "cfg_pipeline.json"
    plan_path = spec.work_dir / "cfg_pipeline.plan.json"
    lock_path = spec.work_dir / "cfg_pipeline.lock"
    spec.work_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(lock_path):
        _write_json_atomic(plan_path, plan)
        _write_status(status_path, spec, status="running", stage="preflight")
        try:
            if not plan["labels_complete"]:
                _write_status(status_path, spec, status="running", stage="steam_labeling")
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node",
                    str(spec.num_processes),
                    "--master_addr",
                    "127.0.0.1",
                    "--master_port",
                    str(spec.master_port),
                    "-m",
                    "clothdojo.training.steam_advantages",
                    "--config",
                    str(spec.config_path),
                ]
                environment = os.environ.copy()
                # Labeling is single-node. Avoid slow or unavailable pod-level
                # InfiniBand discovery and bind rendezvous traffic to loopback.
                environment.setdefault("NCCL_IB_DISABLE", "1")
                environment.setdefault("NCCL_SOCKET_IFNAME", "lo")
                environment.setdefault("GLOO_SOCKET_IFNAME", "lo")
                subprocess.run(
                    command,
                    cwd=_project_root(),
                    env=environment,
                    check=True,
                )

            _write_status(status_path, spec, status="running", stage="cfg_data_gate")
            gate = inspect_cfg_training(cfg_config, cfg_overrides)
            _write_status(status_path, spec, status="running", stage="cfg_training")
            cfg = compose_config(cfg_config, cfg_overrides)
            run_config(kind="cfg", cfg=cfg, cfg_manifest=gate)
        except BaseException as exc:
            _write_status(
                status_path,
                spec,
                status="failed",
                stage="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _write_status(status_path, spec, status="complete", stage="complete")
    return {
        "status_path": str(status_path),
        "plan_path": str(plan_path),
        "data_gate": gate,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _require_within(path: Path, root: Path, context: str) -> None:
    path = path.resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{context} must live under ClothDojo {root}: {path}")


def _require_within_any(path: Path, roots: tuple[Path, ...], context: str) -> None:
    path = path.resolve()
    resolved_roots = tuple(root.resolve() for root in roots)
    if not any(path == root or root in path.parents for root in resolved_roots):
        expected = ", ".join(str(root) for root in resolved_roots)
        raise ValueError(f"{context} must live under one of [{expected}]: {path}")


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"CFG pipeline is already running: {path}") from exc
        yield


def _write_status(
    path: Path,
    spec: SteamAdvantageSpec,
    *,
    status: str,
    stage: str,
    error: str | None = None,
) -> None:
    value = {
        "schema": "clothdojo_cfg_pipeline_status",
        "format_version": 1,
        "status": status,
        "stage": stage,
        "pid": os.getpid(),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if error is not None:
        value["error"] = error
    _write_json_atomic(path, value)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
