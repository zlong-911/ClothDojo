"""Thin entry points over RLINF's SFT runner."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TrainingKind = Literal["pi05", "steam", "cfg"]


@dataclass(frozen=True)
class _Runtime:
    validate_cfg: Callable[[Any], Any]
    cluster_type: type
    placement_type: type
    runner_type: type
    worker_type: type
    multiprocessing: Any
    omega_conf: Any


def run(
    *,
    kind: TrainingKind,
    config_path: str | Path,
    overrides: Sequence[str] = (),
    steam_member_index: int | None = None,
) -> None:
    """Compose one ClothDojo config and run it through the selected RLINF worker."""

    cfg = compose_config(config_path, overrides)
    run_config(
        kind=kind,
        cfg=cfg,
        steam_member_index=steam_member_index,
    )


def check(
    *,
    kind: TrainingKind,
    config_path: str | Path,
    overrides: Sequence[str] = (),
    steam_member_index: int | None = None,
) -> dict[str, Any]:
    """Compose and validate a config without launching training workers."""

    cfg = compose_config(config_path, overrides)
    if kind == "pi05":
        from clothdojo.training.pi05_preflight import inspect_pi05_training

        return inspect_pi05_training(cfg)
    if kind == "steam":
        if steam_member_index is None:
            raise ValueError("STEAM training requires a member index")
        from clothdojo.training.steam import inspect_steam_training

        return inspect_steam_training(cfg, steam_member_index)
    runtime = _load_runtime(kind)
    cfg = runtime.validate_cfg(cfg)
    resolved = runtime.omega_conf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved RLINF config must be a mapping")
    return resolved


def compose_config(config_path: str | Path, overrides: Sequence[str] = ()) -> Any:
    """Compose a Hydra config rooted at the directory containing ``config_path``."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    if path.suffix != ".yaml":
        raise ValueError(f"Training config must be a .yaml file: {path}")

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(path.parent), version_base="1.1"):
        return compose(config_name=path.stem, overrides=list(overrides))


def run_config(
    *,
    kind: TrainingKind,
    cfg: Any,
    steam_member_index: int | None = None,
) -> None:
    """Run an already composed config without modifying RLINF source code."""

    data_root = cfg.get("data", {}).get("data_root")
    if data_root:
        os.environ["HF_LEROBOT_HOME"] = str(data_root)

    if kind == "pi05":
        from clothdojo.training.pi05_preflight import (
            inspect_pi05_training,
            write_pi05_launch_manifest,
        )

        manifest = inspect_pi05_training(cfg)
        manifest_path = write_pi05_launch_manifest(cfg, manifest)
        logging.info("Pi0.5 launch manifest: %s", manifest_path)
    elif kind == "steam":
        if steam_member_index is None:
            raise ValueError("STEAM training requires a member index")
        from clothdojo.training.steam import (
            inspect_steam_training,
            write_steam_launch_manifest,
        )

        manifest = inspect_steam_training(cfg, steam_member_index)
        manifest_path = write_steam_launch_manifest(cfg, manifest)
        from clothdojo.training.rlinf_steam import prepare_steam_driver_runtime

        prepare_steam_driver_runtime()
        logging.info("STEAM launch manifest: %s", manifest_path)

    # Importing the scheduler initializes Ray on this RLINF revision, so keep
    # the runtime import strictly after the launch gate and manifest write.
    runtime = _load_runtime(kind)
    cfg = runtime.validate_cfg(cfg)
    if kind != "steam":
        runtime.multiprocessing.set_start_method("spawn", force=True)
    logging.info(
        "RLINF %s config:\n%s",
        kind,
        json.dumps(runtime.omega_conf.to_container(cfg, resolve=True), indent=2),
    )

    cluster = runtime.cluster_type(cluster_cfg=cfg.cluster)
    if kind == "steam":
        from clothdojo.training.rlinf_steam import configure_steam_cluster

        configure_steam_cluster(cluster, cfg)
    placement = runtime.placement_type(cfg, cluster)
    actor_group = runtime.worker_type.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )

    runner = runtime.runner_type(cfg=cfg, actor=actor_group)
    runner.init_workers()
    runner.run()


def _load_runtime(kind: TrainingKind) -> _Runtime:
    import torch.multiprocessing as multiprocessing
    from omegaconf import OmegaConf
    from rlinf.config import validate_cfg
    from rlinf.runners.sft_runner import SFTRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    if kind == "pi05":
        from clothdojo.training.rlinf_pi05 import ClothDojoPi05SftWorker

        worker_type = ClothDojoPi05SftWorker
    elif kind == "steam":
        from clothdojo.training.rlinf_steam import ClothDojoSteamSftWorker

        worker_type = ClothDojoSteamSftWorker
    elif kind == "cfg":
        from clothdojo.training.rlinf_cfg import ClothDojoCfgWorker, enable

        enable()
        worker_type = ClothDojoCfgWorker
    else:
        raise ValueError(f"Unknown RLINF training kind: {kind}")

    return _Runtime(
        validate_cfg=validate_cfg,
        cluster_type=Cluster,
        placement_type=HybridComponentPlacement,
        runner_type=SFTRunner,
        worker_type=worker_type,
        multiprocessing=multiprocessing,
        omega_conf=OmegaConf,
    )
