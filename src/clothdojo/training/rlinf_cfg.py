# Portions derived from RLINF, Copyright 2026 The RLINF Authors,
# licensed under the Apache License, Version 2.0.

"""ClothDojo's narrow RLINF CFG extension.

RLINF's CFG model and worker currently omit ``actor.model.openpi_data`` when
they construct OpenPI's data config.  Piper needs that config to disable the
physical ALOHA axis conversion and to select the externally computed norm
statistics.  This module changes only that argument plumbing; model, dataset,
sampler, optimizer and checkpoint behavior remain upstream RLINF behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MODEL_TYPE = "clothdojo_cfg_model"
EXTENSION_MODULE = "clothdojo.training.rlinf_cfg"
_REGISTERED = False
_PROJECT_OPENPI_CACHE = Path(__file__).resolve().parents[3] / "data/caches/openpi"


def _configure_project_openpi_cache() -> Path:
    """Bind OpenPI downloads to ClothDojo's checked local asset cache."""

    tokenizer = _PROJECT_OPENPI_CACHE / "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(
            "CFG requires the project-owned OpenPI tokenizer cache: "
            f"{tokenizer}"
        )
    os.environ["OPENPI_DATA_HOME"] = str(_PROJECT_OPENPI_CACHE)
    return _PROJECT_OPENPI_CACHE


def register() -> None:
    """Register the project CFG model in each RLINF process."""

    global _REGISTERED
    if _REGISTERED:
        return
    from rlinf.models import register_model

    register_model(MODEL_TYPE, _build_cfg_model, category="embodied")
    _REGISTERED = True


def enable() -> None:
    """Enable the extension in the driver and future RLINF workers."""

    _configure_project_openpi_cache()
    configured = os.environ.get("RLINF_EXT_MODULE")
    if configured not in {None, EXTENSION_MODULE}:
        raise RuntimeError(
            f"CFG requires RLINF_EXT_MODULE={EXTENSION_MODULE!r}, but it is already {configured!r}"
        )
    os.environ["RLINF_EXT_MODULE"] = EXTENSION_MODULE
    register()


def _openpi_config_kwargs(
    cfg: Any,
    *,
    batch_size: int | None = None,
    repo_id: str | None = None,
) -> dict[str, Any]:
    """Build the shared OpenPI config arguments used by model and loader."""

    kwargs: dict[str, Any] = {
        "config_name": cfg.openpi.config_name,
        "model_path": cfg.model_path,
        "data_kwargs": cfg.get("openpi_data"),
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if repo_id is not None:
        kwargs["repo_id"] = repo_id
    return kwargs


def _build_cfg_model(cfg: Any, torch_dtype: Any = None) -> Any:
    """Build upstream's CFG model with ClothDojo's OpenPI data arguments."""

    del torch_dtype
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as checkpoints
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
        OpenPi0Config,
        OpenPi0ForCFGActionPrediction,
    )

    from clothdojo.rollout.checkpoint import (
        strict_reload_matching_rlinf_openpi_checkpoint,
    )

    train_config = get_openpi_config(**_openpi_config_kwargs(cfg))
    model_config = OpenPi0Config(**train_config.model.__dict__)
    for key, value in cfg.openpi.items():
        model_config.__dict__[key] = value

    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    model = OpenPi0ForCFGActionPrediction(model_config)
    if model_config.train_expert_only:
        model.freeze_vlm()

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    checkpoint_validation = strict_reload_matching_rlinf_openpi_checkpoint(
        model,
        checkpoint_dir,
    )
    expected_weights = cfg.get("expected_weights_sha256")
    if expected_weights is not None and (
        not isinstance(expected_weights, str)
        or checkpoint_validation["weights_sha256"] != expected_weights
    ):
        raise RuntimeError("CFG initialization weights changed after preflight")
    model.clothdojo_checkpoint_validation = checkpoint_validation
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    norm_stats = data_config.norm_stats
    if norm_stats is None:
        data_kwargs = cfg.get("openpi_data") or {}
        norm_path = data_kwargs.get("norm_stats_path")
        if norm_path is None:
            if data_config.asset_id is None:
                raise ValueError("CFG requires an OpenPI asset_id or norm_stats_path")
            norm_stats = checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
        else:
            norm_dir = Path(norm_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)

    model.setup_wrappers(
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
    )
    return model


def _worker_type() -> type:
    """Create the worker subclass lazily so ordinary imports stay lightweight."""

    from rlinf.workers.sft.fsdp_cfg_worker import FSDPCfgWorker

    class ClothDojoCfgWorker(FSDPCfgWorker):
        """Upstream CFG worker with OpenPI data kwargs passed through."""

        def build_dataloader(self):
            try:
                import lerobot.datasets.lerobot_dataset as lerobot_dataset
            except ModuleNotFoundError:
                import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
            import openpi.training.data_loader as openpi_data_loader
            import openpi.transforms as transforms
            from rlinf.data.datasets.recap.cfg_model import (
                AdvantagePreservingDataset,
                CFGDataLoaderImpl,
                CfgMixtureDataset,
            )
            from rlinf.data.datasets.recap.utils import cast_image_features
            from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            from clothdojo.data.lerobot import open_lerobot_dataset
            from clothdojo.training.cfg import resolve_cfg_episode_selection

            data_cfg = self.cfg.get("data", {})
            datasets_config = data_cfg.get("train_data_paths", [])
            if not datasets_config:
                raise ValueError("data.train_data_paths must contain a CFG dataset")

            first_path = datasets_config[0]["dataset_path"]
            config = get_openpi_config(
                **_openpi_config_kwargs(
                    self.cfg.actor.model,
                    batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                    repo_id=first_path,
                )
            )
            data_config = config.data.create(config.assets_dirs, config.model)
            model_transforms = self._build_model_transforms(data_config)
            norm_stats = data_config.norm_stats or {}
            advantage_tag = data_cfg.get("advantage_tag")

            datasets_with_weights = []
            for dataset_entry in datasets_config:
                data_path = dataset_entry["dataset_path"]
                dataset_root = resolve_lerobot_dataset_root(data_path)
                weight = dataset_entry.get("weight", 1.0)
                dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_path, root=dataset_root)
                episodes_array = resolve_cfg_episode_selection(
                    episodes=dataset_entry.get("episodes"),
                    episodes_path=dataset_entry.get("episodes_path"),
                    total_episodes=int(dataset_meta.total_episodes),
                )
                episodes = [int(value) for value in episodes_array]
                base_dataset = open_lerobot_dataset(
                    data_path,
                    root=dataset_root,
                    episodes=episodes,
                    delta_timestamps={
                        key: [
                            step / dataset_meta.fps for step in range(config.model.action_horizon)
                        ]
                        for key in data_config.action_sequence_keys
                    },
                )
                base_dataset.hf_dataset = cast_image_features(base_dataset.hf_dataset)
                if episodes is not None:
                    self._fix_episode_data_index(base_dataset, episodes)
                if data_config.prompt_from_task:
                    base_dataset = openpi_data_loader.TransformedDataset(
                        base_dataset,
                        [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
                    )
                transformed_dataset = openpi_data_loader.TransformedDataset(
                    base_dataset,
                    [
                        *data_config.repack_transforms.inputs,
                        *data_config.data_transforms.inputs,
                        transforms.Normalize(
                            norm_stats, use_quantiles=data_config.use_quantile_norm
                        ),
                        *model_transforms,
                    ],
                )
                advantages = self._load_advantages_lookup(data_path, advantage_tag)
                final_dataset = AdvantagePreservingDataset(
                    base_dataset=base_dataset,
                    transformed_dataset=transformed_dataset,
                    advantages_lookup=advantages,
                )
                datasets_with_weights.append((final_dataset, weight))
                if self._rank == 0:
                    self.log_info(
                        f"Loaded CFG dataset: {data_path} "
                        f"({len(final_dataset)} samples, weight={weight})"
                    )

            combined_dataset = CfgMixtureDataset(
                datasets=datasets_with_weights,
                mode="train",
                balance_dataset_weights=data_cfg.get("balance_dataset_weights", True),
                seed=data_cfg.get("seed", 42),
            )
            torch_loader = self._create_torch_dataloader(
                combined_dataset, config, openpi_data_loader
            )
            data_loader = CFGDataLoaderImpl(data_config, torch_loader)
            return data_loader, data_loader.data_config()

    return ClothDojoCfgWorker


ClothDojoCfgWorker = _worker_type()
