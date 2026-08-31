# Portions derived from RLINF, Copyright 2026 The RLINF Authors,
# licensed under the Apache License, Version 2.0.

"""ClothDojo's RLINF CFG extension.

RLINF's CFG model and worker currently omit ``actor.model.openpi_data`` when
they construct OpenPI's data config.  Piper needs that config to disable the
physical ALOHA axis conversion and to select the externally computed norm
statistics. The upstream CFG model also uses a different checkpoint layout
from RLINF's current ``openpi_rlinf`` implementation. This module keeps the
current project SFT/serving layout while adding CFG routing and mixture data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MODEL_TYPE = "clothdojo_cfg_model"
EXTENSION_MODULE = "clothdojo.training.rlinf_cfg"
_REGISTERED = False
_PROJECT_OPENPI_CACHE = Path(__file__).resolve().parents[3] / "data/caches/openpi"


class _AlignedAdvantageDataset:
    """Restore one prevalidated boolean label without a per-frame Python dict."""

    def __init__(self, transformed_dataset: Any, advantages: Any):
        if len(transformed_dataset) != len(advantages):
            raise ValueError(
                "CFG transformed samples and aligned advantages differ: "
                f"{len(transformed_dataset)} != {len(advantages)}"
            )
        self._dataset = transformed_dataset
        self._advantages = advantages

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._dataset[index]
        sample["advantage"] = bool(self._advantages[index])
        return sample


def _load_aligned_advantages(
    data_path: str | Path,
    *,
    advantage_tag: str | None,
    hf_dataset: Any,
) -> Any:
    """Load and exactly align CFG labels using compact vectorized frame keys."""

    import numpy as np
    import pyarrow.parquet as pq

    from clothdojo.training.cfg import _validate_advantage_columns

    root = Path(data_path)
    filename = f"advantages_{advantage_tag}.parquet" if advantage_tag else "advantages.parquet"
    path = root / "meta" / filename
    if not path.is_file():
        raise FileNotFoundError(f"CFG advantage labels do not exist: {path}")
    table = pq.read_table(
        path,
        columns=["episode_index", "frame_index", "advantage"],
        memory_map=True,
    )
    _validate_advantage_columns(table, path)
    label_episode = np.asarray(
        table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    label_frame = np.asarray(
        table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    labels = np.asarray(
        table["advantage"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.bool_,
    )
    sample_episode = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    sample_frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    for name, values in (
        ("label episode_index", label_episode),
        ("label frame_index", label_frame),
        ("sample episode_index", sample_episode),
        ("sample frame_index", sample_frame),
    ):
        if np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max):
            raise ValueError(f"CFG {name} cannot be encoded as a frame key: {path}")
    label_keys = (label_episode << 32) | label_frame
    sample_keys = (sample_episode << 32) | sample_frame
    if len(label_keys) == len(sample_keys) and np.array_equal(label_keys, sample_keys):
        return labels

    order = np.argsort(label_keys, kind="stable")
    sorted_keys = label_keys[order]
    if len(sorted_keys) > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError(f"CFG advantage labels contain duplicate frame keys: {path}")
    positions = np.searchsorted(sorted_keys, sample_keys)
    in_range = positions < len(sorted_keys)
    matches = np.zeros(len(sample_keys), dtype=np.bool_)
    matches[in_range] = sorted_keys[positions[in_range]] == sample_keys[in_range]
    if not np.all(matches):
        missing = np.flatnonzero(~matches)[:5]
        examples = [(int(sample_episode[index]), int(sample_frame[index])) for index in missing]
        raise ValueError(
            f"CFG samples are missing {int((~matches).sum())} advantage labels "
            f"from {path}; first keys: {examples}"
        )
    return labels[order[positions]]


def _configure_project_openpi_cache() -> Path:
    """Bind OpenPI downloads to ClothDojo's checked local asset cache."""

    cache = (
        Path(os.environ.get("OPENPI_DATA_HOME", str(_PROJECT_OPENPI_CACHE))).expanduser().resolve()
    )
    tokenizer = cache / "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(
            f"CFG requires the project-owned OpenPI tokenizer cache: {tokenizer}"
        )
    os.environ["OPENPI_DATA_HOME"] = str(cache)
    return cache


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
    """Build project CFG on RLINF's current ``openpi_rlinf`` core."""

    from clothdojo.rollout.checkpoint import (
        strict_reload_matching_rlinf_openpi_checkpoint,
    )
    from clothdojo.training.cfg_model import build_current_openpi_cfg_model

    model = build_current_openpi_cfg_model(cfg, torch_dtype=torch_dtype)
    # ``model_path`` remains the canonical checkpoint recorded by the gate and
    # run manifest. Workers may read an identical, hash-checked node-local copy
    # to avoid multiplying shared-filesystem traffic by world size.
    load_checkpoint = cfg.get("load_checkpoint_path", cfg.model_path)
    checkpoint_validation = strict_reload_matching_rlinf_openpi_checkpoint(
        model,
        load_checkpoint,
    )
    expected_weights = cfg.get("expected_weights_sha256")
    if expected_weights is not None and (
        not isinstance(expected_weights, str)
        or checkpoint_validation["weights_sha256"] != expected_weights
    ):
        raise RuntimeError("CFG initialization weights changed after preflight")
    model.clothdojo_checkpoint_validation = checkpoint_validation
    model.clothdojo_checkpoint_validation["canonical_model_path"] = str(
        Path(str(cfg.model_path)).expanduser().resolve()
    )
    model.clothdojo_checkpoint_validation["load_checkpoint_path"] = str(
        Path(str(load_checkpoint)).expanduser().resolve()
    )
    return model


def _worker_type() -> type:
    """Create the worker subclass lazily so ordinary imports stay lightweight."""

    from rlinf.workers.sft.fsdp_cfg_worker import FSDPCfgWorker

    class ClothDojoCfgWorker(FSDPCfgWorker):
        """Upstream CFG worker with OpenPI data kwargs passed through."""

        def set_global_step(self, global_step: int) -> None:
            # Only actual iterator exhaustion advances the data epoch. Inferring
            # it with floor(len(loader) / grad_accum) drops tail micro-batches.
            restored_step = getattr(self, "_restored_step", None)
            if restored_step is not None:
                if global_step != restored_step:
                    raise ValueError("CFG runner step disagrees with restored checkpoint")
                del self._restored_step
            self.global_step = global_step
            if hasattr(self.model, "set_global_step"):
                self.model.set_global_step(global_step)

        def load_checkpoint(self, load_path: str) -> None:
            import torch
            from rlinf.utils.utils import set_rng_state

            from clothdojo.training.cfg_data_runtime import validate_runtime_state

            path = Path(load_path) / f"cfg_runtime_rank_{self._rank}.pt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"CFG exact resume requires per-rank runtime state: {path}. "
                    "Older checkpoints remain usable for inference or initialization."
                )
            state = torch.load(path, map_location="cpu", weights_only=False)
            validate_runtime_state(
                state,
                identity=self._cfg_runtime_identity,
                batches_per_epoch=len(self.data_loader),
            )
            super().load_checkpoint(load_path)
            self._data_epoch = state["epoch"]
            self._current_epoch = state["epoch"]
            self._data_iter_offset = state["consumed_batches"]
            self.data_loader._data_loader.sampler.restore(self._data_epoch, self._data_iter_offset)
            self.data_iter = iter(self.data_loader)
            self._restored_step = state["step"]
            # Upstream DCP also loads RNG, but does not namespace it per rank.
            # Restore our rank's RNG last, after model/optimizer/loader work.
            set_rng_state(state["rng"])

        def save_checkpoint(self, save_path: str, step: int = 0) -> None:
            """Save current-core weights with an explicit CFG format contract."""

            import torch
            from rlinf.utils.utils import get_rng_state

            from clothdojo.training.cfg_data_runtime import validate_runtime_state

            super().save_checkpoint(save_path, step)
            runtime = {
                "identity": self._cfg_runtime_identity,
                "step": int(step),
                "epoch": self._data_epoch,
                "consumed_batches": self._data_iter_offset,
                "rng": get_rng_state(),
            }
            validate_runtime_state(
                runtime,
                identity=self._cfg_runtime_identity,
                batches_per_epoch=len(self.data_loader),
            )
            runtime_path = Path(save_path) / f"cfg_runtime_rank_{self._rank}.pt"
            temporary = runtime_path.with_suffix(".tmp")
            torch.save(runtime, temporary)
            os.replace(temporary, runtime_path)
            if torch.distributed.get_rank() == 0:
                contract = {
                    "schema": "clothdojo_openpi_rlinf_cfg_checkpoint",
                    "format_version": 1,
                    "step": int(step),
                    "weights": {
                        "path": "model_state_dict/full_weights.pt",
                        "purpose": "inference and rollout",
                        "source": "online",
                    },
                    "online": {
                        "path": "dcp_checkpoint",
                        "purpose": "resume with optimizer, scheduler and per-rank runtime state",
                    },
                    "runtime": {"path": "cfg_runtime_rank_{rank}.pt"},
                }
                (Path(save_path) / "clothdojo_weights.json").write_text(
                    json.dumps(contract, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            torch.distributed.barrier()

        def build_dataloader(self):
            import hashlib

            import numpy as np
            from omegaconf import OmegaConf
            from rlinf.utils.utils import seed_everything

            from clothdojo.training.cfg_data_runtime import (
                make_cfg_torch_loader,
                resume_fingerprint,
            )

            # Upstream CFG bypasses the SFT worker's seed initialization.
            seed_everything(int(self.cfg.actor.seed) + self._rank)
            try:
                import lerobot.datasets.lerobot_dataset as lerobot_dataset
            except ModuleNotFoundError:
                import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
            import openpi.training.data_loader as openpi_data_loader
            import openpi.transforms as transforms
            from rlinf.data.datasets.recap.cfg_model import (
                CFGDataLoaderImpl,
                CfgMixtureDataset,
            )
            from rlinf.data.datasets.recap.utils import cast_image_features
            from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            from clothdojo.data.lerobot import open_lerobot_dataset
            from clothdojo.training.cfg import resolve_cfg_episode_selection
            from clothdojo.training.openpi import TrainingPrompt

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

            datasets_with_weights = []
            dataset_fingerprints = []
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
                advantages = _load_aligned_advantages(
                    data_path,
                    advantage_tag=dataset_entry.get("advantage_tag"),
                    hf_dataset=base_dataset.hf_dataset,
                )
                digest = hashlib.sha256()
                # Hash already-loaded alignment/labels once, not at checkpoints.
                # This also detects changed episodes_path contents or relabeling.
                for column in ("episode_index", "frame_index"):
                    digest.update(
                        np.asarray(base_dataset.hf_dataset[column], dtype=np.int64).tobytes()
                    )
                digest.update(np.asarray(advantages, dtype=np.bool_).tobytes())
                digest.update((Path(dataset_root) / "meta/info.json").read_bytes())
                dataset_fingerprints.append(digest.hexdigest())
                if data_config.prompt_from_task:
                    base_dataset = openpi_data_loader.TransformedDataset(
                        base_dataset,
                        [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
                    )
                transformed_dataset = openpi_data_loader.TransformedDataset(
                    base_dataset,
                    [
                        TrainingPrompt(self.cfg.actor.model.openpi_data.default_prompt),
                        *data_config.repack_transforms.inputs,
                        *data_config.data_transforms.inputs,
                        transforms.Normalize(
                            norm_stats, use_quantiles=data_config.use_quantile_norm
                        ),
                        *model_transforms,
                    ],
                )
                final_dataset = _AlignedAdvantageDataset(
                    transformed_dataset,
                    advantages,
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
            torch_loader = make_cfg_torch_loader(
                combined_dataset,
                batch_size=int(self.cfg.actor.micro_batch_size),
                rank=self._rank,
                world_size=self._world_size,
                seed=int(data_cfg.get("seed", 42)),
                num_workers=int(data_cfg.get("num_workers", config.num_workers)),
            )
            manifest_path = Path(str(self.cfg.runner.logger.log_path)) / "cfg_launch_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._cfg_runtime_identity = {
                "rank": self._rank,
                "world_size": self._world_size,
                "micro_batch_size": int(self.cfg.actor.micro_batch_size),
                "gradient_accumulation": self.gradient_accumulation,
                "fingerprint": resume_fingerprint(
                    {
                        "data": OmegaConf.to_container(self.cfg.data, resolve=True),
                        "actor": OmegaConf.to_container(self.cfg.actor, resolve=True),
                        "datasets": dataset_fingerprints,
                        "norm_stats": manifest["norm_stats"],
                    }
                ),
            }
            data_loader = CFGDataLoaderImpl(data_config, torch_loader)
            return data_loader, data_loader.data_config()

    return ClothDojoCfgWorker


ClothDojoCfgWorker = _worker_type()
