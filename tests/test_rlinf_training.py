from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from clothdojo.training import rlinf
from clothdojo.training.openpi_rlinf_sft import (
    _action_offset_loss_manifest_from_config,
    _action_offset_weights_from_config,
    _adapt_official_openpi_loader,
    _EpochAdvancingSampler,
    _full_only_anchor_indices,
    _initialize_ema_state,
    _reduce_action_offset_loss,
    _swap_parameter_data_with_ema,
    _update_ema_state,
    _validate_fp32_training_state,
)
from clothdojo.training.steam import configure_steam_member
from clothdojo.training.training_time_rtc import training_time_rtc_manifest_from_config
from clothdojo.training.worker_environment import apply_worker_environment


def test_run_config_delegates_to_rlinf_runner(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        rlinf,
        "inspect_openpi_rlinf_sft",
        lambda cfg, **_kwargs: events.append(("preflight", cfg))
        or {"compatible": True, "openpi_data_home": "/openpi-cache"},
    )
    monkeypatch.setattr(
        rlinf,
        "write_openpi_rlinf_launch_manifest",
        lambda cfg, manifest: events.append(("manifest", cfg, manifest))
        or Path("/output/clothdojo_launch.json"),
    )

    class Multiprocessing:
        @staticmethod
        def set_start_method(method, *, force):
            events.append(("start_method", method, force))

    class OmegaConf:
        @staticmethod
        def to_container(cfg, *, resolve):
            assert resolve is True
            return {"resolved": True}

    class Cluster:
        def __init__(self, *, cluster_cfg):
            events.append(("cluster", cluster_cfg))

    class Placement:
        def __init__(self, cfg, cluster):
            events.append(("placement", cfg, cluster))

        def get_strategy(self, component):
            events.append(("strategy", component))
            return "actor-placement"

    class Worker:
        @classmethod
        def create_group(cls, cfg):
            events.append(("create_group", cfg))
            return cls()

        def launch(self, cluster, *, name, placement_strategy):
            events.append(("launch", cluster, name, placement_strategy))
            return "actor-group"

    class Runner:
        def __init__(self, *, cfg, actor):
            events.append(("runner", cfg, actor))

        def init_workers(self):
            events.append(("init_workers",))

        def run(self):
            events.append(("run",))

    runtime = rlinf._Runtime(
        validate_cfg=lambda cfg: cfg,
        cluster_type=Cluster,
        placement_type=Placement,
        runner_type=Runner,
        worker_type=Worker,
        multiprocessing=Multiprocessing,
        omega_conf=OmegaConf,
    )
    monkeypatch.setattr(rlinf, "_load_runtime", lambda kind: runtime)
    cfg = SimpleNamespace(
        cluster="cluster-config",
        actor=SimpleNamespace(group_name="ActorGroup"),
        get=lambda key, default: default,
    )

    rlinf.run_config(kind="openpi-rlinf", cfg=cfg)

    assert events[0] == ("preflight", cfg)
    assert os.environ["OPENPI_DATA_HOME"] == "/openpi-cache"
    assert events[1] == (
        "manifest",
        cfg,
        {"compatible": True, "openpi_data_home": "/openpi-cache"},
    )
    assert events[2] == ("start_method", "spawn", True)
    assert ("strategy", "actor") in events
    assert ("runner", cfg, "actor-group") in events
    assert events[-2:] == [("init_workers",), ("run",)]

def test_openpi_rlinf_runtime_uses_current_worker() -> None:
    from clothdojo.training.openpi_rlinf_sft import OpenPiRlinfSftWorker

    assert rlinf._load_runtime("openpi-rlinf").worker_type is OpenPiRlinfSftWorker


def test_worker_environment_overrides_ray_actor_defaults(monkeypatch) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "clothdojo": {
                "worker_environment": {
                    "GLOO_SOCKET_IFNAME": "br0",
                    "NCCL_SOCKET_IFNAME": "br0",
                    "NCCL_IB_DISABLE": "0",
                    "NCCL_IB_GID_INDEX": None,
                }
            }
        }
    )
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth1")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth1")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    monkeypatch.setenv("NCCL_IB_GID_INDEX", "7")

    apply_worker_environment(cfg)

    assert os.environ["GLOO_SOCKET_IFNAME"] == "br0"
    assert os.environ["NCCL_SOCKET_IFNAME"] == "br0"
    assert os.environ["NCCL_IB_DISABLE"] == "0"
    assert "NCCL_IB_GID_INDEX" not in os.environ


def test_openpi_rlinf_check_does_not_import_scheduler_runtime(monkeypatch) -> None:
    cfg = object()
    monkeypatch.setattr(rlinf, "compose_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        rlinf,
        "_load_runtime",
        lambda kind: pytest.fail("openpi-rlinf --check must not initialize Ray"),
    )
    monkeypatch.setattr(
        rlinf,
        "inspect_openpi_rlinf_sft",
        lambda value, **_kwargs: {"compatible": value is cfg},
    )

    assert rlinf.check(kind="openpi-rlinf", config_path="unused.yaml") == {
        "compatible": True
    }


def test_steam_check_uses_project_preflight_without_rlinf_runtime(monkeypatch) -> None:
    cfg = SimpleNamespace(actor=SimpleNamespace(group_name="SteamMember"))
    monkeypatch.setattr(rlinf, "compose_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        rlinf,
        "_load_runtime",
        lambda kind: pytest.fail("steam --check must not initialize Ray"),
    )
    monkeypatch.setattr(
        "clothdojo.training.steam.inspect_steam_training",
        lambda value, member_index: {
            "compatible": value is cfg,
            "member_index": member_index,
        },
    )

    assert rlinf.check(
        kind="steam",
        config_path="unused.yaml",
        steam_member_index=2,
    ) == {"compatible": True, "member_index": 2}


def test_configure_steam_member_owns_names_and_seeds(monkeypatch) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"actor": {"group_name": "SteamMember"}})
    result = configure_steam_member(cfg, 1)

    assert result == {
        "member_index": 1,
        "head_seed": 101,
        "run_seed": 20360826,
    }
    assert cfg.actor.group_name == "SteamMember1"
    assert os.environ["CLOTHDOJO_STEAM_MEMBER_INDEX"] == "1"
    assert os.environ["CLOTHDOJO_STEAM_HEAD_SEED"] == "101"
    assert os.environ["CLOTHDOJO_STEAM_RUN_SEED"] == "20360826"


def test_steam_shirt_flatten_config_is_self_contained(monkeypatch, tmp_path) -> None:
    for name, value in {
        "CLOTHDOJO_STEAM_PLACEMENT": "0-7",
        "CLOTHDOJO_STEAM_OUTPUT": tmp_path / "output",
        "CLOTHDOJO_STEAM_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_STEAM_SIGLIP_CHECKPOINT": tmp_path / "siglip",
        "CLOTHDOJO_STEAM_GEMMA3_CHECKPOINT": tmp_path / "gemma",
    }.items():
        monkeypatch.setenv(name, str(value))

    project_root = Path(__file__).resolve().parents[1]
    cfg = rlinf.compose_config(
        project_root / "configs/training/steam_value_shirt_flatten.yaml"
    )

    assert cfg.cluster.num_nodes == 4
    assert cfg.cluster.component_placement["actor,env,rollout"] == "0-7"
    assert cfg.runner.max_steps == 30000
    assert cfg.data.train_data_paths[0].dataset_path == str(tmp_path / "dataset")
    assert cfg.data.train_num_workers == 8
    assert cfg.actor.micro_batch_size == 32
    assert cfg.actor.global_batch_size == 512
    assert cfg.actor.model.model_type == "steam_value_model"
    assert cfg.actor.model.precision == "fp32"
    assert cfg.actor.model.include_state is True
    assert cfg.actor.model.freeze_vision_encoder is False
    assert cfg.actor.model.freeze_language_model is False
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype == "bf16"


def test_openpi_rlinf_config_preserves_training_contract(monkeypatch, tmp_path) -> None:
    for name, value in {
        "CLOTHDOJO_LEROBOT_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_PI05_CHECKPOINT": tmp_path / "checkpoint",
        "CLOTHDOJO_PI05_NORM_STATS": tmp_path / "norm-stats",
        "CLOTHDOJO_TRAIN_OUTPUT": tmp_path / "outputs",
        "CLOTHDOJO_OPENPI_CACHE": tmp_path / "openpi-cache",
    }.items():
        monkeypatch.setenv(name, str(value))

    project_root = Path(__file__).resolve().parents[1]
    cfg = rlinf.compose_config(
        project_root / "configs/training/pi05_aloha_openpi_rlinf_lr2x_60k.yaml"
    )

    assert cfg.runner.max_steps == 60000
    assert cfg.runner.max_epochs == -1
    assert cfg.runner.save_interval == 10000
    assert cfg.cluster.num_nodes == 1
    assert cfg.cluster.component_placement.actor == "0-7"
    assert cfg.data.train_data_paths == str(tmp_path / "dataset")
    assert cfg.actor.global_batch_size == 64
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.seed == 43
    assert cfg.actor.optim.total_training_steps == 60000
    assert cfg.actor.optim.lr_scheduler == "openpi_cosine"
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.min_lr == 5.0e-6
    assert cfg.actor.optim.ema_decay == 0.99
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.actor.fsdp_config.save_full_model_weights is False
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.actor.model.model_type == "openpi_rlinf"
    assert cfg.actor.model.openpi.config_name == "pi05_aloha_robotwin"
    assert cfg.actor.model.openpi_data.adapt_to_pi is False
    assert cfg.actor.model.openpi_data.extra_delta_transform is True
    assert cfg.actor.model.openpi_data.norm_stats_path == str(tmp_path / "norm-stats")
    assert cfg.clothdojo.jax_reference is None
    assert cfg.clothdojo.image_augmentation is False
    assert cfg.clothdojo.action_chunk_boundary == "full_only"
    assert cfg.actor.model.openpi_data.default_prompt == "flatten the shirt"

    from omegaconf import OmegaConf

    base = OmegaConf.to_container(
        rlinf.compose_config(
            project_root / "configs/training/pi05_aloha_openpi_rlinf_bs128_30k.yaml"
        ), resolve=True,
    )
    noaug = OmegaConf.to_container(
        rlinf.compose_config(
            project_root / "configs/training/pi05_aloha_openpi_rlinf_noaug_bs128_30k.yaml"
        ), resolve=True,
    )
    assert noaug["clothdojo"]["image_augmentation"] is False
    noaug["runner"]["logger"]["experiment_name"] = base["runner"]["logger"]["experiment_name"]
    assert noaug == base


def test_openpi_rlinf_constant_tail_preserves_cosine_prefix() -> None:
    from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        project_root
        / "configs/training/pi05_aloha_openpi_rlinf_mirrored_noaug_bs256_wd001_40k.yaml"
    )
    cfg = rlinf.compose_config(config_path)
    baseline = rlinf.compose_config(config_path, ["runner.max_steps=30000"])

    def trace(config):
        optim = config.actor.optim
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=optim.lr)
        scheduler = get_lr_scheduler(
            lr_scheduler=optim.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=optim.lr_warmup_steps,
            num_training_steps=optim.total_training_steps,
            num_cycles=optim.num_cycles,
            min_lr=optim.min_lr,
        )
        rates = []
        for _ in range(config.runner.max_steps):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return rates

    rates = trace(cfg)
    assert len(rates) == 40000
    assert rates[:30000] == trace(baseline)
    assert rates[1000] == pytest.approx(5.0e-5)
    assert rates[29999] > 5.0e-6
    assert rates[30000:] == pytest.approx([5.0e-6] * 10000, rel=1e-12, abs=0)


def test_openpi_rlinf_execution_focus_changes_only_loss_reduction(monkeypatch, tmp_path) -> None:
    for name, value in {
        "CLOTHDOJO_LEROBOT_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_PI05_CHECKPOINT": tmp_path / "checkpoint",
        "CLOTHDOJO_PI05_NORM_STATS": tmp_path / "norm-stats",
        "CLOTHDOJO_TRAIN_OUTPUT": tmp_path / "outputs",
        "CLOTHDOJO_OPENPI_CACHE": tmp_path / "openpi-cache",
    }.items():
        monkeypatch.setenv(name, str(value))

    project_root = Path(__file__).resolve().parents[1]
    baseline = rlinf.compose_config(
        project_root / "configs/training/pi05_aloha_openpi_rlinf_lr2x_60k.yaml"
    )
    focused = rlinf.compose_config(
        project_root / "configs/training/pi05_aloha_openpi_rlinf_lr2x_60k_execution_focus.yaml"
    )

    assert baseline.actor.seed == focused.actor.seed == 43
    assert baseline.runner.max_steps == focused.runner.max_steps == 60000
    assert baseline.actor.optim == focused.actor.optim
    assert _action_offset_weights_from_config(baseline) == (1.0,) * 50
    assert _action_offset_weights_from_config(focused) == ((1.0,) * 10 + (0.5,) * 40)
    manifest = _action_offset_loss_manifest_from_config(focused)
    assert manifest["profile"] == "fixed_delay_rtc"
    assert manifest["reduction"] == "batch_mean_of_weight_normalized_action_offset_mean"


def test_openpi_rlinf_training_time_rtc_recipe_keeps_global_batch(monkeypatch, tmp_path) -> None:
    for name, value in {
        "CLOTHDOJO_LEROBOT_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_PI05_CHECKPOINT": tmp_path / "checkpoint",
        "CLOTHDOJO_PI05_NORM_STATS": tmp_path / "norm-stats",
        "CLOTHDOJO_TRAIN_OUTPUT": tmp_path / "outputs",
        "CLOTHDOJO_OPENPI_CACHE": tmp_path / "openpi-cache",
    }.items():
        monkeypatch.setenv(name, str(value))
    project_root = Path(__file__).resolve().parents[1]
    cfg = rlinf.compose_config(
        project_root
        / "configs/training/"
        "pi05_aloha_openpi_rlinf_mirrored_training_time_rtc_fullonly_wd001_40k.yaml"
    )
    assert cfg.cluster.num_nodes == 2
    assert cfg.actor.micro_batch_size == 16
    assert cfg.actor.global_batch_size == 256
    assert training_time_rtc_manifest_from_config(cfg)["delay_support"] == list(range(10))


def test_openpi_rlinf_action_offset_reduction_matches_manual_weighting() -> None:
    per_offset = torch.tensor([[1.0, 3.0, 9.0], [2.0, 4.0, 8.0]])

    loss = _reduce_action_offset_loss(per_offset, (1.0, 1.0, 0.5))

    expected = torch.tensor(
        [
            (1.0 + 3.0 + 0.5 * 9.0) / 2.5,
            (2.0 + 4.0 + 0.5 * 8.0) / 2.5,
        ]
    ).mean()
    torch.testing.assert_close(loss, expected)


def test_openpi_rlinf_contract_is_derived_from_the_named_jax_trial(
    monkeypatch, tmp_path
) -> None:
    checkpoint = tmp_path / "pytorch-base"
    for name, value in {
        "CLOTHDOJO_LEROBOT_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_PI05_CHECKPOINT": checkpoint,
        "CLOTHDOJO_PI05_NORM_STATS": tmp_path / "norm-stats",
        "CLOTHDOJO_TRAIN_OUTPUT": tmp_path / "outputs",
        "CLOTHDOJO_OPENPI_CACHE": tmp_path / "openpi-cache",
        "CLOTHDOJO_PI05_JAX_PARAMS": tmp_path / "jax-params",
        "CLOTHDOJO_PI05_PYTORCH_BASE": checkpoint,
        "CLOTHDOJO_JAX_CHECKPOINT_BASE": tmp_path / "jax-checkpoints",
        "CLOTHDOJO_JAX_GATE_DIR": tmp_path / "jax-gates",
        "CLOTHDOJO_JAX_MATRIX_DIR": tmp_path / "jax-matrix",
    }.items():
        monkeypatch.setenv(name, str(value))

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs/training/pi05_aloha_openpi_rlinf_lr2x_60k.yaml"
    cfg = rlinf.compose_config(config_path)
    # Historical framework comparisons explicitly retain their original inputs.
    cfg.clothdojo.jax_reference = {
        "matrix": "pi05_aloha_jax_lr2x_60k_seed43.yaml",
        "trial": "decay60k_lr2x_seed43",
    }
    cfg.actor.model.openpi_data.default_prompt = "Flatten the shirt."
    contract = {
        "seed": 43,
        "global_batch_size": 64,
        "steps": 60_000,
        "peak_lr": 5.0e-5,
        "warmup_steps": 1_000,
        "decay_steps": 60_000,
        "floor_lr": 5.0e-6,
        "adam_b1": 0.9,
        "adam_b2": 0.95,
        "adam_eps": 1.0e-8,
        "weight_decay": 1.0e-10,
        "clip_gradient_norm": 1.0,
        "ema_decay": 0.99,
    }

    reference = rlinf._inspect_jax_reference(cfg, config_path, contract)

    assert reference["trial"] == "decay60k_lr2x_seed43"
    assert reference["training_contract"] == contract
    with pytest.raises(ValueError, match="differs from its JAX reference"):
        rlinf._inspect_jax_reference(cfg, config_path, {**contract, "seed": 42})

    from omegaconf import OmegaConf

    from clothdojo.training import jax_sft

    OmegaConf.update(
        cfg,
        "loss.action_offset",
        {
            "profile": "fixed_delay_rtc",
            "focus": 0.5,
            "exec_horizon": 10,
            "delay_steps": 0,
        },
        force_add=True,
    )
    with pytest.raises(ValueError, match="action-offset loss differs"):
        rlinf._inspect_jax_reference(cfg, config_path, contract)

    compose_reference = jax_sft.compose_jax_sft_spec
    sampling = jax_sft.DataSamplingSpec("frame", None, None)

    def focused_reference(*args, **kwargs):
        return dataclasses.replace(
            compose_reference(*args, **kwargs),
            action_offset_loss=jax_sft.ActionOffsetLossSpec("fixed_delay_rtc", 0.5, 10, 0),
            data_sampling=sampling,
        )

    monkeypatch.setattr(jax_sft, "compose_jax_sft_spec", focused_reference)
    assert rlinf._inspect_jax_reference(cfg, config_path, contract) is not None
    sampling = jax_sft.DataSamplingSpec("asset_tempered", 0.5, tmp_path / "sampling.json")
    with pytest.raises(ValueError, match="only supports frame sampling"):
        rlinf._inspect_jax_reference(cfg, config_path, contract)


@pytest.mark.parametrize("resume_setting", [{}, {"resume_dir": None}])
def test_openpi_rlinf_preflight_reuses_shared_artifact_contract(
    monkeypatch, tmp_path, resume_setting
) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "runner": {
                "max_steps": 60000,
                "save_interval": 10000,
                "logger": {"log_path": str(tmp_path / "run")},
                **resume_setting,
            },
            "data": {
                "train_data_paths": "/dataset",
                "openpi_data_home": str(tmp_path / "openpi-cache"),
            },
            "actor": {
                "seed": 43,
                "global_batch_size": 64,
                "micro_batch_size": 8,
                "optim": {
                    "lr": 5.0e-5,
                    "lr_warmup_steps": 1000,
                    "total_training_steps": 60000,
                    "min_lr": 5.0e-6,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.95,
                    "adam_eps": 1.0e-8,
                    "weight_decay": 1.0e-10,
                    "clip_grad": 1.0,
                    "ema_decay": 0.99,
                    "lr_scheduler": "openpi_cosine",
                    "num_cycles": 0.5,
                },
                "model": {
                    "model_type": "openpi_rlinf",
                    "precision": "fp32",
                    "num_action_chunks": 50,
                    "openpi": {"config_name": "pi05_aloha_robotwin"},
                    "openpi_data": {
                        "adapt_to_pi": False,
                        "extra_delta_transform": True,
                        "default_prompt": "Flatten the shirt.",
                        "norm_stats_path": "/norm",
                    },
                },
                "fsdp_config": {
                    "save_full_model_weights": False,
                    "mixed_precision": {
                        "param_dtype": "bf16",
                        "reduce_dtype": "fp32",
                        "buffer_dtype": "fp32",
                    }
                },
            },
        }
    )
    tokenizer = tmp_path / "openpi-cache/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer")
    calls = []
    monkeypatch.setattr(
        "clothdojo.training.openpi_artifacts.inspect_openpi_dataset",
        lambda path, **kwargs: calls.append(("dataset", path, kwargs))
        or {"root": path},
    )
    monkeypatch.setattr(
        "clothdojo.training.openpi_artifacts.inspect_openpi_norm_stats",
        lambda path, **kwargs: calls.append(("norm", path, kwargs)) or {"root": path},
    )

    result = rlinf.inspect_openpi_rlinf_sft(cfg)

    assert result["schema"] == "clothdojo_pi05_launch_preflight"
    assert result["format_version"] == 1
    assert result["compatible"] is True
    assert result["image_augmentation"] is False
    assert result["action_chunk_boundary"] == "repeat_last"
    assert result["training_prompt"] == "Flatten the shirt."
    assert calls[0] == (
        "dataset",
        "/dataset",
        {"expected_task": "Flatten the shirt."},
    )
    assert calls[1][0:2] == ("norm", "/norm")
    assert calls[1][2]["data_kwargs"] == {
        "adapt_to_pi": False,
        "extra_delta_transform": True,
        "default_prompt": "Flatten the shirt.",
    }
    assert calls[1][2]["action_chunk_boundary"] == "repeat_last"

    with monkeypatch.context() as patch:
        patch.setattr(rlinf, "_inspect_jax_reference", lambda *_args: {"trial": "reference"})
        source_prompt = "flatten the shirt"
        patch.setattr(
            "clothdojo.training.openpi_artifacts.inspect_openpi_dataset",
            lambda path, **_kwargs: {"root": path, "prompt": source_prompt},
        )
        with pytest.raises(ValueError, match="training prompt differs from its JAX dataset prompt"):
            rlinf.inspect_openpi_rlinf_sft(cfg)
        source_prompt = "Flatten the shirt."
        assert rlinf.inspect_openpi_rlinf_sft(cfg)["training_prompt"] == source_prompt

    cfg.runner.max_steps = 65000
    result = rlinf.inspect_openpi_rlinf_sft(cfg)
    assert result["training_contract"]["steps"] == 65000
    assert result["training_contract"]["decay_steps"] == 60000

    cfg.runner.max_steps = 59000
    with pytest.raises(ValueError, match="warmup_steps < decay_steps <= steps"):
        rlinf.inspect_openpi_rlinf_sft(cfg)
    cfg.runner.max_steps = 60000
    cfg.actor.optim.lr_warmup_steps = 60000
    with pytest.raises(ValueError, match="warmup_steps < decay_steps <= steps"):
        rlinf.inspect_openpi_rlinf_sft(cfg)
    cfg.actor.optim.lr_warmup_steps = 1000

    cfg.clothdojo = {"image_augmentation": False}
    result = rlinf.inspect_openpi_rlinf_sft(cfg)
    assert result["image_augmentation"] is False
    assert result["resolved_config"]["clothdojo"]["image_augmentation"] is False
    cfg.clothdojo.image_augmentation = "false"
    with pytest.raises(TypeError, match="image_augmentation must be a boolean"):
        rlinf.inspect_openpi_rlinf_sft(cfg)


def test_openpi_rlinf_launch_manifest_is_atomic_and_immutable(tmp_path) -> None:
    output = tmp_path / "run"
    cfg = SimpleNamespace(
        runner=SimpleNamespace(logger=SimpleNamespace(log_path=str(output)))
    )
    manifest = {
        "schema": "clothdojo_pi05_launch_preflight",
        "format_version": 1,
        "compatible": True,
    }

    path = rlinf.write_openpi_rlinf_launch_manifest(cfg, manifest)

    assert path == output / "clothdojo_launch.json"
    assert path.read_text() == (
        '{\n  "compatible": true,\n  "format_version": 1,\n'
        '  "schema": "clothdojo_pi05_launch_preflight"\n}\n'
    )
    original = path.read_bytes()
    for repeat in (manifest, {**manifest, "compatible": False}):
        with pytest.raises(FileExistsError, match="new output directory"):
            rlinf.write_openpi_rlinf_launch_manifest(cfg, repeat)
        assert path.read_bytes() == original


@pytest.mark.parametrize("entry", [rlinf.check, rlinf.run])
@pytest.mark.parametrize("resume_dir", ["/old/checkpoints/global_step_10000", ""])
def test_sft_rejects_resume_before_launch(monkeypatch, tmp_path, entry, resume_dir):
    from omegaconf import OmegaConf

    output = tmp_path / "run"
    cfg = OmegaConf.create(
        {"runner": {"resume_dir": resume_dir, "logger": {"log_path": str(output)}}}
    )
    monkeypatch.setattr(rlinf, "compose_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        rlinf, "_load_runtime", lambda kind: pytest.fail("must reject before initializing Ray")
    )

    with pytest.raises(ValueError, match="SFT resume is disabled"):
        entry(kind="openpi-rlinf", config_path="unused.yaml")
    assert not output.exists()


@pytest.mark.parametrize("has_checkpoint", [False, True])
def test_sft_rejects_existing_output_even_without_launch_manifest(tmp_path, has_checkpoint):
    from omegaconf import OmegaConf

    output = tmp_path / "run"
    output.mkdir()
    checkpoint = output / "legacy" / "checkpoints" / "global_step_10000" / "weights.pt"
    if has_checkpoint:
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"old weights")
    cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(output)}}})

    with pytest.raises(FileExistsError, match="new output directory"):
        rlinf.inspect_openpi_rlinf_sft(cfg)
    with pytest.raises(FileExistsError, match="new output directory"):
        rlinf.write_openpi_rlinf_launch_manifest(cfg, {})
    assert not (output / "clothdojo_launch.json").exists()
    if has_checkpoint:
        assert checkpoint.read_bytes() == b"old weights"


def test_sft_worker_rejects_direct_checkpoint_restore():
    from clothdojo.training.openpi_rlinf_sft import OpenPiRlinfSftWorker

    with pytest.raises(ValueError, match="SFT resume is disabled"):
        OpenPiRlinfSftWorker.load_checkpoint(None, "/old/checkpoints/global_step_10000/actor")


def test_openpi_loader_adapter_uses_actor_seed_and_advances_epoch() -> None:
    dataset = TensorDataset(torch.arange(24))
    sampler = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    torch_loader = DataLoader(dataset, batch_size=3, sampler=sampler)
    data_loader = SimpleNamespace(
        _data_loader=SimpleNamespace(_data_loader=torch_loader)
    )

    _adapt_official_openpi_loader(data_loader, seed=42, rank=1)

    adapted = torch_loader.batch_sampler.sampler
    assert isinstance(adapted, _EpochAdvancingSampler)
    assert sampler.seed == 42
    epoch_zero = [int(value) for batch in torch_loader for value in batch[0]]
    epoch_one = [int(value) for batch in torch_loader for value in batch[0]]
    reference_zero = DistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=42, drop_last=True
    )
    reference_one = DistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=42, drop_last=True
    )
    reference_one.set_epoch(1)
    assert epoch_zero == list(reference_zero)
    assert epoch_one == list(reference_one)
    assert epoch_zero != epoch_one


def test_openpi_loader_full_only_excludes_incomplete_episode_tails() -> None:
    from openpi.training.data_loader import TransformedDataset

    raw_dataset = TensorDataset(torch.arange(12))
    raw_dataset.episode_data_index = {
        "from": torch.tensor([0, 7]),
        "to": torch.tensor([7, 12]),
    }
    dataset = TransformedDataset(raw_dataset, [])
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=False,
        drop_last=True,
    )
    torch_loader = DataLoader(dataset, batch_size=2, sampler=sampler)
    data_loader = SimpleNamespace(
        _data_loader=SimpleNamespace(_data_loader=torch_loader)
    )

    _adapt_official_openpi_loader(
        data_loader,
        seed=42,
        rank=0,
        action_horizon=3,
        action_chunk_boundary="full_only",
    )

    actual = [int(value) for batch in torch_loader for value in batch[0]]
    assert actual == [0, 1, 2, 3, 4, 7, 8, 9]
    assert torch.equal(
        _full_only_anchor_indices(dataset, action_horizon=3),
        torch.tensor([0, 1, 2, 3, 4, 7, 8, 9]),
    )


def test_openpi_loader_full_only_keeps_distributed_ranks_equal_and_disjoint() -> None:
    from openpi.training.data_loader import TransformedDataset

    raw_dataset = TensorDataset(torch.arange(13))
    raw_dataset.episode_data_index = {
        "from": torch.tensor([0, 7]),
        "to": torch.tensor([7, 13]),
    }
    dataset = TransformedDataset(raw_dataset, [])
    rank_indices = []
    for rank in range(2):
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
        torch_loader = DataLoader(dataset, batch_size=2, sampler=sampler)
        data_loader = SimpleNamespace(
            _data_loader=SimpleNamespace(_data_loader=torch_loader)
        )
        _adapt_official_openpi_loader(
            data_loader,
            seed=42,
            rank=rank,
            action_horizon=3,
            action_chunk_boundary="full_only",
        )
        rank_indices.append(
            [int(value) for batch in torch_loader for value in batch[0]]
        )

    assert len(rank_indices[0]) == len(rank_indices[1]) == 4
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])
    assert set(rank_indices[0]) | set(rank_indices[1]) <= {
        0,
        1,
        2,
        3,
        4,
        7,
        8,
        9,
        10,
    }


def test_openpi_rlinf_precision_gate_requires_fp32_masters() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()

    _validate_fp32_training_state(model, optimizer)

    with pytest.raises(RuntimeError, match="FP32 masters"):
        _validate_fp32_training_state(model.to(torch.bfloat16), optimizer)


def test_openpi_rlinf_ema_matches_jax_update_and_swaps_reversibly() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 3.0]]))
    ema = _initialize_ema_state(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[5.0, 7.0]]))

    _update_ema_state(model, ema, 0.75)

    torch.testing.assert_close(ema["weight"], torch.tensor([[2.0, 4.0]]))
    _swap_parameter_data_with_ema(model, ema, chunk_elements=1)
    torch.testing.assert_close(model.weight, torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(ema["weight"], torch.tensor([[5.0, 7.0]]))
    _swap_parameter_data_with_ema(model, ema, chunk_elements=1)
    torch.testing.assert_close(model.weight, torch.tensor([[5.0, 7.0]]))
    torch.testing.assert_close(ema["weight"], torch.tensor([[2.0, 4.0]]))
