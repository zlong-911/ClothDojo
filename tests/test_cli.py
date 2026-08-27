from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from clothdojo import cli


def test_environment_check_prints_report(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "clothdojo.diagnostics.inspect_environment",
        lambda *, include_training, smoke: {
            "compatible": True,
            "include_training": include_training,
            "smoke": smoke,
        },
    )

    cli.main(["env", "check", "--training", "--smoke"])

    assert json.loads(capsys.readouterr().out) == {
        "compatible": True,
        "include_training": True,
        "smoke": True,
    }


def test_environment_check_fails_for_incompatible_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "clothdojo.diagnostics.inspect_environment",
        lambda *, include_training, smoke: {"compatible": False},
    )

    with pytest.raises(SystemExit, match="1"):
        cli.main(["env", "check"])


def test_environment_check_rejects_unknown_arguments() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["env", "check", "--typo"])


def test_cli_import_does_not_load_heavy_runtime_dependencies() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = ("hydra", "mujoco", "newton", "rlinf", "torch", "warp")

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                prefixes = tuple(name + "." for name in blocked)
                if fullname in blocked or fullname.startswith(prefixes):
                    raise ImportError(f"blocked heavy import: {fullname}")
                return None

        sys.meta_path.insert(0, Blocker())
        from clothdojo import cli
        cli._parser()
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_jax_sft_check_uses_separate_backend(monkeypatch, capsys) -> None:
    spec = object()
    monkeypatch.setattr(
        cli,
        "_training_config_schema",
        lambda path: "clothdojo_pi05_jax_sft",
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft.compose_jax_sft_spec",
        lambda path, overrides: spec,
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft.inspect_jax_sft",
        lambda value: {"compatible": value is spec},
    )

    cli.main(["train", "openpi-jax", "--config", "jax.yaml", "--check"])

    assert json.loads(capsys.readouterr().out) == {"compatible": True}


def test_jax_sft_run_requires_explicit_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_training_config_schema",
        lambda path: "clothdojo_pi05_jax_sft",
    )
    monkeypatch.setattr(
        "clothdojo.training.jax_sft.compose_jax_sft_spec",
        lambda path, overrides: object(),
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main(["train", "openpi-jax", "--config", "jax.yaml"])


def test_jax_matrix_uses_the_same_training_entry(monkeypatch, capsys) -> None:
    matrix = object()
    monkeypatch.setattr(
        cli,
        "_training_config_schema",
        lambda path: "clothdojo_pi05_jax_matrix",
    )
    monkeypatch.setattr(
        "clothdojo.training.matrix.compose_jax_sft_matrix",
        lambda path: matrix,
    )
    monkeypatch.setattr(
        "clothdojo.training.matrix.inspect_jax_sft_matrix",
        lambda value, **kwargs: {
            "compatible": value is matrix,
            "inspect_artifacts": kwargs["inspect_artifacts"],
        },
    )

    cli.main(["train", "openpi-jax", "--config", "matrix.yaml", "--check"])

    assert json.loads(capsys.readouterr().out) == {
        "compatible": True,
        "inspect_artifacts": True,
    }


@pytest.mark.parametrize("legacy_kind", ["pi05", "pi05-jax", "pi05-jax-matrix"])
def test_training_rejects_removed_openpi_entries(legacy_kind: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["train", legacy_kind, "--config", "unused.yaml", "--check"])


def test_simulation_check_calls_acceptance_entry(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.sim.check.run_shirt_flatten_check",
        lambda assets, **kwargs: call.update(assets=assets, **kwargs) or {"compatible": True},
    )

    cli.main(["sim", "check", "--assets", "/data/assets", "--device", "cuda:1", "--worlds", "4"])

    assert call == {
        "assets": "/data/assets",
        "device": "cuda:1",
        "world_count": 4,
    }
    assert json.loads(capsys.readouterr().out) == {"compatible": True}


def test_make_states_calls_generator(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.rollout.initial_states.generate_initial_state_collection",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs) or {"compatible": True},
    )

    cli.main(
        [
            "rollout",
            "make-states",
            "--seen-assets",
            "2",
            "--unseen-assets",
            "4",
            "--states-per-asset",
            "6",
            "--expert-trajectories",
            "/data/expert",
            "--output",
            "/data/states",
            "--assets",
            "/data/assets",
            "--gpus",
            "2,3",
            "--batch-size",
            "2",
            "--max-candidates",
            "16",
            "--seed",
            "7",
            "--drop-seconds",
            "3.0",
        ]
    )

    assert call == {
        "args": ("/data/assets", "/data/states"),
        "kwargs": {
            "expert_trajectories": "/data/expert",
            "seen_assets": 2,
            "unseen_assets": 4,
            "states_per_asset": 6,
            "gpus": ("2", "3"),
            "candidate_batch": 2,
            "max_candidates": 16,
            "seed": 7,
            "drop_seconds": 3.0,
        },
    }
    assert json.loads(capsys.readouterr().out) == {"compatible": True}


def test_rollout_run_calls_collection_workflow(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.rollout.collection.run_rollout_collection",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs) or {"compatible": True},
    )

    cli.main(
        [
            "rollout",
            "run",
            "--scenario",
            "scenario.yaml",
            "--states",
            "/data/states",
            "--policy",
            "ws://policy:8080",
            "--expected-checkpoint-sha256",
            "ab" * 32,
            "--output",
            "/data/rollouts",
            "--assets",
            "/data/assets",
            "--gpus",
            "3,4",
            "--batch-size",
            "8",
            "--limit-per-asset",
            "2",
            "--seen-assets",
            "5",
            "--unseen-assets",
            "5",
            "--execution",
            "chunk",
            "--exec-horizon",
            "12",
        ]
    )

    assert call == {
        "args": (
            "scenario.yaml",
            "/data/states",
            ("ws://policy:8080",),
            "/data/rollouts",
        ),
        "kwargs": {
            "expected_checkpoint_sha256": "ab" * 32,
            "assets_root": "/data/assets",
            "gpus": ("3", "4"),
            "batch_size": 8,
            "limit_per_asset": 2,
            "seen_assets": 5,
            "unseen_assets": 5,
            "execution": "chunk",
            "exec_horizon": 12,
        },
    }
    assert json.loads(capsys.readouterr().out) == {"compatible": True}


def test_policy_serve_calls_rlinf_boundary(monkeypatch) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.rollout.policy.serve_sft_policy",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs),
    )

    cli.main(
        [
            "policy",
            "serve-sft",
            "--checkpoint",
            "/models/sft",
            "--norm-stats",
            "/data/norm",
            "--device",
            "cuda:4",
            "--port",
            "9000",
        ]
    )

    assert call == {
        "args": ("/models/sft", "/data/norm"),
        "kwargs": {
            "openpi_config": "pi05_aloha_robotwin",
            "action_horizon": 50,
                "num_steps": 10,
                "autocast": True,
                "rtc_execution_horizon": 10,
                "rtc_max_guidance_weight": 10.0,
                "rtc_prefix_schedule": "linear",
                "device": "cuda:4",
            "host": "127.0.0.1",
            "port": 9000,
        },
    }


def test_export_lerobot_detects_rollout_collection(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    source = tmp_path / "rollouts"
    source.mkdir()
    (source / "meta.json").write_text(
        json.dumps({"schema": "clothdojo_rollout"}),
        encoding="utf-8",
    )
    call = {}
    monkeypatch.setattr(
        "clothdojo.data.rollout.export_rollout_collection",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs)
        or {"compatible": True},
    )

    cli.main(
        [
            "data",
            "export-lerobot",
            "--source",
            str(source),
            "--output",
            "/data/lerobot",
            "--assets",
            "/data/assets",
            "--device",
            "cuda:5",
        ]
    )

    assert call == {
        "args": (str(source), "/data/lerobot"),
        "kwargs": {
            "assets_root": "/data/assets",
            "device": "cuda:5",
            "repo_id": None,
            "image_writer_threads": 6,
        },
    }
    assert json.loads(capsys.readouterr().out) == {"compatible": True}


def test_rollout_lerobot_defaults_inside_experiment(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "sft_rtc"
    source.mkdir()
    (source / "meta.json").write_text(
        json.dumps({"schema": "clothdojo_rollout"}), encoding="utf-8"
    )
    call = {}
    monkeypatch.setattr(
        "clothdojo.data.rollout.export_rollout_collection",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs)
        or {"compatible": True},
    )

    cli.main(["data", "export-lerobot", "--source", str(source)])

    assert call["args"] == (str(source), str(source / "lerobot"))


def test_training_command_calls_thin_rlinf_adapter(monkeypatch) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.training.rlinf.run",
        lambda **kwargs: call.update(kwargs),
    )

    cli.main(
        [
            "train",
            "steam",
            "--config",
            "configs/training/steam.yaml",
            "--member-index",
            "1",
            "data.k=8",
        ]
    )

    assert call == {
        "kind": "steam",
        "config_path": "configs/training/steam.yaml",
        "overrides": ["data.k=8"],
        "steam_member_index": 1,
    }


def test_training_check_does_not_call_runner(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.training.rlinf.check",
        lambda **kwargs: call.update(kwargs) or {"runner": {"max_steps": 10000}},
    )
    monkeypatch.setattr(
        "clothdojo.training.rlinf.run",
        lambda **kwargs: pytest.fail("training runner must not be called"),
    )

    cli.main(
        [
            "train",
            "openpi-rlinf",
            "--config",
            "configs/training/pi05_aloha_openpi_rlinf_lr2x_60k.yaml",
            "--check",
            "runner.max_steps=2",
        ]
    )

    assert call == {
        "kind": "openpi-rlinf",
        "config_path": "configs/training/pi05_aloha_openpi_rlinf_lr2x_60k.yaml",
        "overrides": ["runner.max_steps=2"],
        "steam_member_index": None,
    }
    assert json.loads(capsys.readouterr().out) == {"runner": {"max_steps": 10000}}


def test_steam_merge_calls_canonical_assembly(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.training.steam.merge_steam_members",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs)
        or Path("/models/steam/actor"),
    )

    cli.main(
        [
            "steam",
            "merge",
            "--member",
            "/models/member0/actor",
            "--member",
            "/models/member1/actor",
            "--member",
            "/models/member2/actor",
            "--output",
            "/models/steam/actor",
        ]
    )

    assert call == {
        "args": (
            [
                "/models/member0/actor",
                "/models/member1/actor",
                "/models/member2/actor",
            ],
            "/models/steam/actor",
        ),
        "kwargs": {"overwrite": False},
    }
    assert json.loads(capsys.readouterr().out) == {
        "checkpoint": "/models/steam/actor"
    }


def test_norm_stats_command_calls_openpi_preparation(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.training.openpi.compute_norm_stats",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs) or {"frames": 12},
    )

    cli.main(
        [
            "data",
            "compute-norm-stats",
            "--dataset",
            "/data/good",
            "--output",
            "/data/stats",
            "--block-size",
            "128",
        ]
    )

    assert call == {
        "args": ("/data/good", "/data/stats"),
        "kwargs": {"config_name": "pi05_aloha_robotwin", "block_size": 128},
    }
    assert json.loads(capsys.readouterr().out) == {"frames": 12}


def test_cfg_data_check_calls_static_preflight(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.training.cfg.inspect_cfg_training",
        lambda *args: call.update(args=args) or {"compatible": True},
    )

    cli.main(
        [
            "data",
            "check-cfg",
            "--config",
            "configs/training/cfg_pi05_aloha.yaml",
            "data.seed=7",
        ]
    )

    assert call == {
        "args": ("configs/training/cfg_pi05_aloha.yaml", ["data.seed=7"]),
    }
    assert json.loads(capsys.readouterr().out) == {"compatible": True}
