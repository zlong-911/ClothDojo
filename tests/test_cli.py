from __future__ import annotations

import json
import subprocess
import sys
import textwrap

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


def test_rtc_rollout_check_calls_acceptance_entry(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.rollout.check.run_rtc_rollout_check",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs) or {"compatible": True},
    )

    cli.main(
        [
            "rollout",
            "check-rtc",
            "--scenario",
            "configs/rollout/gate.yaml",
            "--assets",
            "/data/assets",
            "--trajectory-output",
            "/data/trajectory",
            "--lerobot-output",
            "/data/lerobot",
            "--device",
            "cuda:2",
        ]
    )

    assert call == {
        "args": (
            "configs/rollout/gate.yaml",
            "/data/assets",
            "/data/trajectory",
            "/data/lerobot",
        ),
        "kwargs": {"device": "cuda:2"},
    }
    assert json.loads(capsys.readouterr().out) == {"compatible": True}


def test_make_states_calls_generator(monkeypatch, capsys) -> None:
    call = {}
    monkeypatch.setattr(
        "clothdojo.rollout.initial_states.generate_initial_states",
        lambda *args, **kwargs: call.update(args=args, kwargs=kwargs) or {"compatible": True},
    )

    cli.main(
        [
            "rollout",
            "make-states",
            "--garment",
            "shirt",
            "--count",
            "4",
            "--output",
            "/data/states.npz",
            "--assets",
            "/data/assets",
            "--device",
            "cuda:2",
            "--batch-size",
            "2",
            "--max-candidates",
            "16",
            "--seed",
            "7",
            "--hold-seconds",
            "1.0",
            "--settle-seconds",
            "2.0",
        ]
    )

    assert call == {
        "args": ("/data/assets", "/data/states.npz"),
        "kwargs": {
            "garment_id": "shirt",
            "state_count": 4,
            "candidate_batch": 2,
            "max_candidates": 16,
            "seed": 7,
            "hold_seconds": 1.0,
            "settle_seconds": 2.0,
            "device": "cuda:2",
        },
    }
    assert json.loads(capsys.readouterr().out) == {"compatible": True}


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
            "data.k=8",
        ]
    )

    assert call == {
        "kind": "steam",
        "config_path": "configs/training/steam.yaml",
        "overrides": ["data.k=8"],
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
            "pi05",
            "--config",
            "configs/training/pi05_aloha.yaml",
            "--check",
            "runner.max_steps=2",
        ]
    )

    assert call == {
        "kind": "pi05",
        "config_path": "configs/training/pi05_aloha.yaml",
        "overrides": ["runner.max_steps=2"],
    }
    assert json.loads(capsys.readouterr().out) == {"runner": {"max_steps": 10000}}


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
