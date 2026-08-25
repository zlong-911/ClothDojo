from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clothdojo.training.matrix import (
    JaxSftMatrix,
    MatrixTrial,
    compose_jax_sft_matrix,
    run_jax_sft_matrix,
)


def test_matrix_resolves_base_config_and_trials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MATRIX_STATUS", str(tmp_path / "status"))
    (tmp_path / "base.yaml").write_text("base", encoding="utf-8")
    path = tmp_path / "matrix.yaml"
    path.write_text(
        """
schema: clothdojo_pi05_jax_matrix
format_version: 1
name: trial-matrix
base_config: base.yaml
status_dir: ${oc.env:MATRIX_STATUS}
continue_on_error: true
stages: [pilot, full]
trials:
  - name: first
    overrides: [experiment.profile=decay_40k_lr_1x]
""",
        encoding="utf-8",
    )

    matrix = compose_jax_sft_matrix(path)

    assert matrix.base_config == tmp_path / "base.yaml"
    assert matrix.status_path == tmp_path / "status/trial-matrix/status.json"
    assert matrix.trials[0].stages == ("pilot", "full")


def test_matrix_rejects_duplicate_trial_names(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MATRIX_STATUS", str(tmp_path / "status"))
    path = tmp_path / "matrix.yaml"
    path.write_text(
        """
schema: clothdojo_pi05_jax_matrix
format_version: 1
name: duplicate
base_config: base.yaml
status_dir: ${oc.env:MATRIX_STATUS}
trials:
  - {name: same, overrides: []}
  - {name: same, overrides: []}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        compose_jax_sft_matrix(path)


def test_matrix_reports_failure_after_continuing_to_later_trials(
    tmp_path: Path, monkeypatch
) -> None:
    matrix = JaxSftMatrix(
        name="continue-after-failure",
        config_path=tmp_path / "matrix.yaml",
        base_config=tmp_path / "base.yaml",
        status_dir=tmp_path / "status",
        continue_on_error=True,
        trials=(
            MatrixTrial(name="fails", overrides=(), stages=("pilot",)),
            MatrixTrial(name="passes", overrides=(), stages=("pilot",)),
        ),
    )
    monkeypatch.setattr(
        "clothdojo.training.matrix.compose_jax_sft_spec",
        lambda *_args: SimpleNamespace(
            tuning_profile="profile",
            run_name="run",
            warmup_steps=1,
            peak_lr=1.0,
            decay_steps=2,
            decay_lr=0.1,
        ),
    )
    monkeypatch.setattr(
        "clothdojo.training.matrix._inspect_in_subprocess",
        lambda *_args: {"artifact_sha256": "preflight"},
    )
    monkeypatch.setattr("clothdojo.training.matrix._compatible_gate", lambda *_args: None)
    calls = iter((RuntimeError("failed gate"), None))

    def run_gate(*_args) -> None:
        error = next(calls)
        if error is not None:
            raise error

    monkeypatch.setattr("clothdojo.training.matrix._run_gate_in_subprocess", run_gate)
    monkeypatch.setattr(
        "clothdojo.training.matrix._read_gate", lambda *_args: {"compatible": True}
    )
    monkeypatch.setattr(
        "clothdojo.training.matrix.gate_manifest_path",
        lambda *_args: tmp_path / "pilot.json",
    )

    report = run_jax_sft_matrix(matrix)

    assert report["state"] == "failed"
    assert [trial["state"] for trial in report["trials"]] == ["failed", "completed"]
    assert json.loads(matrix.status_path.read_text(encoding="utf-8"))["state"] == "failed"


def test_matrix_records_compose_failure_and_continues(
    tmp_path: Path, monkeypatch
) -> None:
    base_config = (
        Path(__file__).resolve().parents[1]
        / "configs/training/pi05_aloha_jax.yaml"
    )
    paths = {
        "CLOTHDOJO_LEROBOT_DATASET": tmp_path / "dataset",
        "CLOTHDOJO_PI05_NORM_STATS": tmp_path / "norm",
        "CLOTHDOJO_PI05_JAX_PARAMS": tmp_path / "params",
        "CLOTHDOJO_PI05_PYTORCH_BASE": tmp_path / "pytorch",
        "CLOTHDOJO_OPENPI_CACHE": tmp_path / "cache",
        "CLOTHDOJO_JAX_CHECKPOINT_BASE": tmp_path / "checkpoints",
        "CLOTHDOJO_JAX_GATE_DIR": tmp_path / "gates",
    }
    for name, path in paths.items():
        monkeypatch.setenv(name, str(path))
    matrix = JaxSftMatrix(
        name="continue-after-compose-failure",
        config_path=tmp_path / "matrix.yaml",
        base_config=base_config,
        status_dir=tmp_path / "status",
        continue_on_error=True,
        trials=(
            MatrixTrial(
                name="fails",
                overrides=("experiment.profile=unknown",),
                stages=("pilot",),
            ),
            MatrixTrial(name="passes", overrides=(), stages=("pilot",)),
        ),
    )
    monkeypatch.setattr(
        "clothdojo.training.matrix._inspect_in_subprocess",
        lambda *_args: {"artifact_sha256": "good"},
    )
    monkeypatch.setattr("clothdojo.training.matrix._compatible_gate", lambda *_args: None)
    monkeypatch.setattr("clothdojo.training.matrix._run_gate_in_subprocess", lambda *_args: None)
    monkeypatch.setattr(
        "clothdojo.training.matrix._read_gate", lambda *_args: {"compatible": True}
    )
    monkeypatch.setattr(
        "clothdojo.training.matrix.gate_manifest_path",
        lambda *_args: tmp_path / "pilot.json",
    )

    report = run_jax_sft_matrix(matrix)

    assert report["state"] == "failed"
    assert report["trials"][0]["state"] == "failed"
    assert "unknown JAX SFT tuning profile" in report["trials"][0]["error"]
    assert report["trials"][1]["state"] == "completed"
    persisted = json.loads(matrix.status_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "failed"
