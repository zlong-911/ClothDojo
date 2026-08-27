from __future__ import annotations

import hashlib
import json

from clothdojo.rollout import conversion


def test_jax_conversion_publishes_only_after_layout_and_norm_checks(
    monkeypatch,
    tmp_path,
) -> None:
    run = tmp_path / "jax_run"
    source = run / "40000"
    (source / "params").mkdir(parents=True)
    launch = run / "clothdojo_launch.json"
    launch.write_text('{"schema":"clothdojo_pi05_jax_preflight"}\n', encoding="utf-8")
    norm = tmp_path / "norm"
    norm.mkdir()
    (norm / "norm_stats.json").write_text("{}\n", encoding="utf-8")
    (norm / "clothdojo.json").write_text("{}\n", encoding="utf-8")
    parity_config = tmp_path / "jax.yaml"
    parity_config.write_text("schema: test\n", encoding="utf-8")
    calls = []

    def fake_converter(**kwargs):
        calls.append(kwargs)
        output = conversion.Path(kwargs["output_path"])
        output.mkdir()
        (output / "model.safetensors").write_bytes(b"weights")
        return {
            "schema": "clothdojo_jax_openpi_rlinf_mapping",
            "compatible": True,
            "mapped_tensors": 667,
            "strict_load": True,
            "unexpected": [],
        }

    monkeypatch.setattr(
        conversion,
        "inspect_rlinf_openpi_checkpoint",
        lambda *args, **kwargs: {"weights_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        conversion,
        "validate_sft_norm_binding",
        lambda *args, **kwargs: {"norm_stats_sha256": "b" * 64},
    )
    output = tmp_path / "rollout_checkpoint"

    result = conversion.convert_jax_checkpoint(
        source,
        output,
        norm,
        parity_config,
        _converter=fake_converter,
        _parity_checker=lambda *args, **kwargs: {
            "schema": "clothdojo_jax_openpi_rlinf_parity",
            "format_version": 1,
            "compatible": True,
            "weights_sha256": "a" * 64,
            "norm_stats_sha256": "b" * 64,
        },
    )

    assert calls == [
        {
            "checkpoint_dir": str(source.resolve()),
            "output_path": str((tmp_path / ".rollout_checkpoint.tmp").resolve()),
            "norm_stats": str(norm.resolve()),
            "openpi_config": "pi05_aloha_robotwin",
            "action_horizon": 50,
        }
    ]
    assert result["compatible"] is True
    assert output.is_dir()
    assert not (tmp_path / ".rollout_checkpoint.tmp").exists()
    assert (output / "clothdojo_launch.json").read_bytes() == launch.read_bytes()
    metadata = json.loads((output / "clothdojo_conversion.json").read_text())
    assert metadata["weights_sha256"] == "a" * 64
    assert (output / "clothdojo_parity.json").is_file()
    assert metadata["source_launch_manifest_sha256"] == hashlib.sha256(
        launch.read_bytes()
    ).hexdigest()
