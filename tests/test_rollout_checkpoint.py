from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from clothdojo.rollout.checkpoint import (
    ExpectedCheckpointLayout,
    TensorSpec,
    _expected_legacy_openpi_layout,
    inspect_rlinf_openpi_checkpoint,
    resolve_rlinf_openpi_weights,
    strict_reload_rlinf_openpi_checkpoint,
)


def _layout() -> ExpectedCheckpointLayout:
    return ExpectedCheckpointLayout(
        tensors={
            "embedding.weight": TensorSpec((4, 2), "float32"),
            "lm_head.weight": TensorSpec((4, 2), "float32"),
            "projection.weight": TensorSpec((2, 2), "bfloat16"),
        },
        tied_groups=(("embedding.weight", "lm_head.weight"),),
    )


def _write_full_weights(root: Path, state_dict: dict[str, torch.Tensor]) -> Path:
    path = root / "actor/model_state_dict/full_weights.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    return path


def _inspect(root: Path) -> dict:
    return inspect_rlinf_openpi_checkpoint(
        root,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
        action_env_dim=14,
        num_steps=5,
        expected_layout=_layout(),
    )


class _TiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.projection = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)


def test_strict_checkpoint_accepts_exact_keys_dtypes_and_shared_alias(tmp_path: Path) -> None:
    shared = torch.zeros(4, 2)
    weights = _write_full_weights(
        tmp_path,
        {
            "embedding.weight": shared,
            "lm_head.weight": shared,
            "projection.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        },
    )

    manifest = _inspect(tmp_path)

    assert manifest["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert manifest["dtype_counts"] == {"bfloat16": 1, "float32": 2}
    assert manifest["payload_materialized"] is False
    assert manifest["tied_aliases"] == [
        {
            "stored_keys": ["embedding.weight", "lm_head.weight"],
            "omitted_keys": [],
            "semantics": "shared_checkpoint_storage",
        }
    ]
    model = _TiedModel()
    reload_manifest = strict_reload_rlinf_openpi_checkpoint(
        model,
        tmp_path,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
        action_env_dim=14,
        num_steps=5,
        expected_layout=_layout(),
    )
    torch.testing.assert_close(model.embedding.weight, torch.zeros(4, 2))
    assert reload_manifest["canonicalized_aliases"][0]["canonical_source_key"] == ("lm_head.weight")


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing", "missing model keys"),
        ("unexpected", "unexpected model keys"),
        ("shape", "shape mismatch"),
        ("dtype", "dtype mismatch"),
    ),
)
def test_strict_checkpoint_rejects_every_layout_mismatch(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    shared = torch.zeros(4, 2)
    state_dict = {
        "embedding.weight": shared,
        "lm_head.weight": shared,
        "projection.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
    }
    if mutation == "missing":
        del state_dict["projection.weight"]
    elif mutation == "unexpected":
        state_dict["other.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state_dict["projection.weight"] = torch.zeros(3, 2, dtype=torch.bfloat16)
    elif mutation == "dtype":
        state_dict["projection.weight"] = torch.zeros(2, 2)
    _write_full_weights(tmp_path, state_dict)

    with pytest.raises(ValueError, match=match):
        _inspect(tmp_path)


def test_safetensors_accepts_only_explicit_tied_alias_omission(tmp_path: Path) -> None:
    save_file(
        {
            "embedding.weight": torch.zeros(4, 2),
            "projection.weight": torch.zeros(2, 2),
        },
        tmp_path / "model.safetensors",
    )

    manifest = _inspect(tmp_path)

    assert manifest["weights_format"] == "safetensors"
    assert manifest["stored_dtype_contract"] == "released_base_all_float32"
    assert manifest["tied_aliases"] == [
        {
            "stored_keys": ["embedding.weight"],
            "omitted_keys": ["lm_head.weight"],
            "semantics": "same_parameter_alias_omission",
        }
    ]
    model = _TiedModel()
    reload_manifest = strict_reload_rlinf_openpi_checkpoint(
        model,
        tmp_path,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
        action_env_dim=14,
        num_steps=5,
        expected_layout=_layout(),
    )
    torch.testing.assert_close(model.embedding.weight, torch.zeros(4, 2))
    assert reload_manifest["canonicalized_aliases"][0]["canonical_source_key"] == (
        "embedding.weight"
    )


def test_safetensors_rejects_two_independent_copies_of_tied_weight(tmp_path: Path) -> None:
    save_file(
        {
            "embedding.weight": torch.zeros(4, 2),
            "lm_head.weight": torch.zeros(4, 2),
            "projection.weight": torch.zeros(2, 2),
        },
        tmp_path / "model.safetensors",
    )

    with pytest.raises(ValueError, match="cannot prove shared storage"):
        _inspect(tmp_path)


def test_full_weights_accepts_equal_independent_tied_copies(tmp_path: Path) -> None:
    shared = torch.zeros(4, 2)
    _write_full_weights(
        tmp_path,
        {
            "embedding.weight": shared,
            "lm_head.weight": shared.clone(),
            "projection.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        },
    )

    manifest = _inspect(tmp_path)

    alias = manifest["tied_aliases"][0]
    assert alias["semantics"] == "equal_independent_checkpoint_storage"
    assert alias["comparison"] == {
        "differing_elements": 0,
        "total_elements": 8,
        "max_abs_diff": 0.0,
    }


def test_full_weights_reports_and_canonicalizes_divergent_tied_copy(tmp_path: Path) -> None:
    embedding = torch.zeros(4, 2)
    head = embedding.clone()
    head[1, 1] = 0.125
    _write_full_weights(
        tmp_path,
        {
            "embedding.weight": embedding,
            "lm_head.weight": head,
            "projection.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        },
    )

    manifest = _inspect(tmp_path)

    alias = manifest["tied_aliases"][0]
    assert alias == {
        "stored_keys": ["embedding.weight", "lm_head.weight"],
        "omitted_keys": [],
        "semantics": "canonicalized_divergent_checkpoint_storage",
        "canonical_source_key": "lm_head.weight",
        "comparison": {
            "differing_elements": 1,
            "total_elements": 8,
            "max_abs_diff": 0.125,
        },
    }

    model = _TiedModel()
    reload_manifest = strict_reload_rlinf_openpi_checkpoint(
        model,
        tmp_path,
        openpi_config="pi05_aloha_robotwin",
        action_horizon=50,
        action_env_dim=14,
        num_steps=5,
        expected_layout=_layout(),
    )

    torch.testing.assert_close(model.embedding.weight, head)
    assert model.embedding.weight is model.lm_head.weight
    assert reload_manifest["canonicalized_aliases"][0]["canonical_source_key"] == ("lm_head.weight")


def test_safetensors_rejects_post_conversion_dtype_in_base_artifact(tmp_path: Path) -> None:
    save_file(
        {
            "embedding.weight": torch.zeros(4, 2),
            "projection.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        },
        tmp_path / "model.safetensors",
    )

    with pytest.raises(ValueError, match="dtype mismatch"):
        _inspect(tmp_path)


def test_resolver_matches_rlinf_loader_precedence_and_ignores_dcp(tmp_path: Path) -> None:
    _write_full_weights(tmp_path, {"actor": torch.zeros(1)})
    direct = tmp_path / "model_state_dict/full_weights.pt"
    direct.parent.mkdir(parents=True)
    torch.save({"direct": torch.zeros(1)}, direct)
    dcp = tmp_path / "actor/dcp_checkpoint"
    dcp.mkdir(parents=True)
    (dcp / ".metadata").write_bytes(b"metadata")
    (dcp / "__0_0.distcp").write_bytes(b"optimizer and shards")

    resolved = resolve_rlinf_openpi_weights(tmp_path)

    assert resolved.paths == (direct,)


def test_resolver_rejects_dcp_only_checkpoint(tmp_path: Path) -> None:
    dcp = tmp_path / "actor/dcp_checkpoint"
    dcp.mkdir(parents=True)
    (dcp / ".metadata").write_bytes(b"metadata")
    (dcp / "__0_0.distcp").write_bytes(b"optimizer and shards")

    with pytest.raises(FileNotFoundError, match="inference weights do not exist"):
        resolve_rlinf_openpi_weights(tmp_path)


def test_meta_layout_locks_pinned_pi05_keys_dtypes_and_alias() -> None:
    layout = _expected_legacy_openpi_layout("pi05_aloha_robotwin", 50, 14, 5)

    assert len(layout.tensors) == 813
    assert Counter(spec.dtype for spec in layout.tensors.values()) == {
        "bfloat16": 691,
        "float32": 122,
    }
    assert layout.tied_groups == (
        (
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
            "paligemma_with_expert.paligemma.lm_head.weight",
        ),
    )
