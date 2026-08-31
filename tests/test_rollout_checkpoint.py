from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from clothdojo.rollout.checkpoint import (
    ExpectedCheckpointLayout,
    TensorSpec,
    _read_pytorch_metadata,
    inspect_rlinf_openpi_checkpoint,
    resolve_rlinf_openpi_weights,
    strict_reload_matching_rlinf_openpi_checkpoint,
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
        action_horizon=50,
        expected_layout=_layout(),
    )


def test_pytorch_metadata_falls_back_to_meta_for_fsdp_tensor_rebuild(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "full_weights.pt"
    path.write_bytes(b"checkpoint")
    calls = []

    def load(source, **kwargs):
        calls.append((source, kwargs))
        if kwargs["map_location"] == "cpu":
            raise RuntimeError("FakeTensor cannot rebuild FSDP Tensor metadata")
        return {"weight": torch.empty((3, 2), device="meta")}

    monkeypatch.setattr(torch, "load", load)

    result = _read_pytorch_metadata(path)

    assert result["weight"].spec == TensorSpec((3, 2), "float32")
    assert result["weight"].storage_identity is None
    assert [call[1]["map_location"] for call in calls] == ["cpu", "meta"]
    assert all(call[1]["mmap"] is True for call in calls)
    assert all(call[1]["weights_only"] is True for call in calls)


def test_strict_checkpoint_accepts_layout_without_tied_weights(tmp_path: Path) -> None:
    layout = ExpectedCheckpointLayout(
        tensors={"projection.weight": TensorSpec((2, 2), "float32")},
        tied_groups=(),
    )
    save_file(
        {"projection.weight": torch.zeros(2, 2)},
        tmp_path / "model.safetensors",
    )

    manifest = inspect_rlinf_openpi_checkpoint(
        tmp_path,
        action_horizon=50,
        expected_layout=layout,
    )

    assert manifest["tied_aliases"] == []


def test_native_sft_contract_identifies_wrapper_state_dict(tmp_path: Path) -> None:
    weights = _write_full_weights(
        tmp_path,
        {
            "model.embedding.weight": torch.zeros(4, 2),
            "model.lm_head.weight": torch.zeros(4, 2),
            "model.projection.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        },
    )
    actor = weights.parents[1]
    (actor / "clothdojo_weights.json").write_text(
        '{"schema":"clothdojo_openpi_rlinf_checkpoint","format_version":1,'
        '"ema":{"decay":0.99,"path":"model_state_dict/full_weights.pt",'
        '"purpose":"inference and rollout"}}'
    )
    layout = ExpectedCheckpointLayout(
        tensors={
            "model.embedding.weight": TensorSpec((4, 2), "float32"),
            "model.lm_head.weight": TensorSpec((4, 2), "float32"),
            "model.projection.weight": TensorSpec((2, 2), "bfloat16"),
        },
        tied_groups=(),
    )

    manifest = inspect_rlinf_openpi_checkpoint(
        actor,
        action_horizon=50,
        expected_layout=layout,
    )

    assert manifest["state_dict_layout"] == "sft_wrapper"
    assert manifest["stored_dtype_contract"] == "native_openpi_rlinf_wrapper_float32"


def test_native_cfg_contract_identifies_wrapper_from_step_root(tmp_path: Path) -> None:
    weights = _write_full_weights(
        tmp_path,
        {
            "model.embedding.weight": torch.zeros(4, 2),
            "model.lm_head.weight": torch.zeros(4, 2),
            "model.projection.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        },
    )
    actor = weights.parents[1]
    (actor / "clothdojo_weights.json").write_text(
        '{"schema":"clothdojo_openpi_rlinf_cfg_checkpoint","format_version":1,'
        '"weights":{"path":"model_state_dict/full_weights.pt",'
        '"purpose":"inference and rollout","source":"online"}}'
    )
    layout = ExpectedCheckpointLayout(
        tensors={
            "model.embedding.weight": TensorSpec((4, 2), "float32"),
            "model.lm_head.weight": TensorSpec((4, 2), "float32"),
            "model.projection.weight": TensorSpec((2, 2), "bfloat16"),
        },
        tied_groups=(),
    )

    manifest = inspect_rlinf_openpi_checkpoint(
        tmp_path,
        action_horizon=50,
        expected_layout=layout,
    )

    assert manifest["state_dict_layout"] == "cfg_wrapper"


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
        action_horizon=50,
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
        action_horizon=50,
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


def test_matching_model_reload_validates_cfg_style_runtime_layout(tmp_path: Path) -> None:
    shared = torch.zeros(4, 2)
    _write_full_weights(
        tmp_path,
        {
            "embedding.weight": shared,
            "lm_head.weight": shared,
            "projection.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        },
    )
    model = _TiedModel()

    manifest = strict_reload_matching_rlinf_openpi_checkpoint(model, tmp_path)

    assert manifest["strict"] is True
    assert manifest["tensor_count"] == 3
    torch.testing.assert_close(model.embedding.weight, torch.zeros(4, 2))


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
        action_horizon=50,
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
    from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
        load_full_wrapper_weights,
        resolve_full_weights,
    )

    actor = _write_full_weights(tmp_path, {"model.weight": torch.full((1, 1), 2.0)})
    direct = tmp_path / "model_state_dict/full_weights.pt"
    direct.parent.mkdir(parents=True)
    torch.save({"model.weight": torch.full((1, 1), -3.0)}, direct)
    dcp = tmp_path / "actor/dcp_checkpoint"
    dcp.mkdir(parents=True)
    (dcp / ".metadata").write_bytes(b"metadata")
    (dcp / "__0_0.distcp").write_bytes(b"optimizer and shards")

    resolved = resolve_rlinf_openpi_weights(tmp_path)

    loaded_path = resolve_full_weights(tmp_path)
    assert resolved.paths == (loaded_path,) == (actor,)
    wrapper = torch.nn.Module()
    wrapper.model = torch.nn.Linear(1, 1, bias=False)
    load_full_wrapper_weights(wrapper, loaded_path, expect_rlt=False)
    assert wrapper.model.weight.item() == 2.0
    manifest = inspect_rlinf_openpi_checkpoint(
        tmp_path,
        action_horizon=50,
        expected_layout=ExpectedCheckpointLayout(
            tensors={"model.weight": TensorSpec((1, 1), "float32")}, tied_groups=()
        ),
    )
    assert manifest["weights_sha256"] == hashlib.sha256(actor.read_bytes()).hexdigest()
    assert resolve_rlinf_openpi_weights(direct).paths == (resolve_full_weights(direct),)


def test_resolver_rejects_dcp_only_checkpoint(tmp_path: Path) -> None:
    dcp = tmp_path / "actor/dcp_checkpoint"
    dcp.mkdir(parents=True)
    (dcp / ".metadata").write_bytes(b"metadata")
    (dcp / "__0_0.distcp").write_bytes(b"optimizer and shards")

    with pytest.raises(FileNotFoundError, match="inference weights do not exist"):
        resolve_rlinf_openpi_weights(tmp_path)
