"""Strict, allocation-free inspection of RLINF OpenPI weights."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class ExpectedCheckpointLayout:
    tensors: dict[str, TensorSpec]
    tied_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ResolvedInferenceWeights:
    root: Path
    format: str
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _StoredTensor:
    spec: TensorSpec
    storage_identity: tuple[Any, ...] | None


_SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
}


def resolve_rlinf_openpi_weights(checkpoint: str | Path) -> ResolvedInferenceWeights:
    """Resolve the one OpenPI state dict used by ClothDojo training and serving."""

    root = Path(checkpoint).expanduser().resolve()
    if root.is_file():
        if root.suffix == ".pt":
            return ResolvedInferenceWeights(root=root.parent, format="pytorch", paths=(root,))
        if root.suffix == ".safetensors":
            return ResolvedInferenceWeights(
                root=root.parent,
                format="safetensors",
                paths=(root,),
            )
        raise ValueError(f"unsupported OpenPI checkpoint file: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"RLINF OpenPI checkpoint does not exist: {root}")

    # Match openpi_rlinf.utils.rlt_utils.resolve_full_weights, used by serving.
    for relative in (
        "actor/model_state_dict/full_weights.pt",
        "model_state_dict/full_weights.pt",
        "full_weights.pt",
    ):
        candidate = root / relative
        if candidate.is_file():
            return ResolvedInferenceWeights(root=root, format="pytorch", paths=(candidate,))

    safetensors = tuple(sorted(root.glob("*.safetensors")))
    if safetensors:
        return ResolvedInferenceWeights(root=root, format="safetensors", paths=safetensors)
    raise FileNotFoundError(
        "RLINF OpenPI inference weights do not exist; expected "
        f"{root / 'model_state_dict/full_weights.pt'}, "
        f"{root / 'actor/model_state_dict/full_weights.pt'}, "
        f"{root / 'full_weights.pt'}, or root-level *.safetensors"
    )


def inspect_rlinf_openpi_checkpoint(
    checkpoint: str | Path,
    *,
    action_horizon: int,
    expected_layout: ExpectedCheckpointLayout | None = None,
) -> dict[str, Any]:
    """Prove that the selected checkpoint exactly fits the inference model.

    PyTorch checkpoint payloads are inspected with fake tensors, and safetensors
    through their headers. Neither path materializes real parameter storage.
    """

    resolved = resolve_rlinf_openpi_weights(checkpoint)
    expected = expected_layout or _expected_openpi_rlinf_layout(int(action_horizon))
    state_dict_layout = _state_dict_layout(resolved)
    if expected_layout is None and state_dict_layout in {"sft_wrapper", "cfg_wrapper"}:
        expected = _wrapper_layout(expected)
    actual = _read_stored_tensors(resolved)
    aliases = _validate_layout(actual, expected, resolved=resolved)
    digest = _weights_sha256(resolved)
    dtype_counts = Counter(tensor.spec.dtype for tensor in actual.values())
    return {
        "schema": "clothdojo_rlinf_openpi_checkpoint",
        "format_version": 1,
        "compatible": True,
        "loader": "rlinf.models.embodiment.openpi_rlinf",
        "checkpoint_root": str(resolved.root),
        "weights_format": resolved.format,
        "state_dict_layout": state_dict_layout,
        "weights_paths": [str(path) for path in resolved.paths],
        "weights_sha256": digest,
        "weights_size_bytes": sum(path.stat().st_size for path in resolved.paths),
        "tensor_count": len(actual),
        "tensor_elements": sum(math.prod(tensor.spec.shape) for tensor in actual.values()),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "stored_dtype_contract": (
            (
                "released_base_all_float32"
                if expected_layout is not None
                else "source_float32_runtime_bfloat16"
            )
            if resolved.format == "safetensors"
            else (
                "native_openpi_rlinf_wrapper_float32"
                if state_dict_layout in {"sft_wrapper", "cfg_wrapper"}
                else "post_conversion_model_state_dict_exact"
            )
        ),
        "expected_state_dict_keys": len(expected.tensors),
        "tied_aliases": aliases,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "dtype_mismatches": [],
        "payload_materialized": False,
    }


def rlinf_openpi_weights_sha256(checkpoint: str | Path) -> str:
    """Hash only the weights that the RLINF OpenPI loader will execute."""

    return _weights_sha256(resolve_rlinf_openpi_weights(checkpoint))


def strict_reload_rlinf_openpi_checkpoint(
    model: Any,
    checkpoint: str | Path,
    *,
    action_horizon: int,
    expected_layout: ExpectedCheckpointLayout | None = None,
) -> dict[str, Any]:
    """Reload validated weights with strict keys and deterministic alias semantics.

    Legacy ``strict=False`` loads tied keys in state-dict traversal order. RLINF's
    Pi0.5 model visits ``lm_head`` last, so old checkpoints with two divergent
    copies execute the ``lm_head`` value. Pointing every alias at that canonical
    tensor preserves the old result while making the final load strict.
    """

    resolved = resolve_rlinf_openpi_weights(checkpoint)
    expected = expected_layout or _expected_openpi_rlinf_layout(int(action_horizon))
    return _strict_reload(model, resolved, expected)


def strict_reload_matching_rlinf_openpi_checkpoint(
    model: Any,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Validate and strictly reload weights against an already-built OpenPI model.

    This is used by the CFG subclass, whose state-dict layout is owned by the
    pinned RLINF implementation.
    """

    resolved = resolve_rlinf_openpi_weights(checkpoint)
    expected = _layout_from_model(model)
    actual = _read_stored_tensors(resolved)
    aliases = _validate_layout(actual, expected, resolved=resolved)
    reload_manifest = _strict_reload(model, resolved, expected)
    return {
        **reload_manifest,
        "tensor_count": len(actual),
        "tied_aliases": aliases,
    }


def _strict_reload(
    model: Any,
    resolved: ResolvedInferenceWeights,
    expected: ExpectedCheckpointLayout,
) -> dict[str, Any]:
    """Strictly load one already-resolved, already-validated state dict."""

    if resolved.format == "safetensors" and len(resolved.paths) == 1:
        from safetensors.torch import load_model

        stored_keys = set(_read_safetensors_metadata(resolved.paths))
        canonicalized = _canonicalized_alias_manifest(expected, stored_keys)
        missing, unexpected = load_model(model, resolved.paths[0], strict=True)
        if missing or unexpected:
            raise RuntimeError(
                "strict safetensors reload returned incompatible keys: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return {
            "strict": True,
            "weights_sha256": _weights_sha256(resolved),
            "canonicalized_aliases": canonicalized,
        }

    state_dict = _load_state_dict_for_strict_reload(resolved)
    canonicalized = _canonicalized_alias_manifest(expected, set(state_dict))
    for group in expected.tied_groups:
        present = [key for key in group if key in state_dict]
        canonical_key = group[-1] if group[-1] in state_dict else present[-1]
        canonical = state_dict[canonical_key]
        for key in group:
            state_dict[key] = canonical
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict RLINF checkpoint reload returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    return {
        "strict": True,
        "weights_sha256": _weights_sha256(resolved),
        "canonicalized_aliases": canonicalized,
    }


def _layout_from_model(model: Any) -> ExpectedCheckpointLayout:
    tensors = {
        key: TensorSpec(tuple(int(dim) for dim in value.shape), _torch_dtype_name(value.dtype))
        for key, value in model.state_dict().items()
    }
    aliases: defaultdict[int, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].append(name)
    tied_groups = tuple(tuple(names) for names in aliases.values() if len(names) > 1)
    return ExpectedCheckpointLayout(tensors=tensors, tied_groups=tied_groups)


def _wrapper_layout(layout: ExpectedCheckpointLayout) -> ExpectedCheckpointLayout:
    return ExpectedCheckpointLayout(
        tensors={f"model.{key}": value for key, value in layout.tensors.items()},
        tied_groups=tuple(
            tuple(f"model.{key}" for key in group) for group in layout.tied_groups
        ),
    )


def _state_dict_layout(resolved: ResolvedInferenceWeights) -> str:
    contract_candidates = (
        resolved.root / "clothdojo_weights.json",
        resolved.paths[0].parent.parent / "clothdojo_weights.json",
    )
    contract = next((path for path in contract_candidates if path.is_file()), None)
    if contract is None:
        return "pi0_core"
    try:
        value = json.loads(contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read RLINF checkpoint contract {contract}: {exc}") from exc
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError(f"unsupported RLINF checkpoint contract: {contract}")
    schema = value.get("schema")
    if schema == "clothdojo_openpi_rlinf_checkpoint":
        valid = value.get("ema") == {
            "decay": 0.99,
            "path": "model_state_dict/full_weights.pt",
            "purpose": "inference and rollout",
        }
    elif schema == "clothdojo_openpi_rlinf_cfg_checkpoint":
        valid = value.get("weights") == {
            "path": "model_state_dict/full_weights.pt",
            "purpose": "inference and rollout",
            "source": "online",
        }
    else:
        valid = False
    if not valid:
        raise ValueError(f"unsupported RLINF checkpoint contract: {contract}")
    return "sft_wrapper" if schema == "clothdojo_openpi_rlinf_checkpoint" else "cfg_wrapper"


def _canonicalized_alias_manifest(
    expected: ExpectedCheckpointLayout,
    stored_keys: set[str],
) -> list[dict[str, Any]]:
    canonicalized: list[dict[str, Any]] = []
    for group in expected.tied_groups:
        present = [key for key in group if key in stored_keys]
        if not present:
            raise ValueError(f"strict reload has no stored tied-weight key: {group}")
        canonicalized.append(
            {
                "keys": list(group),
                "canonical_source_key": group[-1] if group[-1] in stored_keys else present[-1],
                "semantics": "last_model_traversal_key",
            }
        )
    return canonicalized


@lru_cache(maxsize=8)
def _expected_openpi_rlinf_layout(action_horizon: int) -> ExpectedCheckpointLayout:
    """Construct the current RLinf Pi0.5 core without allocating parameters."""

    import torch
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config

    config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=action_horizon,
        max_token_len=200,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="float32",
        pcd=False,
    )
    with torch.device("meta"):
        model = config.create()
    tensors = {
        key: TensorSpec(
            tuple(int(dim) for dim in value.shape),
            _torch_dtype_name(value.dtype),
        )
        for key, value in model.state_dict().items()
    }
    del model
    return ExpectedCheckpointLayout(tensors=tensors, tied_groups=())


def _read_stored_tensors(resolved: ResolvedInferenceWeights) -> dict[str, _StoredTensor]:
    if resolved.format == "pytorch":
        return _read_pytorch_metadata(resolved.paths[0])
    return _read_safetensors_metadata(resolved.paths)


def _read_pytorch_metadata(path: Path) -> dict[str, _StoredTensor]:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    try:
        with FakeTensorMode():
            state_dict = torch.load(
                path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
    except Exception as fake_error:
        # FSDP full-state checkpoints can retain Tensor rebuild metadata that
        # FakeTensorMode cannot wrap. A meta-device load still reads only the
        # restricted weights-only metadata and never materializes the payload.
        try:
            state_dict = torch.load(
                path,
                map_location="meta",
                mmap=True,
                weights_only=True,
            )
        except Exception as meta_error:
            raise ValueError(
                f"cannot inspect RLINF full_weights.pt safely: {path}: "
                f"fake={fake_error}; meta={meta_error}"
            ) from meta_error
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError(f"RLINF full_weights.pt must contain a non-empty state dict: {path}")

    result: dict[str, _StoredTensor] = {}
    for key, tensor in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(tensor):
            raise TypeError(f"RLINF full_weights.pt contains a non-tensor state entry: {key!r}")
        storage = tensor.untyped_storage()
        checkpoint_offset = (
            None if tensor.device.type == "meta" else getattr(storage, "_checkpoint_offset", None)
        )
        storage_identity = None
        if checkpoint_offset is not None:
            storage_identity = (
                int(checkpoint_offset),
                int(storage.nbytes()),
                int(tensor.storage_offset()),
                tuple(int(value) for value in tensor.stride()),
                tuple(int(value) for value in tensor.shape),
                _torch_dtype_name(tensor.dtype),
            )
        result[key] = _StoredTensor(
            TensorSpec(
                tuple(int(dim) for dim in tensor.shape),
                _torch_dtype_name(tensor.dtype),
            ),
            storage_identity,
        )
    return result


def _read_safetensors_metadata(paths: tuple[Path, ...]) -> dict[str, _StoredTensor]:
    from safetensors import safe_open

    result: dict[str, _StoredTensor] = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key in result:
                    raise ValueError(f"duplicate safetensors checkpoint key {key!r}: {path}")
                tensor = stream.get_slice(key)
                stored_dtype = str(tensor.get_dtype())
                try:
                    dtype = _SAFETENSORS_DTYPES[stored_dtype]
                except KeyError as exc:
                    raise ValueError(
                        f"unsupported safetensors dtype {stored_dtype!r} for {key!r}"
                    ) from exc
                result[key] = _StoredTensor(
                    TensorSpec(tuple(int(dim) for dim in tensor.get_shape()), dtype),
                    None,
                )
    if not result:
        raise ValueError(f"RLINF safetensors checkpoint contains no tensors: {paths}")
    return result


def _validate_layout(
    actual: dict[str, _StoredTensor],
    expected: ExpectedCheckpointLayout,
    *,
    resolved: ResolvedInferenceWeights,
) -> list[dict[str, Any]]:
    expected_keys = set(expected.tensors)
    actual_keys = set(actual)
    unexpected = sorted(actual_keys - expected_keys)
    if unexpected:
        raise ValueError(f"RLINF checkpoint contains unexpected model keys: {unexpected[:8]}")

    shape_mismatches = {
        key: {"expected": expected.tensors[key].shape, "actual": actual[key].spec.shape}
        for key in sorted(actual_keys)
        if actual[key].spec.shape != expected.tensors[key].shape
    }
    if shape_mismatches:
        raise ValueError(
            f"RLINF checkpoint tensor shape mismatch: {dict(list(shape_mismatches.items())[:8])}"
        )
    dtype_mismatches = {
        key: {
            "expected": (
                "float32" if resolved.format == "safetensors" else expected.tensors[key].dtype
            ),
            "actual": actual[key].spec.dtype,
        }
        for key in sorted(actual_keys)
        if actual[key].spec.dtype
        != ("float32" if resolved.format == "safetensors" else expected.tensors[key].dtype)
    }
    if dtype_mismatches:
        raise ValueError(
            f"RLINF checkpoint tensor dtype mismatch: {dict(list(dtype_mismatches.items())[:8])}"
        )

    missing = expected_keys - actual_keys
    accepted_aliases: list[dict[str, Any]] = []
    tied_keys: set[str] = set()
    for group in expected.tied_groups:
        tied_keys.update(group)
        present = [key for key in group if key in actual_keys]
        absent = [key for key in group if key in missing]
        if len(present) == 1 and len(absent) == len(group) - 1:
            accepted_aliases.append(
                {
                    "stored_keys": present,
                    "omitted_keys": absent,
                    "semantics": "same_parameter_alias_omission",
                }
            )
            missing.difference_update(absent)
            continue
        if len(present) != len(group):
            continue
        identities = [actual[key].storage_identity for key in present]
        if resolved.format != "pytorch":
            raise ValueError(
                "RLINF checkpoint stores every tied parameter key but cannot prove shared "
                f"storage: {present}"
            )
        if all(identity is not None for identity in identities) and len(set(identities)) == 1:
            accepted_aliases.append(
                {
                    "stored_keys": present,
                    "omitted_keys": [],
                    "semantics": "shared_checkpoint_storage",
                }
            )
            continue
        comparison = _compare_tied_tensor_values(resolved.paths[0], present)
        if comparison["differing_elements"]:
            accepted_aliases.append(
                {
                    "stored_keys": present,
                    "omitted_keys": [],
                    "semantics": "canonicalized_divergent_checkpoint_storage",
                    "canonical_source_key": group[-1],
                    "comparison": comparison,
                }
            )
            continue
        accepted_aliases.append(
            {
                "stored_keys": present,
                "omitted_keys": [],
                "semantics": "equal_independent_checkpoint_storage",
                "comparison": comparison,
            }
        )

    if missing:
        raise ValueError(f"RLINF checkpoint is missing model keys: {sorted(missing)[:8]}")
    if len(accepted_aliases) != len(expected.tied_groups):
        raise ValueError(
            "RLINF checkpoint tied-weight contract differs from the meta-device model; "
            f"expected={expected.tied_groups}, accepted={accepted_aliases}, "
            f"tied_keys={sorted(tied_keys)}"
        )
    return accepted_aliases


def _load_state_dict_for_strict_reload(resolved: ResolvedInferenceWeights) -> dict[str, Any]:
    import torch

    if resolved.format == "pytorch":
        try:
            state_dict = torch.load(
                resolved.paths[0],
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError(
                f"cannot mmap RLINF full_weights.pt for strict reload: {resolved.paths[0]}: {exc}"
            ) from exc
        if not isinstance(state_dict, dict):
            raise TypeError("RLINF full_weights.pt does not contain a state dict")
        return state_dict

    from safetensors.torch import load_file

    state_dict: dict[str, Any] = {}
    for path in resolved.paths:
        shard = load_file(path, device="cpu")
        duplicates = sorted(set(state_dict) & set(shard))
        if duplicates:
            raise ValueError(
                f"duplicate safetensors checkpoint keys during reload: {duplicates[:8]}"
            )
        state_dict.update(shard)
    return state_dict


def _compare_tied_tensor_values(path: Path, keys: list[str]) -> dict[str, int | float]:
    """Compare independent tied-weight copies without reading the whole checkpoint into RAM."""

    import torch

    try:
        state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot mmap RLINF tied weights for comparison: {path}: {exc}") from exc
    reference = state_dict[keys[0]].reshape(-1)
    candidates = [state_dict[key].reshape(-1) for key in keys[1:]]
    total = int(reference.numel())
    differing = 0
    max_abs_diff = 0.0
    chunk_elements = 1024 * 1024
    with torch.no_grad():
        for candidate in candidates:
            if candidate.numel() != total:
                raise ValueError(f"tied checkpoint tensors have different sizes: {keys}")
            for start in range(0, total, chunk_elements):
                end = min(start + chunk_elements, total)
                left = reference[start:end]
                right = candidate[start:end]
                difference_mask = left != right
                chunk_differing = int(torch.count_nonzero(difference_mask).item())
                differing += chunk_differing
                if chunk_differing:
                    absolute = torch.abs(
                        left[difference_mask].float() - right[difference_mask].float()
                    )
                    absolute = torch.nan_to_num(
                        absolute,
                        nan=float("inf"),
                        posinf=float("inf"),
                        neginf=float("inf"),
                    )
                    chunk_max = float(torch.max(absolute).item())
                    max_abs_diff = max(max_abs_diff, chunk_max)
    del state_dict
    return {
        "differing_elements": differing,
        "total_elements": total * len(candidates),
        "max_abs_diff": max_abs_diff,
    }


def _weights_sha256(resolved: ResolvedInferenceWeights) -> str:
    if len(resolved.paths) == 1:
        return _file_sha256(resolved.paths[0])
    digest = hashlib.sha256()
    for path in resolved.paths:
        relative = path.relative_to(resolved.root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        _update_file_digest(digest, path)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    _update_file_digest(digest, path)
    return digest.hexdigest()


def _update_file_digest(digest: Any, path: Path) -> None:
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)


def _torch_dtype_name(dtype: Any) -> str:
    value = str(dtype)
    return value.removeprefix("torch.")
