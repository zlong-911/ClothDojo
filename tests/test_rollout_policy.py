from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import clothdojo.rollout.policy as policy_module
from clothdojo.rollout.policy import (
    DELTA_ACTION_MASK,
    RTC_GUIDANCE,
    RemoteSFTPolicy,
    _load_action_quantile_scale,
    _rebase_delta_model_actions,
    checkpoint_sha256,
    serve_sft_policy,
)
from clothdojo.rollout.render import CAMERA_NAMES

CHECKPOINT_SHA = "ab" * 32
NORM_SHA = "cd" * 32
PROVENANCE_SHA = "de" * 32
MANIFEST_SHA = "ef" * 32
DATASET_INFO_SHA = "12" * 32
DATASET_SIDECAR_SHA = "34" * 32


def _metadata(**changes):
    rtc_guidance = RTC_GUIDANCE.to_json_dict()
    rtc_guidance["execution_horizon"] = 2
    identity = {
        "protocol_version": 2,
        "backend": "rlinf-sft",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "norm_stats_sha256": NORM_SHA,
        "norm_provenance_sha256": PROVENANCE_SHA,
        "training_manifest_sha256": MANIFEST_SHA,
        "dataset_info_sha256": DATASET_INFO_SHA,
        "dataset_sidecar_sha256": DATASET_SIDECAR_SHA,
        "openpi_config": "pi05_aloha_robotwin",
        "action_dim": 14,
        "action_horizon": 4,
        "camera_names": list(CAMERA_NAMES),
        "transport_image_size": [224, 224],
        "batched_inference": True,
        "num_steps": 10,
        "rtc_guidance": rtc_guidance,
    }
    identity.update(changes)
    return {"clothdojo": identity}


class _Client:
    def __init__(self, metadata=None):
        self.metadata = metadata or _metadata()
        self.requests = []

    def get_server_metadata(self):
        return self.metadata

    def infer(self, request):
        self.requests.append(request)
        return {
            "actions": np.zeros((1, 4, 14), dtype=np.float32),
            "model_actions": np.full((1, 4, 32), 7.0, dtype=np.float32),
        }


def _observation():
    return {
        "observation.state": np.zeros((1, 14), dtype=np.float32),
        **{
            f"observation.images.{name}": np.zeros((1, 48, 64, 3), dtype=np.uint8)
            for name in CAMERA_NAMES
        },
    }


def _batched_observation(batch_size: int = 3):
    return {
        "observation.state": np.zeros((batch_size, 14), dtype=np.float32),
        **{
            f"observation.images.{name}": np.zeros(
                (batch_size, 48, 64, 3), dtype=np.uint8
            )
            for name in CAMERA_NAMES
        },
    }


def test_remote_sft_policy_checks_identity_and_sends_rtc_context() -> None:
    client = _Client()
    policy = RemoteSFTPolicy(
        "ws://127.0.0.1:8080",
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        action_horizon=4,
        client_factory=lambda **_: client,
    )
    policy.reset(17, mode="rtc", exec_horizon=2)
    previous_observation = _observation()
    previous_observation["observation.state"].fill(3.0)
    current_observation = _observation()
    current_observation["observation.state"].fill(5.0)
    policy.infer(previous_observation, "Flatten the shirt.")

    output = policy.infer(current_observation, "Flatten the shirt.")

    assert output.shape == (1, 4, 14)
    assert client.requests[0]["clothdojo.seed"] == 17
    request = client.requests[1]
    assert request["clothdojo.rtc"]["executed_horizon"] == 2
    assert set(request["clothdojo.rtc"]) == {
        "previous_model_actions",
        "previous_state",
        "executed_horizon",
    }
    assert request["clothdojo.rtc"]["previous_model_actions"].shape == (1, 4, 32)
    np.testing.assert_array_equal(
        request["clothdojo.rtc"]["previous_model_actions"],
        7.0,
    )
    assert request["clothdojo.rtc"]["previous_state"].shape == (1, 14)
    np.testing.assert_array_equal(request["clothdojo.rtc"]["previous_state"], 3.0)
    assert request["observation.images.cam_high"].shape == (1, 224, 224, 3)
    assert "clothdojo.seed" not in request


def test_remote_policy_transports_one_batched_request() -> None:
    class BatchedClient(_Client):
        def infer(self, request):
            self.requests.append(request)
            batch = request["observation.state"].shape[0]
            return {
                "actions": np.zeros((batch, 4, 14), dtype=np.float32),
                "model_actions": np.zeros((batch, 4, 32), dtype=np.float32),
            }

    client = BatchedClient()
    policy = RemoteSFTPolicy(
        "ws://127.0.0.1:8080",
        action_horizon=4,
        client_factory=lambda **_: client,
    )
    policy.reset(17, mode="chunk", exec_horizon=2)

    output = policy.infer(_batched_observation(), "Flatten the shirt.")

    assert output.shape == (3, 4, 14)
    assert client.requests[0]["observation.images.cam_high"].shape == (3, 224, 224, 3)


def test_remote_sft_policy_rejects_wrong_checkpoint() -> None:
    with pytest.raises(ValueError, match="identity mismatch"):
        RemoteSFTPolicy(
            "ws://host:8080",
            expected_checkpoint_sha256=CHECKPOINT_SHA,
            action_horizon=4,
            client_factory=lambda **_: _Client(_metadata(checkpoint_sha256="ef" * 32)),
        )


def test_remote_sft_policy_records_server_checkpoint_without_expected_hash() -> None:
    policy = RemoteSFTPolicy(
        "ws://host:8080",
        action_horizon=4,
        client_factory=lambda **_: _Client(),
    )

    assert policy.identity["checkpoint_sha256"] == CHECKPOINT_SHA


def test_remote_policy_accepts_nondefault_lerobot_rtc_parameters() -> None:
    guidance = RTC_GUIDANCE.to_json_dict()
    guidance.update(
        max_guidance_weight=10.0,
        prefix_attention_schedule="linear",
    )
    policy = RemoteSFTPolicy(
        "ws://host:8080",
        action_horizon=4,
        client_factory=lambda **_: _Client(_metadata(rtc_guidance=guidance)),
    )

    assert policy.identity["rtc_guidance"] == guidance


def test_remote_policy_accepts_openpi_without_rtc_metadata() -> None:
    metadata = _metadata(backend="openpi")
    del metadata["clothdojo"]["rtc_guidance"]
    policy = RemoteSFTPolicy(
        "ws://host:8080",
        action_horizon=4,
        client_factory=lambda **_: _Client(metadata),
    )

    assert policy.identity["backend"] == "openpi"


def test_openpi_without_rtc_capability_rejects_rtc_request() -> None:
    metadata = _metadata(backend="openpi")
    del metadata["clothdojo"]["rtc_guidance"]
    policy = RemoteSFTPolicy(
        "ws://host:8080",
        action_horizon=4,
        client_factory=lambda **_: _Client(metadata),
    )
    with pytest.raises(ValueError, match="RTC-capable policy server"):
        policy.reset(17, mode="rtc", exec_horizon=2)


def test_openpi_with_rtc_capability_sends_rtc_request() -> None:
    client = _Client(_metadata(backend="openpi"))
    policy = RemoteSFTPolicy(
        "ws://host:8080",
        action_horizon=4,
        client_factory=lambda **_: client,
    )
    policy.reset(17, mode="rtc", exec_horizon=2)
    policy.infer(_observation(), "Flatten the shirt.")

    policy.infer(_observation(), "Flatten the shirt.")

    assert client.requests[-1]["clothdojo.rtc"]["executed_horizon"] == 2


def test_checkpoint_identity_tracks_executed_weights_instead_of_dcp(tmp_path: Path) -> None:
    dcp = tmp_path / "actor/dcp_checkpoint"
    dcp.mkdir(parents=True)
    (dcp / ".metadata").write_bytes(b"checkpoint metadata")
    shard = dcp / "__0_0.distcp"
    shard.write_bytes(b"first optimizer and weight shards")
    weights = tmp_path / "actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"executed weights")

    first = checkpoint_sha256(tmp_path)
    shard.write_bytes(b"other optimizer and weight shards")
    assert checkpoint_sha256(tmp_path) == first
    weights.write_bytes(b"other executed weights")
    assert checkpoint_sha256(tmp_path) != first


def test_checkpoint_identity_is_independent_of_checkpoint_root(tmp_path: Path) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for root in roots:
        checkpoint = root / "actor/model_state_dict/full_weights.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"weights")

    assert checkpoint_sha256(roots[0]) == checkpoint_sha256(roots[1])


def test_serve_validates_checkpoint_before_constructing_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    norm = tmp_path / "norm_stats.json"
    norm.write_text("{}", encoding="utf-8")
    events: list[str] = []

    def inspect(*args, **kwargs):
        del args, kwargs
        events.append("validated")
        return {
            "schema": "clothdojo_rlinf_openpi_checkpoint",
            "format_version": 1,
            "loader": "rlinf.models.embodiment.openpi",
            "weights_format": "pytorch",
            "weights_sha256": CHECKPOINT_SHA,
            "tensor_count": 813,
            "dtype_counts": {"bfloat16": 691, "float32": 122},
            "stored_dtype_contract": "post_conversion_model_state_dict_exact",
            "expected_state_dict_keys": 813,
            "tied_aliases": [],
        }

    class FakePolicy:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            assert events == ["validated"]
            events.append("model")

    class FakeServer:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def serve_forever(self):
            events.append("served")

    import openpi.serving.websocket_policy_server as server_module

    monkeypatch.setattr(policy_module, "inspect_rlinf_openpi_checkpoint", inspect)
    monkeypatch.setattr(
        policy_module,
        "validate_sft_norm_binding",
        lambda *args, **kwargs: {
            "norm_stats_sha256": NORM_SHA,
            "norm_provenance_sha256": PROVENANCE_SHA,
            "training_manifest_sha256": MANIFEST_SHA,
            "dataset_info_sha256": DATASET_INFO_SHA,
            "dataset_sidecar_sha256": DATASET_SIDECAR_SHA,
        },
    )
    monkeypatch.setattr(policy_module, "_RLinfSFTPolicy", FakePolicy)
    monkeypatch.setattr(server_module, "WebsocketPolicyServer", FakeServer)

    identity = serve_sft_policy(checkpoint, norm, device="cpu")

    assert events == ["validated", "model", "served"]
    assert identity["checkpoint_sha256"] == CHECKPOINT_SHA
    assert identity["checkpoint_validation"]["tensor_count"] == 813
    assert identity["checkpoint_validation"]["strict_reload"] is True


def test_rtc_delta_actions_are_rebased_to_the_current_state() -> None:
    model_actions = np.zeros((1, 4, 32), dtype=np.float32)
    previous_state = np.zeros((1, 14), dtype=np.float32)
    current_state = np.ones((1, 14), dtype=np.float32)
    scale = np.full((14,), 2.0, dtype=np.float32)

    result = _rebase_delta_model_actions(
        model_actions,
        previous_state=previous_state,
        current_state=current_state,
        action_scale=scale,
    )

    expected = np.zeros((14,), dtype=np.float32)
    expected[[*range(6), *range(7, 13)]] = -0.5
    np.testing.assert_allclose(result[0, 0, :14], expected)
    np.testing.assert_array_equal(result[..., 14:], 0.0)


def test_rtc_delta_rebase_is_independent_per_batch_row() -> None:
    model_actions = np.zeros((2, 4, 32), dtype=np.float32)
    previous_state = np.asarray([[0.0] * 14, [3.0] * 14], dtype=np.float32)
    current_state = np.asarray([[2.0] * 14, [1.0] * 14], dtype=np.float32)

    result = _rebase_delta_model_actions(
        model_actions,
        previous_state=previous_state,
        current_state=current_state,
        action_scale=np.ones((14,), dtype=np.float32),
    )

    np.testing.assert_array_equal(result[0, :, :14][:, DELTA_ACTION_MASK], -2.0)
    np.testing.assert_array_equal(result[1, :, :14][:, DELTA_ACTION_MASK], 2.0)


def test_action_quantile_scale_is_loaded_from_norm_stats(tmp_path: Path) -> None:
    q01 = np.arange(14, dtype=np.float32)
    q99 = q01 + 4.0
    (tmp_path / "norm_stats.json").write_text(
        json.dumps({"norm_stats": {"actions": {"q01": q01.tolist(), "q99": q99.tolist()}}}),
        encoding="utf-8",
    )

    np.testing.assert_allclose(_load_action_quantile_scale(tmp_path), 2.0000005)
