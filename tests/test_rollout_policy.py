from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.rollout.engine import RTCContext
from clothdojo.rollout.policy import (
    RTC_GUIDANCE,
    RemoteSFTPolicy,
    _load_action_quantile_scale,
    _rebase_delta_model_actions,
    checkpoint_sha256,
)
from clothdojo.rollout.render import CAMERA_NAMES

CHECKPOINT_SHA = "ab" * 32
NORM_SHA = "cd" * 32


def _metadata(**changes):
    identity = {
        "protocol_version": 1,
        "backend": "rlinf-sft",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "norm_stats_sha256": NORM_SHA,
        "openpi_config": "pi05_aloha_robotwin",
        "action_dim": 14,
        "action_horizon": 4,
        "camera_names": list(CAMERA_NAMES),
        "transport_image_size": [224, 224],
        "num_steps": 10,
        "rtc_guidance": RTC_GUIDANCE.to_json_dict(),
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
            "model_actions": np.zeros((1, 4, 32), dtype=np.float32),
        }


def _observation():
    return {
        "observation.state": np.zeros((1, 14), dtype=np.float32),
        **{
            f"observation.images.{name}": np.zeros((1, 48, 64, 3), dtype=np.uint8)
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
    policy.reset(17)
    policy.infer(_observation(), "Flatten the shirt.")
    context = RTCContext(
        previous_model_actions=np.ones((1, 4, 32), dtype=np.float32),
        executed_horizon=2,
        predicted_delay_steps=3,
        chunk_id=3,
    )

    output = policy.infer(_observation(), "Flatten the shirt.", rtc=context)

    assert output.actions.shape == (1, 4, 14)
    assert client.requests[0]["clothdojo.seed"] == 17
    request = client.requests[1]
    assert request["clothdojo.rtc"]["executed_horizon"] == 2
    assert request["clothdojo.rtc"]["delay_steps"] == 3
    assert request["clothdojo.rtc"]["previous_state"].shape == (1, 14)
    assert request["observation.images.cam_high"].shape == (1, 224, 224, 3)
    assert "clothdojo.seed" not in request


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


def test_checkpoint_identity_covers_rlinf_metadata_and_weight_shards(tmp_path: Path) -> None:
    checkpoint = tmp_path / "actor" / "dcp_checkpoint"
    metadata = checkpoint / ".metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"checkpoint metadata")
    shard = checkpoint / "__0_0.distcp"
    shard.write_bytes(b"first weights")

    first = checkpoint_sha256(tmp_path)
    shard.write_bytes(b"other weights")
    second = checkpoint_sha256(tmp_path)

    assert first != second


def test_checkpoint_identity_is_independent_of_checkpoint_root(tmp_path: Path) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for root in roots:
        checkpoint = root / "actor" / "dcp_checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / ".metadata").write_bytes(b"metadata")
        (checkpoint / "__0_0.distcp").write_bytes(b"weights")

    assert checkpoint_sha256(roots[0]) == checkpoint_sha256(roots[1])


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


def test_action_quantile_scale_is_loaded_from_norm_stats(tmp_path: Path) -> None:
    q01 = np.arange(14, dtype=np.float32)
    q99 = q01 + 4.0
    (tmp_path / "norm_stats.json").write_text(
        json.dumps(
            {"norm_stats": {"actions": {"q01": q01.tolist(), "q99": q99.tolist()}}}
        ),
        encoding="utf-8",
    )

    np.testing.assert_allclose(_load_action_quantile_scale(tmp_path), 2.0000005)
