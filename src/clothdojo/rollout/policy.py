"""Small RLINF SFT service and strict remote rollout client."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.rollout.artifacts import validate_sft_norm_binding
from clothdojo.rollout.checkpoint import (
    inspect_rlinf_openpi_checkpoint,
    rlinf_openpi_weights_sha256,
    strict_reload_rlinf_openpi_checkpoint,
)
from clothdojo.rollout.engine import ExecutionMode
from clothdojo.rollout.guided_rtc import GuidedRTCConfig, sample_guided_actions
from clothdojo.rollout.newton import POLICY_ACTION_DIM
from clothdojo.rollout.render import CAMERA_NAMES

POLICY_PROTOCOL_VERSION = 2
TRANSPORT_IMAGE_SIZE = (224, 224)
RTC_GUIDANCE = GuidedRTCConfig()
DELTA_ACTION_MASK = np.asarray(
    [True] * 6 + [False] + [True] * 6 + [False],
    dtype=bool,
)


class RemoteSFTPolicy:
    """Connect the rollout process to one identity-checked SFT service."""

    def __init__(
        self,
        endpoint: str,
        *,
        expected_checkpoint_sha256: str | None = None,
        action_horizon: int,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("policy endpoint must be a non-empty string")
        if expected_checkpoint_sha256 is not None:
            _require_sha256(expected_checkpoint_sha256, "expected checkpoint")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        _bypass_proxy_for_loopback(endpoint)
        if client_factory is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            client_factory = WebsocketClientPolicy
        self._client = client_factory(host=endpoint)
        metadata = self._client.get_server_metadata()
        self.identity = _validate_server_metadata(
            metadata,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            action_horizon=action_horizon,
        )
        self._pending_seed: int | None = None
        self._session_id = uuid.uuid4().hex
        self._mode: ExecutionMode | None = None
        self._exec_horizon: int | None = None
        self._previous_model_actions: np.ndarray | None = None
        self._previous_model_state: np.ndarray | None = None

    def reset(
        self,
        seed: int,
        *,
        mode: ExecutionMode,
        exec_horizon: int,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("policy seed must be a non-negative integer")
        if mode not in {"chunk", "rtc"}:
            raise ValueError("mode must be 'chunk' or 'rtc'")
        action_horizon = int(self.identity["action_horizon"])
        if not 1 <= exec_horizon <= action_horizon:
            raise ValueError("exec_horizon must be in [1, action_horizon]")
        if mode == "rtc":
            if exec_horizon == action_horizon:
                raise ValueError("RTC exec_horizon must be smaller than action_horizon")
            guidance = self.identity.get("rtc_guidance")
            if guidance is None:
                raise ValueError("RTC execution requires an RTC-capable policy server")
            server_horizon = int(guidance["execution_horizon"])
            if exec_horizon != server_horizon:
                raise ValueError(
                    "RTC exec_horizon must match the policy server guidance horizon: "
                    f"{exec_horizon} != {server_horizon}"
                )
        self._pending_seed = int(seed)
        self._mode = mode
        self._exec_horizon = int(exec_horizon)
        self._previous_model_actions = None
        self._previous_model_state = None

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
    ) -> np.ndarray:
        if self._mode is None or self._exec_horizon is None:
            raise RuntimeError("policy must be reset before inference")
        request = _prepare_remote_observation(observation)
        if (
            request["observation.state"].shape[0] > 1
            and self.identity.get("batched_inference") is not True
        ):
            raise ValueError("policy server does not advertise batched inference")
        request["prompt"] = prompt
        request["clothdojo.session_id"] = self._session_id
        if self._pending_seed is not None:
            request["clothdojo.seed"] = self._pending_seed
            self._pending_seed = None
        if self._mode == "rtc" and self._previous_model_actions is not None:
            if self._previous_model_state is None:
                raise RuntimeError("RTC replan has no previous model-state reference")
            request["clothdojo.rtc"] = {
                "previous_model_actions": self._previous_model_actions.copy(),
                "previous_state": self._previous_model_state.copy(),
                "executed_horizon": self._exec_horizon,
            }
        result = self._client.infer(request)
        if not isinstance(result, dict):
            raise ValueError("policy server response must be an object")
        actions = _finite_actions(result.get("actions"), POLICY_ACTION_DIM, "actions")
        batch_size = int(request["observation.state"].shape[0])
        expected_prefix = (batch_size, int(self.identity["action_horizon"]))
        if actions.shape[:2] != expected_prefix:
            raise ValueError(
                "policy response batch/horizon "
                f"{actions.shape[:2]} does not match request {expected_prefix}"
            )
        model_actions = result.get("model_actions")
        if model_actions is not None:
            model_actions = _finite_actions(model_actions, None, "model_actions")
            if model_actions.shape[:2] != actions.shape[:2]:
                raise ValueError("model_actions must match the action batch and horizon")
        if self._mode == "rtc":
            if model_actions is None:
                raise ValueError("RTC policy response must include model_actions")
            self._previous_model_actions = model_actions.copy()
            self._previous_model_state = np.asarray(
                request["observation.state"], dtype=np.float32
            ).copy()
        return actions


class _RLinfSFTPolicy:
    """RLINF's released OpenPI model behind ClothDojo's NumPy boundary."""

    def __init__(
        self,
        checkpoint: Path,
        norm_stats: Path,
        *,
        openpi_config: str,
        action_horizon: int,
        num_steps: int,
        autocast: bool,
        device: str,
        rtc_guidance: GuidedRTCConfig,
        expected_checkpoint_sha256: str,
    ) -> None:
        import torch
        from omegaconf import OmegaConf
        from rlinf.models import get_model

        config = OmegaConf.create(
            {
                "precision": None,
                "model_path": str(checkpoint),
                "model_type": "openpi",
                "is_lora": False,
                "load_to_device": False,
                "num_action_chunks": action_horizon,
                "num_steps": num_steps,
                "action_dim": POLICY_ACTION_DIM,
                "add_value_head": False,
                "openpi": {
                    "config_name": openpi_config,
                    "num_images_in_input": len(CAMERA_NAMES),
                    "action_chunk": action_horizon,
                    "action_env_dim": POLICY_ACTION_DIM,
                    "num_steps": num_steps,
                },
                "openpi_data": {
                    "adapt_to_pi": False,
                    "extra_delta_transform": True,
                    "norm_stats_path": str(norm_stats),
                },
            }
        )
        model = get_model(config)
        if model is None:
            raise RuntimeError("pinned RLINF did not register its OpenPI model")
        strict_reload = strict_reload_rlinf_openpi_checkpoint(
            model,
            checkpoint,
            openpi_config=openpi_config,
            action_horizon=action_horizon,
            action_env_dim=POLICY_ACTION_DIM,
            num_steps=num_steps,
        )
        if strict_reload["weights_sha256"] != expected_checkpoint_sha256:
            raise RuntimeError("SFT checkpoint changed between validation and strict model load")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.autocast = bool(autocast)
        self.rtc_guidance = rtc_guidance
        self._delta_action_scale = _load_action_quantile_scale(norm_stats)

    def _guided_infer(
        self,
        env_observation: dict[str, Any],
        previous_actions: Any,
    ) -> tuple[Any, Any]:
        from rlinf.models.embodiment.openpi.openpi_action_model import _model

        processed = self.model.precision_processor(
            self.model.input_transform(
                self.model.obs_processor(env_observation),
                transpose=False,
            )
        )
        observation = _model.Observation.from_dict(processed)
        model_actions = sample_guided_actions(
            self.model,
            observation,
            previous_actions,
            inference_delay=0,
            config=self.rtc_guidance,
        )
        actions = self.model.output_transform(
            {"actions": model_actions, "state": observation.state}
        )["actions"]
        return actions, model_actions

    def reset(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        rtc: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        state = np.asarray(observation["observation.state"], dtype=np.float32)
        main = np.asarray(observation[f"observation.images.{CAMERA_NAMES[0]}"])
        wrists = np.stack(
            [
                observation[f"observation.images.{CAMERA_NAMES[1]}"],
                observation[f"observation.images.{CAMERA_NAMES[2]}"],
            ],
            axis=1,
        )
        env_observation = {
            "main_images": torch.from_numpy(np.array(main, copy=True, order="C")),
            "wrist_images": torch.from_numpy(np.array(wrists, copy=True, order="C")),
            "extra_view_images": None,
            "states": torch.from_numpy(np.array(state, copy=True, order="C")),
            "task_descriptions": [prompt] * len(state),
        }
        previous_remaining = None
        if rtc is not None:
            previous_model_actions = _rebase_delta_model_actions(
                rtc["previous_model_actions"],
                previous_state=rtc["previous_state"],
                current_state=state,
                action_scale=self._delta_action_scale,
            )
            previous_remaining = torch.from_numpy(
                previous_model_actions[:, rtc["executed_horizon"] :]
            ).to(self.device)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.autocast and self.device.type == "cuda",
            ),
        ):
            if previous_remaining is None:
                actions, forward = self.model.predict_action_batch(
                    env_observation,
                    mode="eval",
                    compute_values=False,
                )
                model_actions = forward.get("model_actions")
                if model_actions is None:
                    model_actions = forward["forward_inputs"]["model_action"].reshape(
                        actions.shape[0], actions.shape[1], -1
                    )
            else:
                actions, model_actions = self._guided_infer(
                    env_observation,
                    previous_remaining,
                )
        return _to_numpy(actions), _to_numpy(model_actions)


class _PolicySessionRouter:
    def __init__(self, policy: _RLinfSFTPolicy) -> None:
        self._policy = policy
        self._seeded_sessions: set[str] = set()

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        request = dict(request)
        prompt = request.pop("prompt", None)
        session = request.pop("clothdojo.session_id", None)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(session, str) or not session:
            raise ValueError("clothdojo.session_id must be a non-empty string")
        seed = request.pop("clothdojo.seed", None)
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("clothdojo.seed must be a non-negative integer")
            self._policy.reset(seed)
            self._seeded_sessions.add(session)
        if session not in self._seeded_sessions:
            raise ValueError("a policy session must provide a seed on its first request")
        rtc = _validate_rtc_request(request.pop("clothdojo.rtc", None))
        actions, model_actions = self._policy.infer(request, prompt, rtc)
        return {
            "actions": actions,
            "model_actions": model_actions,
        }


def serve_sft_policy(
    checkpoint: str | Path,
    norm_stats: str | Path,
    *,
    openpi_config: str = "pi05_aloha_robotwin",
    action_horizon: int = 50,
    num_steps: int = 10,
    autocast: bool = True,
    rtc_execution_horizon: int = 10,
    rtc_max_guidance_weight: float = 10.0,
    rtc_prefix_schedule: str = "linear",
    device: str = "cuda:0",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> dict[str, Any]:
    """Load an accepted SFT checkpoint and serve the rollout-only protocol."""

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    norm_path = Path(norm_stats).expanduser().resolve()
    rtc_guidance = GuidedRTCConfig(
        execution_horizon=rtc_execution_horizon,
        max_guidance_weight=rtc_max_guidance_weight,
        prefix_attention_schedule=rtc_prefix_schedule,
    )
    _validate_serve_arguments(
        checkpoint_path,
        norm_path,
        openpi_config=openpi_config,
        action_horizon=action_horizon,
        num_steps=num_steps,
        port=port,
    )
    checkpoint_manifest = inspect_rlinf_openpi_checkpoint(
        checkpoint_path,
        openpi_config=openpi_config,
        action_horizon=action_horizon,
        action_env_dim=POLICY_ACTION_DIM,
        num_steps=num_steps,
    )
    norm_identity = validate_sft_norm_binding(
        checkpoint_path,
        norm_path,
        openpi_config=openpi_config,
        action_horizon=action_horizon,
    )
    identity = {
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "backend": "rlinf-sft",
        "checkpoint_sha256": checkpoint_manifest["weights_sha256"],
        "checkpoint_validation": {
            "strict_reload": True,
            **{
                key: checkpoint_manifest[key]
                for key in (
                    "schema",
                    "format_version",
                    "loader",
                    "weights_format",
                    "tensor_count",
                    "dtype_counts",
                    "stored_dtype_contract",
                    "expected_state_dict_keys",
                    "tied_aliases",
                )
            },
        },
        **norm_identity,
        "openpi_config": openpi_config,
        "action_dim": POLICY_ACTION_DIM,
        "action_horizon": action_horizon,
        "camera_names": list(CAMERA_NAMES),
        "transport_image_size": list(TRANSPORT_IMAGE_SIZE),
        "batched_inference": True,
        "num_steps": num_steps,
        "autocast": bool(autocast),
        "rtc_guidance": rtc_guidance.to_json_dict(),
    }
    policy = _RLinfSFTPolicy(
        checkpoint_path,
        norm_path,
        openpi_config=openpi_config,
        action_horizon=action_horizon,
        num_steps=num_steps,
        autocast=autocast,
        device=device,
        rtc_guidance=rtc_guidance,
        expected_checkpoint_sha256=checkpoint_manifest["weights_sha256"],
    )
    print(json.dumps({"clothdojo": identity}, sort_keys=True), flush=True)
    WebsocketPolicyServer(
        _PolicySessionRouter(policy),
        host=host,
        port=port,
        metadata={"clothdojo": identity},
    ).serve_forever()
    return identity


def checkpoint_sha256(checkpoint: str | Path) -> str:
    return rlinf_openpi_weights_sha256(checkpoint)


def _validate_server_metadata(
    metadata: Any,
    *,
    expected_checkpoint_sha256: str | None,
    action_horizon: int,
) -> dict[str, Any]:
    if not isinstance(metadata, dict) or not isinstance(metadata.get("clothdojo"), dict):
        raise ValueError("policy server is missing ClothDojo metadata")
    identity = dict(metadata["clothdojo"])
    backend = identity.get("backend")
    if backend not in {"rlinf-sft", "openpi"}:
        raise ValueError(f"unsupported policy server backend: {backend!r}")
    expected = {
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "action_dim": POLICY_ACTION_DIM,
        "action_horizon": action_horizon,
        "camera_names": list(CAMERA_NAMES),
        "transport_image_size": list(TRANSPORT_IMAGE_SIZE),
    }
    if expected_checkpoint_sha256 is not None:
        expected["checkpoint_sha256"] = expected_checkpoint_sha256
    mismatches = {
        key: {"expected": value, "actual": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise ValueError(f"policy server identity mismatch: {mismatches}")
    _require_sha256(identity.get("checkpoint_sha256"), "server checkpoint")
    _require_sha256(identity.get("norm_stats_sha256"), "server norm stats")
    if not isinstance(identity.get("openpi_config"), str) or not identity["openpi_config"]:
        raise ValueError("policy server is missing openpi_config")
    if backend == "rlinf-sft":
        _require_sha256(identity.get("norm_provenance_sha256"), "server norm provenance")
        _require_sha256(identity.get("training_manifest_sha256"), "server training manifest")
        _require_sha256(identity.get("dataset_info_sha256"), "server dataset info")
        _require_sha256(identity.get("dataset_sidecar_sha256"), "server dataset sidecar")
    if backend == "rlinf-sft" or identity.get("rtc_guidance") is not None:
        _validate_rtc_guidance_metadata(identity.get("rtc_guidance"))
    return identity


def _validate_rtc_guidance_metadata(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("policy server is missing RTC guidance metadata")
    required = {
        "implementation",
        "execution_horizon",
        "max_guidance_weight",
        "prefix_attention_schedule",
        "delta_rebase",
    }
    if set(value) != required:
        raise ValueError("policy server RTC guidance metadata has invalid fields")
    if value["implementation"] != "lerobot_guided_upstream":
        raise ValueError("policy server RTC implementation is unsupported")
    if value["delta_rebase"] is not True:
        raise ValueError("policy server RTC delta rebase must be enabled")
    GuidedRTCConfig(
        execution_horizon=value["execution_horizon"],
        max_guidance_weight=value["max_guidance_weight"],
        prefix_attention_schedule=value["prefix_attention_schedule"],
    )


def _prepare_remote_observation(observation: dict[str, np.ndarray]) -> dict[str, Any]:
    from openpi_client import image_tools

    required = {"observation.state", *(f"observation.images.{name}" for name in CAMERA_NAMES)}
    if set(observation) != required:
        raise ValueError(
            "policy observation fields differ: "
            f"missing={sorted(required - set(observation))}, "
            f"extra={sorted(set(observation) - required)}"
        )
    state = np.asarray(observation["observation.state"], dtype=np.float32)
    if (
        state.ndim != 2
        or state.shape[0] < 1
        or state.shape[1] != POLICY_ACTION_DIM
        or not np.isfinite(state).all()
    ):
        raise ValueError("observation.state must be finite float32[batch, 14]")
    batch_size = int(state.shape[0])

    def prepare(name: str) -> tuple[str, np.ndarray]:
        key = f"observation.images.{name}"
        image = np.asarray(observation[key])
        if image.ndim != 4 or image.shape[0] != batch_size or image.shape[-1] != 3:
            raise ValueError(f"{key} must have shape (batch, H, W, 3)")
        resized = image_tools.resize_with_pad(image, *TRANSPORT_IMAGE_SIZE)
        converted = image_tools.convert_to_uint8(resized)
        if converted.dtype != np.uint8:
            raise TypeError(f"{key} could not be converted to uint8")
        return key, np.ascontiguousarray(converted)

    with ThreadPoolExecutor(max_workers=len(CAMERA_NAMES)) as executor:
        images = dict(executor.map(prepare, CAMERA_NAMES))
    return {"observation.state": np.ascontiguousarray(state), **images}


def _validate_rtc_request(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    required = {
        "previous_model_actions",
        "previous_state",
        "executed_horizon",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("clothdojo.rtc has invalid fields")
    previous = _finite_actions(value["previous_model_actions"], None, "previous_model_actions")
    previous_state = np.asarray(value["previous_state"], dtype=np.float32)
    if (
        previous_state.shape != (previous.shape[0], POLICY_ACTION_DIM)
        or not np.isfinite(previous_state).all()
    ):
        raise ValueError(
            "clothdojo.rtc.previous_state must be finite float32[batch, 14]"
        )
    result = {
        "previous_model_actions": previous,
        "previous_state": np.ascontiguousarray(previous_state),
    }
    executed_horizon = value["executed_horizon"]
    if (
        isinstance(executed_horizon, bool)
        or not isinstance(executed_horizon, int)
        or not 1 <= executed_horizon < previous.shape[1]
    ):
        raise ValueError(
            "clothdojo.rtc.executed_horizon must be in [1, previous horizon)"
        )
    result["executed_horizon"] = executed_horizon
    return result


def _load_action_quantile_scale(norm_stats: Path) -> np.ndarray:
    source = norm_stats / "norm_stats.json" if norm_stats.is_dir() else norm_stats
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        action_stats = payload["norm_stats"]["actions"]
        q01 = np.asarray(action_stats["q01"], dtype=np.float32)
        q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read action quantiles from {source}: {exc}") from exc
    if q01.shape != (POLICY_ACTION_DIM,) or q99.shape != q01.shape:
        raise ValueError("action normalization quantiles must both have shape (14,)")
    scale = (q99 - q01 + 1.0e-6) / 2.0
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("action normalization quantile scales must be finite and positive")
    return np.ascontiguousarray(scale)


def _rebase_delta_model_actions(
    previous_model_actions: np.ndarray,
    *,
    previous_state: np.ndarray,
    current_state: np.ndarray,
    action_scale: np.ndarray,
) -> np.ndarray:
    """Keep old absolute joint targets fixed when the delta-action base moves."""

    actions = np.asarray(previous_model_actions, dtype=np.float32)
    previous = np.asarray(previous_state, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    scale = np.asarray(action_scale, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[2] < POLICY_ACTION_DIM:
        raise ValueError("previous model actions must have shape (batch, horizon, >=14)")
    expected_state = (actions.shape[0], POLICY_ACTION_DIM)
    if previous.shape != expected_state or current.shape != previous.shape:
        raise ValueError("previous and current policy states must have shape (batch, 14)")
    if scale.shape != (POLICY_ACTION_DIM,) or np.any(scale <= 0.0):
        raise ValueError("action scale must be positive with shape (14,)")
    shift = np.zeros_like(previous)
    shift[:, DELTA_ACTION_MASK] = (
        previous[:, DELTA_ACTION_MASK] - current[:, DELTA_ACTION_MASK]
    ) / scale[DELTA_ACTION_MASK]
    result = actions.copy()
    result[..., :POLICY_ACTION_DIM] += shift[:, None, :]
    return np.ascontiguousarray(result)


def _finite_actions(value: Any, action_dim: int | None, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if (
        result.ndim != 3
        or result.shape[0] < 1
        or result.shape[1] < 1
        or (action_dim is not None and result.shape[2] != action_dim)
        or not np.isfinite(result).all()
    ):
        suffix = "" if action_dim is None else f" with action_dim={action_dim}"
        raise ValueError(f"{name} must be finite with shape (batch, horizon, dim){suffix}")
    return np.ascontiguousarray(result)


def _validate_serve_arguments(
    checkpoint: Path,
    norm_stats: Path,
    *,
    openpi_config: str,
    action_horizon: int,
    num_steps: int,
    port: int,
) -> None:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"SFT checkpoint directory does not exist: {checkpoint}")
    norm_file = norm_stats / "norm_stats.json" if norm_stats.is_dir() else norm_stats
    if not norm_file.is_file():
        raise FileNotFoundError(f"norm_stats.json does not exist: {norm_stats}")
    if not openpi_config.strip():
        raise ValueError("openpi_config must be non-empty")
    if min(action_horizon, num_steps, port) < 1:
        raise ValueError("action_horizon, num_steps and port must be positive")


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} SHA-256 must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} SHA-256 must contain 64 hexadecimal characters") from exc


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _bypass_proxy_for_loopback(endpoint: str) -> None:
    if not any(host in endpoint for host in ("127.0.0.1", "localhost", "[::1]")):
        return
    for name in ("NO_PROXY", "no_proxy"):
        values = [value for value in os.environ.get(name, "").split(",") if value]
        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in values:
                values.append(host)
        os.environ[name] = ",".join(values)
