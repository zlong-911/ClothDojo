"""Checked RLINF SFT/CFG services and one strict remote rollout client."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

from clothdojo.rendering import CAMERA_NAMES, POLICY_TRANSPORT_SIZE
from clothdojo.rollout.artifacts import (
    validate_cfg_checkpoint_binding,
    validate_jax_reference_binding,
    validate_sft_norm_binding,
)
from clothdojo.rollout.checkpoint import (
    inspect_rlinf_openpi_checkpoint,
    rlinf_openpi_weights_sha256,
    strict_reload_matching_rlinf_openpi_checkpoint,
)
from clothdojo.rollout.engine import ExecutionMode
from clothdojo.rollout.guided_rtc import (
    GuidedRTCConfig,
    sample_guided_cfg_openpi_rlinf_actions,
    sample_guided_openpi_rlinf_actions,
)
from clothdojo.rollout.newton import POLICY_ACTION_DIM

POLICY_PROTOCOL_VERSION = 2
IMAGE_PREPROCESSING_POLICIES = ("none", "training_fixed_crop")
RTC_GUIDANCE = GuidedRTCConfig()
DELTA_ACTION_MASK = np.asarray(
    [True] * 6 + [False] + [True] * 6 + [False],
    dtype=bool,
)


def _validate_image_preprocessing(value: Any) -> None:
    if value not in IMAGE_PREPROCESSING_POLICIES:
        raise ValueError(
            f"image_preprocessing must be one of {IMAGE_PREPROCESSING_POLICIES}, got {value!r}"
        )


def _preprocess_inference_images(observation: Any, image_preprocessing: str) -> Any:
    """Replay historical FSDP BF16 image preprocessing without changing model mode."""

    _validate_image_preprocessing(image_preprocessing)
    if image_preprocessing == "none":
        return observation

    import torch
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module

    if any(image.device.type != "cuda" for image in observation.images.values()):
        raise ValueError("training_fixed_crop requires CUDA to reproduce BF16 image operations")
    # FSDP cast_root_forward_inputs rounds images before augmentation. Reuse
    # the pinned pure preprocessing function, including rotate(0) and the wrist
    # range round-trip: both matter at BF16 precision. rng=None is deterministic;
    # this does not enable model training, dropout, or random augmentation.
    prepared = dataclasses.replace(
        observation,
        images={key: image.to(torch.bfloat16) for key, image in observation.images.items()},
    )
    return model_module.preprocess_observation(prepared, train=True, rng=None)


def _install_inference_image_preprocessing(model: Any, image_preprocessing: str) -> None:
    """Wrap the single observation boundary shared by SFT, CFG and guided RTC."""

    _validate_image_preprocessing(image_preprocessing)
    method = model._observation_dict_to_device
    installed = getattr(method, "_clothdojo_image_preprocessing", None)
    if installed is not None:
        if installed != image_preprocessing:
            raise ValueError("inference image preprocessing is already installed differently")
        return
    if image_preprocessing == "none":
        return
    original = method.__func__

    @wraps(original)
    def prepare(instance: Any, processed: dict[str, Any]) -> Any:
        return _preprocess_inference_images(original(instance, processed), image_preprocessing)

    prepare._clothdojo_image_preprocessing = image_preprocessing
    model._observation_dict_to_device = MethodType(prepare, model)


class RemotePolicy:
    """Connect rollout to one identity-checked SFT or CFG policy service."""

    def __init__(
        self,
        endpoint: str,
        *,
        expected_checkpoint_sha256: str | None = None,
        expected_image_preprocessing: str = "none",
        action_horizon: int,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("policy endpoint must be a non-empty string")
        if expected_checkpoint_sha256 is not None:
            _require_sha256(expected_checkpoint_sha256, "expected checkpoint")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        _validate_image_preprocessing(expected_image_preprocessing)
        _bypass_proxy_for_loopback(endpoint)
        if client_factory is None:
            self._client = _connect_remote_policy(endpoint)
        else:
            self._client = client_factory(host=endpoint)
        metadata = self._client.get_server_metadata()
        self.identity = _validate_server_metadata(
            metadata,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            action_horizon=action_horizon,
            expected_image_preprocessing=expected_image_preprocessing,
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
        self._pending_seed = int(seed)
        self._mode = mode
        self._exec_horizon = int(exec_horizon)
        self._previous_model_actions = None
        self._previous_model_state = None

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        *,
        inference_delay_steps: int = 0,
    ) -> np.ndarray:
        if self._mode is None or self._exec_horizon is None:
            raise RuntimeError("policy must be reset before inference")
        if (
            isinstance(inference_delay_steps, bool)
            or not isinstance(inference_delay_steps, int)
            or inference_delay_steps < 0
        ):
            raise ValueError("inference_delay_steps must be a non-negative integer")
        if self._mode == "chunk" and inference_delay_steps:
            raise ValueError("chunk inference does not accept an RTC delay")
        if self._mode == "rtc":
            guidance_horizon = int(self.identity["rtc_guidance"]["guidance_horizon"])
            if inference_delay_steps > guidance_horizon:
                raise ValueError(
                    "RTC inference delay must not exceed the policy guidance horizon: "
                    f"{inference_delay_steps} > {guidance_horizon}"
                )
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
                "inference_delay_steps": inference_delay_steps,
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
            model_actions = _finite_actions(
                model_actions,
                POLICY_ACTION_DIM,
                "model_actions",
            )
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


def _connect_remote_policy(endpoint: str) -> Any:
    """Connect without treating a long synchronous inference as a dead peer."""

    import websockets.sync.client
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    class _LongInferenceClient(WebsocketClientPolicy):
        def _wait_for_server(self) -> tuple[Any, dict[str, Any]]:
            headers = (
                {"Authorization": f"Api-Key {self._api_key}"}
                if self._api_key
                else None
            )
            connection = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                additional_headers=headers,
                ping_timeout=None,
            )
            metadata = msgpack_numpy.unpackb(connection.recv())
            return connection, metadata

    return _LongInferenceClient(host=endpoint)


class _RLinfSFTPolicy:
    """RLinf's current ``openpi_rlinf`` model behind the NumPy boundary."""

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
        image_preprocessing: str = "none",
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        _configure_openpi_data_home()
        from rlinf.models import get_model

        config = OmegaConf.create(
            {
                "precision": "bf16",
                "model_path": str(checkpoint),
                "model_type": "openpi_rlinf",
                "pi05": True,
                "is_lora": False,
                "lora_rank": 32,
                "load_to_device": False,
                "num_action_chunks": action_horizon,
                "num_steps": num_steps,
                "action_dim": POLICY_ACTION_DIM,
                "add_value_head": False,
                "openpi": {
                    "task": "eval",
                    "config_name": openpi_config,
                    "model_action_dim": 32,
                    "paligemma_variant": "gemma_2b",
                    "action_expert_variant": "gemma_300m",
                    "max_token_len": 200,
                    "use_rlt": False,
                },
                "openpi_data": {
                    "adapt_to_pi": False,
                    "extra_delta_transform": True,
                    "default_prompt": "flatten the shirt",
                    "norm_stats_path": str(norm_stats),
                },
            }
        )
        model = get_model(config)
        if model is None:
            raise RuntimeError("pinned RLINF did not register openpi_rlinf")
        if rlinf_openpi_weights_sha256(checkpoint) != expected_checkpoint_sha256:
            raise RuntimeError("SFT checkpoint changed while the model was loading")
        self.device = torch.device(device)
        if autocast:
            raise ValueError("openpi_rlinf uses native BF16; autocast is not a mode")
        self.model = model.to(self.device).eval().requires_grad_(False)
        _install_inference_image_preprocessing(self.model, image_preprocessing)
        self.autocast = False
        self.rtc_guidance = rtc_guidance
        self.num_steps = int(num_steps)
        self._delta_action_scale = _load_action_quantile_scale(norm_stats)

    def _guided_infer(
        self,
        env_observation: dict[str, Any],
        previous_actions: Any,
        *,
        inference_delay_steps: int,
    ) -> tuple[Any, Any]:
        processed = self.model.input_transform(
            self.model._repack_env_obs(env_observation),
            transpose=False,
        )
        observation = self.model._observation_dict_to_device(processed)
        model_actions = sample_guided_openpi_rlinf_actions(
            self.model.model,
            observation,
            previous_actions,
            inference_delay=inference_delay_steps,
            num_steps=self.num_steps,
            config=self.rtc_guidance,
        )
        actions = self.model.output_transform(
            {"actions": model_actions, "state": observation.state}
        )["actions"]
        return actions, model_actions

    @staticmethod
    def _model_actions_from_forward(actions: Any, forward: dict[str, Any]) -> Any:
        return forward["forward_inputs"]["model_action"].reshape(
            actions.shape[0], actions.shape[1], -1
        )

    def reset(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def prepare_model_observation(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
    ) -> Any:
        """Apply the exact serving input path and return the model observation."""

        env_observation = self._to_env_observation(observation, prompt)
        processed = self.model.input_transform(
            self.model._repack_env_obs(env_observation),
            transpose=False,
        )
        return self.model._observation_dict_to_device(processed)

    @staticmethod
    def _to_env_observation(
        observation: dict[str, np.ndarray],
        prompt: str,
    ) -> dict[str, Any]:
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
        return {
            "main_images": torch.from_numpy(np.array(main, copy=True, order="C")),
            "wrist_images": torch.from_numpy(np.array(wrists, copy=True, order="C")),
            "extra_view_images": None,
            "states": torch.from_numpy(np.array(state, copy=True, order="C")),
            "task_descriptions": [prompt] * len(state),
        }

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        rtc: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        state = np.asarray(observation["observation.state"], dtype=np.float32)
        env_observation = self._to_env_observation(observation, prompt)
        previous_remaining = None
        if rtc is not None:
            inference_delay_steps = rtc["inference_delay_steps"]
            if inference_delay_steps > self.rtc_guidance.guidance_horizon:
                raise ValueError(
                    "RTC inference delay exceeds the policy guidance horizon"
                )
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
                model_actions = self._model_actions_from_forward(actions, forward)
            else:
                actions, model_actions = self._guided_infer(
                    env_observation,
                    previous_remaining,
                    inference_delay_steps=inference_delay_steps,
                )
        return (
            _to_numpy(actions),
            _to_numpy(model_actions[..., :POLICY_ACTION_DIM]),
        )


class _RLinfCFGPolicy(_RLinfSFTPolicy):
    """RLINF's CFG OpenPI model behind the same rollout boundary."""

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
        guidance_scale: float,
        guidance_type: str,
        positive_only_conditional: bool,
        expected_checkpoint_sha256: str,
        image_preprocessing: str = "none",
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        _configure_openpi_data_home()
        from rlinf.models import get_model

        from clothdojo.training.rlinf_cfg import MODEL_TYPE, enable

        enable()
        config = OmegaConf.create(
            {
                "precision": None,
                "model_path": str(checkpoint),
                "model_type": MODEL_TYPE,
                "expected_weights_sha256": expected_checkpoint_sha256,
                "pi05": True,
                "use_proprio": True,
                "num_steps": num_steps,
                "is_lora": False,
                "lora_rank": 32,
                "load_to_device": False,
                "num_action_chunks": action_horizon,
                "action_dim": POLICY_ACTION_DIM,
                "add_value_head": False,
                "openpi": {
                    "task": "cfg",
                    "config_name": openpi_config,
                    "num_images_in_input": len(CAMERA_NAMES),
                    "action_chunk": action_horizon,
                    "action_env_dim": POLICY_ACTION_DIM,
                    "num_steps": num_steps,
                    "model_action_dim": 32,
                    "paligemma_variant": "gemma_2b",
                    "action_expert_variant": "gemma_300m",
                    "discrete_state_input": True,
                    "max_token_len": 200,
                    "use_rlt": False,
                    "cfgrl_guidance_scale": guidance_scale,
                    "unconditional_prob": 0.1,
                    "train_expert_only": False,
                    "guidance_type": guidance_type,
                    "positive_only_conditional": positive_only_conditional,
                },
                "openpi_data": {
                    "adapt_to_pi": False,
                    "extra_delta_transform": True,
                    "default_prompt": "flatten the shirt",
                    "norm_stats_path": str(norm_stats),
                },
            }
        )
        model = get_model(config)
        if model is None:
            raise RuntimeError("pinned RLINF did not register its CFG OpenPI model")
        strict_reload = getattr(model, "clothdojo_checkpoint_validation", None)
        if not isinstance(strict_reload, dict):
            strict_reload = strict_reload_matching_rlinf_openpi_checkpoint(
                model,
                checkpoint,
            )
        if strict_reload["weights_sha256"] != expected_checkpoint_sha256:
            raise RuntimeError("CFG checkpoint changed while the model was loading")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        _install_inference_image_preprocessing(self.model, image_preprocessing)
        self.autocast = bool(autocast)
        self.rtc_guidance = rtc_guidance
        self.num_steps = int(num_steps)
        self._delta_action_scale = _load_action_quantile_scale(norm_stats)
        self.checkpoint_validation = strict_reload

    def _guided_infer(
        self,
        env_observation: dict[str, Any],
        previous_actions: Any,
        *,
        inference_delay_steps: int,
    ) -> tuple[Any, Any]:
        unconditional, conditional = self.model.prepare_cfg_observations(
            env_observation
        )
        model_actions = sample_guided_cfg_openpi_rlinf_actions(
            self.model.model,
            unconditional,
            conditional,
            previous_actions,
            inference_delay=inference_delay_steps,
            num_steps=self.num_steps,
            guidance_scale=self.model.routing.guidance_scale,
            config=self.rtc_guidance,
        )
        actions = self.model.output_transform(
            {"actions": model_actions, "state": unconditional.state}
        )["actions"]
        return actions, model_actions

    @staticmethod
    def _model_actions_from_forward(actions: Any, forward: dict[str, Any]) -> Any:
        del actions
        return forward["forward_inputs"]["raw_actions"]


class _OpenPIJAXPolicy:
    """Original OpenPI/JAX policy behind ClothDojo's batched rollout boundary."""

    def __init__(
        self,
        checkpoint: Path,
        norm_stats: Path,
        training_config: Path,
        *,
        action_horizon: int,
        num_steps: int,
    ) -> None:
        import jax
        import jax.numpy as jnp
        from openpi import transforms
        from openpi.models import model as model_module
        from openpi.shared import nnx_utils
        from openpi.training import checkpoints

        from clothdojo.training.jax_sft import (
            build_openpi_train_config,
            compose_jax_sft_spec,
        )

        spec = compose_jax_sft_spec(training_config)
        if not spec.norm_stats.samefile(norm_stats):
            raise ValueError("JAX serving config norm_stats differs from --norm-stats")
        config = build_openpi_train_config(spec)
        if config.model.action_horizon != action_horizon:
            raise ValueError("JAX model action horizon differs from serving horizon")
        data_config = config.data.create(config.assets_dirs, config.model)
        norm_dir = norm_stats if norm_stats.is_dir() else norm_stats.parent
        norm_values = checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
        self._input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(spec.default_prompt),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_values,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = transforms.compose(
            [
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    norm_values,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.data_transforms.outputs,
            ]
        )
        self._model = config.model.load(
            model_module.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
        )
        self._model.eval()
        self._sample_actions = nnx_utils.module_jit(self._model.sample_actions)
        self._rng = jax.random.key(0)
        self._num_steps = int(num_steps)

    def reset(self, seed: int) -> None:
        import jax

        self._rng = jax.random.key(seed)

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        rtc: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rtc is not None:
            raise ValueError("the OpenPI/JAX reference supports chunk execution only")

        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_module

        states = np.asarray(observation["observation.state"], dtype=np.float32)
        transformed = []
        for index, state in enumerate(states):
            item = {
                "state": state,
                "images": {
                    name: np.moveaxis(
                        observation[f"observation.images.{name}"][index], -1, 0
                    )
                    for name in CAMERA_NAMES
                },
                "prompt": prompt,
            }
            transformed.append(self._input_transform(item))
        inputs = jax.tree.map(
            lambda *values: jnp.stack(values),
            *transformed,
        )
        model_observation = model_module.Observation.from_dict(inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        model_actions = np.asarray(
            self._sample_actions(
                sample_rng,
                model_observation,
                num_steps=self._num_steps,
            ),
            dtype=np.float32,
        )
        actions = np.stack(
            [
                np.asarray(
                    self._output_transform(
                        {"actions": action, "state": state}
                    )["actions"],
                    dtype=np.float32,
                )
                for action, state in zip(
                    model_actions,
                    np.asarray(model_observation.state),
                    strict=True,
                )
            ]
        )
        return actions, model_actions[..., :POLICY_ACTION_DIM]


class _PolicySessionRouter:
    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self._active_session: str | None = None
        self._lock = threading.Lock()

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        # One endpoint is assigned exclusively to one rollout worker. Serialize
        # access anyway so an accidental second client cannot interleave RNG and
        # RTC history in the stateful backend.
        with self._lock:
            return self._infer(request)

    def _infer(self, request: dict[str, Any]) -> dict[str, Any]:
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
            self._active_session = session
        if session != self._active_session:
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
    image_preprocessing: str = "none",
    rtc_guidance_horizon: int = 10,
    rtc_max_guidance_weight: float = 10.0,
    rtc_prefix_schedule: str = "linear",
    device: str = "cuda:0",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> dict[str, Any]:
    """Load an accepted SFT checkpoint and serve the rollout-only protocol."""

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    _validate_image_preprocessing(image_preprocessing)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    norm_path = Path(norm_stats).expanduser().resolve()
    rtc_guidance = GuidedRTCConfig(
        guidance_horizon=rtc_guidance_horizon,
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
    if rtc_guidance_horizon > action_horizon:
        raise ValueError("RTC guidance horizon must not exceed the action horizon")
    checkpoint_manifest = inspect_rlinf_openpi_checkpoint(
        checkpoint_path,
        action_horizon=action_horizon,
    )
    norm_identity = validate_sft_norm_binding(
        checkpoint_path,
        norm_path,
        openpi_config=openpi_config,
        action_horizon=action_horizon,
        weights_sha256=checkpoint_manifest["weights_sha256"],
    )
    identity = {
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "backend": "rlinf-sft",
        "checkpoint_sha256": checkpoint_manifest["weights_sha256"],
        "checkpoint_validation": {
            "exact_layout": True,
            **{
                key: checkpoint_manifest[key]
                for key in (
                    "schema",
                    "format_version",
                    "loader",
                    "weights_format",
                    "state_dict_layout",
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
        "transport_image_size": list(POLICY_TRANSPORT_SIZE),
        "batched_inference": True,
        "num_steps": num_steps,
        "inference_precision": "native_bfloat16",
        "image_preprocessing": image_preprocessing,
        "rtc_guidance": rtc_guidance.to_json_dict(),
    }
    policy = _RLinfSFTPolicy(
        checkpoint_path,
        norm_path,
        openpi_config=openpi_config,
        action_horizon=action_horizon,
        num_steps=num_steps,
        autocast=False,
        device=device,
        rtc_guidance=rtc_guidance,
        expected_checkpoint_sha256=checkpoint_manifest["weights_sha256"],
        image_preprocessing=image_preprocessing,
    )
    print(json.dumps({"clothdojo": identity}, sort_keys=True), flush=True)
    WebsocketPolicyServer(
        _PolicySessionRouter(policy),
        host=host,
        port=port,
        metadata={"clothdojo": identity},
    ).serve_forever()
    return identity


def serve_jax_policy(
    checkpoint: str | Path,
    norm_stats: str | Path,
    training_config: str | Path,
    *,
    action_horizon: int = 50,
    num_steps: int = 10,
    image_preprocessing: str = "none",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> dict[str, Any]:
    """Serve the original OpenPI/JAX EMA as a chunk-only reference."""

    _validate_image_preprocessing(image_preprocessing)
    if image_preprocessing != "none":
        raise ValueError("OpenPI/JAX only supports image_preprocessing='none'")

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    norm_path = Path(norm_stats).expanduser().resolve()
    config_path = Path(training_config).expanduser().resolve()
    if not (checkpoint_path / "params" / "_METADATA").is_file():
        raise FileNotFoundError("JAX checkpoint has no Orbax params metadata")
    if not config_path.is_file():
        raise FileNotFoundError(f"JAX training config does not exist: {config_path}")
    if action_horizon < 1 or num_steps < 1 or not 1 <= port <= 65535:
        raise ValueError("invalid JAX serving horizon, sampling steps, or port")
    norm_file = norm_path / "norm_stats.json" if norm_path.is_dir() else norm_path
    if not norm_file.is_file():
        raise FileNotFoundError(f"normalization stats do not exist: {norm_file}")

    checkpoint_identity = _tree_sha256(checkpoint_path / "params")
    artifact_identity = validate_jax_reference_binding(
        checkpoint_path,
        norm_path,
        config_path,
        action_horizon=action_horizon,
    )
    identity = {
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "backend": "openpi",
        "checkpoint_sha256": checkpoint_identity,
        **artifact_identity,
        "openpi_config": "pi05_aloha_robotwin",
        "action_dim": POLICY_ACTION_DIM,
        "action_horizon": action_horizon,
        "camera_names": list(CAMERA_NAMES),
        "transport_image_size": list(POLICY_TRANSPORT_SIZE),
        "batched_inference": True,
        "num_steps": num_steps,
        "inference_precision": "bfloat16",
        "image_preprocessing": image_preprocessing,
        "rtc_guidance": None,
    }
    policy = _OpenPIJAXPolicy(
        checkpoint_path,
        norm_path,
        config_path,
        action_horizon=action_horizon,
        num_steps=num_steps,
    )
    if _tree_sha256(checkpoint_path / "params") != checkpoint_identity:
        raise RuntimeError("JAX checkpoint changed while the model was loading")
    print(json.dumps({"clothdojo": identity}, sort_keys=True), flush=True)
    WebsocketPolicyServer(
        _PolicySessionRouter(policy),
        host=host,
        port=port,
        metadata={"clothdojo": identity},
    ).serve_forever()
    return identity


def serve_cfg_policy(
    checkpoint: str | Path,
    norm_stats: str | Path,
    *,
    openpi_config: str = "pi05_aloha_robotwin",
    action_horizon: int = 50,
    num_steps: int = 5,
    image_preprocessing: str = "none",
    guidance_scale: float = 1.0,
    guidance_type: str = "positive",
    positive_only_conditional: bool = False,
    autocast: bool = True,
    rtc_guidance_horizon: int = 10,
    rtc_max_guidance_weight: float = 10.0,
    rtc_prefix_schedule: str = "linear",
    device: str = "cuda:0",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> dict[str, Any]:
    """Serve an RLINF CFG checkpoint for chunk or composed CFG+RTC rollout."""

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    _validate_image_preprocessing(image_preprocessing)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    norm_path = Path(norm_stats).expanduser().resolve()
    _validate_serve_arguments(
        checkpoint_path,
        norm_path,
        openpi_config=openpi_config,
        action_horizon=action_horizon,
        num_steps=num_steps,
        port=port,
    )
    if not np.isfinite(guidance_scale) or guidance_scale < 0.0:
        raise ValueError("CFG guidance_scale must be finite and non-negative")
    if guidance_type not in {"positive", "negative", "no_guide"}:
        raise ValueError("CFG guidance_type must be positive, negative or no_guide")
    if positive_only_conditional and guidance_type == "negative":
        raise ValueError("negative CFG guidance requires negative-conditioned training")
    rtc_guidance = GuidedRTCConfig(
        guidance_horizon=rtc_guidance_horizon,
        max_guidance_weight=rtc_max_guidance_weight,
        prefix_attention_schedule=rtc_prefix_schedule,
    )
    if rtc_guidance_horizon > action_horizon:
        raise ValueError("RTC guidance horizon must not exceed the action horizon")
    weights_sha256 = rlinf_openpi_weights_sha256(checkpoint_path)
    cfg_binding = validate_cfg_checkpoint_binding(
        checkpoint_path,
        norm_path,
        openpi_config=openpi_config,
        action_horizon=action_horizon,
        action_dim=POLICY_ACTION_DIM,
        positive_only_conditional=positive_only_conditional,
    )
    policy = _RLinfCFGPolicy(
        checkpoint_path,
        norm_path,
        openpi_config=openpi_config,
        action_horizon=action_horizon,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        guidance_type=guidance_type,
        positive_only_conditional=positive_only_conditional,
        autocast=autocast,
        device=device,
        rtc_guidance=rtc_guidance,
        expected_checkpoint_sha256=weights_sha256,
        image_preprocessing=image_preprocessing,
    )
    identity = {
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "backend": "rlinf-cfg",
        "checkpoint_sha256": weights_sha256,
        "checkpoint_validation": {
            "strict_reload": True,
            "tensor_count": policy.checkpoint_validation["tensor_count"],
            "tied_aliases": policy.checkpoint_validation["tied_aliases"],
        },
        **cfg_binding,
        "openpi_config": openpi_config,
        "action_dim": POLICY_ACTION_DIM,
        "action_horizon": action_horizon,
        "camera_names": list(CAMERA_NAMES),
        "transport_image_size": list(POLICY_TRANSPORT_SIZE),
        "batched_inference": True,
        "num_steps": num_steps,
        "autocast": bool(autocast),
        "image_preprocessing": image_preprocessing,
        "cfg_guidance": {
            "type": guidance_type,
            "scale": float(guidance_scale),
            "positive_only_conditional": bool(positive_only_conditional),
        },
        "rtc_guidance": rtc_guidance.to_json_dict(),
    }
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"checkpoint tree has no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validate_server_metadata(
    metadata: Any,
    *,
    expected_checkpoint_sha256: str | None,
    action_horizon: int,
    expected_image_preprocessing: str = "none",
) -> dict[str, Any]:
    if not isinstance(metadata, dict) or not isinstance(metadata.get("clothdojo"), dict):
        raise ValueError("policy server is missing ClothDojo metadata")
    identity = dict(metadata["clothdojo"])
    # Servers predating this option had the deterministic no-crop eval path.
    identity.setdefault("image_preprocessing", "none")
    backend = identity.get("backend")
    if backend not in {"rlinf-sft", "rlinf-cfg", "openpi"}:
        raise ValueError(f"unsupported policy server backend: {backend!r}")
    _validate_image_preprocessing(identity.get("image_preprocessing"))
    if backend == "openpi" and identity["image_preprocessing"] != "none":
        raise ValueError("OpenPI/JAX only supports image_preprocessing='none'")
    expected = {
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "action_dim": POLICY_ACTION_DIM,
        "action_horizon": action_horizon,
        "camera_names": list(CAMERA_NAMES),
        "transport_image_size": list(POLICY_TRANSPORT_SIZE),
        "image_preprocessing": expected_image_preprocessing,
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
    if backend in {"rlinf-sft", "rlinf-cfg", "openpi"}:
        _require_sha256(identity.get("norm_provenance_sha256"), "server norm provenance")
        _require_sha256(identity.get("training_manifest_sha256"), "server training manifest")
    if backend in {"rlinf-sft", "openpi"}:
        _require_sha256(identity.get("dataset_info_sha256"), "server dataset info")
        _require_sha256(identity.get("dataset_sidecar_sha256"), "server dataset sidecar")
        source_framework = identity.get("source_framework")
        if source_framework not in {"jax", "pytorch"}:
            raise ValueError("policy server has an unsupported SFT source framework")
        if backend == "openpi" and source_framework != "jax":
            raise ValueError("the OpenPI reference backend requires JAX artifacts")
        if source_framework == "jax":
            if backend == "rlinf-sft":
                _require_sha256(identity.get("parity_report_sha256"), "server conversion parity")
            else:
                _require_sha256(
                    identity.get("checkpoint_contract_sha256"),
                    "server JAX checkpoint contract",
                )
    if backend == "rlinf-cfg":
        _validate_cfg_guidance_metadata(identity.get("cfg_guidance"))
    if backend == "rlinf-sft" or identity.get("rtc_guidance") is not None:
        _validate_rtc_guidance_metadata(identity.get("rtc_guidance"))
    return identity


def _validate_rtc_guidance_metadata(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("policy server is missing RTC guidance metadata")
    required = {
        "implementation",
        "guidance_horizon",
        "max_guidance_weight",
        "prefix_attention_schedule",
        "delta_rebase",
    }
    if set(value) != required:
        raise ValueError("policy server RTC guidance metadata has invalid fields")
    if value["implementation"] != "physical_intelligence_guided_vjp":
        raise ValueError("policy server RTC implementation is unsupported")
    if value["delta_rebase"] is not True:
        raise ValueError("policy server RTC delta rebase must be enabled")
    GuidedRTCConfig(
        guidance_horizon=value["guidance_horizon"],
        max_guidance_weight=value["max_guidance_weight"],
        prefix_attention_schedule=value["prefix_attention_schedule"],
    )


def _validate_cfg_guidance_metadata(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "type",
        "scale",
        "positive_only_conditional",
    }:
        raise ValueError("policy server CFG guidance metadata is invalid")
    if value["type"] not in {"positive", "negative", "no_guide"}:
        raise ValueError("policy server CFG guidance type is invalid")
    scale = value["scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale):
        raise ValueError("policy server CFG guidance scale is invalid")
    if scale < 0.0 or not isinstance(value["positive_only_conditional"], bool):
        raise ValueError("policy server CFG guidance metadata is invalid")


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
    raw_images = {}
    for name in CAMERA_NAMES:
        key = f"observation.images.{name}"
        image = np.asarray(observation[key])
        if image.ndim != 4 or image.shape[0] != batch_size or image.shape[-1] != 3:
            raise ValueError(f"{key} must have shape (batch, H, W, 3)")
        raw_images[name] = image

    def prepare(name: str) -> tuple[str, np.ndarray]:
        key = f"observation.images.{name}"
        image = raw_images[name]
        resized = image_tools.resize_with_pad(image, *POLICY_TRANSPORT_SIZE)
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
        "inference_delay_steps",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("clothdojo.rtc has invalid fields")
    previous = _finite_actions(
        value["previous_model_actions"],
        POLICY_ACTION_DIM,
        "previous_model_actions",
    )
    previous_state = np.asarray(value["previous_state"], dtype=np.float32)
    if (
        previous_state.shape != (previous.shape[0], POLICY_ACTION_DIM)
        or not np.isfinite(previous_state).all()
    ):
        raise ValueError("clothdojo.rtc.previous_state must be finite float32[batch, 14]")
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
        raise ValueError("clothdojo.rtc.executed_horizon must be in [1, previous horizon)")
    result["executed_horizon"] = executed_horizon
    inference_delay_steps = value["inference_delay_steps"]
    if (
        isinstance(inference_delay_steps, bool)
        or not isinstance(inference_delay_steps, int)
        or not 0 <= inference_delay_steps < executed_horizon
    ):
        raise ValueError(
            "clothdojo.rtc.inference_delay_steps must be in [0, executed_horizon)"
        )
    result["inference_delay_steps"] = inference_delay_steps
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


def _configure_openpi_data_home() -> None:
    """Bind offline RLINF serving to the project-owned OpenPI asset cache."""

    configured = os.environ.get("CLOTHDOJO_OPENPI_CACHE")
    if not configured:
        return
    cache = Path(configured).expanduser().resolve()
    tokenizer = cache / "big_vision" / "paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"OpenPI tokenizer does not exist in cache: {tokenizer}")
    current = os.environ.get("OPENPI_DATA_HOME")
    if current is not None and Path(current).expanduser().resolve() != cache:
        raise ValueError("OPENPI_DATA_HOME differs from CLOTHDOJO_OPENPI_CACHE")
    os.environ["OPENPI_DATA_HOME"] = str(cache)


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
    if actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[2] != POLICY_ACTION_DIM:
        raise ValueError("previous model actions must have shape (batch, horizon, 14)")
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
