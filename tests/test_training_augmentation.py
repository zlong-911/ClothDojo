from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from clothdojo.training.augmentation import install_training_augmentation


@pytest.fixture(autouse=True)
def cpu_rng():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(previous_threads)


def observation(batch=2):
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model

    return model.Observation(
        images={
            key: torch.linspace(-0.8, 0.8, batch * 224 * 224 * 3).reshape(batch, 224, 224, 3)
            for key in model.IMAGE_KEYS
        },
        image_masks={key: torch.ones(batch, dtype=torch.bool) for key in model.IMAGE_KEYS},
        state=torch.zeros(batch, 32),
        tokenized_prompt=torch.zeros(batch, 8, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(batch, 8, dtype=torch.bool),
    )


class RecordingCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def compute_loss(self, observation, actions, *, train=False, **kwargs):
        self.received = observation, actions, train, kwargs
        return observation.images["base_0_rgb"].mean() * self.weight


def test_random_augmentation_replays_global_rng_and_preserves_loss_arguments():
    core = RecordingCore()
    state_keys = tuple(core.state_dict())
    install_training_augmentation(core, enabled=True)
    installed_loss = core.compute_loss
    install_training_augmentation(core, enabled=True)
    assert core.compute_loss is installed_loss
    assert tuple(core.state_dict()) == state_keys
    obs = observation()
    actions = torch.zeros(2, 4, 32)
    kwargs = {
        "rng": torch.Generator().manual_seed(99),
        "noise": torch.ones_like(actions),
        "time": torch.full((2,), 0.5),
    }
    noise_rng_state = kwargs["rng"].get_state().clone()
    state = torch.get_rng_state()
    core.compute_loss(obs, actions, train=True, **kwargs)
    first, received_actions, train, received_kwargs = core.received
    assert received_actions is actions and train is False
    assert all(received_kwargs[key] is value for key, value in kwargs.items())
    assert torch.equal(kwargs["rng"].get_state(), noise_rng_state)
    core.compute_loss(obs, actions, train=True)
    second = core.received[0]
    assert all(not torch.equal(first.images[key], second.images[key]) for key in obs.images)
    torch.set_rng_state(state)
    core.compute_loss(obs, actions, train=True)
    restored = core.received[0]
    assert all(torch.equal(first.images[key], restored.images[key]) for key in obs.images)
    assert not torch.equal(obs.images["base_0_rgb"], first.images["base_0_rgb"])


@pytest.mark.parametrize("enabled", [False, True])
def test_installation_is_idempotent_but_rejects_a_different_policy(enabled):
    core = RecordingCore()
    install_training_augmentation(core, enabled=enabled)
    installed_loss = core.compute_loss
    install_training_augmentation(core, enabled=enabled)
    assert core.compute_loss is installed_loss
    with pytest.raises(ValueError, match="another policy"):
        install_training_augmentation(core, enabled=not enabled)
    assert core.compute_loss is installed_loss


@pytest.mark.parametrize("enabled", ["false", 0, None])
def test_installation_rejects_non_boolean_policy(enabled):
    with pytest.raises(TypeError, match="must be a bool"):
        install_training_augmentation(RecordingCore(), enabled=enabled)


@pytest.mark.parametrize("explicit_policy", [False, True])
def test_disabled_augmentation_preserves_images_rng_and_all_loss_arguments(explicit_policy):
    core = RecordingCore()
    install_training_augmentation(core, **({"enabled": False} if explicit_policy else {}))
    assert core.compute_loss._clothdojo_training_augmentation is False
    obs, actions = observation(), torch.zeros(2, 4, 32)
    kwargs = {
        "rng": torch.Generator().manual_seed(99),
        "noise": torch.ones_like(actions),
        "time": torch.full((2,), 0.5),
        "future_argument": object(),
    }
    global_state = torch.get_rng_state()
    noise_state = kwargs["rng"].get_state()
    core.compute_loss(obs, actions, train=True, **kwargs)
    received_obs, received_actions, train, received_kwargs = core.received
    assert received_obs is obs and received_actions is actions and train is False
    assert received_kwargs.keys() == kwargs.keys()
    assert all(received_kwargs[key] is value for key, value in kwargs.items())
    assert torch.equal(torch.get_rng_state(), global_state)
    assert torch.equal(kwargs["rng"].get_state(), noise_state)


@pytest.mark.parametrize("enabled", [False, True])
def test_eval_is_untouched_and_deepcopy_binds_to_copied_parameters(enabled):
    core = RecordingCore()
    install_training_augmentation(core, enabled=enabled)
    clone = copy.deepcopy(core)
    obs = observation()
    state = torch.get_rng_state()
    clone.compute_loss(obs, None, train=False).backward()
    assert clone.received[0] is obs
    assert torch.equal(state, torch.get_rng_state())
    assert clone.weight.grad is not None and core.weight.grad is None
    assert not hasattr(core, "received")


def tiny_core(monkeypatch, dtype):
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import gemma, pi0, siglip
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config

    # Exercise upstream tensor math on CPU without invoking its CUDA-oriented
    # compile toolchain; production code keeps the original compiled function.
    monkeypatch.setattr(gemma, "_apply_rope", gemma._apply_rope.__wrapped__)
    monkeypatch.setattr(gemma, "_fused_gated_residual", gemma._fused_gated_residual.__wrapped__)
    monkeypatch.setattr(gemma, "gelu_glu", gemma.gelu_glu.__wrapped__)

    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda _: gemma.Config(
            width=32, depth=2, mlp_dim=64, num_heads=4, num_kv_heads=1, head_dim=8
        ),
    )
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 64)
    monkeypatch.setattr(
        siglip,
        "_decode_variant",
        lambda _: {
            "width": 32,
            "depth": 2,
            "num_heads": 4,
            "mlp_dim": 64,
            "patch_size": (112, 112),
        },
    )
    core = pi0.Pi0(
        Pi0Config(pi05=True, action_dim=32, action_horizon=4, max_token_len=8, dtype=dtype)
    ).to(getattr(torch, dtype))
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.normal_(0, 0.1)
    return core


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("enabled", [False, True])
def test_core_matches_explicit_upstream_image_policy_loss_and_gradients(
    monkeypatch, dtype, enabled
):
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model

    core = tiny_core(monkeypatch, dtype)
    reference = copy.deepcopy(core)
    install_training_augmentation(core, enabled=enabled)
    obs = observation()
    # FSDP may cast inputs before dispatch; enabled augmentation still needs FP32.
    obs = dataclasses.replace(
        obs, images={k: v.to(getattr(torch, dtype)) for k, v in obs.images.items()}
    )
    actions = torch.randn(2, 4, 32)
    noise, time = torch.randn_like(actions), torch.full((2,), 0.5)
    seed = torch.get_rng_state()
    loss = core.compute_loss(obs, actions, train=True, noise=noise, time=time)
    if not enabled:
        assert torch.equal(torch.get_rng_state(), seed)
    torch.set_rng_state(seed)
    if enabled:
        fp32_obs = dataclasses.replace(obs, images={k: v.float() for k, v in obs.images.items()})
        prepared = model.preprocess_observation(fp32_obs, train=True, rng=torch.default_generator)
    else:
        prepared = model.preprocess_observation(obs, train=False)
    expected = reference.compute_loss(prepared, actions, train=False, noise=noise, time=time)
    assert torch.equal(loss, expected)
    loss.mean().backward()
    expected.mean().backward()
    assert_matching_gradients(core, reference)


def assert_matching_gradients(left, right):
    nonzero = 0
    for a, b in zip(left.parameters(), right.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)
            nonzero += bool(torch.count_nonzero(a.grad))
    assert nonzero > 0


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_disabled_augmentation_preserves_flow_sampling_and_gradients(monkeypatch, dtype):
    core = tiny_core(monkeypatch, dtype)
    reference = copy.deepcopy(core)
    install_training_augmentation(core, enabled=False)
    obs, actions = observation(), torch.randn(2, 4, 32)
    rng = torch.Generator().manual_seed(99)
    reference_rng = torch.Generator().manual_seed(99)
    seed = torch.get_rng_state()
    # No explicit noise/time: the flow objective must still sample both.
    loss = core.compute_loss(obs, actions, train=True, rng=rng)
    actual_rng_state = torch.get_rng_state()
    torch.set_rng_state(seed)
    expected = reference.compute_loss(obs, actions, train=False, rng=reference_rng)
    assert not torch.equal(actual_rng_state, seed)
    assert torch.equal(torch.get_rng_state(), actual_rng_state)
    assert torch.equal(rng.get_state(), reference_rng.get_state())
    assert torch.equal(loss, expected)
    loss.mean().backward()
    expected.mean().backward()
    assert_matching_gradients(core, reference)


@pytest.mark.parametrize("channels_first", [False, True])
@pytest.mark.parametrize("explicit_policy", [False, True])
def test_disabled_augmentation_keeps_deterministic_preprocessing(
    monkeypatch, channels_first, explicit_policy
):
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model

    core = tiny_core(monkeypatch, "float32")
    install_training_augmentation(core, **({"enabled": False} if explicit_policy else {}))
    obs, actions = observation(), torch.zeros(2, 4, 32)
    if channels_first:
        obs = dataclasses.replace(
            obs,
            images={
                key: value[:, :192, :160].permute(0, 3, 1, 2) for key, value in obs.images.items()
            },
            image_masks={},
        )
    upstream_preprocess = model.preprocess_observation
    expected = upstream_preprocess(obs, train=False)
    preprocessed = []

    def record_preprocessing(observation, **kwargs):
        assert kwargs["train"] is False
        result = upstream_preprocess(observation, **kwargs)
        preprocessed.append(result)
        return result

    monkeypatch.setattr(model, "preprocess_observation", record_preprocessing)
    seed = torch.get_rng_state()
    core.compute_loss(obs, actions, train=True, noise=actions, time=torch.full((2,), 0.5))
    assert torch.equal(seed, torch.get_rng_state())
    assert len(preprocessed) == 1
    for key in obs.images:
        assert preprocessed[0].images[key].shape == (2, 224, 224, 3)
        assert torch.equal(preprocessed[0].images[key], expected.images[key])
        assert torch.equal(preprocessed[0].image_masks[key], expected.image_masks[key])
        if not channels_first:
            # Native-resolution images must retain all pixels, including borders.
            assert preprocessed[0].images[key] is obs.images[key]


@pytest.mark.parametrize(
    "setting",
    [
        None,
        {},
        {"image_augmentation": True},
        {"image_augmentation": False},
        {"image_augmentation": "false"},
        {"image_augmentation": 0},
        {"image_augmentation": None},
    ],
)
def test_sft_worker_installs_configured_image_policy(monkeypatch, setting):
    from omegaconf import OmegaConf

    from clothdojo.training.openpi_rlinf_sft import FSDPVlaSftWorker, OpenPiRlinfSftWorker

    monkeypatch.setattr(FSDPVlaSftWorker, "init_worker", lambda self: None)
    worker = object.__new__(OpenPiRlinfSftWorker)
    raw = {"actor": {"optim": {"ema_decay": 0.99}}}
    if setting is not None:
        raw["clothdojo"] = setting
    worker.cfg = OmegaConf.create(raw)
    worker.model = torch.nn.Module()
    worker.model.model = RecordingCore()
    worker.optimizer = SimpleNamespace(state={})
    worker._action_offset_weights = (1.0,)
    expected = (setting or {}).get("image_augmentation", False)
    if not isinstance(expected, bool):
        with pytest.raises(TypeError, match="must be a bool"):
            worker.init_worker()
        return
    worker.init_worker()
    assert worker.model.model.compute_loss._clothdojo_training_augmentation is expected


@pytest.mark.parametrize("enabled", [False, True])
def test_native_inference_and_eval_loss_are_unchanged(monkeypatch, enabled):
    core = tiny_core(monkeypatch, "float32").eval()
    obs, actions = observation(), torch.randn(2, 4, 32)
    noise, time = torch.randn_like(actions), torch.full((2,), 0.5)
    with torch.no_grad():
        expected_actions = core.sample_actions(obs, num_steps=2, noise=noise)
        expected_loss = core.compute_loss(obs, actions, train=False, noise=noise, time=time)
        install_training_augmentation(core, enabled=enabled)
        actual_actions = core.sample_actions(obs, num_steps=2, noise=noise)
        actual_loss = core.compute_loss(obs, actions, train=False, noise=noise, time=time)
    assert torch.equal(actual_actions, expected_actions)
    assert torch.equal(actual_loss, expected_loss)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_default_cfg_matches_sft_loss_and_gradients(monkeypatch, dtype, checkpointing):
    from rlinf.models.embodiment.openpi_rlinf.sft_action_model import OpenPiPytorchSFTActionModel

    from clothdojo.training.cfg_model import CfgRoutingConfig, ClothDojoOpenPiCfgModel

    core = tiny_core(monkeypatch, dtype)
    cfg = ClothDojoOpenPiCfgModel(
        copy.deepcopy(core),
        num_steps=2,
        action_env_dim=14,
        routing=CfgRoutingConfig(True, 0.1, "positive", 1.0),
    )
    sft = OpenPiPytorchSFTActionModel(core, num_steps=2, action_env_dim=14)
    install_training_augmentation(sft.model)
    assert cfg.model.compute_loss._clothdojo_training_augmentation is False
    assert sft.model.compute_loss._clothdojo_training_augmentation is False
    if checkpointing:
        cfg.gradient_checkpointing_enable()
        sft.gradient_checkpointing_enable()
    base = observation()
    obs = SimpleNamespace(
        **vars(base),
        tokenized_positive_guidance_prompt=base.tokenized_prompt + 1,
        tokenized_positive_guidance_prompt_mask=base.tokenized_prompt_mask,
        tokenized_negative_guidance_prompt=base.tokenized_prompt + 2,
        tokenized_negative_guidance_prompt_mask=base.tokenized_prompt_mask,
    )
    actions, advantage = torch.randn(2, 4, 32), torch.tensor([True, False])
    seed = torch.get_rng_state()
    conditional = (torch.rand(2) > 0.1) & advantage
    expected_obs = dataclasses.replace(
        base,
        tokenized_prompt=torch.where(
            conditional[:, None], obs.tokenized_positive_guidance_prompt, base.tokenized_prompt
        ),
    )
    expected_loss = sft(data=(expected_obs, actions))
    expected_loss.backward()
    torch.set_rng_state(seed)
    loss, _ = cfg({"observation": obs, "actions": actions, "advantage": advantage})
    loss.backward()
    assert torch.equal(loss, expected_loss)
    assert_matching_gradients(cfg, sft)


@pytest.mark.parametrize("enabled", [False, True])
def test_weighted_sft_uses_same_image_policy_core(monkeypatch, enabled):
    from rlinf.models.embodiment.openpi_rlinf.sft_action_model import OpenPiPytorchSFTActionModel

    from clothdojo.training.openpi_rlinf_sft import (
        _install_action_offset_reduction,
        _reduce_action_offset_loss,
    )

    core = tiny_core(monkeypatch, "float32")
    install_training_augmentation(core, enabled=enabled)
    reference = copy.deepcopy(core)
    sft = OpenPiPytorchSFTActionModel(core, num_steps=2, action_env_dim=14)
    weights = (4.0, 2.0, 1.0, 1.0)
    _install_action_offset_reduction(sft, weights)
    obs, actions = observation(), torch.randn(2, 4, 32)
    seed = torch.get_rng_state()
    loss = sft(data=(obs, actions))
    torch.set_rng_state(seed)
    expected = _reduce_action_offset_loss(reference.compute_loss(obs, actions, train=True), weights)
    assert torch.equal(loss, expected)
    loss.backward()
    expected.backward()
    assert_matching_gradients(core, reference)
