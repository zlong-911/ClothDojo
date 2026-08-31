from __future__ import annotations

import dataclasses
import json
from types import MethodType

import numpy as np
import pytest
import torch

from clothdojo.rollout import policy


@pytest.fixture(autouse=True)
def cpu_rng():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(previous_threads)


def observation(shape=(2, 224, 224, 3), dtype=torch.float32, device="cpu"):
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model

    return model.Observation(
        images={
            key: torch.rand(shape, dtype=dtype, device=device) * 2 - 1
            for key in model.IMAGE_KEYS
        },
        image_masks={
            key: torch.ones(shape[0], dtype=torch.bool, device=device)
            for key in model.IMAGE_KEYS
        },
        state=torch.ones(shape[0], 32, device=device),
        tokenized_prompt=torch.ones(shape[0], 8, dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones(shape[0], 8, dtype=torch.bool, device=device),
    )


@pytest.mark.parametrize("shape", [(2, 224, 224, 3), (2, 3, 224, 224), (2, 180, 320, 3)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="historical BF16 preprocessing uses CUDA")
def test_training_crop_matches_fsdp_root_cast_and_upstream_preprocessing(shape):
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model
    from torch.distributed.utils import _cast_forward_inputs

    # Serving Observation.from_dict always converts uint8 images to float32,
    # independently of the BF16 model weight dtype.
    obs = observation(shape, device="cuda:0")
    snapshots = {key: value.clone() for key, value in obs.images.items()}
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    _, casted = _cast_forward_inputs(torch.bfloat16, observation=obs)
    expected = model.preprocess_observation(casted["observation"], train=True, rng=None)
    actual = policy._preprocess_inference_images(obs, "training_fixed_crop")
    uncropped = model.preprocess_observation(obs, train=False)
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_state)
    assert not torch.equal(actual.images["base_0_rgb"], uncropped.images["base_0_rgb"])
    for key in obs.images:
        assert torch.equal(obs.images[key], snapshots[key])
        assert actual.images[key].dtype == torch.bfloat16
        assert torch.isfinite(actual.images[key]).all()
        torch.testing.assert_close(actual.images[key], expected.images[key], rtol=0, atol=0)
    for field in dataclasses.fields(obs):
        if field.name not in {"images", "image_masks"}:
            assert getattr(actual, field.name) is getattr(obs, field.name)
    for key in obs.image_masks:
        assert actual.image_masks[key] is obs.image_masks[key]


def test_none_is_identity_and_training_crop_rejects_unsupported_cpu():
    obs = observation()
    assert policy._preprocess_inference_images(obs, "none") is obs
    with pytest.raises(ValueError, match="requires CUDA"):
        policy._preprocess_inference_images(obs, "training_fixed_crop")
    with pytest.raises(ValueError, match="image_preprocessing"):
        policy._preprocess_inference_images(obs, "legacy_fixed_crop")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="historical BF16 preprocessing uses CUDA")
def test_training_crop_passes_no_rng_and_bf16_images(monkeypatch):
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model

    obs = observation(device="cuda:0")
    original = model.preprocess_observation

    def deterministic_training_pixels(observation, *, train=False, **kwargs):
        assert train is True
        assert kwargs == {"rng": None}
        assert all(image.dtype == torch.bfloat16 for image in observation.images.values())
        return original(observation, train=train, **kwargs)

    monkeypatch.setattr(model, "preprocess_observation", deterministic_training_pixels)
    policy._preprocess_inference_images(obs, "training_fixed_crop")


@pytest.mark.parametrize("backend", ["sft", "cfg"])
@pytest.mark.parametrize("image_preprocessing", ["none", "training_fixed_crop"])
def test_actual_wrapper_chunk_rtc_and_cfg_branches_crop_exactly_once(backend, image_preprocessing):
    from rlinf.models.embodiment.openpi_rlinf.eval_action_model import OpenPiPytorchEvalActionModel
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module

    from clothdojo.training.cfg_model import CfgRoutingConfig, ClothDojoOpenPiCfgModel

    # Use the actual SFT/CFG wrappers and RTC samplers. Only replace the neural
    # network and tokenizer/data transforms; all observation routing is real.
    class Core(torch.nn.Module):
        action_horizon = 4
        action_dim = 32

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.received = []

        def compute_loss(self, *args, **kwargs):
            raise AssertionError("inference must not invoke training loss")

        def sample_actions(self, obs, **kwargs):
            self.build_prefix_cache(model_module.preprocess_observation(obs, train=False))
            return torch.zeros(obs.state.shape[0], self.action_horizon, self.action_dim)

        def build_prefix_cache(self, obs):
            self.received.append(obs)
            return None, None, None

        def run_suffix(self, obs, latent, *args):
            return latent * 0.1

        def velocity_from_suffix(self, suffix):
            return suffix

    if image_preprocessing != "none" and not torch.cuda.is_available():
        pytest.skip("historical BF16 preprocessing uses CUDA")
    device = "cuda:0" if image_preprocessing != "none" else "cpu"
    core = Core().to(device)
    wrapper_type = OpenPiPytorchEvalActionModel if backend == "sft" else ClothDojoOpenPiCfgModel
    kwargs = {} if backend == "sft" else {
        "routing": CfgRoutingConfig(False, 0.1, "positive", 1.5)
    }
    wrapper = wrapper_type(core, num_steps=2, action_env_dim=14, action_chunk=4, **kwargs)
    original_obs = observation(device=device)

    def input_transform(self, raw, transpose=False):
        assert transpose is False
        return {"image": dict(original_obs.images), "image_mask": original_obs.image_masks,
                "state": original_obs.state, "tokenized_prompt": original_obs.tokenized_prompt,
                "tokenized_prompt_mask": original_obs.tokenized_prompt_mask}

    def output_transform(self, data):
        return {"actions": data["actions"][..., :14]}

    wrapper.input_transform = MethodType(input_transform, wrapper)
    wrapper.output_transform = MethodType(output_transform, wrapper)
    wrapper.eval().requires_grad_(False)
    policy._install_inference_image_preprocessing(wrapper, image_preprocessing)
    installed = wrapper._observation_dict_to_device
    policy._install_inference_image_preprocessing(wrapper, image_preprocessing)
    assert wrapper._observation_dict_to_device == installed
    serving = object.__new__(policy._RLinfSFTPolicy if backend == "sft" else policy._RLinfCFGPolicy)
    serving.model, serving.device, serving.autocast = wrapper, torch.device(device), False
    serving.rtc_guidance = policy.GuidedRTCConfig(guidance_horizon=2)
    serving.num_steps, serving._delta_action_scale = 2, np.ones(14)
    raw = {"observation.state": np.zeros((2, 14), dtype=np.float32), **{
        f"observation.images.{name}": np.zeros((2, 224, 224, 3), dtype=np.uint8)
        for name in policy.CAMERA_NAMES
    }}
    # Chunk and the first RTC request both use the no-history path.
    for _ in range(2):
        actions, model_actions = serving.infer(raw, "Flatten the shirt.", None)
        assert actions.shape == model_actions.shape == (2, 4, 14)
    serving.infer(raw, "Flatten the shirt.", {
        "previous_model_actions": model_actions,
        "previous_state": raw["observation.state"],
        "executed_horizon": 2, "inference_delay_steps": 1,
    })
    expected = policy._preprocess_inference_images(original_obs, image_preprocessing)
    assert len(core.received) == (3 if backend == "sft" else 6)
    for received in core.received:
        for key in expected.images:
            torch.testing.assert_close(received.images[key], expected.images[key], rtol=0, atol=0)
        assert torch.equal(received.state, original_obs.state)
    assert not wrapper.training and not core.training


def test_jax_rejects_training_crop_before_loading_artifacts():
    with pytest.raises(ValueError, match="OpenPI/JAX"):
        policy.serve_jax_policy(
            "missing", "missing", "missing", image_preprocessing="training_fixed_crop"
        )


def test_scoring_rejects_mixed_image_preprocessing(tmp_path):
    from test_evaluation import _offline_evaluation

    from clothdojo.evaluation.collection import score_evaluation_condition

    condition, protocol, templates, _ = _offline_evaluation(tmp_path)
    meta_path = condition / "meta.json"
    metadata = json.loads(meta_path.read_text())
    metadata["policy"]["image_preprocessing"] = "training_fixed_crop"
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="image preprocessing differs"):
        score_evaluation_condition(condition, protocol, templates)


@pytest.mark.parametrize("backend", ["sft", "cfg"])
def test_serve_cli_passes_explicit_training_crop(monkeypatch, backend):
    from clothdojo import cli

    recorded = {}
    monkeypatch.setattr(policy, f"serve_{backend}_policy", lambda *args, **kw: recorded.update(kw))
    cli.main([
        "policy", f"serve-{backend}", "--checkpoint", "checkpoint", "--norm-stats", "norm",
        "--image-preprocessing", "training_fixed_crop",
    ])
    assert recorded["image_preprocessing"] == "training_fixed_crop"


def test_benchmark_cli_passes_explicit_training_crop(monkeypatch):
    from clothdojo import cli
    from clothdojo.evaluation import collection

    recorded = {}
    monkeypatch.setattr(
        collection, "run_evaluation_condition", lambda *a, **kw: recorded.update(kw)
    )
    cli.main([
        "benchmark", "run", "--condition-id", "training_crop", "--scenario", "scenario",
        "--protocol", "protocol", "--states", "states", "--templates", "templates",
        "--output", "output", "--image-preprocessing", "training_fixed_crop",
    ])
    assert recorded["image_preprocessing"] == "training_fixed_crop"
