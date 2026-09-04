from __future__ import annotations

import ast
import inspect
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler

from clothdojo.training.openpi import TrainingPrompt


class RecordingTokenizer:
    def __init__(self):
        self.prompts = []

    def tokenize(self, prompt, state=None):
        self.prompts.append(prompt)
        return np.asarray([1, 2], dtype=np.int32), np.asarray([True, True])


@pytest.mark.parametrize("prompt", ["flatten the shirt", "Flatten the shirt."])
def test_training_prompt_is_picklable_preserves_exact_text_and_source(prompt):
    transform = pickle.loads(pickle.dumps(TrainingPrompt(prompt)))
    source = {"prompt": np.asarray("Original task."), "state": np.zeros(32)}
    result = transform(source)

    assert result is not source
    assert result["prompt"].item() == prompt
    assert source["prompt"].item() == "Original task."
    assert result["state"] is source["state"]


@pytest.mark.parametrize("prompt", [None, 4, "", "   "])
def test_training_prompt_rejects_empty_or_non_string_text(prompt):
    with pytest.raises(ValueError, match="non-empty string"):
        TrainingPrompt(prompt)


@pytest.mark.parametrize("eval_dataset", [False, True])
@pytest.mark.parametrize("prompt", ["flatten the shirt", "Flatten the shirt."])
def test_sft_worker_overrides_task_before_official_tokenization(
    monkeypatch, eval_dataset, prompt
):
    from openpi import transforms
    from openpi.training.data_loader import DataLoaderImpl, TorchDataLoader, TransformedDataset

    from clothdojo.training.openpi_rlinf_sft import (
        FSDPVlaSftWorker,
        OpenPiRlinfSftWorker,
        _EpochAdvancingSampler,
    )

    # Mirror upstream nesting: LeRobot task lookup happens inside the outer
    # repack/model pipeline. The configured text must win even when task metadata
    # differs, without rewriting any source sample or starting loader workers.
    samples = [
        {"task_index": np.asarray(0), "state": np.zeros(32), "actions": np.zeros((4, 32))}
        for _ in range(2)
    ]
    tasks = {0: "Flatten the shirt."}
    raw_dataset = TransformedDataset(samples, [transforms.PromptFromLeRobotTask(tasks)])
    tokenizer = RecordingTokenizer()
    dataset = TransformedDataset(
        raw_dataset,
        [
            transforms.RepackTransform(
                {"prompt": "prompt", "state": "state", "actions": "actions"}
            ),
            transforms.TokenizePrompt(tokenizer, discrete_state_input=True),
        ],
    )
    original_pipeline = dataset._transform
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=0)
    official_loader = TorchDataLoader(
        dataset, local_batch_size=2, sampler=sampler, num_workers=0, framework="pytorch"
    )
    data_config = SimpleNamespace()
    upstream_loader = DataLoaderImpl(data_config, official_loader)
    upstream_calls = []

    def build_upstream(worker, data_paths, eval_dataset=False):
        upstream_calls.append((data_paths, eval_dataset))
        assert dataset._transform is original_pipeline
        assert tokenizer.prompts == []
        return upstream_loader, data_config

    monkeypatch.setattr(FSDPVlaSftWorker, "build_dataloader", build_upstream)
    worker = object.__new__(OpenPiRlinfSftWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "seed": 43,
                "model": {
                    "num_action_chunks": 4,
                    "openpi_data": {"default_prompt": prompt},
                },
            }
        }
    )
    worker._rank = 0
    result, actual_config = worker.build_dataloader(["dataset"], eval_dataset=eval_dataset)

    assert result is upstream_loader and actual_config is data_config
    assert upstream_calls == [(["dataset"], eval_dataset)]
    assert tokenizer.prompts == []  # Configuration must not eagerly decode data.
    batch = next(iter(result._data_loader))
    assert tokenizer.prompts == [prompt, prompt]
    assert torch.equal(batch["tokenized_prompt"], torch.tensor([[1, 2], [1, 2]]))
    assert all("prompt" not in sample for sample in samples)
    assert tasks == {0: "Flatten the shirt."}
    installed_sampler = official_loader.torch_loader.batch_sampler.sampler
    if eval_dataset:
        assert installed_sampler is sampler and sampler.seed == 0
    else:
        assert isinstance(installed_sampler, _EpochAdvancingSampler)
        assert installed_sampler._sampler is sampler and sampler.seed == 43


def test_cfg_places_same_training_prompt_before_guidance_tokenization():
    from openpi.training.data_loader import TransformedDataset
    from rlinf.data.datasets.recap.cfg_model import TokenizePromptWithGuidance

    from clothdojo.training import rlinf_cfg

    # Guard the actual CFG loader construction order, then exercise its shared
    # transform with the official guidance tokenizer (without downloading a model).
    tree = ast.parse(inspect.getsource(rlinf_cfg))
    pipelines = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
        and node.elts
        and isinstance(node.elts[0], ast.Call)
        and isinstance(node.elts[0].func, ast.Name)
        and node.elts[0].func.id == "TrainingPrompt"
    ]
    assert len(pipelines) == 1
    assert isinstance(pipelines[0].elts[-1], ast.Starred)
    assert ast.unparse(pipelines[0].elts[-1].value) == "model_transforms"
    tokenizer = RecordingTokenizer()
    source = {"prompt": "Flatten the shirt.", "state": np.zeros(32)}
    dataset = TransformedDataset(
        [source],
        [TrainingPrompt("flatten the shirt"), TokenizePromptWithGuidance(tokenizer, True)],
    )
    result = dataset[0]
    assert tokenizer.prompts == [
        "flatten the shirt",
        "flatten the shirt\nAdvantage: positive",
        "flatten the shirt\nAdvantage: negative",
    ]
    assert "tokenized_positive_guidance_prompt" in result
    assert source["prompt"] == "Flatten the shirt."
