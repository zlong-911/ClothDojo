"""Training image policy for the shared OpenPI_RLinf flow objective."""

from __future__ import annotations

import dataclasses
from functools import wraps
from types import MethodType
from typing import Any


def install_training_augmentation(core: Any, *, enabled: bool = False) -> None:
    """Set upstream augmentation once on an SFT/CFG core, not globally.

    Upstream uses one optional generator for CPU image augmentation and device
    flow noise. Its default disables random augmentation; passing a CPU
    generator to the entire loss instead breaks CUDA noise sampling. Prepare
    training images separately using the checkpointed global CPU RNG, leaving
    the original loss's noise/time RNG and inference path untouched. Disabled
    augmentation is the default and uses upstream train=False preprocessing,
    avoiding the fixed top-left crop that train=True applies without a generator.
    """

    import torch
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module

    if not isinstance(enabled, bool):
        raise TypeError("image augmentation enabled must be a bool")
    installed = getattr(core.compute_loss, "_clothdojo_training_augmentation", None)
    if installed is not None:
        if installed != enabled:
            raise ValueError("training image augmentation is already installed with another policy")
        return
    # Capture the unbound function, not this core: deepcopy must bind the loss
    # to the copied model rather than silently training the original parameters.
    original_compute_loss = core.compute_loss.__func__

    @wraps(original_compute_loss)
    def compute_loss(model: Any, observation: Any, actions: Any, *, train=False, **kwargs):
        if train and enabled:
            observation = dataclasses.replace(
                observation,
                images={
                    key: image.to(dtype=torch.float32) for key, image in observation.images.items()
                },
            )
            observation = model_module.preprocess_observation(
                observation, train=True, rng=torch.default_generator
            )
        # Keep deterministic layout/resize/mask preprocessing in the original
        # loss, without entering its augmentation or fixed-crop fallback branch.
        return original_compute_loss(model, observation, actions, train=False, **kwargs)

    compute_loss._clothdojo_training_augmentation = enabled
    core.compute_loss = MethodType(compute_loss, core)
