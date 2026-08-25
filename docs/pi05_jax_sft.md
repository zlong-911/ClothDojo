# Pi0.5 OpenPI/JAX SFT reference

OpenPI/JAX is the authoritative SFT optimizer for the next comparison. RLINF
continues to own policy serving, rollout and CFG after the accepted JAX EMA
checkpoint is converted to its PyTorch format. This split avoids asking the two
frameworks to be numerically interchangeable during optimization.

## Fixed contract

The executable config is `configs/training/pi05_aloha_jax.yaml`. It starts from
the released Pi0.5 Orbax parameter tree and uses the reviewed-good v2 dataset:
719 episodes, 1,287,887 frames, and a 65-frame export floor. The three model
views are `cam_high`, `cam_left_wrist` and `cam_right_wrist`; Piper's 14D state
and action are padded to the model's 32D width, and the action horizon is 50.

Optimization follows the OpenPI ALOHA reference: global batch 64 over eight
devices, seed 42, AdamW `0.9/0.95`, epsilon `1e-8`, weight decay `1e-10`, global
norm clip 1.0, and EMA 0.99. The learning rate warms up for 1,000 updates to
`2.5e-5`, decays with OpenPI's cosine schedule to `2.5e-6` at update 30,000,
then remains at that floor through update 40,000. Model parameters, gradients,
AdamW state and EMA remain FP32; the model's matrix-heavy forward computation
uses BF16.

The loader selects PyAV explicitly. The pinned TorchCodec package remains
useful to RLINF on hosts with matching FFmpeg libraries, but package presence
alone does not make it usable. PyAV is the portable JAX baseline and its real
throughput is measured by the pilot gate.

## Launch gates

Set these paths on the training host:

```bash
export CLOTHDOJO_LEROBOT_DATASET=/absolute/path/to/shirt_flatten_reviewed_good_v2
export CLOTHDOJO_PI05_NORM_STATS=/absolute/path/to/pi05_aloha_shirt_flatten_reviewed_good_v2
export CLOTHDOJO_PI05_JAX_PARAMS=/absolute/path/to/pi05_base/params
export CLOTHDOJO_PI05_PYTORCH_BASE=/absolute/path/to/pi05_base_hf_snapshot
export CLOTHDOJO_OPENPI_CACHE=/absolute/path/to/openpi_cache
export CLOTHDOJO_JAX_CHECKPOINT_BASE=/absolute/path/to/data/checkpoints/openpi_jax
export CLOTHDOJO_JAX_GATE_DIR=/absolute/path/to/outputs/training_gates
```

Then run the gates in order:

```bash
clothdojo train pi05-jax --config configs/training/pi05_aloha_jax.yaml --check
clothdojo train pi05-jax --config configs/training/pi05_aloha_jax.yaml --gate initialize
clothdojo train pi05-jax --config configs/training/pi05_aloha_jax.yaml --gate overfit
clothdojo train pi05-jax --config configs/training/pi05_aloha_jax.yaml --gate pilot
clothdojo train pi05-jax --config configs/training/pi05_aloha_jax.yaml --gate parity
clothdojo train pi05-jax --config configs/training/pi05_aloha_jax.yaml --gate full
```

`initialize` restores the full Orbax tree and checks actual FP32 update state.
`overfit` repeats one batch for 200 updates. `pilot` trains 1,000 shuffled
batches and measures the production decoder pool. `parity` compares one fixed
inference Euler step against the released PyTorch conversion of the same base.
`full` refuses to start if any gate is missing, failed, or refers to different
data, normalization, base metadata or tokenizer artifacts.

Weights are never uploaded to W&B. Metric windows are averaged before logging,
so OpenPI and RLINF curves can be compared without mistaking per-batch noise for
a different loss definition.

## Parameter matrix

`configs/training/pi05_aloha_jax_matrix.yaml` is the reviewed follow-up matrix.
It does not accept arbitrary optimizer drift: each trial selects a profile whose
batch, sample budget, logging interval and complete LR schedule are validated in
code. The first two trials use batch 64 and isolate LR scale under a cosine decay
that ends at update 40,000. The batch-128 trial runs 20,000 updates, warms up for
500 updates, logs every 50 updates and saves every 5,000 updates. Those choices
preserve the 2.56-million-sample budget, 64,000-sample warmup, 6,400-sample log
window and 640,000-sample checkpoint spacing of the batch-64 runs.

The matrix runs `initialize`, `overfit`, a 3,000-update shuffled pilot, `parity`
and `full` for each trial. Its status is written atomically below `outputs/`.
A failed trial is recorded and the next reviewed trial starts, so a local error
does not silently discard the remainder of the queue. W&B includes
`samples_seen`, which is the comparison x-axis when batch sizes differ.

`configs/training/pi05_aloha_jax_matrix_replication.yaml` keeps the completed
reference host occupied with the 0.5× LR bracket followed by a seed-43 repeat
of the conservative 1× slow-decay candidate. The repeat separates a real
schedule improvement from ordinary data-order and diffusion-noise variance.

`configs/training/pi05_aloha_jax_matrix_repair.yaml` is a one-trial recovery
queue for the 2× LR run whose full gate was deliberately rejected after its
preflight identity changed during profile development. It rebuilds every gate
under one immutable identity before starting full training.

## Checkpoint meaning

OpenPI checkpoints already contain both weight views without duplicating them:

- `params/` is the EMA inference tree and is the source for JAX-to-PyTorch
  conversion, SFT evaluation and CFG initialization.
- `train_state/` contains online parameters plus AdamW state and is used for
  exact resume or update analysis.

Each completed step directory receives `clothdojo_weights.json` describing
those roles. A converted EMA artifact must retain the source step and this
manifest identity; an online checkpoint must never be silently substituted.
