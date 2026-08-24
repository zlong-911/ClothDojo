# Pi0.5 SFT baseline

This baseline is the PyTorch/RLINF branch used to compare plain SFT with later
CFG training. It starts from the released Hugging Face `lerobot/pi05_base`
snapshot and produces online weights. It is not presented as a bitwise
reproduction of the older JAX/OpenPI run, whose inference checkpoint used EMA
0.99.

## Resolved experiment contract

| Item | Value | Reference and rationale |
| --- | --- | --- |
| Base | `lerobot/pi05_base` snapshot `7de663972b7817d2c4cf2d84c821153dfea772e9` | Released PyTorch Pi0.5 base used by RLINF. Preflight checks its 812 F32 tensors against all 813 model state keys and accepts only the verified tied embedding/output-head alias. |
| Model | `pi05_aloha_robotwin`, internal action/state width 32, action horizon 50, environment width 14 | OpenPI Pi0.5 architecture and the pinned RLINF Robotwin ALOHA data config. |
| Dataset | `shirt_flatten_reviewed_good_v2`: 719 episodes, 1,287,887 frames | Final bad-complement selection. A 65-frame quality floor removes 19 bad-split fragments (411 frames); it leaves 644 full and 75 partial episodes. |
| Cameras | high, left wrist, right wrist | The OpenPI three-image input contract. `cam_low` remains in LeRobot provenance but is not passed to the model. |
| Prompt | exact dataset task `flatten the shirt` | `prompt_from_task=true` makes LeRobot task metadata authoritative. The YAML `default_prompt` is only a missing-key fallback. |
| Piper transforms | `adapt_to_pi=false`, `extra_delta_transform=true` | Piper already uses the configured ALOHA vector order; the delta mask follows the referenced ALOHA joint-delta/gripper-absolute policy. |
| Normalization | Pi0.5 quantile statistics after the training transforms | Stats are recomputed for v2 and preflight requires exact dataset info/sidecar hashes, config name, transform kwargs and horizon. |
| Batch | global 64 = 8 ranks x local 8, accumulation 1 | Matches the historical OpenPI ALOHA batch and the verified RLINF worker behavior. |
| Optimizer | AdamW, LR `2.5e-5`, betas `0.9/0.95`, epsilon `1e-8`, weight decay `1e-10`, global clip 1 | OpenPI ALOHA optimizer settings. |
| LR schedule | `openpi_cosine`, warmup 1,000, schedule length 30,000, floor `2.5e-6` | Matches OpenPI's referenced 30k warmup-plus-decay schedule. RLINF clamps at the floor from step 30k through 40k. Generic Hugging Face `cosine` is forbidden here because it interpreted the absolute floor against AdamW's default LR and drove the previous run to about `6.25e-8`. |
| Run length | 40,000 optimizer updates | ClothDojo experiment choice: approximately two passes over v2 at global batch 64. It is intentionally longer than the 20k historical run and does not change the referenced 30k LR schedule. |
| Precision | OpenPI selected BF16 parameters, with patch embedding, norms and action/time projections retained in FP32; generic AMP off | Matches the pinned PyTorch OpenPI path. `precision: null` does not mean the constructed model is entirely FP32. |
| Randomness | base seed 42; process RNG uses `42 + rank`; distributed sampler uses common seed 42 and advances epoch | Historical PyTorch OpenPI rank convention plus the required distributed-sampler contract. ClothDojo's narrow worker adapter repairs the missing RLINF propagation and epoch advance. |
| Loss | flow-matching MSE averaged over batch, horizon 50 and all padded 32 action dimensions | Same mathematical reduction as OpenPI. The 18 padded dimensions remain part of the referenced model contract. |
| Checkpoints | 10k, 20k, 30k, 40k; online weights | Intermediate comparison points. EMA is deliberately absent in both the SFT result and the CFG initialization branch. |
| Resume | not exact; do not use for the reference run | Pinned official OpenPI loader is not stateful, so RLINF does not restore its data position. An interrupted reference run must restart from base until loader-state support is added. |

RLINF's schema also requires sampling/value/FSDP fields that are inert for this
SFT path, such as five ODE sampling steps, disabled value head and LoRA,
disabled generic AMP, and reshard flags under `no_shard`. They remain pinned by
the executable preflight contract so an upstream semantic change fails visibly;
they are not treated as tuned SFT hyperparameters.

`clothdojo train pi05 --check` is the launch gate. It validates the resolved
configuration, dataset, task, camera schema, normalization provenance and base
checkpoint layout without reading the 14 GB tensor payload. A real launch
writes the same evidence once to `pi05_launch_manifest.json` in a new, empty
output directory before RLINF initializes Ray.

The SFT and later CFG comparison must use this raw 40k SFT checkpoint as their
common policy state. Comparing it directly with the old JAX EMA checkpoint is a
separate cross-framework experiment and must be labelled accordingly.
