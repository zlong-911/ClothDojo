# Offline RTC to CFG

This document fixes the interchange contract for the next data-agent slice. It
does not make the old rollout launcher a dependency. ClothDojo now owns the RTC
execution/rendering/publication mechanics; the accepted-SFT policy connection
and STEAM labeling still gate training data.

## Pipeline

```text
accepted SFT checkpoint
          │
          ├── RTC policy rollout ──→ rollout LeRobot v2.1 ──┐
          │                                                │
expert LeRobot v2.1 ──┬── STEAM value/advantage labeling ──┤
                      │                                    │
                      └────────────────────────────────────┴──→ CFG mixture
```

RTC changes how the frozen SFT policy samples actions. It does not change the
offline dataset schema. The rollout exporter must publish a complete ClothDojo
LeRobot v2.1 dataset with the same 50 Hz Piper feature contract as the expert
dataset. Datasets, checkpoints, advantage labels and run outputs remain outside
the Git repository.

The old `run_steam_pool_rollouts.py` and `export_steam_rollouts.py` are evidence,
not migration targets. Their fixed asset list, GPU pairs, episode-number split,
symlink staging and `data/runtime` layout are deliberately excluded. The first
replacement accepts an explicit scenario and two external output roots, writes
through the ClothDojo trajectory boundary, and passes one real-GPU acceptance
case before any pool launcher is added. See `docs/rtc_rollout.md`.

## CFG input contract

`configs/training/cfg_pi05_aloha.yaml` consumes exactly two datasets:

- one `type: sft` expert dataset with sampling weight `0.75`;
- one `type: rollout` RTC dataset with sampling weight `0.25`.

`balance_dataset_weights` stays false, so these values are source probabilities
rather than length-adjusted weights. Both datasets must have:

- pinned LeRobot `v2.1` metadata and a complete `meta/clothdojo.json` export;
- no explicit `training_eligible: false` in `meta/clothdojo.json`; newly
  generated RTC exports must set it to `true`;
- identical `robot_type`, FPS and state/action/image feature definitions;
- `observation.state` and `action` as `float32[14]`;
- `cam_high`, `cam_left_wrist` and `cam_right_wrist` video features;
- one boolean row for every selected episode/frame in
  `meta/advantages_<tag>.parquet`.

The same advantage tag is used for both sources. Expert data must also be
labeled by the value model; it must not be declared positive wholesale. The
historical `rollout q30 / expert q80` rule is experiment evidence, not a default
embedded in the CFG trainer. The future advantage-labeling workflow owns and
records that decision.

The CFG policy is initialized from the accepted SFT checkpoint. It reuses the
normalization statistics computed from the expert dataset. The preflight checks
the expert `meta/info.json` hash against the norm-stat provenance, preventing a
rollout dataset or a stale export from silently becoming the normalization
source.

## Remote gate

After rollout export and STEAM advantage labeling have produced their external
artifacts, set:

```bash
export CLOTHDOJO_CFG_EXPERT_DATASET=/path/to/expert_lerobot_v21
export CLOTHDOJO_CFG_ROLLOUT_DATASET=/path/to/rtc_rollout_lerobot_v21
export CLOTHDOJO_CFG_ADVANTAGE_TAG=steam_value_v1
export CLOTHDOJO_CFG_INIT_CHECKPOINT=/path/to/accepted_sft_checkpoint
export CLOTHDOJO_PI05_NORM_STATS=/path/to/expert_norm_stats
export CLOTHDOJO_TRAIN_OUTPUT=/path/to/cfg_runs
```

Run the data gate before invoking RLINF:

```bash
.venv/bin/clothdojo data check-cfg \
  --config configs/training/cfg_pi05_aloha.yaml
```

The report resolves both roots, checks the dataset schemas and exact advantage
coverage, reports positive/negative counts and confirms the effective
`0.75/0.25` probabilities. Then validate the RLINF placement and batch contract
on the training host:

```bash
.venv/bin/clothdojo train cfg \
  --config configs/training/cfg_pi05_aloha.yaml \
  --check
```

The initial config is a 400-update acceptance gate with checkpoints every 50
updates. Checkpoints 50 and 100 must be evaluated against the same frozen
scenario manifest before extending the run. Do not increase the horizon by
editing the old launcher or by copying its long episode list into this config.

## Current readiness

The CFG config, data preflight and narrow RLINF extension are implemented and
tested. The extension only passes `model_path` and `openpi_data` consistently to
RLINF's OpenPI config builder, which is required for `adapt_to_pi=false` and the
external norm statistics.

The user-facing RTC scheduler, real Newton three-camera observations, trajectory
recording and transactional LeRobot v2.1 publication are implemented and have
passed one GPU acceptance run. That gate uses a deterministic hold-position
policy and stamps both outputs `training_eligible: false`; the CFG preflight
rejects them by design.

The next rollout slice connects the remotely accepted SFT checkpoint to the
plain policy interface, records its immutable identity, and only then permits
`training_eligible: true`. STEAM value training and advantage labeling also do
not yet have owned ClothDojo configs. Until both land, `data check-cfg` remains
the final executable training gate.
