# RLINF delta audit

## Decision

ClothDojo uses upstream RLINF commit
`7d07a4212ee6858cc333e1d4fab7a37256d1f839` (version 0.4.0, 2026-08-20).
It is 53 commits newer than the old project's base commit
`c5ca51cc21c007a41d287159f9e1b14e0200000e`.

The old project changed 19 RLINF files. None currently justifies a fork:

- already-upstreamed behavior is deleted;
- setup and runtime mutations are deleted;
- project experiments remain outside the baseline;
- general defects become isolated upstream candidates;
- project-specific model behavior uses RLINF's model registry and
  `RLINF_EXT_MODULE`, rather than modifying RLINF files.

RLINF's current extension module mechanism supports registered models, but its
environment path still validates a closed `SupportedEnvType` enum. ClothDojo
will not hide that gap behind import-time monkey-patching. Offline SFT can use
the thin adapter now; online RLINF rollout waits for a small environment
registry upstream or another explicitly reviewed integration boundary.

## File-by-file result

| # | Old modified file | Result | Reason |
|---|---|---|---|
| 1 | `examples/.../compute_advantages_ensemble.py` | Defer experiment | Episode subsets and robot-state input belong to the later dual-state experiment. First-batch tensor logging is diagnostic residue. |
| 2 | `requirements/embodied/download_assets.sh` | Upstream candidate | The old change puts the tokenizer under `big_vision/`, matching OpenPI's cache key. Current RLINF still downloads it one directory too high. ClothDojo will use OpenPI's public cache path until this is fixed upstream. |
| 3 | `requirements/install.sh` | Delete | It creates a nested project venv and installs source trees imperatively. `requirements.txt` now owns one environment. |
| 4 | `rlinf/data/datasets/recap/cfg_model.py` | Defer performance work | Batched video lookup is an optimization coupled to private LeRobot internals, not baseline correctness. |
| 5 | `rlinf/data/datasets/recap/common.py` | Defer performance work | Locality block sampling and batched indexing need a measured decoder bottleneck before adoption. |
| 6 | `rlinf/data/datasets/recap/utils.py` | Delete compatibility code | New upstream has a LeRobot compatibility layer. ClothDojo initially pins RLINF's own LeRobot v2.1 revision and converts at the data boundary. |
| 7 | `rlinf/data/datasets/steam/pair_dataset.py` | Split, no baseline change | LeRobot v3 shims are unnecessary on the pinned v2.1 baseline; PyAV thread monkey-patching is deleted. Privileged scores, state conditioning and episode filtering remain experiments. |
| 8 | `rlinf/data/process/distributed.py` | Upstream candidate | The file moved to `rlinf/algorithms/offline/process/`, but upstream still initializes NCCL before binding `LOCAL_RANK`. The fix is general and small. |
| 9 | `rlinf/envs/__init__.py` | Upstream candidate | Latest RLINF still uses a closed enum and dispatch chain. A public environment registry is the correct fix; ClothDojo will not patch the enum. |
| 10 | `rlinf/envs/action_utils.py` | Same environment gap | Action preparation must be part of the same registered environment contract, not another ClothDojo branch inside RLINF. |
| 11 | `rlinf/models/embodiment/openpi/openpi_action_model.py` | Delete | Upstream commit `657faae5` added a maintained RTC implementation that supersedes the old local version. |
| 12 | `rlinf/models/embodiment/openpi_cfg/__init__.py` | Project model extension | The clean baseline proves CFG drops `model_path` and `openpi_data`, so external stats and Piper's `adapt_to_pi=false` are ignored. `clothdojo.training.rlinf_cfg` passes those arguments through the public registry. |
| 13 | `rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py` | Defer, then upstream or registered subclass | Per-action guidance masking and state-aware prompting are research behavior, not required for the first π0.5/STEAM baseline. |
| 14 | `rlinf/models/embodiment/value_model/steam/configuration.py` | Defer experiment | Robot-state conditioning changes the STEAM research model and artifact contract. |
| 15 | `rlinf/models/embodiment/value_model/steam/modeling_critic.py` | Defer experiment | This only propagates the dual-state model change. |
| 16 | `rlinf/models/embodiment/value_model/steam/modeling_steam.py` | Split, no baseline change | Dual-state fusion is deferred, debug synchronization is deleted, and prompt trimming is a general performance candidate only after measurement. |
| 17 | `rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py` | Upstream candidate | Explicit π0.5 selection and float32 construction are general conversion correctness fixes. Asset-path semantics need a standalone reproducer before proposing a change. |
| 18 | `rlinf/workers/sft/fsdp_cfg_worker.py` | Narrow project subclass | Pinning RLINF's LeRobot v2.1 removes the v3 shims. The subclass preserves upstream loading and changes only the missing `openpi_data` argument plumbing; locality sampling and decoder thread caps remain deferred performance work. |
| 19 | `rlinf/workers/sft/fsdp_steam_sft_worker.py` | Defer experiment | State and privileged-score arguments only support the deferred research variants. |

## Fork gate

A fork may be considered only when all of these are recorded:

1. the failure reproduces on the pinned clean upstream revision;
2. configuration, the model registry, a project adapter and a subclass are all
   insufficient;
3. the required change has a focused test and minimal diff;
4. upstream contribution is impractical for a documented reason;
5. maintaining that diff is cheaper than delaying the affected workflow.

No current delta passes this gate.

The CFG extension is not a fork exception: it uses RLINF's project extension
mechanism, is selected only by ClothDojo's CFG config, and has a test that locks
the shared model/dataloader OpenPI arguments.
