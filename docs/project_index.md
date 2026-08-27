# Project index

- `requirements.txt`: the single readable environment definition.
- `pyproject.toml`: package metadata, CLI registration and tool configuration.
- `src/clothdojo/cli.py`: the only user-facing command dispatcher.
- `src/clothdojo/diagnostics.py`: installation and runtime compatibility check.
- `src/clothdojo/training/rlinf.py`: `openpi-rlinf`, STEAM and CFG training adapter.
- `src/clothdojo/training/openpi_rlinf_sft.py`: current RLINF OpenPI worker boundary.
- `src/clothdojo/training/steam.py`: STEAM critic data/hyperparameter gates,
  launch manifest and three-member checkpoint merge.
- `src/clothdojo/training/rlinf_steam.py`: deterministic RLINF STEAM loader,
  state/full-model training and FP32-master runtime adapter.
- `src/clothdojo/training/steam_dataloader.py`: forkserver-safe loader entrypoint.
- `configs/training/steam_value_shirt_flatten.yaml`: shared production contract
  for three independent 8-GPU STEAM evaluator members.

- `src/clothdojo/training/rlinf_cfg.py`: registered RLINF CFG model/worker
  extension that preserves ClothDojo's OpenPI data settings.
- `src/clothdojo/training/cfg.py`: static two-dataset CFG compatibility and
  exact advantage-coverage gate.
- `src/clothdojo/training/steam_advantages.py`: owned distributed STEAM
  inference, quantile labeling and atomic provenance publication.
- `src/clothdojo/training/cfg_pipeline.py`: locked and status-recorded
  STEAM-labeling → data-gate → CFG transition.
- `src/clothdojo/training/episode_selection.py`: explicit reviewed episode
  split shared by labeling and CFG sampling.
- `src/clothdojo/training/openpi.py`: exact ALOHA normalization statistics
  using the same transforms as RLINF training.
- `src/clothdojo/training/jax_sft.py`: `openpi-jax` SFT entry, strict Orbax
  initialization, short-run gates and EMA/online checkpoint roles.
- `src/clothdojo/training/matrix.py`: durable sequential JAX SFT parameter
  matrix that records per-gate state and continues with the next reviewed trial
  after a failed trial.
- `configs/training/pi05_aloha_jax.yaml`: 8-GPU OpenPI/JAX 40,000-update
  contract aligned to the original OpenPI optimizer and precision policy.
- `configs/training/pi05_aloha_jax_matrix.yaml`: paired LR/schedule/batch
  experiments with sample-equivalent logging and checkpoint intervals.
- `configs/training/pi05_aloha.yaml`: reviewed-good ALOHA π0.5 RLINF
  contract for the 8-GPU, 40,000-update baseline.
- `configs/training/cfg_pi05_aloha.yaml`: guarded expert/RTC CFG mixture,
  external artifact references and 50-update evaluation checkpoints.
- `configs/training/steam_advantages_pi05_aloha.yaml`: accepted STEAM
  checkpoint, two-view scoring inputs, q80/q30 labels and eight-GPU inference.
- `configs/rollout/shirt_flatten.yaml`: production 1,800-step rollout scenario.
- `configs/evaluation/shirt_flatten.yaml`: calibration-stage, fingerprinted
  flattening success and timeout protocol.
- `src/clothdojo/rollout/initial_states.py`: validated garment-specific initial
  state pool and batched Newton generator.
- `src/clothdojo/rollout/policy.py`: identity-checked remote client and released
  RLINF SFT policy service.
- `src/clothdojo/rollout/guided_rtc.py`: minimal adapter matching LeRobot
  guided RTC for RLINF's OpenPI flow sampler.
- `src/clothdojo/rollout/collection.py`: dynamic multi-GPU scheduling,
  fixed-size same-asset world batches and canonical multi-trajectory manifest.
- `src/clothdojo/rollout/engine.py`: the one synchronized batch loop for chunk
  and zero-delay RTC; one world is simply a batch of one.
- `src/clothdojo/rollout/newton.py`: exact batched ALOHA action mapping and
  per-world atomic trajectory publication.
- `src/clothdojo/rollout/render.py`: three-view tiled RGB observations from the
  live simulation model.
- `src/clothdojo/evaluation/`: canonical factorized metric, GPU success
  tracker, compact condition runner and exact paired summarizer.
- `src/clothdojo/trajectory/`: trajectory dataset definition and atomic I/O.
- `src/clothdojo/trajectory/newton.py`: one-way Newton-to-trajectory adapter.
- `src/clothdojo/data/annotations.py`: bad-only annotation complement and
  deterministic output episode plan.
- `src/clothdojo/data/review_source.py`: narrow read-only adapter for the old
  single-episode LeRobot v3 review cache.
- `src/clothdojo/data/lerobot.py`: transactional pinned-LeRobot v2.1 writer,
  export provenance and STEAM episode selection.
- `src/clothdojo/data/v3_indexed_view.py`: atomic one-way v3-to-v2.1
  publication without video re-encoding.
- `src/clothdojo/data/indexed_lerobot.py`: project-owned per-camera timestamp
  index reader used by CFG and STEAM.
- `src/clothdojo/data/rollout.py`: offline trajectory replay and transactional
  multi-episode LeRobot rollout export.
- `configs/migration/cfg_*_v3_indexed_view.yaml`: exact historical CFG input
  migration contracts.
- `src/clothdojo/assets/garments.py`: minimal source/selection/converted garment
  storage shared by the collection console and simulator.
- `src/clothdojo/collection/config.py`: strict collection profiles and the two
  user-facing teleoperation choices; runtime installation paths are excluded.
- `src/clothdojo/collection/assets.py`: paginated garment curation over the
  canonical selection file.
- `src/clothdojo/collection/review.py`: trajectory/review index and
  revision-guarded atomic annotations.
- `src/clothdojo/collection/hil.py`: rollout review and immutable HIL
  intervention plans plus completed-branch indexing.
- `src/clothdojo/collection/continuation.py`: validated terminal-state discovery,
  consumption tracking and immutable continuation plans.
- `src/clothdojo/collection/workspace.py`: stable view-model adapter between
  the preserved old workspace and ClothDojo's canonical catalogs.
- `src/clothdojo/collection/web.py`: local FastAPI control plane and static
  four-panel browser workspace.
- `src/clothdojo/collection/supervisor.py`: local SQLite GPU leases, fixed
  process launch, PID reconciliation and bounded session logs.
- `src/clothdojo/collection/static/`: the old console's DOM, CSS and interaction
  design, mechanically split into HTML/CSS/JavaScript without a Node runtime.
- `src/clothdojo/teleop/`: fixed CloudXR/Isaac application entry, OpenXR input,
  Newton 1.5 IK/control, full-state cold restore, Fabric/NOCS display bridge and
  canonical recording.
- `src/clothdojo/teleop/teleop.kit`: packaged minimal Isaac RTX/OpenXR
  experience; it intentionally excludes Isaac's Warp extension.
- `src/clothdojo/sim/cloth.py`: converted-USDC to public Style3D builder boundary.
- `src/clothdojo/sim/scene.py`: minimal Piper/table/garment scene and Newton 1.5
  batch replication.
- `src/clothdojo/sim/runtime.py`: explicit MuJoCo/Style3D control and reset.
- `src/clothdojo/sim/grasp.py`: fixed-allocation expert four-seed/two-ring
  connected cloth-patch grasp.
- `src/clothdojo/sim/check.py`: executable full-scene acceptance gate.
- `assets/robots/piper/piper.usda`: simplified entry to the Piper USD closure,
  including the authored 99-contact/27-proxy expert collision layer; no URDF or
  custom collision cache is required at runtime.
- `assets/`: project-local source, preview and converted assets; ignored by Git.
- `docs/architecture.md`: system boundaries and dependency direction.
- `docs/migration.md`: ordered migration with acceptance gates.
- `docs/trajectory.md`: temporal alignment, fields and persistence guarantees.
- `docs/lerobot_export.md`: reviewed data selection, conversion and acceptance
  contract.
- `docs/pi05_sft.md`: evidence-backed Pi0.5 SFT contract and launch preflight.
- `docs/pi05_jax_sft.md`: authoritative JAX SFT contract, launch gates and
  conversion boundary into RLINF.
- `src/clothdojo/data/legacy_collection.py`: one-way, resumable import and
  manifest normalization for validated pre-ClothDojo collections.
- `docs/newton_15_audit.md`: public API changes and physics acceptance gates.
- `docs/rlinf_delta_audit.md`: disposition of all 19 old RLINF source changes.
- `docs/rtc_rollout.md`: RTC execution, rendering and publication acceptance
  contract.
- `docs/evaluation.md`: canonical targets, stable success semantics, condition
  pairing and benchmark output contract.
- `docs/collection.md`: collection data ownership, web/control boundary and
  fixed NVIDIA Isaac Teleop + CloudXR runtime architecture.
- `docs/offline_cfg.md`: RTC rollout-to-CFG artifact contract, readiness and
  remote launch gate.
- `docs/todo.md`: deferred decisions with long-term risk.
