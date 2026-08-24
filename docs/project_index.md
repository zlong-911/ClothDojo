# Project index

- `requirements.txt`: the single readable environment definition.
- `pyproject.toml`: package metadata, CLI registration and tool configuration.
- `src/clothdojo/cli.py`: the only user-facing command dispatcher.
- `src/clothdojo/diagnostics.py`: installation and runtime compatibility check.
- `src/clothdojo/training/rlinf.py`: thin π0.5, STEAM and CFG SFT adapter.
- `src/clothdojo/training/rlinf_cfg.py`: registered RLINF CFG model/worker
  extension that preserves ClothDojo's OpenPI data settings.
- `src/clothdojo/training/cfg.py`: static two-dataset CFG compatibility and
  exact advantage-coverage gate.
- `src/clothdojo/training/openpi.py`: exact ALOHA normalization statistics
  using the same transforms as RLINF training.
- `configs/training/pi05_aloha.yaml`: reviewed-good ALOHA π0.5 RLINF
  contract for the 8-GPU, 40,000-update baseline.
- `configs/training/cfg_pi05_aloha.yaml`: guarded expert/RTC CFG mixture,
  external artifact references and 50-update evaluation checkpoints.
- `configs/rollout/shirt_flatten.yaml`: production 1,800-step rollout scenario.
- `src/clothdojo/rollout/initial_states.py`: validated garment-specific initial
  state pool and batched Newton generator.
- `src/clothdojo/rollout/policy.py`: identity-checked remote client and released
  RLINF SFT policy service.
- `src/clothdojo/rollout/guided_rtc.py`: minimal adapter matching LeRobot
  guided RTC for RLINF's OpenPI flow sampler.
- `src/clothdojo/rollout/collection.py`: batch initial-state consumption and
  canonical multi-trajectory manifest.
- `src/clothdojo/rollout/engine.py`: synchronous chunk and zero-delay RTC execution.
- `src/clothdojo/rollout/newton.py`: exact ALOHA action mapping and one-world
  Newton trajectory episode.
- `src/clothdojo/rollout/render.py`: three-view RGB observations from a
  separate Newton visual twin.
- `src/clothdojo/trajectory/`: trajectory dataset definition and atomic I/O.
- `src/clothdojo/trajectory/newton.py`: one-way Newton-to-trajectory adapter.
- `src/clothdojo/data/annotations.py`: bad-only annotation complement and
  deterministic output episode plan.
- `src/clothdojo/data/review_source.py`: narrow read-only adapter for the old
  single-episode LeRobot v3 review cache.
- `src/clothdojo/data/lerobot.py`: transactional pinned-LeRobot v2.1 writer,
  export provenance and STEAM episode selection.
- `src/clothdojo/data/rollout.py`: offline trajectory replay and transactional
  multi-episode LeRobot rollout export.
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
- `src/clothdojo/data/legacy_collection.py`: one-way, resumable import and
  manifest normalization for validated pre-ClothDojo collections.
- `docs/newton_15_audit.md`: public API changes and physics acceptance gates.
- `docs/rlinf_delta_audit.md`: disposition of all 19 old RLINF source changes.
- `docs/rtc_rollout.md`: RTC execution, rendering and publication acceptance
  contract.
- `docs/collection.md`: collection data ownership, web/control boundary and
  fixed NVIDIA Isaac Teleop + CloudXR runtime architecture.
- `docs/offline_cfg.md`: RTC rollout-to-CFG artifact contract, readiness and
  remote launch gate.
- `docs/todo.md`: deferred decisions with long-term risk.
