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
- `configs/rollout/shirt_flatten_rtc_gate.yaml`: fixed one-scenario RTC
  mechanics acceptance manifest.
- `src/clothdojo/rollout/rtc.py`: dependency-free RTC delay scheduler.
- `src/clothdojo/rollout/engine.py`: policy/episode interfaces and asynchronous
  action-chunk execution.
- `src/clothdojo/rollout/newton.py`: exact ALOHA action mapping and one-world
  Newton trajectory episode.
- `src/clothdojo/rollout/render.py`: three-view RGB observations from a
  separate Newton visual twin.
- `src/clothdojo/rollout/check.py`: real-GPU hold-policy acceptance workflow.
- `src/clothdojo/trajectory/`: trajectory dataset definition and atomic I/O.
- `src/clothdojo/trajectory/newton.py`: one-way Newton-to-trajectory adapter.
- `src/clothdojo/data/annotations.py`: bad-only annotation complement and
  deterministic output episode plan.
- `src/clothdojo/data/review_source.py`: narrow read-only adapter for the old
  single-episode LeRobot v3 review cache.
- `src/clothdojo/data/lerobot.py`: transactional pinned-LeRobot v2.1 writer,
  export provenance and STEAM episode selection.
- `src/clothdojo/data/rollout.py`: transactional live single-episode LeRobot
  rollout writer.
- `src/clothdojo/assets/garments.py`: minimal source/selection/converted garment
  storage shared by the future web console and simulator.
- `src/clothdojo/sim/cloth.py`: converted-USDC to public Style3D builder boundary.
- `src/clothdojo/sim/scene.py`: minimal Piper/table/garment scene and Newton 1.5
  batch replication.
- `src/clothdojo/sim/runtime.py`: explicit MuJoCo/Style3D control and reset.
- `src/clothdojo/sim/grasp.py`: fixed-allocation connected cloth-patch grasp.
- `src/clothdojo/sim/check.py`: executable full-scene acceptance gate.
- `assets/robots/piper/piper.usda`: simplified entry to the minimal Piper USD
  closure; no URDF or custom collision cache is required at runtime.
- `assets/`: project-local source, preview and converted assets; ignored by Git.
- `docs/architecture.md`: system boundaries and dependency direction.
- `docs/migration.md`: ordered migration with acceptance gates.
- `docs/trajectory.md`: temporal alignment, fields and persistence guarantees.
- `docs/lerobot_export.md`: reviewed data selection, conversion and acceptance
  contract.
- `docs/newton_15_audit.md`: public API changes and physics acceptance gates.
- `docs/rlinf_delta_audit.md`: disposition of all 19 old RLINF source changes.
- `docs/rtc_rollout.md`: RTC execution, rendering and publication acceptance
  contract.
- `docs/offline_cfg.md`: RTC rollout-to-CFG artifact contract, readiness and
  remote launch gate.
- `docs/todo.md`: deferred decisions with long-term risk.
