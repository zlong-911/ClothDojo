# Migration from newton-cloth-rl

Migration is behavior-first. A file is copied only when it is the smallest
implementation that satisfies a migrated acceptance test.

## Phases

1. **Runtime baseline** — pin Newton 1.5 and the training dependency set;
   establish smoke tests without compatibility shims.
2. **Newton environment** — migrate one `shirt_flatten` scene and validate reset,
   controlled stepping and finite execution under Newton 1.5.
3. **Trajectory dataset** — define the fixed state/action format and keep its
   Newton adapter outside simulation.
4. **Rollout** — run a frozen policy, including RTC sampling, through the
   environment and publish trajectories through one explicit exporter.
5. **Benchmark** — define randomization cases and paired topology/texture
   evaluation suites independently of training.
6. **Data agent** — expose π0.5, STEAM and offline CFG through the small adapter
   over the pinned upstream RLINF API; address online environment registration
   upstream before integrating RLINF rollout.
7. **Collection** — migrate local teleoperation first; add review, HIL and
   cluster control only when their separate workflows are required.

## Explicitly not migrated

- `patches/` and setup-time source mutation;
- `data/runtime/` and nested virtual environments;
- historical experiment launchers and one-off monitoring scripts;
- accumulated outputs, caches and local model copies;
- Newton 1.4 compatibility paths.

## Current status

- [x] Independent ClothDojo repository created.
- [x] Newton 1.5 dependency baseline declared.
- [x] Newton 1.5 source delta audited and deterministic CPU smoke entry added.
- [x] Core Newton 1.5 environment resolved and deterministic CPU smoke passed.
- [x] Full training requirements resolved and `env check --training` passed.
- [x] Fixed trajectory dataset, strict reader and atomic writer implemented.
- [x] Newton-to-trajectory conversion isolated in a one-way adapter and tested
  with independent multi-world array copies.
- [x] Imported the required `000004` collection through a one-way validated
  migration into `data/collections/shirt_flatten`: 690 complete trajectories,
  two incomplete records, 668 annotations and 690 four-camera review exports.
  All manifests now use the strict ClothDojo schema and no old-project path is
  a runtime dependency.
- [x] Reviewed `000004` trajectories and v3 review exports resynchronized;
  bad-complement planning, v3 read-only adaptation, pinned LeRobot v2.1 writing,
  explicit episode extent metadata and atomic publication implemented.
- [x] CFG's accepted 375-episode expert set and 264-episode rollout pool cross
  an atomic v3 indexed-media boundary into project-owned v2.1 datasets. Camera
  clocks, source hashes and the 242-episode rollout train split are explicit;
  no old-project path is needed at runtime.
- [x] Full 1,288,298-frame reviewed LeRobot v1 export published and reloaded
  through pinned LeRobot; the active SFT run uses this immutable artifact.
- [x] Published the v2 reviewed export with the 65-frame temporal floor: 719
  episodes and 1,287,887 frames. The active v1 training input remains immutable;
  exact retained MP4s are hard-linked while Parquet indices and metadata are
  rebuilt and reloaded independently.
- [x] Minimal garment source, front/back preview, accept/reject and converted
  storage contract implemented independently of the old web service.
- [x] Old garment assets copied without modifying the source project: 35 source
  archives, 33,158 front/back candidates, 2,485 decisions and 757 converted
  garments now use the ClothDojo layout and names.
- [x] Preserve the old four-panel web design for teleoperation, trajectory
  review, HIL and asset curation. Its DOM/CSS/interaction code is now local
  static source, with a small view-model adapter over ClothDojo catalogs.
- [ ] Complete the local collection workflow. Asset decisions, trajectory
  browsing, revision-guarded annotations, HIL rollout browsing and immutable
  intervention plans use ClothDojo contracts. Local GPU leases, the fixed Isaac
  Teleop/CloudXR supervisor, Newton 1.5 IK/control and canonical recording are
  implemented. HIL full-state cold restore, source-hashed intervention recording,
  terminal-state continuation leases, Y/B single-arm Home, asynchronous grasp
  haptics and teleop NOCS display are migrated. Asset conversion/deletion and
  concurrent Quest acceptance remain. None of the old cluster or runtime modules
  is a dependency.
- [x] Piper's 13-layer USD closure lives under the simplified `piper.usda`
  entry. The old cache geometry was migrated once into an authored USD layer;
  the runtime uses Newton 1.5's public importer and does not load the old cache.
- [x] Minimal Newton 1.5 `shirt_flatten` model builder implemented with a
  single-world template, zero-spacing replication, public Style3D construction
  and `ArticulationView` selection.
- [x] Minimal runtime, expert four-seed/two-ring connected-patch grasp and
  selected-world reset implemented behind one `clothdojo sim check` acceptance
  entry. Attach/release thresholds and the gripper's 2.5% soft-limit margin are
  preserved.
- [x] Exact pinned environment builds and steps the `shirt_flatten` scene.
- [x] Pin the Newton 1.5.0 fork commit containing the Style3D
  `numpy.arctan2`/`numpy.power` corrections; no site-package mutation or NumPy
  upgrade is required.
- [x] Move the old VF/EE cloth self-friction into the fork's formal Style3D
  collision solve, expose the 63-candidate capacity publicly and enable the old
  `mu=0.35`, `epsilon=0.01`, untangling-suppression configuration in ClothDojo.
- [x] Local `shirt_flatten` runtime passes reset, control and finite-state GPU
  gates; bitwise repeatability is reported as a diagnostic, not required for
  Style3D compatibility.
- [x] Remove visual USD shapes from the physics-only importer. The 69 native
  colliders remain present but disabled; the active envelope is the migrated
  126-shape expert set (99 cloth/external-contact shapes plus 27 self-collision
  proxies).
- [x] Resolve the intermittent OpenUSD import status-139 failure with its
  supported single-thread Work/TBB setting. The minimal importer passed 8/8
  fresh processes and the ordinary full GPU CLI passed 3/3 without debug
  timing or global CUDA synchronization.
- [x] Old RLINF modifications audited against current upstream 0.4.0.
- [x] Minimal upstream RLINF SFT adapter established.
- [x] Offline CFG data contract, exact advantage-coverage preflight, guarded
  expert/rollout config and narrow registered RLINF extension implemented.
- [x] Single-scenario RTC scheduling, real Newton three-camera rendering,
  trajectory persistence and transactional LeRobot v2.1 publication pass a
  real-GPU hold-policy acceptance test. Its output is explicitly ineligible for
  training; old fixed asset/GPU/episode launchers remain read-only evidence.
- [x] Garment-specific initial-state pool generation, seeded free-drop dynamic
  relaxation, strict table-contact validation and arbitrary particle restore
  pass a real-GPU acceptance check.
- [x] RLINF SFT service, deterministic seen/unseen asset selection, per-asset
  initial-state pools, multi-GPU worker queues, teleoperation-layout RTC
  trajectories and colocated offline LeRobot replay are implemented behind
  stable commands. The earlier single-asset replay/export path passed a real-GPU
  six-frame acceptance check; the new multi-asset queue still needs its remote
  acceptance run.
- [x] RLINF's approximate RTC correction is replaced at the project adapter
  boundary by LeRobot's released guided RTC correction without modifying either
  dependency. A same-checkpoint, same-state 400-step GPU run completed with
  bounded joint velocity and a published three-camera LeRobot episode.
- [x] Offline RTC now uses synchronous five-step replanning with zero inference
  delay. On the same checkpoint and initial state, guided RTC reduced command
  step p95 from 0.640 to 0.308 and state jerk p95 from 0.0171 to 0.0098 versus
  independent chunk-5, while retaining the cloth grasp for 342/400 frames.
- [x] Benchmark evaluation is a separate no-trajectory workflow with a strict
  protocol, asset-hashed canonical targets, device-batched deformable/rigid
  metrics, release/speed/dwell success, compact per-asset publication and exact
  checkpoint/mode pairing. The old evaluation launchers are evidence only;
  their schemas and fixed asset lists were not copied.
- [ ] Connect the newly trained accepted SFT checkpoint and publish the first
  real policy collection with `training_eligible: true`.
- [x] Owned STEAM advantage labeling produces atomic tagged labels and
  provenance consumed by the CFG preflight without importing the old repo.
- [ ] Owned STEAM value-training config reproduces the accepted evaluator.
- [ ] π0.5 training acceptance test passed.
- [x] Guarded OpenPI/JAX Pi0.5 SFT entry implemented with exact reviewed-v2
  artifact checks, mandatory Orbax initialization, FP32 update/EMA validation,
  fixed-batch overfit and shuffled pilot gates. Formal 40k acceptance remains
  pending the parity gate and completed run.
- [ ] STEAM acceptance test passed.
- [ ] Benchmark generation acceptance test passed.
