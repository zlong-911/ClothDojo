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
- [ ] Convert required old canonical trajectories offline into the current
  manifest before migrating a dataset workflow.
- [x] Reviewed `000004` trajectories and v3 review exports resynchronized;
  bad-complement planning, v3 read-only adaptation, pinned LeRobot v2.1 writing,
  explicit episode extent metadata and atomic publication implemented.
- [ ] Run the full 1,288,298-frame reviewed LeRobot export after the project
  environment installation completes, then reload it through pinned LeRobot.
- [x] Minimal garment source, front/back preview, accept/reject and converted
  storage contract implemented independently of the old web service.
- [x] Old garment assets copied without modifying the source project: 35 source
  archives, 33,158 front/back candidates, 2,485 decisions and 757 converted
  garments now use the ClothDojo layout and names.
- [ ] Local collection workflow migrated; the old combined web console is not a
  package dependency or migration unit.
- [x] Piper's minimal 12-layer USD closure copied under the simplified
  `piper.usda` entry; native Newton 1.5 collision shapes replace the old custom
  collision-cache loader.
- [x] Minimal Newton 1.5 `shirt_flatten` model builder implemented with a
  single-world template, zero-spacing replication, public Style3D construction
  and `ArticulationView` selection.
- [x] Minimal runtime, connected-patch grasp and selected-world reset implemented
  behind one `clothdojo sim check` acceptance entry.
- [x] Exact pinned environment builds and steps the `shirt_flatten` scene.
- [ ] Land the proposed Newton Style3D `numpy.arctan2` fix upstream and verify
  it from a clean released installation; the local validation reaches a full
  GPU step without changing ClothDojo's NumPy pin.
- [x] Local `shirt_flatten` runtime passes reset, control and finite-state GPU
  gates; bitwise repeatability is reported as a diagnostic, not required for
  Style3D compatibility.
- [x] Remove visual USD shapes from the physics-only importer, reducing Piper
  from 515 mixed shapes to 69 physical shapes.
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
- [ ] Accepted SFT checkpoint connected through the remote policy boundary and
  one RTC rollout published with `training_eligible: true`.
- [ ] Owned STEAM value and advantage-labeling configs produce the tagged
  metadata consumed by the CFG preflight.
- [ ] π0.5 training acceptance test passed.
- [ ] STEAM acceptance test passed.
- [ ] Benchmark generation acceptance test passed.
