# Architecture

## System boundary

```text
Agent bootstrap
  teleoperation → expert dataset → RLINF π0.5 → RTC rollout dataset
                          │                         │
                          └──── STEAM labeling ─────┴→ CFG policy
                                                        │
                                                        ▼
Benchmark
  frozen agent → factor randomization → trajectory generation → evaluation
                                      ├─ topology
                                      └─ texture
```

Agent bootstrap produces a versioned policy artifact. Benchmark generation
consumes that artifact through a policy interface; it does not know how the
policy was trained.

## Dependency direction

```text
CLI ──→ diagnostics | sim check | train | rollout | benchmark | collect

simulation ──→ assets
trajectory format ──→ NumPy + files
trajectory/Newton adapter ──→ simulation + trajectory format
data export ──→ trajectory annotations + rendered reviews + pinned LeRobot
rendering ──→ simulation
rollout ──→ simulation + rendering + policy + trajectory format
benchmark ──→ rollout + metrics
training/RLINF adapter ──→ exported dataset + pinned RLINF
collection ──→ simulation + rendering + trajectory format
```

Only modules with a migrated user-facing workflow are created. There is no
generic `contracts`, `workflows`, `agent` or registry layer. Simulation owns
physics state and stepping and never imports the trajectory dataset. The
one-way Newton adapter copies state into plain NumPy records when collection or
rollout asks for persistence.

RLINF must not import ClothDojo application modules. Model-specific behavior can
use RLINF's model registry and `RLINF_EXT_MODULE`. Offline CFG uses that public
extension boundary to pass ClothDojo's OpenPI data settings into the upstream
model and dataloader; it does not mutate installed RLINF files. Its environment
dispatch is still a closed enum, so online rollout integration is deliberately
not hidden behind import-time monkey-patching: ClothDojo will first propose a
narrow environment registry upstream. A fork remains a last resort subject to
the gate in `docs/rlinf_delta_audit.md`.

The agent-training interchange initially uses the LeRobot revision pinned by
RLINF itself. ClothDojo's trajectory dataset is independent of LeRobot; an
explicit exporter converts between them. The trajectory format is a storage
definition, not a central domain layer. Its `s_t --a_t--> s_t+1` alignment and
persistence rules are defined in `docs/trajectory.md`. Rendered observations
and trainer-specific columns live in explicit exports rather than expanding
the stored physics state.

Expert and RTC rollout exports meet at the offline CFG boundary. Both remain
ordinary pinned-LeRobot datasets; STEAM advantage labels are tagged metadata
beside each dataset. `docs/offline_cfg.md` owns their compatibility and launch
gate, while the rollout module continues to own policy execution and trajectory
publication.

RTC execution is independent of RLINF internals. A small scheduler maintains at
most one asynchronous replan, measures its latency in control steps and resumes
the returned chunk at that measured offset. The policy boundary receives plain
NumPy observations plus the previous model-space chunk and timing context. This
lets a remote accepted-SFT service implement RLINF/OpenPI inference without
making RLINF an environment or simulation dependency.

The physics scene retains only collision shapes. Rollout rendering builds a
second, topology-identical Newton model with visual shapes and copies the live
body/particle state into the public camera path. Thus rendering cannot add 446
visual-only shapes back into collision detection. The first rollout slice owns
three fixed RGB cameras; richer benchmark appearance randomization remains a
later rendering workflow.

Reviewed legacy data crosses one additional, temporary format boundary. A
narrow read-only adapter streams the old LeRobot v3 review cache with
PyArrow/PyAV, while the pinned LeRobot v2.1 writer owns the final dataset. Bad
labels are interpreted before the writer boundary; trainer selection metadata
lives in `meta/clothdojo.json` rather than being added to the physics
trajectory schema. See `docs/lerobot_export.md`.

## Environment

`requirements.txt` is the only environment definition. Newton 1.5, Warp 1.16,
MuJoCo/MJWarp 3.11 and USD Core 26.3 move together as one tested unit. Only
Newton's `sim` extra is installed: ClothDojo reads converted USD at runtime and
does not pull the unrelated Open3D/CoACD/mesh-conversion stack. RLINF uses
current upstream 0.4.0 while Torch remains 2.7.1, the explicit requirement of
`rlinf-openpi` and a supported Newton 1.5 baseline. Python 3.11 is required
because OpenPI's `dm-tree` dependency is unavailable for Python 3.12. Formal
artifacts record the complete resolved package set so a historical run remains
diagnosable even when indirect dependencies later change.

The few extra training pins reproduce OpenPI's own tested compatibility
overrides (`ml-dtypes`, `tensorstore`) and resolve its LeRobot boundary
(`fsspec`, Hugging Face Hub, Zarr and OpenCV). They are not a copied lockfile.
OpenPI and that LeRobot revision currently declare the full and headless OpenCV
distributions respectively; ClothDojo pins both to OpenPI's identical 4.11
build and tracks removal of that upstream packaging overlap in `docs/todo.md`.

Isaac Sim 6.0 ships a different Newton stack. It must not be installed into the
core environment. If retained for offline RTX rendering, it is an external tool
that consumes a trajectory dataset and emits a rendered dataset.

## Storage

Assets are physically stored below the project-local `assets/` directory and
are ignored by Git. Garments use one deliberately small layout:

```text
assets/garments/
  source/
    archives/                    original dataset archives
    index.jsonl                  source lookup used during conversion
    previews/<asset_id>/
      front.png
      back.png
  selection.json                accepted and rejected ID lists
  converted/<asset_id>/
    <asset_id>.usdc
```

The web console reads the two previews, updates `selection.json` atomically and
considers the intersection of accepted and converted IDs collectable. Rejecting
an asset never removes its source or converted file. Conversion writes a hidden
temporary sibling directory and installs the completed result with one atomic
rename; there is no permanent staging or trash area.

Robot and prop packages live under `assets/robots/` and `assets/props/`.
Datasets, checkpoints, caches and run outputs remain separate generated
artifacts and are not mixed with reusable scene assets.

Piper is stored as one 12-layer USD closure at
`assets/robots/piper/piper.usda`. The simulation imports its authored collision
shapes directly and leaves visual geometry to the rendering boundary. The old
URDF companion, source mesh tree and custom Newton collision cache are not
runtime dependencies.

The physics process sets OpenUSD's supported `PXR_WORK_THREAD_LIMIT=1` default
before Newton imports OpenUSD. USD parsing is a one-time setup operation, and
serial Work/TBB traversal avoids a measured native race on the current
high-core-count host without serializing Warp simulation. An explicit external
setting is respected. OpenUSD 26.3 cannot reliably lower this limit after
Newton initializes pxr, so a host application that imports Newton first must
set the same environment variable at process startup; ClothDojo fails clearly
if that ordering contract is violated.

## Newton scene

`shirt_flatten` builds one Piper, table and Style3D garment template, then uses
Newton 1.5's public `ModelBuilder.replicate()` at zero physical spacing. Viewer
offsets may separate worlds visually without changing their physics
coordinates. `ArticulationView` owns batched robot selection and control-array
layout.

Piper remains on `SolverMuJoCo` position targets and the garment remains on
`SolverStyle3D` for the first parity baseline. Style3D is not a supported
`SolverCoupled` participant in Newton 1.5, so the application retains one
explicit two-solver stepping boundary. A VBD coupled scene is only eligible
after it matches the observable cloth acceptance metrics.

That boundary uses two public `CollisionPipeline` instances: a rigid-only
pipeline with `soft_contact_max=0` for MuJoCo, and a full rigid/soft pipeline
for Style3D. This avoids changing state fields temporarily or teaching either
solver about the other solver's contact representation.
