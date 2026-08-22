# Newton 1.5 migration audit

## Decision

ClothDojo targets the released Newton 1.5.0 stack directly. The old runtime
stepper is not copied because it combines valid project behavior with Newton
1.4 compatibility code and private Style3D internals.

The first executable acceptance point is:

```bash
clothdojo env check --smoke
```

It builds a tiny model with vector gravity, allocates contacts from an explicit
`CollisionPipeline`, and runs the deterministic public XPBD step path twice on
CPU. The exact requirements environment must pass this before the
`shirt_flatten` GPU scene is accepted.

## Source audit

The following 1.5 changes affect the old implementation:

| Old behavior | ClothDojo behavior |
|---|---|
| Construct and later assign scalar gravity | Use a three-component gravity vector; scalar `ModelBuilder.gravity` is deprecated. |
| Allocate contacts through `Model.contacts()` | Allocate with `CollisionPipeline.contacts()`; the model helper is deprecated. |
| Leave target layout at the legacy default | Set `newton.use_coord_layout_targets = True` before model construction. |
| Depend on default atomic execution | Enable deterministic contact ordering and use solver determinism only where the released public API supports this scene. |
| Pass a reset mask of length `world_count` | Use the 1.5 mask shape `world_count + 1`, whose final element selects global entities. |
| Set `shape_material_kf = 1e3` as a generic contact constant | Do not carry the value blindly: 1.5 maps it to MuJoCo per-contact friction `solref`, changing its physical meaning. |
| Import Style3D collision kernels from `newton._src` | Delete the private self-friction extension. Only public Newton APIs may enter the baseline. |
| Resize internal Style3D broad-phase arrays | Use official capacity controls if exposed; otherwise reproduce overflow on 1.5 and upstream a focused public option. |
| Load render geometry into the physics-only process | Import authored collision shapes only; visual geometry belongs to the later renderer boundary. |
| Let OpenUSD use unrestricted host concurrency | Default OpenUSD Work/TBB to one import thread before Newton loads pxr; GPU simulation remains parallel. |
| Reflect over constructor signatures to support several Newton versions | Call only the pinned 1.5 API. There is no 1.4 compatibility branch. |

Newton 1.5 also fixes Style3D divergence on isolated vertices and memory growth
without CUDA graph capture. Those fixes are adopted through the release; no
local equivalent is retained.

The runtime installs `newton[sim]`, USD Core and Newton USD schemas directly.
The broad `importers` extra is intentionally excluded because all simulation
assets are already USD and no runtime path performs mesh simplification,
convex decomposition or format conversion.

## Scene architecture decision

The first parity baseline remains MuJoCo for Piper and Style3D for garments.
Newton 1.5's `SolverCoupled` cannot contain `SolverStyle3D` because Style3D
does not implement the required public `CouplingInterface`. Switching the cloth
to VBD would enable the new coupled full-surface contact and masked deformable
reset, but would also change the anisotropic material model and trajectory
distribution. That is a later measured experiment, not part of migration.

The new joint-impedance controller is likewise not used in the baseline. It
produces torque commands, while the existing teleoperation and policy contract
commands MuJoCo position targets. Replacing that contract during migration
would conflate an architecture change with a control-system change.

The parts adopted immediately are:

- build one world template and use `ModelBuilder.replicate()` with zero physical
  spacing for every batch size;
- use `newton.selection.ArticulationView` instead of project-owned batched
  joint-index arithmetic;
- consume the Piper USD's native collision shapes instead of the old custom
  collision-cache loader;
- set the unrelated shape `kf` gain to zero explicitly and retain friction in
  `mu` while the baseline uses pyramidal MuJoCo contacts.

The remaining manual MuJoCo/Style3D step boundary is intentional. It should be
one small project runtime, not disguised as a coupled solver or expanded with
version reflection. The rigid phase uses a public collision pipeline with
`soft_contact_max=0`; the cloth phase uses a separate full pipeline. No state
field is temporarily removed to control collision behavior.

## Acceptance boundary

The CPU smoke proves installation and core public API behavior only. A
`shirt_flatten` migration is complete only after the exact pinned environment
passes a GPU test covering model construction, exact reset, controlled stepping
and finite state. Bitwise repeatability is reported for diagnosis but is not a
Style3D compatibility requirement. Trajectory serialization is tested
separately.

Physics parity is evaluated from observable outcomes, not by preserving old
private buffers. Cloth self-friction and the changed MuJoCo friction reference
must be measured explicitly before teleoperation collection or benchmark data
generation begins.

## Release compatibility and numerical behavior

The pinned core environment passes the Newton CPU smoke, but the first
Style3D garment build currently stops inside Newton 1.5's released
`_compute_edge_bending_data`: its anisotropic path calls `numpy.atan2`, which
is unavailable in the project's NumPy 1.26.4 baseline. Newton's current
upstream source retains the same call, while NumPy 2 exposes that alias.

A locally proposed upstream change from `numpy.atan2` to `numpy.arctan2` proves
that the garment build can proceed without changing ClothDojo's NumPy pin. It
is validation evidence only: a clean installation of the released Newton 1.5.0
still contains the failing call, so ClothDojo does not depend on that
site-package edit.

Newton 1.5.0 applies
`SolverMuJoCo(deterministic=RUN_TO_RUN)` to every loaded MuJoCo Warp module.
MuJoCo Warp 3.11.0's sensor module contains one kernel that mixes
`atomic_max` and `atomic_add` reductions on the same output, which Warp 1.16.0
rejects in deterministic mode. The shirt scene has zero sensors, but module
compilation fails before that fact can exclude the kernel. The same code is
still present in the current Newton and MuJoCo Warp upstream sources.

Running the public default solver mode gets through one complete control step,
keeps every value finite and restores reset state exactly. Across two
reset-and-step passes, rigid state differs
by at most about `1e-8`, while Style3D particle position differs by up to
`6.9e-3 m` and particle velocity by up to `1.91 m/s`. Style3D exposes no public
determinism option in Newton 1.5.0.

This lack of bitwise identity is accepted as a property of the GPU cloth
system, not a failed migration. ClothDojo therefore keeps deterministic contact
ordering, uses MuJoCo Warp's public `NOT_GUARANTEED` mode, and treats finite
state plus exact reset as the runtime compatibility boundary. Benchmark
reproducibility will be evaluated from fixed scenario manifests and
distribution-level task metrics rather than exact particle arrays.

Do not patch NumPy, Newton, MuJoCo Warp or Style3D during project setup. Land
the builder fix upstream, then rerun the exact clean-install GPU acceptance
test.

## OpenUSD import stability

The ordinary GPU CLI originally exited intermittently with native status 139,
while faulthandler, GDB and `CUDA_LAUNCH_BLOCKING=1` changed timing enough to
complete. A minimal reproducer reduced the failure to one
`ModelBuilder.add_usd()` call; it did not require Style3D or a simulation step.
OpenUSD stage opening alone was stable, and BLAS/OpenMP thread limits did not
remove the failure. Only OpenUSD's own `PXR_WORK_THREAD_LIMIT=1` control was
stable in isolation.

ClothDojo therefore sets that official OpenUSD default before Newton imports
pxr and imports collision shapes only. If a host imports Newton first without
the environment setting, the scene builder fails before USD parsing and tells
the host to configure process startup; OpenUSD 26.3's Work API cannot reliably
lower the initialized 72-thread limit. The minimal importer passed eight fresh
processes consecutively and the complete ordinary GPU CLI passed three. No
faulthandler, sleep, global CUDA synchronization or source patch is part of the
runtime.
