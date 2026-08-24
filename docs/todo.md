# Deferred decisions

## Newton 1.5 GPU acceptance

Newton 1.5's anisotropic Style3D cloth builder calls `numpy.atan2`, an alias
provided by NumPy 2 but not the current NumPy 1.26.4 pin. The CPU Newton smoke
passes. A locally proposed `numpy.arctan2` upstream fix lets the GPU scene pass
this build stage without changing NumPy, but the released package is not fixed
yet.

The local validation completes a finite GPU step and an exact reset. Style3D
does not produce bitwise-identical repeated steps, which is accepted as a
numerical property rather than treated as a package compatibility failure.
Rerun the clean-install GPU acceptance after the upstream builder fix lands.
Do not add a global alias or setup-time patch.

## Benchmark statistical reproducibility

GPU Style3D execution is not bitwise repeatable: parallel numerical reduction
differences can be amplified by cloth contact dynamics. Before publishing the
benchmark, define fixed scenario manifests and distribution-level tolerances
for task metrics across repeated runs. Do not require exact particle-array
identity or serialize one implementation-specific reduction order into the
dataset contract.

## RLINF online environment registry

Current upstream RLINF 0.4.0 loads extension modules and supports registered
models, but environment validation and action preparation still use a closed
enum. Before online RLINF rollout is migrated, propose one small upstream
registry covering environment construction and action preparation. Do not use
an import-time monkey-patch as an interim architecture.

The completed source-delta review and fork gate are in
`docs/rlinf_delta_audit.md`.

## RLINF CFG OpenPI data arguments

Pinned RLINF 0.4.0's CFG model builder calls `get_openpi_config` without the
configured model path or `actor.model.openpi_data`; its CFG dataloader passes
the model path but still drops `openpi_data`. That would silently restore the
physical-ALOHA axis conversion and ignore ClothDojo's external normalization
statistics.

ClothDojo currently uses the public model registry, `RLINF_EXT_MODULE`, and one
`FSDPCfgWorker.build_dataloader` subclass to pass the missing arguments while
retaining upstream model, dataset, optimizer and checkpoint behavior. Recheck
this gap whenever the RLINF pin moves. Remove the project extension once
upstream CFG passes the same data arguments; do not turn it into a general
compatibility layer.

## Newton 1.5 cloth physics acceptance

The old project implemented cloth self-friction by importing private Style3D
kernels and replacing internal collision objects. ClothDojo will not migrate
that coupling. Newton 1.5 also gives the old `shape_material_kf = 1e3` value a
new MuJoCo friction-reference meaning, so carrying the numeric value would not
preserve behavior.

Before collecting new teleoperation or benchmark base data, compare the pinned
1.5 public baseline against representative old episodes for contact stability,
self-contact sliding, grasp retention and final cloth spread. If public Newton
cannot meet the measured requirement, produce a minimal upstream reproducer;
do not restore private imports or setup-time patches. The API audit is in
`docs/newton_15_audit.md`.

## RTX renderer boundary

Isaac Sim 6.0 embeds Newton 1.2 while ClothDojo targets Newton 1.5. Installing
both into one Python environment would create an ambiguous physics stack. Before
benchmark randomization is migrated, decide whether Newton's current renderer
meets topology/texture benchmark needs. Retain Isaac only if its added rendering
capability is measured to be necessary; then freeze it as an external container
that exchanges trajectory and rendered datasets with ClothDojo.

## Newton deformable smooth shading

Newton 1.5 interpolates authored normals for static Mesh shapes, but its
deformable triangle-mesh camera path still exposes only face normals. The same
gap remains in the current upstream source. ClothDojo therefore maintains one
small render adapter: a non-colliding visual garment Mesh follows the physics
particles, recomputes vertex normals on the GPU and updates the tiled camera's
copied normal buffer. Remove this adapter when Newton exposes deformable vertex
normals through a released camera API.

## Asset content identity

The initial asset store intentionally uses readable asset IDs without a hash
registry. This keeps curation and conversion small, but editing an installed
USDC in place could make old data and current geometry share the same ID. Before
publishing the first benchmark, decide whether immutability alone is sufficient
or whether trajectory manifests should record the converted file hash. Do not add
an asset-set or revision system before that concrete need exists.

## OpenPI OpenCV package ownership

The released `rlinf-openpi` dependency chain requires `opencv-python`, while
OpenPI's pinned LeRobot revision requires `opencv-python-headless`. OpenPI's own
resolved environment contains both at 4.11.0.86, so ClothDojo currently pins
that same pair rather than allowing two independently changing `cv2` builds.
The distributions still overlap file ownership. Before relying on OpenCV in a
long-lived service environment, request that upstream OpenPI and LeRobot agree
on one distribution and then remove the redundant package; do not solve this
with a local shim package or fork.
