# Collection console and XR runtime

ClothDojo preserves the validated teleoperation path from `newton-cloth-rl`:

```text
Pico 4 Ultra / NVIDIA Isaac Teleop web client
  ↕ CloudXR WSS, media and OpenXR input
Isaac Teleop + Isaac Kit RTX
  ↕ same-process render bridge
Newton 1.5 simulation, IK, grasp and trajectory recording
```

Newton is the only owner of physics and canonical actions. Isaac owns XR
composition and RTX display but never steps a second physics scene. The web
console is a control plane: it allocates a GPU slot, starts or stops a runtime,
publishes the official client URL and exposes bounded logs. XR poses and Newton
state never pass through the console server.

## Environment and process boundary

Developers create and activate only the project `.venv` declared by
`requirements.txt`; `clothdojo collect serve` is the only console entry point.
The collection YAML contains no Python executable, library directory, web
client directory, temporary runtime root, EULA switch or CloudXR port. Those
are installation details owned by one fixed supervisor and its bootstrap.

This operationally behaves as one project environment, while preserving the
necessary process isolation: Isaac Sim 6 requires Python 3.12 and carries
Newton 1.0/Warp 1.14, whereas ClothDojo requires Python 3.11 and Newton
1.5/Warp 1.16. Installing both sets into one `site-packages` directory would
silently replace ClothDojo's physics stack, so the supervisor invokes Isaac as
an application executable rather than exposing another Python environment to
workflow code or users.

The fixed application bundle overlays Newton 1.5/Warp 1.16 in its own Python
3.12 environment and starts a minimal Kit experience that deliberately omits
`omni.warp.core`. This preserves the old same-process GPU Fabric bridge without
allowing Isaac Sim's pinned Newton 1.0 package to own physics. The application
bundle is machine provisioning, not a project workflow environment.

Physics-only processes set the OpenUSD startup default to one worker before
importing pxr. The Isaac entry point also uses SimulationApp's public
`limit_cpu_threads=1`: SimulationApp otherwise overwrites the environment with
its 32-thread default before USDPhysics starts, reproducing a native descriptor
traversal race. This restriction affects host scheduling, not CUDA Graph GPU
parallelism. Measured 1-thread and 32-thread sessions had the same low frame
cadence, so CloudXR low-FPS pacing is configured separately rather than treating
TBB as the rendering bottleneck.

The supervisor launches one fixed ClothDojo teleoperation entry with a
server-authored session plan; browsers cannot submit commands, paths,
environment variables or ports.

A session plan contains only resolved project identities and paths:

- mode: `collection` or `intervention`;
- profile, garment and GPU identities;
- the canonical trajectory output root;
- fixed CloudXR ports and the public client endpoint;
- for HIL, an immutable rollout source plus one or more sample indices.
- for continuation, an immutable leased set of terminal trajectory states.

Collection and HIL use the same GPU lease, supervisor and teleoperation loop.
HIL differs only in its validated reset source and intervention provenance.
Continuation profiles use the same loop and record source-profile, episode,
terminal sample and manifest hashes in every result.

The CUDA teleoperation runtime captures fixed-allocation Newton physics and IK
graphs by default. Capture failure is fatal on GPU rather than a silent eager
fallback; CPU-only tests retain the eager path. The live status records both
graph identities and rolling loop/phase timings so performance acceptance can
verify the executed path. IK is submitted only when Cartesian targets change,
matching the validated old-project control loop. CloudXR 6.2 is given a fixed
20 FPS `max-fps` pacing hint because this application renders materially below
the 72/90 Hz headset rate. The hint lets the runtime schedule that low-rate
producer directly instead of repeatedly missing headset display deadlines; it
does not change the headset refresh rate or the Newton simulation timestep.

## Data ownership

The console does not introduce another database of trajectories or assets.

- `assets/garments/selection.json` owns garment accept/reject decisions.
- Canonical trajectory directories and their manifests own episode state.
- `trajectories/annotations/*.segments.json` owns review decisions.
- Rollout `meta.json` files own policy-generated episode membership.
- Intervention trajectory manifests own their rollout breakpoint provenance.
- SQLite below `outputs/collection/` owns only disposable GPU leases, process
  identity and job status.

Formal trajectories, rollouts and interventions remain below `data/`. Runtime
logs, thumbnail caches and the disposable control database remain below
`outputs/`. A service restart can rebuild every data-facing index from the
formal files.

## HTTP boundary

The console is one local FastAPI service and preserves the old browser
workspace's DOM, CSS, theme, bilingual labels, keyboard review controls and
four-panel layout. The 80 KB inline source is mechanically separated into
HTML, CSS and JavaScript; this changes code ownership, not the design.

The old FastAPI/cluster implementation is not copied. A dedicated workspace
view-model translates ClothDojo's canonical catalogs into the stable browser
response shapes, while the backend remains split by resource:

- assets are filtered and paginated;
- canonical trajectory and rollout catalogs remain independently queryable;
- immutable images and video support normal browser caching/range requests;
- selection and annotation writes are atomic;
- annotation updates use a revision digest to prevent lost concurrent edits;
- live sessions and logs are separate from formal-data indexes.

Review-source and episode indexes are cached against directory metadata and
invalidated after an annotation write or source-directory change. This keeps a
full 690-trajectory collection compatible with the preserved two-second browser
poll without weakening on-disk validation or making SQLite authoritative.

There is no second Node environment or generated frontend lock. Asset curation,
trajectory review, teleoperation and HIL remain four panels in one workspace.
Collection teleoperation uses the fixed supervisor and canonical Newton 1.5
writer. HIL cold restore and terminal-state continuation are implemented with
strict runtime/asset/source hashes. Asset conversion and deletion remain
disabled instead of silently calling old-project scripts.

The live controller preserves the operator-facing controls: Y/B return the
left/right arm independently to startup Home; release then squeeze is required
to resume tracking. A fixed-allocation GPU probe drives continuous per-hand
grasp-region vibration through asynchronous pinned-memory readback. The render
cloth carries rest-mesh NOCS as a vertex primvar, so color identity remains
stable while the simulated particles deform.

## Migration order

1. Asset selection and trajectory review, which need no GPU runtime.
2. Local GPU leases and the fixed Isaac Teleop/CloudXR supervisor boundary.
   Implemented with process-identity reconciliation, fixed ports and bounded
   logs; concurrent Quest acceptance remains.
3. Newton 1.5 teleoperation and canonical recording. Implemented and covered by
   real-GPU IK/recording and Isaac application-bundle startup checks; headset
   input remains the hardware acceptance gate.
4. HIL rollout browsing, cold restore and intervention recording through the
   same runtime. Implemented; headset acceptance remains.
5. Multi-host aggregation only after the local workflow passes concurrent
   Quest acceptance; the old AFO-specific cluster manager is not a migration
   unit.

Each step lands with a user-visible acceptance test. The old repository remains
read-only evidence and is never imported by ClothDojo.
