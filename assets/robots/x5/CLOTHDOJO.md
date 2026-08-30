# ARX X5A asset provenance

`robodojo/ARX.usd`, `X5A.urdf` and `meshes/` come from
`RoboDojo-Benchmark/RoboDojo` revision
`cfb06d1dadcdf03ccc923102b0e3f3b0a68dfc43`. The Hugging Face dataset card
declares Apache-2.0. The fixed workstation references the complete upstream
`ARX.usd` twice. Its robot geometry, inertias, fingers, `camera_base`, full-resolution
camera mesh and fixed camera joints therefore stay in one upstream-authored tree.
ClothDojo only overlays the two Kai0 root poses, fixed-base joints, physical X5
limits, home state, collision flags and dataset camera semantics. `x5.xml` remains
the MuJoCo retarget model; it is not the visual source for the workstation USD.

The dark arm parts, fingers and wrist-camera mount retain RoboDojo's bound
`/X5A/Looks/defaultmaterial` unchanged (linear diffuse 0.02, metallic 0.8,
roughness 0.2). The prior custom blue-gray overrides are removed. The white
link3 and wrist-camera appearance fallbacks remain because their upstream
material bindings are unresolved; they are not claimed as restored materials.
The Kai0 integrated plate binds the same material as the camera column, as
requested for this simulation installation.

The source URDF uses placeholder arm ranges. ClothDojo replaces them with the
X5 position limits from `real-stanford/arx5-sdk` revision
`a8890c9bae94464abd1cb7c5e4da7c4a62104a3a`, whose `LICENSE` is MIT. The
home configuration and semantic attach frame are migrated from the old
ClothDojo project. The fixed simulation installation uses
`OpenDriveLab/kai0`'s complete `base-integrated.STEP`; the arm origins are aligned
to mounting-hole centers measured directly from that complete CAD, rather than
from the separate-base drawing. The previous Kai0 camera bottom bracket is not
used. The third-person camera is the old project's complete RoboDojo
`camera_stand/00000` D455 assembly with only its separate 90 x 300 mm base plate
removed. Its 66.1 mm foot is centered on the outer four holes of Kai0's central
six-hole region and seated on the integrated plate.

The two patterns are not mechanically compatible: the RoboDojo foot is a
54.1 mm square four-hole pattern, while Kai0 uses a 20 + 30 mm by 70 mm six-hole
pattern, leaving an 8.21 mm best-fit maximum hole-center residual. This placement
is an explicit simulation assembly inference, not a claim that the real parts
bolt together without an adapter.

## Dynamics provenance

The mechanical links retain the complete RoboDojo USD's mass, center of mass,
diagonal inertia and principal axes without substitution. Reconstructing their
full inertia tensors in link coordinates agrees with the accompanying
`X5A.urdf`; all nine inertial bodies per arm have positive, physically admissible
inertia eigenvalues. The Stanford SDK's `X5.urdf` has different link dimensions
and inertias, so its inertias are not mixed into this X5A assembly.

The actuator baseline is RoboDojo's configuration for this exact `ARX.usd`,
not the generic drive defaults inside its exported USD:
[RoboDojo x5.py, revision 76e16bf8feb6e5934ddb291386bf531ff826c621](https://github.com/robodojo-benchmark/RoboDojo/blob/76e16bf8feb6e5934ddb291386bf531ff826c621/env/robot_manager/robot_config/x5.py).
The external asset package retains `provenance/robodojo_x5.py` and the upstream
Apache-2.0 license (`LICENSE.robodojo`); Git records the pinned source URL above.
The six revolute joints use
4400 N m/rad stiffness, 40 N m s/rad damping, 100 N m simulated effort limit,
5 rad/s simulated velocity limit and 0.01 kg m² armature. Finger drives retain
2300 N/m stiffness, 100 N s/m damping and 100 N effort limit. These are simulation
controller settings, not manufacturer-verified motor performance.

USD angular drive gains are multiplied by pi/180, angular limits and velocities
by 180/pi; effort and armature are not angular-unit-scaled. The installed Newton
PhysX schema resolver performs the inverse angular conversions on import.
`build_workstation.py` embeds the actuator settings; the tracked `x5.usda` also
fixes armature and velocity overrides so rebuilding the external payload cannot
discard the audited parameters. No runtime robot-specific dynamics override is needed.
Importing the velocity value is not proof of an enforced speed constraint:
Newton's current `SolverMuJoCo` does not consume `joint_velocity_limit` as a
hard joint-speed limit.

The SDK X5 controller's torque limits [30, 40, 30, 15, 10, 10] N m and velocity
limits [5, 5, 5.5, 5.5, 5, 5] rad/s are research controller safety settings. They
are retained as audit comparisons only, not substituted into the same-USD
simulation baseline. The previously selected SDK position ranges remain intact.

## Wrist camera optics

The physical `camera` bodies in `ARX.usd` do not contain Camera prims. At the
user's request, both wrists use **RoboDojo's complete simulation preset**, not
the previously evaluated manufacturer-nominal RGB optical frame. Offline
provenance is `provenance/robodojo_wrist_camera.json`; runtime reads only the
fixed `x5.usda`. `_author_camera` rebuilds the same definition without a network
connection or an old-project dependency.

The source `Assets/Robots/x5/robot_config.yml` belongs to the separately pinned
asset revision `c037c1b3183a030724e694d69a08cb62369ed285`; the code template and
camera runtime belong to revision `76e16bf8feb6e5934ddb291386bf531ff826c621`.
The cached source ARX USD and the installed ARX USD are byte-identical.

| Setting | Fixed value |
| --- | --- |
| Parent body | Each arm's `camera` |
| Local position | `(0,0,0.001)` m |
| Local rotation | intrinsic XYZ Euler `(0,-80,-90)` degrees |
| Convention | USD camera -Z forward, +Y up; PINHOLE child is identity |
| Focal / horizontal / vertical aperture | `13 / 20.955 / 15.71625` |
| Nominal raster | `640 x 480` |
| Horizontal / vertical field | approximately `77.735 / 62.304` degrees |
| Clipping | `(0.0001,10)` m |

No extra 180-degree output roll, ROS conversion, mirror or centering crop is
applied. The source pose is already upright at home. Both arms use identical
local settings, and the grasp center projects near pixel `(320,289)` at home.
This virtual camera is near the housing center, rather than the physical D435
RGB lens; its wider field and additional downward tilt are explicit simulation
choices, not a manufacturer calibration claim. Clipping is a renderer setting,
not a sensor measurement range.

Home, attach, physical camera/holder and gripper geometry, materials, inertia
and third-person cameras remain unchanged. The physical mesh registration
landmarks are retained in the provenance JSON for geometry checks only; they
do not define the selected virtual optical frame. The source camera/holder
rear-hole mismatch (45 mm versus 38.4 mm) remains unresolved.

Native USD review honors this projection. All X5 apertures are now 4:3, so the
existing policy renderer again uses its historical horizontal-as-vertical FOV
branch; this asset change does not fix that separate benchmark compatibility
boundary or pass USD clipping through the runtime CameraSpec. Do not equate
native review images with policy-renderer pixels.

## Known unresolved upstream parameters

- Each `camera_base` and `camera` has no authored mass or center of mass in the
  USD and no inertial record in the URDF. Their upstream diagonal inertia is
  uniformly 0.0001 kg m², without a validated physical source. No fabricated mass
  or inertia is added; importer inference is not certified asset data.
- The URDF/USD finger speed limit is the generic 1000 m/s value, and the matched
  RoboDojo runtime configuration supplies no replacement. SDK gripper motor
  torque cannot be copied to a prismatic finger force without a transmission model.
- RoboDojo's runtime example disables gravity. Reusing its actuator parameters
  does not change ClothDojo's gravity policy or prove stability under gravity;
  that must be validated separately by simulation.
