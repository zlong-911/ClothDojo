# Trajectory dataset format

The trajectory dataset stores state/action sequences produced by collection or
rollout. It is independent of Newton Python objects, RLINF, LeRobot and
rendered observations. Simulation does not depend on this format; the one-way
Newton adapter copies runtime state into these plain NumPy records.

## Temporal alignment

A trajectory contains states `s0 ... sN` and actions `a0 ... aN-1`:

```text
s0 --a0--> s1 --a1--> ... --aN-1--> sN
```

The training sample at index `t` is `(state_t, action_t)`. The terminal state
`sN` is retained for transition analysis but is not a training sample. State
and action indices are contiguous `int64` values starting at zero.

## Directory layout

```text
trajectory/
  manifest.json
  states.npz
  actions.npz
```

`manifest.json` records schema identity, completion status, timebase, shapes,
scene fingerprints and counts. `states.npz` has `N + 1` rows and `actions.npz`
has `N` rows. A complete zero-action trajectory contains one state and may omit
`actions.npz`.

The current manifest has `format_version: 1`. The reader accepts only this
fixed format and requires an explicit `coordinate_frame`. Older
`newton-cloth-rl` manifests are converted offline before use; compatibility
branches do not live in the runtime reader.

## Arrays

State arrays use `float32` except grasp flags, counts and pin indices, which use
`int32`:

- physics: `joint_q`, `joint_qd`, `body_q`, `body_qd`, `particle_q`,
  `particle_qd`;
- gripper state: `gripper_open_state`, `grasp_attached`,
  `grasp_previous_open`, `grasp_pin_count`, `grasp_pins`, `grasp_pin_local`.

Action arrays are `joint_target` and `gripper_open_command`, both `float32`.
Shapes are fixed within a trajectory and recorded in the manifest.

## Commit and validation behavior

The writer buffers bounded chunks, then atomically publishes the NPZ archives
before changing the manifest status to `complete`. An interrupted recording is
therefore never mistaken for a complete trajectory.

The reader rejects unknown fields, wrong dtypes or shapes, non-contiguous
indices and broken `N + 1`/`N` alignment. Training validation checks finite
values only for paired state/action ranges, intentionally excluding the
unpaired terminal state.

Rendered images and policy-specific dataset fields belong to explicit export
formats. They are not added to the trajectory dataset merely to serve one
trainer or benchmark.
