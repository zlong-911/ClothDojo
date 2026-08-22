# RTC rollout acceptance

This workflow proves the local rollout substrate before an accepted SFT
checkpoint is allowed to generate CFG input. It owns one `shirt_flatten`
scenario, asynchronous Real-Time Chunking, Newton observations and two external
artifacts. It does not own policy training or cluster scheduling.

## Command

```bash
.venv/bin/clothdojo rollout check-rtc \
  --scenario configs/rollout/shirt_flatten_rtc_gate.yaml \
  --assets assets \
  --trajectory-output /path/to/rtc_gate_trajectory \
  --lerobot-output /path/to/rtc_gate_lerobot \
  --device cuda:0
```

Both output paths must be absent. Datasets and trajectories stay outside the
repository. A failed LeRobot publication leaves the hidden sibling staging
directory for diagnosis and never installs the requested output root.

## Execution contract

The initial chunk is synchronous. Once its minimum execution horizon is
reached, one policy inference runs concurrently with simulation. When it
finishes, the scheduler measures whole control steps since the request and
starts the replacement chunk at that offset. It retains a bounded delay history
and supplies the maximum observed delay to later policy calls. Only one request
may be in flight; if a response is late, execution holds the last finite action.

Policy state and actions use the exact ALOHA order:

```text
left joint1..6, left gripper openness,
right joint1..6, right gripper openness
```

The policy receives `observation.state`, the three RGB images, the task prompt
and, for replans, the previous model-space chunk, executed horizon, predicted
delay and chunk ID. Arrays are copied at the asynchronous boundary.

## Rendering and outputs

The solver model remains physics-only. The renderer builds a separate
topology-identical visual model and synchronizes its live body and particle
state before using Newton's public tiled camera. The first contract fixes three
`640x480` uint8 RGB views: `cam_high`, `cam_left_wrist` and
`cam_right_wrist`.

The trajectory is published before the LeRobot sidecar that refers to it. The
LeRobot writer then encodes one H.264 video per camera, validates frame counts,
checks v2.1 metadata, reloads the dataset through the pinned library, writes
`meta/clothdojo.json`, and atomically installs the output root.

## Training eligibility

`check-rtc` uses `hold_acceptance`, which repeats the observed arm and gripper
state for an entire action chunk. It validates timing, rendering, physics and
storage only. Both its trajectory manifest and LeRobot sidecar contain
`training_eligible: false`; `clothdojo data check-cfg` rejects that dataset.

The next slice must connect an accepted remote SFT policy, pin checkpoint and
normalization identities in provenance, validate returned model/action chunks,
and pass one reviewed scenario before it may publish
`training_eligible: true`. Pool launch, retries and GPU assignment remain out of
scope until that single-policy boundary is accepted.
