# Policy evaluation

Evaluation is a benchmark workflow, not a training-rollout option. It consumes
the same fixed initial-state pool and policy interface, but it never publishes
training trajectories.

## Protocol

`configs/evaluation/shirt_flatten.yaml` defines one fingerprinted protocol:

- a fixed simulated-time horizon;
- a fixed metric sampling rate;
- an asset-specific canonical target and explicit planar symmetry;
- normalized deformable and rigid distances;
- release, cloth-speed and consecutive-sample requirements for success.

The checked-in protocol has `stage: calibration`. Its thresholds and horizon
must be calibrated on reference demonstrations and frozen before reporting a
benchmark. Changing any field changes the protocol fingerprint, so results from
different versions cannot be combined accidentally.

For current particles `X` and canonical particles `G`, evaluation first finds
the best table-plane rigid transform of `G` by corresponding-particle
Procrustes alignment. It reports:

```text
d_deform = mean ||X - align(G)|| / canonical_scale
d_rigid  = mean ||align(G) - G|| / canonical_scale
d_total  = 0.65 d_deform + 0.35 d_rigid
reward   = -d_total
```

The `planar_180` variant is evaluated explicitly and the lower total distance
is selected. For pure flattening, `rigid_distance_max: null` means pose is a
diagnostic rather than a success requirement. A later placement/folding task
can enable a rigid threshold without redefining the metric.

Success is absorbing. It is confirmed only after every sampled state in the
configured dwell window satisfies the distance threshold, both grippers are
released, and cloth RMS speed is low. Results record both the first qualifying
sample (`stable_since_step`) and the later confirmation sample
(`confirmed_step`). A policy that is still manipulating at the horizon is a
timeout unless it already achieved confirmed success. Terminal and best-settled
distances plus `success_at_end` remain diagnostics; none can turn a timeout into
success.

This keeps the useful parts of common benchmark practice: a hard horizon,
per-step or macro-step success checks, early-success retention, and separate
`success_once`/terminal diagnostics. It deliberately avoids forced release,
post-time settling, or selecting an arbitrary best frame as the primary result.

## Canonical targets

Each asset in the fixed initial-state pool gets one deterministic target. It is
generated from the asset's exact default pose, with no randomization: the cloth
free-settles on the table while robot-particle collision is disabled. The table
still collides normally, so the robot cannot bias the target or interfere with
unseen assets. Generation requires a full low-speed window after the configured
minimum settle time and has an explicit timeout.

```bash
.venv/bin/clothdojo benchmark make-templates \
  --states data/states \
  --protocol configs/evaluation/shirt_flatten.yaml \
  --gpus 2,3,4,5,6,7 \
  --output data/benchmarks/shirt_flatten_calibration
```

The command is resumable and writes one compact file per asset plus atomic root
metadata. Each template records asset, topology, scene-physics, protocol and
source hashes. A single stored target represents both allowed poses: the metric
constructs the `+180°` alternative around the configured table center, so a
second template is neither stored nor simulated. Templates never come from
training trajectories or the checkpoint being evaluated.

## Conditions and pairing

One condition is exactly one checkpoint plus one inference configuration. Run
it into its own immutable directory:

```bash
clothdojo benchmark run \
  --condition-id step40000_chunk_h10 \
  --scenario configs/rollout/shirt_flatten.yaml \
  --protocol configs/evaluation/shirt_flatten.yaml \
  --states data/states \
  --templates data/benchmarks/shirt_flatten_calibration/templates \
  --policy ws://127.0.0.1:8080 \
  --execution chunk --exec-horizon 10 \
  --gpus 0 --batch-size 4 \
  --output data/evaluations/step40000_chunk_h10
```

Repeat with the other checkpoint or `--execution rtc`. The scheduler derives
policy seeds only from the ordered initial-state batch and replicate index, so
the same case layout has the same stochastic seed across conditions.

Use the same execution horizon for chunk and RTC when the goal is to isolate
RTC guidance. Comparing a full 50-action chunk with RTC at 10 actions is also a
valid system comparison, but it changes replanning frequency as well and must
be named and interpreted accordingly. The current runner is synchronous and
zero-delay; a real-time RTC claim requires a separately frozen latency model.

Summarize in baseline-first order:

```bash
clothdojo benchmark summarize \
  --condition data/evaluations/step40000_chunk_h10 \
  --condition data/evaluations/step40000_rtc_h10
```

The command rejects mismatched protocols, initial-state pools or case sets. It
reports micro success rate, equal-asset macro success rate, failure-censored
restricted mean completion time, success-only median time, terminal distance,
terminal cloth RMS speed, success-at-end rate, and paired win/loss/tie counts.
The primary table should use success rate at the fixed horizon and time to
confirmed success; dense reward is diagnostic. Terminal distance and speed
together distinguish a settled near-miss from a policy that is still moving at
the cutoff without changing the fixed-horizon verdict.

## Runtime path

Same-asset initial states share one Newton 1.5 replicated model. Physics uses
the existing CUDA Graph, cameras use one tiled sensor update, and policy input
uses one batch request. Metric sampling stays on Newton's CUDA stream through a
zero-copy Warp-to-Torch view. Only four scalar curves at the protocol sampling
rate are copied after the rollout; full particle trajectories are not written.

Each GPU worker publishes one compact `assets/<asset_id>.npz`. The coordinator
then atomically writes `results.jsonl` and marks `meta.json` complete. Completed
asset files are reused after interruption, while root metadata is written only
by the coordinator.

## Reference behavior

- [RTC](https://papers.nips.cc/paper_files/paper/2025/file/300ccb2187dedd4edcc07f7e76d8e553-Paper-Conference.pdf)
  uses task-specific hard time limits and timestamps completed substeps.
- [CALVIN](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py)
  and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/lifelong/metric.py)
  retain success detected during a fixed rollout and stop successful episodes.
- [RLinf's ManiSkill adapter](https://github.com/RLinf/RLinf/blob/main/rlinf/envs/maniskill/maniskill_env.py)
  reports both success-once and success-at-end.
- [ClothFunnels](https://github.com/real-stanford/cloth-funnels/blob/main/cloth_funnels/learning/utils.py)
  is the reference for separating deformable shape error from rigid alignment
  error; ClothDojo uses its own corresponding-particle, device-batched metric.
