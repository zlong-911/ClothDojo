# Reviewed LeRobot export

`clothdojo data export-lerobot` is the single offline path from a local
ClothDojo reviewed trajectory collection to the LeRobot v2.1 format pinned by
RLINF. Export reads only the current ClothDojo trajectory format.

## Selection rule

Annotations use action indices and inclusive endpoints. A segment whose label
is exactly `bad` removes data. Overlapping or adjacent bad ranges are merged,
and connected components of their complement become output LeRobot episodes
only when they contain at least 65 frames.

- a missing annotation means the entire trajectory is good;
- an empty annotation means the entire trajectory is good;
- an explicit `good` range does not gate unmarked frames;
- `review_status` is provenance and never gates selection;
- fragments shorter than 65 frames are removed and counted separately from bad
  frames. At 50 Hz this is 1.3 seconds.

The 65-frame floor is a data-quality policy, not a loader compatibility
requirement. The pinned LeRobot loader can clamp a 50-step Pi0.5 action chunk
to the terminal action and reports an `action_is_pad` mask, although the
current OpenPI ALOHA repack does not train with that mask. STEAM likewise can
clamp the second frame of a pair to the episode terminal. A 50-frame episode
contains one Pi0.5 target without terminal padding; 65 frames is the stricter
shared floor because it also contains one STEAM pair at the true `k=64`
stride. Shorter fragments would contain neither kind of unclamped target.

All frames in the exported LeRobot dataset are consequently good and belong to
a minimally usable temporal fragment. Bad and short-fragment frames are
represented separately in export provenance; neither is copied to the output.

## Source and output formats

The completed review cache was produced by LeRobot v3.0, while RLINF pins
LeRobot v2.1. The exporter uses a narrow read-only adapter for the former:
PyArrow reads state/action Parquet and PyAV streams the four synchronized MP4
files. Frames are handed to the pinned v2.1 `LeRobotDataset` writer, so
ClothDojo does not install or import two LeRobot versions.

The thin writer adapter explicitly uses H.264 at CRF 23 and encodes the four
cameras concurrently. This avoids the corrupted streams observed with the
pinned writer's bundled SVT-AV1 encoder without modifying site-packages. Every
video is fully decoded and its frame count is checked immediately after the
episode is encoded, then checked again during final dataset validation. The
codec, quality setting and camera parallelism are recorded in
`meta/clothdojo.json`.

The exporter first requires an exact one-to-one mapping between complete
source trajectories and review exports. It validates each review's provenance,
identity range, frame count, feature schema and required files. Source indexes
and metadata hashes are checked again before publication. Output is written to
`.<name>.tmp` and atomically renamed only after LeRobot metadata, episode
lengths and every decoded video pass validation. An existing output or
unfinished staging directory is never overwritten.

When a stricter selection is an exact subset of an already completed
ClothDojo v2.1 export, `--reuse-from` avoids decoding and re-encoding identical
media. It verifies every retained segment by source range and annotation
digest, rebuilds Parquet episode/global indices and all standard metadata, and
hard-links immutable MP4 files into the atomic output when the filesystem
supports links. Filesystems that reject links fall back to a byte copy without
re-encoding; the publication mode is recorded in the sidecar. Either output is
independent of the source pathname. The fixed LeRobot version must reload the
republished result before publication.
Republishing only needs trajectory manifests, annotations and the JSON metadata
of each review export; its source index may therefore omit the large review
Parquet and MP4 payloads. Fresh export and ordinary plan validation retain the
strict default and require those payloads.

## Episode metadata

Standard LeRobot metadata remains untouched. `meta/clothdojo.json` supplies the
project provenance and an entry for every output episode:

- `quality: "good"`;
- `source_trajectory`, `action_start`, and `action_end_exclusive`;
- `source_episode_complete`: the source manifest was complete;
- `full_trajectory`: the output range covers the complete source action
  timeline;
- `contains_source_terminal`: the output range reaches the source action
  timeline's end;
- `review_status` and `review_complete`, which are informational only;
- annotation digest and source review export identity.

New exports record SHA-256 identities for each review's `info.json` and
provenance JSON. Republication rejects a recorded identity mismatch. The first
reviewed-good export predates these two fields, so its one-time stricter
republication is additionally anchored by its completed sidecar and the exact
source-range/annotation-digest match; the resulting export contains the hashes
and is strict for all later reuse.

`counts.represented_source_trajectories` distinguishes input trajectories from
sources that still contribute an output episode. Short-filter provenance is
kept under `dropped_short`: `ranges` records every rejected source interval and
whether it covered a full source trajectory, while
`source_trajectories_fully_removed` lists sources whose only good ranges were
removed by the minimum-length policy. Numeric full/partial range and removed
source counts remain in `counts` for quick audit.

The explicit names avoid overloading a single ambiguous `complete` flag.
`episode_sets.good`, `episode_sets.full_trajectory`, and
`episode_sets.partial_trajectory` contain directly loadable LeRobot episode
indices. `episode_sets.annotation_present` and `episode_sets.review_complete`
keep those two notions distinct. STEAM can use:

```python
from clothdojo.data.lerobot import load_episode_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

episodes = load_episode_indices(dataset_root, trajectory_extent="all")
dataset = LeRobotDataset(
    repo_id="local/shirt_flatten_reviewed",
    root=dataset_root,
    episodes=episodes,
)
```

Use `trajectory_extent="full"` only if an experiment explicitly requires
unsplit full trajectories. LeRobot's standard `info.json` split remains its
contiguous `train` range; arbitrary quality selection is not encoded as a fake
standard split.

## Commands and synchronized baseline

Validate and resynchronize indexes without decoding videos or writing output:

```bash
clothdojo data export-lerobot \
  --source data/collections/shirt_flatten \
  --plan-only
```

Write a new external dataset directory:

```bash
clothdojo data export-lerobot \
  --source data/collections/shirt_flatten \
  --reuse-from data/datasets/shirt_flatten_reviewed_good_v1 \
  --output data/datasets/shirt_flatten_reviewed_good_v2 \
  --min-segment-frames 65 \
  --repo-id local/shirt_flatten_reviewed_good_v2
```

The synchronized 2026-08-22 source snapshot has 690 complete trajectories and
1,306,567 source frames. Its 48 bad ranges remove 18,269 frames; the 65-frame
floor additionally removes 19 fragments containing 411 frames. Acceptance is
therefore exactly 719 output episodes and 1,287,887 frames: 644 full-trajectory
episodes and 75 partial-trajectory episodes, representing 689 of the 690 input
trajectories. All 19 rejected fragments were produced by bad-range splitting;
none was an originally short complete trajectory. One source has no remaining
output episode because both of its good fragments were shorter than the floor.
The 22 trajectories without an annotation are included as full good
trajectories.
