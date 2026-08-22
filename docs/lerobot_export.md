# Reviewed LeRobot export

`clothdojo data export-lerobot` is the single offline path from the old
reviewed trajectory collection to the LeRobot v2.1 format pinned by RLINF. It
does not modify or copy the source trajectory collection into this Git
repository.

## Selection rule

Annotations use action indices and inclusive endpoints. Only a segment whose
label is exactly `bad` removes data. Overlapping or adjacent bad ranges are
merged, and every non-empty connected component of their complement becomes
one output LeRobot episode.

- a missing annotation means the entire trajectory is good;
- an empty annotation means the entire trajectory is good;
- an explicit `good` range does not gate unmarked frames;
- `review_status` is provenance and never gates selection;
- there is no minimum fragment length and therefore no second, implicit data
  deletion rule.

All frames in the exported LeRobot dataset are consequently good. Bad frames
are represented only by the source annotation and the export provenance; they
are not copied to the output.

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
  --source /path/to/newton-cloth-rl/data/shirt_flatten/000004 \
  --plan-only
```

Write a new external dataset directory:

```bash
clothdojo data export-lerobot \
  --source /path/to/newton-cloth-rl/data/shirt_flatten/000004 \
  --output /path/to/datasets/shirt_flatten_reviewed_good_v1 \
  --repo-id local/shirt_flatten_reviewed_good_v1
```

The synchronized 2026-08-22 source snapshot has 690 complete trajectories and
1,306,567 source frames. Its 48 bad ranges remove 18,269 frames. Acceptance is
therefore exactly 738 output episodes and 1,288,298 frames: 644 full-trajectory
episodes and 94 partial-trajectory episodes. The 22 trajectories without an
annotation are included as full good trajectories.
