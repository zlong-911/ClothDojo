from __future__ import annotations

import json
from pathlib import Path

import pytest

from clothdojo.data.annotations import build_good_segment_plan


def _trajectory(root: Path, trajectory_id: str, actions: int, *, status: str = "complete") -> None:
    path = root / "trajectories" / trajectory_id
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "status": status,
                "state_count": actions + 1,
                "action_count": actions,
            }
        ),
        encoding="utf-8",
    )


def _annotation(root: Path, trajectory_id: str, segments: list[dict], **extra) -> None:
    path = root / "trajectories" / "annotations"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{trajectory_id}.segments.json").write_text(
        json.dumps(
            {
                "schema": "sim1_segments_v1",
                "segments": segments,
                **extra,
            }
        ),
        encoding="utf-8",
    )


def test_missing_annotation_defaults_to_complete_good_episode(tmp_path: Path) -> None:
    _trajectory(tmp_path, "unmarked", 5)

    plan = build_good_segment_plan(tmp_path)

    assert plan.default_good_trajectories == 1
    assert plan.output_episodes == 1
    assert plan.output_frames == 5
    assert plan.full_trajectory_episodes == 1
    assert plan.segments[0].full_trajectory is True
    assert plan.segments[0].annotation_present is False


def test_only_bad_ranges_split_and_good_label_does_not_select_data(tmp_path: Path) -> None:
    _trajectory(tmp_path, "reviewed", 12)
    _annotation(
        tmp_path,
        "reviewed",
        [
            {"start_frame": 0, "end_frame": 1, "label": "good"},
            {"start_frame": 2, "end_frame": 3, "label": "bad"},
            {"start_frame": 4, "end_frame": 4, "label": "bad"},
            {"start_frame": 9, "end_frame": 10, "label": "bad"},
        ],
        review_status="in_progress",
    )

    plan = build_good_segment_plan(tmp_path)

    assert [(segment.action_start, segment.action_end_exclusive) for segment in plan.segments] == [
        (0, 2),
        (5, 9),
        (11, 12),
    ]
    assert all(segment.full_trajectory is False for segment in plan.segments)
    assert plan.bad_ranges == 2
    assert plan.dropped_bad_frames == 5
    assert plan.output_frames == 7
    assert {segment.review_status for segment in plan.segments} == {"in_progress"}


def test_empty_annotation_is_complete_even_if_review_status_is_not_complete(
    tmp_path: Path,
) -> None:
    _trajectory(tmp_path, "empty", 3)
    _annotation(tmp_path, "empty", [], review_status="in_progress")

    segment = build_good_segment_plan(tmp_path).segments[0]

    assert segment.full_trajectory is True
    assert segment.review_status == "in_progress"


def test_incomplete_trajectory_is_skipped(tmp_path: Path) -> None:
    _trajectory(tmp_path, "complete", 3)
    _trajectory(tmp_path, "recording", 3, status="recording")

    plan = build_good_segment_plan(tmp_path)

    assert plan.source_trajectories == 1
    assert plan.skipped_incomplete_trajectories == 1


def test_bad_can_remove_a_whole_source_and_both_timeline_boundaries(tmp_path: Path) -> None:
    _trajectory(tmp_path, "all_bad", 5)
    _annotation(
        tmp_path,
        "all_bad",
        [{"start_frame": 0, "end_frame": 4, "label": "bad"}],
    )
    _trajectory(tmp_path, "bad_edges", 5)
    _annotation(
        tmp_path,
        "bad_edges",
        [
            {"start_frame": 0, "end_frame": 0, "label": "bad"},
            {"start_frame": 4, "end_frame": 4, "label": "bad"},
        ],
    )

    plan = build_good_segment_plan(tmp_path)

    assert plan.source_trajectories == 2
    assert plan.source_action_counts == (("all_bad", 5), ("bad_edges", 5))
    assert [
        (
            segment.source_trajectory,
            segment.action_start,
            segment.action_end_exclusive,
        )
        for segment in plan.segments
    ] == [("bad_edges", 1, 4)]
    assert plan.dropped_bad_frames == 7


@pytest.mark.parametrize(
    "segment",
    [
        {"start_frame": -1, "end_frame": 0, "label": "bad"},
        {"start_frame": 2, "end_frame": 1, "label": "bad"},
        {"start_frame": 0, "end_frame": 3, "label": "bad"},
        {"start_frame": 0, "end_frame": 1, "label": "suboptimal"},
    ],
)
def test_invalid_or_legacy_drop_range_is_rejected(tmp_path: Path, segment: dict) -> None:
    _trajectory(tmp_path, "episode", 3)
    _annotation(tmp_path, "episode", [segment])

    with pytest.raises(RuntimeError):
        build_good_segment_plan(tmp_path)


def test_orphan_annotation_is_rejected(tmp_path: Path) -> None:
    _trajectory(tmp_path, "episode", 3)
    _annotation(tmp_path, "missing", [])

    with pytest.raises(RuntimeError, match="do not match"):
        build_good_segment_plan(tmp_path)
