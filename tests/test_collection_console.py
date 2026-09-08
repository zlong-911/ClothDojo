from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from clothdojo.collection.assets import AssetCatalog
from clothdojo.collection.config import (
    CollectionConfig,
    CollectionProfile,
    TeleoperationConfig,
    load_config,
)
from clothdojo.collection.hil import HILCatalog
from clothdojo.collection.review import AnnotationConflictError, ReviewCatalog
from clothdojo.collection.review_export import ReviewExportManager
from clothdojo.collection.supervisor import NoAvailableSlotError, TeleopSupervisor
from clothdojo.collection.web import build_cloudxr_client_url, create_app
from clothdojo.collection.workspace import CollectionWorkspace
from clothdojo.teleop.install import TeleopInstallation
from clothdojo.teleop.session import AssetSwitchRequest, SessionPlan, SessionStatusWriter


def _manifest(path: Path, *, asset_id: str = "shirt_a", actions: int = 5) -> None:
    path.mkdir(parents=True)
    value = {
        "format_version": 1,
        "alignment": "states_n_plus_1_actions_n",
        "status": "complete",
        "control_fps": 50.0,
        "sim_substeps": 6,
        "units": "m",
        "coordinate_convention": "right_handed_z_up",
        "coordinate_frame": "world_local",
        "max_actions": 100,
        "state_count": actions + 1,
        "action_count": actions,
        "state_shapes": {},
        "action_shapes": {},
        "scene": {
            "task_name": "shirt_flatten",
            "robot_name": "piper",
            "asset_fingerprints": {"garment": "abc"},
            "config_fingerprints": {"scene": "def"},
        },
        "metadata": {"asset_id": asset_id},
    }
    (path / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


def _profile(tmp_path: Path) -> CollectionProfile:
    return CollectionProfile(
        id="shirt_flatten",
        display_name="T 恤展平",
        scene="shirt_flatten",
        dataset_root=tmp_path / "dataset",
        hil_dataset_root=tmp_path / "hil-dataset",
        rollout_root=tmp_path / "rollouts",
        max_actions=3600,
    )


def test_hil_claim_completion_scan_ignores_empty_and_unfinished_results(tmp_path):
    from clothdojo.collection.intervention_leases import completed_points

    for name, actions, status in (("saved", 5, "complete"), ("empty", 0, "complete"),
                                   ("aborted", 5, "aborted")):
        path = tmp_path / name
        _manifest(path, actions=actions)
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = status
        manifest["metadata"].update(source="human_intervention", source_run_id="run",
                                     source_episode_id=name, source_sample_index=10)
        manifest_path.write_text(json.dumps(manifest))
    assert completed_points(tmp_path) == {("run", "saved", 10)}


def test_collection_config_exposes_workflow_choices_not_runtime_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "collection.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
project_root: ..
assets_root: assets
state_dir: outputs/collection
listen: {host: 127.0.0.1, port: 8080}
teleoperation:
  client_host: 10.0.0.8
  gpu_ids: [0, 2]
  standby_gpu_ids: [1, 3]
profiles:
  shirt_flatten:
    display_name: T 恤展平
    scene: shirt_flatten
    dataset_root: data/collections/shirt_flatten
    hil_dataset_root: data/collections/shirt_flatten/hil
    rollout_root: data/collections/shirt_flatten/rollouts
    max_actions: 3600
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.project_root == tmp_path.absolute()
    assert config.profiles["shirt_flatten"].trajectory_root == (
        tmp_path / "data/collections/shirt_flatten/trajectories"
    )
    assert config.profiles["shirt_flatten"].hil_trajectory_root == (
        tmp_path / "data/collections/shirt_flatten/hil/trajectories"
    )
    assert config.teleoperation.gpu_ids == (0, 2)
    assert config.teleoperation.standby_gpu_ids == (1, 3)
    assert config.teleoperation.client_host == "10.0.0.8"


def test_asset_catalog_filters_and_atomically_updates_selection(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    previews = assets / "garments/source/previews/shirt_a"
    previews.mkdir(parents=True)
    (previews / "front.png").write_bytes(b"front")
    (previews / "back.png").write_bytes(b"back")
    converted = assets / "garments/converted/shirt_a"
    converted.mkdir(parents=True)
    (converted / "shirt_a.usdc").write_bytes(b"usd")

    catalog = AssetCatalog(assets)
    assert catalog.page(status="pending")["counts"]["pending"] == 1
    result = catalog.decide("shirt_a", "accepted")
    assert result["collectable"] is True
    assert catalog.page(status="collectable")["assets"][0]["id"] == "shirt_a"
    assert json.loads((assets / "garments/selection.json").read_text())["accepted"] == ["shirt_a"]


def test_review_annotation_uses_revision_and_merges_adjacent_ranges(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    _manifest(profile.trajectory_root / "episode_a", actions=8)
    catalog = ReviewCatalog(profile)
    initial = catalog.annotation("episode_a")
    assert initial["revision"] is None
    assert catalog.episodes()["episodes"][0]["bad_segment_count"] == 0

    saved = catalog.save_annotation(
        "episode_a",
        review_status="complete",
        expected_revision=None,
        segments=[
            {"start_frame": 1, "end_frame": 2, "label": "bad"},
            {"start_frame": 3, "end_frame": 4, "label": "bad"},
        ],
    )
    assert saved["segments"] == [{"start_frame": 1, "end_frame": 4, "label": "bad"}]
    assert saved["revision"]
    assert catalog.episodes()["episodes"][0]["bad_segment_count"] == 1
    with pytest.raises(AnnotationConflictError):
        catalog.save_annotation(
            "episode_a",
            review_status="complete",
            expected_revision=None,
            segments=[],
        )


def test_review_catalog_reads_id_addressed_review_video(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    _manifest(profile.trajectory_root / "episode_a")
    catalog = ReviewCatalog(profile)
    assert catalog.episodes()["episodes"][0]["review_ready"] is False

    video = profile.review_root / "videos" / "cam_high" / "episode_a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"review")
    assert catalog.video("episode_a", "cam_high") == video
    assert catalog.episodes()["episodes"][0]["cameras"] == ["cam_high"]


def test_review_catalog_deletes_trajectory_and_review_artifacts(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    trajectory = profile.trajectory_root / "episode_a"
    _manifest(trajectory)
    catalog = ReviewCatalog(profile)
    catalog.save_annotation(
        "episode_a",
        review_status="complete",
        segments=[],
        expected_revision=None,
    )
    for camera in ("cam_high", "cam_low"):
        video = profile.review_root / "videos" / camera / "episode_a.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"review")

    deleted = catalog.delete_trajectory("episode_a")

    assert deleted["archived_path"] is None
    assert deleted["removed_review_videos"] == 2
    assert deleted["removed_annotation"] is True
    assert not trajectory.exists()
    assert catalog.episodes()["episodes"] == []
    with pytest.raises(FileNotFoundError, match="unknown trajectory"):
        catalog.delete_trajectory("episode_a")


def test_review_catalog_does_not_delete_recording_trajectory(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    trajectory = profile.trajectory_root / "episode_a"
    _manifest(trajectory)
    manifest_path = trajectory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "recording"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="while it is recording"):
        ReviewCatalog(profile).delete_trajectory("episode_a")
    assert trajectory.is_dir()


def test_hil_plan_binds_rollout_and_trajectory_hashes(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    run = profile.rollout_root / "run_a"
    trajectory = run / "trajectories/shirt_a_000000"
    _manifest(trajectory, actions=6)
    meta = {
        "schema": "clothdojo_rollout",
        "format_version": 1,
        "status": "complete",
        "training_eligible": True,
        "camera_names": ["cam_high"],
        "policy": {"checkpoint_sha256": "123"},
        "trajectories": [
            {
                "asset_id": "shirt_a",
                "initial_state_index": 0,
                "seed": 7,
                "path": "trajectories/shirt_a_000000",
                "action_count": 6,
            }
        ],
    }
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    catalog = HILCatalog(profile)
    lerobot_video = (
        run
        / "lerobot/videos/observation.images.cam_high/chunk-000/episode_000000.mp4"
    )
    lerobot_video.parent.mkdir(parents=True)
    lerobot_video.write_bytes(b"training export")
    assert catalog.runs()[0]["episodes"][0]["cameras"] == []
    with pytest.raises(FileNotFoundError):
        catalog.video("run_a", "shirt_a_000000", "cam_high")

    review_video = run / "review/videos/cam_high/shirt_a_000000.mp4"
    review_video.parent.mkdir(parents=True)
    review_video.write_bytes(b"review")
    assert catalog.video("run_a", "shirt_a_000000", "cam_high") == review_video
    assert catalog.runs()[0]["episodes"][0]["cameras"] == ["cam_high"]
    (run / "hil_review.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_hil_review_v1",
                "marks": {"shirt_a_000000": {"updated_at": "legacy"}},
            }
        ),
        encoding="utf-8",
    )
    assert catalog.runs()[0]["episodes"][0]["good_sample_index"] == 5
    saved_breakpoint = catalog.set_breakpoint(
        "run_a", "shirt_a_000000", sample_index=3, marked=True
    )["breakpoints"]
    assert [item["sample_index"] for item in saved_breakpoint] == [3]
    assert catalog.runs()[0]["episodes"][0]["breakpoints"] == saved_breakpoint
    plan = catalog.intervention_plan(
        [{"run_id": "run_a", "episode_id": "shirt_a_000000", "sample_index": 3}]
    )
    assert plan["asset_id"] == "shirt_a"
    assert len(plan["runs"][0]["rollout_meta_sha256"]) == 64
    assert len(plan["breakpoints"][0]["manifest_sha256"]) == 64
    assert catalog.set_completion(
        "run_a", "shirt_a_000000", sample_index=4
    )["good_sample_index"] == 4
    assert catalog.runs()[0]["episodes"][0]["good_sample_index"] == 4
    assert catalog.set_completion(
        "run_a", "shirt_a_000000", sample_index=None
    )["good_sample_index"] is None
    episode = catalog.runs()[0]["episodes"][0]
    assert episode["good_sample_index"] is None
    assert [item["sample_index"] for item in episode["breakpoints"]] == [3]
    assert catalog.set_breakpoint(
        "run_a", "shirt_a_000000", sample_index=3, marked=False
    )["breakpoints"] == []
    assert catalog.runs()[0]["episodes"][0]["breakpoints"] == []
    with pytest.raises(ValueError, match="completion point is outside"):
        catalog.set_completion("run_a", "shirt_a_000000", sample_index=6)
    with pytest.raises(ValueError, match="breakpoint is outside"):
        catalog.set_breakpoint(
            "run_a", "shirt_a_000000", sample_index=6, marked=True
        )
    intervention = profile.hil_trajectory_root / "teleop_branch"
    _manifest(intervention, actions=4)
    manifest_path = intervention / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["metadata"].update(
        {
            "source": "human_intervention",
            "source_run_id": "run_a",
            "source_episode_id": "shirt_a_000000",
            "source_sample_index": 3,
        }
    )
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    interventions = catalog.interventions()[("run_a", "shirt_a_000000")]
    assert interventions[0].pop("collected_at")
    assert interventions == [
        {
            "id": "teleop_branch",
            "sample_index": 3,
            "action_count": 4,
            "fps": 50.0,
            "cameras": [],
        }
    ]
    intervention_video = (
        profile.hil_dataset_root
        / "review/videos/cam_high/teleop_branch.mp4"
    )
    intervention_video.parent.mkdir(parents=True)
    intervention_video.write_bytes(b"intervention review")
    assert catalog.primary_intervention_camera("teleop_branch") == "cam_high"
    assert catalog.intervention_video("teleop_branch", "cam_high") == intervention_video
    assert catalog.interventions()[("run_a", "shirt_a_000000")][0]["cameras"] == [
        "cam_high"
    ]
    pending_plan = catalog.intervention_plan(
        [
            {"run_id": "run_a", "episode_id": "shirt_a_000000", "sample_index": 3},
            {"run_id": "run_a", "episode_id": "shirt_a_000000", "sample_index": 4},
        ]
    )
    assert [point["source_sample_index"] for point in pending_plan["breakpoints"]] == [4]
    with pytest.raises(ValueError, match="already have completed HIL"):
        catalog.intervention_plan(
            [
                {
                    "run_id": "run_a",
                    "episode_id": "shirt_a_000000",
                    "sample_index": 3,
                }
            ]
        )
    deleted = catalog.delete_intervention("teleop_branch")
    archived = profile.hil_dataset_root / "rejected" / "teleop_branch"
    assert deleted["archived_path"] == str(archived)
    assert deleted["removed_review_videos"] == 1
    assert not intervention.exists()
    assert archived.is_dir()
    assert json.loads((archived / "rejection.json").read_text())["reason"] == (
        "operator_deleted"
    )
    assert not intervention_video.exists()
    assert catalog.interventions() == {}
    assert [
        point["sample_index"]
        for point in catalog.runs()[0]["episodes"][0]["breakpoints"]
    ] == [3]


def test_hil_rollout_plan_rejects_formal_unseen_overlap(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    run = profile.rollout_root / "run_a"
    _manifest(run / "trajectories/shirt_a_000000")
    meta = {
        "schema": "clothdojo_rollout",
        "format_version": 1,
        "status": "complete",
        "asset_selection": {"asset_ids": ["shirt_a"]},
        "trajectories": [
            {
                "asset_id": "shirt_a",
                "path": "trajectories/shirt_a_000000",
                "action_count": 5,
            }
        ],
    }
    meta_path = run / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    formal_path = tmp_path / "formal.json"
    formal_path.write_text(
        json.dumps({"assets": [{"asset_id": "shirt_a", "split": "unseen"}]}),
        encoding="utf-8",
    )
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def identity_digest(values: list[str]) -> str:
        return hashlib.sha256(
            ("\n".join(sorted(values)) + "\n").encode()
        ).hexdigest()
    plan = {
        "schema": "clothdojo_hil_rollout_plan",
        "status": "ready",
        "candidate_asset_count": 1,
        "candidate_trajectory_count": 1,
        "rollout": {
            "meta_sha256": digest(meta_path),
            "asset_ids_sha256": identity_digest(["shirt_a"]),
        },
        "formal_panel": {
            "meta_path": str(formal_path),
            "meta_sha256": digest(formal_path),
            "unseen_asset_count": 1,
            "unseen_asset_ids_sha256": identity_digest(["shirt_a"]),
            "unseen_intersection_count": 1,
            "asset_intersection_count": 1,
        },
    }
    (run / "hil_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps the formal unseen"):
        HILCatalog(profile)._metadata(run)


def test_preserved_workspace_uses_canonical_catalogs(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    previews = assets_root / "garments/source/previews/shirt_a"
    previews.mkdir(parents=True)
    (previews / "front.png").write_bytes(b"front")
    (previews / "back.png").write_bytes(b"back")
    converted = assets_root / "garments/converted/shirt_a"
    converted.mkdir(parents=True)
    (converted / "shirt_a.usdc").write_bytes(b"usd")

    profile = _profile(tmp_path)
    _manifest(profile.trajectory_root / "episode_a", actions=8)
    asset_catalog = AssetCatalog(assets_root)
    asset_catalog.decide("shirt_a", "accepted")
    reviews = {profile.id: ReviewCatalog(profile)}
    hil = {profile.id: HILCatalog(profile)}
    config = CollectionConfig(
        project_root=tmp_path,
        assets_root=assets_root,
        state_dir=tmp_path / "outputs/collection",
        host="127.0.0.1",
        port=8080,
        teleoperation=TeleoperationConfig(client_host="auto", gpu_ids=(0,)),
        profiles={profile.id: profile},
    )

    workspace = CollectionWorkspace(config, asset_catalog, reviews, hil)
    snapshot = workspace.snapshot()
    assert snapshot["tasks"][0]["episodes"][0]["id"] == "episode_a"
    assert snapshot["tasks"][0]["episodes"][0]["deletable"] is True
    assert snapshot["tasks"][0]["assets"] == [
        {"id": "shirt_a", "episode_count": 1, "default": False}
    ]
    assert snapshot["slots"][0]["status"] == "offline"
    assert snapshot["metrics"] == {"available_slots": 0, "total_slots": 1}

    class AvailableSupervisor:
        @staticmethod
        def slots() -> list[dict[str, object]]:
            return [{"id": "gpu-0", "status": "available"}]

    workspace.supervisor = AvailableSupervisor()  # type: ignore[assignment]
    assert workspace.snapshot()["metrics"] == {"available_slots": 1, "total_slots": 1}
    page = workspace.asset_page(
        status="accepted",
        query="",
        page=1,
        page_size=48,
    )
    assert page["assets"][0]["trajectory_count"] == 1
    assert "conversion_available" not in page["assets"][0]


def test_workspace_shows_migrated_continuation_results_with_legacy_profile_ids(
    tmp_path: Path,
) -> None:
    flat = _profile(tmp_path)
    fold = CollectionProfile(
        id="shirt_fold",
        display_name="T 恤折叠",
        scene="shirt_fold",
        dataset_root=tmp_path / "fold-dataset",
        hil_dataset_root=tmp_path / "fold-hil-dataset",
        rollout_root=tmp_path / "fold-rollouts",
        max_actions=3600,
        continuation_source_profile=flat.id,
    )
    _manifest(fold.trajectory_root / "fold_a", actions=7)
    manifest_path = fold.trajectory_root / "fold_a/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scene"]["task_name"] = "shirt_fold"
    manifest["metadata"].update(
        {
            "source": "task_continuation",
            "profile_id": "shirt_fold_v1",
            "source_profile_id": "shirt_flatten_v1",
            "source_episode_id": "flat_legacy_a",
            "source_sample_index": 123,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = CollectionConfig(
        project_root=tmp_path,
        assets_root=tmp_path / "assets",
        state_dir=tmp_path / "outputs/collection",
        host="127.0.0.1",
        port=8080,
        teleoperation=TeleoperationConfig(client_host="auto", gpu_ids=(0,)),
        profiles={flat.id: flat, fold.id: fold},
    )
    workspace = CollectionWorkspace(
        config,
        AssetCatalog(config.assets_root),
        {flat.id: ReviewCatalog(flat), fold.id: ReviewCatalog(fold)},
        {flat.id: HILCatalog(flat), fold.id: HILCatalog(fold)},
    )

    fold_task = next(task for task in workspace.snapshot()["tasks"] if task["id"] == fold.id)

    assert fold_task["episode_count"] == 1
    assert len(fold_task["episodes"]) == 1
    assert fold_task["episodes"][0]["status"] == "complete"
    assert fold_task["episodes"][0]["source_profile_id"] == "shirt_flatten_v1"
    assert fold_task["episodes"][0]["source_episode_id"] == "flat_legacy_a"
    assert fold_task["episodes"][0]["result_episode_id"] == "fold_a"
    assert fold_task["episodes"][0]["deletable"] is True


def test_hil_collection_tracks_completed_and_pending_points() -> None:
    runs = [
        {
            "id": "run_a",
            "episodes": [
                {
                    "id": "rollout_a",
                    "asset_id": "shirt_a",
                    "action_count": 20,
                    "fps": 50.0,
                    "cameras": ["cam_high"],
                    "breakpoints": [
                        {"sample_index": 3, "created_at": "2026-09-02T01:00:00Z"},
                        {"sample_index": 8, "created_at": "2026-09-02T02:00:00Z"},
                    ],
                }
            ],
        }
    ]
    interventions = {
        ("run_a", "rollout_a"): [
            {
                "id": "hil_a",
                "sample_index": 3,
                "action_count": 11,
                "collected_at": "2026-09-02T03:00:00Z",
            },
            {
                "id": "hil_old",
                "sample_index": 12,
                "action_count": 7,
                "collected_at": "2026-09-02T00:00:00Z",
            },
        ]
    }

    collection = CollectionWorkspace._hil_collection(runs, interventions)

    assert collection["marked_point_count"] == 2
    assert collection["pending_count"] == 1
    assert collection["completed_point_count"] == 2
    assert collection["trajectory_count"] == 2
    assert collection["frame_count"] == 18
    assert [item["sample_index"] for item in collection["items"]] == [3, 12, 8]
    assert collection["items"][0]["collected_at"] == "2026-09-02T03:00:00Z"
    assert collection["items"][2]["status"] == "pending"
    assert collection["items"][0]["result_count"] == 1
    assert collection["items"][1]["marked"] is False
    assert collection["items"][0]["source_cameras"] == ["cam_high"]


def test_collection_console_preserves_old_workspace_design() -> None:
    static = Path(__file__).parents[1] / "src/clothdojo/collection/static"
    index = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")
    styles = (static / "styles.css").read_text(encoding="utf-8")

    for element_id in (
        "slots",
        "episodes",
        "hil",
        "assets",
        "qr-modal",
        "vr-device-select",
    ):
        assert f'id="{element_id}"' in index
    assert "function renderSlot" in script
    assert "function renderHilCollectionData" in script
    assert "function hilPendingPoints" in script
    assert "asset_count:new Set(points.map(point=>point.asset_id)).size" in script
    assert "hilPendingPoints(state.taskId).filter" not in script
    assert "'<strong>${t(\"service_offline\")}</strong>'" not in script
    assert "expected_revision" in script
    assert 'deviceProfile:"pico4ultra"' in script
    assert 'deviceProfile:"quest3"' in script
    assert 'deviceProfile:"quest3s"' in script
    assert 'deviceProfile:"quest2"' in script
    assert 'useQuestColorWorkaround:"false"' in script
    assert 'cloudxrAutoRefreshMode:"any"' in script
    assert "const clientOrigin=new URL" in script
    assert "serverIP:target.hostname" in script
    assert "proxyUrl:`${clientOrigin.origin}/_clothdojo/${slot.signaling_port}`" in script
    assert 'localStorage.setItem("clothdojo-vr-device"' in script
    assert "let hilRefreshPromise=null" in script
    assert "let refreshPromise=null" in script
    assert 'const result=await pollRequest("/api/hil/snapshot",state.hilEtag)' in script
    assert 'pollRequest("/api/snapshot",state.snapshotEtag)' in script
    assert 'if(!document.hidden&&state.activePane!=="hil")refresh()' in script
    assert "video.playbackRate=hilState.rate" in script
    assert 'onchange="setHilRate(this.value)"' in script
    assert "playbackRateOptions(hilState.rate)" in script
    assert "good_sample_index" in script
    assert 'onclick="hilToggleCompletion()"' in script
    assert '::intervention' in script
    assert 'slot_id:slotId' in script
    assert 'id="hil-start"' not in index
    assert 'id="review-export"' in index
    assert 'id="hil-review-export"' not in index
    assert 'id="task-review-export"' not in index
    assert 'id="hil-profile"' not in index
    assert 'id="hil-vr"' not in index
    assert 'id="hil-stop"' not in index
    assert "function hilToggleReview" in script
    assert "function toggleHilCollectionReview" in script
    assert "function deleteTrajectory" in script
    assert "function deleteHilTrajectory" not in script
    assert "function deleteEpisode" not in script
    assert "function startReviewExport" in script
    assert 'request("/api/review-export",{method:"POST"})' in script
    assert 'request(`/api/sessions/${encodeURIComponent(id)}/asset`' in script
    assert "left.collected_at||left.id" in script
    assert ".runtime{" in styles
    assert ".review-pane-actions button{flex:0 0 auto;white-space:nowrap}" in styles


def test_cloudxr_client_url_uses_validated_pico_profile() -> None:
    url = urlparse(build_cloudxr_client_url("33.229.154.175:44401", 44388))
    query = parse_qs(url.query)

    assert (url.scheme, url.hostname, url.port, url.path) == (
        "https",
        "33.229.154.175",
        44401,
        "/client/",
    )
    assert query["deviceProfile"] == ["pico4ultra"]
    assert query["useQuestColorWorkaround"] == ["false"]
    assert query["cloudxrAutoRefreshMode"] == ["any"]
    assert query["proxyUrl"] == ["https://33.229.154.175:44401/_clothdojo/44388"]


@pytest.mark.parametrize(
    ("device", "tex_sub_image", "color_workaround"),
    [
        ("quest3", "true", "true"),
        ("quest3s", "true", "true"),
        ("quest2", "true", "false"),
    ],
)
def test_cloudxr_client_url_uses_selected_quest_profile(
    device: str, tex_sub_image: str, color_workaround: str
) -> None:
    query = parse_qs(
        urlparse(build_cloudxr_client_url("33.229.154.175:44401", 44388, device)).query
    )

    assert query["deviceProfile"] == [device]
    assert query["enableTexSubImage2D"] == [tex_sub_image]
    assert query["useQuestColorWorkaround"] == [color_workaround]
    assert query["perEyeWidth"] == ["1280"]
    assert query["enablePoseSmoothing"] == ["true"]


def test_collection_frontend_and_teleop_experience_are_packaged() -> None:
    project = Path(__file__).parents[1]
    package_data = tomllib.loads((project / "pyproject.toml").read_text())["tool"]["setuptools"][
        "package-data"
    ]

    assert package_data["clothdojo.collection"] == [
        "static/*.html",
        "static/*.js",
        "static/*.css",
    ]
    assert package_data["clothdojo.teleop"] == ["teleop.kit"]
    assert (project / "src/clothdojo/teleop/teleop.kit").is_file()


def test_preserved_workspace_http_contract(tmp_path: Path) -> None:
    previews = tmp_path / "assets/garments/source/previews/shirt_a"
    previews.mkdir(parents=True)
    (previews / "front.png").write_bytes(b"front")
    (previews / "back.png").write_bytes(b"back")
    converted = tmp_path / "assets/garments/converted/shirt_a"
    converted.mkdir(parents=True)
    (converted / "shirt_a.usdc").write_bytes(b"usd")
    rollout = tmp_path / "data/collections/shirt_flatten/rollouts/run_a"
    _manifest(rollout / "trajectories/shirt_a_000000", actions=6)
    (rollout / "meta.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_rollout",
                "format_version": 1,
                "status": "complete",
                "camera_names": [],
                "trajectories": [
                    {
                        "asset_id": "shirt_a",
                        "path": "trajectories/shirt_a_000000",
                        "action_count": 6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    intervention = (
        tmp_path
        / "data/collections/shirt_flatten/hil/trajectories/hil_a"
    )
    _manifest(intervention, actions=4)
    intervention_manifest = json.loads(
        (intervention / "manifest.json").read_text(encoding="utf-8")
    )
    intervention_manifest["metadata"].update(
        {
            "source": "human_intervention",
            "source_run_id": "run_a",
            "source_episode_id": "shirt_a_000000",
            "source_sample_index": 2,
        }
    )
    (intervention / "manifest.json").write_text(
        json.dumps(intervention_manifest), encoding="utf-8"
    )
    intervention_video = (
        tmp_path
        / "data/collections/shirt_flatten/hil/review/videos/cam_high/hil_a.mp4"
    )
    intervention_video.parent.mkdir(parents=True)
    intervention_video.write_bytes(b"intervention review")
    regular = tmp_path / "data/collections/shirt_flatten/trajectories/regular_a"
    _manifest(regular, actions=5)
    regular_video = (
        tmp_path
        / "data/collections/shirt_flatten/review/videos/cam_high/regular_a.mp4"
    )
    regular_video.parent.mkdir(parents=True)
    regular_video.write_bytes(b"regular review")
    config_path = tmp_path / "collection.yaml"
    config_path.write_text(
        """
project_root: .
assets_root: assets
state_dir: outputs/collection
listen: {host: 127.0.0.1, port: 8080}
teleoperation: {client_host: auto, gpu_ids: [0]}
profiles:
  shirt_flatten:
    display_name: T 恤展平
    scene: shirt_flatten
    dataset_root: data/collections/shirt_flatten
    hil_dataset_root: data/collections/shirt_flatten/hil
    rollout_root: data/collections/shirt_flatten/rollouts
    max_actions: 3600
""".strip(),
        encoding="utf-8",
    )

    with TestClient(create_app(config_path)) as client:
        api_routes = {
            route.path for route in client.app.routes if route.path.startswith("/api/")
        }
        assert not any(path.startswith("/api/v1/") for path in api_routes)
        assert {
            "/api/snapshot",
            "/api/assets",
            "/api/hil/snapshot",
            "/api/review-export",
            "/api/sessions/{session_id}/asset",
        } <= api_routes
        assert client.get("/").status_code == 200
        assert "ClothDojo Collection" in client.get("/").text
        snapshot = client.get("/api/snapshot")
        assert snapshot.status_code == 200
        assert snapshot.json()["slots"][0]["status"] == "offline"
        assert snapshot.json()["tasks"][0]["episodes"][0]["deletable"] is True
        assert snapshot.headers["etag"]
        unchanged = client.get(
            "/api/snapshot",
            headers={"If-None-Match": snapshot.headers["etag"]},
        )
        assert unchanged.status_code == 304
        assets = client.get("/api/assets", params={"page_size": 1})
        assert assets.status_code == 200
        assert assets.json()["assets"][0]["id"] == "shirt_a"
        deleted_regular = client.delete(
            "/api/tasks/shirt_flatten/episodes/regular_a"
        )
        assert deleted_regular.status_code == 200
        assert deleted_regular.json()["removed_review_videos"] == 1
        assert not regular.exists()
        assert not regular_video.exists()
        completion_url = (
            "/api/hil/profiles/shirt_flatten/runs/run_a/"
            "episodes/shirt_a_000000/completion"
        )
        completion = client.put(completion_url, json={"sample_index": 3})
        assert completion.status_code == 200
        assert completion.json()["good_sample_index"] == 3
        hil_snapshot = client.get("/api/hil/snapshot")
        assert hil_snapshot.status_code == 200
        assert hil_snapshot.json()["review_export"] == {
            "status": "idle",
            "excluded_gpu_ids": [0],
        }
        assert client.get("/api/review-export").json() == {
            "status": "idle",
            "excluded_gpu_ids": [0],
        }
        intervention_payload = hil_snapshot.json()["profiles"][0]["runs"][0][
            "episodes"
        ][0]["interventions"][0]
        assert intervention_payload["id"] == "hil_a"
        assert intervention_payload["cameras"] == ["cam_high"]
        intervention_response = client.get(
            "/api/hil/profiles/shirt_flatten/interventions/hil_a/videos/primary"
        )
        assert intervention_response.status_code == 200
        assert intervention_response.content == b"intervention review"
        assert hil_snapshot.json()["profiles"][0]["runs"][0]["episodes"][0][
            "good_sample_index"
        ] == 3
        breakpoint_url = (
            "/api/hil/profiles/shirt_flatten/runs/run_a/"
            "episodes/shirt_a_000000/breakpoint"
        )
        breakpoint = client.put(
            breakpoint_url, json={"sample_index": 2, "marked": True}
        )
        assert breakpoint.status_code == 200
        assert breakpoint.json()["breakpoints"][0]["sample_index"] == 2
        deadline = time.monotonic() + 2.0
        while True:
            hil_snapshot = client.get("/api/hil/snapshot")
            hil_profile = hil_snapshot.json()["profiles"][0]
            saved = hil_profile["runs"][0]["episodes"][0]["breakpoints"]
            if saved or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        assert saved[0]["sample_index"] == 2
        assert hil_profile["collection"]["marked_point_count"] == 1
        assert hil_profile["collection"]["pending_count"] == 0
        assert hil_profile["collection"]["trajectory_count"] == 1
        assert client.put(
            breakpoint_url, json={"sample_index": 2, "marked": False}
        ).json()["breakpoints"] == []
        assert client.put(completion_url, json={"sample_index": None}).status_code == 200
        deleted = client.delete(
            "/api/hil/profiles/shirt_flatten/interventions/hil_a"
        )
        assert deleted.status_code == 200
        assert deleted.json()["source_sample_index"] == 2
        assert not intervention.exists()
        assert (
            tmp_path
            / "data/collections/shirt_flatten/hil/rejected/hil_a/rejection.json"
        ).is_file()
        assert not intervention_video.exists()
        deadline = time.monotonic() + 2.0
        while True:
            hil_snapshot = client.get("/api/hil/snapshot").json()
            item = hil_snapshot["profiles"][0]["collection"]["items"][0]
            if item["status"] == "pending" or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        assert item["status"] == "pending"
        assert item["marked"] is True
        assert item["results"] == []


def test_review_export_covers_all_sources_and_excludes_teleop_gpus(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    _manifest(profile.trajectory_root / "regular_a", actions=40)
    for index, frames in enumerate((30, 20, 10)):
        _manifest(profile.hil_trajectory_root / f"hil_{index}", actions=frames)
    rollout = profile.rollout_root / "run_a"
    _manifest(rollout / "trajectories/rollout_a", actions=50)
    (rollout / "meta.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_rollout",
                "status": "complete",
                "trajectories": [
                    {"path": "trajectories/rollout_a", "action_count": 50}
                ],
            }
        ),
        encoding="utf-8",
    )
    config = CollectionConfig(
        project_root=tmp_path,
        assets_root=tmp_path / "assets",
        state_dir=tmp_path / "state",
        host="127.0.0.1",
        port=8080,
        teleoperation=TeleoperationConfig(
            client_host="auto",
            gpu_ids=(0,),
            standby_gpu_ids=(2,),
        ),
        profiles={profile.id: profile},
    )
    commands = []

    class CompleteProcess:
        pid = 12345

        @staticmethod
        def poll():
            return 0

    def launch(command, _log_path):
        commands.append(command)
        return CompleteProcess()

    manager = ReviewExportManager(
        config,
        gpu_detector=lambda: [0, 1, 2, 3],
        launcher=launch,
    )
    started = manager.start()
    deadline = time.monotonic() + 2.0
    while manager.status()["status"] in {"starting", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    finished = manager.status()

    assert started["excluded_gpu_ids"] == [0, 2]
    assert finished["status"] == "complete"
    assert finished["gpu_ids"] == [1, 3]
    assert finished["source_count"] == 3
    assert finished["completed_trajectory_count"] == 5
    assert {command[command.index("--device") + 1] for command in commands} == {
        "cuda:1",
        "cuda:3",
    }
    exported = [
        value
        for command in commands
        for index, value in enumerate(command)
        if index > 0 and command[index - 1] == "--trajectory"
    ]
    assert sorted(exported) == [
        "hil_0",
        "hil_1",
        "hil_2",
        "regular_a",
        "rollout_a",
    ]


def test_session_plan_and_status_are_atomic(tmp_path: Path) -> None:
    plan = SessionPlan(
        session_id="session_test",
        mode="collection",
        profile_id="shirt_flatten",
        scene="shirt_flatten",
        asset_id="shirt_a",
        asset_path=tmp_path / "shirt.usdc",
        assets_root=tmp_path / "assets",
        output_root=tmp_path / "data/trajectories",
        session_root=tmp_path / "session",
        status_path=tmp_path / "session/status.json",
        gpu_id=0,
        endpoint="10.0.0.8:44401",
        signaling_port=44388,
        media_port=33434,
        proxy_port=44401,
        max_actions=3600,
        standby_gpu_id=1,
    )
    path = plan.write(tmp_path / "session/plan.json")
    assert SessionPlan.read(path) == plan
    status = SessionStatusWriter(plan.status_path, session_id=plan.session_id)
    status.write("waiting_for_vr", gpu_id=0)
    payload = json.loads(plan.status_path.read_text(encoding="utf-8"))
    assert payload["state"] == "waiting_for_vr"
    assert payload["sequence"] == 1

    switch = AssetSwitchRequest(
        request_id="switch_test",
        session_id=plan.session_id,
        asset_id="shirt_b",
        asset_path=tmp_path / "shirt_b.usdc",
        requested_at="2026-09-04T00:00:00+00:00",
    )
    assert AssetSwitchRequest.read(switch.write(plan.switch_request_path)) == switch


def test_local_supervisor_leases_gpu_and_stops_process(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    converted = assets_root / "garments/converted/shirt_a"
    converted.mkdir(parents=True)
    (converted / "shirt_a.usdc").write_bytes(b"usd")
    converted_b = assets_root / "garments/converted/shirt_b"
    converted_b.mkdir(parents=True)
    (converted_b / "shirt_b.usdc").write_bytes(b"usd")
    profile = _profile(tmp_path)
    fold_profile = CollectionProfile(
        id="shirt_fold",
        display_name="T 恤折叠",
        scene="shirt_fold",
        dataset_root=tmp_path / "fold-dataset",
        hil_dataset_root=tmp_path / "fold-hil-dataset",
        rollout_root=tmp_path / "fold-rollouts",
        max_actions=3600,
        continuation_source_profile=profile.id,
    )
    config = CollectionConfig(
        project_root=tmp_path,
        assets_root=assets_root,
        state_dir=tmp_path / "outputs/collection",
        host="127.0.0.1",
        port=8080,
        teleoperation=TeleoperationConfig(
            client_host="10.0.0.8",
            gpu_ids=(0,),
            standby_gpu_ids=(1,),
        ),
        profiles={profile.id: profile, fold_profile.id: fold_profile},
    )
    runtime_root = tmp_path / "teleop-runtime"
    python = runtime_root / "bin/python"
    libraries = runtime_root / "cloudxr-system-libs/root/usr/lib/x86_64-linux-gnu"
    client = runtime_root / "cloudxr-web-client"
    python.parent.mkdir(parents=True)
    python.touch()
    libraries.mkdir(parents=True)
    client.mkdir()
    installation = TeleopInstallation(runtime_root, python, libraries, client, None)
    launched_plans = []

    def launch(plan, _plan_path, log_path):
        launched_plans.append(plan)
        log = log_path.open("ab", buffering=0)
        try:
            return subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()

    supervisor = TeleopSupervisor(config, installation=installation, launcher=launch)
    supervisor._occupied_ports = lambda _slot: []
    assert supervisor.slots()[0]["status"] == "available"
    session = supervisor.start_session(
        profile_id=profile.id,
        asset_id="shirt_a",
        slot_id="gpu-0",
    )
    assert session["status"] == "starting"
    with pytest.raises(NoAvailableSlotError):
        supervisor.start_session(profile_id=profile.id, asset_id="shirt_a")
    session_plan = SessionPlan.read(
        config.state_dir / f"sessions/{session['id']}/plan.json"
    )
    assert session_plan.asset_id == "shirt_a"
    assert session_plan.standby_gpu_id == 1
    assert session_plan.output_root == profile.trajectory_root
    SessionStatusWriter(
        config.state_dir / f"sessions/{session['id']}/status.json",
        session_id=session["id"],
    ).write(
        "collecting",
        physics_cuda_graph=True,
        ik_cuda_graph=True,
        loop_fps=72.0,
        frame_ms=13.8,
        app_ms=7.5,
        physics_submit_ms=0.4,
        ik_updates=1,
        restoration_phase="locked",
        intervention_phase="locked",
    )
    reported = supervisor.session(session["id"])
    assert reported["physics_cuda_graph"] is True
    assert reported["ik_cuda_graph"] is True
    assert reported["loop_fps"] == 72.0
    assert reported["physics_submit_ms"] == 0.4
    assert reported["restoration_phase"] == "locked"
    assert reported["intervention_phase"] == "locked"
    switch = supervisor.request_asset_switch(session["id"], "shirt_b")
    assert switch["asset_id"] == "shirt_b"
    switch_request = AssetSwitchRequest.read(session_plan.switch_request_path)
    assert switch_request.session_id == session["id"]
    assert switch_request.asset_id == "shirt_b"
    with pytest.raises(RuntimeError, match="already pending"):
        supervisor.request_asset_switch(session["id"], "shirt_b")
    session_plan.switch_request_path.unlink()
    SessionStatusWriter(
        config.state_dir / f"sessions/{session['id']}/status.json",
        session_id=session["id"],
    ).write("collecting", asset_id="shirt_b", active_gpu_id=1, standby_gpu_id=0)
    assert supervisor.session(session["id"])["asset_id"] == "shirt_b"
    with pytest.raises(ValueError, match="already active"):
        supervisor.request_asset_switch(session["id"], "shirt_b")
    supervisor.stop_session(session["id"])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = supervisor.session(session["id"])
        if result["status"] == "stopped":
            break
        time.sleep(0.05)
    assert result["status"] == "stopped"

    fold_intervention = supervisor.start_session(
        profile_id=fold_profile.id,
        asset_id="shirt_a",
        slot_id="gpu-0",
        mode="intervention",
        intervention_plan={"breakpoints": [{"asset_id": "shirt_a", "source_run_id": "run",
            "source_episode_id": "a", "source_sample_index": 1}]},
    )
    fold_plan = launched_plans[-1]
    assert fold_plan.mode == "intervention"
    assert fold_plan.scene == "shirt_fold"
    assert fold_plan.intervention_plan is not None
    assert fold_plan.continuation_plan is None
    with supervisor._connect() as database:
        assert database.execute("SELECT asset_id FROM intervention_leases").fetchone()[0] \
            == "shirt_a"
    supervisor.stop_session(fold_intervention["id"])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = supervisor.session(fold_intervention["id"])
        if result["status"] == "stopped":
            break
        time.sleep(0.05)
    assert result["status"] == "stopped"
    assert supervisor.slots()[0]["status"] == "available"

    with supervisor._connect() as database:
        assert database.execute("SELECT COUNT(*) FROM intervention_leases").fetchone()[0] == 0

    intervention = supervisor.start_session(
        profile_id=profile.id,
        asset_id="shirt_a",
        slot_id="gpu-0",
        mode="intervention",
        intervention_plan={"breakpoints": [{"asset_id": "shirt_a", "source_run_id": "run",
            "source_episode_id": "a", "source_sample_index": 1}]},
    )
    assert launched_plans[-1].mode == "intervention"
    assert launched_plans[-1].output_root == profile.hil_trajectory_root
    supervisor.stop_session(intervention["id"])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = supervisor.session(intervention["id"])
        if result["status"] == "stopped":
            break
        time.sleep(0.05)
    assert result["status"] == "stopped"


    def fail_launch(*_args):
        raise RuntimeError("launch failed")

    supervisor._launcher = fail_launch
    with pytest.raises(RuntimeError, match="launch failed"):
        supervisor.start_session(
            profile_id=profile.id, asset_id="shirt_a", slot_id="gpu-0", mode="intervention",
            intervention_plan={"breakpoints": [{"asset_id": "shirt_a", "source_run_id": "run",
                "source_episode_id": "a", "source_sample_index": 1}]},
        )
    with supervisor._connect() as database:
        assert database.execute("SELECT COUNT(*) FROM intervention_leases").fetchone()[0] == 0
    assert supervisor.slots()[0]["status"] == "available"


def test_local_supervisor_ports_are_stable_for_physical_gpu_ids(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    config = CollectionConfig(
        project_root=tmp_path,
        assets_root=tmp_path / "assets",
        state_dir=tmp_path / "outputs/collection",
        host="127.0.0.1",
        port=8000,
        teleoperation=TeleoperationConfig(client_host="10.0.0.8", gpu_ids=(2, 5)),
        profiles={profile.id: profile},
    )
    supervisor = TeleopSupervisor(config)

    assert supervisor._slot_specs() == [
        {
            "id": "gpu-2",
            "gpu_id": 2,
            "standby_gpu_id": None,
            "endpoint": "10.0.0.8:44401",
            "signaling_port": 44408,
            "media_port": 33454,
            "proxy_port": 44401,
        },
        {
            "id": "gpu-5",
            "gpu_id": 5,
            "standby_gpu_id": None,
            "endpoint": "10.0.0.8:44401",
            "signaling_port": 44438,
            "media_port": 33484,
            "proxy_port": 44401,
        },
    ]
