from __future__ import annotations

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
from clothdojo.collection.supervisor import NoAvailableSlotError, TeleopSupervisor
from clothdojo.collection.web import build_cloudxr_client_url, create_app
from clothdojo.collection.workspace import CollectionWorkspace
from clothdojo.teleop.install import TeleopInstallation
from clothdojo.teleop.session import SessionPlan, SessionStatusWriter


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
        rollout_root=tmp_path / "rollouts",
        intervention_root=tmp_path / "interventions",
        max_actions=3600,
    )


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
profiles:
  shirt_flatten:
    display_name: T 恤展平
    scene: shirt_flatten
    dataset_root: data/collections/shirt_flatten
    rollout_root: data/rollouts/shirt_flatten
    intervention_root: data/interventions/shirt_flatten
    max_actions: 3600
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.project_root == tmp_path.absolute()
    assert config.profiles["shirt_flatten"].trajectory_root == (
        tmp_path / "data/collections/shirt_flatten/trajectories"
    )
    assert config.teleoperation.gpu_ids == (0, 2)
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


def test_review_source_index_is_cached_until_review_directory_changes(
    tmp_path: Path, monkeypatch
) -> None:
    profile = _profile(tmp_path)
    profile.review_root.mkdir(parents=True)
    calls = []

    def discover(_root):
        calls.append(True)
        return {"episode_a": object()}

    monkeypatch.setattr("clothdojo.collection.review.discover_review_sources", discover)
    catalog = ReviewCatalog(profile)
    assert catalog._review_sources() == {"episode_a": catalog._review_sources()["episode_a"]}
    assert len(calls) == 1

    (profile.review_root / "review-new").mkdir()
    catalog._review_sources()
    assert len(calls) == 2


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
    plan = catalog.intervention_plan("run_a", [{"episode_id": "shirt_a_000000", "sample_index": 3}])
    assert plan["asset_id"] == "shirt_a"
    assert len(plan["rollout_meta_sha256"]) == 64
    assert len(plan["breakpoints"][0]["manifest_sha256"]) == 64
    assert catalog.set_good("run_a", "shirt_a_000000", good=True)["good"] is True
    assert catalog.runs()[0]["episodes"][0]["good"] is True
    intervention = profile.intervention_root / "teleop_branch"
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
    assert catalog.interventions()[("run_a", "shirt_a_000000")] == [
        {"id": "teleop_branch", "sample_index": 3, "action_count": 4}
    ]


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
    assert page["assets"][0]["conversion_available"] is False


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
    assert "'<strong>${t(\"service_offline\")}</strong>'" not in script
    assert "expected_revision" in script
    assert 'deviceProfile:"pico4ultra"' in script
    assert 'deviceProfile:"quest3"' in script
    assert 'deviceProfile:"quest3s"' in script
    assert 'deviceProfile:"quest2"' in script
    assert 'useQuestColorWorkaround:"false"' in script
    assert 'cloudxrAutoRefreshMode:"any"' in script
    assert 'localStorage.setItem("clothdojo-vr-device"' in script
    assert ".runtime{" in styles
    assert ".review-pane-actions button{flex:0 0 auto;white-space:nowrap}" in styles


def test_cloudxr_client_url_uses_validated_pico_profile() -> None:
    url = urlparse(build_cloudxr_client_url("33.229.154.175:44401"))
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
        urlparse(build_cloudxr_client_url("33.229.154.175:44401", device)).query
    )

    assert query["deviceProfile"] == [device]
    assert query["enableTexSubImage2D"] == [tex_sub_image]
    assert query["useQuestColorWorkaround"] == [color_workaround]
    assert query["perEyeWidth"] == ["1280"]
    assert query["enablePoseSmoothing"] == ["true"]


def test_collection_frontend_and_teleop_experience_are_packaged() -> None:
    project = Path(__file__).parents[1]
    package_data = tomllib.loads((project / "pyproject.toml").read_text())["tool"][
        "setuptools"
    ]["package-data"]

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
    rollout_root: data/rollouts/shirt_flatten
    intervention_root: data/interventions/shirt_flatten
    max_actions: 3600
""".strip(),
        encoding="utf-8",
    )

    with TestClient(create_app(config_path)) as client:
        assert client.get("/").status_code == 200
        assert "ClothDojo Collection" in client.get("/").text
        snapshot = client.get("/api/snapshot")
        assert snapshot.status_code == 200
        assert snapshot.json()["slots"][0]["status"] == "offline"
        assets = client.get("/api/assets", params={"page_size": 1})
        assert assets.status_code == 200
        assert assets.json()["assets"][0]["id"] == "shirt_a"
        assert client.get("/api/hil/snapshot").status_code == 200


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
    )
    path = plan.write(tmp_path / "session/plan.json")
    assert SessionPlan.read(path) == plan
    status = SessionStatusWriter(plan.status_path, session_id=plan.session_id)
    status.write("waiting_for_vr", gpu_id=0)
    payload = json.loads(plan.status_path.read_text(encoding="utf-8"))
    assert payload["state"] == "waiting_for_vr"
    assert payload["sequence"] == 1


def test_local_supervisor_leases_gpu_and_stops_process(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    converted = assets_root / "garments/converted/shirt_a"
    converted.mkdir(parents=True)
    (converted / "shirt_a.usdc").write_bytes(b"usd")
    profile = _profile(tmp_path)
    config = CollectionConfig(
        project_root=tmp_path,
        assets_root=assets_root,
        state_dir=tmp_path / "outputs/collection",
        host="127.0.0.1",
        port=8080,
        teleoperation=TeleoperationConfig(client_host="10.0.0.8", gpu_ids=(0,)),
        profiles={profile.id: profile},
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

    def launch(plan, _plan_path, log_path):
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
    assert (
        SessionPlan.read(config.state_dir / f"sessions/{session['id']}/plan.json").asset_id
        == "shirt_a"
    )
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
    )
    reported = supervisor.session(session["id"])
    assert reported["physics_cuda_graph"] is True
    assert reported["ik_cuda_graph"] is True
    assert reported["loop_fps"] == 72.0
    assert reported["physics_submit_ms"] == 0.4
    supervisor.stop_session(session["id"])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = supervisor.session(session["id"])
        if result["status"] == "stopped":
            break
        time.sleep(0.05)
    assert result["status"] == "stopped"
    assert supervisor.slots()[0]["status"] == "available"
