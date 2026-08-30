from dataclasses import replace
from types import SimpleNamespace

import pytest

from clothdojo.cli import _parser, main
from clothdojo.robots import robot_ids, robot_profile


def test_discovery_and_cli_do_not_load_assets(monkeypatch):
    import clothdojo.robots as robots

    def fail(*args, **kwargs):
        raise AssertionError("discovery must work before deployment")

    monkeypatch.setattr(robots, "_load_workstation", fail)
    assert set(robot_ids()) == {"piper", "aloha2", "x5", "ur5e_2f85", "fr3_duo"}
    parser = _parser()
    args = parser.parse_args(["assets", "prepare-workstations", "--source", "external"])
    assert args.source == "external"
    assert args.output == "assets"
    args = parser.parse_args(["sim", "render-workstations"])
    assert args.robot == "all"
    assert tuple(args.views) == ("overall", "left", "right")
    with pytest.raises(SystemExit):
        parser.parse_args(["sim", "render-workstations", "--robot", "panda"])


def test_render_all_uses_only_registered_workstations(monkeypatch):
    from clothdojo.sim import workstation_review

    calls = []

    def render(robot, **kwargs):
        calls.append((robot, kwargs))
        return {"robot": robot}

    monkeypatch.setattr(workstation_review, "render_workstation_review", render)
    main(["sim", "render-workstations", "--assets", "deployed", "--egl-device", "3"])
    assert tuple(robot for robot, _ in calls) == robot_ids()
    assert all(args["assets_root"] == "deployed" and args["egl_device"] == 3 for _, args in calls)
    assert all(args["views"] == ("overall", "left", "right") for _, args in calls)


def test_render_wrist_views_dispatch(monkeypatch):
    from clothdojo.sim import workstation_review

    calls = []
    monkeypatch.setattr(
        workstation_review, "render_workstation_review",
        lambda robot, **kwargs: calls.append((robot, kwargs)),
    )
    main(["sim", "render-workstations", "--robot", "ur5e_2f85",
          "--views", "left_wrist", "right_wrist"])
    assert calls[0][0] == "ur5e_2f85"
    assert calls[0][1]["views"] == ("left_wrist", "right_wrist")


def test_workstation_assets_cli_dispatch(monkeypatch):
    from clothdojo.assets import workstations

    calls = []
    monkeypatch.setattr(
        workstations, "prepare_workstation_assets", lambda *args: calls.append(args)
    )
    main(["assets", "prepare-workstations", "--source", "shared", "--output", "local"])
    assert calls == [("shared", "local")]
    monkeypatch.setattr(
        workstations, "inspect_workstation_assets", lambda *args: calls.append(args)
    )
    main(["assets", "check-workstations", "--assets", "local"])
    assert calls[-1] == ("local",)


def test_missing_camera_export_fails_before_creating_dataset(tmp_path, monkeypatch):
    import clothdojo.trajectory as trajectory
    from clothdojo.data import rollout

    monkeypatch.setattr(trajectory, "TrajectoryReader", lambda *_: SimpleNamespace())
    # Production FR3 now has cam_low. Deliberately remove it to keep testing
    # the missing-camera guard, rather than expecting a broken production asset.
    profile = robot_profile("fr3_duo")
    assert "cam_low" in {camera.name for camera in profile.cameras}
    incomplete = replace(profile, cameras=tuple(
        camera for camera in profile.cameras if camera.name != "cam_low"
    ))
    monkeypatch.setattr(rollout, "_trajectory_robot", lambda _: incomplete)

    def fail(*args, **kwargs):
        raise AssertionError("must reject unsupported camera set before creating writer")

    monkeypatch.setattr(rollout, "RolloutLeRobotWriter", fail)
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match=r"fr3_duo.*missing cam_low"):
        rollout._export_collection_manifest(
            tmp_path, output, manifest={"scenario": {}, "trajectories": [{"path": "t"}]},
            source_kind="retarget_collection", camera_factory=fail, rendering={},
            repo_id=None, image_writer_threads=1,
        )
    assert not output.exists()
