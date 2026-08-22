"""Executable single-scenario RTC/trajectory acceptance gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clothdojo.data.rollout import RolloutLeRobotWriter
from clothdojo.rollout.engine import run_rtc_episode
from clothdojo.rollout.newton import HoldChunkPolicy, NewtonTrajectoryEpisode
from clothdojo.rollout.render import CAMERA_NAMES, ShirtFlattenCameraObservation
from clothdojo.rollout.scenario import load_rtc_scenario
from clothdojo.sim.runtime import ShirtFlattenRuntime
from clothdojo.sim.scene import (
    ShirtFlattenSpec,
    build_shirt_flatten,
    build_shirt_flatten_render_scene,
)
from clothdojo.trajectory import TrajectoryReader


def run_rtc_rollout_check(
    scenario_path: str | Path,
    assets_root: str | Path,
    trajectory_output: str | Path,
    lerobot_output: str | Path,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run a hold-policy RTC episode through the real Newton trajectory path."""

    scenario = load_rtc_scenario(scenario_path)
    trajectory_output = Path(trajectory_output).expanduser().resolve()
    lerobot_output = Path(lerobot_output).expanduser().resolve()
    try:
        scene = build_shirt_flatten(
            assets_root,
            spec=ShirtFlattenSpec(garment_id=scenario.garment_id, world_count=1),
            device=device,
        )
        runtime = ShirtFlattenRuntime(scene)
        render_scene = build_shirt_flatten_render_scene(
            assets_root,
            spec=scene.spec,
            device=device,
        )
        camera = ShirtFlattenCameraObservation(scene, render_scene)
        policy = HoldChunkPolicy(scenario.action_horizon)
        frame_sink = RolloutLeRobotWriter(
            lerobot_output,
            fps=scene.spec.control_fps,
            prompt=scenario.prompt,
            scenario=scenario.to_json_dict(),
            trajectory=str(trajectory_output),
            policy={"kind": "hold_acceptance"},
            training_eligible=False,
        )
        with NewtonTrajectoryEpisode(
            runtime,
            trajectory_output,
            scenario,
            assets_root=assets_root,
            observation_provider=camera,
            frame_sink=frame_sink,
        ) as episode:
            rtc = run_rtc_episode(episode, policy, scenario)
        reader = TrajectoryReader(trajectory_output)
        reader.validate_training_data()
        compatible = (
            reader.manifest.action_count == scenario.steps
            and reader.manifest.state_count == scenario.steps + 1
        )
        return {
            "compatible": compatible,
            "device": device,
            "scenario": scenario.to_json_dict(),
            "trajectory": str(trajectory_output),
            "lerobot": str(lerobot_output),
            "state_count": reader.manifest.state_count,
            "action_count": reader.manifest.action_count,
            "rtc": rtc.to_json_dict(),
            "policy_kind": "hold_acceptance",
            "training_eligible": False,
            "renderer_migrated": True,
            "camera_names": list(CAMERA_NAMES),
        }
    except Exception as error:
        return {
            "compatible": False,
            "device": device,
            "scenario": scenario.to_json_dict(),
            "trajectory": str(trajectory_output),
            "lerobot": str(lerobot_output),
            "error": f"{type(error).__name__}: {error}",
        }
