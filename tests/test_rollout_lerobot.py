from __future__ import annotations

import json

import numpy as np

from clothdojo.data.rollout import RolloutLeRobotWriter, _features
from clothdojo.rollout.render import CAMERA_NAMES


class _Writer:
    def __init__(self) -> None:
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


def test_live_rollout_export_is_atomic_and_explicitly_ineligible(tmp_path) -> None:
    target = tmp_path / "rollout"
    writer = _Writer()
    sink = RolloutLeRobotWriter(
        target,
        fps=50,
        prompt="Flatten the shirt.",
        scenario={"scenario_id": "gate", "fingerprint": "abc"},
        trajectory="/data/trajectory",
        policy={"kind": "hold_acceptance"},
        training_eligible=False,
        image_height=2,
        image_width=4,
        _writer_factory=lambda **kwargs: writer,
    )
    observation = {
        "observation.state": np.zeros((1, 14), dtype=np.float32),
        **{
            f"observation.images.{camera}": np.zeros((1, 2, 4, 3), dtype=np.uint8)
            for camera in CAMERA_NAMES
        },
    }

    sink.add_frame(observation, np.ones(14, dtype=np.float32))
    assert not target.exists()
    sink.close()

    assert target.is_dir()
    assert writer.saved == 1
    assert writer.frames[0]["task"] == "Flatten the shirt."
    sidecar = json.loads((target / "meta/clothdojo.json").read_text())
    assert sidecar["source"]["kind"] == "rtc_rollout"
    assert sidecar["training_eligible"] is False
    assert sidecar["counts"] == {"output_episodes": 1, "output_frames": 1}


def test_rollout_features_match_expert_policy_order() -> None:
    features = _features(480, 640)

    assert features["observation.state"]["shape"] == (14,)
    assert features["observation.state"]["names"][6] == "left_gripper_open_state"
    assert features["observation.state"]["names"][13] == "right_gripper_open_state"
    assert features["action"]["names"][6] == "left_gripper_open_cmd"
    assert features["action"]["names"][13] == "right_gripper_open_cmd"
