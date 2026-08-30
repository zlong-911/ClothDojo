from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from clothdojo.data.rollout import (
    CAMERA_NAMES,
    RolloutLeRobotWriter,
    _export_canonical_collection,
    _features,
    _restore_render_state,
    _TrajectoryRenderState,
)
from clothdojo.robots import (
    ALOHA2,
    FR3_DUO,
    robot_profile,
)


class _Writer:
    def __init__(self) -> None:
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


class _Array:
    def __init__(self, shape) -> None:
        self.values = np.zeros(shape, dtype=np.float32)

    @property
    def shape(self):
        return self.values.shape

    def assign(self, values) -> None:
        self.values[...] = values


class _Robot:
    def get_attribute(self, name, state):
        return getattr(state, name)

    def set_attribute(self, name, state, value) -> None:
        assert getattr(state, name) is value


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
    assert "observation.images.cam_low" in features

    aloha = robot_profile(ALOHA2)
    aloha_features = _features(480, 640, arm_joint_names=aloha.arm_joint_names)
    assert aloha_features["observation.state"]["names"][0] == "left/waist"
    assert aloha_features["action"]["names"][7] == "right/waist_target"

    fr3 = robot_profile(FR3_DUO)
    fr3_features = _features(480, 640, arm_joint_names=fr3.arm_joint_names)
    assert fr3_features["observation.state"]["shape"] == (16,)
    assert fr3_features["observation.state"]["names"][7] == "left_gripper_open_state"
    assert fr3_features["action"]["names"][8] == "right/fr3v2_joint1_target"


def test_rollout_writer_can_publish_multiple_episodes(tmp_path) -> None:
    target = tmp_path / "rollouts"
    writer = _Writer()
    sink = RolloutLeRobotWriter(
        target,
        fps=50,
        prompt="Flatten the shirt.",
        scenario={"scenario_id": "batch"},
        trajectory="/data/collection",
        policy={"kind": "rlinf_sft"},
        training_eligible=True,
        source_kind="rollout_collection",
        rendering={"backend": "isaac_rtx", "base_seed": 4, "layouts": []},
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

    sink.add_frame(observation, np.zeros(14, dtype=np.float32))
    sink.save_episode()
    sink.add_frame(observation, np.zeros(14, dtype=np.float32))
    sink.close()

    sidecar = json.loads((target / "meta/clothdojo.json").read_text())
    assert writer.saved == 2
    assert sidecar["counts"] == {"output_episodes": 2, "output_frames": 2}
    assert sidecar["episode_sets"]["full_trajectory"] == [0, 1]
    assert sidecar["source"]["kind"] == "rollout_collection"
    assert sidecar["rendering"] == {
        "backend": "isaac_rtx",
        "base_seed": 4,
        "layouts": [],
    }


def test_render_replay_restores_world_local_state_and_rebuilds_articulation(
    monkeypatch,
) -> None:
    state = SimpleNamespace(
        joint_q=_Array((2,)),
        joint_qd=_Array((2,)),
        body_q=_Array((1, 7)),
        body_qd=_Array((1, 6)),
        particle_q=_Array((2, 3)),
        particle_qd=_Array((2, 3)),
    )
    model = SimpleNamespace(world_count=1, state=lambda: state)
    scene = SimpleNamespace(
        model=model,
        robot=_Robot(),
        world_offsets=np.asarray([[10.0, 2.0, 0.0]], dtype=np.float32),
    )
    replay = _TrajectoryRenderState(scene)
    local = SimpleNamespace(
        joint_q=np.asarray([0.0, 1.0], dtype=np.float32),
        joint_qd=np.zeros(2, dtype=np.float32),
        body_q=np.asarray([[1.0, 3.0, 5.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        body_qd=np.zeros((1, 6), dtype=np.float32),
        particle_q=np.asarray([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32),
        particle_qd=np.zeros((2, 3), dtype=np.float32),
    )
    fk_calls = []
    monkeypatch.setattr(
        "newton.eval_fk",
        lambda *args: fk_calls.append(args),
    )

    _restore_render_state(replay, local)

    np.testing.assert_allclose(state.joint_q.values, [0.0, 1.0])
    np.testing.assert_allclose(state.body_q.values[:, :3], [[11, 5, 5]])
    np.testing.assert_allclose(
        state.particle_q.values,
        [[11, 3, 0], [12, 3, 0]],
    )
    assert fk_calls == [(model, state.joint_q, state.joint_qd, state)]


def test_canonical_export_owns_backend_camera_lifecycle(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    manifest = {
        "scenario": {"prompt": "Flatten the shirt."},
        "policy": {"kind": "test"},
        "training_eligible": True,
        "trajectories": [
            {"path": "episode_0", "asset_id": "shirt_a", "action_count": 1},
            {"path": "episode_1", "asset_id": "shirt_b", "action_count": 1},
        ],
    }
    state = SimpleNamespace(
        joint_q=np.arange(14, dtype=np.float32),
        gripper_open_state=np.asarray([0.25, 0.75], dtype=np.float32),
    )
    action = SimpleNamespace(
        joint_target=np.arange(12, dtype=np.float32),
        gripper_open_command=np.asarray([0.5, 1.0], dtype=np.float32),
    )
    sample = SimpleNamespace(state=state, action=action)

    class Reader:
        manifest = SimpleNamespace(
            control_fps=50,
            action_count=1,
            scene=SimpleNamespace(
                robot_id="piper",
                task_name="shirt_flatten",
            ),
        )

        def __init__(self, _path) -> None:
            pass

        def state_at(self, index):
            assert index == 0
            return state

        def iter_samples(self, **_kwargs):
            return iter((sample,))

    class Sink:
        def __init__(self, *_args, **kwargs) -> None:
            self.kwargs = kwargs
            self.frames = []
            self.saved = 0
            self.closed = False
            self.aborted = False

        def add_frame(self, observation, policy_action) -> None:
            self.frames.append((observation, policy_action))

        def save_episode(self) -> None:
            self.saved += 1

        def close(self) -> None:
            self.closed = True

        def abort(self) -> None:
            self.aborted = True

    cameras = []

    class Camera:
        def __init__(self, asset_id) -> None:
            self.asset_id = asset_id
            self.episodes = []
            self.closed = False
            cameras.append(self)

        def begin_episode(self, episode_index, *, source_episode_id) -> None:
            self.episodes.append((episode_index, source_episode_id))

        def __call__(self, received_state):
            assert received_state is state
            return {
                f"observation.images.{name}": np.zeros((1, 2, 4, 3), dtype=np.uint8)
                for name in CAMERA_NAMES
            }

        def close(self) -> None:
            self.closed = True

    sink = Sink()
    monkeypatch.setattr(
        "clothdojo.data.rollout._load_canonical_collection",
        lambda _root: (manifest, "rollout_collection"),
    )
    monkeypatch.setattr("clothdojo.trajectory.TrajectoryReader", Reader)
    monkeypatch.setattr("clothdojo.data.rollout.RolloutLeRobotWriter", lambda *_a, **_kw: sink)
    monkeypatch.setattr(
        "clothdojo.data.rollout.inspect_rollout_collection_export",
        lambda _root: {"renderer": "newton"},
    )

    result = _export_canonical_collection(
        source,
        tmp_path / "output",
        camera_factory=lambda asset_id, _state, _environment: Camera(asset_id),
        rendering={"backend": "isaac_rtx", "layouts": []},
        repo_id=None,
        image_writer_threads=2,
    )

    assert [camera.asset_id for camera in cameras] == ["shirt_a", "shirt_b"]
    assert [camera.episodes for camera in cameras] == [[(0, "episode_0")], [(1, "episode_1")]]
    assert all(camera.closed for camera in cameras)
    assert sink.saved == 2
    assert sink.closed and not sink.aborted
    np.testing.assert_array_equal(
        sink.frames[0][0]["observation.state"],
        [[0, 1, 2, 3, 4, 5, 0.25, 8, 9, 10, 11, 12, 13, 0.75]],
    )
    assert result["renderer"] == "isaac_rtx"
