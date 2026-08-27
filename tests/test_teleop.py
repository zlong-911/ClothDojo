from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from clothdojo.teleop.cloudxr_slot import _app_started, _startup_timeout, _write_overrides
from clothdojo.teleop.control import (
    BimanualTeleopController,
    TeleopSafetyConfig,
    TeleopState,
    TimestampedXRFrame,
    XRHandFrame,
)
from clothdojo.teleop.ik import DualArmIKController
from clothdojo.teleop.install import TeleopInstallation
from clothdojo.teleop.interaction import ButtonEdges, HomeReturn
from clothdojo.teleop.isaac_app import (
    _configure_openusd_startup,
    _prime_changed_inputs,
    _simulation_app_config,
    _targets_changed,
)
from clothdojo.teleop.openxr import IsaacOpenXRBackend


def _pose(x: float, y: float, z: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = (x, y, z)
    return value


def test_teleop_ik_enables_cuda_graph_by_default() -> None:
    assert DualArmIKController.__dataclass_fields__["use_cuda_graph"].default is True


def test_teleop_only_updates_ik_when_cartesian_targets_change() -> None:
    left = _pose(0.5, 0.25, 1.0)
    right = _pose(0.5, -0.25, 1.0)

    assert not _targets_changed(left.copy(), right.copy(), left, right)
    moved = left.copy()
    moved[0, 3] += 0.01
    assert _targets_changed(moved, right, left, right)


def test_home_return_requires_release_then_squeeze() -> None:
    home = HomeReturn()
    home.start()

    assert not home.update(1.0, 0.5)
    assert home.active
    assert not home.update(0.0, 0.5)
    assert home.update(1.0, 0.5)
    assert not home.active


def test_reconnected_controller_buttons_are_primed_without_edges() -> None:
    edges = ButtonEdges()
    frame = TimestampedXRFrame(
        sequence=1,
        receive_time_s=1.0,
        left=XRHandFrame(buttons={"x": True, "y": True}, valid=True),
        right=XRHandFrame(buttons={"a": True, "b": True}, valid=True),
    )

    assert _prime_changed_inputs(
        edges,
        frame,
        left_input_valid=False,
        right_input_valid=False,
    ) == (True, True)
    assert not edges.rising("reset", True)
    assert not edges.rising("left_home", True)
    assert not edges.rising("record", True)
    assert not edges.rising("right_home", True)


def test_grasp_haptics_are_continuous_and_independent_per_hand() -> None:
    class Device:
        def __init__(self) -> None:
            self.value = 0.0

        @staticmethod
        def get_output_names():
            return ["haptic"]

        @staticmethod
        def has_output(_name):
            return True

        def set_output_value(self, _name, value):
            self.value = value

    backend = IsaacOpenXRBackend(xr_core=object())
    left, right = Device(), Device()
    backend._devices = {"left": left, "right": right}

    backend.set_grasp_haptics(True, False)
    assert (left.value, right.value) == (0.25, 0.0)
    backend.set_grasp_haptics(False, True)
    assert (left.value, right.value) == (0.0, 0.25)
    backend.stop_haptics()
    assert (left.value, right.value) == (0.0, 0.0)


def test_cloudxr_pacing_targets_the_measured_teleop_rate(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CLOUDXR_VULKAN_GPU_INDEX", raising=False)
    plan = SimpleNamespace(signaling_port=44388, media_port=33434, proxy_port=44401)

    config = _write_overrides(plan, tmp_path).read_text(encoding="utf-8")

    assert "NV_MAX_FPS=20\n" in config
    assert "NV_CXR_GPU_INDEX_VULKAN=0\n" in config


def test_nonzero_gpu_keeps_physical_cuda_and_vulkan_indices_aligned(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    installation = TeleopInstallation(
        root=tmp_path / "runtime",
        python=tmp_path / "python",
        cloudxr_library_dir=tmp_path / "libraries",
        web_client_dir=tmp_path / "client",
        nvenc_filter=None,
    )
    environment = installation.environment(
        gpu_id=1,
        slot_root=tmp_path / "slot",
        source_root=tmp_path / "src",
    )
    plan = SimpleNamespace(gpu_id=1)
    config = _simulation_app_config(plan, tmp_path / "isaac")

    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["WARP_CACHE_PATH"] == str(tmp_path / "warp-cache")
    assert config["active_gpu"] == 1
    assert config["physics_gpu"] == 1
    assert config["enable_crashreporter"] is False


def test_cloudxr_startup_state_converges_only_after_isaac_is_ready() -> None:
    assert not _app_started(None)
    assert not _app_started({"state": "starting", "phase": "building_newton"})
    assert _app_started({"state": "waiting_for_vr"})
    assert _app_started({"state": "collecting"})
    assert _startup_timeout("building_newton") > _startup_timeout("starting_isaac")


def test_isaac_limits_openusd_traversal_before_kit_startup(monkeypatch) -> None:
    monkeypatch.delenv("PXR_WORK_THREAD_LIMIT", raising=False)
    _configure_openusd_startup()
    assert os.environ["PXR_WORK_THREAD_LIMIT"] == "1"

    monkeypatch.setenv("PXR_WORK_THREAD_LIMIT", "2")
    _configure_openusd_startup()
    assert os.environ["PXR_WORK_THREAD_LIMIT"] == "2"


def test_bimanual_controller_preserves_clutch_and_stale_input_safety() -> None:
    left_home = _pose(0.5, 0.25, 1.0)
    right_home = _pose(0.5, -0.25, 1.0)
    controller = BimanualTeleopController(
        left_home,
        right_home,
        TeleopSafetyConfig(stale_after_s=0.15, clutch_threshold=0.5),
    )

    released = TimestampedXRFrame(
        sequence=1,
        receive_time_s=1.0,
        left=XRHandFrame(pose=_pose(0.0, 0.2, 1.2), trigger=0.25, valid=True),
        right=XRHandFrame(pose=_pose(0.0, -0.2, 1.2), trigger=0.75, valid=True),
    )
    ready = controller.update(released, lambda: (left_home, right_home), now_s=1.0)
    assert ready.state == TeleopState.READY
    np.testing.assert_allclose(ready.gripper_open, (0.75, 0.25))

    anchored = TimestampedXRFrame(
        sequence=2,
        receive_time_s=1.01,
        left=XRHandFrame(pose=_pose(0.0, 0.2, 1.2), squeeze=1.0, valid=True),
        right=released.right,
    )
    first = controller.update(anchored, lambda: (left_home, right_home), now_s=1.01)
    assert first.left_engaged is True
    np.testing.assert_allclose(first.left_pose, left_home)

    moved = TimestampedXRFrame(
        sequence=3,
        receive_time_s=1.02,
        left=XRHandFrame(pose=_pose(0.1, 0.2, 1.2), squeeze=1.0, valid=True),
        right=released.right,
    )
    engaged = controller.update(moved, lambda: (left_home, right_home), now_s=1.02)
    assert engaged.state == TeleopState.ENGAGED
    np.testing.assert_allclose(engaged.left_pose[:3, 3], (0.6, 0.25, 1.0))

    stale = controller.update(moved, lambda: (left_home, right_home), now_s=2.0)
    assert stale.state == TeleopState.HOLD
    assert stale.left_engaged is False
