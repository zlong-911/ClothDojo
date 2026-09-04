"""Isaac RTX/OpenXR frontend with Newton 1.5 as the only physics owner."""

from __future__ import annotations

# The minimal Kit experience excludes omni.warp.core so the application bundle's
# Newton 1.5 / Warp 1.16 pair is the only Warp runtime in this process.
import argparse
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from clothdojo.teleop.session import SessionPlan, SessionStatusWriter


def _configure_openusd_startup() -> None:
    """Avoid the USDPhysics traversal race without serializing Kit tasking."""

    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")


def main() -> None:
    # Must precede SimulationApp: USDPhysics is initialized during Kit startup.
    _configure_openusd_startup()
    parser = argparse.ArgumentParser(description="run ClothDojo Isaac XR teleoperation")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = SessionPlan.read(args.plan)
    status = SessionStatusWriter(plan.status_path, session_id=plan.session_id)
    stop = threading.Event()
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    def request_stop(_number, _frame) -> None:
        stop.set()

    for number in previous:
        signal.signal(number, request_stop)
    app = None
    try:
        if plan.mode == "intervention" and plan.intervention_plan is None:
            raise RuntimeError("an intervention session requires a cold-restore plan")
        status.write("starting", phase="starting_isaac", gpu_id=plan.gpu_id)
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
        portable = Path(os.environ.get("XDG_CACHE_HOME", plan.session_root)) / "kit-portable"
        portable.mkdir(parents=True, exist_ok=True)
        sys.argv = [sys.argv[0], "--portable-root", str(portable)]

        import isaacsim
        from isaacsim import SimulationApp

        isaac_root = Path(isaacsim.__file__).resolve().parent
        app = SimulationApp(
            _simulation_app_config(plan, isaac_root),
            experience=str(Path(__file__).with_name("teleop.kit")),
        )
        status.write("starting", phase="building_newton", gpu_id=plan.gpu_id)
        runtime = _build_newton_runtime(plan)
        _run(app, runtime, plan, status, stop)
        status.write("stopped", phase="operator_stop" if stop.is_set() else "app_exit")
    except BaseException as exc:
        status.write("failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        if app is not None and app.is_running():
            app.close()


def _build_newton_runtime(plan: SessionPlan):
    """Build physics after Kit startup but before any ClothDojo USD import."""

    import warp as wp

    device = wp.get_device()
    if not device.is_cuda or device.ordinal != plan.gpu_id:
        raise RuntimeError(
            "Warp did not adopt Kit's active GPU context: "
            f"expected cuda:{plan.gpu_id}, got {device}"
        )
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

    with wp.ScopedDevice(device):
        scene = build_scene(
            plan.assets_root,
            task_name=plan.scene,
            spec=ShirtFlattenSpec(garment_id=plan.asset_id, world_count=1),
            device=device,
        )
        runtime = ShirtFlattenRuntime(scene)
        expected_mimics = 2
        if runtime.model.constraint_mimic_count != expected_mimics:
            raise RuntimeError(
                "Piper gripper mimic constraints are incomplete: "
                f"expected {expected_mimics}, got {runtime.model.constraint_mimic_count}"
            )
        print(
            f"Piper gripper mimic constraints ready: {expected_mimics}",
            flush=True,
        )
        return runtime


def _simulation_app_config(plan: SessionPlan, isaac_root: Path) -> dict:
    return {
        "headless": False,
        "renderer": "RasterizationLighting",
        "width": 1280,
        "height": 720,
        "multi_gpu": False,
        "active_gpu": plan.gpu_id,
        "physics_gpu": plan.gpu_id,
        # A native startup failure must return control to the supervisor; the
        # bundled crash reporter can otherwise symbolicate for many minutes.
        "enable_crashreporter": False,
        # SimulationApp otherwise overwrites PXR_WORK_THREAD_LIMIT with its
        # 32-thread default before USDPhysics initializes.
        "limit_cpu_threads": 1,
        "extra_args": [
            "--/rtx/hydra/supportMultiTickRate=false",
            "--/rtx/rendering/perSensorTickTlas=false",
            "--ext-folder",
            str(isaac_root / "extscache"),
        ],
    }


def _run(
    app, runtime, plan: SessionPlan, status: SessionStatusWriter, stop: threading.Event
) -> None:
    import newton
    import omni.kit.app
    import omni.usd
    import warp as wp

    if newton.__version__ != "1.5.0" or wp.__version__ != "1.16.0":
        raise RuntimeError(
            "teleoperation bundle must provide Newton 1.5.0 and Warp 1.16.0; "
            f"got Newton {newton.__version__}, Warp {wp.__version__}"
        )
    extensions = omni.kit.app.get_app().get_extension_manager()
    for name in (
        "omni.kit.xr.core",
        "omni.kit.xr.system.openxr",
        "omni.kit.xr.ui.window.viewport",
    ):
        extensions.set_extension_enabled(name, True)
    for _ in range(10):
        app.update()

    from clothdojo.teleop.control import BimanualTeleopController, TeleopSafetyConfig
    from clothdojo.teleop.ik import DualArmIKController
    from clothdojo.teleop.interaction import ButtonEdges, HomeReturn, XRLocomotion
    from clothdojo.teleop.isaac_fabric import NewtonFabricBridge
    from clothdojo.teleop.isaac_stage import build_render_stage, configure_xr
    from clothdojo.teleop.openxr import IsaacOpenXRBackend
    from clothdojo.teleop.recording import TeleopRecorder
    from clothdojo.teleop.restoration import RestorationQueue

    stage_path = build_render_stage(runtime, plan)
    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"Isaac could not open render stage: {stage_path}")
    for _ in range(30):
        app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac USD context has no render stage")
    device = runtime.model.device
    bridge = NewtonFabricBridge(runtime.model, context, device=device)
    anchor = configure_xr(
        app,
        stage,
        (-0.3, 0.0, 1.45),
        (0.65, 0.0, 0.8),
    )
    status.write("waiting_for_vr", gpu_id=plan.gpu_id)
    backend = IsaacOpenXRBackend()
    ik = DualArmIKController(runtime, iterations=10)
    left_home, right_home = ik.measured_pose_matrices()
    controller = BimanualTeleopController(
        left_home,
        right_home,
        TeleopSafetyConfig(
            ee_min_x_m=0.0,
            ee_min_z_m=0.82,
            stale_after_s=0.15,
            clutch_threshold=0.5,
            translation_scale=1.0,
        ),
    )
    recorder = TeleopRecorder(runtime, plan)
    restore_queue = RestorationQueue.load(plan, runtime)
    grasp_probe = runtime.grasp.region_probe()
    edges = ButtonEdges()
    left_return = HomeReturn()
    right_return = HomeReturn()
    locomotion = XRLocomotion(move_speed_mps=0.5, turn_speed_degps=60.0, deadzone=0.18)
    arm_command = ik.solve()
    gripper_command = wp.ones((1, 2), dtype=wp.float32, device=device)
    print(
        "CUDA Graph: "
        f"physics={'enabled' if runtime.cuda_graph_enabled else 'disabled'}, "
        f"ik={'enabled' if ik.cuda_graph_enabled else 'disabled'}",
        flush=True,
    )
    previous_connected = False
    last_step = time.monotonic()
    next_status = last_step + 1.0
    report_started = last_step
    report_frames = 0
    report_ik_updates = 0
    report_seconds = {
        "control": 0.0,
        "ik_submit": 0.0,
        "physics_submit": 0.0,
        "fabric_submit": 0.0,
        "app": 0.0,
        "frame": 0.0,
    }
    last_left_target = left_home.copy()
    last_right_target = right_home.copy()
    left_input_valid = False
    right_input_valid = False
    restoration_phase = "waiting" if restore_queue is not None else None
    current_origin = None
    completed_attempt = None
    advance_after_finalize = False

    def prepare_restored_point(point, *, preserve_completed: bool = False) -> bool:
        nonlocal arm_command, current_origin, restoration_phase, completed_attempt
        nonlocal last_left_target, last_right_target, last_step
        if point is None:
            restoration_phase = "complete"
            current_origin = None
            completed_attempt = None
            backend.set_grasp_haptics(False, False)
            return False
        ik.reset_from_state()
        restored_left, restored_right = ik.measured_pose_matrices()
        restored_gripper = restore_queue.current_gripper_open
        assert restored_gripper is not None
        controller.reset_from_state(restored_left, restored_right, restored_gripper)
        arm_command = ik.solve()
        gripper_command.assign(restored_gripper.reshape(1, 2))
        last_left_target = restored_left.copy()
        last_right_target = restored_right.copy()
        left_return.clear()
        right_return.clear()
        grasp_probe.discard()
        backend.set_grasp_haptics(False, False)
        current_origin = point.recording_metadata()
        if not preserve_completed:
            completed_attempt = None
        restoration_phase = "waiting"
        last_step = time.monotonic()
        backend.pulse_reset_complete()
        label = "HIL" if point.kind == "human_intervention" else "CONTINUE"
        print(
            f"  [{label}] restored {point.source_episode_id}; "
            f"remaining={restore_queue.remaining}; waiting for safe takeover",
            flush=True,
        )
        return True

    def restore_next(*, requeue_current: bool = False) -> bool:
        if restore_queue is None:
            return False
        return prepare_restored_point(
            restore_queue.restore_next(
                runtime,
                requeue_current=requeue_current,
            )
        )

    def restore_current(*, preserve_completed: bool = False) -> bool:
        if restore_queue is None:
            return False
        return prepare_restored_point(
            restore_queue.restore_current(runtime),
            preserve_completed=preserve_completed,
        )

    def accept_completed_attempt(completed) -> None:
        nonlocal completed_attempt
        previous_attempt = completed_attempt
        completed_attempt = completed
        if previous_attempt is None or previous_attempt == completed:
            return
        rejected = recorder.reject_completed(
            previous_attempt,
            reason="superseded_by_rerecording",
        )
        print(f"  [RESTORE] replaced previous attempt {rejected}", flush=True)

    def restoration_status() -> dict[str, object]:
        continuation_phase = None
        if plan.continuation_plan is not None:
            continuation_phase = "exhausted" if restoration_phase == "complete" else "active"
        return {
            "restoration_phase": restoration_phase,
            "intervention_phase": restoration_phase if plan.mode == "intervention" else None,
            "continuation_phase": continuation_phase,
            "continuation_remaining": (
                None if restore_queue is None else restore_queue.remaining
            ),
        }

    if restore_queue is not None:
        restore_next()
    frame_index = 0
    reset_action = "save/next scene" if restore_queue is not None else "reset"
    print(
        "XR controls: squeeze=clutch, trigger=gripper, A=start/save recording, "
        f"X={reset_action}, Y/B=left/right Home, both sticks=reset view",
        flush=True,
    )
    try:
        while app.is_running() and not stop.is_set():
            frame_started = time.perf_counter()
            now = time.monotonic()
            had_finalization = recorder.is_finalizing
            completed = recorder.poll_finalization()
            if completed is not None:
                print(f"  [REC] saved {completed}", flush=True)
                if restoration_phase == "finalizing":
                    accept_completed_attempt(completed)
                    if advance_after_finalize:
                        advance_after_finalize = False
                        completed_attempt = None
                        restore_next()
                    else:
                        restore_current(preserve_completed=True)
            elif had_finalization and not recorder.is_finalizing:
                if restoration_phase == "aborting":
                    restore_current(preserve_completed=True)
            frame = backend.poll()
            graspable = grasp_probe.poll()
            if graspable is not None:
                backend.set_grasp_haptics(*graspable)
            connected = bool(frame.left.valid or frame.right.valid)
            if connected and not previous_connected:
                anchor.arm_initial_rebind()
                backend.pulse_connected()
            elif previous_connected and not connected and recorder.is_recording:
                recorder.abort("xr_connection_lost")
                if restoration_phase == "recording":
                    restoration_phase = "aborting"
                print("  [REC] aborted after XR disconnect", flush=True)
            previous_connected = connected

            left_input_valid, right_input_valid = _prime_changed_inputs(
                edges,
                frame,
                left_input_valid=left_input_valid,
                right_input_valid=right_input_valid,
            )
            record_pressed = edges.rising("record", _button(frame.right, "a"))
            reset_pressed = edges.rising("reset", _button(frame.left, "x"))
            left_home_pressed = edges.rising("left_home", _button(frame.left, "y"))
            right_home_pressed = edges.rising("right_home", _button(frame.right, "b"))

            frozen = False
            if restoration_phase is not None:
                if restoration_phase in {"waiting", "ready"}:
                    ready = controller.ready_for_restored_takeover(frame, now_s=now)
                    restoration_phase = "ready" if ready else "waiting"
                    if reset_pressed:
                        requeue_current = completed_attempt is None
                        completed_attempt = None
                        restore_next(requeue_current=requeue_current)
                        frozen = True
                    elif ready and record_pressed:
                        path = recorder.start(origin=current_origin)
                        restoration_phase = "recording"
                        backend.pulse_recording_start()
                        print(f"  [RESTORE] recording {path}", flush=True)
                    else:
                        frozen = True
                elif restoration_phase == "recording":
                    if record_pressed or reset_pressed:
                        path = recorder.stop()
                        backend.pulse_recording_stop()
                        if path is None:
                            if reset_pressed:
                                completed_attempt = None
                                restore_next(requeue_current=True)
                            else:
                                restore_current(preserve_completed=True)
                                print(
                                    "  [RESTORE] discarded empty recording; restored current point",
                                    flush=True,
                                )
                        else:
                            advance_after_finalize = reset_pressed
                            restoration_phase = "finalizing"
                            suffix = "; next scene after save" if reset_pressed else ""
                            print(f"  [RESTORE] finalizing {path}{suffix}", flush=True)
                        frozen = True
                elif restoration_phase == "finalizing":
                    if reset_pressed:
                        advance_after_finalize = True
                    frozen = True
                elif restoration_phase in {"aborting", "complete"}:
                    frozen = True
            else:
                if record_pressed:
                    if recorder.is_recording:
                        path = recorder.stop()
                        backend.pulse_recording_stop()
                        if path is None:
                            print("  [REC] discarded empty recording", flush=True)
                        else:
                            print(f"  [REC] finalizing {path}", flush=True)
                    elif recorder.is_finalizing:
                        print("  [REC] waiting for trajectory finalization", flush=True)
                    else:
                        path = recorder.start(origin=current_origin)
                        backend.pulse_recording_start()
                        print(f"  [REC] recording {path}", flush=True)
                if left_home_pressed:
                    left_return.start()
                    controller.reset_targets(left_pose=left_home)
                    print("  [HOME] left arm returning to Home", flush=True)
                if right_home_pressed:
                    right_return.start()
                    controller.reset_targets(right_pose=right_home)
                    print("  [HOME] right arm returning to Home", flush=True)
                if left_return.update(frame.left.squeeze, controller.config.clutch_threshold):
                    controller.reset_targets(left_pose=left_home)
                    print("  [HOME] left arm resumed VR control", flush=True)
                if right_return.update(frame.right.squeeze, controller.config.clutch_threshold):
                    controller.reset_targets(right_pose=right_home)
                    print("  [HOME] right arm resumed VR control", flush=True)
                if reset_pressed:
                    if recorder.is_recording:
                        path = recorder.stop(reason="operator_reset")
                        if path is not None:
                            print(f"  [REC] finalizing {path}", flush=True)
                    grasp_probe.discard()
                    backend.set_grasp_haptics(False, False)
                    runtime.reset()
                    ik.reset_from_state()
                    left_home, right_home = ik.measured_pose_matrices()
                    controller.reset(left_home, right_home)
                    arm_command = ik.solve()
                    last_left_target = left_home.copy()
                    last_right_target = right_home.copy()
                    left_return.clear()
                    right_return.clear()
                    backend.pulse_reset_complete()
                    print("  [RESET] environment reset", flush=True)

            both_sticks = _button(frame.left, "thumbstick_click") and _button(
                frame.right, "thumbstick_click"
            )
            if edges.rising("locomotion_reset", both_sticks):
                anchor.rebind()
            movement = locomotion.update(
                _stick(frame.left),
                _stick(frame.right),
                min(0.1, max(0.0, now - last_step)),
                enabled=frame.left.squeeze < 0.5 and frame.right.squeeze < 0.5,
            )
            last_step = now
            if movement is not None:
                anchor.update_flight(*movement)

            if frozen:
                bridge.update(runtime.state_0)
                app.update()
                frame_index += 1
                if time.monotonic() >= next_status:
                    status.write(
                        "collecting" if connected else "waiting_for_vr",
                        gpu_id=plan.gpu_id,
                        teleop_state="hold" if connected else "disconnected",
                        recording=recorder.is_recording,
                        recording_finalizing=recorder.is_finalizing,
                        recording_frames=recorder.frame_count,
                        frame=frame_index,
                        **restoration_status(),
                    )
                    next_status = time.monotonic() + 1.0
                continue

            control_started = time.perf_counter()
            targets = controller.update(frame, ik.measured_pose_matrices, now_s=now)
            report_seconds["control"] += time.perf_counter() - control_started
            if _targets_changed(
                targets.left_pose,
                targets.right_pose,
                last_left_target,
                last_right_target,
            ):
                ik_started = time.perf_counter()
                ik.set_targets(targets.left_pose, targets.right_pose)
                arm_command = ik.solve()
                report_seconds["ik_submit"] += time.perf_counter() - ik_started
                report_ik_updates += 1
                last_left_target = targets.left_pose.copy()
                last_right_target = targets.right_pose.copy()
            ik.override_home(
                arm_command,
                left=left_return.active,
                right=right_return.active,
            )
            gripper_command.assign(targets.gripper_open.reshape(1, 2))
            physics_started = time.perf_counter()
            reached_limit = recorder.step(arm_command, gripper_command)
            report_seconds["physics_submit"] += time.perf_counter() - physics_started
            if reached_limit:
                backend.pulse_recording_stop()
                if restoration_phase == "recording":
                    restoration_phase = "finalizing"
                print("  [REC] maximum action count reached; finalizing", flush=True)
            grasp_probe.submit(runtime.state_0)
            fabric_started = time.perf_counter()
            bridge.update(runtime.state_0)
            report_seconds["fabric_submit"] += time.perf_counter() - fabric_started
            app_started = time.perf_counter()
            app.update()
            report_seconds["app"] += time.perf_counter() - app_started
            report_seconds["frame"] += time.perf_counter() - frame_started
            report_frames += 1
            frame_index += 1
            status_now = time.monotonic()
            if status_now >= next_status:
                state = "collecting" if connected else "waiting_for_vr"
                report_elapsed = max(status_now - report_started, 1.0e-9)
                divisor = max(report_frames, 1)
                status.write(
                    state,
                    gpu_id=plan.gpu_id,
                    teleop_state=targets.state.value,
                    recording=recorder.is_recording,
                    recording_finalizing=recorder.is_finalizing,
                    recording_frames=recorder.frame_count,
                    frame=frame_index,
                    ik_status=ik.solution_status(),
                    physics_cuda_graph=runtime.cuda_graph_enabled,
                    ik_cuda_graph=ik.cuda_graph_enabled,
                    loop_fps=report_frames / report_elapsed,
                    frame_ms=1000.0 * report_seconds["frame"] / divisor,
                    control_ms=1000.0 * report_seconds["control"] / divisor,
                    ik_submit_ms=1000.0 * report_seconds["ik_submit"] / divisor,
                    ik_updates=report_ik_updates,
                    physics_submit_ms=1000.0 * report_seconds["physics_submit"] / divisor,
                    fabric_submit_ms=1000.0 * report_seconds["fabric_submit"] / divisor,
                    app_ms=1000.0 * report_seconds["app"] / divisor,
                    **restoration_status(),
                )
                next_status = status_now + 1.0
                report_started = status_now
                report_frames = 0
                report_ik_updates = 0
                for name in report_seconds:
                    report_seconds[name] = 0.0
    finally:
        try:
            if recorder.is_recording:
                if restoration_phase == "recording":
                    recorder.abort("session_stop")
                else:
                    recorder.stop(reason="session_stop")
            completed = recorder.close()
            if completed is not None:
                print(f"  [REC] saved {completed}", flush=True)
                if restoration_phase == "finalizing":
                    accept_completed_attempt(completed)
        finally:
            backend.set_grasp_haptics(False, False)
            backend.close()
            anchor.close()


def _button(hand, name: str) -> bool:
    return bool(hand.buttons.get(name, False))


def _prime_changed_inputs(
    edges,
    frame,
    *,
    left_input_valid: bool,
    right_input_valid: bool,
) -> tuple[bool, bool]:
    """Suppress face-button edges caused solely by controller reconnects."""

    left_valid = bool(frame.left.valid)
    right_valid = bool(frame.right.valid)
    left_changed = left_valid != left_input_valid
    right_changed = right_valid != right_input_valid
    if left_changed:
        edges.prime(
            {
                "left_home": _button(frame.left, "y"),
                "reset": _button(frame.left, "x"),
            }
        )
    if right_changed:
        edges.prime(
            {
                "record": _button(frame.right, "a"),
                "right_home": _button(frame.right, "b"),
            }
        )
    if left_changed or right_changed:
        edges.prime(
            {
                "locomotion_reset": _button(frame.left, "thumbstick_click")
                and _button(frame.right, "thumbstick_click")
            }
        )
    return left_valid, right_valid


def _targets_changed(
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    previous_left_pose: np.ndarray,
    previous_right_pose: np.ndarray,
) -> bool:
    return not (
        np.array_equal(left_pose, previous_left_pose)
        and np.array_equal(right_pose, previous_right_pose)
    )


def _stick(hand) -> tuple[float, float]:
    return (
        float(hand.buttons.get("thumbstick_x", 0.0)),
        float(hand.buttons.get("thumbstick_y", 0.0)),
    )


if __name__ == "__main__":
    main()
