"""Supervise CloudXR runtime, WSS client host and one Isaac teleop app."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from clothdojo.teleop.session import SessionPlan, SessionStatusWriter

_RUNTIME_WORKER = """\
import sys
sys.path = [path for path in sys.path if path]
from isaacteleop.cloudxr.runtime import run
run()
"""
_SHUTDOWN_TIMEOUT_S = 60.0
_APP_STARTUP_PHASE_TIMEOUT_S = 150.0
_NEWTON_BUILD_TIMEOUT_S = 300.0
_RENDERER_BUILD_TIMEOUT_S = 900.0
_FAILED_STARTUP_SHUTDOWN_TIMEOUT_S = 5.0
_MAX_STREAM_FPS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="run one fixed ClothDojo CloudXR slot")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = SessionPlan.read(args.plan)
    status = SessionStatusWriter(plan.status_path, session_id=plan.session_id)
    try:
        raise SystemExit(run_slot(plan, status))
    except BaseException as exc:
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            status.write("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def run_slot(plan: SessionPlan, status: SessionStatusWriter) -> int:
    with _short_install_dir(plan) as install_dir:
        return _run_slot(plan, status, install_dir)


def _run_slot(plan: SessionPlan, status: SessionStatusWriter, install_dir: Path) -> int:
    from isaacteleop.cloudxr.env_config import EnvConfig
    from isaacteleop.cloudxr.runtime import check_eula, wait_for_runtime_ready_sync

    status.write("starting", phase="configuring_cloudxr", gpu_id=plan.gpu_id)
    override = _write_overrides(plan, install_dir)
    environment = EnvConfig.from_args(str(install_dir), override)
    check_eula(accept_eula=True)
    logs = environment.ensure_logs_dir()
    runtime_log_path = logs / "runtime-launcher.log"
    runtime_log = runtime_log_path.open("ab", buffering=0)
    runtime = subprocess.Popen(
        [sys.executable, "-c", _RUNTIME_WORKER],
        env=_runtime_environment(),
        stdin=subprocess.DEVNULL,
        stdout=runtime_log,
        stderr=subprocess.STDOUT,
    )
    runtime_log.close()
    try:
        if not wait_for_runtime_ready_sync(is_process_alive=lambda: runtime.poll() is None):
            raise RuntimeError(
                "CloudXR runtime did not become ready; inspect "
                + str(plan.session_root / "cloudxr/logs")
            )
        status.write(
            "starting",
            phase="cloudxr_ready",
            gpu_id=plan.gpu_id,
            signaling_port=plan.signaling_port,
            media_port=plan.media_port,
            proxy_port=plan.proxy_port,
        )
        return asyncio.run(_run_services(plan, runtime, status))
    finally:
        _terminate_sync(runtime)


async def _run_services(
    plan: SessionPlan,
    runtime: subprocess.Popen,
    status: SessionStatusWriter,
) -> int:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, stop.set)
    app = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "clothdojo.teleop.isaac_app",
            "--plan",
            str(plan.session_root / "plan.json"),
        ],
        cwd=plan.session_root,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    startup_deadline = loop.time() + _APP_STARTUP_PHASE_TIMEOUT_S
    startup_phase = None
    app_started = False
    try:
        while True:
            if stop.is_set():
                status.write("stopping", phase="operator_stop", gpu_id=plan.gpu_id)
                await _terminate(app)
                status.write("stopped", phase="operator_stop", gpu_id=plan.gpu_id)
                return 0
            if runtime.poll() is not None:
                raise RuntimeError(f"CloudXR runtime exited with code {runtime.returncode}")
            if app.poll() is not None:
                if app.returncode == 0:
                    status.write("stopped", phase="teleop_exit", gpu_id=plan.gpu_id)
                return int(app.returncode)
            current = None if app_started else _read_status(plan.status_path)
            if current is not None and current.get("state") == "failed":
                await _terminate(app, timeout_s=_FAILED_STARTUP_SHUTDOWN_TIMEOUT_S)
                raise RuntimeError(str(current.get("error") or "Isaac teleoperation failed"))
            app_started = app_started or _app_started(current)
            phase = None if current is None else current.get("phase")
            if not app_started and phase is not None and phase != startup_phase:
                startup_phase = phase
                startup_deadline = loop.time() + _startup_timeout(phase)
            if loop.time() >= startup_deadline and not app_started:
                await _terminate(app, timeout_s=_FAILED_STARTUP_SHUTDOWN_TIMEOUT_S)
                raise RuntimeError(
                    "Isaac teleoperation did not finish startup within "
                    f"{_startup_timeout(phase):.0f}s "
                    f"in phase {phase or 'unknown'}"
                )
            await asyncio.sleep(0.5)
    finally:
        await _terminate(app)


def _write_overrides(plan: SessionPlan, install_dir: Path) -> Path:
    config = install_dir / "config"
    config.mkdir(parents=True, exist_ok=True)
    path = config / "slot.env"
    temporary = config / ".slot.env.tmp"
    vulkan_gpu = int(os.environ.get("CLOUDXR_VULKAN_GPU_INDEX", "0"))
    temporary.write_text(
        "\n".join(
            (
                f"NV_CXR_GPU_INDEX_VULKAN={vulkan_gpu}",
                # CloudXR 6.2 uses this hint to pace applications that render
                # well below the HMD refresh rate. Without it, a 40--50 ms
                # frame can repeatedly miss the 72 Hz display deadline and
                # fall onto an approximately 10 Hz cadence.
                f"NV_MAX_FPS={_MAX_STREAM_FPS}",
                f"NV_CXR_SERVER_PORT={plan.signaling_port}",
                f"NV_CXR_MEDIA_PORT={plan.media_port}",
                f"PROXY_PORT={plan.proxy_port}",
                "",
            )
        ),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return path


@contextmanager
def _short_install_dir(plan: SessionPlan):
    """Keep Monado's Unix socket under Linux SUN_LEN while persisting files in outputs."""
    storage = plan.session_root / "cloudxr"
    storage.mkdir(parents=True, exist_ok=True)
    aliases = Path(tempfile.gettempdir()) / "clothdojo-cxr"
    aliases.mkdir(mode=0o700, parents=True, exist_ok=True)
    alias = aliases / plan.session_id.removeprefix("session_")
    if os.path.lexists(alias):
        raise RuntimeError(f"stale CloudXR runtime alias exists: {alias}")
    alias.symlink_to(storage, target_is_directory=True)
    try:
        yield alias
    finally:
        alias.unlink(missing_ok=True)


def _runtime_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    preload = environment.pop("CLOUDXR_LD_PRELOAD", "").strip()
    if preload:
        inherited = environment.get("LD_PRELOAD", "").strip()
        environment["LD_PRELOAD"] = ":".join(item for item in (preload, inherited) if item)
    return environment


def _app_started(status: dict | None) -> bool:
    return status is not None and status.get("state") in {"waiting_for_vr", "collecting"}


def _startup_timeout(phase: object) -> float:
    if phase == "building_newton":
        return _NEWTON_BUILD_TIMEOUT_S
    if phase == "building_renderer":
        return _RENDERER_BUILD_TIMEOUT_S
    return _APP_STARTUP_PHASE_TIMEOUT_S


def _read_status(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


async def _terminate(process: subprocess.Popen, *, timeout_s: float = _SHUTDOWN_TIMEOUT_S) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = asyncio.get_running_loop().time() + timeout_s
    while process.poll() is None and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await asyncio.to_thread(process.wait)


def _terminate_sync(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


if __name__ == "__main__":
    main()
