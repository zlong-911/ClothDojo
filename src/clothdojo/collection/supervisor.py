"""Small local GPU lease and teleoperation process supervisor."""

from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clothdojo.assets import GarmentLibrary
from clothdojo.collection.config import CollectionConfig
from clothdojo.collection.continuation import build_continuation_plan
from clothdojo.collection.intervention_leases import claim_asset, reserve_legacy_sessions
from clothdojo.teleop.install import TeleopInstallation
from clothdojo.teleop.session import AssetSwitchRequest, SessionPlan

ACTIVE_STATES = ("starting", "waiting_for_vr", "collecting", "stopping")
TERMINAL_STATES = ("stopped", "failed")
SIGNALING_PORT = 44388
MEDIA_PORT = 33434
PROXY_PORT = 44401
PORT_STRIDE = 10
_GATEWAY_START_TIMEOUT_S = 10.0
_GATEWAY_STOP_TIMEOUT_S = 5.0


class NoAvailableSlotError(RuntimeError):
    """Raised when the requested local GPU cannot be leased."""


Launcher = Callable[[SessionPlan, Path, Path], subprocess.Popen]


class TeleopSupervisor:
    """Own only disposable process state; formal episodes own their own state."""

    def __init__(
        self,
        config: CollectionConfig,
        *,
        installation: TeleopInstallation | None = None,
        launcher: Launcher | None = None,
    ) -> None:
        self.config = config
        self.installation = installation or TeleopInstallation.discover(config.project_root)
        self._launcher = launcher or self._launch
        self._processes: dict[str, subprocess.Popen] = {}
        self._gateway_process: subprocess.Popen | None = None
        self._gateway_error: str | None = None
        self._lock = threading.RLock()
        self.database_path = config.state_dir / "control.sqlite3"
        self.sessions_root = config.state_dir / "sessions"
        self.slots_root = config.state_dir / "slots"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.slots_root.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def slots(self) -> list[dict[str, Any]]:
        self.reconcile()
        with closing(self._connect()) as database:
            active = {
                row["slot_id"]: dict(row)
                for row in database.execute(
                    "SELECT * FROM sessions WHERE status IN (?, ?, ?, ?)",
                    ACTIVE_STATES,
                )
            }
        installation_problem = self._gateway_error or self.installation.problem()
        result = []
        for slot in self._slot_specs():
            session = active.get(slot["id"])
            conflicts = [] if session is not None else self._occupied_ports(slot)
            if session is not None:
                status = str(session["status"])
                error = session.get("error")
            elif installation_problem is not None:
                status = "offline"
                error = installation_problem
            elif conflicts:
                status = "port_conflict"
                error = "local ports already in use: " + ", ".join(map(str, conflicts))
            else:
                status = "available"
                error = None
            result.append(
                {
                    **slot,
                    "worker_id": None,
                    "status": status,
                    "error": error,
                    "session": None if session is None else self._public_session(session),
                    "export": None,
                }
            )
        return result

    def start_gateway(self) -> None:
        """Start the single HTTPS/WSS front door shared by every GPU slot."""
        if self._gateway_process is not None and self._gateway_process.poll() is None:
            return
        problem = self.installation.problem()
        if problem is not None:
            self._gateway_error = problem
            return
        if not _can_bind(PROXY_PORT, socket.SOCK_STREAM):
            self._gateway_error = f"CloudXR gateway port is already in use: {PROXY_PORT}"
            return
        gateway_root = self.config.state_dir / "cloudxr"
        gateway_root.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.installation.python),
            "-m",
            "clothdojo.teleop.cloudxr_gateway",
            "--install-dir",
            str(gateway_root),
            "--proxy-port",
            str(PROXY_PORT),
        ]
        for slot in self._slot_specs():
            command.extend(("--signaling-port", str(slot["signaling_port"])))
        environment = self.installation.gateway_environment(
            source_root=self.config.project_root / "src"
        )
        log_path = gateway_root / "gateway.log"
        log = log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                command,
                cwd=self.config.project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()
        deadline = time.monotonic() + _GATEWAY_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self._gateway_error = f"CloudXR gateway exited with code {process.returncode}"
                return
            if not _can_bind(PROXY_PORT, socket.SOCK_STREAM):
                self._gateway_process = process
                self._gateway_error = None
                return
            time.sleep(0.05)
        _terminate_process_group(process, timeout_s=_GATEWAY_STOP_TIMEOUT_S)
        self._gateway_error = "CloudXR gateway did not start listening"

    def stop_gateway(self) -> None:
        process = self._gateway_process
        self._gateway_process = None
        if process is None or process.poll() is not None:
            return
        _terminate_process_group(process, timeout_s=_GATEWAY_STOP_TIMEOUT_S)

    def start_session(
        self,
        *,
        profile_id: str,
        asset_id: str,
        slot_id: str | None = None,
        mode: str = "collection",
        intervention_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_id = _identifier(profile_id, "profile_id")
        asset_id = _identifier(asset_id, "asset_id")
        profile = self.config.profiles.get(profile_id)
        if profile is None:
            raise KeyError(f"unknown collection profile: {profile_id}")
        if mode not in {"collection", "intervention"}:
            raise ValueError(f"unknown teleoperation mode: {mode}")
        if mode == "intervention" and intervention_plan is None:
            raise ValueError("intervention sessions require an immutable plan")
        problem = self.installation.problem()
        if problem is not None:
            raise RuntimeError(problem)
        asset_path = GarmentLibrary(self.config.assets_root).converted(asset_id)
        self.reconcile()
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        created_at = _utc_now()

        with self._lock:
            with closing(self._connect()) as database:
                database.execute("BEGIN IMMEDIATE")
                slot = self._select_slot(database, slot_id)
                if intervention_plan is not None:
                    reserve_legacy_sessions(database, profile_id)
                    claimed = claim_asset(
                        database, profile_id=profile_id, session_id=session_id,
                        points=intervention_plan["breakpoints"],
                        output_root=profile.hil_trajectory_root, visited=set(),
                    )
                    if not claimed:
                        raise NoAvailableSlotError("HIL points are already claimed or completed")
                    asset_id = claimed[0]["asset_id"]
                    asset_path = GarmentLibrary(self.config.assets_root).converted(asset_id)
                    intervention_plan = {**intervention_plan, "asset_id": asset_id,
                        "allocation": {"database_path": str(self.database_path.resolve()),
                                       "session_id": session_id,
                                       "initial_points": claimed}}
                if (
                    intervention_plan is not None
                    and len(intervention_plan.get("assets", ())) > 1
                    and slot["standby_gpu_id"] is None
                ):
                    raise RuntimeError(
                        "cross-asset intervention requires a standby GPU"
                    )
                session_root = self.sessions_root / session_id
                status_path = session_root / "status.json"
                log_path = session_root / "teleop.log"
                plan_path = session_root / "plan.json"
                intervention_path = (
                    session_root / "intervention-plan.json"
                    if intervention_plan is not None
                    else None
                )
                continuation_plan = None
                continuation_path = None
                if mode == "collection" and profile.continuation_source_profile is not None:
                    leased = {
                        str(row["source_episode_id"])
                        for row in database.execute(
                            """
                            SELECT source_episode_id FROM continuation_leases
                            WHERE source_profile_id = ?
                            """,
                            (profile.continuation_source_profile,),
                        )
                    }
                    continuation_plan = build_continuation_plan(
                        self.config,
                        profile,
                        asset_id=asset_id,
                        leased_episode_ids=leased,
                    )
                    continuation_path = session_root / "continuation-plan.json"
                output_root = (
                    profile.hil_trajectory_root
                    if mode == "intervention"
                    else profile.trajectory_root
                )
                plan = SessionPlan(
                    session_id=session_id,
                    mode=mode,
                    profile_id=profile_id,
                    scene=profile.scene,
                    asset_id=asset_id,
                    asset_path=asset_path,
                    assets_root=self.config.assets_root,
                    output_root=output_root,
                    session_root=session_root,
                    status_path=status_path,
                    gpu_id=int(slot["gpu_id"]),
                    standby_gpu_id=slot["standby_gpu_id"],
                    endpoint=str(slot["endpoint"]),
                    signaling_port=int(slot["signaling_port"]),
                    media_port=int(slot["media_port"]),
                    proxy_port=int(slot["proxy_port"]),
                    max_actions=profile.max_actions,
                    intervention_plan=intervention_path,
                    continuation_plan=continuation_path,
                )
                database.execute(
                    """
                    INSERT INTO sessions (
                        id, slot_id, gpu_id, standby_gpu_id, profile_id, asset_id, mode, status,
                        endpoint, output_root, plan_path, status_path, log_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'starting', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        slot["id"],
                        slot["gpu_id"],
                        slot["standby_gpu_id"],
                        profile_id,
                        asset_id,
                        mode,
                        slot["endpoint"],
                        str(output_root),
                        str(plan_path),
                        str(status_path),
                        str(log_path),
                        created_at,
                    ),
                )
                if continuation_plan is not None:
                    database.executemany(
                        """
                        INSERT INTO continuation_leases (
                            source_profile_id, source_episode_id, session_id
                        ) VALUES (?, ?, ?)
                        """,
                        [
                            (
                                profile.continuation_source_profile,
                                source["source_episode_id"],
                                session_id,
                            )
                            for source in continuation_plan["sources"]
                        ],
                    )
                database.commit()

            try:
                session_root.mkdir(parents=True, exist_ok=False)
                if intervention_path is not None:
                    _atomic_json(intervention_path, intervention_plan)
                if continuation_path is not None:
                    _atomic_json(continuation_path, continuation_plan)
                plan.write(plan_path)
                process = self._launcher(plan, plan_path, log_path)
                start_ticks = _pid_start_ticks(process.pid)
            except Exception as exc:
                self._finish(session_id, "failed", error=f"{type(exc).__name__}: {exc}")
                raise
            self._processes[session_id] = process
            with closing(self._connect()) as database:
                database.execute(
                    "UPDATE sessions SET pid = ?, pid_start_ticks = ?, started_at = ? WHERE id = ?",
                    (process.pid, start_ticks, _utc_now(), session_id),
                )
                database.commit()
        return self.session(session_id)

    def request_asset_switch(self, session_id: str, asset_id: str) -> dict[str, Any]:
        """Atomically request a hot asset replacement in a live collection session."""

        asset_id = _identifier(asset_id, "asset_id")
        asset_path = GarmentLibrary(self.config.assets_root).converted(asset_id)
        self.reconcile()
        with self._lock:
            row = self._session_row(session_id)
            if row["status"] not in {"waiting_for_vr", "collecting"}:
                raise RuntimeError("asset switching requires a running teleoperation session")
            if row["mode"] != "collection":
                raise RuntimeError("asset switching is not available during intervention")
            plan = SessionPlan.read(row["plan_path"])
            if plan.standby_gpu_id is None:
                raise RuntimeError("this teleoperation slot has no standby GPU")
            if plan.continuation_plan is not None:
                raise RuntimeError("asset switching is not available during continuation")
            status = _read_status(Path(row["status_path"])) or {}
            current_asset = str(status.get("asset_id") or row["asset_id"])
            if asset_id == current_asset:
                raise ValueError(f"asset {asset_id} is already active")
            if status.get("recording") or status.get("recording_finalizing"):
                raise RuntimeError("finish saving the current recording before switching assets")
            request_path = plan.switch_request_path
            if request_path.exists():
                raise RuntimeError("an asset switch is already pending")
            request = AssetSwitchRequest(
                request_id=uuid.uuid4().hex,
                session_id=session_id,
                asset_id=asset_id,
                asset_path=asset_path,
                requested_at=_utc_now(),
            )
            request.write(request_path)
        return {
            "request_id": request.request_id,
            "session_id": session_id,
            "asset_id": asset_id,
        }

    def stop_session(self, session_id: str) -> dict[str, Any]:
        row = self._session_row(session_id)
        if row["status"] in TERMINAL_STATES:
            return self._public_session(row)
        pid = row.get("pid")
        if pid is None or not _same_process(int(pid), row.get("pid_start_ticks")):
            self._finish(session_id, "failed", error="teleoperation process disappeared")
            return self.session(session_id)
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            self._finish(session_id, "failed", error="teleoperation process disappeared")
            return self.session(session_id)
        with closing(self._connect()) as database:
            database.execute("UPDATE sessions SET status = 'stopping' WHERE id = ?", (session_id,))
            database.commit()
        return self.session(session_id)

    def session(self, session_id: str) -> dict[str, Any]:
        self.reconcile()
        return self._public_session(self._session_row(session_id))

    def session_log(self, session_id: str, *, max_bytes: int = 64 * 1024) -> str:
        row = self._session_row(session_id)
        path = Path(row["log_path"])
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - max_bytes))
                return stream.read(max_bytes).decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def reconcile(self) -> None:
        with self._lock:
            with closing(self._connect()) as database:
                rows = [
                    dict(row)
                    for row in database.execute(
                        "SELECT * FROM sessions WHERE status IN (?, ?, ?, ?)",
                        ACTIVE_STATES,
                    )
                ]
            for row in rows:
                session_id = str(row["id"])
                runtime_status = _read_status(Path(row["status_path"]))
                reported = None if runtime_status is None else runtime_status.get("state")
                reported_asset = (
                    None if runtime_status is None else runtime_status.get("asset_id")
                )
                if (
                    reported in ACTIVE_STATES
                    and reported != row["status"]
                    and row["status"] != "stopping"
                ):
                    with closing(self._connect()) as database:
                        database.execute(
                            "UPDATE sessions SET status = ? WHERE id = ?",
                            (reported, session_id),
                        )
                        database.commit()
                    row["status"] = reported
                if reported_asset is not None and reported_asset != row["asset_id"]:
                    with closing(self._connect()) as database:
                        database.execute(
                            "UPDATE sessions SET asset_id = ? WHERE id = ?",
                            (reported_asset, session_id),
                        )
                        database.commit()
                    row["asset_id"] = reported_asset
                process = self._processes.get(session_id)
                if process is not None:
                    exit_code = process.poll()
                    if exit_code is None:
                        continue
                    self._processes.pop(session_id, None)
                else:
                    pid = row.get("pid")
                    if pid is not None and _same_process(int(pid), row.get("pid_start_ticks")):
                        continue
                    exit_code = row.get("exit_code")
                final = (
                    "stopped"
                    if row["status"] == "stopping" or reported == "stopped" or exit_code == 0
                    else "failed"
                )
                self._finish(
                    session_id,
                    final,
                    exit_code=exit_code,
                    error=None if runtime_status is None else runtime_status.get("error"),
                )

    def _launch(self, plan: SessionPlan, plan_path: Path, log_path: Path) -> subprocess.Popen:
        slot_root = self.slots_root / f"gpu-{plan.gpu_id}"
        slot_root.mkdir(parents=True, exist_ok=True)
        environment = self.installation.environment(
            gpu_id=plan.gpu_id,
            slot_root=slot_root,
            source_root=self.config.project_root / "src",
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab", buffering=0)
        try:
            return subprocess.Popen(
                [
                    str(self.installation.python),
                    "-m",
                    "clothdojo.teleop.cloudxr_slot",
                    "--plan",
                    str(plan_path),
                ],
                cwd=self.config.project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()

    def _public_session(self, row: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: row.get(key)
            for key in (
                "id",
                "slot_id",
                "gpu_id",
                "standby_gpu_id",
                "profile_id",
                "asset_id",
                "mode",
                "status",
                "endpoint",
                "output_root",
                "pid",
                "exit_code",
                "error",
                "created_at",
                "started_at",
                "stopped_at",
            )
        }
        result["worker_id"] = None
        status = _read_status(Path(row["status_path"]))
        if status is not None:
            for key in (
                "phase",
                "asset_id",
                "active_gpu_id",
                "standby_gpu_id",
                "asset_switch_phase",
                "asset_switch_error",
                "teleop_state",
                "recording",
                "recording_finalizing",
                "recording_frames",
                "physics_cuda_graph",
                "ik_cuda_graph",
                "ik_status",
                "loop_fps",
                "frame_ms",
                "control_ms",
                "ik_submit_ms",
                "ik_updates",
                "physics_submit_ms",
                "fabric_submit_ms",
                "app_ms",
                "restoration_phase",
                "intervention_phase",
                "continuation_phase",
                "continuation_remaining",
            ):
                if key in status:
                    result[key] = status[key]
        return result

    def _slot_specs(self) -> list[dict[str, Any]]:
        host = _client_host(self.config.teleoperation.client_host)
        standby_gpu_ids: tuple[int | None, ...] = (
            self.config.teleoperation.standby_gpu_ids
            or (None,) * len(self.config.teleoperation.gpu_ids)
        )
        return [
            {
                "id": f"gpu-{gpu_id}",
                "gpu_id": gpu_id,
                "standby_gpu_id": standby_gpu_id,
                "endpoint": f"{host}:{PROXY_PORT}",
                "signaling_port": SIGNALING_PORT + gpu_id * PORT_STRIDE,
                "media_port": MEDIA_PORT + gpu_id * PORT_STRIDE,
                "proxy_port": PROXY_PORT,
            }
            for gpu_id, standby_gpu_id in zip(
                self.config.teleoperation.gpu_ids,
                standby_gpu_ids,
                strict=True,
            )
        ]

    def _slot(self, slot_id: str) -> dict[str, Any]:
        for slot in self._slot_specs():
            if slot["id"] == slot_id:
                return slot
        raise KeyError(f"unknown teleoperation slot: {slot_id}")

    def _select_slot(self, database: sqlite3.Connection, requested: str | None) -> dict:
        active = {
            row["slot_id"]
            for row in database.execute(
                "SELECT slot_id FROM sessions WHERE status IN (?, ?, ?, ?)",
                ACTIVE_STATES,
            )
        }
        candidates = [self._slot(requested)] if requested else self._slot_specs()
        for slot in candidates:
            if slot["id"] not in active and not self._occupied_ports(slot):
                return slot
        raise NoAvailableSlotError(
            f"{requested} is not available" if requested else "all configured GPUs are in use"
        )

    @staticmethod
    def _occupied_ports(slot: dict[str, Any]) -> list[int]:
        checks = (
            (int(slot["signaling_port"]), socket.SOCK_STREAM),
            (int(slot["media_port"]), socket.SOCK_DGRAM),
        )
        return [port for port, kind in checks if not _can_bind(port, kind)]

    def _session_row(self, session_id: str) -> dict[str, Any]:
        with closing(self._connect()) as database:
            row = database.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown teleoperation session: {session_id}")
        return dict(row)

    def _finish(
        self,
        session_id: str,
        status: str,
        *,
        exit_code: int | None = None,
        error: str | None = None,
    ) -> None:
        with closing(self._connect()) as database:
            database.execute(
                """
                UPDATE sessions
                SET status = ?, stopped_at = ?, exit_code = ?, error = ?
                WHERE id = ?
                """,
                (status, _utc_now(), exit_code, error, session_id),
            )
            database.execute(
                "DELETE FROM continuation_leases WHERE session_id = ?",
                (session_id,),
            )
            database.execute("DELETE FROM intervention_leases WHERE session_id = ?",
                             (session_id,))
            database.commit()

    def _initialize_database(self) -> None:
        with closing(self._connect()) as database:
            database.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    slot_id TEXT NOT NULL,
                    gpu_id INTEGER NOT NULL,
                    standby_gpu_id INTEGER,
                    profile_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    output_root TEXT NOT NULL,
                    plan_path TEXT NOT NULL,
                    status_path TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    pid INTEGER,
                    pid_start_ticks INTEGER,
                    exit_code INTEGER,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    stopped_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_session_per_slot
                ON sessions(slot_id)
                WHERE status IN ('starting', 'waiting_for_vr', 'collecting', 'stopping');
                CREATE TABLE IF NOT EXISTS continuation_leases (
                    source_profile_id TEXT NOT NULL,
                    source_episode_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    PRIMARY KEY (source_profile_id, source_episode_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS intervention_leases (
                    profile_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    PRIMARY KEY (profile_id, asset_id)
                );
                """
            )
            columns = {
                str(row[1]) for row in database.execute("PRAGMA table_info(sessions)")
            }
            if "standby_gpu_id" not in columns:
                database.execute("ALTER TABLE sessions ADD COLUMN standby_gpu_id INTEGER")
                database.commit()

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.database_path, timeout=10.0)
        database.row_factory = sqlite3.Row
        return database


def _identifier(value: str, name: str) -> str:
    value = str(value).strip()
    if not value or len(value) > 160 or any(character in value for character in "/\\\0"):
        raise ValueError(f"{name} is not a valid identifier")
    return value


def _read_status(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_json(path: Path, value: dict[str, Any] | None) -> None:
    if value is None:
        raise ValueError("cannot write an empty teleoperation plan")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _pid_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def _same_process(pid: int, expected_start_ticks: int | None) -> bool:
    if expected_start_ticks is None:
        return False
    try:
        return _pid_start_ticks(pid) == int(expected_start_ticks)
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def _can_bind(port: int, kind: socket.SocketKind) -> bool:
    try:
        with socket.socket(socket.AF_INET, kind) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False


def _terminate_process_group(process: subprocess.Popen, *, timeout_s: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return


def _client_host(configured: str) -> str:
    if configured != "auto":
        return configured
    try:
        resolved = socket.gethostbyname(socket.gethostname())
        if not resolved.startswith("127."):
            return resolved
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("8.8.8.8", 53))
            return str(probe.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
