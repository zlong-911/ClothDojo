"""Transactional asset ownership for continuous HIL collection."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from clothdojo.trajectory import TrajectoryReader


def point_key(point: dict) -> tuple[str, str, int]:
    return (point["source_run_id"], point["source_episode_id"], point["source_sample_index"])


def completed_points(output_root: Path) -> set[tuple[str, str, int]]:
    result = set()
    for path in output_root.glob("*/manifest.json"):
        if path.parent.is_symlink():
            continue
        try:
            manifest = TrajectoryReader(path.parent).manifest
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            continue
        if manifest.status == "complete" and manifest.action_count > 0 \
                and manifest.metadata.get("source") == "human_intervention":
            try:
                result.add(point_key(manifest.metadata))
            except KeyError:
                continue
    return result


def reserve_legacy_sessions(database: sqlite3.Connection, profile_id: str) -> None:
    """Old live workers own their entire immutable queues until they exit."""
    for session_id, plan_path in database.execute(
        "SELECT id, plan_path FROM sessions WHERE profile_id = ? AND mode = 'intervention' "
        "AND status IN ('starting','waiting_for_vr','collecting','stopping')", (profile_id,)
    ):
        if not Path(plan_path).is_file() and database.execute(
            "SELECT 1 FROM intervention_leases WHERE session_id = ?", (session_id,)
        ).fetchone():
            # A new session commits its first lease before atomically writing its plan.
            continue
        session = json.loads(Path(plan_path).read_text())
        pool = json.loads(Path(session["intervention_plan"]).read_text())
        if "allocation" in pool:
            continue
        assets = [a["asset_id"] for a in pool.get("assets", [])] or [pool["asset_id"]]
        database.executemany(
            "INSERT OR IGNORE INTO intervention_leases VALUES (?, ?, ?)",
            [(profile_id, asset, session_id) for asset in assets],
        )


def claim_asset(database: sqlite3.Connection, *, profile_id: str, session_id: str,
                points: list[dict], output_root: Path, visited: set[str]) -> list[dict]:
    """Caller holds BEGIN IMMEDIATE; claim one asset and only unfinished unique points."""
    finished = completed_points(output_root)
    occupied = {r[0] for r in database.execute(
        "SELECT asset_id FROM intervention_leases WHERE profile_id = ? AND session_id != ?",
        (profile_id, session_id))}
    available = {}
    for point in points:
        if point["asset_id"] not in occupied | visited and point_key(point) not in finished:
            available.setdefault(point_key(point), point)
    asset = next((p["asset_id"] for p in available.values()), None)
    database.execute("DELETE FROM intervention_leases WHERE session_id = ?", (session_id,))
    if asset is None:
        return []
    database.execute("INSERT INTO intervention_leases VALUES (?, ?, ?)",
                     (profile_id, asset, session_id))
    return [p for p in available.values() if p["asset_id"] == asset]


def claim_next(database_path: Path, **kwargs) -> list[dict]:
    database = sqlite3.connect(database_path, timeout=30)
    try:
        database.execute("BEGIN IMMEDIATE")
        row = database.execute("SELECT status FROM sessions WHERE id = ?",
                               (kwargs["session_id"],)).fetchone()
        if not row or row[0] not in ("starting", "waiting_for_vr", "collecting"):
            raise RuntimeError("HIL session is no longer authorized to claim work")
        reserve_legacy_sessions(database, kwargs["profile_id"])
        points = claim_asset(database, **kwargs)
        database.commit()
        return points
    finally:
        database.close()


def assert_owned(database_path: Path, profile_id: str, session_id: str, asset_id: str) -> None:
    database = sqlite3.connect(database_path, timeout=30)
    try:
        owner = database.execute(
            "SELECT 1 FROM intervention_leases l JOIN sessions s ON s.id=l.session_id "
            "WHERE l.profile_id=? AND l.session_id=? AND l.asset_id=? "
            "AND s.status IN ('starting','waiting_for_vr','collecting')",
            (profile_id, session_id, asset_id),
        ).fetchone()
        if owner is None:
            raise RuntimeError("HIL session no longer owns this garment")
    finally:
        database.close()
