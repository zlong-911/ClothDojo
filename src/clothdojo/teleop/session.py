"""Immutable session plans and atomic runtime status publication."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class SessionPlan:
    """The complete, server-authored input to one local teleoperation process."""

    session_id: str
    mode: Literal["collection", "intervention"]
    profile_id: str
    scene: str
    asset_id: str
    asset_path: Path
    assets_root: Path
    output_root: Path
    session_root: Path
    status_path: Path
    gpu_id: int
    endpoint: str
    signaling_port: int
    media_port: int
    proxy_port: int
    max_actions: int
    intervention_plan: Path | None = None

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in tuple(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
        return {"schema": "clothdojo_teleop_session", "format_version": 1, **value}

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return destination

    @classmethod
    def read(cls, path: str | Path) -> SessionPlan:
        source = Path(path).expanduser().resolve()
        value = json.loads(source.read_text(encoding="utf-8"))
        if value.pop("schema", None) != "clothdojo_teleop_session":
            raise ValueError("not a ClothDojo teleoperation session plan")
        if value.pop("format_version", None) != 1:
            raise ValueError("unsupported teleoperation session plan version")
        for key in (
            "asset_path",
            "assets_root",
            "output_root",
            "session_root",
            "status_path",
            "intervention_plan",
        ):
            if value.get(key) is not None:
                value[key] = Path(value[key]).expanduser().resolve()
        return cls(**value)


class SessionStatusWriter:
    """Publish the latest child-process state without exposing partial JSON."""

    def __init__(self, path: str | Path, *, session_id: str) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.sequence = 0
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if existing.get("session_id") == session_id:
                self.sequence = int(existing.get("sequence", 0))
        except (AttributeError, FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            pass

    def write(self, state: str, **fields: Any) -> None:
        self.sequence += 1
        payload = {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "updated_at": datetime.now(UTC).isoformat(),
            "state": state,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
