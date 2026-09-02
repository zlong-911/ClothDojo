"""Deterministic identity for the project-owned Python runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path


def runtime_source_sha256(source_root: str | Path | None = None) -> str:
    """Hash relative names and bytes for every project-owned Python source file."""

    root = (
        Path(source_root).expanduser().resolve()
        if source_root is not None
        else Path(__file__).resolve().parent
    )
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise ValueError(f"runtime source tree contains no Python files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()
