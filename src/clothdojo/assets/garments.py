"""Minimal garment storage used by curation and simulation."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Decision = Literal["pending", "accepted", "rejected"]
_ASSET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@dataclass(frozen=True, slots=True)
class Garment:
    id: str
    decision: Decision
    converted: bool
    front: Path
    back: Path


class GarmentLibrary:
    """Own the small on-disk contract shared by the web UI and simulator."""

    def __init__(self, assets_root: str | Path):
        self.root = Path(assets_root).expanduser().absolute() / "garments"
        self.source_root = self.root / "source"
        self.archives_root = self.source_root / "archives"
        self.source_index = self.source_root / "index.jsonl"
        self.previews_root = self.source_root / "previews"
        self.selection_path = self.root / "selection.json"
        self.converted_root = self.root / "converted"
        self._lock = threading.RLock()
        self._garment_cache: tuple[tuple[tuple[int, int], ...], tuple[Garment, ...]] | None = None

    def garments(self) -> tuple[Garment, ...]:
        stamp = self._catalog_stamp()
        with self._lock:
            if self._garment_cache is not None and self._garment_cache[0] == stamp:
                return self._garment_cache[1]
            accepted, rejected = self._selection()
            converted = self._converted_ids()
            ids = sorted(self._preview_ids() | accepted | rejected | converted)
            garments = tuple(
                Garment(
                    id=asset_id,
                    decision=(
                        "accepted"
                        if asset_id in accepted
                        else "rejected"
                        if asset_id in rejected
                        else "pending"
                    ),
                    converted=asset_id in converted,
                    front=self.previews_root / asset_id / "front.png",
                    back=self.previews_root / asset_id / "back.png",
                )
                for asset_id in ids
            )
            self._garment_cache = (stamp, garments)
            return garments

    def decide(self, asset_id: str, decision: Literal["accepted", "rejected"]) -> None:
        asset_id = _asset_id(asset_id)
        if asset_id not in self._known_ids():
            raise KeyError(f"unknown garment: {asset_id}")
        accepted, rejected = self._selection()
        accepted.discard(asset_id)
        rejected.discard(asset_id)
        (accepted if decision == "accepted" else rejected).add(asset_id)
        self._write_selection(accepted, rejected)

    def preview(self, asset_id: str, view: Literal["front", "back"]) -> Path:
        asset_id = _asset_id(asset_id)
        path = self.previews_root / asset_id / f"{view}.png"
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing {view} preview for garment {asset_id}")
        return path

    def converted(self, asset_id: str) -> Path:
        asset_id = _asset_id(asset_id)
        path = self.converted_root / asset_id / f"{asset_id}.usdc"
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"garment is not converted: {asset_id}")
        return path

    def collectable_ids(self) -> tuple[str, ...]:
        return tuple(
            item.id for item in self.garments() if item.decision == "accepted" and item.converted
        )

    def install_converted(self, asset_id: str, source: str | Path) -> Path:
        """Install one completed conversion without exposing a partial directory."""

        asset_id = _asset_id(asset_id)
        if asset_id not in self._known_ids():
            raise KeyError(f"unknown garment: {asset_id}")
        source = Path(source)
        if source.suffix != ".usdc" or not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"conversion output must be a non-empty .usdc file: {source}")
        target = self.converted_root / asset_id
        if target.exists():
            raise FileExistsError(f"converted garment already exists: {asset_id}")

        self.converted_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{asset_id}.", dir=self.converted_root))
        try:
            shutil.copy2(source, temporary / f"{asset_id}.usdc")
            os.replace(temporary, target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        self._invalidate()
        return target / f"{asset_id}.usdc"

    def _known_ids(self) -> set[str]:
        return {item.id for item in self.garments()}

    def _preview_ids(self) -> set[str]:
        if self.source_index.is_file():
            ids = set()
            with self.source_index.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        value = json.loads(line)
                        asset_id = value["id"]
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        raise ValueError(
                            f"invalid garment source index line {line_number}"
                        ) from exc
                    ids.add(_asset_id(asset_id))
            return ids
        if not self.previews_root.is_dir():
            return set()
        return {
            path.name
            for path in self.previews_root.iterdir()
            if path.is_dir() and not path.is_symlink() and _is_asset_id(path.name)
        }

    def _converted_ids(self) -> set[str]:
        if not self.converted_root.is_dir():
            return set()
        return {
            path.name
            for path in self.converted_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and _is_asset_id(path.name)
            and (path / f"{path.name}.usdc").is_file()
        }

    def _selection(self) -> tuple[set[str], set[str]]:
        if not self.selection_path.is_file():
            return set(), set()
        value = json.loads(self.selection_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"accepted", "rejected"}:
            raise ValueError("selection.json must contain accepted and rejected lists")
        accepted = _id_set(value["accepted"], "accepted")
        rejected = _id_set(value["rejected"], "rejected")
        overlap = accepted & rejected
        if overlap:
            raise ValueError(f"garments cannot be accepted and rejected: {sorted(overlap)}")
        return accepted, rejected

    def _write_selection(self, accepted: set[str], rejected: set[str]) -> None:
        self.selection_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"accepted": sorted(accepted), "rejected": sorted(rejected)},
            indent=2,
        )
        descriptor, temporary = tempfile.mkstemp(
            prefix=".selection.", dir=self.selection_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.selection_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self._invalidate()

    def _catalog_stamp(self) -> tuple[tuple[int, int], ...]:
        preview_source = self.source_index if self.source_index.is_file() else self.previews_root
        return tuple(
            _path_stamp(path) for path in (preview_source, self.selection_path, self.converted_root)
        )

    def _invalidate(self) -> None:
        with self._lock:
            self._garment_cache = None


def _id_set(value: object, name: str) -> set[str]:
    if not isinstance(value, list):
        raise ValueError(f"selection.{name} must be a list")
    return {_asset_id(item) for item in value}


def _asset_id(value: object) -> str:
    if not isinstance(value, str) or not _is_asset_id(value):
        raise ValueError(f"invalid garment id: {value!r}")
    return value


def _is_asset_id(value: str) -> bool:
    return _ASSET_ID.fullmatch(value) is not None


def _path_stamp(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)
