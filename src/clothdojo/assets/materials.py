"""Curated visual materials and their on-disk contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

MaterialRole = Literal["cloth", "table", "ground"]

DEFAULT_MATERIAL_CATALOG = (
    Path(__file__).resolve().parents[3] / "configs/rendering/materials.json"
)
_ASSET_ID = re.compile(r"[a-z0-9][a-z0-9_]*\Z")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_USER_AGENT = "ClothDojo/0.1 material preparation"
_SYSTEM_CA_BUNDLES = (
    Path("/etc/ssl/certs/ca-certificates.crt"),
    Path("/etc/pki/tls/certs/ca-bundle.crt"),
)


@dataclass(frozen=True, slots=True)
class MaterialAsset:
    id: str
    role: MaterialRole
    color_channels: tuple[str, ...]
    size_mm: tuple[float, float]

    def directory(self, root: Path) -> Path:
        return root / "materials" / self.role / self.id

    def base_color_name(self, color_channel: str) -> str:
        if color_channel not in self.color_channels:
            raise ValueError(f"unknown color channel {color_channel!r} for {self.id}")
        if len(self.color_channels) == 1:
            return "base_color.jpg"
        return f"base_color_{color_channel}.jpg"

    def uv_scale(self) -> tuple[float, float]:
        # Cloth panel coordinates are authored in centimetres; table and ground
        # UVs are authored in metres. Poly Haven dimensions are millimetres.
        unit_mm = 10.0 if self.role == "cloth" else 1000.0
        return tuple(unit_mm / size for size in self.size_mm)


@dataclass(frozen=True, slots=True)
class LightingAsset:
    id: str
    exposure_offset: float


@dataclass(frozen=True, slots=True)
class MaterialCatalog:
    materials: tuple[MaterialAsset, ...]
    lighting: tuple[LightingAsset, ...]
    source_files_hashes: tuple[tuple[str, str], ...]
    resolution: str

    def material(self, asset_id: str) -> MaterialAsset:
        matches = [item for item in self.materials if item.id == asset_id]
        if len(matches) != 1:
            raise KeyError(f"unknown material asset: {asset_id}")
        return matches[0]

    def pool(self, role: MaterialRole) -> tuple[MaterialAsset, ...]:
        result = tuple(item for item in self.materials if item.role == role)
        if not result:
            raise ValueError(f"material catalog has no {role} assets")
        return result

    def source_hash(self, asset_id: str) -> str:
        try:
            return dict(self.source_files_hashes)[asset_id]
        except KeyError:
            raise KeyError(f"material catalog has no source hash for {asset_id}") from None


def load_material_catalog(path: str | Path = DEFAULT_MATERIAL_CATALOG) -> MaterialCatalog:
    """Load and validate the project-owned Poly Haven selection."""

    return _load_material_catalog(str(Path(path).expanduser().resolve()))


@cache
def _load_material_catalog(path: str) -> MaterialCatalog:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"material catalog is missing: {source}") from None
    if not isinstance(value, dict) or value.get("schema") != "clothdojo_visual_material_catalog":
        raise ValueError(f"invalid material catalog schema: {source}")
    source_value = value.get("source")
    resolution = source_value.get("working_resolution") if isinstance(source_value, dict) else None
    if resolution not in {"1k", "2k", "4k", "8k"}:
        raise ValueError(f"invalid material working resolution: {resolution!r}")

    material_values = value.get("materials")
    lighting_values = value.get("lighting")
    hash_values = value.get("source_files_hashes")
    if not isinstance(material_values, list) or not isinstance(lighting_values, list):
        raise ValueError("material catalog assets must be lists")
    if not isinstance(hash_values, dict):
        raise ValueError("material catalog source_files_hashes must be an object")

    materials = tuple(_parse_material(item) for item in material_values)
    lighting = tuple(_parse_lighting(item) for item in lighting_values)
    ids = [item.id for item in (*materials, *lighting)]
    if len(ids) != len(set(ids)):
        raise ValueError("material catalog asset ids must be unique")
    if set(hash_values) != set(ids):
        raise ValueError("material catalog source hashes do not match its assets")
    hashes = tuple(
        sorted((_asset_id(key), _source_hash(value)) for key, value in hash_values.items())
    )
    for role in ("cloth", "table", "ground"):
        if not any(item.role == role for item in materials):
            raise ValueError(f"material catalog has no {role} assets")
    if not lighting:
        raise ValueError("material catalog has no lighting assets")
    return MaterialCatalog(
        materials=materials,
        lighting=lighting,
        source_files_hashes=hashes,
        resolution=resolution,
    )


def require_polyhaven_assets(root: str | Path, catalog: MaterialCatalog) -> Path:
    """Validate the small runtime subset without hashing it on every export."""

    target = Path(root).expanduser().resolve()
    missing = [
        str(path.relative_to(target))
        for path in _required_asset_paths(target, catalog)
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise FileNotFoundError(f"Poly Haven visual assets are incomplete: {preview}{suffix}")
    return target


def _required_asset_paths(root: Path, catalog: MaterialCatalog) -> tuple[Path, ...]:
    paths = []
    for item in catalog.materials:
        directory = item.directory(root)
        color_names = [item.base_color_name(channel) for channel in item.color_channels]
        for name in (*color_names, "normal_gl.jpg", "roughness.jpg"):
            paths.append(directory / name)
    for item in catalog.lighting:
        paths.append(root / "lighting" / f"{item.id}.hdr")
    return tuple(paths)


def prepare_polyhaven_assets(
    output: str | Path,
    catalog: MaterialCatalog,
) -> dict[str, int | str]:
    """Download the catalog's exact 2K working set and verify official MD5s."""

    target = Path(output).expanduser().resolve()
    paths = _required_asset_paths(target, catalog)
    try:
        require_polyhaven_assets(target, catalog)
    except FileNotFoundError:
        pass
    else:
        return {
            "output": str(target),
            "downloaded_files": 0,
            "reused_files": len(paths),
            "bytes": sum(path.stat().st_size for path in paths),
        }
    downloaded = 0
    reused = 0
    total_bytes = 0
    for item in catalog.materials:
        info = _api_json(f"https://api.polyhaven.com/info/{item.id}")
        if info.get("files_hash") != catalog.source_hash(item.id):
            raise RuntimeError(f"Poly Haven source files changed for {item.id}")
        files = _api_json(f"https://api.polyhaven.com/files/{item.id}")
        directory = item.directory(target)
        for channel in item.color_channels:
            entry = _file_entry(files, channel, catalog.resolution, "jpg")
            changed, size = _download_verified(entry, directory / item.base_color_name(channel))
            downloaded += changed
            reused += not changed
            total_bytes += size
        for channel, name in (("nor_gl", "normal_gl.jpg"), ("Rough", "roughness.jpg")):
            entry = _file_entry(files, channel, catalog.resolution, "jpg")
            changed, size = _download_verified(entry, directory / name)
            downloaded += changed
            reused += not changed
            total_bytes += size
    for item in catalog.lighting:
        info = _api_json(f"https://api.polyhaven.com/info/{item.id}")
        if info.get("files_hash") != catalog.source_hash(item.id):
            raise RuntimeError(f"Poly Haven source files changed for {item.id}")
        files = _api_json(f"https://api.polyhaven.com/files/{item.id}")
        entry = _file_entry(files, "hdri", catalog.resolution, "hdr")
        changed, size = _download_verified(entry, target / "lighting" / f"{item.id}.hdr")
        downloaded += changed
        reused += not changed
        total_bytes += size
    require_polyhaven_assets(target, catalog)
    return {
        "output": str(target),
        "downloaded_files": downloaded,
        "reused_files": reused,
        "bytes": total_bytes,
    }


def _parse_material(value: object) -> MaterialAsset:
    if not isinstance(value, dict):
        raise ValueError("material catalog entries must be objects")
    asset_id = _asset_id(value.get("id"))
    role = value.get("role")
    channels = value.get("color_channels")
    size = value.get("size_mm")
    if role not in {"cloth", "table", "ground"}:
        raise ValueError(f"invalid role for {asset_id}: {role!r}")
    if not isinstance(channels, list) or not channels or not all(
        isinstance(channel, str) and channel for channel in channels
    ):
        raise ValueError(f"invalid color channels for {asset_id}")
    if len(channels) != len(set(channels)):
        raise ValueError(f"duplicate color channels for {asset_id}")
    if not isinstance(size, list) or len(size) != 2 or not all(
        isinstance(dimension, (int, float)) and not isinstance(dimension, bool) and dimension > 0
        for dimension in size
    ):
        raise ValueError(f"invalid physical size for {asset_id}")
    return MaterialAsset(
        id=asset_id,
        role=role,
        color_channels=tuple(channels),
        size_mm=(float(size[0]), float(size[1])),
    )


def _parse_lighting(value: object) -> LightingAsset:
    if not isinstance(value, dict):
        raise ValueError("lighting catalog entries must be objects")
    asset_id = _asset_id(value.get("id"))
    exposure_offset = value.get("exposure_offset")
    if not isinstance(exposure_offset, (int, float)) or isinstance(
        exposure_offset, bool
    ):
        raise ValueError(f"invalid lighting exposure offset for {asset_id}")
    return LightingAsset(
        id=asset_id,
        exposure_offset=float(exposure_offset),
    )


def _asset_id(value: object) -> str:
    if not isinstance(value, str) or _ASSET_ID.fullmatch(value) is None:
        raise ValueError(f"invalid Poly Haven asset id: {value!r}")
    return value


def _source_hash(value: object) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise ValueError(f"invalid Poly Haven files hash: {value!r}")
    return value


def _api_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60, context=_https_context()) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"Poly Haven API returned a non-object: {url}")
    return value


def _file_entry(files: dict, channel: str, resolution: str, extension: str) -> dict:
    try:
        entry = files[channel][resolution][extension]
        url = entry["url"]
        md5 = entry["md5"]
        size = entry["size"]
    except (KeyError, TypeError):
        raise RuntimeError(
            f"Poly Haven file is unavailable: {channel}/{resolution}/{extension}"
        ) from None
    if not isinstance(url, str) or not isinstance(md5, str) or not isinstance(size, int):
        raise RuntimeError(f"Poly Haven file metadata is invalid: {channel}/{resolution}")
    return {"url": url, "md5": md5, "size": size}


def _download_verified(entry: dict, target: Path) -> tuple[int, int]:
    if target.is_file() and target.stat().st_size == entry["size"]:
        if _md5(target) == entry["md5"]:
            return 0, entry["size"]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    for attempt in range(3):
        request = urllib.request.Request(entry["url"], headers={"User-Agent": _USER_AGENT})
        digest = hashlib.md5(usedforsecurity=False)
        size = 0
        try:
            response = urllib.request.urlopen(request, timeout=120, context=_https_context())
            with response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != entry["size"] or digest.hexdigest() != entry["md5"]:
                raise RuntimeError(
                    f"Poly Haven download failed integrity check: {entry['url']}"
                )
            os.replace(temporary, target)
            return 1, size
        except (OSError, RuntimeError, urllib.error.URLError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@cache
def _https_context() -> ssl.SSLContext:
    paths = ssl.get_default_verify_paths()
    if paths.cafile is not None or paths.capath is not None:
        return ssl.create_default_context()
    bundle = next((path for path in _SYSTEM_CA_BUNDLES if path.is_file()), None)
    if bundle is None:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=str(bundle))
