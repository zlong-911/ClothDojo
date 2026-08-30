"""Offline deployment of the accepted, project-local robot workstations.

The directory is a snapshot, not a recipe for reassembling upstream CAD. USD
references must remain relative and inside it; renderer-provided MDL modules
are reported separately. No asset IDs, hashes, or mutable download aliases are
introduced here.
"""

from __future__ import annotations

import filecmp
import os
import shutil
import tempfile
from pathlib import Path

# Keep discovery usable before any USD payloads have been installed. In
# particular, importing this module must not construct RobotProfile objects.
WORKSTATION_IDS = ("piper", "aloha2", "x5", "ur5e_2f85", "fr3_duo")
_RENDERER_MODULES = frozenset({"OmniPBR.mdl", "OmniGlass.mdl", "gltf/pbr.mdl"})
_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})


def _asset_file(path: Path, root: Path) -> Path:
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"workstation dependency escapes assets root: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty workstation asset: {path}")
    return path


def _usd_dependencies(entry: Path, root: Path) -> tuple[set[Path], set[str]]:
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
    from pxr import UsdUtils

    files: set[Path] = set()
    renderer_modules: set[str] = set()
    pending = [entry]
    while pending:
        path = _asset_file(pending.pop(), root)
        if path in files:
            continue
        files.add(path)
        if path.suffix.lower() not in _USD_SUFFIXES:
            continue
        # OpenUSD extracts sublayers, references, payloads AND asset-valued
        # attributes (e.g. texture files), including unselected variants.
        for bucket in UsdUtils.ExtractExternalReferences(str(path)):
            for reference in bucket:
                if reference in _RENDERER_MODULES:
                    renderer_modules.add(reference)
                    continue
                if Path(reference).is_absolute() or ":" in reference:
                    raise ValueError(
                        f"workstation references must be portable relative paths: "
                        f"{path}: {reference}"
                    )
                candidate = Path(os.path.abspath(path.parent / reference))
                pending.append(candidate)
    return files, renderer_modules


def inspect_workstation_assets(assets: str | Path = "assets") -> dict:
    """Check the five runtime USD dependency closures without loading Newton."""

    root = Path(assets).expanduser().resolve()
    workstations = {}
    all_files: set[Path] = set()
    for robot_id in WORKSTATION_IDS:
        entry = root / "robots" / robot_id / f"{robot_id}.usda"
        files, modules = _usd_dependencies(entry, root)
        all_files.update(files)
        workstations[robot_id] = {
            "entry": entry.relative_to(root).as_posix(),
            "files": sorted(path.relative_to(root).as_posix() for path in files),
            "renderer_modules": sorted(modules),
        }
    return {
        "assets": str(root),
        "workstations": workstations,
        "runtime_files": len(all_files),
        "runtime_bytes": sum(path.stat().st_size for path in all_files),
    }


def prepare_workstation_assets(source: str | Path, output: str | Path = "assets") -> dict:
    """Copy a complete shared asset snapshot, refusing every existing mismatch.

    Both paths name an assets directory containing ``robots/``. In addition to
    the runtime closure, retain the five robot directories' source meshes,
    MJCF/URDFs and provenance, plus shared camera components. Deprecated robots
    and generated Python caches are not part of this deployment.
    """

    source_root = Path(source).expanduser().resolve()
    target_root = Path(output).expanduser().resolve()
    source_report = inspect_workstation_assets(source_root)
    files = set()
    for name in (*WORKSTATION_IDS, "common"):
        directory = source_root / "robots" / name
        for path in directory.rglob("*"):
            if any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(directory).parts
            ):
                continue
            if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
                _asset_file(path, source_root)
                files.add(path.relative_to(source_root))
    # A new relative dependency outside the usual component directories must
    # still be copied; closure checks cannot silently lose that dependency.
    for item in source_report["workstations"].values():
        files.update(Path(path) for path in item["files"])

    missing = []
    reused = 0
    for relative in sorted(files):
        target = target_root / relative
        if not target.resolve().is_relative_to(target_root):
            raise ValueError(f"workstation destination escapes assets root: {target}")
        if target.exists() or target.is_symlink():
            if not target.is_file() or not filecmp.cmp(
                source_root / relative, target, shallow=False
            ):
                raise FileExistsError(
                    f"workstation asset differs; refusing to overwrite {target}. "
                    "Use an asset snapshot matching this checkout or an empty destination."
                )
            reused += 1
        else:
            missing.append(relative)

    # Check all conflicts before copying anything. Each missing file is
    # installed atomically, without replacing a file created concurrently.
    for relative in missing:
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".install-", dir=target.parent)
        os.close(descriptor)
        try:
            shutil.copy2(source_root / relative, temporary)
            os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)

    report = inspect_workstation_assets(target_root)
    return {
        **report,
        "source": str(source_root),
        "copied_files": len(missing),
        "reused_files": reused,
    }
