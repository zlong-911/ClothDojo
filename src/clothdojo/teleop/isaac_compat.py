"""Compatibility boundary between Isaac Replicator and project-owned Warp."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

_PROJECT_WARP_PACKAGE_PATH: str | None = None


def configure_replicator_warp() -> None:
    """Keep Replicator on Newton's Warp instead of mixing bundled modules."""

    global _PROJECT_WARP_PACKAGE_PATH

    import warp as wp

    if _PROJECT_WARP_PACKAGE_PATH is None:
        _PROJECT_WARP_PACKAGE_PATH = str(Path(wp.__file__).resolve().parent)
    project_path = Path(_PROJECT_WARP_PACKAGE_PATH)
    for name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if (
            name.startswith("warp.")
            and module_file is not None
            and not Path(module_file).resolve().is_relative_to(project_path)
        ):
            del sys.modules[name]
    wp.__path__[:] = [_PROJECT_WARP_PACKAGE_PATH]
    importlib.invalidate_caches()

    # Replicator's decal node imports its bundled, deprecated ``warp.sim`` but
    # never uses it.  Keep that import from mixing Warp 1.8 modules into the
    # project-owned Warp runtime used by Newton.
    if "warp.sim" not in sys.modules:
        sys.modules["warp.sim"] = ModuleType("warp.sim")
    wp.sim = sys.modules["warp.sim"]

    if not hasattr(wp, "context"):
        wp.context = SimpleNamespace(Kernel=wp.Kernel, Device=wp.Device)
    from warp._src import types as internal_types

    for name in (
        "array",
        "np_dtype_to_warp_type",
        "type_size_in_bytes",
        "warp_type_to_np_dtype",
    ):
        if not hasattr(wp.types, name):
            setattr(wp.types, name, getattr(internal_types, name))
    array_type = wp.types.array
    if not getattr(array_type, "_clothdojo_replicator_compatible", False):
        original_init = array_type.__init__

        def compatible_init(self, *args, owner=None, **kwargs):
            original_init(self, *args, **kwargs)

        array_type.__init__ = compatible_init
        array_type._clothdojo_replicator_compatible = True

    for module_name in (
        "warp.examples",
        "warp.fem",
        "warp.fem.linalg",
        "warp.optim",
        "warp.optim.linear",
        "warp.sparse",
    ):
        importlib.import_module(module_name)
