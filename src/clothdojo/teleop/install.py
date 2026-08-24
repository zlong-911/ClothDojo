"""Discovery of the fixed Isaac Teleop application bundle."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TeleopInstallation:
    """Machine installation details kept outside collection YAML."""

    root: Path
    python: Path
    cloudxr_library_dir: Path
    web_client_dir: Path
    nvenc_filter: Path | None

    @classmethod
    def discover(cls, project_root: str | Path) -> TeleopInstallation:
        project = Path(project_root).expanduser().resolve()
        override = os.environ.get("CLOTHDOJO_TELEOP_RUNTIME")
        if override:
            root = Path(override).expanduser().resolve()
        elif project.parent.name == "repos":
            root = project.parent.parent / "tools" / "clothdojo-teleop-runtime"
        else:
            root = project / ".teleop-runtime"
        filter_path = root / "nvenc-rm-filter.so"
        return cls(
            root=root,
            python=root / "bin" / "python",
            cloudxr_library_dir=root / "cloudxr-system-libs" / "root/usr/lib/x86_64-linux-gnu",
            web_client_dir=root / "cloudxr-web-client",
            nvenc_filter=filter_path if filter_path.is_file() else None,
        )

    def problem(self) -> str | None:
        required = {
            "Isaac Teleop Python": self.python,
            "CloudXR libraries": self.cloudxr_library_dir,
            "CloudXR web client": self.web_client_dir,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            return "teleoperation runtime is incomplete (" + "; ".join(missing) + ")"
        return None

    def environment(self, *, gpu_id: int, slot_root: Path, source_root: Path) -> dict[str, str]:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        environment["XDG_CACHE_HOME"] = str(slot_root / "cache")
        environment["WARP_CACHE_PATH"] = str(slot_root / "warp-cache")
        environment["PXR_WORK_THREAD_LIMIT"] = "1"
        environment["TELEOP_WEB_CLIENT_STATIC_DIR"] = str(self.web_client_dir)
        existing_python_path = environment.get("PYTHONPATH", "").strip()
        environment["PYTHONPATH"] = ":".join(
            item for item in (str(source_root), existing_python_path) if item
        )
        library_dir = str(self.cloudxr_library_dir)
        inherited = environment.get("LD_LIBRARY_PATH", "").strip()
        environment["LD_LIBRARY_PATH"] = ":".join(item for item in (library_dir, inherited) if item)
        if self.nvenc_filter is not None:
            environment["CLOUDXR_LD_PRELOAD"] = str(self.nvenc_filter)
            environment["CLOUDXR_VULKAN_GPU_INDEX"] = "0"
            environment["NVENC_RM_GPU_ID"] = _nvidia_rm_gpu_id(gpu_id)
        _ensure_loopback_no_proxy(environment)
        return environment


def _nvidia_rm_gpu_id(gpu_id: int) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pci_bus_id = result.stdout.strip()
    try:
        _, bus, device_function = pci_bus_id.rsplit(":", maxsplit=2)
        device, function = device_function.split(".", maxsplit=1)
        rm_gpu_id = (int(bus, 16) << 8) | (int(device, 16) << 3) | int(function, 16)
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"invalid PCI bus ID returned for GPU {gpu_id}: {pci_bus_id!r}") from exc
    return f"0x{rm_gpu_id:08x}"


def _ensure_loopback_no_proxy(environment: dict[str, str]) -> None:
    entries: list[str] = []
    for name in ("NO_PROXY", "no_proxy"):
        entries.extend(item.strip() for item in environment.get(name, "").split(",") if item)
    for host in ("localhost", "127.0.0.1", "::1"):
        if host not in entries:
            entries.append(host)
    value = ",".join(dict.fromkeys(entries))
    environment["NO_PROXY"] = value
    environment["no_proxy"] = value
