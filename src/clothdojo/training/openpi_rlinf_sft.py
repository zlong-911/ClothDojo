"""Worker boundary for RLINF's current ``openpi_rlinf`` SFT model."""

from __future__ import annotations

from typing import Any

from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def _select_pyav_backend() -> None:
    """Select the supported decoder before RLINF constructs the LeRobot dataset."""

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    lerobot_dataset.get_safe_default_codec = lambda: "pyav"


class OpenPiRlinfSftWorker(FSDPVlaSftWorker):
    """Current OpenPI_RLinf worker with the host-supported video decoder."""

    def __init__(self, cfg: Any) -> None:
        _select_pyav_backend()
        super().__init__(cfg)
