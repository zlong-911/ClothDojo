"""View models for the preserved collection-console interface.

The browser contract deliberately lives here instead of leaking into the data
catalogs.  This keeps the old, proven workspace free to evolve independently
from ClothDojo's canonical storage formats.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from clothdojo.collection.assets import AssetCatalog, AssetStatus
from clothdojo.collection.config import CollectionConfig
from clothdojo.collection.hil import HILCatalog
from clothdojo.collection.review import ReviewCatalog

if TYPE_CHECKING:
    from clothdojo.collection.supervisor import TeleopSupervisor


class CollectionWorkspace:
    """Translate canonical catalogs into the stable browser response shapes."""

    def __init__(
        self,
        config: CollectionConfig,
        assets: AssetCatalog,
        reviews: dict[str, ReviewCatalog],
        hil: dict[str, HILCatalog],
        supervisor: TeleopSupervisor | None = None,
    ) -> None:
        self.config = config
        self.assets = assets
        self.reviews = reviews
        self.hil = hil
        self.supervisor = supervisor

    def snapshot(self) -> dict[str, Any]:
        tasks = []
        asset_ids = tuple(self.assets.library.collectable_ids())
        for profile_id, profile in self.config.profiles.items():
            episodes = self._episodes(profile_id)
            counts = Counter(
                str(item["asset_id"])
                for item in episodes
                if item["status"] == "complete" and item["asset_id"] is not None
            )
            tasks.append(
                {
                    "id": profile.id,
                    "display_name": profile.display_name,
                    "continuation": False,
                    "episode_count": len(episodes),
                    "frame_count": sum(int(item["action_count"]) for item in episodes),
                    "duration_seconds": sum(
                        int(item["action_count"]) / float(item["control_fps"])
                        for item in episodes
                        if float(item["control_fps"]) > 0
                    ),
                    "assets": [
                        {
                            "id": asset_id,
                            "episode_count": counts[asset_id],
                            "default": False,
                        }
                        for asset_id in asset_ids
                    ],
                    "episodes": [self._episode_view(profile_id, item) for item in episodes],
                }
            )
        slots = self.slots()
        return {
            "slots": slots,
            "tasks": tasks,
            "metrics": {
                "available_slots": sum(slot["status"] == "available" for slot in slots),
                "total_slots": len(slots),
            },
        }

    def asset_page(
        self,
        *,
        status: AssetStatus,
        query: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        value = self.assets.page(
            status=status,
            query=query,
            page=page,
            page_size=page_size,
        )
        trajectory_counts = Counter(
            str(item["asset_id"])
            for profile_id in self.config.profiles
            for item in self._episodes(profile_id)
            if item["status"] == "complete" and item["asset_id"] is not None
        )
        value["counts"] = {
            **value["counts"],
            "convertible": value["counts"]["accepted"] - value["counts"]["collectable"],
            "converting": 0,
        }
        value["assets"] = [
            {
                **item,
                "trajectory_count": trajectory_counts[item["id"]],
                "conversion_status": None,
                "conversion_error": None,
                # Conversion is migrated as its own workflow; the preserved UI
                # must not offer a button that launches an undefined command.
                "conversion_available": False,
                "deletion_available": False,
            }
            for item in value["assets"]
        ]
        return value

    def annotation_index(self, profile_id: str) -> dict[str, Any]:
        return {
            item["id"]: self.reviews[profile_id].annotation(item["id"])
            for item in self._episodes(profile_id)
            if item["status"] == "complete"
        }

    def hil_snapshot(self) -> dict[str, Any]:
        profiles = []
        for profile_id, profile in self.config.profiles.items():
            runs = self.hil[profile_id].runs()
            profiles.append(
                {
                    "id": profile.id,
                    "display_name": profile.display_name,
                    "assets": [
                        {"id": asset_id}
                        for asset_id in sorted(
                            {
                                str(episode["asset_id"])
                                for run in runs
                                for episode in run["episodes"]
                            }
                        )
                    ],
                    "runs": [
                        {
                            **run,
                            "episodes": [
                                {**episode, "interventions": []} for episode in run["episodes"]
                            ],
                        }
                        for run in runs
                    ],
                }
            )
        return {"profiles": profiles, "slots": self.slots()}

    def slots(self) -> list[dict[str, Any]]:
        if self.supervisor is not None:
            return self.supervisor.slots()
        return [
            {
                "id": f"gpu-{gpu_id}",
                "worker_id": None,
                "gpu_id": gpu_id,
                "status": "offline",
                "error": "Isaac Teleop supervisor is not configured",
                "endpoint": None,
                "proxy_port": 44401 + index * 10,
                "session": None,
                "export": None,
            }
            for index, gpu_id in enumerate(self.config.teleoperation.gpu_ids)
        ]

    def rollout_run_for_episode(self, profile_id: str, episode_id: str) -> str:
        matches = [
            run["id"]
            for run in self.hil[profile_id].runs()
            if any(item["id"] == episode_id for item in run["episodes"])
        ]
        if len(matches) != 1:
            raise KeyError(f"rollout episode is not unique: {episode_id}")
        return str(matches[0])

    def primary_hil_camera(
        self,
        profile_id: str,
        run_id: str,
        episode_id: str,
    ) -> str:
        for run in self.hil[profile_id].runs():
            if run["id"] != run_id:
                continue
            for episode in run["episodes"]:
                if episode["id"] == episode_id and episode["cameras"]:
                    return str(episode["cameras"][0])
        raise FileNotFoundError(f"HIL video is unavailable: {run_id}/{episode_id}")

    def _episodes(self, profile_id: str) -> list[dict[str, Any]]:
        catalog = self.reviews[profile_id]
        first = catalog.episodes(page=1, page_size=200)
        result = list(first["episodes"])
        for page in range(2, int(first["page_count"]) + 1):
            result.extend(catalog.episodes(page=page, page_size=200)["episodes"])
        return result

    @staticmethod
    def _episode_view(profile_id: str, item: dict[str, Any]) -> dict[str, Any]:
        ready = bool(item["review_ready"])
        return {
            "id": item["id"],
            "status": item["status"],
            "stop_reason": None,
            "reviewable": item["status"] == "complete",
            "usable": item["status"] == "complete",
            "action_count": item["action_count"],
            "duration_seconds": (
                item["action_count"] / item["control_fps"] if item["control_fps"] > 0 else None
            ),
            "session_id": None,
            "profile_id": profile_id,
            "asset_id": item["asset_id"] or "unknown",
            "abort_reason": None,
            "deletable": False,
            "review": (
                {
                    "status": "ready",
                    "cameras": item["cameras"],
                    "fps": item["control_fps"],
                    "frames": item["action_count"],
                }
                if ready
                else None
            ),
        }
