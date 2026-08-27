"""View models for the collection-console interface."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from clothdojo.collection.assets import AssetCatalog, AssetStatus
from clothdojo.collection.config import CollectionConfig
from clothdojo.collection.continuation import continuation_sources
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
            continuation = profile.continuation_source_profile is not None
            display_episodes = (
                self._continuation_episode_rows(profile_id, episodes)
                if continuation
                else [self._episode_view(profile_id, item) for item in episodes]
            )
            tasks.append(
                {
                    "id": profile.id,
                    "display_name": profile.display_name,
                    "continuation": continuation,
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
                    "episodes": display_episodes,
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
        value["assets"] = [
            {
                **item,
                "trajectory_count": trajectory_counts[item["id"]],
            }
            for item in value["assets"]
        ]
        return value

    def annotation_index(self, profile_id: str) -> dict[str, Any]:
        return {
            item["id"]: item["annotation"]
            for item in self._episodes(profile_id)
            if item["status"] == "complete"
        }

    def hil_snapshot(self) -> dict[str, Any]:
        profiles = []
        for profile_id, profile in self.config.profiles.items():
            runs = self.hil[profile_id].runs()
            interventions = self.hil[profile_id].interventions()
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
                                {
                                    **episode,
                                    "interventions": interventions.get(
                                        (run["id"], episode["id"]), []
                                    ),
                                }
                                for episode in run["episodes"]
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
                "proxy_port": 44401 + gpu_id * 10,
                "session": None,
                "export": None,
            }
            for gpu_id in self.config.teleoperation.gpu_ids
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

    def _continuation_episode_rows(
        self,
        profile_id: str,
        destination_episodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        profile = self.config.profiles[profile_id]
        source_profile_id = profile.continuation_source_profile
        assert source_profile_id is not None
        sources = continuation_sources(self.config, profile)
        source_profile_ids = {
            source_profile_id,
            *(item["source_profile_id"] for item in sources),
        }
        results = {}
        for result in destination_episodes:
            metadata = result.get("metadata")
            if (
                result.get("status") == "complete"
                and int(result.get("action_count", 0)) > 0
                and isinstance(metadata, dict)
                and metadata.get("source") == "task_continuation"
                and metadata.get("source_profile_id") in source_profile_ids
                and isinstance(metadata.get("source_episode_id"), str)
            ):
                results.setdefault(metadata["source_episode_id"], result)

        rows = []
        for source in sources:
            source_id = source["source_episode_id"]
            result = results.get(source_id)
            if result is None:
                rows.append(
                    {
                        "id": source_id,
                        "status": "pending",
                        "stop_reason": None,
                        "reviewable": False,
                        "usable": False,
                        "action_count": None,
                        "duration_seconds": None,
                        "session_id": None,
                        "profile_id": profile_id,
                        "asset_id": source["asset_id"],
                        "abort_reason": None,
                        "deletable": False,
                        "review": None,
                        "source": "task_continuation",
                        "source_profile_id": source_profile_id,
                        "source_episode_id": source_id,
                        "source_sample_index": source["source_sample_index"],
                        "result_episode_id": None,
                    }
                )
                continue
            row = self._episode_view(profile_id, result)
            row.update(
                {
                    "id": source_id,
                    "source": "task_continuation",
                    "source_profile_id": source_profile_id,
                    "source_episode_id": source_id,
                    "source_sample_index": source["source_sample_index"],
                    "result_episode_id": result["id"],
                }
            )
            rows.append(row)
        rows.sort(key=lambda item: item["id"], reverse=True)
        return rows

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
