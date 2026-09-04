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
            collection = self._hil_collection(runs, interventions)
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
                    "collection": collection,
                }
            )
        return {"profiles": profiles, "slots": self.slots()}

    @staticmethod
    def _hil_collection(
        runs: list[dict[str, Any]],
        interventions: dict[tuple[str, str], list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Build one management row per rollout intervention point."""

        items = []
        for run in runs:
            run_id = str(run["id"])
            for episode in run["episodes"]:
                episode_id = str(episode["id"])
                results_by_sample: dict[int, list[dict[str, Any]]] = {}
                for result in interventions.get((run_id, episode_id), []):
                    results_by_sample.setdefault(int(result["sample_index"]), []).append(
                        result
                    )
                breakpoints = {
                    int(point["sample_index"]): point
                    for point in episode.get("breakpoints", [])
                }
                for sample_index in sorted(set(breakpoints) | set(results_by_sample)):
                    point = breakpoints.get(sample_index)
                    results = sorted(
                        results_by_sample.get(sample_index, []),
                        key=lambda item: str(item["id"]),
                    )
                    collected_at = max(
                        (
                            str(result["collected_at"])
                            for result in results
                            if result.get("collected_at")
                        ),
                        default=None,
                    )
                    items.append(
                        {
                            "id": f"{run_id}/{episode_id}/{sample_index}",
                            "run_id": run_id,
                            "episode_id": episode_id,
                            "sample_index": sample_index,
                            "asset_id": str(episode["asset_id"]),
                            "source_action_count": int(episode["action_count"]),
                            "source_fps": float(episode["fps"]),
                            "source_cameras": list(episode.get("cameras", [])),
                            "created_at": point.get("created_at") if point else None,
                            "collected_at": collected_at,
                            "marked": point is not None,
                            "status": "complete" if results else "pending",
                            "results": results,
                            "result_count": len(results),
                            "result_action_count": sum(
                                int(result["action_count"]) for result in results
                            ),
                        }
                    )
        items.sort(
            key=lambda item: (
                item["collected_at"] is not None,
                str(item["collected_at"] or ""),
                item["run_id"],
                item["episode_id"],
                item["sample_index"],
            ),
            reverse=True,
        )
        return {
            "items": items,
            "marked_point_count": sum(item["marked"] for item in items),
            "pending_count": sum(
                item["marked"] and item["status"] == "pending" for item in items
            ),
            "completed_point_count": sum(
                item["status"] == "complete" for item in items
            ),
            "trajectory_count": sum(item["result_count"] for item in items),
            "frame_count": sum(item["result_action_count"] for item in items),
        }

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
                "proxy_port": 44401,
                "session": None,
                "export": None,
            }
            for gpu_id in self.config.teleoperation.gpu_ids
        ]

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
        results: dict[str, list[dict[str, Any]]] = {}
        for result in destination_episodes:
            metadata = result.get("metadata")
            if (
                result.get("status") == "complete"
                and int(result.get("action_count", 0)) > 0
                and isinstance(metadata, dict)
                and metadata.get("source") == "task_continuation"
                and isinstance(metadata.get("source_episode_id"), str)
            ):
                results.setdefault(metadata["source_episode_id"], []).append(result)

        rows = []
        for source in sources:
            source_id = source["source_episode_id"]
            source_results = results.pop(source_id, [])
            if not source_results:
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
            rows.extend(
                self._continuation_result_row(
                    profile_id,
                    result,
                    source_id=source_id,
                    default_source_profile_id=source_profile_id,
                    default_sample_index=source["source_sample_index"],
                )
                for result in source_results
            )
        for source_id, historical_results in results.items():
            rows.extend(
                self._continuation_result_row(
                    profile_id,
                    result,
                    source_id=source_id,
                    default_source_profile_id=source_profile_id,
                    default_sample_index=None,
                )
                for result in historical_results
            )
        rows.sort(key=lambda item: item["id"], reverse=True)
        return rows

    def _continuation_result_row(
        self,
        profile_id: str,
        result: dict[str, Any],
        *,
        source_id: str,
        default_source_profile_id: str,
        default_sample_index: int | None,
    ) -> dict[str, Any]:
        metadata = result.get("metadata")
        assert isinstance(metadata, dict)
        row = self._episode_view(profile_id, result)
        row.update(
            {
                "id": source_id,
                "source": "task_continuation",
                "source_profile_id": metadata.get(
                    "source_profile_id", default_source_profile_id
                ),
                "source_episode_id": source_id,
                "source_sample_index": metadata.get(
                    "source_sample_index", default_sample_index
                ),
                "result_episode_id": result["id"],
            }
        )
        return row

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
            "deletable": item["status"] != "recording",
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
