"""Paginated garment curation over the canonical asset library."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from clothdojo.assets import GarmentLibrary

AssetStatus = Literal["all", "pending", "accepted", "rejected", "converted", "collectable"]


class AssetCatalog:
    def __init__(self, assets_root: str | Path):
        self.library = GarmentLibrary(assets_root)
        self._lock = threading.RLock()

    def page(
        self,
        *,
        status: AssetStatus = "pending",
        query: str = "",
        page: int = 1,
        page_size: int = 48,
    ) -> dict:
        if status not in {"all", "pending", "accepted", "rejected", "converted", "collectable"}:
            raise ValueError(f"unknown asset status: {status}")
        if page < 1:
            raise ValueError("page must be positive")
        if not 1 <= page_size <= 96:
            raise ValueError("page_size must be between 1 and 96")
        needle = query.strip().casefold()
        with self._lock:
            garments = self.library.garments()
        collectable = set(self.library.collectable_ids())
        counts = {
            "all": len(garments),
            "pending": sum(item.decision == "pending" for item in garments),
            "accepted": sum(item.decision == "accepted" for item in garments),
            "rejected": sum(item.decision == "rejected" for item in garments),
            "converted": sum(item.converted for item in garments),
            "collectable": len(collectable),
        }
        filtered = [
            item
            for item in garments
            if (not needle or needle in item.id.casefold())
            and (
                status == "all"
                or status == item.decision
                or (status == "converted" and item.converted)
                or (status == "collectable" and item.id in collectable)
            )
        ]
        total = len(filtered)
        page_count = max(1, (total + page_size - 1) // page_size)
        page = min(page, page_count)
        start = (page - 1) * page_size
        return {
            "counts": counts,
            "total": total,
            "page": page,
            "page_size": page_size,
            "page_count": page_count,
            "assets": [
                {
                    "id": item.id,
                    "decision": item.decision,
                    "converted": item.converted,
                    "collectable": item.id in collectable,
                    "preview": item.front.is_file() and item.back.is_file(),
                }
                for item in filtered[start : start + page_size]
            ],
        }

    def decide(self, asset_id: str, decision: Literal["accepted", "rejected"]) -> dict:
        with self._lock:
            self.library.decide(asset_id, decision)
            garment = next(item for item in self.library.garments() if item.id == asset_id)
            collectable = asset_id in self.library.collectable_ids()
        return {
            "id": garment.id,
            "decision": garment.decision,
            "converted": garment.converted,
            "collectable": collectable,
        }

    def preview(self, asset_id: str, view: Literal["front", "back"]) -> Path:
        return self.library.preview(asset_id, view)
