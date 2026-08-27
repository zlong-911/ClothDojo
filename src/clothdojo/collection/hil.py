"""Browse canonical rollout collections and build immutable HIL plans."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clothdojo.collection.config import CollectionProfile
from clothdojo.rollout.collection import (
    ROLLOUT_COLLECTION_FORMAT_VERSION,
    ROLLOUT_COLLECTION_SCHEMA,
)
from clothdojo.trajectory import TrajectoryReader

HIL_REVIEW_SCHEMA = "clothdojo_hil_review_v1"
INTERVENTION_PLAN_SCHEMA = "clothdojo_intervention_plan_v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class HILCatalog:
    def __init__(self, profile: CollectionProfile):
        self.profile = profile
        self._lock = threading.RLock()
        self._runs_cache: tuple[tuple[tuple[str, int, int], ...], list[dict[str, Any]]] | None = (
            None
        )

    def runs(self) -> list[dict[str, Any]]:
        roots = self._run_roots()
        stamp = tuple(
            (str(root), root.stat().st_mtime_ns, root.stat().st_size) for root in roots
        )
        with self._lock:
            if self._runs_cache is not None and self._runs_cache[0] == stamp:
                return self._runs_cache[1]
        result = []
        for root in roots:
            try:
                metadata = self._metadata(root)
                marks = self._marks(root)
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            episodes = []
            for index, item in enumerate(metadata["trajectories"]):
                relative = Path(item["path"])
                episode_id = relative.name
                manifest = TrajectoryReader(root / relative).manifest
                cameras = [
                    camera
                    for camera in metadata.get("camera_names", [])
                    if self._video_path(root, index, str(camera)).is_file()
                ]
                episodes.append(
                    {
                        "id": episode_id,
                        "asset_id": item["asset_id"],
                        "action_count": item["action_count"],
                        "fps": manifest.control_fps,
                        "initial_state_index": item.get("initial_state_index"),
                        "seed": item.get("seed"),
                        "cameras": cameras,
                        "good": episode_id in marks,
                    }
                )
            result.append(
                {
                    "id": root.name,
                    "status": metadata["status"],
                    "training_eligible": metadata.get("training_eligible"),
                    "checkpoint_sha256": metadata.get("policy", {}).get("checkpoint_sha256"),
                    "episodes": episodes,
                }
            )
        result.sort(key=lambda item: item["id"], reverse=True)
        with self._lock:
            self._runs_cache = (stamp, result)
        return result

    def interventions(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Index completed HIL branches by rollout run and source episode."""

        result: dict[tuple[str, str], list[dict[str, Any]]] = {}
        root = self.profile.intervention_root.resolve()
        if not root.is_dir():
            return result
        for manifest_path in sorted(root.glob("*/manifest.json")):
            trajectory = manifest_path.parent
            if trajectory.is_symlink() or trajectory.resolve().parent != root:
                continue
            try:
                manifest = TrajectoryReader(trajectory).manifest
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                continue
            metadata = manifest.metadata
            run_id = metadata.get("source_run_id")
            episode_id = metadata.get("source_episode_id")
            sample_index = metadata.get("source_sample_index")
            if (
                manifest.status != "complete"
                or manifest.action_count <= 0
                or metadata.get("source") != "human_intervention"
                or not isinstance(run_id, str)
                or not isinstance(episode_id, str)
                or isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
            ):
                continue
            result.setdefault((run_id, episode_id), []).append(
                {
                    "id": trajectory.name,
                    "sample_index": sample_index,
                    "action_count": manifest.action_count,
                }
            )
        return result

    def video(self, run_id: str, episode_id: str, camera: str) -> Path:
        run_id = _identity(run_id, "run id")
        episode_id = _identity(episode_id, "episode id")
        camera = _identity(camera, "camera")
        root = self._run_root(run_id)
        metadata = self._metadata(root)
        for index, item in enumerate(metadata["trajectories"]):
            if Path(item["path"]).name == episode_id:
                if camera not in metadata.get("camera_names", []):
                    break
                path = self._video_path(root, index, camera)
                if path.is_file() and not path.is_symlink():
                    return path
                break
        raise FileNotFoundError(f"HIL video is unavailable: {run_id}/{episode_id}/{camera}")

    def set_good(self, run_id: str, episode_id: str, *, good: bool) -> dict[str, Any]:
        run_id = _identity(run_id, "run id")
        episode_id = _identity(episode_id, "episode id")
        root = self._run_root(run_id)
        metadata = self._metadata(root)
        known = {Path(item["path"]).name for item in metadata["trajectories"]}
        if episode_id not in known:
            raise KeyError(f"unknown rollout trajectory: {episode_id}")
        with self._lock:
            marks = self._marks(root)
            if good:
                marks[episode_id] = {"updated_at": _utc_now()}
            else:
                marks.pop(episode_id, None)
            _atomic_json(
                root / "hil_review.json",
                {"schema": HIL_REVIEW_SCHEMA, "marks": marks},
            )
            self._runs_cache = None
        return {"episode_id": episode_id, "good": good}

    def intervention_plan(
        self,
        run_id: str,
        breakpoints: object,
    ) -> dict[str, Any]:
        run_id = _identity(run_id, "run id")
        if not isinstance(breakpoints, list) or not breakpoints:
            raise ValueError("an intervention requires at least one breakpoint")
        root = self._run_root(run_id)
        metadata = self._metadata(root)
        episodes = {Path(item["path"]).name: item for item in metadata["trajectories"]}
        normalized = []
        asset_id: str | None = None
        runtime: dict[str, Any] | None = None
        for raw in breakpoints:
            if not isinstance(raw, dict) or set(raw) != {"episode_id", "sample_index"}:
                raise ValueError("each breakpoint needs episode_id and sample_index")
            episode_id = _identity(raw["episode_id"], "episode id")
            sample_index = raw["sample_index"]
            if isinstance(sample_index, bool) or not isinstance(sample_index, int):
                raise ValueError("breakpoint sample_index must be an integer")
            item = episodes.get(episode_id)
            if item is None:
                raise KeyError(f"unknown rollout trajectory: {episode_id}")
            if not 0 <= sample_index < int(item["action_count"]):
                raise ValueError(
                    f"breakpoint is outside rollout trajectory: {episode_id}:{sample_index}"
                )
            current_asset = str(item["asset_id"])
            if asset_id is None:
                asset_id = current_asset
            elif asset_id != current_asset:
                raise ValueError("one intervention plan cannot mix garment assets")
            trajectory = root / item["path"]
            reader = TrajectoryReader(trajectory)
            current_runtime = {
                "task_name": reader.manifest.scene.task_name,
                "robot_name": reader.manifest.scene.robot_name,
                "control_fps": reader.manifest.control_fps,
                "sim_substeps": reader.manifest.sim_substeps,
                "asset_fingerprints": reader.manifest.scene.asset_fingerprints,
                "config_fingerprints": reader.manifest.scene.config_fingerprints,
            }
            if runtime is None:
                runtime = current_runtime
            elif runtime != current_runtime:
                raise ValueError("one intervention plan cannot mix runtime identities")
            normalized.append(
                {
                    "source_episode_id": episode_id,
                    "source_sample_index": sample_index,
                    "trajectory": str(trajectory),
                    "manifest_sha256": _sha256(trajectory / "manifest.json"),
                }
            )
        return {
            "schema": INTERVENTION_PLAN_SCHEMA,
            "profile_id": self.profile.id,
            "run_id": run_id,
            "rollout_root": str(root),
            "rollout_meta_sha256": _sha256(root / "meta.json"),
            "asset_id": asset_id,
            "runtime": runtime,
            "breakpoints": normalized,
        }

    def _metadata(self, root: Path) -> dict[str, Any]:
        value = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("rollout metadata must be an object")
        if value.get("schema") != ROLLOUT_COLLECTION_SCHEMA:
            raise ValueError("unsupported rollout schema")
        if value.get("format_version") != ROLLOUT_COLLECTION_FORMAT_VERSION:
            raise ValueError("unsupported rollout version")
        if value.get("status") != "complete":
            raise ValueError("rollout is not complete")
        trajectories = value.get("trajectories")
        if not isinstance(trajectories, list):
            raise ValueError("rollout trajectories must be a list")
        for item in trajectories:
            if not isinstance(item, dict):
                raise ValueError("rollout trajectory entry must be an object")
            relative = item.get("path")
            if not isinstance(relative, str) or Path(relative).parent != Path("trajectories"):
                raise ValueError("rollout trajectory path must be flat")
            if not isinstance(item.get("asset_id"), str):
                raise ValueError("rollout trajectory asset_id is invalid")
            if not isinstance(item.get("action_count"), int):
                raise ValueError("rollout trajectory action_count is invalid")
        return value

    def _run_roots(self) -> list[Path]:
        base = self.profile.rollout_root
        roots = [base] if (base / "meta.json").is_file() else []
        if base.is_dir():
            roots.extend(
                path.parent for path in base.glob("*/meta.json") if not path.parent.is_symlink()
            )
        return sorted(set(roots))

    def _run_root(self, run_id: str) -> Path:
        matches = [root for root in self._run_roots() if root.name == run_id]
        if len(matches) != 1:
            raise KeyError(f"unknown rollout run: {run_id}")
        return matches[0]

    @staticmethod
    def _video_path(root: Path, episode_index: int, camera: str) -> Path:
        review = root / "review" / "videos" / camera / f"episode_{episode_index:06d}.mp4"
        if review.is_file():
            return review
        return (
            root
            / "lerobot"
            / "videos"
            / f"observation.images.{camera}"
            / "chunk-000"
            / f"episode_{episode_index:06d}.mp4"
        )

    @staticmethod
    def _marks(root: Path) -> dict[str, dict[str, str]]:
        path = root / "hil_review.json"
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != HIL_REVIEW_SCHEMA:
            raise RuntimeError(f"invalid HIL review file: {path}")
        marks = value.get("marks")
        if not isinstance(marks, dict):
            raise RuntimeError(f"invalid HIL marks: {path}")
        return dict(marks)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _identity(value: object, name: str) -> str:
    text = str(value)
    if not _ID.fullmatch(text):
        raise ValueError(f"invalid {name}: {value!r}")
    return text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
