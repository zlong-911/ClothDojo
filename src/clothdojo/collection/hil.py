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
from clothdojo.collection.trajectory_store import remove_trajectory, trajectory_path
from clothdojo.rollout.collection import (
    ROLLOUT_COLLECTION_FORMAT_VERSION,
    ROLLOUT_COLLECTION_SCHEMA,
)
from clothdojo.trajectory import TrajectoryReader

HIL_REVIEW_SCHEMA = "clothdojo_hil_review_v3"
PREVIOUS_HIL_REVIEW_SCHEMA = "clothdojo_hil_review_v2"
LEGACY_HIL_REVIEW_SCHEMA = "clothdojo_hil_review_v1"
INTERVENTION_PLAN_SCHEMA = "clothdojo_intervention_plan_v3"
HIL_ROLLOUT_PLAN_SCHEMA = "clothdojo_hil_rollout_plan"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class HILCatalog:
    def __init__(self, profile: CollectionProfile):
        self.profile = profile
        self._lock = threading.RLock()
        self._runs_cache: tuple[tuple[Any, ...], list[dict[str, Any]]] | None = None

    def runs(self) -> list[dict[str, Any]]:
        roots = self._run_roots()
        stamp = tuple(_run_stamp(root) for root in roots)
        with self._lock:
            if self._runs_cache is not None and self._runs_cache[0] == stamp:
                return self._runs_cache[1]
        result = []
        for root in roots:
            try:
                metadata = self._metadata(root)
                review = self._review(root)
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            episodes = []
            for item in metadata["trajectories"]:
                relative = Path(item["path"])
                episode_id = relative.name
                manifest = TrajectoryReader(root / relative).manifest
                cameras = [
                    camera
                    for camera in metadata.get("camera_names", [])
                    if self._video_path(root, episode_id, str(camera)).is_file()
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
                        "good_sample_index": _completion_sample(
                            review["marks"].get(episode_id), int(item["action_count"])
                        ),
                        "breakpoints": _breakpoint_marks(
                            review["breakpoints"].get(episode_id),
                            int(item["action_count"]),
                        ),
                    }
                )
            result.append(
                {
                    "id": root.name,
                    "status": metadata["status"],
                    "training_eligible": metadata.get("training_eligible"),
                    "checkpoint_sha256": metadata.get("policy", {}).get("checkpoint_sha256"),
                    "training_dataset_root": metadata.get("policy", {}).get(
                        "training_dataset_root"
                    ),
                    "dataset_sidecar_sha256": metadata.get("policy", {}).get(
                        "dataset_sidecar_sha256"
                    ),
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
        root = self.profile.hil_trajectory_root.resolve()
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
                    "fps": manifest.control_fps,
                    "collected_at": _trajectory_collected_at(trajectory),
                    "cameras": _intervention_cameras(
                        self.profile.hil_dataset_root, trajectory.name
                    ),
                }
            )
        return result

    def video(self, run_id: str, episode_id: str, camera: str) -> Path:
        run_id = _identity(run_id, "run id")
        episode_id = _identity(episode_id, "episode id")
        camera = _identity(camera, "camera")
        root = self._run_root(run_id)
        metadata = self._metadata(root)
        for item in metadata["trajectories"]:
            if Path(item["path"]).name == episode_id:
                if camera not in metadata.get("camera_names", []):
                    break
                path = self._video_path(root, episode_id, camera)
                if path.is_file() and not path.is_symlink():
                    return path
                break
        raise FileNotFoundError(f"HIL video is unavailable: {run_id}/{episode_id}/{camera}")

    def intervention_video(self, trajectory_id: str, camera: str) -> Path:
        trajectory_id = _identity(trajectory_id, "trajectory id")
        camera = _identity(camera, "camera")
        trajectory = self.profile.hil_trajectory_root / trajectory_id
        if (
            trajectory.is_symlink()
            or trajectory.parent != self.profile.hil_trajectory_root
            or not (trajectory / "manifest.json").is_file()
        ):
            raise FileNotFoundError(f"unknown HIL trajectory: {trajectory_id}")
        path = (
            self.profile.hil_dataset_root
            / "review"
            / "videos"
            / camera
            / f"{trajectory_id}.mp4"
        )
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"HIL review video is unavailable: {trajectory_id}/{camera}"
            )
        return path

    def primary_intervention_camera(self, trajectory_id: str) -> str:
        trajectory_id = _identity(trajectory_id, "trajectory id")
        cameras = _intervention_cameras(self.profile.hil_dataset_root, trajectory_id)
        if not cameras:
            raise FileNotFoundError(f"HIL review video is unavailable: {trajectory_id}")
        return cameras[0]

    def delete_intervention(self, trajectory_id: str) -> dict[str, Any]:
        """Archive one HIL result and restore its rollout breakpoint to pending."""

        trajectory_id = _identity(trajectory_id, "trajectory id")
        source = trajectory_path(self.profile.hil_dataset_root, trajectory_id)
        manifest = TrajectoryReader(source).manifest
        metadata = manifest.metadata
        run_id = metadata.get("source_run_id")
        episode_id = metadata.get("source_episode_id")
        sample_index = metadata.get("source_sample_index")
        if (
            manifest.status != "complete"
            or metadata.get("source") != "human_intervention"
            or not isinstance(run_id, str)
            or not isinstance(episode_id, str)
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
        ):
            raise ValueError(f"trajectory is not a completed HIL result: {trajectory_id}")

        with self._lock:
            self.set_breakpoint(
                run_id,
                episode_id,
                sample_index=sample_index,
                marked=True,
            )
            removed = remove_trajectory(
                self.profile.hil_dataset_root,
                trajectory_id,
                disposition="reject",
            )
            self._runs_cache = None

        return {
            **removed,
            "source_run_id": run_id,
            "source_episode_id": episode_id,
            "source_sample_index": sample_index,
        }

    def set_completion(
        self,
        run_id: str,
        episode_id: str,
        *,
        sample_index: int | None,
    ) -> dict[str, Any]:
        run_id = _identity(run_id, "run id")
        episode_id = _identity(episode_id, "episode id")
        root = self._run_root(run_id)
        metadata = self._metadata(root)
        action_counts = {
            Path(item["path"]).name: int(item["action_count"])
            for item in metadata["trajectories"]
        }
        if episode_id not in action_counts:
            raise KeyError(f"unknown rollout trajectory: {episode_id}")
        if sample_index is not None and (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < action_counts[episode_id]
        ):
            raise ValueError(
                f"completion point is outside rollout trajectory: {episode_id}:{sample_index}"
            )
        with self._lock:
            review = self._review(root)
            marks = _normalized_marks(review["marks"], action_counts)
            if sample_index is not None:
                marks[episode_id] = {
                    "sample_index": sample_index,
                    "updated_at": _utc_now(),
                }
            else:
                marks.pop(episode_id, None)
            _atomic_json(
                root / "hil_review.json",
                {
                    "schema": HIL_REVIEW_SCHEMA,
                    "marks": marks,
                    "breakpoints": _normalized_breakpoints(
                        review["breakpoints"], action_counts
                    ),
                },
            )
            self._runs_cache = None
        return {"episode_id": episode_id, "good_sample_index": sample_index}

    def set_breakpoint(
        self,
        run_id: str,
        episode_id: str,
        *,
        sample_index: int,
        marked: bool,
    ) -> dict[str, Any]:
        """Persist one rollout intervention point immediately beside its review."""

        run_id = _identity(run_id, "run id")
        episode_id = _identity(episode_id, "episode id")
        root = self._run_root(run_id)
        metadata = self._metadata(root)
        action_counts = {
            Path(item["path"]).name: int(item["action_count"])
            for item in metadata["trajectories"]
        }
        action_count = action_counts.get(episode_id)
        if action_count is None:
            raise KeyError(f"unknown rollout trajectory: {episode_id}")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < action_count
        ):
            raise ValueError(
                f"breakpoint is outside rollout trajectory: {episode_id}:{sample_index}"
            )
        with self._lock:
            review = self._review(root)
            breakpoints = _normalized_breakpoints(
                review["breakpoints"], action_counts
            )
            points = breakpoints.setdefault(episode_id, [])
            existing = next(
                (point for point in points if point["sample_index"] == sample_index),
                None,
            )
            if marked and existing is None:
                points.append({"sample_index": sample_index, "created_at": _utc_now()})
            elif not marked and existing is not None:
                points.remove(existing)
            if not points:
                breakpoints.pop(episode_id, None)
            _atomic_json(
                root / "hil_review.json",
                {
                    "schema": HIL_REVIEW_SCHEMA,
                    "marks": _normalized_marks(review["marks"], action_counts),
                    "breakpoints": breakpoints,
                },
            )
            self._runs_cache = None
        return {
            "episode_id": episode_id,
            "breakpoints": breakpoints.get(episode_id, []),
        }

    def intervention_plan(
        self,
        breakpoints: object,
    ) -> dict[str, Any]:
        if not isinstance(breakpoints, list) or not breakpoints:
            raise ValueError("an intervention requires at least one breakpoint")
        run_cache: dict[str, tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]] = {}
        run_provenance: dict[str, dict[str, str]] = {}
        normalized = []
        asset_runtime: dict[str, dict[str, Any]] = {}
        for raw in breakpoints:
            if not isinstance(raw, dict) or set(raw) != {
                "run_id",
                "episode_id",
                "sample_index",
            }:
                raise ValueError("each breakpoint needs run_id, episode_id and sample_index")
            run_id = _identity(raw["run_id"], "run id")
            episode_id = _identity(raw["episode_id"], "episode id")
            sample_index = raw["sample_index"]
            if isinstance(sample_index, bool) or not isinstance(sample_index, int):
                raise ValueError("breakpoint sample_index must be an integer")
            if run_id not in run_cache:
                root = self._run_root(run_id)
                metadata = self._metadata(root)
                episodes = {
                    Path(item["path"]).name: item for item in metadata["trajectories"]
                }
                run_cache[run_id] = (root, metadata, episodes)
                run_provenance[run_id] = {
                    "run_id": run_id,
                    "rollout_root": str(root),
                    "rollout_meta_sha256": _sha256(root / "meta.json"),
                }
            root, _, episodes = run_cache[run_id]
            item = episodes.get(episode_id)
            if item is None:
                raise KeyError(f"unknown rollout trajectory: {run_id}/{episode_id}")
            if not 0 <= sample_index < int(item["action_count"]):
                raise ValueError(
                    f"breakpoint is outside rollout trajectory: {episode_id}:{sample_index}"
                )
            current_asset = str(item["asset_id"])
            trajectory = root / item["path"]
            reader = TrajectoryReader(trajectory)
            current_runtime = {
                "task_name": reader.manifest.scene.task_name,
                "robot_id": reader.manifest.scene.robot_id,
                "control_fps": reader.manifest.control_fps,
                "sim_substeps": reader.manifest.sim_substeps,
                "asset_fingerprints": reader.manifest.scene.asset_fingerprints,
            }
            previous_runtime = asset_runtime.setdefault(current_asset, current_runtime)
            if previous_runtime != current_runtime:
                raise ValueError("one garment asset cannot mix runtime identities")
            normalized.append(
                {
                    "asset_id": current_asset,
                    "source_run_id": run_id,
                    "source_episode_id": episode_id,
                    "source_sample_index": sample_index,
                    "trajectory": str(trajectory),
                    "manifest_sha256": _sha256(trajectory / "manifest.json"),
                }
            )
        asset_order = {asset_id: index for index, asset_id in enumerate(asset_runtime)}
        normalized.sort(key=lambda item: asset_order[str(item["asset_id"])])
        return {
            "schema": INTERVENTION_PLAN_SCHEMA,
            "profile_id": self.profile.id,
            "asset_id": normalized[0]["asset_id"],
            "assets": [
                {"asset_id": asset_id, "runtime": runtime}
                for asset_id, runtime in asset_runtime.items()
            ],
            "runs": list(run_provenance.values()),
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
        self._validate_rollout_plan(root, value)
        return value

    def _validate_rollout_plan(self, root: Path, metadata: dict[str, Any]) -> None:
        path = root / "hil_plan.json"
        if not path.is_file():
            return
        plan = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict) or plan.get("schema") != HIL_ROLLOUT_PLAN_SCHEMA:
            raise ValueError("unsupported HIL rollout plan")
        if plan.get("status") != "ready":
            raise ValueError("HIL rollout plan is not ready")
        rollout = plan.get("rollout")
        formal = plan.get("formal_panel")
        if not isinstance(rollout, dict) or not isinstance(formal, dict):
            raise ValueError("HIL rollout plan provenance is invalid")
        if rollout.get("meta_sha256") != _sha256(root / "meta.json"):
            raise ValueError("HIL rollout metadata changed after planning")
        asset_ids = metadata.get("asset_selection", {}).get("asset_ids")
        if (
            not isinstance(asset_ids, list)
            or any(not isinstance(item, str) or not item for item in asset_ids)
            or len(asset_ids) != len(set(asset_ids))
        ):
            raise ValueError("planned HIL rollout asset ids are invalid")
        if rollout.get("asset_ids_sha256") != _identity_set_sha256(asset_ids):
            raise ValueError("HIL rollout asset set changed after planning")
        if plan.get("candidate_asset_count") != len(asset_ids):
            raise ValueError("HIL rollout candidate asset count changed")
        if plan.get("candidate_trajectory_count") != len(metadata["trajectories"]):
            raise ValueError("HIL rollout candidate trajectory count changed")

        formal_path = Path(str(formal.get("meta_path", ""))).expanduser()
        if not formal_path.is_absolute():
            formal_path = root.parents[2] / formal_path
        formal_path = formal_path.resolve()
        if formal.get("meta_sha256") != _sha256(formal_path):
            raise ValueError("formal panel metadata changed after HIL planning")
        formal_meta = json.loads(formal_path.read_text(encoding="utf-8"))
        formal_assets = formal_meta.get("assets") if isinstance(formal_meta, dict) else None
        if not isinstance(formal_assets, list):
            raise ValueError("formal panel asset list is invalid")
        panel_ids = {
            item.get("asset_id")
            for item in formal_assets
            if isinstance(item, dict) and isinstance(item.get("asset_id"), str)
        }
        unseen_ids = {
            item.get("asset_id")
            for item in formal_assets
            if isinstance(item, dict)
            and item.get("split") == "unseen"
            and isinstance(item.get("asset_id"), str)
        }
        if len(panel_ids) != len(formal_assets):
            raise ValueError("formal panel contains invalid or duplicate asset ids")
        if formal.get("unseen_asset_count") != len(unseen_ids):
            raise ValueError("formal unseen asset count changed after HIL planning")
        if formal.get("unseen_asset_ids_sha256") != _identity_set_sha256(unseen_ids):
            raise ValueError("formal unseen asset set changed after HIL planning")
        selected = set(asset_ids)
        unseen_intersection = selected & unseen_ids
        if formal.get("unseen_intersection_count") != len(unseen_intersection):
            raise ValueError("HIL rollout/formal unseen intersection changed")
        if unseen_intersection:
            raise ValueError("HIL rollout overlaps the formal unseen asset set")
        if formal.get("asset_intersection_count") != len(selected & panel_ids):
            raise ValueError("HIL rollout/formal panel intersection changed")

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
    def _video_path(root: Path, episode_id: str, camera: str) -> Path:
        return root / "review" / "videos" / camera / f"{episode_id}.mp4"

    @staticmethod
    def _review(root: Path) -> dict[str, dict[str, Any]]:
        path = root / "hil_review.json"
        if not path.is_file():
            return {"marks": {}, "breakpoints": {}}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") not in {
            HIL_REVIEW_SCHEMA,
            PREVIOUS_HIL_REVIEW_SCHEMA,
            LEGACY_HIL_REVIEW_SCHEMA,
        }:
            raise RuntimeError(f"invalid HIL review file: {path}")
        schema = value["schema"]
        marks = value.get("marks")
        if (
            not isinstance(marks, dict)
            or any(
                not isinstance(key, str) or not isinstance(mark, dict)
                for key, mark in marks.items()
            )
        ):
            raise RuntimeError(f"invalid HIL marks: {path}")
        if schema != LEGACY_HIL_REVIEW_SCHEMA and any(
            "sample_index" not in mark for mark in marks.values()
        ):
            raise RuntimeError(f"invalid HIL completion marks: {path}")
        breakpoints = value.get("breakpoints", {})
        if (
            not isinstance(breakpoints, dict)
            or any(
                not isinstance(key, str) or not isinstance(points, list)
                for key, points in breakpoints.items()
            )
        ):
            raise RuntimeError(f"invalid HIL breakpoints: {path}")
        return {"marks": dict(marks), "breakpoints": dict(breakpoints)}

    @staticmethod
    def _marks(root: Path) -> dict[str, dict[str, Any]]:
        """Compatibility accessor for callers that only need completion marks."""

        return HILCatalog._review(root)["marks"]


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


def _identity_set_sha256(values: object) -> str:
    if not isinstance(values, (list, set, tuple)):
        raise ValueError("identity set must be a sequence")
    normalized = sorted(str(value) for value in values)
    return hashlib.sha256(("\n".join(normalized) + "\n").encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _completion_sample(mark: object, action_count: int) -> int | None:
    if mark is None:
        return None
    if not isinstance(mark, dict):
        raise RuntimeError("invalid HIL completion mark")
    sample_index = mark.get("sample_index")
    if sample_index is None:
        return action_count - 1 if action_count > 0 else None
    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or not 0 <= sample_index < action_count
    ):
        raise RuntimeError("HIL completion mark is outside its rollout trajectory")
    return sample_index


def _breakpoint_marks(points: object, action_count: int) -> list[dict[str, Any]]:
    if points is None:
        return []
    if not isinstance(points, list):
        raise RuntimeError("invalid HIL breakpoint list")
    result = []
    seen = set()
    for point in points:
        if not isinstance(point, dict):
            raise RuntimeError("invalid HIL breakpoint mark")
        sample_index = point.get("sample_index")
        created_at = point.get("created_at")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < action_count
            or not isinstance(created_at, str)
            or not created_at
        ):
            raise RuntimeError("HIL breakpoint mark is outside its rollout trajectory")
        if sample_index in seen:
            continue
        seen.add(sample_index)
        result.append({"sample_index": sample_index, "created_at": created_at})
    return result


def _normalized_breakpoints(
    breakpoints: dict[str, Any],
    action_counts: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for episode_id, points in breakpoints.items():
        if episode_id not in action_counts:
            continue
        normalized = _breakpoint_marks(points, action_counts[episode_id])
        if normalized:
            result[episode_id] = normalized
    return result


def _normalized_marks(
    marks: dict[str, Any],
    action_counts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    result = {}
    for episode_id, mark in marks.items():
        if episode_id not in action_counts:
            continue
        completion = _completion_sample(mark, action_counts[episode_id])
        if completion is None:
            continue
        result[episode_id] = {
            "sample_index": completion,
            "updated_at": str(mark.get("updated_at") or _utc_now()),
        }
    return result


def _run_stamp(root: Path) -> tuple[Any, ...]:
    stat = root.stat()
    videos = []
    for path in root.glob("review/videos/*/*.mp4"):
        try:
            video_stat = path.stat()
        except FileNotFoundError:
            continue
        videos.append((str(path.relative_to(root)), video_stat.st_mtime_ns, video_stat.st_size))
    return str(root), stat.st_mtime_ns, stat.st_size, tuple(sorted(videos))


def _intervention_cameras(root: Path, trajectory_id: str) -> list[str]:
    videos = root / "review" / "videos"
    result = []
    for path in videos.glob(f"*/{trajectory_id}.mp4"):
        if path.is_file() and not path.is_symlink() and _ID.fullmatch(path.parent.name):
            result.append(path.parent.name)
    return sorted(set(result))


def _trajectory_collected_at(trajectory: Path) -> str:
    match = re.fullmatch(r"teleop_(\d{8}T\d{6}Z)_[A-Za-z0-9]+", trajectory.name)
    if match is not None:
        collected = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=UTC
        )
    else:
        collected = datetime.fromtimestamp(
            (trajectory / "manifest.json").stat().st_mtime,
            tz=UTC,
        )
    return collected.isoformat().replace("+00:00", "Z")
