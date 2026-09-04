"""Strict configuration for collection workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TeleoperationConfig:
    """User-facing teleoperation choices, never runtime installation details."""

    client_host: str
    gpu_ids: tuple[int, ...]

    def for_client_host(self, host: str) -> TeleoperationConfig:
        host = host.strip()
        if not host:
            raise ValueError("XR client host must not be empty")
        return replace(self, client_host=host)


@dataclass(frozen=True, slots=True)
class CollectionProfile:
    id: str
    display_name: str
    scene: str
    dataset_root: Path
    hil_dataset_root: Path
    rollout_root: Path
    max_actions: int
    continuation_source_profile: str | None = None

    @property
    def trajectory_root(self) -> Path:
        return self.dataset_root / "trajectories"

    @property
    def hil_trajectory_root(self) -> Path:
        return self.hil_dataset_root / "trajectories"

    @property
    def review_root(self) -> Path:
        return self.dataset_root / "review"


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    project_root: Path
    assets_root: Path
    state_dir: Path
    host: str
    port: int
    teleoperation: TeleoperationConfig
    profiles: dict[str, CollectionProfile]


def load_config(path: str | Path) -> CollectionConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"cannot read collection config: {config_path}: {exc}") from exc
    root = _mapping(value, "collection config")
    _keys(
        root,
        {
            "project_root",
            "assets_root",
            "state_dir",
            "listen",
            "teleoperation",
            "profiles",
        },
        "collection config",
    )
    project_root = _path(root["project_root"], config_path.parent)
    assets_root = _path(root["assets_root"], project_root)
    state_dir = _path(root["state_dir"], project_root)

    listen = _mapping(root["listen"], "listen")
    _keys(listen, {"host", "port"}, "listen")
    host = str(listen["host"]).strip()
    if not host:
        raise ValueError("listen.host must not be empty")
    port = _port(listen["port"], "listen.port")

    teleoperation_value = _mapping(root["teleoperation"], "teleoperation")
    _keys(teleoperation_value, {"client_host", "gpu_ids"}, "teleoperation")
    teleoperation = TeleoperationConfig(
        client_host=str(teleoperation_value["client_host"]).strip(),
        gpu_ids=_gpu_ids(teleoperation_value["gpu_ids"]),
    )
    if not teleoperation.client_host:
        raise ValueError("teleoperation.client_host must not be empty")

    profiles_value = _mapping(root["profiles"], "profiles")
    if not profiles_value:
        raise ValueError("profiles must not be empty")
    profiles: dict[str, CollectionProfile] = {}
    for raw_id, raw_profile in profiles_value.items():
        profile_id = _identity(raw_id, "profile id")
        profile = _mapping(raw_profile, f"profiles.{profile_id}")
        _keys(
            profile,
            {
                "display_name",
                "scene",
                "dataset_root",
                "hil_dataset_root",
                "rollout_root",
                "max_actions",
            },
            f"profiles.{profile_id}",
            optional={"continuation_source_profile"},
        )
        display_name = str(profile["display_name"]).strip()
        scene = _identity(profile["scene"], f"profiles.{profile_id}.scene")
        if not display_name:
            raise ValueError(f"profiles.{profile_id}.display_name must not be empty")
        profiles[profile_id] = CollectionProfile(
            id=profile_id,
            display_name=display_name,
            scene=scene,
            dataset_root=_path(profile["dataset_root"], project_root),
            hil_dataset_root=_path(profile["hil_dataset_root"], project_root),
            rollout_root=_path(profile["rollout_root"], project_root),
            max_actions=_positive_integer(
                profile["max_actions"], f"profiles.{profile_id}.max_actions"
            ),
            continuation_source_profile=(
                None
                if profile.get("continuation_source_profile") is None
                else _identity(
                    profile["continuation_source_profile"],
                    f"profiles.{profile_id}.continuation_source_profile",
                )
            ),
        )
    for profile in profiles.values():
        source = profile.continuation_source_profile
        if source is None:
            continue
        if source == profile.id:
            raise ValueError("continuation source profile must differ from its destination")
        if source not in profiles:
            raise ValueError(f"unknown continuation source profile: {source}")
    return CollectionConfig(
        project_root=project_root,
        assets_root=assets_root,
        state_dir=state_dir,
        host=host,
        port=port,
        teleoperation=teleoperation,
        profiles=profiles,
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping")
    return value


def _keys(
    value: dict[str, Any],
    expected: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = set() if optional is None else optional
    missing = expected - set(value)
    unknown = set(value) - expected - optional
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")


def _path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _identity(value: object, name: str) -> str:
    text = str(value)
    if not text or not text.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid {name}: {value!r}")
    return text


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _port(value: object, name: str) -> int:
    result = _positive_integer(value, name)
    if result > 65535:
        raise ValueError(f"{name} must be in [1, 65535]")
    return result


def _gpu_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("teleoperation.gpu_ids must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError("teleoperation.gpu_ids must contain non-negative integers")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError("teleoperation.gpu_ids must not contain duplicates")
    return result
