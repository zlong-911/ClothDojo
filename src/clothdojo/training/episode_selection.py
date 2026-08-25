"""Explicit episode selections shared by offline training workflows."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any


def read_episode_selection(
    path: str | Path,
    *,
    total_episodes: int,
) -> tuple[int, ...]:
    selection_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read episode selection: {selection_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"episode selection must be an object: {selection_path}")
    if payload.get("schema") != "clothdojo_episode_selection":
        raise ValueError(f"unsupported episode selection schema: {selection_path}")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported episode selection version: {selection_path}")
    return validate_episode_selection(payload.get("episodes"), total_episodes)


def validate_episode_selection(value: Any, total_episodes: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("episode selection must be a list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError("episode selection entries must be integers")
    episodes = tuple(value)
    if not episodes:
        raise ValueError("episode selection must be non-empty")
    if any(left >= right for left, right in pairwise(episodes)):
        raise ValueError("episode selection must be strictly increasing")
    if min(episodes) < 0 or max(episodes) >= total_episodes:
        raise ValueError("episode selection is outside the dataset")
    return episodes
