"""Task identities independent of robot workstation selection."""

from __future__ import annotations

from dataclasses import dataclass

SHIRT_FLATTEN = "shirt_flatten"
SHIRT_FOLD = "shirt_fold"


@dataclass(frozen=True, slots=True)
class TaskProfile:
    task_name: str
    scene_available: bool


_PROFILES = {
    SHIRT_FLATTEN: TaskProfile(task_name=SHIRT_FLATTEN, scene_available=True),
    # The identity is reserved now so data can distinguish fold from flatten.
    # Its assets and randomization contract have not been defined yet.
    SHIRT_FOLD: TaskProfile(task_name=SHIRT_FOLD, scene_available=False),
}


def task_profile(task_name: str) -> TaskProfile:
    try:
        return _PROFILES[task_name]
    except KeyError as exc:
        raise ValueError(f"unknown task profile: {task_name!r}") from exc
