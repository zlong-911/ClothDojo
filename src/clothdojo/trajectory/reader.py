"""Strict reader for ClothDojo trajectory datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from clothdojo.trajectory.schema import (
    ACTION_ARRAY_DTYPES,
    SAMPLE_INDEX_DTYPE,
    STATE_ARRAY_DTYPES,
    TRAJECTORY_ALIGNMENT,
    TRAJECTORY_FORMAT_VERSION,
    WORLD_LOCAL_FRAME,
    TrajectoryAction,
    TrajectoryManifest,
    TrajectorySample,
    TrajectoryState,
)

_STATE_KEYS = ("sample_index", *STATE_ARRAY_DTYPES)
_ACTION_KEYS = ("sample_index", *ACTION_ARRAY_DTYPES)
_DTYPES = {
    "sample_index": SAMPLE_INDEX_DTYPE,
    **STATE_ARRAY_DTYPES,
    **ACTION_ARRAY_DTYPES,
}


class TrajectoryReader:
    """Read and validate one immutable trajectory directory."""

    def __init__(self, trajectory_dir: str | Path, *, require_complete: bool = True):
        self.trajectory_dir = Path(trajectory_dir)
        with (self.trajectory_dir / "manifest.json").open("r", encoding="utf-8") as file:
            self.manifest = TrajectoryManifest.from_json_dict(json.load(file))
        self._validate_manifest(require_complete=require_complete)
        self._state_arrays: dict[str, np.ndarray] | None = None
        self._action_arrays: dict[str, np.ndarray] | None = None

    def iter_states(self) -> Iterator[TrajectoryState]:
        arrays = self._states()
        for index in range(self.manifest.state_count):
            yield self._state_at(arrays, index)

    def iter_actions(self) -> Iterator[TrajectoryAction]:
        if self.manifest.action_count == 0:
            return
        arrays = self._actions()
        for index in range(self.manifest.action_count):
            yield self._action_at(arrays, index)

    def iter_samples(
        self,
        *,
        action_start: int = 0,
        action_end_exclusive: int | None = None,
    ) -> Iterator[TrajectorySample]:
        """Yield state/action pairs over one half-open action range."""

        start, end = self._action_range(action_start, action_end_exclusive)
        if start == end:
            return
        states = self._states()
        actions = self._actions()
        for index in range(start, end):
            yield TrajectorySample(
                state=self._state_at(states, index),
                action=self._action_at(actions, index),
            )

    def state_at(self, sample_index: int) -> TrajectoryState:
        index = self._sample_index(sample_index)
        if index >= self.manifest.state_count:
            raise IndexError(f"sample_index {index} is outside [0, {self.manifest.state_count})")
        return self._state_at(self._states(), index)

    def terminal_state(self) -> TrajectoryState:
        return self.state_at(self.manifest.state_count - 1)

    def state_arrays(
        self, fields: Iterable[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Return selected validated read-only arrays for vectorized consumers."""

        if fields is None:
            return dict(self._states())
        selected = tuple(fields)
        if len(set(selected)) != len(selected) or any(
            field not in STATE_ARRAY_DTYPES for field in selected
        ):
            raise ValueError("state array fields must be unique known state fields")
        return self._load_arrays(
            self.trajectory_dir / "states.npz",
            _STATE_KEYS,
            self.manifest.state_count,
            self.manifest.state_shapes,
            load_keys=("sample_index", *selected),
        )

    def validate_training_data(
        self,
        *,
        action_ranges: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        """Validate archives and finite values used as training samples."""

        states = self._states()
        action_count = self.manifest.action_count
        ranges = (
            ((0, action_count),)
            if action_ranges is None
            else tuple(self._action_range(start, end) for start, end in action_ranges)
        )
        if action_count == 0:
            return
        actions = self._actions()
        for start, end in ranges:
            for field in STATE_ARRAY_DTYPES:
                self._require_finite(
                    states[field][start:end],
                    kind="state",
                    field=field,
                    sample_offset=start,
                )
            for field in ACTION_ARRAY_DTYPES:
                self._require_finite(
                    actions[field][start:end],
                    kind="action",
                    field=field,
                    sample_offset=start,
                )

    def _states(self) -> dict[str, np.ndarray]:
        if self._state_arrays is None:
            self._state_arrays = self._load_arrays(
                self.trajectory_dir / "states.npz",
                _STATE_KEYS,
                self.manifest.state_count,
                self.manifest.state_shapes,
            )
        return self._state_arrays

    def _actions(self) -> dict[str, np.ndarray]:
        if self._action_arrays is None:
            self._action_arrays = self._load_arrays(
                self.trajectory_dir / "actions.npz",
                _ACTION_KEYS,
                self.manifest.action_count,
                self.manifest.action_shapes,
            )
        return self._action_arrays

    @staticmethod
    def _state_at(arrays: dict[str, np.ndarray], index: int) -> TrajectoryState:
        return TrajectoryState(
            sample_index=int(arrays["sample_index"][index]),
            **{field: arrays[field][index] for field in STATE_ARRAY_DTYPES},
        )

    @staticmethod
    def _action_at(arrays: dict[str, np.ndarray], index: int) -> TrajectoryAction:
        return TrajectoryAction(
            sample_index=int(arrays["sample_index"][index]),
            **{field: arrays[field][index] for field in ACTION_ARRAY_DTYPES},
        )

    def _action_range(
        self,
        action_start: int,
        action_end_exclusive: int | None,
    ) -> tuple[int, int]:
        end_value = (
            self.manifest.action_count if action_end_exclusive is None else action_end_exclusive
        )
        start = self._integer("action_start", action_start)
        end = self._integer("action_end_exclusive", end_value)
        if start < 0 or end < start or end > self.manifest.action_count:
            raise IndexError(
                "action range must satisfy "
                f"0 <= start <= end <= {self.manifest.action_count}, got [{start}, {end})"
            )
        return start, end

    @staticmethod
    def _sample_index(value: int) -> int:
        index = TrajectoryReader._integer("sample_index", value)
        if index < 0:
            raise IndexError("sample_index must be non-negative")
        return index

    @staticmethod
    def _integer(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        return int(value)

    def _require_finite(
        self,
        values: np.ndarray,
        *,
        kind: str,
        field: str,
        sample_offset: int,
    ) -> None:
        finite = np.isfinite(values)
        if finite.all():
            return
        valid_samples = finite.reshape(finite.shape[0], -1).all(axis=1)
        sample_index = sample_offset + int(np.flatnonzero(~valid_samples)[0])
        raise RuntimeError(
            f"trajectory contains non-finite training {kind}: {self.trajectory_dir}, "
            f"field={field}, sample={sample_index}"
        )

    def _load_arrays(
        self,
        path: Path,
        required_keys: tuple[str, ...],
        expected_count: int,
        expected_shapes: dict[str, list[int]],
        load_keys: tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        if not path.is_file():
            raise RuntimeError(f"trajectory data file is missing: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            actual_keys = set(data.files)
            expected_keys = set(required_keys)
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    f"{path.name} fields differ: missing={missing}, unexpected={unexpected}"
                )
            selected_keys = required_keys if load_keys is None else load_keys
            if any(key not in required_keys for key in selected_keys):
                raise ValueError(f"{path.name} requested unknown fields")
            arrays = {key: data[key] for key in selected_keys}

        wrong_dtypes = {
            key: str(arrays[key].dtype)
            for key in selected_keys
            if arrays[key].dtype != _DTYPES[key]
        }
        if wrong_dtypes:
            raise RuntimeError(f"{path.name} has invalid dtypes: {wrong_dtypes}")
        scalar_fields = [key for key in selected_keys if arrays[key].ndim < 1]
        if scalar_fields:
            raise RuntimeError(f"{path.name} fields lack a sample axis: {scalar_fields}")

        lengths = {key: int(arrays[key].shape[0]) for key in selected_keys}
        if any(length != expected_count for length in lengths.values()):
            raise RuntimeError(
                f"{path.name} count mismatch: manifest={expected_count} arrays={lengths}"
            )
        if arrays["sample_index"].shape != (expected_count,):
            raise RuntimeError(
                f"{path.name} sample_index shape must be ({expected_count},), "
                f"got {arrays['sample_index'].shape}"
            )
        if set(expected_shapes) != set(required_keys) - {"sample_index"}:
            raise RuntimeError(f"{path.name} manifest shape fields are incomplete")
        for key in selected_keys:
            if key == "sample_index":
                continue
            expected_shape = expected_shapes[key]
            actual_shape = list(arrays[key].shape[1:])
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"{path.name} field {key} shape mismatch: "
                    f"manifest={expected_shape} array={actual_shape}"
                )

        expected_indices = np.arange(expected_count, dtype=SAMPLE_INDEX_DTYPE)
        actual_indices = arrays["sample_index"]
        if not np.array_equal(actual_indices, expected_indices):
            position = int(np.flatnonzero(actual_indices != expected_indices)[0])
            raise RuntimeError(
                f"{path.name} sample_index {int(actual_indices[position])} is not "
                f"contiguous at position {position}"
            )
        for array in arrays.values():
            array.setflags(write=False)
        return arrays

    def _validate_manifest(self, *, require_complete: bool) -> None:
        manifest = self.manifest
        supported = {
            "format version": (manifest.format_version, TRAJECTORY_FORMAT_VERSION),
            "alignment": (manifest.alignment, TRAJECTORY_ALIGNMENT),
            "coordinate frame": (manifest.coordinate_frame, WORLD_LOCAL_FRAME),
        }
        for name, (actual, expected) in supported.items():
            if actual != expected:
                raise RuntimeError(f"unsupported trajectory {name}: {actual!r}")
        if require_complete and manifest.status != "complete":
            raise RuntimeError(f"trajectory is not complete: status={manifest.status!r}")
        if manifest.status not in {"recording", "complete", "aborted"}:
            raise RuntimeError(f"unsupported trajectory status: {manifest.status!r}")
        if manifest.control_fps <= 0.0 or manifest.sim_substeps <= 0:
            raise RuntimeError("trajectory control_fps and sim_substeps must be positive")
        if manifest.max_actions <= 0:
            raise RuntimeError("trajectory max_actions must be positive")
        if manifest.state_count == manifest.action_count == 0:
            if manifest.status == "complete":
                raise RuntimeError("a complete trajectory must contain an initial state")
            return
        if manifest.state_count != manifest.action_count + 1:
            raise RuntimeError(
                "trajectory count invariant failed: "
                f"states={manifest.state_count} actions={manifest.action_count}"
            )
        if manifest.action_count > manifest.max_actions:
            raise RuntimeError("trajectory action_count exceeds max_actions")
