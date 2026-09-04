"""Strict reader for ClothDojo trajectory datasets."""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
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
_DERIVED_BASE_FIELDS = ("particle_q", "particle_qd")
_DERIVED_OVERRIDE_FIELDS = tuple(
    field for field in STATE_ARRAY_DTYPES if field not in _DERIVED_BASE_FIELDS
)
_DTYPES = {
    "sample_index": SAMPLE_INDEX_DTYPE,
    **STATE_ARRAY_DTYPES,
    **ACTION_ARRAY_DTYPES,
}


def _stored_array_view(path: Path, archive: zipfile.ZipFile, key: str) -> np.ndarray:
    """Map a numeric NPY member without reading the other trajectory frames."""
    info = archive.getinfo(f"{key}.npy")
    if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
        raise ValueError("stored trajectory member size or compression differs")
    if info.flag_bits & 1:
        raise ValueError("encrypted trajectory arrays are unsupported")
    # ZipFile.open validates the local header, name and member boundaries.
    with archive.open(info) as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
        else:
            raise ValueError(f"unsupported trajectory NPY version {version}")
        payload_offset = member.tell()
    if dtype.hasobject:
        raise ValueError("object trajectory arrays are unsupported")
    count = int(np.prod(shape, dtype=np.int64))
    if count * dtype.itemsize != info.file_size - payload_offset:
        raise ValueError(f"trajectory array payload size differs: {key}")
    if count == 0:
        return np.empty(shape, dtype=dtype)
    with path.open("rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("invalid trajectory ZIP local header")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
    offset = info.header_offset + 30 + name_length + extra_length + payload_offset
    return np.memmap(
        path, mode="r", dtype=dtype, offset=offset, shape=shape,
        order="F" if fortran else "C",
    )


class TrajectoryReader:
    """Read and validate one immutable trajectory directory."""

    def __init__(self, trajectory_dir: str | Path, *, require_complete: bool = True):
        self.trajectory_dir = Path(trajectory_dir)
        with (self.trajectory_dir / "manifest.json").open("r", encoding="utf-8") as file:
            manifest_payload = json.load(file)
        self.manifest = TrajectoryManifest.from_json_dict(
            self._normalize_legacy_manifest_payload(manifest_payload)
        )
        self._validate_manifest(require_complete=require_complete)
        self._state_arrays: dict[str, np.ndarray] | None = None
        self._action_arrays: dict[str, np.ndarray] | None = None
        self._base_reader: TrajectoryReader | None = None
        self._base_sample_indices: np.ndarray | None = None
        self._base_spatial_linear: np.ndarray | None = None
        self._base_spatial_translation: np.ndarray | None = None
        storage = self.manifest.metadata.get("storage")
        if isinstance(storage, dict) and storage.get("kind") in {
            "derived_overlay",
            "derived_overlay_v1",
        }:
            self._base_reader = self._open_base_reader(storage)

    @staticmethod
    def _normalize_legacy_manifest_payload(payload: object) -> dict:
        if not isinstance(payload, dict):
            raise RuntimeError("trajectory manifest must be a JSON object")
        values = dict(payload)
        if "format_version" in values:
            return values
        if (
            values.get("schema_version") != "newton_cloth_episode"
            or values.get("state_profile") != "full"
        ):
            return values
        values.pop("schema_version")
        values.pop("state_profile")
        values["format_version"] = TRAJECTORY_FORMAT_VERSION
        return values

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
        self, fields: Iterable[str] | None = None, *, mmap: bool = False
    ) -> dict[str, np.ndarray]:
        """Return selected validated read-only arrays for vectorized consumers.

        ``mmap`` avoids loading unsampled frames from uncompressed recordings.
        Shape/dtype/index validation is unchanged; as with NPY memory maps,
        partial reads do not verify the full ZIP payload CRC. Compressed or
        derived recordings retain the normal reader path.
        """

        if fields is None:
            return dict(self._states())
        selected = tuple(fields)
        if len(set(selected)) != len(selected) or any(
            field not in STATE_ARRAY_DTYPES for field in selected
        ):
            raise ValueError("state array fields must be unique known state fields")
        if self._base_reader is not None:
            return self._overlay_state_arrays(selected)
        return self._load_arrays(
            self.trajectory_dir / "states.npz",
            _STATE_KEYS,
            self.manifest.state_count,
            self.manifest.state_shapes,
            load_keys=("sample_index", *selected),
            mmap=mmap,
        )

    def action_arrays(self) -> dict[str, np.ndarray]:
        """Return all validated read-only action arrays for vectorized consumers."""

        return dict(self._actions())

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
            if self._base_reader is None:
                self._state_arrays = self._load_arrays(
                    self.trajectory_dir / "states.npz",
                    _STATE_KEYS,
                    self.manifest.state_count,
                    self.manifest.state_shapes,
                )
            else:
                fields = self._overlay_state_arrays(tuple(STATE_ARRAY_DTYPES))
                sample_index = np.arange(
                    self.manifest.state_count, dtype=SAMPLE_INDEX_DTYPE
                )
                sample_index.setflags(write=False)
                self._state_arrays = {"sample_index": sample_index, **fields}
        return self._state_arrays

    def _overlay_state_arrays(self, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
        if self._base_reader is None:
            raise RuntimeError("derived trajectory base reader is missing")
        base_fields = tuple(field for field in fields if field in _DERIVED_BASE_FIELDS)
        override_fields = tuple(field for field in fields if field in _DERIVED_OVERRIDE_FIELDS)
        result: dict[str, np.ndarray] = {}
        if base_fields:
            base_arrays = self._base_reader.state_arrays(base_fields)
            if self._base_sample_indices is None:
                result.update(base_arrays)
            else:
                for field, values in base_arrays.items():
                    selected = values[self._base_sample_indices]
                    selected.setflags(write=False)
                    result[field] = selected
            if self._base_spatial_linear is not None:
                for field in base_fields:
                    transformed = np.asarray(result[field]) @ self._base_spatial_linear.T
                    if field == "particle_q":
                        transformed = transformed + self._base_spatial_translation
                    transformed = np.asarray(transformed, dtype=np.float32)
                    transformed.setflags(write=False)
                    result[field] = transformed
        if override_fields:
            overrides = self._load_arrays(
                self.trajectory_dir / "states.npz",
                ("sample_index", *_DERIVED_OVERRIDE_FIELDS),
                self.manifest.state_count,
                {
                    field: self.manifest.state_shapes[field]
                    for field in _DERIVED_OVERRIDE_FIELDS
                },
                load_keys=("sample_index", *override_fields),
            )
            overrides.pop("sample_index")
            result.update(overrides)
        return {field: result[field] for field in fields}

    def _open_base_reader(self, storage: dict) -> TrajectoryReader:
        value = storage.get("base_trajectory")
        expected_hash = storage.get("base_manifest_sha256")
        if not isinstance(value, str) or not value:
            raise RuntimeError("derived trajectory base_trajectory is missing")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeError("derived trajectory base manifest fingerprint is missing")
        base_path = Path(value).expanduser()
        if not base_path.is_absolute():
            base_path = self.trajectory_dir / base_path
        base_path = base_path.resolve()
        if base_path == self.trajectory_dir.resolve():
            raise RuntimeError("derived trajectory cannot reference itself")
        manifest_path = base_path / "manifest.json"
        try:
            actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(f"cannot read derived trajectory base: {base_path}") from exc
        if actual_hash != expected_hash:
            raise RuntimeError("derived trajectory base manifest fingerprint differs")
        base = TrajectoryReader(base_path)
        if base.manifest.control_fps != self.manifest.control_fps:
            raise RuntimeError("derived trajectory base timeline differs")
        index_file = storage.get("base_sample_indices")
        if index_file is None:
            if (
                base.manifest.state_count != self.manifest.state_count
                or base.manifest.action_count != self.manifest.action_count
            ):
                raise RuntimeError("derived trajectory base timeline differs")
        else:
            if not isinstance(index_file, str) or not index_file:
                raise RuntimeError("derived trajectory base_sample_indices is invalid")
            index_path = self.trajectory_dir / index_file
            try:
                indices = np.load(index_path, allow_pickle=False)
            except OSError as exc:
                raise RuntimeError("derived trajectory base sample map is missing") from exc
            if indices.dtype != SAMPLE_INDEX_DTYPE or indices.shape != (
                self.manifest.state_count,
            ):
                raise RuntimeError("derived trajectory base sample map shape or dtype differs")
            if (
                indices[0] != 0
                or indices[-1] != base.manifest.state_count - 1
                or np.any(np.diff(indices) < 0)
                or np.any(indices < 0)
                or np.any(indices >= base.manifest.state_count)
            ):
                raise RuntimeError("derived trajectory base sample map is invalid")
            indices.setflags(write=False)
            self._base_sample_indices = indices
        for field in _DERIVED_BASE_FIELDS:
            if base.manifest.state_shapes.get(field) != self.manifest.state_shapes.get(field):
                raise RuntimeError(f"derived trajectory base {field} shape differs")
        transform = storage.get("base_spatial_transform")
        if transform is not None:
            self._set_base_spatial_transform(transform)
        return base

    def _set_base_spatial_transform(self, transform: object) -> None:
        if not isinstance(transform, dict) or set(transform) != {"linear", "translation"}:
            raise RuntimeError("derived trajectory base spatial transform is invalid")
        linear = np.asarray(transform["linear"], dtype=np.float64)
        translation = np.asarray(transform["translation"], dtype=np.float64)
        if linear.shape != (3, 3) or translation.shape != (3,):
            raise RuntimeError("derived trajectory base spatial transform shape differs")
        if not np.isfinite(linear).all() or not np.isfinite(translation).all():
            raise RuntimeError("derived trajectory base spatial transform is non-finite")
        if not np.allclose(linear @ linear.T, np.eye(3), atol=1.0e-7, rtol=0.0):
            raise RuntimeError("derived trajectory base spatial transform is not orthogonal")
        determinant = float(np.linalg.det(linear))
        if not np.isclose(abs(determinant), 1.0, atol=1.0e-7, rtol=0.0):
            raise RuntimeError("derived trajectory base spatial transform determinant differs")
        self._base_spatial_linear = linear.astype(np.float32)
        self._base_spatial_translation = translation.astype(np.float32)

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
        mmap: bool = False,
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
            if mmap:
                with zipfile.ZipFile(path) as archive:
                    arrays = {
                        key: (
                            _stored_array_view(path, archive, key)
                            if archive.getinfo(f"{key}.npy").compress_type == zipfile.ZIP_STORED
                            else data[key]
                        )
                        for key in selected_keys
                    }
            else:
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
