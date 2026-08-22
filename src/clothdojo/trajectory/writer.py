"""Bounded, atomic writer for ClothDojo trajectory datasets."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from clothdojo.trajectory.schema import (
    ACTION_ARRAY_DTYPES,
    SAMPLE_INDEX_DTYPE,
    STATE_ARRAY_DTYPES,
    SceneMetadata,
    TrajectoryAction,
    TrajectoryManifest,
    TrajectoryState,
)


class TrajectoryWriter:
    """Stream one trajectory in bounded chunks and atomically commit it."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        control_fps: float,
        sim_substeps: int,
        max_actions: int = 3600,
        chunk_size: int = 64,
        units: str = "m",
        scene: SceneMetadata | None = None,
        metadata: dict | None = None,
        compressed: bool = True,
    ) -> None:
        positive = {
            "control_fps": control_fps,
            "sim_substeps": sim_substeps,
            "max_actions": max_actions,
            "chunk_size": chunk_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "manifest.json"
        self.states_path = self.output_dir / "states.npz"
        self.actions_path = self.output_dir / "actions.npz"
        self._chunks_dir = self.output_dir / ".chunks"
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"trajectory output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_dir.mkdir()

        self.manifest = TrajectoryManifest(
            control_fps=float(control_fps),
            sim_substeps=int(sim_substeps),
            max_actions=int(max_actions),
            units=str(units),
            scene=scene or SceneMetadata(),
            metadata=dict(metadata or {}),
        )
        self._chunk_size = int(chunk_size)
        self._compressed = bool(compressed)
        self._state_chunks: _ChunkedArchive | None = None
        self._action_chunks: _ChunkedArchive | None = None
        self._closed = False
        self._write_manifest()

    def write_initial_state(self, state: TrajectoryState) -> None:
        self._require_open()
        if self._state_chunks is not None:
            raise RuntimeError("initial state has already been written")
        if int(state.sample_index) != 0:
            raise ValueError(f"initial state sample_index must be 0, got {state.sample_index}")

        normalized = self._normalize_state(state, establish_shapes=True)
        self._state_chunks = _ChunkedArchive(
            self._chunks_dir,
            name="states",
            chunk_size=self._chunk_size,
            template=normalized,
        )
        self._state_chunks.append(normalized)
        self.manifest.state_count = 1

    def append_transition(self, action: TrajectoryAction, next_state: TrajectoryState) -> None:
        self._require_open()
        if self._state_chunks is None:
            raise RuntimeError("write_initial_state() must be called before append_transition()")
        source_index = self.manifest.action_count
        if source_index >= self.manifest.max_actions:
            raise RuntimeError(f"trajectory exceeds max_actions={self.manifest.max_actions}")
        if int(action.sample_index) != source_index:
            raise ValueError(
                f"action sample_index {action.sample_index} does not match "
                f"source state {source_index}"
            )
        if int(next_state.sample_index) != source_index + 1:
            raise ValueError(
                f"next state sample_index {next_state.sample_index} must equal {source_index + 1}"
            )

        normalized_action = self._normalize_action(
            action,
            establish_shapes=self._action_chunks is None,
        )
        normalized_state = self._normalize_state(next_state, establish_shapes=False)
        if self._action_chunks is None:
            self._action_chunks = _ChunkedArchive(
                self._chunks_dir,
                name="actions",
                chunk_size=self._chunk_size,
                template=normalized_action,
            )
        self._action_chunks.append(normalized_action)
        self._state_chunks.append(normalized_state)
        self.manifest.action_count += 1
        self.manifest.state_count += 1

    def close(self) -> None:
        if self._closed:
            return
        if self._state_chunks is None:
            raise RuntimeError("cannot complete a trajectory without an initial state")
        if self.manifest.state_count != self.manifest.action_count + 1:
            raise RuntimeError("trajectory state/action count invariant failed")

        self._state_chunks.finalize(
            self.states_path,
            self.manifest.state_count,
            compressed=self._compressed,
        )
        if self.manifest.action_count:
            if self._action_chunks is None:
                raise RuntimeError("trajectory action buffer is missing")
            self._action_chunks.finalize(
                self.actions_path,
                self.manifest.action_count,
                compressed=self._compressed,
            )
        self.manifest.status = "complete"
        self._write_manifest()
        self._closed = True
        shutil.rmtree(self._chunks_dir, ignore_errors=True)

    def abort(self) -> None:
        if self._closed:
            return
        self.manifest.status = "aborted"
        self.manifest.state_count = 0
        self.manifest.action_count = 0
        self.manifest.state_shapes = {}
        self.manifest.action_shapes = {}
        self._write_manifest()
        self.states_path.unlink(missing_ok=True)
        self.actions_path.unlink(missing_ok=True)
        shutil.rmtree(self._chunks_dir, ignore_errors=True)
        self._closed = True

    def _normalize_state(
        self,
        state: TrajectoryState,
        *,
        establish_shapes: bool,
    ) -> dict[str, np.ndarray]:
        payload = self._normalize_record(state, STATE_ARRAY_DTYPES)
        shapes = {name: list(payload[name].shape) for name in STATE_ARRAY_DTYPES}
        if establish_shapes:
            self.manifest.state_shapes = shapes
        elif shapes != self.manifest.state_shapes:
            raise ValueError(
                f"state shapes changed: expected {self.manifest.state_shapes}, got {shapes}"
            )
        return payload

    def _normalize_action(
        self,
        action: TrajectoryAction,
        *,
        establish_shapes: bool,
    ) -> dict[str, np.ndarray]:
        payload = self._normalize_record(action, ACTION_ARRAY_DTYPES)
        shapes = {name: list(payload[name].shape) for name in ACTION_ARRAY_DTYPES}
        if establish_shapes:
            self.manifest.action_shapes = shapes
        elif shapes != self.manifest.action_shapes:
            raise ValueError(
                f"action shapes changed: expected {self.manifest.action_shapes}, got {shapes}"
            )
        return payload

    @staticmethod
    def _normalize_record(record, dtypes: dict[str, np.dtype]) -> dict[str, np.ndarray]:
        payload = {"sample_index": np.asarray(int(record.sample_index), dtype=SAMPLE_INDEX_DTYPE)}
        payload.update(
            {
                name: np.ascontiguousarray(getattr(record, name), dtype=dtype).copy()
                for name, dtype in dtypes.items()
            }
        )
        return payload

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed trajectory")

    def _write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self.manifest.to_json_dict())

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class _ChunkedArchive:
    def __init__(
        self,
        directory: Path,
        *,
        name: str,
        chunk_size: int,
        template: dict[str, np.ndarray],
    ) -> None:
        self.directory = directory
        self.name = name
        self.chunk_size = chunk_size
        self._template = {key: (value.dtype, value.shape) for key, value in template.items()}
        self._arrays = {
            key: np.empty((chunk_size, *shape), dtype=dtype)
            for key, (dtype, shape) in self._template.items()
        }
        self._size = 0
        self._paths: list[Path] = []

    def append(self, record: dict[str, np.ndarray]) -> None:
        for key, value in record.items():
            self._arrays[key][self._size] = value
        self._size += 1
        if self._size == self.chunk_size:
            self._flush()

    def finalize(
        self,
        output_path: Path,
        expected_count: int,
        *,
        compressed: bool,
    ) -> None:
        self._flush()
        self._arrays = {}
        _merge_chunks(
            output_path,
            self._paths,
            self._template,
            expected_count,
            compressed=compressed,
        )

    def _flush(self) -> None:
        if self._size == 0:
            return
        path = self.directory / f"{self.name}-{len(self._paths):06d}.npz"
        payload = {key: value[: self._size] for key, value in self._arrays.items()}
        _atomic_save_npz(path, payload, compressed=False)
        self._paths.append(path)
        self._size = 0


def _merge_chunks(
    output_path: Path,
    chunk_paths: list[Path],
    template: dict[str, tuple[np.dtype, tuple[int, ...]]],
    expected_count: int,
    *,
    compressed: bool,
) -> None:
    staging_dir = output_path.parent / f".{output_path.stem}-merge"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir()
    arrays = {
        key: np.lib.format.open_memmap(
            staging_dir / f"{key}.npy",
            mode="w+",
            dtype=dtype,
            shape=(expected_count, *shape),
        )
        for key, (dtype, shape) in template.items()
    }
    offset = 0
    try:
        for chunk_path in chunk_paths:
            with np.load(chunk_path, allow_pickle=False) as chunk:
                count = int(chunk["sample_index"].shape[0])
                for key, target in arrays.items():
                    target[offset : offset + count] = chunk[key]
            offset += count
        if offset != expected_count:
            raise RuntimeError(
                f"{output_path.stem} chunk count mismatch: "
                f"expected={expected_count} actual={offset}"
            )
        for array in arrays.values():
            array.flush()
        _atomic_save_npz(output_path, arrays, compressed=compressed)
    finally:
        arrays.clear()
        shutil.rmtree(staging_dir, ignore_errors=True)


def _atomic_save_npz(
    path: Path,
    payload: dict[str, np.ndarray],
    *,
    compressed: bool,
) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("wb") as file:
        save = np.savez_compressed if compressed else np.savez
        save(file, **payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)
    _fsync_directory(path.parent)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
