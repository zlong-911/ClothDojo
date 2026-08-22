"""Executable acceptance check for the first migrated simulation slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clothdojo.sim.runtime import ShirtFlattenRuntime
from clothdojo.sim.scene import ShirtFlattenSpec, build_shirt_flatten


def run_shirt_flatten_check(
    assets_root: str | Path,
    *,
    device: str = "cuda:0",
    world_count: int = 1,
) -> dict[str, object]:
    """Build, step, reset and repeat the Newton 1.5 scene."""

    try:
        import warp as wp

        scene = build_shirt_flatten(
            assets_root,
            spec=ShirtFlattenSpec(world_count=world_count),
            device=device,
        )
        runtime = ShirtFlattenRuntime(scene)
        initial = _snapshot(runtime)
        arm_target = wp.array(
            scene.arms.get_attribute("joint_q", runtime.state_0).numpy()[:, 0].astype(np.float32),
            dtype=wp.float32,
            device=device,
        )
        gripper_q = scene.grippers.get_attribute("joint_q", runtime.state_0).numpy()[:, 0]
        gripper_lower = scene.grippers.get_attribute("joint_limit_lower", scene.model).numpy()[:, 0]
        gripper_upper = scene.grippers.get_attribute("joint_limit_upper", scene.model).numpy()[:, 0]
        gripper_open = wp.array(
            _gripper_openness(gripper_q, gripper_lower, gripper_upper),
            dtype=wp.float32,
            device=device,
        )

        runtime.step(arm_target, gripper_open)
        first = _snapshot(runtime)

        runtime.reset()
        reset = _snapshot(runtime)
        reset_exact = _snapshots_equal(initial, reset)
        runtime.step(arm_target, gripper_open)
        repeated = _snapshot(runtime)

        finite = _snapshot_finite(first) and _snapshot_finite(repeated)
        bitwise_repeatable = _snapshots_equal(first, repeated)
        compatible = finite and reset_exact
        return {
            "compatible": compatible,
            "device": str(device),
            "world_count": world_count,
            "bodies": int(scene.model.body_count),
            "particles": int(scene.model.particle_count),
            "finite": finite,
            "reset_exact": reset_exact,
            "bitwise_repeatable": bitwise_repeatable,
        }
    except Exception as error:
        return {
            "compatible": False,
            "device": str(device),
            "world_count": world_count,
            "error": f"{type(error).__name__}: {error}",
        }


def _snapshot(runtime: ShirtFlattenRuntime) -> tuple[np.ndarray, ...]:
    state = runtime.state_0
    values = [
        state.joint_q.numpy().copy(),
        state.joint_qd.numpy().copy(),
        state.body_q.numpy().copy(),
        state.body_qd.numpy().copy(),
        state.particle_q.numpy().copy(),
        state.particle_qd.numpy().copy(),
    ]
    for grasp in runtime.grasp.capture_all_states():
        values.extend(value.copy() for value in grasp.values())
    return tuple(values)


def _snapshot_finite(snapshot: tuple[np.ndarray, ...]) -> bool:
    return all(np.isfinite(value).all() for value in snapshot)


def _snapshots_equal(
    left: tuple[np.ndarray, ...],
    right: tuple[np.ndarray, ...],
) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(first, second) for first, second in zip(left, right, strict=True)
    )


def _gripper_openness(q, lower, upper) -> np.ndarray:
    controlled = (0, 2)
    return np.clip(
        (q[:, controlled] - lower[:, controlled]) / (upper[:, controlled] - lower[:, controlled]),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
