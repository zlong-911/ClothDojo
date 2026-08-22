import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.sim.check import _gripper_openness
from clothdojo.sim.scene import (
    ShirtFlattenSpec,
    _body_indices,
    _joint_dofs,
    _joint_indices,
)


def test_scene_sets_safe_openusd_concurrency_before_newton_import() -> None:
    env = os.environ.copy()
    env.pop("PXR_WORK_THREAD_LIMIT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import clothdojo.sim.scene; print(os.environ['PXR_WORK_THREAD_LIMIT'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "1"


def test_scene_rejects_unlimited_openusd_after_newton_was_imported() -> None:
    env = os.environ.copy()
    env.pop("PXR_WORK_THREAD_LIMIT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import newton; import clothdojo.sim.scene as scene; "
            "scene._limit_openusd_workers(); from pxr import Work; "
            "print(Work.GetConcurrencyLimit())",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "set it before importing Newton" in result.stderr


def test_sim_runtime_does_not_depend_on_trajectory_dataset() -> None:
    path = Path(__file__).parents[1] / "src" / "clothdojo" / "sim" / "runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("clothdojo.trajectory") for module in imports)


def test_piper_asset_is_a_small_self_contained_usd_closure() -> None:
    root = Path(__file__).parents[1] / "assets" / "robots" / "piper"
    if not root.is_dir():
        pytest.skip("project-local assets are installed separately from Git")
    expected = {
        "piper.usda",
        "piper_dual_m.usda",
        "payloads/base.usda",
        "payloads/geometries.usd",
        "payloads/installation.usdc",
        "payloads/installation_materials.usda",
        "payloads/instances.usda",
        "payloads/materials.usda",
        "payloads/robot.usda",
        "payloads/Physics/mujoco.usda",
        "payloads/Physics/physics.usda",
        "payloads/Physics/physx.usda",
    }

    assert {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()} == expected
    assert "@./piper_dual_m.usda@" in (root / "piper.usda").read_text(encoding="utf-8")


def test_joint_selection_resolves_named_one_dof_joints() -> None:
    builder = SimpleNamespace(
        joint_label=["root", "/robot/left/joint1", "/robot/right/joint1"],
        joint_dof_dim=[(0, 0), (0, 1), (0, 1)],
    )

    joints = _joint_indices(builder, ("left/joint1", "right/joint1"))

    assert joints == [1, 2]
    assert _joint_dofs(builder, joints) == [0, 1]


def test_joint_selection_rejects_ambiguous_asset_labels() -> None:
    builder = SimpleNamespace(
        joint_label=["/a/left/joint1", "/b/left/joint1"],
        joint_dof_dim=[(0, 1), (0, 1)],
    )

    with pytest.raises(ValueError, match="expected one Piper joint"):
        _joint_indices(builder, ("left/joint1",))


def test_end_effector_selection_preserves_left_right_order() -> None:
    builder = SimpleNamespace(
        body_label=[
            "/robot/left/base/link6",
            "/robot/right/base/link6",
        ]
    )

    assert _body_indices(builder, ("left/link6", "right/link6")) == [0, 1]


def test_scene_requires_at_least_one_world() -> None:
    with pytest.raises(ValueError, match="world_count"):
        ShirtFlattenSpec(world_count=0)


def test_gripper_openness_uses_one_leader_joint_per_side() -> None:
    q = np.asarray([[0.02, -0.01, 0.04, -0.03]], dtype=np.float32)
    lower = np.asarray([[0.0, -0.04, 0.0, -0.04]], dtype=np.float32)
    upper = np.asarray([[0.04, 0.0, 0.04, 0.0]], dtype=np.float32)

    openness = _gripper_openness(q, lower, upper)

    np.testing.assert_allclose(openness, [[0.5, 1.0]])
