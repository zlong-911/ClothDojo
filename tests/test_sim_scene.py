import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo.sim.check import _gripper_openness
from clothdojo.sim.runtime import ShirtFlattenRuntime
from clothdojo.sim.scene import (
    ARM_JOINTS,
    GRIPPER_JOINTS,
    ShirtFlattenSpec,
    _body_indices,
    _configure_robot,
    _joint_dofs,
    _joint_indices,
    _nocs_visual_mesh,
    _vertex_normals,
    rest_vertices_to_nocs,
)


class _RobotBuilder:
    def __init__(self) -> None:
        labels = tuple(
            f"/robot/{side}/joint{joint}"
            for side in ("left", "right")
            for joint in range(1, 9)
        )
        self.joint_label = list(labels)
        self.joint_dof_dim = [(0, 1)] * len(labels)
        self.joint_limit_lower = np.full(len(labels), -10.0)
        self.joint_limit_upper = np.full(len(labels), 10.0)
        self.joint_q = np.zeros(len(labels))
        self.joint_target_q = np.zeros(len(labels))
        self.joint_target_mode = np.zeros(len(labels), dtype=np.int32)
        self.joint_target_ke = np.zeros(len(labels))
        self.joint_target_kd = np.zeros(len(labels))
        self.joint_effort_limit = np.zeros(len(labels))
        self.joint_armature = np.zeros(len(labels))
        self.custom_attributes = {
            "mujoco:gravcomp": SimpleNamespace(values=None),
            "mujoco:jnt_actgravcomp": SimpleNamespace(values=None),
        }
        self.mimics: list[dict[str, object]] = []

    def add_constraint_mimic(self, **kwargs) -> None:
        self.mimics.append(kwargs)


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


def test_scene_preserves_host_openusd_concurrency_after_newton_import() -> None:
    env = os.environ.copy()
    env.pop("PXR_WORK_THREAD_LIMIT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import newton; from pxr import Work; before = Work.GetConcurrencyLimit(); "
            "import clothdojo.sim.scene; print(before, Work.GetConcurrencyLimit())",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    before, after = (int(value) for value in result.stdout.split())
    assert after == before


def test_sim_runtime_does_not_depend_on_trajectory_dataset() -> None:
    path = Path(__file__).parents[1] / "src" / "clothdojo" / "sim" / "runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("clothdojo.trajectory") for module in imports)


def test_sim_runtime_enables_cuda_graph_by_default() -> None:
    assert ShirtFlattenRuntime.__init__.__kwdefaults__ == {"use_cuda_graph": True}

    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.spec = SimpleNamespace(sim_substeps=3)
    with pytest.raises(RuntimeError, match="even sim_substeps"):
        runtime._capture_step_graph()


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


def test_piper_control_matches_legacy_limits_and_gripper_mimics() -> None:
    builder = _RobotBuilder()
    newton = SimpleNamespace(JointTargetMode=SimpleNamespace(POSITION=1))

    arm_joints, gripper_joints, arm_dofs = _configure_robot(
        newton, builder, ShirtFlattenSpec(), body_count=21
    )

    assert arm_joints == _joint_indices(builder, ARM_JOINTS)
    assert gripper_joints == _joint_indices(builder, GRIPPER_JOINTS)
    np.testing.assert_allclose(builder.joint_limit_lower[arm_dofs][[0, 6]], -0.5 * np.pi)
    np.testing.assert_allclose(builder.joint_limit_upper[arm_dofs][[0, 6]], 0.5 * np.pi)
    assert builder.mimics == [
        {
            "joint0": gripper_joints[1],
            "joint1": gripper_joints[0],
            "coef0": 0.0,
            "coef1": -1.0,
            "label": "left_gripper_mimic",
        },
        {
            "joint0": gripper_joints[3],
            "joint1": gripper_joints[2],
            "coef0": 0.0,
            "coef1": -1.0,
            "label": "right_gripper_mimic",
        },
    ]


def test_scene_requires_at_least_one_world() -> None:
    with pytest.raises(ValueError, match="world_count"):
        ShirtFlattenSpec(world_count=0)


def test_gripper_openness_uses_one_leader_joint_per_side() -> None:
    q = np.asarray([[0.02, -0.01, 0.04, -0.03]], dtype=np.float32)
    lower = np.asarray([[0.0, -0.04, 0.0, -0.04]], dtype=np.float32)
    upper = np.asarray([[0.04, 0.0, 0.04, 0.0]], dtype=np.float32)

    openness = _gripper_openness(q, lower, upper)

    np.testing.assert_allclose(openness, [[0.5, 1.0]])


def test_vertex_normals_are_area_weighted_and_normalized() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    normals = _vertex_normals(vertices, triangles)

    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (4, 1)))


def test_nocs_matches_expert_data_encoding_and_reserves_black() -> None:
    vertices = np.asarray(
        [[-1.0, 2.0, 7.0], [0.0, 4.0, 7.0], [1.0, 6.0, 7.0]],
        dtype=np.float32,
    )

    colors = rest_vertices_to_nocs(vertices)

    np.testing.assert_array_equal(
        np.rint(colors * 255.0).astype(np.uint8),
        [[1, 1, 1], [128, 128, 1], [255, 255, 1]],
    )


def test_nocs_visual_mesh_uses_face_varying_public_texture_atlas() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    visual, indices, normals, uvs, texture = _nocs_visual_mesh(
        vertices,
        triangles,
        vertices,
    )

    np.testing.assert_array_equal(visual, vertices[triangles.reshape(-1)])
    np.testing.assert_array_equal(indices, np.arange(6, dtype=np.int32))
    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (6, 1)))
    assert uvs.shape == (6, 2)
    assert texture.dtype == np.uint8
    assert texture.shape == (8, 16, 4)
    expected = np.rint(rest_vertices_to_nocs(vertices)[triangles.reshape(-1)] * 255).astype(
        np.uint8
    )
    pixel_x = np.floor(uvs[:, 0] * texture.shape[1]).astype(int)
    pixel_y = np.floor((1.0 - uvs[:, 1]) * texture.shape[0]).astype(int)
    np.testing.assert_array_equal(texture[pixel_y, pixel_x, :3], expected)
