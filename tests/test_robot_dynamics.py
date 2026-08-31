"""Source-backed USD dynamics regressions; no GPU or stepping required."""
import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.robots import ALOHA2, robot_profile
from clothdojo.sim.scene import _add_robot, _joint_dofs, _joint_indices, _register_solver_attributes


def test_exported_usd_joint_dynamics_remain_in_si_units() -> None:
    import newton
    import warp as wp
    from pxr import Usd

    root = Path(__file__).parents[1] / "assets" / "robots" / "aloha2"
    stage = Usd.Stage.Open(str(root / "aloha2.usda"))
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(
        newton,
        wp,
        builder,
        root / "aloha2.usda",
        robot_profile(ALOHA2),
        load_visual_shapes=False,
    )

    builder_attributes = {
        "newton:damping": "joint_damping",
        "newton:limitStiffness": "joint_limit_ke",
        "newton:limitDamping": "joint_limit_kd",
        "newton:velocityLimit": "joint_velocity_limit",
    }
    joint_by_label = {str(label): index for index, label in enumerate(builder.joint_label)}
    checked = 0
    for prim in stage.Traverse():
        label = prim.GetAttribute("clothdojo:jointLabel")
        if not label or not label.HasAuthoredValue() or str(label.Get()) not in joint_by_label:
            continue
        joint = joint_by_label[str(label.Get())]
        if sum(builder.joint_dof_dim[joint]) != 1:
            continue
        dof = int(builder.joint_qd_start[joint])
        for usd_name, builder_name in builder_attributes.items():
            attribute = prim.GetAttribute(usd_name)
            if attribute and attribute.HasAuthoredValue() and np.isfinite(attribute.Get()):
                assert getattr(builder, builder_name)[dof] == pytest.approx(attribute.Get())
                checked += 1
    assert checked == 64

    profile = robot_profile(ALOHA2)
    arm = np.asarray(profile.arm_dof_indices)
    # Only the two load-limited passive terms are tuned; no effort is added.
    np.testing.assert_allclose(
        np.asarray(builder.joint_damping)[arm], [5.76, 20, 10, 2, 6.28, 1.2] * 2,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(builder.joint_effort_limit)[arm], [35, 144, 59, 22, 35, 35] * 2,
    )


def test_aloha_usd_preserves_passive_finger_equalities() -> None:
    import newton
    import warp as wp

    root = Path(__file__).parents[1] / "assets" / "robots" / "aloha2"
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(
        newton, wp, builder, root / "aloha2.usda", robot_profile(ALOHA2),
        load_visual_shapes=False,
    )
    leaders = _joint_indices(builder, ("left/left_finger", "right/left_finger"))
    followers = _joint_indices(builder, ("left/right_finger", "right/right_finger"))
    assert builder.constraint_mimic_joint0 == followers
    assert builder.constraint_mimic_joint1 == leaders
    assert builder.constraint_mimic_coef0 == [0.0, 0.0]
    assert builder.constraint_mimic_coef1 == [1.0, 1.0]
    assert builder.constraint_mimic_enabled == [True, True]
    # Do not replace the mechanical coupling with a second independent drive.
    np.testing.assert_allclose(
        np.asarray(builder.joint_target_ke)[_joint_dofs(builder, leaders)], 2000.0
    )
    np.testing.assert_array_equal(
        np.asarray(builder.joint_target_ke)[_joint_dofs(builder, followers)], 0.0
    )


def test_x5_import_preserves_same_usd_upstream_actuator_baseline() -> None:
    import newton
    import warp as wp

    profile = robot_profile("x5")
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(
        newton, wp, builder, Path(__file__).parents[1] / "assets" / profile.asset,
        profile, load_visual_shapes=False,
    )
    dofs = [
        builder.joint_qd_start[builder.joint_q_start.index(coord)]
        for coord in profile.arm_coord_indices
    ]
    assert len(dofs) == 12
    # RoboDojo x5.py loads the same ARX.usd. Effort is a simulation cap,
    # and SolverMuJoCo does not enforce the imported velocity-limit field.
    for field, value in (
        ("joint_target_ke", 4400.0), ("joint_target_kd", 40.0),
        ("joint_armature", 0.01), ("joint_effort_limit", 100.0),
        ("joint_velocity_limit", 5.0),
    ):
        np.testing.assert_allclose(np.asarray(getattr(builder, field))[dofs], value, rtol=1e-6)


def test_ur5_import_uses_tuned_drives_and_preserves_passive_dynamics_and_inertia() -> None:
    import newton
    import warp as wp

    profile = robot_profile("ur5e_2f85")
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(
        newton, wp, builder, Path(__file__).parents[1] / "assets" / profile.asset,
        profile, load_visual_shapes=False,
    )
    springs = builder.custom_attributes["mujoco:dof_passive_stiffness"].values
    refs = builder.custom_attributes["mujoco:dof_springref"].values
    assert set(springs) == set(refs) == {8, 12, 28, 32}
    np.testing.assert_allclose([v.value for v in springs.values()], 0.05)
    np.testing.assert_allclose([v.value for v in refs.values()], 2.62)
    np.testing.assert_allclose(np.asarray(builder.joint_damping)[list(springs)], 0.00125)
    # Fixed Menagerie MJCF, compiled with MuJoCo 3.11: inferred from the
    # original mesh/default density, not a manufacturer measurement.
    mounts = [
        i for i, label in enumerate(builder.body_label)
        if label.endswith("/gripper_base_mount")
    ]
    assert len(mounts) == 2
    np.testing.assert_allclose(np.asarray(builder.body_mass)[mounts], 0.150002855, rtol=1e-6)
    arm = np.asarray(profile.arm_dof_indices)
    # USD owns this simulation response tuning; retarget applies no multiplier.
    np.testing.assert_allclose(
        np.asarray(builder.joint_target_kd)[arm], [120, 125, 95, 18, 16, 15] * 2,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(builder.joint_target_ke)[arm], [2000, 2000, 2000, 500, 500, 500] * 2,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(builder.joint_effort_limit)[arm], [150, 150, 150, 28, 28, 28] * 2,
    )


def test_piper_import_restores_legacy_velocity_baseline() -> None:
    import newton
    import warp as wp

    profile = robot_profile("piper")
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    _add_robot(
        newton, wp, builder, Path(__file__).parents[1] / "assets" / profile.asset,
        profile, load_visual_shapes=False,
    )
    dofs = [
        builder.joint_qd_start[builder.joint_q_start.index(coord)]
        for coord in profile.arm_coord_indices
    ]
    np.testing.assert_allclose(np.asarray(builder.joint_velocity_limit)[dofs], 3.0)


@pytest.fixture(scope="module")
def fr3_import():
    import newton

    root = Path(__file__).parents[1] / "assets/robots/fr3_duo"
    if not (root / "payloads/workstation.usdc").is_file():
        pytest.skip("FR3 payload is installed separately from Git")
    builder = newton.ModelBuilder()
    _register_solver_attributes(newton, builder)
    imported = builder.add_usd(
        str(root / "fr3_duo.usda"), floating=False, load_visual_shapes=False,
        schema_resolvers=[newton.usd.SchemaResolverNewton(), newton.usd.SchemaResolverPhysx()],
    )
    source = json.loads((root / "provenance/dynamics.json").read_text())
    return builder, imported, source


def test_fr3_full_body_local_inertias_match_verified_sources(fr3_import) -> None:
    builder, imported, source = fr3_import
    assert builder.body_count == 110
    assert len(source["restored_bodies"]) == 47
    for expected in source["restored_bodies"]:
        body = imported["path_body_map"][expected["path"]]
        assert builder.body_mass[body] == pytest.approx(expected["mass"], abs=3e-7)
        # Newton's float32 world/body transform roundtrip introduces sub-micron error.
        np.testing.assert_allclose(builder.body_com[body], expected["com"], atol=3e-7)
        # Full tensor detects inverse principal-axis rotation; comparing only
        # diagonal eigenvalues would miss the original FR3 conversion bug.
        np.testing.assert_allclose(
            np.asarray(builder.body_inertia[body]).reshape(3, 3), expected["tensor"],
            atol=3e-8, rtol=1e-5, err_msg=expected["path"],
        )


def test_fr3_joint_import_converts_degrees_and_preserves_source_limits(fr3_import) -> None:
    builder, imported, source = fr3_import
    assert len(source["arm_joints"]) == 14
    for expected in source["arm_joints"]:
        joint = imported["path_joint_map"][expected["path"]]
        dof = builder.joint_qd_start[joint]
        fields = {
            "joint_target_ke": expected["kp_Nm_per_rad"],
            "joint_target_kd": expected["kd_Nm_s_per_rad"],
            "joint_effort_limit": expected["limits"]["effort"],
            "joint_velocity_limit": expected["limits"]["velocity"],
            "joint_limit_lower": expected["limits"]["lower"],
            "joint_limit_upper": expected["limits"]["upper"],
            "joint_damping": expected["passive_damping_Nm_s_per_rad"],
            "joint_friction": expected["friction_Nm"],
        }
        for name, value in fields.items():
            assert getattr(builder, name)[dof] == pytest.approx(value, rel=1e-6)
