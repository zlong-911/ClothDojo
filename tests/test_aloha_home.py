"""The fixed ALOHA workstation opens both grippers without moving the arms."""

from pathlib import Path

import numpy as np
import pytest

from clothdojo.robots import robot_profile

ROOT = Path(__file__).parents[1] / "assets/robots/aloha2"
FINGERS = (8, 9, 18, 19)


def test_aloha_home_uses_open_endpoints_and_synchronized_joint_state():
    from pxr import Usd, UsdPhysics

    profile = robot_profile("aloha2")
    stage = Usd.Stage.Open(str(ROOT / "aloha2.usda"))
    source = Usd.Stage.Open(str(ROOT / "payloads/workstation.usdc"))
    home = np.asarray(profile.home_joint_q)
    original = np.asarray(source.GetDefaultPrim().GetAttribute("clothdojo:homeJointQ").Get())
    np.testing.assert_array_equal(
        home[list(profile.arm_coord_indices)], original[list(profile.arm_coord_indices)]
    )
    assert profile.gripper_open_positions == (0.037, 0.037)
    np.testing.assert_array_equal(
        home[list(profile.gripper_coord_indices)], profile.gripper_open_positions * 2
    )
    assert all(
        opened > closed
        for opened, closed in zip(
            profile.gripper_open_positions, profile.gripper_closed_positions, strict=True
        )
    )
    for joint_id in FINGERS:
        prim = stage.GetPrimAtPath(f"/Robot/Joints/joint_{joint_id:03}")
        joint = UsdPhysics.PrismaticJoint(prim)
        assert joint.GetLowerLimitAttr().Get() < 0.037 < joint.GetUpperLimitAttr().Get()
        assert prim.GetAttribute("state:linear:physics:position").Get() == pytest.approx(0.037)
        if joint_id in (8, 18):
            assert prim.GetAttribute("drive:linear:physics:targetPosition").Get() == pytest.approx(
                0.037
            )
        else:
            # Followers retain the source's single-drive / equality design.
            assert not prim.GetAttribute("drive:linear:physics:targetPosition").HasAuthoredValue()
            assert prim.GetAttribute("newton:mimicCoef1").Get() == 1.0


def test_aloha_direct_usd_import_and_default_transforms_match_open_home():
    import newton
    import warp as wp
    from pxr import Gf, Usd, UsdGeom

    profile = robot_profile("aloha2")
    stage = Usd.Stage.Open(str(ROOT / "aloha2.usda"))
    # Deliberately do not use _add_robot's profile override: a direct USD
    # consumer must also receive the authored open joint positions.
    with wp.ScopedDevice("cpu"):
        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            floating=False,
            load_visual_shapes=False,
            load_static_visual_shapes=False,
            skip_mesh_approximation=True,
            schema_resolvers=[newton.usd.SchemaResolverNewton(), newton.usd.SchemaResolverPhysx()],
        )
        np.testing.assert_allclose(builder.joint_q, profile.home_joint_q, atol=1e-7, rtol=0)
        model = builder.finalize(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    cache = UsdGeom.XformCache()
    for path, pose in zip(builder.body_label, state.body_q.numpy(), strict=True):
        expected = Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[6]), *map(float, pose[3:6])))
        expected.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)), expected,
            atol=2e-6, rtol=0, err_msg=str(path),
        )


def test_aloha_only_finger_default_transforms_move_from_source():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(ROOT / "aloha2.usda"))
    source = Usd.Stage.Open(str(ROOT / "payloads/workstation.usdc"))
    finger_paths = {f"/Robot/Bodies/body_{index:03}" for index in FINGERS}
    for old in source.Traverse():
        path = str(old.GetPath())
        if path in finger_paths or not old.IsA(UsdGeom.Xformable):
            continue
        # Includes arm bodies, camera local transforms, attach frames and
        # the complete installation. No camera or geometry authoring changes.
        new = stage.GetPrimAtPath(old.GetPath())
        np.testing.assert_array_equal(
            UsdGeom.Xformable(new).GetLocalTransformation(),
            UsdGeom.Xformable(old).GetLocalTransformation(),
            err_msg=path,
        )
        if old.IsA(UsdGeom.Camera):
            for attr in old.GetAttributes():
                assert new.GetAttribute(attr.GetName()).Get() == attr.Get()

    cache = UsdGeom.XformCache()
    for first, second in ((8, 9), (18, 19)):
        spans = []
        for value in (source, stage):
            origins = [
                np.asarray(cache.GetLocalToWorldTransform(
                    value.GetPrimAtPath(f"/Robot/Bodies/body_{index:03}")
                ))[3, :3]
                for index in (first, second)
            ]
            spans.append(np.linalg.norm(origins[1] - origins[0]))
        # Both fingers move outward by 0.037 - 0.0084 m.
        assert spans[1] - spans[0] == pytest.approx(2 * (0.037 - 0.0084), abs=2e-6)
