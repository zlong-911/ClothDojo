"""Freeze UR5e's accepted Robotiq model into FR3, retaining its VMKit mount.

Offline USD conversion only. The two binary components contain the gripper and
its installation; the original workstation payload and UR5e remain untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import newton
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent
COMPONENT = ROOT / "components/robotiq_2f85.usdc"
INSTALLATION = ROOT / "components/robotiq_installation.usdc"


def set_transform(prim, matrix):
    xform = UsdGeom.Xformable(prim)
    attr = prim.GetAttribute("xformOp:transform")
    if not attr:
        attr = xform.AddTransformOp().GetAttr()
    attr.Set(matrix)
    xform.SetXformOpOrder([UsdGeom.XformOp(attr)])


def remap_path(path, mapping):
    if any(path.HasPrefix(new) for new in mapping.values()):
        return path
    for old, new in sorted(mapping.items(), key=lambda item: -len(str(item[0]))):
        if path.HasPrefix(old):
            return path.ReplacePrefix(old, new)
    raise ValueError(f"unexpected external gripper relationship: {path}")


def build_component():
    source = Usd.Stage.Open(str(ROOT.parent / "ur5e_2f85/ur5e_2f85.usda"))
    flat = source.Flatten()
    COMPONENT.parent.mkdir(exist_ok=True)
    stage = Usd.Stage.CreateNew(str(COMPONENT))
    stage.SetDefaultPrim(stage.DefinePrim("/Robotiq", "Xform"))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().customLayerData = {
        "source": "UR5e accepted USD; Menagerie da76818e269b82289eba39808e2fb91d679d6994",
        "control": ("Two 2.5 Nm position drives, kp=50, kd=5; "
                    "unchanged UR5e closed loops and passive dynamics"),
    }
    mapping = {}
    source_bodies = []
    for index in range(8, 21):
        prim = source.GetPrimAtPath(f"/Robot/Bodies/body_{index:03d}")
        name = prim.GetAttribute("clothdojo:bodyLabel").Get().split("/")[-1]
        mapping[prim.GetPath()] = Sdf.Path(f"/Robotiq/Bodies/{name}")
        source_bodies.append(prim)
    for index in range(9, 23):
        mapping[Sdf.Path(f"/Robot/Joints/joint_{index:03d}")] = Sdf.Path(
            f"/Robotiq/Joints/joint_{index - 9:02d}"
        )
    mapping[Sdf.Path("/Robot/Materials/SourceMJCF")] = Sdf.Path("/Robotiq/Materials")
    stage.DefinePrim("/Robotiq/Bodies", "Scope")
    stage.DefinePrim("/Robotiq/Joints", "Scope")
    for old, new in mapping.items():
        if not Sdf.CopySpec(flat, old, stage.GetRootLayer(), new):
            raise RuntimeError(f"cannot copy {old}")
    base = stage.GetPrimAtPath("/Robotiq/Bodies/gripper_base")
    # FR3 keeps its own physical cameras and TCP; these are UR5e semantics.
    for child in list(base.GetChildren()):
        if child.GetName() != "Geometry":
            stage.RemovePrim(child.GetPath())
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(source_bodies[0]).GetInverse()
    for prim in source_bodies:
        set_transform(stage.GetPrimAtPath(mapping[prim.GetPath()]),
                      cache.GetLocalToWorldTransform(prim) * inverse)
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            if rel.GetTargets():
                rel.SetTargets([remap_path(path, mapping) for path in rel.GetTargets()])
        for attr in prim.GetAttributes():
            if attr.GetConnections():
                attr.SetConnections([remap_path(path, mapping) for path in attr.GetConnections()])
    stage.GetRootLayer().Save()


def baseline_stage():
    """Read current non-gripper overrides; never regenerate installation edits."""
    entry = Sdf.Layer.FindOrOpen(str(ROOT / "fr3_duo.usda"))
    layer = Sdf.Layer.CreateAnonymous()
    layer.TransferContent(entry)
    layer.subLayerPaths = [str((ROOT / path).resolve()) for path in entry.subLayerPaths
                           if Path(path).name != INSTALLATION.name]
    return Usd.Stage.Open(layer)


def build_installation():
    baseline = baseline_stage()
    layer = Sdf.Layer.CreateNew(str(INSTALLATION))
    # Compose against the existing asset in memory, but save only our overlay.
    session = Sdf.Layer.CreateAnonymous()
    session.subLayerPaths = [layer.identifier, baseline.GetRootLayer().identifier]
    stage = Usd.Stage.Open(session)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Robot"))
    stage.SetEditTarget(layer)
    cache = UsdGeom.XformCache()
    bodies = {p.GetAttribute("clothdojo:bodyLabel").Get(): p for p in baseline.Traverse()
              if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    # Menagerie closes along Y, the VMKit model along X. Same shaft/Z origin.
    alignment = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), -90))
    old_to_new = alignment.GetInverse()
    new_roots = []
    for side in ("left", "right"):
        old_base = bodies[f"{side}/robotiq_85_base_link"]
        old_base_path = old_base.GetPath()
        removed = {p.GetPath() for label, p in bodies.items()
                   if label and label.startswith(f"{side}/robotiq_85_")
                   and not label.endswith("base_adapter_link")}
        assert len(removed) == 9
        mount = stage.DefinePrim(f"/Robot/Duo/Robotiq/{side}", "Xform")
        mount.GetReferences().AddReference("./robotiq_2f85.usdc", "/Robotiq")
        parent_world = cache.GetLocalToWorldTransform(baseline.GetPrimAtPath("/Robot/Duo"))
        set_transform(mount, alignment * cache.GetLocalToWorldTransform(old_base)
                      * parent_world.GetInverse())
        new_base_path = mount.GetPath().AppendPath("Bodies/gripper_base")
        for path in removed:
            stage.OverridePrim(path).SetActive(False)
        for prim in baseline.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            ends = [prim.GetRelationship(f"physics:body{i}").GetTargets() for i in (0, 1)]
            if any(path in removed - {old_base_path} for paths in ends for path in paths):
                stage.OverridePrim(prim.GetPath()).SetActive(False)
                continue
            for end, paths in enumerate(ends):
                if paths != [old_base_path]:
                    continue
                joint = stage.GetPrimAtPath(prim.GetPath())
                joint.GetRelationship(f"physics:body{end}").SetTargets([new_base_path])
                pos = prim.GetAttribute(f"physics:localPos{end}").Get()
                rot = prim.GetAttribute(f"physics:localRot{end}").Get()
                frame = Gf.Matrix4d().SetRotate(Gf.Quatd(rot))
                frame.SetTranslateOnly(Gf.Vec3d(pos))
                frame = frame * old_to_new
                joint.GetAttribute(f"physics:localPos{end}").Set(Gf.Vec3f(frame.ExtractTranslation()))
                joint.GetAttribute(f"physics:localRot{end}").Set(Gf.Quatf(frame.ExtractRotationQuat()))
        for prim in Usd.PrimRange(mount):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.GetAttribute("clothdojo:bodyLabel").Set(f"{side}/{prim.GetName()}")
            if prim.IsA(UsdPhysics.Joint):
                child = prim.GetRelationship("physics:body1").GetTargets()[0].name
                prim.GetAttribute("clothdojo:jointLabel").Set(f"{side}/{child}/{prim.GetName()}")
        new_roots.append(mount.GetPath())

    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    imported = builder.add_usd(
        stage, floating=False, load_visual_shapes=False,
        schema_resolvers=[newton.usd.SchemaResolverNewton(), newton.usd.SchemaResolverPhysx()],
    )
    # Match the runtime's float32 FK, rather than accumulating the donor's
    # tiny CAD matrix shear through a second installation transform.
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    poses = state.body_q.numpy()
    cache = UsdGeom.XformCache()
    for path, index in imported["path_body_map"].items():
        if not str(path).startswith("/Robot/Duo/Robotiq/"):
            continue
        prim = stage.GetPrimAtPath(path)
        pose = poses[index]
        world = Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[6]), *map(float, pose[3:6])))
        world.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
        set_transform(prim, world * cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse())
    names = [None] * builder.body_count
    for path, index in imported["path_body_map"].items():
        names[index] = stage.GetPrimAtPath(path).GetAttribute("clothdojo:bodyLabel").Get()
    assert all(names) and len(set(names)) == len(names)
    parents = [-1] * builder.body_count
    for j, child in enumerate(builder.joint_child):
        if builder.joint_type[j] != newton.JointType.BALL:
            parents[child] = builder.joint_parent[j]
    assert all(parent < child for child, parent in enumerate(parents))
    arm_names = list(baseline.GetDefaultPrim().GetAttribute("clothdojo:armJointNames").Get())
    by_label = {stage.GetPrimAtPath(path).GetAttribute("clothdojo:jointLabel").Get(): index
                for path, index in imported["path_joint_map"].items()}
    arms = [by_label[name] for name in arm_names]
    drivers = [imported["path_joint_map"][str(root.AppendPath(f"Joints/joint_{index:02d}"))]
               for root in new_roots for index in (0, 6)]
    metadata = {
        "bodyNames": (Sdf.ValueTypeNames.StringArray, names),
        "bodyParents": (Sdf.ValueTypeNames.IntArray, parents),
        "armCoordIndices": (Sdf.ValueTypeNames.IntArray, [builder.joint_q_start[j] for j in arms]),
        "armDofIndices": (Sdf.ValueTypeNames.IntArray, [builder.joint_qd_start[j] for j in arms]),
        "gripperCoordIndices": (
            Sdf.ValueTypeNames.IntArray, [builder.joint_q_start[j] for j in drivers]),
        "gripperDofIndices": (
            Sdf.ValueTypeNames.IntArray, [builder.joint_qd_start[j] for j in drivers]),
        "homeJointQ": (Sdf.ValueTypeNames.DoubleArray, list(builder.joint_q)),
        "jointDofCount": (Sdf.ValueTypeNames.Int, builder.joint_dof_count),
        "gripperClosedPositions": (Sdf.ValueTypeNames.DoubleArray, [.8, .8]),
        "gripperOpenPositions": (Sdf.ValueTypeNames.DoubleArray, [0., 0.]),
    }
    root = stage.GetPrimAtPath("/Robot")
    for key, (kind, value) in metadata.items():
        root.CreateAttribute(f"clothdojo:{key}", kind, custom=True).Set(value)
    layer.customLayerData = {
        "source": "build_robotiq.py; fixed UR5e gripper, unchanged FR3 arm/mount/cameras/TCP",
        "alignment": "Menagerie base to legacy VMKit base: Rz(-90deg), no translation",
    }
    layer.Save()
    print({key: value for key, (_, value) in metadata.items() if key != "bodyNames"}, flush=True)


if __name__ == "__main__":
    build_component()
    build_installation()
