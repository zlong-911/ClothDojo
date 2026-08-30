"""Compose the fixed Kai0 workstation from the official RoboDojo X5 USD."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "payloads/workstation.usdc"
DEFAULT_OUTPUT = HERE / "payloads/workstation.next.usdc"
ROBO_DOJO_X5 = HERE / "robodojo/ARX.usd"
ROBO_DOJO_CAMERA_STAND = HERE / "robodojo/camera_stand.usdc"

# Center the old stand foot on the widest four-hole subset of Kai0's central
# six-hole pattern, with its lowest surface seated on the integrated plate.
CAMERA_STAND_POSITION = (0.261, 0.0, 0.81)
CAM_HIGH_TRANSFORM = Gf.Matrix4d(
    0.0,
    -1.0,
    0.0,
    0.0,
    0.7224967097223222,
    0.0,
    0.6913743591140898,
    0.0,
    -0.6913743591140898,
    0.0,
    0.7224967097223222,
    0.0,
    0.3179578738615954,
    -0.0225080347427568,
    1.34981,
    1.0,
)

ARM_ROOTS = {
    "left": (0.25, 0.28625912, 0.81),
    "right": (0.25, -0.28625912, 0.81),
}
BODY_ORDER = (
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "camera_base",
    "camera",
    "link7",
    "link8",
)
BODY_PARENTS_SINGLE = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 6, 6)
ARM_LIMITS_DEG = (
    (-179.90875244140625, 150.00035095214844),
    (-2.8647890090942383, 200.5352325439453),
    (-5.729578018188477, 183.34649658203125),
    (-91.67324829101562, 88.80845642089844),
    (-89.95437622070312, 89.95437622070312),
    (-114.59156036376953, 114.59156036376953),
)

# RoboDojo 76e16bf8feb6e5934ddb291386bf531ff826c621 uses these with this
# exact ARX.usd (see provenance/robodojo_x5.py). USD angular gains use
# torque/degree, velocity uses degree/s, and armature remains kg*m^2.
ARM_STIFFNESS_RAD = 4400.0
ARM_DAMPING_RAD = 40.0
ARM_ARMATURE = 0.01
ARM_VELOCITY_LIMIT_RAD = 5.0


def restore_source_joint_dynamics(stage: Usd.Stage) -> None:
    """Bake the two missing same-USD RoboDojo actuator fields, without reassembly."""
    for side in ARM_ROOTS:
        for joint_index in range(1, 7):
            joint = stage.GetPrimAtPath(f"/Robot/Arms/{side}/joints/joint{joint_index}")
            if not joint:
                raise RuntimeError(f"missing X5 joint: {side}/joint{joint_index}")
            joint.CreateAttribute(
                "physxJoint:armature", Sdf.ValueTypeNames.Float, custom=False
            ).Set(ARM_ARMATURE)
            joint.GetAttribute("physxJoint:maxJointVelocity").Set(
                math.degrees(ARM_VELOCITY_LIMIT_RAD)
            )


def _copy_source(source: Path, output: Path) -> Usd.Stage:
    layer = Sdf.Layer.FindOrOpen(str(source))
    if layer is None:
        raise RuntimeError(f"could not open source workstation layer: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not layer.Export(str(output)):
        raise RuntimeError(f"could not create workstation layer: {output}")
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"could not reopen workstation layer: {output}")
    return stage


def _set_custom(prim: Usd.Prim, name: str, type_name, value) -> None:
    prim.CreateAttribute(name, type_name, custom=True).Set(value)


def _author_camera(parent: Usd.Prim, name: str) -> None:
    camera = UsdGeom.Camera.Define(parent.GetStage(), parent.GetPath().AppendChild(name))
    camera.CreateFocalLengthAttr(1.93)
    camera.CreateHorizontalApertureAttr(3.896)
    camera.CreateVerticalApertureAttr(2.922)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(
        Gf.Matrix4d(
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0109,
            1.0,
        )
    )


def _material(stage: Usd.Stage, path: str, color: tuple[float, float, float]):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.52)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _author_arm(stage: Usd.Stage, side: str, position: tuple[float, float, float]) -> None:
    root = UsdGeom.Xform.Define(stage, f"/Robot/Arms/{side}")
    root.GetPrim().GetReferences().AddReference("../robodojo/ARX.usd")
    root.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))

    root_joint = UsdPhysics.FixedJoint.Define(stage, f"/Robot/Arms/{side}/joints/root_joint")
    root_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Robot/Arms/{side}/base_link")])
    root_joint.CreateLocalPos0Attr(Gf.Vec3f(*position))
    root_joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    root_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    root_joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    _set_custom(
        root_joint.GetPrim(),
        "clothdojo:jointLabel",
        Sdf.ValueTypeNames.String,
        f"{side}/fixed_base",
    )

    looks = {
        "white": _material(stage, f"/Robot/Arms/{side}/Looks/White", (0.72, 0.75, 0.78)),
        "camera": _material(stage, f"/Robot/Arms/{side}/Looks/Camera", (0.045, 0.05, 0.055)),
    }

    for body_name in BODY_ORDER:
        body = stage.GetPrimAtPath(f"/Robot/Arms/{side}/{body_name}")
        if not body:
            raise RuntimeError(f"official RoboDojo X5 is missing {body_name}")
        _set_custom(body, "clothdojo:bodyLabel", Sdf.ValueTypeNames.String, f"{side}/{body_name}")
        collisions = stage.GetPrimAtPath(f"{body.GetPath()}/collisions")
        if collisions:
            collisions.SetActive(False)
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in prim.GetPath().pathString:
                continue
            visual_only = body_name == "camera"
            _set_custom(
                prim, "clothdojo:shapeFlags", Sdf.ValueTypeNames.Int, 1 if visual_only else 7
            )
            _set_custom(prim, "clothdojo:collisionGroup", Sdf.ValueTypeNames.Int, 1)
            _set_custom(
                prim,
                "clothdojo:material",
                Sdf.ValueTypeNames.FloatArray,
                [1.0, 2500.0, 100.0, 1000.0],
            )
            _set_custom(prim, "clothdojo:contact", Sdf.ValueTypeNames.FloatArray, [0.0, 0.1])
            # Preserve the complete USD's authored dark metal material. Only
            # keep the existing white/camera fallbacks whose upstream bindings
            # are unresolved; never replace valid source materials by body name.
            if body_name in ("camera", "link3"):
                material_name = "camera" if body_name == "camera" else "white"
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(looks[material_name])
            if visual_only and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

    for joint_index in range(1, 9):
        joint = stage.GetPrimAtPath(f"/Robot/Arms/{side}/joints/joint{joint_index}")
        if not joint:
            raise RuntimeError(f"official RoboDojo X5 is missing joint{joint_index}")
        _set_custom(
            joint, "clothdojo:jointLabel", Sdf.ValueTypeNames.String, f"{side}/joint{joint_index}"
        )
        if joint_index <= 6:
            lower, upper = ARM_LIMITS_DEG[joint_index - 1]
            joint.GetAttribute("physics:lowerLimit").Set(lower)
            joint.GetAttribute("physics:upperLimit").Set(upper)
            joint.GetAttribute("drive:angular:physics:stiffness").Set(
                math.radians(ARM_STIFFNESS_RAD)
            )
            joint.GetAttribute("drive:angular:physics:damping").Set(math.radians(ARM_DAMPING_RAD))
            joint.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(
                0.0
            )
        else:
            joint.GetAttribute("drive:linear:physics:stiffness").Set(2300.0)
            joint.GetAttribute("drive:linear:physics:damping").Set(100.0)
            joint.CreateAttribute("state:linear:physics:position", Sdf.ValueTypeNames.Float).Set(
                0.044
            )

    link6 = stage.GetPrimAtPath(f"/Robot/Arms/{side}/link6")
    attach = UsdGeom.Xform.Define(stage, link6.GetPath().AppendChild("clothdojo_attach"))
    attach.AddTranslateOp().Set(Gf.Vec3d(0.152602, 0.0, -0.00014))
    attach.AddOrientOp().Set(Gf.Quatf(0.5, Gf.Vec3f(0.5, 0.5, 0.5)))
    _set_custom(
        attach.GetPrim(), "clothdojo:role", Sdf.ValueTypeNames.String, "end_effector_attach"
    )
    _set_custom(
        attach.GetPrim(), "clothdojo:parentBody", Sdf.ValueTypeNames.String, f"{side}/link6"
    )

    camera_body = stage.GetPrimAtPath(f"/Robot/Arms/{side}/camera")
    camera_name = f"cam_{side}_wrist"
    _author_camera(camera_body, camera_name)
    camera = stage.GetPrimAtPath(camera_body.GetPath().AppendChild(camera_name))
    _set_custom(camera, "clothdojo:parentBody", Sdf.ValueTypeNames.String, f"{side}/camera")


def _author_camera_stand(stage: Usd.Stage) -> None:
    stand = UsdGeom.Xform.Define(stage, "/Robot/Installation/CameraStand")
    stand.GetPrim().GetReferences().AddReference("../robodojo/camera_stand.usdc")
    stand.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_STAND_POSITION))
    _set_custom(
        stand.GetPrim(),
        "clothdojo:mountInference",
        Sdf.ValueTypeNames.String,
        "foot centered on Kai0 central outer four holes; 8.21 mm maximum hole-center residual",
    )

    camera = stage.GetPrimAtPath("/Robot/Semantics/cam_high")
    if not camera or not camera.IsA(UsdGeom.Camera):
        raise RuntimeError("source workstation is missing cam_high")
    camera.GetAttribute("xformOp:transform").Set(CAM_HIGH_TRANSFORM)
    _set_custom(camera, "clothdojo:cameraModel", Sdf.ValueTypeNames.String, "RealSense D455")
    _set_custom(
        camera,
        "clothdojo:visualMount",
        Sdf.ValueTypeNames.String,
        "/Robot/Installation/CameraStand/D455",
    )


def restore_dark_materials_and_match_base(stage: Usd.Stage) -> None:
    """Remove obsolete shading overrides; reuse the column material for Kai0."""
    for side in ARM_ROOTS:
        for body_name in BODY_ORDER:
            if body_name in ("link3", "camera"):
                continue
            body = stage.GetPrimAtPath(f"/Robot/Arms/{side}/{body_name}")
            for prim in Usd.PrimRange(body):
                if prim.IsA(UsdGeom.Mesh) and "/visuals/" in str(prim.GetPath()):
                    prim.RemoveProperty("material:binding")
                    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                    expected = f"/Robot/Arms/{side}/Looks/defaultmaterial"
                    if not material or str(material.GetPath()) != expected:
                        raise RuntimeError(f"unresolved upstream X5 material: {prim.GetPath()}")
        stage.RemovePrim(f"/Robot/Arms/{side}/Looks/Dark")
        stage.RemovePrim(f"/Robot/Arms/{side}/Looks/CameraMount")

    column = stage.GetPrimAtPath("/Robot/Installation/CameraStand/Column")
    material = UsdShade.MaterialBindingAPI(column).ComputeBoundMaterial()[0]
    if not material:
        raise RuntimeError("X5 column is missing its material")
    plate = stage.GetPrimAtPath("/Robot/Installation/Kai0BaseIntegrated")
    UsdShade.MaterialBindingAPI.Apply(plate).Bind(material)


def build(source: Path, output: Path) -> None:
    if not ROBO_DOJO_X5.is_file():
        raise FileNotFoundError(f"missing official RoboDojo X5 USD: {ROBO_DOJO_X5}")
    if not ROBO_DOJO_CAMERA_STAND.is_file():
        raise FileNotFoundError(f"missing RoboDojo camera stand USD: {ROBO_DOJO_CAMERA_STAND}")
    stage = _copy_source(source, output)
    for path in ("/Robot/Bodies", "/Robot/Joints", "/Robot/Installation/CameraStand"):
        stage.RemovePrim(path)
    stage.RemovePrim("/Robot/Arms")
    stage.RemovePrim("/Robot/Installation/Kai0CameraBottomBracket")

    root = stage.GetPrimAtPath("/Robot")
    _set_custom(
        root,
        "clothdojo:installationDescription",
        Sdf.ValueTypeNames.String,
        (
            "Kai0 integrated base with two RoboDojo ARX X5 robots and a directly "
            "mounted RoboDojo D455 stand"
        ),
    )
    _set_custom(
        root,
        "clothdojo:robotSource",
        Sdf.ValueTypeNames.String,
        "RoboDojo-Benchmark/RoboDojo Assets/Robots/x5/ARX.usd",
    )
    root.RemoveProperty("clothdojo:wristCameraVisual")

    body_names = [f"{side}/{name}" for side in ARM_ROOTS for name in BODY_ORDER]
    body_parents = list(BODY_PARENTS_SINGLE) + [
        -1 if parent < 0 else parent + len(BODY_ORDER) for parent in BODY_PARENTS_SINGLE
    ]
    _set_custom(root, "clothdojo:bodyNames", Sdf.ValueTypeNames.StringArray, body_names)
    _set_custom(root, "clothdojo:bodyParents", Sdf.ValueTypeNames.IntArray, body_parents)

    for side, position in ARM_ROOTS.items():
        _author_arm(stage, side, position)
    restore_source_joint_dynamics(stage)
    _author_camera_stand(stage)
    restore_dark_materials_and_match_base(stage)

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.GetPath().pathString.startswith(
            "/Robot/Installation/"
        ):
            continue
        _set_custom(prim, "clothdojo:shapeFlags", Sdf.ValueTypeNames.Int, 1)
        _set_custom(prim, "clothdojo:collisionGroup", Sdf.ValueTypeNames.Int, 1)
        _set_custom(
            prim, "clothdojo:material", Sdf.ValueTypeNames.FloatArray, [1.0, 2500.0, 100.0, 1000.0]
        )
        _set_custom(prim, "clothdojo:contact", Sdf.ValueTypeNames.FloatArray, [0.0, 0.1])

    stage.SetDefaultPrim(root)
    stage.GetRootLayer().customLayerData = {
        "installation": (
            "OpenDriveLab/kai0 integrated base with the inferred direct-mounted "
            "RoboDojo D455 camera stand"
        ),
        "robot": "two references to the complete RoboDojo ARX.usd",
        "thirdPersonCamera": (
            "RoboDojo camera_stand/00000 without its 90 x 300 mm base plate; "
            "foot centered on the Kai0 central outer four-hole subset"
        ),
        "wristCamera": (
            "RoboDojo camera_base and camera meshes; runtime sensor follows robot_config.yml"
        ),
    }
    stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
