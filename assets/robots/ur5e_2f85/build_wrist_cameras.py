"""Compose the published PickNik flush adapter and RealSense D415 assembly.

This is an offline representation conversion. All geometry and transforms come
from the adjacent source snapshot; no mount geometry or payload inertia is
designed here. The unchanged workstation payload anchors idempotent placement.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "provenance/wrist_camera"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = xyz
    return value


def author_transform(prim, matrix):
    xform = UsdGeom.Xformable(prim)
    attribute = prim.GetAttribute("xformOp:transform")
    if not attribute:
        attribute = xform.AddTransformOp().GetAttr()
    attribute.Set(Gf.Matrix4d(*np.asarray(matrix).T.reshape(-1).tolist()))
    xform.SetXformOpOrder([UsdGeom.XformOp(attribute)])


def material(stage, name, color):
    name = name.replace("-", "_")
    result = UsdShade.Material.Define(stage, f"/WristCameraAssembly/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, f"{result.GetPath()}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def shape_properties(prim, *, collision, label):
    prim.CreateAttribute("clothdojo:shapeFlags", Sdf.ValueTypeNames.Int).Set(6 if collision else 1)
    prim.CreateAttribute("clothdojo:shapeLabel", Sdf.ValueTypeNames.String).Set(label)
    prim.CreateAttribute("clothdojo:collisionGroup", Sdf.ValueTypeNames.Int).Set(1)
    prim.CreateAttribute("clothdojo:contact", Sdf.ValueTypeNames.FloatArray).Set([0, 0.1])
    prim.CreateAttribute("clothdojo:material", Sdf.ValueTypeNames.FloatArray).Set(
        [1, 2500, 100, 1000]
    )
    # Existing bodies have explicit mass properties. Do not derive accessory
    # inertia from mesh density: upstream explicitly lacks trustworthy values.
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0)
    UsdPhysics.MassAPI(prim).CreateDensityAttr(0)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    else:
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)


def mesh(stage, path, vertices, faces, *, collision=False):
    result = UsdGeom.Mesh.Define(stage, path)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    result.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    result.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    result.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    result.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    result.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([vertices.min(0), vertices.max(0)])))
    shape_properties(result.GetPrim(), collision=collision, label=path.rsplit("/", 1)[-1])
    if collision:
        UsdPhysics.MeshCollisionAPI.Apply(result.GetPrim()).CreateApproximationAttr("convexHull")
    return result


def build_component():
    target = ROOT / "components/wrist_camera.usdc"
    target.parent.mkdir(exist_ok=True)
    stage = Usd.Stage.CreateNew(str(target))
    root = UsdGeom.Xform.Define(stage, "/WristCameraAssembly")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.GetRootLayer().customLayerData = {
        "source": "../provenance/wrist_camera/README.md",
        "payloadInertia": "Not modeled: PickNik absent; RealSense tensor explicitly unreliable",
    }

    # Preserve the DAE's exact mesh, authored corner normals, and material split.
    dae = ET.parse(SOURCE / "picknik_ur5_realsense_camera_adapter_rev2.dae")
    geometry = dae.find(".//c:geometry", NS)
    arrays = {}
    for item in geometry.findall("c:mesh/c:source", NS):
        values = item.find("c:float_array", NS)
        arrays[item.get("id")] = np.fromstring(values.text, sep=" ").reshape(-1, 3)
    source_points = arrays["shape0-mesh-positions"]
    source_normals = arrays["shape0-mesh-normals"]
    faces, normals, subsets = [], [], {}
    for item in geometry.findall("c:mesh/c:polylist", NS):
        counts = np.fromstring(item.findtext("c:vcount", namespaces=NS), sep=" ", dtype=int)
        if not np.all(counts == 3):
            raise ValueError("source adapter polygons must be triangles")
        packed = np.fromstring(item.findtext("c:p", namespaces=NS), sep=" ", dtype=int).reshape(
            -1, 2
        )
        subsets[item.get("material")] = list(range(len(faces), len(faces) + len(counts)))
        faces.extend(packed[:, 0].reshape(-1, 3))
        normals.extend(source_normals[packed[:, 1]])
    adapter = mesh(stage, "/WristCameraAssembly/AdapterVisual", source_points, faces)
    adapter.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32)))
    adapter.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    for binding in dae.findall(".//c:library_materials/c:material", NS):
        effect_id = binding.find("c:instance_effect", NS).get("url").lstrip("#")
        effect = dae.find(f".//c:effect[@id='{effect_id}']", NS)
        color = np.fromstring(effect.findtext(".//c:diffuse/c:color", namespaces=NS), sep=" ")[:3]
        name = binding.get("id")
        subset = UsdGeom.Subset.CreateGeomSubset(
            adapter, name.replace("-", "_"), UsdGeom.Tokens.face, subsets[name]
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material(stage, name, color))
    collision = trimesh.load_mesh(
        SOURCE / "picknik_ur5_realsense_camera_adapter_rev2_collision.stl", process=False
    )
    mesh(
        stage,
        "/WristCameraAssembly/AdapterCollision",
        collision.vertices,
        collision.faces,
        collision=True,
    )

    # PickNik adapter -> D415 bottom screw -> camera body, from published Xacro.
    camera_link = transform(
        (0, -0.067, 0.0171), (0, -math.pi / 2 + math.pi / 30, math.pi / 2)
    ) @ transform((0, 0.02, 0.0115))
    camera = UsdGeom.Xform.Define(stage, "/WristCameraAssembly/D415")
    author_transform(camera.GetPrim(), camera_link)
    d415 = trimesh.load_mesh(SOURCE / "d415.stl", process=False)
    visual = mesh(stage, "/WristCameraAssembly/D415/Visual", d415.vertices, d415.faces)
    visual.CreateNormalsAttr(
        Vt.Vec3fArray.FromNumpy(np.asarray(d415.face_normals, dtype=np.float32))
    )
    visual.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    author_transform(
        visual.GetPrim(), transform((0.00987, -0.02, 0), (math.pi / 2, 0, math.pi / 2))
    )
    materials = ET.parse(SOURCE / "_materials.urdf.xacro")
    aluminum = np.fromstring(
        materials.find(".//material[@name='aluminum']/color").get("rgba"), sep=" "
    )[:3]
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(
        material(stage, "D415Aluminum", aluminum)
    )

    # Official simplified collision only; the visible camera remains the exact
    # 425,160-triangle mesh, including its authored front/lens geometry.
    collision = UsdGeom.Cube.Define(stage, "/WristCameraAssembly/D415/Collision")
    collision.CreateSizeAttr(1)
    matrix = transform((0, -0.02, 0))
    matrix[:3, :3] = np.diag([0.02005, 0.099, 0.023])
    author_transform(collision.GetPrim(), matrix)
    shape_properties(collision.GetPrim(), collision=True, label="D415Collision")
    optical = (
        camera_link @ transform((0, 0.015, 0)) @ transform(rpy=(-math.pi / 2, 0, -math.pi / 2))
    )
    usd_camera = optical @ transform(rpy=(math.pi, 0, 0))
    frame = UsdGeom.Xform.Define(stage, "/WristCameraAssembly/ColorOpticalUSD")
    author_transform(frame.GetPrim(), usd_camera)
    stage.GetRootLayer().Save()
    return usd_camera


def main():
    usd_camera = build_component()
    stage = Usd.Stage.Open(str(ROOT / "ur5e_2f85.usda"))
    baseline = Usd.Stage.Open(str(ROOT / "payloads/workstation.usdc"))
    cache = UsdGeom.XformCache()
    flange = transform((0, 0.1, 0), (-math.pi / 2, 0, 0))
    for side, wrist_id, base_id, joint_id in (("left", 6, 8, 7), ("right", 27, 29, 30)):
        wrist_path = f"/Robot/Bodies/body_{wrist_id:03}"
        wrist_world = np.asarray(
            cache.GetLocalToWorldTransform(baseline.GetPrimAtPath(wrist_path))
        ).T
        shift = (wrist_world @ flange)[:3, 2] * 0.007
        for prim in baseline.GetPrimAtPath("/Robot/Bodies").GetChildren():
            label = prim.GetAttribute("clothdojo:bodyLabel").Get()
            if label and label.startswith(f"{side}/gripper_"):
                matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
                matrix[:3, 3] += shift
                author_transform(stage.GetPrimAtPath(prim.GetPath()), matrix)
        joint_path = f"/Robot/Joints/joint_{joint_id:03}"
        original = baseline.GetPrimAtPath(joint_path).GetAttribute("physics:localPos0").Get()
        stage.GetPrimAtPath(joint_path).GetAttribute("physics:localPos0").Set(
            Gf.Vec3f(original[0], original[1] + 0.007, original[2])
        )
        assembly = UsdGeom.Xform.Define(stage, f"{wrist_path}/WristCameraAssembly")
        assembly.GetPrim().GetReferences().SetReferences(
            [Sdf.Reference("./components/wrist_camera.usdc")]
        )
        author_transform(assembly.GetPrim(), flange)
        parent_path = f"/Robot/Bodies/body_{base_id:03}"
        parent_world = np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))
        ).T
        camera = UsdGeom.Camera(stage.GetPrimAtPath(f"{parent_path}/cam_{side}_wrist"))
        author_transform(
            camera.GetPrim(), np.linalg.inv(parent_world) @ wrist_world @ flange @ usd_camera
        )
        # PickNik's authored horizontal FOV is 1.047 radians at 640x480. Retain
        # the existing 13-unit focal scale; the ratio is the published truth.
        camera.CreateFocalLengthAttr(13)
        sensor = ET.parse(SOURCE / "picknik_ur.xacro").find(
            ".//sensor[@name='wrist_mounted_camera_rgbd_sensor']/camera"
        )
        horizontal_fov = float(sensor.findtext("horizontal_fov"))
        horizontal = 26 * math.tan(horizontal_fov / 2)
        camera.CreateHorizontalApertureAttr(horizontal)
        camera.CreateVerticalApertureAttr(
            horizontal
            * float(sensor.findtext("image/height"))
            / float(sensor.findtext("image/width"))
        )
        camera.GetPrim().CreateAttribute("clothdojo:cameraSource", Sdf.ValueTypeNames.String).Set(
            "PickNik UR5e D415 nominal optical chain; source horizontal FOV 1.047rad at 640x480"
        )
        print(side, "gripper_world_shift_m", shift.tolist())
    metadata = dict(stage.GetRootLayer().customLayerData)
    metadata["wristCameraSource"] = "provenance/wrist_camera/README.md"
    metadata["dynamicsUnits"] = (
        "Newton USD revolute coefficients use degrees; source payload is legacy SI. "
        "1e6 rad/s is an effectively unlimited fallback, not a hardware rating. "
        "Robotiq spring values come from fixed Menagerie 2f85.xml."
    )
    stage.GetRootLayer().customLayerData = metadata
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    main()
