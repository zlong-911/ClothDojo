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
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
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


def material(stage, name, color, *, roughness=None, metallic=None):
    name = name.replace("-", "_")
    result = UsdShade.Material.Define(stage, f"/WristCameraAssembly/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, f"{result.GetPath()}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if roughness is not None:
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if metallic is not None:
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def ur5_black_material(stage):
    """Copy the existing arm material, not a visually guessed black swatch."""
    source = Usd.Stage.Open(str(ROOT / "payloads/workstation.usdc"))
    path = "/Robot/Materials/SourceMJCF/Material_source_00_black"
    target = "/WristCameraAssembly/Materials/UR5Black"
    UsdGeom.Scope.Define(stage, "/WristCameraAssembly/Materials")
    if not source.GetPrimAtPath(path) or not Sdf.CopySpec(
        source.GetRootLayer(), path, stage.GetRootLayer(), target
    ):
        raise ValueError("fixed UR5 payload must provide its original black material")
    return UsdShade.Material(stage.GetPrimAtPath(target))


def d415_components(source):
    """Find original CAD pieces without changing a single output vertex/face.

    STL repeats vertices and has non-manifold overlaps. Vertex connectivity,
    rather than manifold face adjacency, recovers its 19 original pieces.
    Component IDs are tied to the adjacent fixed ROS source, not arbitrary CADs.
    """
    welded = source.copy()
    welded.merge_vertices()
    edges = welded.edges
    count, labels = connected_components(
        coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(len(welded.vertices), len(welded.vertices)),
        ),
        directed=False,
    )
    face_labels = labels[welded.faces[:, 0]]
    if count != 19 or len(source.faces) != 425160 or np.count_nonzero(face_labels == 1) != 1592:
        raise ValueError("D415 appearance requires the audited 19-piece ROS STL")
    cover = source.triangles[face_labels == 1].reshape(-1, 3)
    bounds = np.stack((cover.min(0), cover.max(0)))
    if not np.allclose(bounds, [[-.0459, -.0079, -.000675], [.0459, .0079, -.0001]], atol=2e-6):
        raise ValueError("D415 cover geometry differs from the audited source")
    return face_labels, bounds


def bake_d415_front(source, labels, bounds, path):
    """Bake the source cover's rear aperture mask for opaque renderers.

    No camera photo, invented aperture, or changed geometry is used. This is
    a project-authored material approximation, not measured optical appearance.
    Newton does not import PreviewSurface opacity, so a transparent cover alone
    would hide every lens. Orthographic painter projection uses the original
    rear-mask triangles; its three circular and one projector aperture stay
    exactly where the original CAD put them. Detailed optics are not modeled.
    """
    size = (1536, 264)
    panel_rgb = (28, 30, 33)
    image = Image.new("RGB", size, (56, 63, 69))
    draw = ImageDraw.Draw(image)
    selected = (labels == 1) & np.all(
        np.isclose(source.triangles[:, :, 2], -.000675, atol=1e-9), axis=1
    ) & (source.face_normals[:, 2] < -.99)
    triangles = source.triangles[selected]
    if len(triangles) != 308:
        raise ValueError("D415 rear aperture mask differs from the audited source")
    low, high = bounds[:, :2]
    uv = (triangles[:, :, :2] - low) / (high - low)
    pixels = uv * np.asarray((size[0] - 1, -(size[1] - 1))) + (0, size[1] - 1)
    for triangle in pixels:
        draw.polygon([tuple(point) for point in triangle], fill=panel_rgb)
    image.save(path)


def d415_appearance(stage, visual, source, aluminum):
    labels, bounds = d415_components(source)
    bake_d415_front(source, labels, bounds, ROOT / "components/d415_front.png")
    materials = {
        "Housing": material(stage, "D415Aluminum", aluminum, roughness=.35, metallic=.7),
        "Interior": material(stage, "D415Interior", (.012, .013, .015), roughness=.45),
        "FrontCover": material(stage, "D415FrontCover", (.012, .013, .015), roughness=.18),
    }
    front = materials["FrontCover"]
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{front.GetPath()}/PreviewSurface"))
    texture = UsdShade.Shader.Define(stage, f"{front.GetPath()}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./d415_front.png"))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    reader = UsdShade.Shader.Define(stage, f"{front.GetPath()}/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    shader.GetInput("diffuseColor").ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    uv = (source.vertices[:, :2] - bounds[0, :2]) / (bounds[1, :2] - bounds[0, :2])
    UsdGeom.PrimvarsAPI(visual).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    groups = {
        "Housing": np.isin(labels, (0, 11, 18)),
        "FrontCover": labels == 1,
    }
    groups["Interior"] = ~np.logical_or.reduce(list(groups.values()))
    for name, mask in groups.items():
        subset = UsdGeom.Subset.CreateGeomSubset(
            visual, name, UsdGeom.Tokens.face, np.flatnonzero(mask).tolist(),
            "materialBind", UsdGeom.Tokens.partition,
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(materials[name])


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
        "appearance": (
            "User-approved UR5 black adapter and source-geometry D415 front albedo bake; "
            "not OEM textures"
        ),
    }

    # Preserve the DAE's exact mesh and authored corner normals. The user chose
    # the existing UR5 black material for the whole adapter instead of CAD colors.
    dae = ET.parse(SOURCE / "picknik_ur5_realsense_camera_adapter_rev2.dae")
    geometry = dae.find(".//c:geometry", NS)
    arrays = {}
    for item in geometry.findall("c:mesh/c:source", NS):
        values = item.find("c:float_array", NS)
        arrays[item.get("id")] = np.fromstring(values.text, sep=" ").reshape(-1, 3)
    source_points = arrays["shape0-mesh-positions"]
    source_normals = arrays["shape0-mesh-normals"]
    faces, normals = [], []
    for item in geometry.findall("c:mesh/c:polylist", NS):
        counts = np.fromstring(item.findtext("c:vcount", namespaces=NS), sep=" ", dtype=int)
        if not np.all(counts == 3):
            raise ValueError("source adapter polygons must be triangles")
        packed = np.fromstring(item.findtext("c:p", namespaces=NS), sep=" ", dtype=int).reshape(
            -1, 2
        )
        faces.extend(packed[:, 0].reshape(-1, 3))
        normals.extend(source_normals[packed[:, 1]])
    adapter = mesh(stage, "/WristCameraAssembly/AdapterVisual", source_points, faces)
    adapter.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32)))
    adapter.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    UsdShade.MaterialBindingAPI.Apply(adapter.GetPrim()).Bind(ur5_black_material(stage))
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
    d415_appearance(stage, visual, d415, aluminum)

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
