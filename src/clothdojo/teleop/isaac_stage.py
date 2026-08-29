from __future__ import annotations

from pathlib import Path

import numpy as np


class XRAnchorRuntime:
    _ANCHOR_PATH = "/XRAnchor"

    def __init__(
        self,
        xr_core,
        post_sync_event_type: int,
        initial_view_pose,
    ) -> None:
        self._xr_core = xr_core
        self._initial_view_pose = initial_view_pose
        self._post_sync_rebind_armed = False
        self._post_sync_subscription = xr_core.get_message_bus().create_subscription_to_pop_by_type(
            post_sync_event_type,
            self._on_post_sync,
            name="Newton initial XR view",
        )

    def arm_initial_rebind(self) -> None:
        """Apply the initial view after the next completed XR synchronization."""
        self._post_sync_rebind_armed = True

    def _on_post_sync(self, _event) -> None:
        if not self._post_sync_rebind_armed:
            return
        self._post_sync_rebind_armed = False
        self.rebind()
        print("  [VR] initial XR view restored after connection", flush=True)

    def rebind(self) -> None:
        """Restore the identity stage anchor and deterministic initial view."""
        self._xr_core.schedule_set_stage_anchor(self._ANCHOR_PATH)
        self._xr_core.schedule_set_camera(self._initial_view_pose)

    def update_flight(self, translation_delta, yaw_delta: float) -> None:
        """Move the XR origin by one camera-relative input increment."""
        self._xr_core.schedule_move_space_origin_relative_to_camera(
            *[float(value) for value in translation_delta]
        )
        if float(yaw_delta) != 0.0:
            self._xr_core.schedule_rotate_space_origin_relative_to_camera(
                yaw=float(yaw_delta),
                pitch=0.0,
            )

    def close(self) -> None:
        self._post_sync_subscription.unsubscribe()


def build_render_stage(runtime, plan, *, cloth_visual: str = "nocs") -> Path:
    """Author the visual-only stage consumed by Isaac RTX/OpenXR."""
    return _build_render_stage(
        runtime.spec,
        runtime.state_0.particle_q.numpy(),
        plan,
        cloth_visual=cloth_visual,
        robot_profile=runtime.scene.robot_profile,
    )


def build_replay_render_stage(
    initial_points,
    plan,
    *,
    robot_profile,
    cloth_visual: str = "normal",
) -> Path:
    """Author the same stage directly from a recorded trajectory state."""
    from clothdojo.sim.scene import ShirtFlattenSpec

    return _build_render_stage(
        ShirtFlattenSpec(),
        initial_points,
        plan,
        cloth_visual=cloth_visual,
        robot_profile=robot_profile,
    )


def _build_render_stage(
    spec,
    initial_points,
    plan,
    *,
    cloth_visual: str,
    robot_profile,
) -> Path:
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    from clothdojo.sim.cloth import load_garment_mesh

    if cloth_visual not in {"normal", "nocs"}:
        raise ValueError("cloth_visual must be 'normal' or 'nocs'")

    output_path = plan.session_root / "teleop_stage.usda"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    _author_robot(stage, plan.assets_root, robot_profile)

    table_material = _preview_material(stage, "/World/Looks/Table", (0.82, 0.84, 0.87), 0.38)
    table = _define_textured_cube(
        stage, "/World/Table", spec.table_half_extents, physical_uvs=True
    )
    _set_cube(table, spec.table_position, spec.table_half_extents)
    UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(table_material)

    points = np.asarray(initial_points, dtype=np.float32)
    rest_vertices, triangles, panel_vertices, panel_indices = load_garment_mesh(plan.asset_path)
    indices = np.asarray(triangles, dtype=np.int32).reshape(-1)
    cloth = UsdGeom.Mesh.Define(stage, "/World/Cloth")
    cloth.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    cloth.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (indices.size // 3)))
    cloth.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    cloth.CreateDoubleSidedAttr(True)
    cloth.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
    _write_st_primvar(cloth, panel_vertices[panel_indices])
    if cloth_visual == "nocs":
        if len(rest_vertices) != len(points):
            raise RuntimeError(
                "teleoperation NOCS requires one rest vertex per cloth particle"
            )
        _write_nocs_primvar(cloth, rest_vertices)
        cloth_material = _nocs_material(stage, "/World/Looks/ClothNocs")
    else:
        cloth_material = _preview_material(
            stage, "/World/Looks/Cloth", (0.13, 0.36, 0.78), 0.58
        )
    UsdShade.MaterialBindingAPI.Apply(cloth.GetPrim()).Bind(cloth_material)

    ground_material = _preview_material(stage, "/World/Looks/Ground", (0.025, 0.027, 0.032), 0.82)
    ground_extents = (3.5, 3.5, 0.05)
    ground = _define_textured_cube(stage, "/World/Ground", ground_extents, physical_uvs=True)
    _set_cube(ground, (0.0, 0.0, -0.05), ground_extents)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_material)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr(650.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Lighting/Key")
    key.CreateIntensityAttr(2200.0)
    key.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-38.0, 24.0, -28.0))
    stage.GetRootLayer().Save()
    return output_path


def _author_robot(stage, assets_root: Path, robot_profile) -> tuple[str, ...]:
    robot_root = stage.DefinePrim("/World/Robot", "Xform")
    robot_asset = str((assets_root / robot_profile.asset).resolve())
    robot_root.GetReferences().AddReference(robot_asset)
    physics = robot_root.GetVariantSet("Physics")
    if physics and "none" in physics.GetVariantNames():
        physics.SetVariantSelection("none")
    return (robot_root.GetPath().pathString,)


def _author_single_arm_mjcf_visuals(
    stage, mjcf_path: Path, robot_root, *, material_prefix: str
) -> None:
    """Author one pinned Menagerie arm while keeping project-owned installation."""

    import mujoco
    from pxr import UsdGeom

    if not mjcf_path.is_file():
        raise FileNotFoundError(f"missing Menagerie robot asset: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    body_paths: dict[int, str] = {}
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or "/" in name:
            raise RuntimeError(f"single-arm MJCF body name is invalid: {name!r}")
        parent_id = int(model.body_parentid[body_id])
        parent_path = (
            robot_root.GetPath().pathString
            if parent_id == 0
            else body_paths[parent_id]
        )
        path = f"{parent_path}/{name}"
        body = UsdGeom.Xform.Define(stage, path)
        _set_mujoco_pose(body, model.body_pos[body_id], model.body_quat[body_id])
        body_paths[body_id] = path

    materials = {}
    for material_id in range(model.nmat):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id
        ) or f"material_{material_id}"
        rgba = model.mat_rgba[material_id]
        materials[material_id] = _preview_material(
            stage,
            f"/World/Looks/{material_prefix}_{name}",
            _display_to_linear(tuple(float(value) for value in rgba[:3])),
            0.35,
        )
    fallback = _preview_material(
        stage,
        f"/World/Looks/{material_prefix}_default",
        _display_to_linear((0.6, 0.6, 0.6)),
        0.35,
    )
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 2:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise RuntimeError("Menagerie visual geometry must be a mesh")
        material_id = int(model.geom_matid[geom_id])
        _author_mujoco_mesh(
            stage,
            model,
            geom_id,
            f"{body_paths[int(model.geom_bodyid[geom_id])]}/visual_{geom_id}",
            materials.get(material_id, fallback),
        )


def _author_aloha2_mjcf_visuals(stage, mjcf_path: Path, robot_root) -> None:
    """Author the pinned Menagerie meshes without importing its workbench."""

    import mujoco
    from pxr import UsdGeom

    if not mjcf_path.is_file():
        raise FileNotFoundError(f"missing ALOHA 2 Menagerie asset: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if model.nbody != 21 or model.nq != 16:
        raise RuntimeError(
            "pinned ALOHA 2 MJCF topology changed: expected 20 bodies and 16 coordinates"
        )

    aloha_root = UsdGeom.Xform.Define(stage, f"{robot_root.GetPath()}/aloha")
    side_roots = {
        side: UsdGeom.Xform.Define(stage, f"{aloha_root.GetPath()}/{side}")
        for side in ("left", "right")
    }
    # Menagerie authors the plastic as display-space 0.15 gray. PreviewSurface
    # consumes linear color, so use the corresponding linear value instead of
    # making the robot appear middle gray under RTX color management.
    official_black_linear = (0.01960665,) * 3
    material = _preview_material(
        stage, "/World/Looks/Aloha2Black", official_black_linear, 0.35
    )
    body_paths: dict[int, str] = {}
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or "/" not in name:
            raise RuntimeError(f"ALOHA 2 MJCF body has no side-qualified name: {name!r}")
        side, body_name = name.split("/", 1)
        parent_id = int(model.body_parentid[body_id])
        parent_path = (
            side_roots[side].GetPath().pathString
            if parent_id == 0
            else body_paths[parent_id]
        )
        path = f"{parent_path}/{body_name}"
        body = UsdGeom.Xform.Define(stage, path)
        _set_mujoco_pose(
            body,
            model.body_pos[body_id],
            model.body_quat[body_id],
        )
        body_paths[body_id] = path

    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 2:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise RuntimeError("ALOHA 2 visual geometry must be a mesh")
        body_path = body_paths[int(model.geom_bodyid[geom_id])]
        _author_mujoco_mesh(
            stage,
            model,
            geom_id,
            f"{body_path}/visual_{geom_id}",
            material,
        )


def _author_aloha2_installation_visuals(stage, mjcf_path: Path, robot_root) -> None:
    """Author the pinned official frame on ClothDojo's existing workbench."""

    import mujoco
    from pxr import UsdGeom

    if not mjcf_path.is_file():
        raise FileNotFoundError(f"missing ALOHA 2 installation asset: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if model.nbody != 1 or model.ngeom != 31:
        raise RuntimeError("pinned ALOHA 2 installation must contain 31 world meshes")

    frame_root = UsdGeom.Xform.Define(stage, f"{robot_root.GetPath()}/frame")
    black = _preview_material(
        stage, "/World/Looks/Aloha2FrameBlack", _display_to_linear((0.15,) * 3), 0.42
    )
    metal = _preview_material(
        stage,
        "/World/Looks/Aloha2FrameMetal",
        _display_to_linear((0.517, 0.529, 0.537)),
        0.28,
    )
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 1:
            raise RuntimeError("ALOHA installation geometry must use visual group 1")
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise RuntimeError("ALOHA installation geometry must be a mesh")
        material_id = int(model.geom_matid[geom_id])
        material_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
            if material_id >= 0
            else None
        )
        _author_mujoco_mesh(
            stage,
            model,
            geom_id,
            f"{frame_root.GetPath()}/visual_{geom_id}",
            metal if material_name == "metal" else black,
        )


def _author_mujoco_mesh(stage, model, geom_id: int, path: str, material) -> None:
    import mujoco
    from pxr import UsdGeom, UsdShade, Vt

    mesh_id = int(model.geom_dataid[geom_id])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    points = np.asarray(
        model.mesh_vert[vertex_start : vertex_start + vertex_count],
        dtype=np.float32,
    )
    faces = np.asarray(
        model.mesh_face[face_start : face_start + face_count], dtype=np.int32
    )
    normal_start = int(model.mesh_normaladr[mesh_id])
    normal_count = int(model.mesh_normalnum[mesh_id])
    normals = np.asarray(
        model.mesh_normal[normal_start : normal_start + normal_count],
        dtype=np.float32,
    )
    face_normals = np.asarray(
        model.mesh_facenormal[face_start : face_start + face_count], dtype=np.int32
    )
    mesh_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
    ) or f"mesh_{mesh_id}"
    mesh = UsdGeom.Mesh.Define(stage, f"{path}_{mesh_name}")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * face_count))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh.CreateNormalsAttr(
        Vt.Vec3fArray.FromNumpy(normals[face_normals].reshape(-1, 3))
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    _set_mujoco_pose(mesh, model.geom_pos[geom_id], model.geom_quat[geom_id])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _display_to_linear(color: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in color
    )


def _set_mujoco_pose(xformable, position, quaternion_wxyz) -> None:
    from pxr import Gf, UsdGeom

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotateOnly(
        Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*quaternion[1:4]))
    )
    matrix.SetTranslateOnly(
        Gf.Vec3d(*(float(value) for value in position))
    )
    UsdGeom.Xformable(xformable).AddTransformOp().Set(matrix)


def configure_xr(
    app,
    stage,
    initial_view_position,
    initial_view_target,
) -> XRAnchorRuntime:
    import carb
    from omni.kit.xr.core import XRCore, XRCoreEventType
    from pxr import Gf, Usd, UsdGeom

    settings = carb.settings.get_settings()
    settings.set_string("/persistent/xr/profile/vr/system/display", "OpenXR")
    settings.set_string("/xr/profile/vr/tools/layout", "")
    settings.set_string("/xr/profile/vr/gui/layers", "")
    settings.set_bool("/xr/profile/vr/enableNavigationControls", False)
    settings.set_bool("/xr/profile/vr/controllers/visible", False)
    settings.set_bool("/xr/profile/vr/tooltips/visible", False)
    settings.set_string("/xr/profile/vr/renderQuality", "performance")
    settings.set_float("/xr/profile/vr/renderScale", 0.5)
    settings.set_bool("/rtx/post/dlss/enabled", False)
    settings.set_int("/rtx/post/dlss/execMode", 0)
    settings.set_int("/rtx/post/aa/op", 3)
    settings.set_bool("/rtx/reflections/enabled", False)
    settings.set_bool("/rtx/ambientOcclusion/enabled", False)
    settings.set_int("/rtx/directLighting/sampledLighting/autoNumberOfRays", 1)
    settings.set_int("/rtx/directLighting/samplesPerPixel", 1)
    settings.set_bool("/rtx/indirectDiffuse/enabled", False)
    settings.set_bool("/rtx/translucency/enabled", False)
    settings.set_bool("/rtx/raytracing/fractionalCutoutOpacity", False)
    settings.set_bool("/rtx/post/motionBlur/enabled", False)
    settings.set_bool("/rtx/post/dof/enabled", False)
    settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)

    xr = XRCore.get_singleton()
    settings.set_float(
        "/persistent/simulation/defaultMetersPerUnit",
        UsdGeom.GetStageMetersPerUnit(stage),
    )
    UsdGeom.Xform.Define(stage, XRAnchorRuntime._ANCHOR_PATH)
    usd_view_pose = (
        Gf.Matrix4d()
        .SetLookAt(
            Gf.Vec3d(*[float(value) for value in initial_view_position]),
            Gf.Vec3d(*[float(value) for value in initial_view_target]),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        .GetInverse()
    )
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        initial_view = UsdGeom.Xform.Define(stage, "/_xr/initialView")
        initial_view.AddTransformOp().Set(usd_view_pose)
    settings.set_bool("/xr/profile/vr/enabled", True)
    for _ in range(30):
        app.update()
    if xr.is_xr_enabled():
        anchor_runtime = XRAnchorRuntime(
            xr,
            XRCoreEventType.post_sync_update,
            xr.get_world_transform_matrix(initial_view.GetPrim()),
        )
    else:
        raise RuntimeError("OpenXR did not become active; check the active runtime and headset")
    return anchor_runtime


def _set_cube(cube, position, half_extents) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
    xf.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in half_extents]))


def _define_textured_cube(stage, path, half_extents, *, physical_uvs):
    """Create a closed textured box with normalized or meter-scale UVs."""
    from pxr import Gf, Sdf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-1.0, -1.0, -1.0),
            Gf.Vec3f(1.0, -1.0, -1.0),
            Gf.Vec3f(-1.0, 1.0, -1.0),
            Gf.Vec3f(1.0, 1.0, -1.0),
            Gf.Vec3f(-1.0, -1.0, 1.0),
            Gf.Vec3f(1.0, -1.0, 1.0),
            Gf.Vec3f(-1.0, 1.0, 1.0),
            Gf.Vec3f(1.0, 1.0, 1.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [
            4,
            5,
            7,
            6,
            1,
            0,
            2,
            3,
            2,
            6,
            7,
            3,
            0,
            1,
            5,
            4,
            5,
            1,
            3,
            7,
            0,
            4,
            6,
            2,
        ]
    )
    mesh.CreateNormalsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 1.0),
            Gf.Vec3f(0.0, 0.0, -1.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
            Gf.Vec3f(0.0, -1.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(-1.0, 0.0, 0.0),
        ]
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    if physical_uvs:
        size_x, size_y, size_z = (2.0 * float(value) for value in half_extents)
    else:
        size_x = size_y = size_z = 1.0
    face_sizes = (
        (size_x, size_y),
        (size_x, size_y),
        (size_z, size_x),
        (size_x, size_z),
        (size_z, size_y),
        (size_z, size_y),
    )
    face_uvs = []
    for size_u, size_v in face_sizes:
        face_uvs.extend(
            [
                (0.0, 0.0),
                (size_u, 0.0),
                (size_u, size_v),
                (0.0, size_v),
            ]
        )
    st.Set(face_uvs)
    return mesh


def _preview_material(stage, path, color, roughness):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _write_nocs_primvar(cloth, rest_points) -> None:
    from pxr import Gf, Sdf, UsdGeom, Vt

    from clothdojo.sim.scene import rest_vertices_to_nocs

    colors = rest_vertices_to_nocs(rest_points)
    values = Vt.Vec3fArray(len(colors))
    for index, color in enumerate(colors):
        values[index] = Gf.Vec3f(*[float(value) for value in color])
    primvar = UsdGeom.PrimvarsAPI(cloth.GetPrim()).CreatePrimvar(
        "nocs",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex,
    )
    primvar.Set(values)


def _write_st_primvar(cloth, face_uvs) -> None:
    from pxr import Sdf, UsdGeom, Vt

    primvar = UsdGeom.PrimvarsAPI(cloth.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    primvar.Set(Vt.Vec2fArray.FromNumpy(face_uvs))


def _nocs_material(stage, path):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    reader = UsdShade.Shader.Define(stage, f"{path}/NocsReader")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("nocs")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    shader.GetInput("diffuseColor").ConnectToSource(
        reader.ConnectableAPI(),
        "result",
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material
