"""Load the converted garment contract into Newton's public Style3D builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def add_style3d_garment(
    builder,
    path: str | Path,
    spec,
    *,
    mesh: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[int, int]:
    """Add one triangulated, face-varying-UV garment and return its particle slice."""

    import warp as wp
    from newton.solvers import style3d

    vertices, indices, panel_vertices, panel_indices = mesh or load_garment_mesh(path)
    particle_start = int(builder.particle_count)
    style3d.add_cloth_mesh(
        builder,
        pos=wp.vec3(*spec.garment_position),
        rot=wp.quat(*spec.garment_rotation_xyzw),
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=vertices.tolist(),
        indices=indices.tolist(),
        panel_verts=panel_vertices.tolist(),
        panel_indices=panel_indices.tolist(),
        density=spec.cloth_density,
        scale=spec.garment_scale,
        particle_radius=spec.particle_radius,
        tri_aniso_ke=wp.vec3(*spec.stretch_stiffness_uv),
        edge_aniso_ke=wp.vec3(*spec.bend_stiffness_uv),
    )
    return particle_start, int(builder.particle_count)


def load_garment_mesh(
    path: str | Path,
    prim_path: str = "/root/Mesh",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the small mesh contract produced by ClothDojo's converter."""

    from pxr import Usd, UsdGeom

    path = Path(path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"failed to open garment: {path}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"garment must contain a mesh at {prim_path}: {path}")

    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"garment mesh has invalid points: {path}")
    if counts.ndim != 1 or not len(counts) or not np.all(counts == 3):
        raise ValueError(f"garment mesh must be triangulated: {path}")
    if indices.shape != (len(counts) * 3,):
        raise ValueError(f"garment topology is inconsistent: {path}")

    st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    values = st.Get() if st else None
    if values is None or st.GetInterpolation() != UsdGeom.Tokens.faceVarying:
        raise ValueError(f"garment must have face-varying st coordinates: {path}")
    panel_vertices = np.asarray(values, dtype=np.float32)
    authored_indices = st.GetIndices()
    panel_indices = (
        np.asarray(authored_indices, dtype=np.int32)
        if authored_indices
        else np.arange(len(indices), dtype=np.int32)
    )
    if panel_vertices.ndim != 2 or panel_vertices.shape[1] != 2:
        raise ValueError(f"garment st coordinates must be two-dimensional: {path}")
    if panel_indices.shape != indices.shape:
        raise ValueError(f"garment st topology is inconsistent: {path}")
    return vertices, indices, panel_vertices, panel_indices
