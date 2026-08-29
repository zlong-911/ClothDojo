from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _write_points(
    points: wp.array(dtype=wp.vec3f),
    particle_q: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    points[i] = particle_q[i]


class NewtonFabricBridge:
    """Project Newton state into Isaac's reliable cloth and rigid paths.

    Cloth points stay GPU-direct through USDRT. Isaac 5's RTX delegate does not
    reliably publish runtime local-matrix edits for the instanceable ALOHA visual
    meshes, so the small rigid topology is authored through standard USD xforms.
    """

    def __init__(self, model, usd_context, *, device: str) -> None:
        from pxr import Gf, UsdGeom
        from usdrt import Sdf, Usd

        if not str(device).startswith("cuda:"):
            raise ValueError("teleop Fabric requires a CUDA device")
        self._device = device
        self._Gf = Gf
        usd_stage = usd_context.get_stage()
        if usd_stage is None:
            raise RuntimeError("Isaac USD context has no stage")
        self._stage = Usd.Stage.Attach(usd_context.get_stage_id())
        body_paths = _find_render_body_paths(usd_stage, model.body_label)
        self._rigid_xform_ops = []
        for index, label in enumerate(model.body_label):
            path = body_paths.get(index)
            if path is None:
                raise RuntimeError(f"Isaac did not map Newton rigid body {label!r}")
            prim = usd_stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"Isaac rigid-body prim is invalid: {path}")
            self._rigid_xform_ops.append(UsdGeom.Xformable(prim).MakeMatrixXform())

        body_parent = [-1] * int(model.body_count)
        for child, parent in zip(
            model.joint_child.numpy().tolist(),
            model.joint_parent.numpy().tolist(),
            strict=True,
        ):
            body_parent[int(child)] = int(parent)
        self._body_parent = np.asarray(body_parent, dtype=np.int32)
        parent_world = getattr(model, "body_parent_world_q", None)
        if parent_world is None:
            parent_world = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)] * int(
                model.body_count
            )
        self._body_parent_world_q = np.asarray(parent_world, dtype=np.float64)
        # SelectPrims searches Fabric, not the source USD stage. Traversing the
        # attached USDRT stage is the documented population step.
        for _prim in self._stage.Traverse():
            pass

        self._cloth_attr = None
        self._cloth_points = None
        self._cloth_vt_points = None
        cloth = self._stage.GetPrimAtPath(Sdf.Path("/World/Cloth"))
        if cloth and cloth.IsValid():
            # OmniHydra only consumes runtime Fabric point edits from meshes
            # carrying this tag.  Without it RTX keeps rendering the USD
            # points authored when the stage was opened.
            if not cloth.HasAttribute("Deformable"):
                cloth.CreateAttribute(
                    "Deformable", Sdf.ValueTypeNames.PrimTypeTag, True
                )
            attr = cloth.GetAttribute("points")
            if attr and attr.IsValid():
                attr.SyncDataToGpu()
                vt_points = attr.Get()
                if not vt_points.HasFabricGpuData():
                    raise RuntimeError("Isaac Fabric cloth points have no GPU mirror")
                cuda_interface = vt_points.__cuda_array_interface__
                point_count = int(cuda_interface["shape"][0])
                pointer = int(cuda_interface["data"][0])
                if pointer == 0:
                    raise RuntimeError("Isaac Fabric cloth points have a null GPU pointer")
                self._cloth_points = wp.types.array(
                    dtype=wp.vec3f,
                    shape=(point_count,),
                    capacity=point_count * wp.types.type_size_in_bytes(wp.vec3f),
                    ptr=pointer,
                    device=self._device,
                    owner=False,
                    requires_grad=False,
                )
                # Keep the VtArray handle alive for the lifetime of the Warp
                # view into Fabric-owned GPU memory.
                self._cloth_vt_points = vt_points
                self._cloth_attr = attr
                if self._cloth_points.shape[0] != int(model.particle_count):
                    raise RuntimeError(
                        "Isaac Fabric cloth vertex count does not match Newton particles: "
                        f"{self._cloth_points.shape[0]} != {int(model.particle_count)}"
                    )
        if int(model.particle_count) > 0 and self._cloth_points is None:
            raise RuntimeError("Isaac Fabric did not map the combined cloth mesh")
        print(
            f"Isaac replay bridge ready: rigid={len(self._rigid_xform_ops)}, "
            f"cloth={self._cloth_points is not None}, sync=rigid-usd+cloth-gpu",
            flush=True,
        )

    def update(self, state) -> None:
        if self._cloth_points is not None:
            wp.launch(
                _write_points,
                dim=self._cloth_points.shape[0],
                inputs=[self._cloth_points, state.particle_q],
                device=self._device,
            )
            self._cloth_attr.InvalidateCpuData()

        local_poses = _world_to_local_transforms(
            state.body_q.numpy(),
            self._body_parent,
            self._body_parent_world_q,
        )
        for op, pose in zip(self._rigid_xform_ops, local_poses, strict=True):
            matrix = self._Gf.Matrix4d(1.0)
            matrix.SetRotateOnly(
                self._Gf.Quatd(
                    float(pose[6]),
                    self._Gf.Vec3d(*(float(value) for value in pose[3:6])),
                )
            )
            matrix.SetTranslateOnly(
                self._Gf.Vec3d(*(float(value) for value in pose[:3]))
            )
            op.Set(matrix)


def _world_to_local_transforms(
    body_q: np.ndarray,
    body_parent: np.ndarray,
    body_parent_world_q: np.ndarray,
) -> np.ndarray:
    """Convert Newton xyzw world poses to the robot's USD hierarchy."""

    world = np.asarray(body_q, dtype=np.float64)
    parents = np.asarray(body_parent, dtype=np.int32)
    parent_world = np.asarray(body_parent_world_q, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 7:
        raise ValueError("body_q must have shape [body, 7]")
    if parents.shape != (len(world),) or parent_world.shape != world.shape:
        raise ValueError("body hierarchy must match body_q")
    local = np.empty_like(world)
    for index, parent in enumerate(parents):
        reference = world[parent] if parent >= 0 else parent_world[index]
        inverse = _quat_inverse(reference[3:7])
        local[index, :3] = _quat_rotate(inverse, world[index, :3] - reference[:3])
        local[index, 3:7] = _quat_multiply(inverse, world[index, 3:7])
        norm = np.linalg.norm(local[index, 3:7])
        if not np.isfinite(local[index]).all() or norm <= 1.0e-12:
            raise ValueError("body_q contains an invalid rigid transform")
        local[index, 3:7] /= norm
    return local


def _quat_inverse(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm_squared = float(np.dot(value, value))
    if not np.isfinite(norm_squared) or norm_squared <= 1.0e-12:
        raise ValueError("body_q contains an invalid quaternion")
    return np.asarray((-value[0], -value[1], -value[2], value[3])) / norm_squared


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = np.asarray(quaternion[:3])
    value = np.asarray(vector, dtype=np.float64)
    doubled_cross = 2.0 * np.cross(xyz, value)
    return value + quaternion[3] * doubled_cross + np.cross(xyz, doubled_cross)


def _find_render_body_paths(stage, labels) -> dict[int, str]:
    """Match URDF body labels to the shallowest corresponding USD Xform."""
    from pxr import UsdGeom

    candidates: dict[tuple[str | None, str], list[str]] = {}
    rigid_objects: dict[str, str] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        path = prim.GetPath().pathString
        if path.startswith("/World/RigidObjects/") and path.count("/") == 3:
            rigid_objects[prim.GetName()] = path
            continue
        if not path.startswith("/World/Robot/"):
            continue
        parts = path.strip("/").split("/")
        side = "left" if "left" in parts else "right" if "right" in parts else None
        candidates.setdefault((side, prim.GetName()), []).append(path)

    result = {}
    for index, label in enumerate(labels):
        parts = str(label).strip("/").split("/")
        rigid_path = rigid_objects.get(parts[-1])
        if rigid_path is not None:
            result[index] = rigid_path
            continue
        side = "left" if "left" in parts else "right" if "right" in parts else None
        matches = candidates.get((side, parts[-1]), ())
        if matches:
            result[index] = min(matches, key=lambda value: value.count("/"))
    return result
