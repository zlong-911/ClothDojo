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

    Cloth points stay GPU-direct through USDRT on CUDA device 0. USDRT's Fabric
    GPU data is not multi-GPU safe, so other devices author the points through
    Fabric's CPU path. Isaac 5's RTX delegate does not reliably publish runtime
    local-matrix edits for the instanceable ALOHA visual meshes, so the small
    rigid topology is authored through standard USD xforms.
    """

    def __init__(self, model, usd_context, *, device: str) -> None:
        from usdrt import Sdf, Usd, Vt

        if not str(device).startswith("cuda:"):
            raise ValueError("teleop Fabric requires a CUDA device")
        self._device = device
        usd_stage = usd_context.get_stage()
        if usd_stage is None:
            raise RuntimeError("Isaac USD context has no stage")
        self._stage = Usd.Stage.Attach(usd_context.get_stage_id())
        self._rigid_writer = _UsdRigidBodyWriter(usd_stage, model.body_label)
        # SelectPrims searches Fabric, not the source USD stage. Traversing the
        # attached USDRT stage is the documented population step.
        for _prim in self._stage.Traverse():
            pass

        self._cloth_attr = None
        self._cloth_points = None
        self._cloth_vt_points = None
        self._cloth_vt_type = None
        self._cloth_point_count = 0
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
                vt_points = attr.Get()
                self._cloth_point_count = len(vt_points)
                self._cloth_attr = attr
                if self._cloth_point_count != int(model.particle_count):
                    raise RuntimeError(
                        "Isaac Fabric cloth vertex count does not match Newton particles: "
                        f"{self._cloth_point_count} != {int(model.particle_count)}"
                    )
                if _fabric_gpu_direct_supported(self._device):
                    attr.SyncDataToGpu()
                    vt_points = attr.Get()
                    if not vt_points.HasFabricGpuData():
                        raise RuntimeError("Isaac Fabric cloth points have no GPU mirror")
                    cuda_interface = vt_points.__cuda_array_interface__
                    pointer = int(cuda_interface["data"][0])
                    if pointer == 0:
                        raise RuntimeError("Isaac Fabric cloth points have a null GPU pointer")
                    self._cloth_points = wp.array(
                        dtype=wp.vec3f,
                        shape=(self._cloth_point_count,),
                        capacity=(
                            self._cloth_point_count
                            * wp.types.type_size_in_bytes(wp.vec3f)
                        ),
                        ptr=pointer,
                        device=self._device,
                        requires_grad=False,
                    )
                    # Keep the VtArray handle alive for the lifetime of the Warp
                    # view into Fabric-owned GPU memory.
                    self._cloth_vt_points = vt_points
                else:
                    self._cloth_vt_type = Vt.Vec3fArray
        if int(model.particle_count) > 0 and self._cloth_attr is None:
            raise RuntimeError("Isaac Fabric did not map the combined cloth mesh")
        sync = (
            "rigid-usd+cloth-gpu"
            if self._cloth_points is not None
            else "rigid-usd+cloth-fabric-cpu"
        )
        print(
            f"Isaac replay bridge ready: rigid={int(model.body_count)}, "
            f"cloth={self._cloth_attr is not None}, sync={sync}",
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
        elif self._cloth_attr is not None:
            # Fabric GPU arrays currently live on CUDA device 0 even when Isaac
            # physics runs on another device. Writing that pointer from cuda:N
            # causes an asynchronous illegal-memory-access failure. Set the CPU
            # value instead; OmniHydra observes it because the prim is Deformable.
            points = np.asarray(state.particle_q.numpy(), dtype=np.float32)
            if points.shape != (self._cloth_point_count, 3):
                raise ValueError("particle_q must have shape [particle, 3]")
            if not self._cloth_attr.Set(self._cloth_vt_type(points)):
                raise RuntimeError("Isaac Fabric rejected the cloth point update")

        self._rigid_writer.update(state.body_q.numpy())


def _fabric_gpu_direct_supported(device: str) -> bool:
    """USDRT 7.x exposes Fabric GPU arrays only on physical CUDA device 0."""
    return str(device) == "cuda:0"


class _UsdRigidBodyWriter:
    """Write Newton world poses through the actual USD transform hierarchy.

    USD parents may be installation groups or siblings of the physical joint
    parent. Keep authored body stretch and descendant transforms, including
    visual meshes and cameras, while changing only the body's rigid world pose.
    """

    def __init__(self, stage, labels) -> None:
        from pxr import Gf, UsdGeom

        self._Gf = Gf
        self._cache = UsdGeom.XformCache()
        paths = _find_render_body_paths(stage, labels)
        missing = [str(label) for index, label in enumerate(labels) if index not in paths]
        if missing:
            raise RuntimeError(f"Isaac did not map Newton rigid bodies: {missing}")
        if len(set(paths.values())) != len(paths):
            raise RuntimeError("Isaac mapped multiple Newton bodies to the same USD prim")
        self._bodies = []
        # Parents must be updated first even when the Newton body order differs.
        for index, path in sorted(paths.items(), key=lambda item: item[1].count("/")):
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            world = self._cache.GetLocalToWorldTransform(prim)
            rigid = Gf.Matrix4d(1.0)
            rigid.SetRotateOnly(Gf.Transform(world).GetRotation())
            rigid.SetTranslateOnly(world.ExtractTranslation())
            stretch = world * rigid.GetInverse()
            local = xform.GetLocalTransformation()
            reset = xform.GetResetXformStack()
            op = xform.MakeMatrixXform()
            op.Set(local)
            xform.SetResetXformStack(reset)
            self._bodies.append((index, prim, op, stretch, reset))

    def update(self, body_q: np.ndarray) -> None:
        world = np.asarray(body_q, dtype=np.float64)
        if world.shape != (len(self._bodies), 7):
            raise ValueError("body_q must have shape [body, 7]")
        norms = np.linalg.norm(world[:, 3:7], axis=1)
        if not np.isfinite(world).all() or np.any(norms <= 1.0e-12):
            raise ValueError("body_q contains an invalid rigid transform")
        self._cache.Clear()
        for index, prim, op, stretch, reset in self._bodies:
            pose = world[index]
            quaternion = pose[3:7] / norms[index]
            rigid = self._Gf.Matrix4d(1.0)
            rigid.SetRotateOnly(
                self._Gf.Quatd(float(quaternion[3]), self._Gf.Vec3d(*quaternion[:3]))
            )
            rigid.SetTranslateOnly(self._Gf.Vec3d(*pose[:3]))
            desired = stretch * rigid
            parent = (
                self._Gf.Matrix4d(1.0)
                if reset else self._cache.GetParentToWorldTransform(prim)
            )
            op.Set(desired * parent.GetInverse())
            # XformCache does not observe authored changes. Descendants must see
            # their parent's just-written world transform, not the previous frame.
            self._cache.Clear()


def _find_render_body_paths(stage, labels) -> dict[int, str]:
    """Use workstation body labels, retaining name lookup for untagged Piper."""
    from pxr import UsdGeom

    candidates: dict[tuple[str | None, str], list[str]] = {}
    rigid_objects: dict[str, str] = {}
    authored_labels: dict[str, str] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        path = prim.GetPath().pathString
        if path.startswith("/World/RigidObjects/") and path.count("/") == 3:
            rigid_objects[prim.GetName()] = path
            continue
        if not path.startswith("/World/Robot/"):
            continue
        label = prim.GetAttribute("clothdojo:bodyLabel")
        if label and label.HasAuthoredValue():
            name = str(label.Get()).strip("/")
            if name in authored_labels:
                raise RuntimeError(f"duplicate Isaac rigid-body label {name!r}")
            authored_labels[name] = path
        parts = path.strip("/").split("/")
        side = "left" if "left" in parts else "right" if "right" in parts else None
        candidates.setdefault((side, prim.GetName()), []).append(path)

    result = {}
    for index, label in enumerate(labels):
        authored_path = authored_labels.get(str(label).strip("/"))
        if authored_path is not None:
            result[index] = authored_path
            continue
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
