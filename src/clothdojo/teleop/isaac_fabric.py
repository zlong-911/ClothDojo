from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def _write_local_transforms(
    matrices: wp.fabricarray(dtype=wp.mat44d),
    indices: wp.fabricarray(dtype=wp.uint32),
    body_q: wp.array(dtype=wp.transformf),
    body_parent: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    body = int(indices[i])
    transform = body_q[body]
    parent = body_parent[body]
    if parent >= 0:
        transform = wp.transform_multiply(
            wp.transform_inverse(body_q[parent]),
            transform,
        )
    matrices[i] = wp.transpose(wp.mat44d(wp.transform_to_matrix(transform)))


@wp.kernel(enable_backward=False)
def _write_points(
    points: wp.array(dtype=wp.vec3f),
    particle_q: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    points[i] = particle_q[i]


class NewtonFabricBridge:
    """GPU-direct Newton state projection into Isaac Fabric.

    Isaac Sim's bundled Newton FabricManager currently handles rigid bodies only
    and synchronizes the CUDA device every update. This bridge retains its
    ``newton:index`` convention but follows simcloth-bench's GPU-direct path:
    cloth points and rigid transforms are submitted on Warp's current stream,
    then the USDRT attribute invalidation publishes the cloth buffer to Fabric.
    """

    def __init__(self, model, usd_context, *, device: str) -> None:
        from usdrt import Rt, Sdf, Usd

        if not str(device).startswith("cuda:"):
            raise ValueError("teleop Fabric requires a CUDA device")
        self._device = device
        self._stage = Usd.Stage.Attach(usd_context.get_stage_id())
        self._requirements = [
            (
                Sdf.ValueTypeNames.Matrix4d,
                "_localMatrix",
                Usd.Access.ReadWrite,
            ),
            (Sdf.ValueTypeNames.UInt, "newton:index", Usd.Access.Read),
        ]
        body_paths = _find_render_body_paths(usd_context.get_stage(), model.body_label)
        mapped = 0
        for index, _label in enumerate(model.body_label):
            path = body_paths.get(index)
            if path is None:
                continue
            prim = self._stage.GetPrimAtPath(Sdf.Path(path))
            if not prim or not prim.IsValid():
                continue
            prim.CreateAttribute("newton:index", Sdf.ValueTypeNames.UInt, True)
            prim.GetAttribute("newton:index").Set(index)
            xform = Rt.Xformable(prim)
            if not xform.HasLocalXform():
                xform.SetLocalXformFromUsd()
            if not xform.HasWorldXform():
                xform.SetWorldXformFromUsd()
            mapped += 1

        body_parent = [-1] * int(model.body_count)
        for child, parent in zip(
            model.joint_child.numpy().tolist(),
            model.joint_parent.numpy().tolist(),
            strict=True,
        ):
            body_parent[int(child)] = int(parent)
        self._body_parent = wp.array(
            body_parent,
            dtype=wp.int32,
            device=self._device,
        )
        # SelectPrims searches Fabric, not the source USD stage. Traversing the
        # attached USDRT stage is the documented population step.
        for _prim in self._stage.Traverse():
            pass

        self._cloth_attr = None
        self._cloth_points = None
        cloth = self._stage.GetPrimAtPath(Sdf.Path("/World/Cloth"))
        if cloth and cloth.IsValid():
            attr = cloth.GetAttribute("points")
            if attr and attr.IsValid():
                attr.SyncDataToGpu()
                self._cloth_points = wp.array(attr.Get(), dtype=wp.vec3f, device=self._device)
                self._cloth_attr = attr
                if self._cloth_points.shape[0] != int(model.particle_count):
                    raise RuntimeError(
                        "Isaac Fabric cloth vertex count does not match Newton particles: "
                        f"{self._cloth_points.shape[0]} != {int(model.particle_count)}"
                    )
        if mapped == 0:
            raise RuntimeError("Isaac Fabric did not map any Newton rigid bodies")
        if int(model.particle_count) > 0 and self._cloth_points is None:
            raise RuntimeError("Isaac Fabric did not map the combined cloth mesh")
        print(
            f"Isaac Fabric bridge ready: rigid={mapped}, "
            f"cloth={self._cloth_points is not None}, sync=fabric-gpu",
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

        selection = self._stage.SelectPrims(
            require_attrs=self._requirements,
            device=str(self._device),
        )
        if selection.GetCount() == 0:
            diagnostics = {
                name: self._stage.SelectPrims(require_attrs=[requirement], device=device).GetCount()
                for name, requirement, device in (
                    ("index_cuda", self._requirements[1], str(self._device)),
                    ("matrix_cuda", self._requirements[0], str(self._device)),
                    ("index_cpu", self._requirements[1], "cpu"),
                    ("matrix_cpu", self._requirements[0], "cpu"),
                )
            }
            raise RuntimeError(
                f"Isaac Fabric lost all Newton rigid-body mappings: {diagnostics}"
            )
        matrices = wp.fabricarray(selection, "_localMatrix")
        indices = wp.fabricarray(selection, "newton:index")
        wp.launch(
            _write_local_transforms,
            dim=indices.shape[0],
            inputs=[matrices, indices, state.body_q, self._body_parent],
            device=self._device,
        )


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
