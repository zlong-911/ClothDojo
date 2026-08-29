"""Robot workstations whose simulation and rendering truth lives in USD."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from clothdojo.rendering import DATASET_CAMERA_NAMES, CameraSpec

if "pxr.Work" not in sys.modules:
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

PIPER = "piper"
ALOHA2 = "aloha2"
X5 = "x5"
PANDA = "panda"
UR5E_2F85 = "ur5e_2f85"


@dataclass(frozen=True, slots=True)
class AttachFrame:
    parent_body: str
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class InstallationFrame:
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class RobotProfile:
    """Control-facing view of one immutable workstation USD."""

    robot_id: str
    asset: str
    arm_joint_names: tuple[str, ...]
    arm_coord_indices: tuple[int, ...]
    arm_dof_indices: tuple[int, ...]
    gripper_coord_indices: tuple[int, ...]
    gripper_dof_indices: tuple[int, ...]
    joint_dof_count: int
    home_joint_q: tuple[float, ...]
    gripper_closed_positions: tuple[float, float]
    gripper_open_positions: tuple[float, float]
    attach_frames: tuple[AttachFrame, AttachFrame]
    cameras: tuple[CameraSpec, ...]
    body_names: tuple[str, ...]
    body_parents: tuple[int, ...]
    root_body_frames: tuple[InstallationFrame, ...]

    def __post_init__(self) -> None:
        if (
            not self.arm_joint_names
            or len(self.arm_joint_names) != len(self.arm_coord_indices)
            or len(self.arm_joint_names) % 2
        ):
            raise ValueError("robot workstation must define equal non-empty left/right arms")
        if len(self.gripper_coord_indices) != 4:
            raise ValueError("robot workstation must define four gripper coordinates")
        if len(self.arm_dof_indices) != len(self.arm_coord_indices):
            raise ValueError("arm coordinate and velocity layouts must have equal size")
        if len(self.gripper_dof_indices) != len(self.gripper_coord_indices):
            raise ValueError("gripper coordinate and velocity layouts must have equal size")
        if self.joint_dof_count <= max(*self.arm_dof_indices, *self.gripper_dof_indices):
            raise ValueError("robot workstation velocity layout does not cover its DOFs")
        if len(self.home_joint_q) <= max(*self.arm_coord_indices, *self.gripper_coord_indices):
            raise ValueError("robot workstation home pose does not cover its coordinates")
        if tuple(camera.name for camera in self.cameras) != DATASET_CAMERA_NAMES:
            raise ValueError("robot workstation must define the four dataset cameras in order")
        if len(self.body_names) != len(self.body_parents) or not self.body_names:
            raise ValueError("robot workstation body topology is incomplete")
        if any(parent >= index for index, parent in enumerate(self.body_parents)):
            raise ValueError("robot workstation body parents must precede their children")
        if len(self.root_body_frames) != self.body_parents.count(-1):
            raise ValueError("robot workstation root frames do not match its articulations")

    @property
    def arm_dofs_per_side(self) -> int:
        return len(self.arm_coord_indices) // 2

    def to_json_dict(self) -> dict:
        return asdict(self)


def _required(prim, name: str):
    attribute = prim.GetAttribute(name)
    value = attribute.Get() if attribute and attribute.HasAuthoredValue() else None
    if value is None:
        raise RuntimeError(f"robot workstation {prim.GetPath()} is missing {name}")
    return value


def _matrix_tuple(matrix) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(matrix[column][row]) for column in range(4))
        for row in range(4)
    )


def _load_workstation(robot_id: str, asset: str) -> RobotProfile:
    import newton_usd_schemas  # noqa: F401
    from pxr import Usd, UsdGeom

    path = Path(__file__).resolve().parents[2] / "assets" / asset
    stage = Usd.Stage.Open(str(path))
    if stage is None or not stage.GetDefaultPrim():
        raise RuntimeError(f"could not open robot workstation USD: {path}")
    root = stage.GetDefaultPrim()
    if int(_required(root, "clothdojo:workstationSchema")) != 1:
        raise RuntimeError(f"unsupported robot workstation schema: {path}")
    authored_robot_id = str(_required(root, "clothdojo:robotId"))
    if authored_robot_id != robot_id:
        raise RuntimeError(
            f"robot workstation identity mismatch: expected {robot_id}, got {authored_robot_id}"
        )

    cameras = {}
    attach_frames = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera) and prim.GetName() in DATASET_CAMERA_NAMES:
            camera = UsdGeom.Camera(prim)
            horizontal_aperture = round(
                float(camera.GetHorizontalApertureAttr().Get()), 6
            )
            vertical_aperture = round(
                float(camera.GetVerticalApertureAttr().Get()), 6
            )
            if abs(vertical_aperture / horizontal_aperture - 0.75) > 1.0e-6:
                raise RuntimeError(
                    f"robot camera {prim.GetPath()} must use a 640:480 aperture ratio"
                )
            parent_attr = prim.GetAttribute("clothdojo:parentBody")
            parent_body = str(parent_attr.Get()) if parent_attr and parent_attr.Get() else None
            cameras[prim.GetName()] = CameraSpec(
                name=prim.GetName(),
                parent_body=parent_body,
                parent_from_camera=_matrix_tuple(
                    UsdGeom.Xformable(prim).GetLocalTransformation()
                ),
                focal_length=round(float(camera.GetFocalLengthAttr().Get()), 6),
                horizontal_aperture=horizontal_aperture,
            )
        role = prim.GetAttribute("clothdojo:role")
        if not role or role.Get() != "end_effector_attach":
            continue
        parent_body = str(_required(prim, "clothdojo:parentBody"))
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        attach_frames[parent_body] = AttachFrame(
            parent_body=parent_body,
            position=tuple(float(value) for value in translation),
            rotation_xyzw=(
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
                float(rotation.GetReal()),
            ),
        )
    if set(cameras) != set(DATASET_CAMERA_NAMES):
        raise RuntimeError(f"robot workstation does not define all dataset cameras: {path}")
    attach_order = tuple(
        name for name in _required(root, "clothdojo:bodyNames") if name in attach_frames
    )
    if len(attach_order) != 2:
        raise RuntimeError(f"robot workstation does not define two attach frames: {path}")

    body_names = tuple(str(value) for value in _required(root, "clothdojo:bodyNames"))
    body_parents = tuple(int(value) for value in _required(root, "clothdojo:bodyParents"))
    root_values = tuple(float(value) for value in _required(root, "clothdojo:rootBodyFrames"))
    root_count = body_parents.count(-1)
    if len(root_values) != root_count * 7:
        raise RuntimeError(f"robot workstation root-frame array is invalid: {path}")
    root_frames = tuple(
        InstallationFrame(
            position=tuple(root_values[offset : offset + 3]),
            rotation_xyzw=tuple(root_values[offset + 3 : offset + 7]),
        )
        for offset in range(0, len(root_values), 7)
    )
    return RobotProfile(
        robot_id=robot_id,
        asset=asset,
        arm_joint_names=tuple(str(value) for value in _required(root, "clothdojo:armJointNames")),
        arm_coord_indices=tuple(
            int(value) for value in _required(root, "clothdojo:armCoordIndices")
        ),
        arm_dof_indices=tuple(int(value) for value in _required(root, "clothdojo:armDofIndices")),
        gripper_coord_indices=tuple(
            int(value) for value in _required(root, "clothdojo:gripperCoordIndices")
        ),
        gripper_dof_indices=tuple(
            int(value) for value in _required(root, "clothdojo:gripperDofIndices")
        ),
        joint_dof_count=int(_required(root, "clothdojo:jointDofCount")),
        home_joint_q=tuple(float(value) for value in _required(root, "clothdojo:homeJointQ")),
        gripper_closed_positions=tuple(
            float(value) for value in _required(root, "clothdojo:gripperClosedPositions")
        ),
        gripper_open_positions=tuple(
            float(value) for value in _required(root, "clothdojo:gripperOpenPositions")
        ),
        attach_frames=tuple(attach_frames[name] for name in attach_order),
        cameras=tuple(cameras[name] for name in DATASET_CAMERA_NAMES),
        body_names=body_names,
        body_parents=body_parents,
        root_body_frames=root_frames,
    )


_ASSETS = {
    PIPER: "robots/piper/piper.usda",
    ALOHA2: "robots/aloha2/aloha2.usda",
    X5: "robots/x5/x5.usda",
    PANDA: "robots/panda/panda.usda",
    UR5E_2F85: "robots/ur5e_2f85/ur5e_2f85.usda",
}
_PROFILES = {
    robot_id: _load_workstation(robot_id, asset)
    for robot_id, asset in _ASSETS.items()
}


def robot_ids() -> tuple[str, ...]:
    return tuple(_PROFILES)


def robot_profile(robot_id: str) -> RobotProfile:
    try:
        return _PROFILES[robot_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown robot {robot_id!r}; expected one of {', '.join(robot_ids())}"
        ) from exc
