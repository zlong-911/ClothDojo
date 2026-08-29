"""Fixed robot workstation profiles shared by simulation, data and rendering."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from math import radians
from pathlib import Path

from clothdojo.rendering import DATASET_CAMERA_NAMES, CameraSpec

if "pxr.Work" not in sys.modules:
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

PIPER = "piper"
ALOHA2 = "aloha2"
X5 = "x5"
PANDA = "panda"
UR5E_2F85 = "ur5e_2f85"

_PRIMARY = (
    (0.0, 0.8660254037844386, -0.5, 0.253845),
    (-1.0, 0.0, 0.0, 0.001071),
    (0.0, 0.5, 0.8660254037844386, 1.308),
    (0.0, 0.0, 0.0, 1.0),
)
_WRIST = (
    (0.0, 1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0109),
    (0.0, 0.0, 0.0, 1.0),
)
_ALOHA2_WRIST = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -0.906307454435, -0.422618975006, -0.0824748),
    (0.0, 0.422618975006, -0.906307454435, -0.0095955),
    (0.0, 0.0, 0.0, 1.0),
)
_ALOHA2_HIGH = (
    (0.0, 0.906448510586, -0.422316347845, 0.097206),
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 0.422316347845, 0.906448510586, 1.82524),
    (0.0, 0.0, 0.0, 1.0),
)
_ALOHA2_LOW = (
    (0.0, -0.09506026, -0.99547152, 0.023833),
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 0.99547152, -0.09506026, 0.8316055),
    (0.0, 0.0, 0.0, 1.0),
)
_LOW = (
    (0.0, -0.374606593415912, 0.9271838545667874, 1.25),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.9271838545667874, 0.374606593415912, 1.35),
    (0.0, 0.0, 0.0, 1.0),
)


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
class PositionDriveOverride:
    """Newton position drives for an upstream MJCF actuator it cannot import."""

    arm_stiffness: tuple[float, ...]
    arm_damping: tuple[float, ...]
    arm_effort_limit: tuple[float, ...]
    gripper_stiffness: tuple[float, float]
    gripper_damping: tuple[float, float]
    gripper_effort_limit: tuple[float, float]


@dataclass(frozen=True, slots=True)
class RobotProfile:
    robot_id: str
    newton_asset: str
    newton_visual_asset: str
    isaac_asset: str
    installation_asset: str | None
    asset_instances: int
    asset_frame: InstallationFrame
    installations: tuple[InstallationFrame, ...]
    root_body_frames: tuple[InstallationFrame, ...]
    arm_joint_names: tuple[str, ...]
    arm_coord_indices: tuple[int, ...]
    arm_dof_indices: tuple[int, ...]
    gripper_coord_indices: tuple[int, ...]
    gripper_dof_indices: tuple[int, ...]
    joint_dof_count: int
    home_joint_q: tuple[float, ...]
    gripper_closed_positions: tuple[float, float]
    gripper_open_positions: tuple[float, float]
    position_drive_override: PositionDriveOverride | None
    attach_frames: tuple[AttachFrame, AttachFrame]
    cameras: tuple[CameraSpec, ...]
    body_names: tuple[str, ...]
    body_parents: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.arm_joint_names
            or len(self.arm_joint_names) != len(self.arm_coord_indices)
            or len(self.arm_joint_names) % 2
        ):
            raise ValueError(
                "robot profile must define equal non-empty left/right arm coordinates"
            )
        if len(self.gripper_coord_indices) != 4:
            raise ValueError("robot profile must define four gripper coordinates")
        if len(self.arm_dof_indices) != len(self.arm_coord_indices):
            raise ValueError("arm coordinate and velocity layouts must have equal size")
        if len(self.gripper_dof_indices) != len(self.gripper_coord_indices):
            raise ValueError("gripper coordinate and velocity layouts must have equal size")
        if self.asset_instances not in (1, 2):
            raise ValueError("robot profile must use one bimanual or two single-arm assets")
        if len(self.home_joint_q) <= max(*self.arm_coord_indices, *self.gripper_coord_indices):
            raise ValueError("robot profile home pose does not cover its coordinates")
        if self.joint_dof_count <= max(*self.arm_dof_indices, *self.gripper_dof_indices):
            raise ValueError("robot profile velocity layout does not cover its DOFs")
        if tuple(camera.name for camera in self.cameras) != DATASET_CAMERA_NAMES:
            raise ValueError("robot profile must define the four dataset cameras in order")
        if len(self.body_names) != len(self.body_parents) or not self.body_names:
            raise ValueError("robot profile body topology is incomplete")
        if any(parent >= index for index, parent in enumerate(self.body_parents)):
            raise ValueError("robot profile body parents must precede their children")
        if len(self.installations) != self.body_parents.count(-1):
            raise ValueError("robot installations must match the number of root rigid bodies")
        if len(self.root_body_frames) != self.body_parents.count(-1):
            raise ValueError("robot root frames must match the number of root rigid bodies")
        drive = self.position_drive_override
        if drive is not None:
            arm_count = self.arm_dofs_per_side
            if any(
                len(values) != arm_count
                for values in (
                    drive.arm_stiffness,
                    drive.arm_damping,
                    drive.arm_effort_limit,
                )
            ):
                raise ValueError("position-drive arm arrays must match one arm")
            if (
                any(value <= 0.0 for value in drive.arm_stiffness)
                or any(value < 0.0 for value in drive.arm_damping)
                or any(value <= 0.0 for value in drive.arm_effort_limit)
                or any(value <= 0.0 for value in drive.gripper_stiffness)
                or any(value < 0.0 for value in drive.gripper_damping)
                or any(value <= 0.0 for value in drive.gripper_effort_limit)
            ):
                raise ValueError("position-drive gains and limits are invalid")

    @property
    def arm_dofs_per_side(self) -> int:
        return len(self.arm_coord_indices) // 2

    def to_json_dict(self) -> dict:
        return asdict(self)


def _cameras(
    left_mount: str,
    right_mount: str,
    *,
    wrist_transform=_WRIST,
    wrist_focal_length=13.0,
    wrist_horizontal_aperture=15.71625,
    high_transform=_PRIMARY,
    low_transform=_LOW,
    high_focal_length=10.0,
    high_horizontal_aperture=14.266,
    low_focal_length=10.0,
    low_horizontal_aperture=14.266,
) -> tuple[CameraSpec, ...]:
    return (
        CameraSpec(
            "cam_high",
            None,
            high_transform,
            high_focal_length,
            high_horizontal_aperture,
        ),
        CameraSpec(
            "cam_low",
            None,
            low_transform,
            low_focal_length,
            low_horizontal_aperture,
        ),
        CameraSpec(
            "cam_left_wrist",
            left_mount,
            wrist_transform,
            wrist_focal_length,
            wrist_horizontal_aperture,
        ),
        CameraSpec(
            "cam_right_wrist",
            right_mount,
            wrist_transform,
            wrist_focal_length,
            wrist_horizontal_aperture,
        ),
    )


@dataclass(frozen=True, slots=True)
class _UsdWorkstation:
    home_joint_q: tuple[float, ...]
    gripper_closed_positions: tuple[float, float]
    gripper_open_positions: tuple[float, float]
    attach_frames: tuple[AttachFrame, AttachFrame]
    cameras: tuple[CameraSpec, ...]


def _usd_workstation(asset: str) -> _UsdWorkstation:
    """Load fixed reset, grasp and camera semantics from a robot workstation USD."""

    import newton_usd_schemas  # noqa: F401
    from pxr import Usd, UsdGeom

    path = Path(__file__).resolve().parents[2] / "assets" / asset
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open robot camera asset: {path}")
    cameras = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        name = prim.GetName()
        if name not in DATASET_CAMERA_NAMES:
            continue
        camera = UsdGeom.Camera(prim)
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        parent_attr = prim.GetAttribute("clothdojo:parentBody")
        parent_body = str(parent_attr.Get()) if parent_attr and parent_attr.Get() else None
        horizontal_aperture = round(float(camera.GetHorizontalApertureAttr().Get()), 6)
        vertical_aperture = round(float(camera.GetVerticalApertureAttr().Get()), 6)
        if abs(vertical_aperture / horizontal_aperture - 0.75) > 1.0e-6:
            raise RuntimeError(f"robot camera {prim.GetPath()} must use the 640:480 aperture ratio")
        cameras[name] = CameraSpec(
            name=name,
            parent_body=parent_body,
            parent_from_camera=tuple(
                tuple(float(matrix[column][row]) for column in range(4))
                for row in range(4)
            ),
            focal_length=round(float(camera.GetFocalLengthAttr().Get()), 6),
            horizontal_aperture=horizontal_aperture,
        )
    if set(cameras) != set(DATASET_CAMERA_NAMES):
        raise RuntimeError(f"robot camera asset does not define {DATASET_CAMERA_NAMES}: {path}")

    home_joint_q = []
    gripper_limits = []
    for side in ("left", "right"):
        for joint in range(1, 9):
            prim = stage.GetPrimAtPath(f"/piper_dual/Physics/{side}/joint{joint}")
            if not prim:
                raise RuntimeError(f"robot workstation is missing {side}/joint{joint}: {path}")
            angular = prim.GetAttribute("state:angular:physics:position")
            linear = prim.GetAttribute("state:linear:physics:position")
            angular_value = angular.Get() if angular and angular.HasAuthoredValue() else None
            linear_value = linear.Get() if linear and linear.HasAuthoredValue() else None
            if angular_value is not None and linear_value is not None:
                raise RuntimeError(f"robot joint {prim.GetPath()} has two position states")
            home_joint_q.append(
                radians(float(angular_value))
                if angular_value is not None
                else round(float(linear_value), 6) if linear_value is not None else 0.0
            )
            if joint in (7, 8):
                gripper_limits.append(
                    (
                        round(float(prim.GetAttribute("physics:lowerLimit").Get()), 6),
                        round(float(prim.GetAttribute("physics:upperLimit").Get()), 6),
                    )
                )

    attach_by_parent = {}
    for prim in stage.Traverse():
        role = prim.GetAttribute("clothdojo:role")
        if not role or role.Get() != "end_effector_attach":
            continue
        parent = prim.GetAttribute("clothdojo:parentBody").Get()
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        attach_by_parent[str(parent)] = AttachFrame(
            parent_body=str(parent),
            position=tuple(float(value) for value in translation),
            rotation_xyzw=(
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
                float(rotation.GetReal()),
            ),
        )
    attach_order = ("left/link6", "right/link6")
    if set(attach_by_parent) != set(attach_order):
        raise RuntimeError(f"robot workstation does not define both attach frames: {path}")

    # Limits are ordered left joint7/joint8, then right joint7/joint8.
    left_leader, left_follower, right_leader, right_follower = gripper_limits
    if left_leader != right_leader or left_follower != right_follower:
        raise RuntimeError("robot workstation gripper limits are not symmetric")
    return _UsdWorkstation(
        home_joint_q=tuple(home_joint_q),
        gripper_closed_positions=(left_leader[0], left_follower[1]),
        gripper_open_positions=(left_leader[1], left_follower[0]),
        attach_frames=tuple(attach_by_parent[parent] for parent in attach_order),
        cameras=tuple(cameras[name] for name in DATASET_CAMERA_NAMES),
    )


_PIPER_WORKSTATION = _usd_workstation("robots/piper/piper.usda")


_PROFILES = {
    PIPER: RobotProfile(
        robot_id=PIPER,
        newton_asset="robots/piper/piper.usda",
        newton_visual_asset="robots/piper/piper.usda",
        isaac_asset="robots/piper/piper.usda",
        installation_asset=None,
        asset_instances=1,
        asset_frame=InstallationFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        installations=(InstallationFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
        root_body_frames=(InstallationFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
        arm_joint_names=tuple(
            f"{side}/joint{joint}" for side in ("left", "right") for joint in range(1, 7)
        ),
        arm_coord_indices=(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13),
        arm_dof_indices=(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13),
        gripper_coord_indices=(6, 7, 14, 15),
        gripper_dof_indices=(6, 7, 14, 15),
        joint_dof_count=16,
        home_joint_q=_PIPER_WORKSTATION.home_joint_q,
        gripper_closed_positions=_PIPER_WORKSTATION.gripper_closed_positions,
        gripper_open_positions=_PIPER_WORKSTATION.gripper_open_positions,
        position_drive_override=None,
        attach_frames=_PIPER_WORKSTATION.attach_frames,
        cameras=_PIPER_WORKSTATION.cameras,
        body_names=(
            "world_base",
            *(
                f"{side}/{link}"
                for side in ("left", "right")
                for link in (
                    "base_link",
                    "link1",
                    "link2",
                    "link3",
                    "link4",
                    "link5",
                    "link6",
                    "camera",
                    "link7",
                    "link8",
                )
            ),
        ),
        body_parents=(
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            7,
            7,
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            17,
            17,
        ),
    ),
    ALOHA2: RobotProfile(
        robot_id=ALOHA2,
        newton_asset="robots/aloha2/aloha.xml",
        newton_visual_asset="robots/aloha2/aloha.xml",
        isaac_asset="robots/aloha2/aloha.xml",
        installation_asset="robots/aloha2/frame.xml",
        asset_instances=1,
        asset_frame=InstallationFrame(
            (0.401, 0.0, 0.8),
            (0.0, 0.0, -0.7071067811865475, 0.7071067811865476),
        ),
        installations=(
            InstallationFrame(
                (0.382, 0.559, 0.82),
                (0.0, 0.0, -0.7071067811865475, 0.7071067811865476),
            ),
            InstallationFrame(
                (0.382, -0.559, 0.82),
                (0.0, 0.0, 0.7071067811865476, 0.7071067811865475),
            ),
        ),
        root_body_frames=(
            InstallationFrame(
                (0.382, 0.559, 0.82),
                (0.0, 0.0, -0.7071067811865475, 0.7071067811865476),
            ),
            InstallationFrame(
                (0.382, -0.559, 0.82),
                (0.0, 0.0, 0.7071067811865476, 0.7071067811865475),
            ),
        ),
        arm_joint_names=tuple(
            f"{side}/{joint}"
            for side in ("left", "right")
            for joint in (
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
            )
        ),
        arm_coord_indices=(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13),
        arm_dof_indices=(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13),
        gripper_coord_indices=(6, 7, 14, 15),
        gripper_dof_indices=(6, 7, 14, 15),
        joint_dof_count=16,
        home_joint_q=(0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.0084, 0.0084) * 2,
        # The pinned custom-finger meshes stop intersecting at about 0.00785 m.
        # Keep a small geometric margin while remaining visually fully closed.
        gripper_closed_positions=(0.0079, 0.0079),
        gripper_open_positions=(0.037, 0.037),
        position_drive_override=None,
        attach_frames=(
            AttachFrame("left/gripper_link", (0.13, 0.0, -0.003), (0.5, 0.5, 0.5, 0.5)),
            AttachFrame("right/gripper_link", (0.13, 0.0, -0.003), (0.5, 0.5, 0.5, 0.5)),
        ),
        cameras=_cameras(
            "left/gripper_base",
            "right/gripper_base",
            wrist_transform=_ALOHA2_WRIST,
            wrist_focal_length=1.93,
            wrist_horizontal_aperture=3.896,
            high_transform=_ALOHA2_HIGH,
            low_transform=_ALOHA2_LOW,
            high_focal_length=1.93,
            high_horizontal_aperture=3.896,
            low_focal_length=1.93,
            low_horizontal_aperture=3.896,
        ),
        body_names=tuple(
            f"{side}/{link}"
            for side in ("left", "right")
            for link in (
                "base_link",
                "shoulder_link",
                "upper_arm_link",
                "upper_forearm_link",
                "lower_forearm_link",
                "wrist_link",
                "gripper_link",
                "gripper_base",
                "left_finger_link",
                "right_finger_link",
            )
        ),
        body_parents=(
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            7,
            -1,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            17,
        ),
    ),
    X5: RobotProfile(
        robot_id=X5,
        newton_asset="robots/x5/x5.xml",
        newton_visual_asset="robots/x5/x5.xml",
        isaac_asset="robots/x5/x5.xml",
        installation_asset=None,
        asset_instances=2,
        asset_frame=InstallationFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        # Migrated from the old project's validated RoboDojo X5 setup.
        installations=(
            InstallationFrame((0.25, 0.3, 0.8), (0.0, 0.0, 0.0, 1.0)),
            InstallationFrame((0.25, -0.3, 0.8), (0.0, 0.0, 0.0, 1.0)),
        ),
        root_body_frames=(
            InstallationFrame((0.25, 0.3, 0.8), (0.0, 0.0, 0.0, 1.0)),
            InstallationFrame((0.25, -0.3, 0.8), (0.0, 0.0, 0.0, 1.0)),
        ),
        arm_joint_names=tuple(
            f"{side}/joint{joint}" for side in ("left", "right") for joint in range(1, 7)
        ),
        arm_coord_indices=(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13),
        arm_dof_indices=(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13),
        gripper_coord_indices=(6, 7, 14, 15),
        gripper_dof_indices=(6, 7, 14, 15),
        joint_dof_count=16,
        home_joint_q=(0.0,) * 6 + (0.044, 0.044) + (0.0,) * 6 + (0.044, 0.044),
        gripper_closed_positions=(0.0, 0.0),
        gripper_open_positions=(0.044, 0.044),
        # These are the locked RoboDojo articulation drives used by the old
        # project. They are environment calibration rather than vendor gains.
        position_drive_override=PositionDriveOverride(
            arm_stiffness=(4400.0,) * 6,
            arm_damping=(40.0,) * 6,
            arm_effort_limit=(100.0,) * 6,
            gripper_stiffness=(2300.0, 2300.0),
            gripper_damping=(100.0, 100.0),
            gripper_effort_limit=(100.0, 100.0),
        ),
        attach_frames=(
            AttachFrame(
                "left/link6",
                (0.152602, 0.0, -0.00014),
                (0.5, 0.5, 0.5, 0.5),
            ),
            AttachFrame(
                "right/link6",
                (0.152602, 0.0, -0.00014),
                (0.5, 0.5, 0.5, 0.5),
            ),
        ),
        cameras=_cameras("left/camera", "right/camera"),
        body_names=tuple(
            f"{side}/{link}"
            for side in ("left", "right")
            for link in (
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
        ),
        body_parents=(
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            6,
            6,
            -1,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            17,
            17,
        ),
    ),
    PANDA: RobotProfile(
        robot_id=PANDA,
        newton_asset="robots/panda/panda.xml",
        newton_visual_asset="robots/panda/panda.xml",
        isaac_asset="robots/panda/panda.xml",
        installation_asset=None,
        asset_instances=2,
        asset_frame=InstallationFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        installations=(
            InstallationFrame((0.25, 0.6, 0.8), (0.0, 0.0, 0.0, 1.0)),
            InstallationFrame((0.25, -0.6, 0.8), (0.0, 0.0, 0.0, 1.0)),
        ),
        root_body_frames=(
            InstallationFrame((0.25, 0.6, 0.8), (0.0, 0.0, 0.0, 1.0)),
            InstallationFrame((0.25, -0.6, 0.8), (0.0, 0.0, 0.0, 1.0)),
        ),
        arm_joint_names=tuple(
            f"{side}/joint{joint}" for side in ("left", "right") for joint in range(1, 8)
        ),
        arm_coord_indices=(0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15),
        arm_dof_indices=(0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15),
        gripper_coord_indices=(7, 8, 16, 17),
        gripper_dof_indices=(7, 8, 16, 17),
        joint_dof_count=18,
        home_joint_q=(
            0.0,
            0.0,
            0.0,
            -1.57079,
            0.0,
            1.57079,
            -0.7853,
            0.04,
            0.04,
        )
        * 2,
        gripper_closed_positions=(0.0, 0.0),
        gripper_open_positions=(0.04, 0.04),
        # Menagerie's affine arm and tendon actuators are not imported by
        # Newton 1.5. These are their exactly equivalent scalar position
        # drives; the gripper values include the tendon's 0.5 coefficients.
        position_drive_override=PositionDriveOverride(
            arm_stiffness=(4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0),
            arm_damping=(450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0),
            arm_effort_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
            gripper_stiffness=(50.0, 50.0),
            gripper_damping=(5.0, 5.0),
            gripper_effort_limit=(50.0, 50.0),
        ),
        attach_frames=(
            AttachFrame(
                "left/hand",
                (0.0, 0.0, 0.107147),
                (0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
            ),
            AttachFrame(
                "right/hand",
                (0.0, 0.0, 0.107147),
                (0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
            ),
        ),
        cameras=_cameras("left/hand", "right/hand"),
        body_names=tuple(
            f"{side}/{link}"
            for side in ("left", "right")
            for link in (
                "link0",
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "link7",
                "hand",
                "left_finger",
                "right_finger",
            )
        ),
        body_parents=(
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            -1,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            19,
        ),
    ),
    UR5E_2F85: RobotProfile(
        robot_id=UR5E_2F85,
        newton_asset="robots/ur5e_2f85/ur5e_2f85.xml",
        newton_visual_asset="robots/ur5e_2f85/ur5e_2f85.xml",
        isaac_asset="robots/ur5e_2f85/ur5e_2f85.xml",
        installation_asset=None,
        asset_instances=2,
        asset_frame=InstallationFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        installations=(
            InstallationFrame((0.25, 0.6, 0.8), (0.0, 0.0, 0.0, 1.0)),
            InstallationFrame((0.25, -0.6, 0.8), (0.0, 0.0, 0.0, 1.0)),
        ),
        # The Menagerie UR root body carries a 180-degree Z rotation inside
        # each independently installed MJCF instance.
        root_body_frames=(
            InstallationFrame((0.25, 0.6, 0.8), (0.0, 0.0, -1.0, 0.0)),
            InstallationFrame((0.25, -0.6, 0.8), (0.0, 0.0, -1.0, 0.0)),
        ),
        arm_joint_names=tuple(
            f"{side}/{joint}"
            for side in ("left", "right")
            for joint in (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            )
        ),
        # Newton imports the two four-bar linkage connect constraints as ball
        # joints (four quaternion coordinates each), so one asset occupies 22
        # coordinates rather than MuJoCo's 14 generalized coordinates.
        arm_coord_indices=(0, 1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27),
        arm_dof_indices=(0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25),
        gripper_coord_indices=(6, 10, 28, 32),
        gripper_dof_indices=(6, 10, 26, 30),
        joint_dof_count=40,
        home_joint_q=(
            2.533765,
            -1.489121,
            2.102872,
            -2.184395,
            -1.571103,
            2.534312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            3.176974,
            -1.489208,
            2.102969,
            -2.184404,
            -1.570822,
            3.176942,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        gripper_closed_positions=(0.8, 0.8),
        gripper_open_positions=(0.0, 0.0),
        # Menagerie's affine arm and tendon actuators are not imported by
        # Newton 1.5. These scalar drives preserve their gains and limits;
        # the gripper values include the tendon's 0.5 coefficients.
        position_drive_override=PositionDriveOverride(
            arm_stiffness=(2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0),
            arm_damping=(400.0, 400.0, 400.0, 100.0, 100.0, 100.0),
            arm_effort_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
            gripper_stiffness=(50.0, 50.0),
            gripper_damping=(5.0, 5.0),
            gripper_effort_limit=(2.5, 2.5),
        ),
        attach_frames=(
            AttachFrame(
                "left/gripper_base",
                (0.0, 0.0, 0.157719),
                (0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
            ),
            AttachFrame(
                "right/gripper_base",
                (0.0, 0.0, 0.157719),
                (0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
            ),
        ),
        cameras=_cameras("left/gripper_base", "right/gripper_base"),
        body_names=tuple(
            f"{side}/{link}"
            for side in ("left", "right")
            for link in (
                "base",
                "shoulder_link",
                "upper_arm_link",
                "forearm_link",
                "wrist_1_link",
                "wrist_2_link",
                "wrist_3_link",
                "gripper_base_mount",
                "gripper_base",
                "gripper_right_driver",
                "gripper_right_coupler",
                "gripper_right_spring_link",
                "gripper_right_follower",
                "gripper_right_pad",
                "gripper_right_silicone_pad",
                "gripper_left_driver",
                "gripper_left_coupler",
                "gripper_left_spring_link",
                "gripper_left_follower",
                "gripper_left_pad",
                "gripper_left_silicone_pad",
            )
        ),
        body_parents=(
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            8,
            11,
            12,
            13,
            8,
            15,
            8,
            17,
            18,
            19,
            -1,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            29,
            32,
            33,
            34,
            29,
            36,
            29,
            38,
            39,
            40,
        ),
    ),
}


def robot_profile(robot_id: str) -> RobotProfile:
    try:
        return _PROFILES[robot_id]
    except KeyError as exc:
        raise ValueError(f"unknown robot profile: {robot_id!r}") from exc
