# UR5e wrist-camera source snapshot

This is the published PickNik **flush camera adapter + retained Robotiq
coupling + D415** arrangement, not a newly designed bracket. PickNik's
[UR5e hardware guide, section 4](https://docs.picknik.ai/6/getting_started/configuration_tutorials/ur5e_hardware_setup_guide/)
places the camera adapter first, flush against the robot flange, followed by
the gripper coupling. Its separate lab simulation also has a 33.1 mm tool
changer; that optional component is not installed in this workstation.

## Immutable sources

- PickNik adapter visual DAE, full STL, collision STL and
  `picknik_ur_camera_adapter.urdf.xacro`:
  [picknik_accessories_non_lfs](https://github.com/PickNikRobotics/picknik_accessories_non_lfs/tree/f6dab9cc7e2807edc4004ac6778f8c2dc9497b53/descriptions/brackets/ur_realsense_camera_adapter),
  commit `f6dab9cc7e2807edc4004ac6778f8c2dc9497b53`, BSD-3-Clause
  (`LICENSE.picknik`). The canonical repository uses Git LFS; this official
  non-LFS mirror supplies the actual meshes.
- Integrated `picknik_ur.xacro`:
  [moveit_pro_example_ws](https://github.com/PickNikRobotics/moveit_pro_example_ws/blob/15977219511b965bde26e55e89bd50090371fd07/src/lab_sim/description/picknik_ur.xacro),
  commit `15977219511b965bde26e55e89bd50090371fd07`, BSD-3-Clause
  (`LICENSE.picknik_example_ws`).
- D415 `d415.stl`, `_d415.urdf.xacro`, `_materials.urdf.xacro`:
  [RealSense ROS description](https://github.com/realsenseai/realsense-ros/tree/bafc21080c5c8e259dadbb309797949aee0dd950/realsense2_description),
  commit `bafc21080c5c8e259dadbb309797949aee0dd950`, Apache-2.0
  (`LICENSE.realsense`).

## Composition and representation

The adapter's published `child_tool` is **7 mm** beyond the input flange.
The existing Robotiq coupling is not a replaceable empty spacer: its local
mesh extends from -7 to +6.9 mm, and its original frame was already 7 mm past
the flange. Its lower face therefore touched the original flange. The camera
adapter is added before it; both complete gripper chains and their attach
frames move 7 mm along the tool axis. Arm home joint angles, bases, table
installation and third-person cameras are unchanged.

Each assembly is rigid geometry on its existing `wrist_3_link`. There are no
additional robot bodies, joints or control coordinates. The generated component
preserves 1,034 adapter visual triangles with the DAE's white/green material
split, all 186 collision triangles, and all **425,160 D415 visual triangles**.
The camera material comes from upstream `aluminum`; no lens colors/textures
are invented. The hidden D415 collision box uses the dimensions and offset
already defined by its official Xacro. It is not the visible camera model.

The camera chain is: flange -> adapter -> D415 bottom screw
`(0,-0.067,0.0171)`, RPY `(0,-pi/2+6deg,pi/2)` -> camera link
`(0,0.02,0.0115)` -> color frame `(0,0.015,0)` -> ROS optical rotation
`(-pi/2,0,-pi/2)` -> USD optical convention. Camera prim paths and semantic
roles remain unchanged, with parent-local transforms recomputed after the
gripper translation. Thus their world poses follow the wrist-mounted sensor,
not an arbitrary point inside the gripper.

The source simulation camera uses `horizontal_fov=1.047` radians at 640x480.
USD retains the old focal-length scale of 13 and derives both apertures from
that ratio; 13 is **not claimed as a measured physical D415 focal length**.
This replaces the old virtual-camera extrinsics/intrinsics. The pre-existing
rollout compatibility path for 4:3 benchmark cameras still treats horizontal
FOV as vertical; this separate known runtime limitation is not silently fixed
by changing the asset.

## Dynamics boundary

**The accessory payload is not included in the robot's mass or inertia.**
PickNik's adapter Xacro contains no inertial block. RealSense lists mass
0.072 kg, but explicitly warns that the following COM/inertia values are not
reliable and should not be used for modeling. We therefore do not merge that
tensor or guess bracket density. Accessory shapes have zero authored mass and
density, and all 42 existing body mass/COM/inertia values remain unchanged.
This is a complete visual/kinematic camera installation, not a fully calibrated
end-effector payload model or real-hardware mechanical certification.

## Offline rebuild

`python assets/robots/ur5e_2f85/build_wrist_cameras.py` reads only this source
directory and the fixed workstation payload. It writes the one internal
`components/wrist_camera.usdc` and updates `ur5e_2f85.usda`; it does not rebuild
the arm or download anything. Repeated builds anchor the 7 mm translation to
the unchanged payload rather than adding it repeatedly. The component and
large source meshes follow the external-asset deployment convention.
