# 开发手册索引

本文件保存开发时可直接查阅的官方手册、API 和固定源码入口。这里只做导航，不记录调研结论或实验经验。

查第三方行为时按以下顺序：

1. 从 `requirements.txt` 确认项目固定版本或 commit；
2. 查对应版本的 API 文档；
3. 文档未说明时查看同一 tag/commit 的源码；
4. 最后再参考 issue、PR 或非官方讨论。

## Newton

- [Newton stable 文档](https://newton-physics.github.io/newton/stable/)
- [Newton API reference](https://newton-physics.github.io/newton/stable/api/newton.html)
- [Worlds 概念](https://newton-physics.github.io/newton/stable/concepts/worlds.html)
- [Collision 与 contact](https://newton-physics.github.io/newton/latest/concepts/collisions.html)
- [Solver 指南](https://newton-physics.github.io/newton/stable/solvers/index.html)
- [可视化、录制和离线 viewer](https://newton-physics.github.io/newton/stable/guide/visualization.html)
- [Newton 官方源码](https://github.com/newton-physics/newton)
- [项目固定 Newton fork commit](https://github.com/zlong-911/newton/tree/363aab40133fa72d8536ec5e15802a466b1f6601)

项目固定的是 Newton 1.5.0 fork，不要使用 1.4 API 或 latest 文档中的未发布接口代替固定源码。

## Warp 与 CUDA

- [Warp 1.16 文档](https://nvidia.github.io/warp/v1.16/)
- [Warp 1.16 runtime、stream 与 CUDA Graph](https://nvidia.github.io/warp/v1.16/user_guide/runtime.html)
- [Warp API reference](https://nvidia.github.io/warp/stable/api_reference/warp.html)
- [Warp 官方源码](https://github.com/NVIDIA/warp)
- [CUDA Programming Guide：CUDA Graph](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [CUDA Runtime API：Graph Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NCCL：CUDA Graph](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)

## MuJoCo 与 OpenUSD

- [MuJoCo 3.11.0 USD 材质转换源码](https://github.com/google-deepmind/mujoco/blob/3.11.0/python/mujoco/usd/objects.py)
- [MuJoCo 3.11.0 Filament：源颜色的 sRGB 语义](https://github.com/google-deepmind/mujoco/blob/3.11.0/src/render/filament/core/material_manager.cc#L244)
- [MuJoCo 3.11.0 Filament：旧材质默认转换倍率](https://github.com/google-deepmind/mujoco/blob/3.11.0/src/render/filament/support/model_objects.h)
- [MuJoCo 3.11.0 Filament：specular-glossiness 材质](https://github.com/google-deepmind/mujoco/blob/3.11.0/src/render/filament/assets/phong_color.mat)

- [MuJoCo 文档](https://mujoco.readthedocs.io/)
- [MuJoCo Warp 文档](https://mujoco.readthedocs.io/en/latest/mjwarp/)
- [MuJoCo 官方源码](https://github.com/google-deepmind/mujoco)
- [MuJoCo Warp 官方源码](https://github.com/google-deepmind/mujoco_warp)
- [MuJoCo Menagerie 官方源码](https://github.com/google-deepmind/mujoco_menagerie)
- [MuJoCo Menagerie ALOHA 2 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/main/aloha)
- [项目固定 ALOHA 2 commit](https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/aloha)
- [MuJoCo Menagerie Franka Panda 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)
- [项目固定 Franka Panda commit](https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/franka_emika_panda)
- [MuJoCo Menagerie UR5e 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e)
- [项目固定 UR5e commit](https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/universal_robots_ur5e)
- [MuJoCo Menagerie Robotiq 2F-85 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85)
- [项目固定 Robotiq 2F-85 commit](https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/robotiq_2f85)
- [Trossen VX300s MJCF](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_vx300s)
- [ARX Robotics 产品与 OpenPI 双臂平台](https://www.arx-x.com/)
- [ARX 官方 X5 仓库](https://github.com/ARXroboticsX/ARX_X5)
- [项目固定 RoboDojo X5A 资产](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/cfb06d1dadcdf03ccc923102b0e3f3b0a68dfc43/Assets/Robots/x5)
- [项目固定 `arx5-sdk` X5 参数](https://github.com/real-stanford/arx5-sdk/tree/a8890c9bae94464abd1cb7c5e4da7c4a62104a3a)
- [同一 ARX.usd 的 RoboDojo 驱动配置](https://github.com/robodojo-benchmark/RoboDojo/blob/76e16bf8feb6e5934ddb291386bf531ff826c621/env/robot_manager/robot_config/x5.py)
- [Franka 官方 3D 资产](https://franka.de/3d-assets)
- [项目审计固定 Franka description](https://github.com/frankarobotics/franka_description/tree/02afaae282d4a8e10d7d2f781b23b3515c303ce5)
- [与 FR3 VMkit 碰撞网格一致的 PickNik Robotiq description](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/2c047340aeb2440f7a60e429264221aab9658707)
- [OpenUSD 文档](https://openusd.org/release/index.html)
- [OpenUSD API reference](https://openusd.org/release/api/index.html)
- [OpenUSD 官方源码](https://github.com/PixarAnimationStudios/OpenUSD)

## RealSense 相机

- [D435 官方 RGB/深度规格与安装接口](https://www.realsenseai.com/products/stereo-depth-camera-d435/)
- [RealSense ROS 4.55.1 D435 安装与名义光学坐标](https://github.com/realsenseai/realsense-ros/blob/4.55.1/realsense2_description/urdf/_d435.urdf.xacro)
- [同版本 D435 原始网格](https://github.com/realsenseai/realsense-ros/blob/4.55.1/realsense2_description/meshes/d435.dae)
- [librealsense v2.55.1 RGB 内参与画幅换算](https://github.com/realsenseai/librealsense/blob/v2.55.1/src/ds/d400/d400-private.cpp)
- [RealSense 投影与逐 stream profile 内参 API](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)

## RLINF、OpenPI 与 LeRobot

- [RLINF 官方源码与使用文档](https://github.com/RLinf/RLinf)
- [项目固定 RLINF commit](https://github.com/RLinf/RLinf/tree/7d07a4212ee6858cc333e1d4fab7a37256d1f839)
- [OpenPI 官方源码](https://github.com/Physical-Intelligence/openpi)
- [OpenPI JAX→PyTorch converter 源码](https://github.com/Physical-Intelligence/openpi/blob/main/examples/convert_jax_model_to_pytorch.py)
- [OpenPI FP32 conversion-intermediate 修复讨论](https://github.com/Physical-Intelligence/openpi/pull/978)
- [OpenPI fine-tuning 指南](https://github.com/Physical-Intelligence/openpi#fine-tuning-a-base-model-on-your-own-data)
- [OpenPI remote inference](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)
- [LeRobot 文档](https://huggingface.co/docs/lerobot/)
- [LeRobot dataset v3 手册](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [LeRobot 官方源码](https://github.com/huggingface/lerobot)
- [项目固定 LeRobot commit](https://github.com/huggingface/lerobot/tree/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5)
- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
- [Physical Intelligence RTC JAX reference](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- [LeRobot v0.4.2 RTCProcessor source](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/policies/rtc/modeling_rtc.py)

项目固定的 LeRobot commit 本身没有 RTC；数据 schema、writer 和视频行为看固定 commit，RTC 只以上述论文、作者代码和后续 LeRobot 实现为算法参考。

## PyTorch、JAX 与分布式训练

- [PyTorch 2.7 文档](https://pytorch.org/docs/2.7/)
- [PyTorch FSDP](https://pytorch.org/docs/2.7/fsdp.html)
- [PyTorch distributed](https://pytorch.org/docs/2.7/distributed.html)
- [JAX 文档](https://docs.jax.dev/)
- [JAX distributed arrays](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [Orbax checkpoint 文档](https://orbax.readthedocs.io/en/latest/)
- [W&B 文档](https://docs.wandb.ai/)

## Isaac、XR 与服务

- [Isaac Sim 6.0 文档](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/)
- [Isaac Sim 5.0 Python 环境安装](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html)
- [Isaac Sim 5.0 系统与 GPU 要求](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html)
- [Isaac Sim 5.0 Replicator snippets](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html)
- [Omniverse UsdPreviewSurface 材质](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/UsdPreviewSurface.html)
- [OpenUSD Exchange SDK 材质 API](https://docs.omniverse.nvidia.com/usd/code-docs/usd-exchange-sdk/latest/api/group__materials.html)
- [Isaac Teleop 官方文档](https://nvidia.github.io/IsaacTeleop/)
- [Isaac Teleop 官方源码](https://github.com/NVIDIA/IsaacTeleop)
- [LeRobot Isaac Teleop 指南](https://huggingface.co/docs/lerobot/main/isaac_teleop)
- [CloudXR SDK 文档](https://docs.nvidia.com/cloudxr-sdk/)
- [CloudXR network setup](https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Uvicorn 文档](https://www.uvicorn.org/)

## 数据、视频与配置

- [Poly Haven 资产库](https://polyhaven.com/)
- [Poly Haven CC0 许可](https://polyhaven.com/license)
- [Poly Haven 官方 API](https://polyhaven.com/our-api)
- [Poly Haven 纹理技术标准](https://docs.polyhaven.com/en/technical-standards/textures)
- [Poly Haven Blender 资产库源码](https://github.com/Poly-Haven/polyhavenassets)
- [NumPy 1.26 文档](https://numpy.org/doc/1.26/)
- [PyArrow 文档](https://arrow.apache.org/docs/python/)
- [PyAV 文档](https://pyav.org/docs/stable/)
- [FFmpeg 文档](https://ffmpeg.org/documentation.html)
- [PyYAML 文档](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Hydra 文档](https://hydra.cc/docs/intro/)
- [OmegaConf 文档](https://omegaconf.readthedocs.io/)

## Python 与仓库工具

- [Python 3.11 文档](https://docs.python.org/3.11/)
- [pytest 文档](https://docs.pytest.org/)
- [Ruff 文档](https://docs.astral.sh/ruff/)
- [Git reference](https://git-scm.com/docs)
- [ripgrep 使用手册](https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md)
- [fd 使用手册](https://github.com/sharkdp/fd#how-to-use)
- [jq 手册](https://jqlang.org/manual/)
- [ShellCheck 说明](https://www.shellcheck.net/)
- [CMake 文档](https://cmake.org/documentation/)
- [Ninja 手册](https://ninja-build.org/manual.html)
