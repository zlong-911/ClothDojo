# FR3 Duo fixed workstation

`fr3_duo.usda` 是固定资产入口，包含校对后的惯性、关节控制/限制和 home 状态覆盖。
`payloads/workstation.usdc` 是已接受桌面装配的完整合成快照，含官方 VMkit、镜像夹爪初态、
原生头部相机、桌面安装件、attach 和材质；按项目已有资产约定在 Git 外单独管理。
当前入口恢复该快照的桌边安装位置，并保留同源 Robotiq 覆盖，不重建或改写 payload。
打开资产需要先部署完整 payload，不能只下载 Git 中的入口文件。

本入口已经脱离 `outputs/`、旧项目绝对路径和官方源 USD 的运行时引用。
它以 `fr3_duo` 加入 `RobotProfile`、Newton 控制与 retarget，替代已停用的 `panda`。
保留官方头部和双腕三路相机、原始 aperture 及镜像夹爪 home。
另外按用户要求加入独立虚拟观察相机 `/Robot/cameras/cam_low`，世界 pose、投影与裁剪参数
与当前 Piper 的 `cam_low` 相同；不增加实体相机或支架，也不作为官方 VMkit 硬件的一部分。
相机定义直接固定在 `fr3_duo.usda`，加载时不引用 Piper 或生成脚本；四路 LeRobot
相机集合为 `cam_high`、`cam_low`、`cam_left_wrist`、`cam_right_wrist`。
任务集成仍需要逐轨迹门禁，`shirt_fold` 场景尚未实现。
旧 `outputs/fr3_duo_migration/` 是迁移过程与审计备份，不再是当前参数入口。

## 当前安装

整套本体及原生相机回到原 payload 的桌边位置；共享安装根为 `(0.13, 0, 0.95)` m，
左右 FR3 基座为 `(0.167, ±0.05018, 1.00)` m。120×80 mm 等截面型材高 150 mm，
从桌面 Z=0.80 m 支撑到 Z=0.95 m；型材后缘距桌后沿 X=0.02 m 约 50.5 mm。
顶部六枚连接螺栓和原穿桌底部六枚螺栓、六枚垫圈均启用。安装按固定约束建模，
不构成桌板承载、紧固件强度或全轨迹避碰认证。
虚拟 `cam_low` 保持原世界位姿。安装位置调整不改变 home、attach 局部标定、
手臂动力学或当前 Robotiq 模型；夹爪的独立替换见下节。
审查仍使用与任务一致的简化无孔桌面；旧孔板 STL 保留作安装几何参考，不改任务碰撞体。

## 来源

- [Franka 官方 FR3Duo_withVMkit.usd](https://franka.de/3d-assets)：本体与完整腕部/头部套件几何。
- [固定 franka_description](https://github.com/frankarobotics/franka_description/tree/02afaae282d4a8e10d7d2f781b23b3515c303ce5)：fr3v2 惯性、限位、被动阻尼/摩擦，以及支座/头部惯性。
- [固定 Menagerie Robotiq](https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/robotiq_2f85)：夹爪采用项目 UR5e 已接受 USD 的完整模型，包含几何、惯性、双驱动、闭环和被动参数；许可证随 `components/LICENSE.robotiq_2f85` 部署。
- [固定 PickNik description](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/2c047340aeb2440f7a60e429264221aab9658707)：保留转接件的已有惯性来源；不再作为当前活动夹爪的来源。
- `provenance/dynamics.json`：47 个有据 body 的质量、COM、完整 tensor 和关节参数；其余63个未独立核定 body 明确单列。

臂 stiffness 保留旧项目调试基线，drive 阻尼经项目 1× effort 响应对照调整为每侧
`[180,170,145,130,20,23,5] Nm·s/rad`；臂官方 effort 限制和被动项不变。
阻尼调整改善了空载响应，不表示任意 retarget 轨迹已通过，尤其不能解决不连续的姿态参考。
控制增益不是厂家标定。相机附件等缺口未猜填；固定 frame 也未通过
虚构质量“补齐”。详见 `docs/robot_dynamics.md`。不构成真机机械安装或动力学认证。

## 统一 Robotiq 夹爪

`components/robotiq_2f85.usdc` 是 UR5e 夹爪的独立快照；不引用 UR5e 工作站运行时。
`components/robotiq_installation.usdc` 停用原 18 个夹爪 body 和原 mimic 链，挂载
26 个同源 body，并保留原 VMKit 转接件、D405、TCP 与 attach。安装绕原底座
Z 固定转 -90°，补偿两套模型 X/Y 开合轴定义差异，不额外平移。
每侧两个 position drive 同受一个 0–1 action 控制，kp=50、kd=5、各限 2.5 Nm；
driver armature=0.005，其余转动关节 armature=0.001，弹簧及闭环约束与 UR5e 一致。
这些是项目仿真模型参数，不是厂家电机额定值或真实夹持力认证。

当前完整 state 为 46 个 q、42 个 DOF、110 个 body；action 仍为 14 臂关节目标加
2 个开合量。旧 26 坐标 FR3 轨迹不能直接用于新资产，必须重新 retarget。
离线转换入口为 `build_robotiq.py`；运行时直接读取固定 USD，不运行拼装脚本。

## 复验

```bash
.venv/bin/clothdojo sim render-workstations --robot fr3_duo --egl-device 1
PXR_WORK_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES='' .venv/bin/pytest tests/test_robot_dynamics.py -q
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/robots/check_dynamics.py fr3_duo
```

复验只读取资产；结果写入 `outputs/robot_dynamics_audit/`，不会重新拼接或覆盖已接受资产。
