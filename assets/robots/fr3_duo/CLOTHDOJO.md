# FR3 Duo accepted workstation

`fr3_duo.usda` 是固定资产入口，包含校对后的惯性、关节控制/限制和 home 状态覆盖。
`payloads/workstation.usdc` 是已接受桌面装配的完整合成快照，含官方 VMkit、镜像夹爪初态、
原生头部相机、桌面安装件、attach 和材质；按项目已有资产约定在 Git 外单独管理。
打开资产需要先部署完整 payload，不能只下载 Git 中的入口文件。

本入口已经脱离 `outputs/`、旧项目绝对路径和官方源 USD 的运行时引用。
它尚未加入 `RobotProfile` 注册表；本轮没有宣称完成 fold/flatten 的正式运行接入。
旧 `outputs/fr3_duo_migration/` 是迁移过程与审计备份，不再是当前参数入口。

## 来源

- [Franka 官方 FR3Duo_withVMkit.usd](https://franka.de/3d-assets)：本体与完整腕部/头部套件几何。
- [固定 franka_description](https://github.com/frankarobotics/franka_description/tree/02afaae282d4a8e10d7d2f781b23b3515c303ce5)：fr3v2 惯性、限位、被动阻尼/摩擦，以及支座/头部惯性。
- [固定 PickNik Robotiq description](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/2c047340aeb2440f7a60e429264221aab9658707)：十类碰撞网格在 body-local 中逐顶点匹配的夹爪/转接件惯性。
- `provenance/dynamics.json`：39 个有据 body 的质量、COM、完整 tensor 和关节参数；其余63个未独立核定 body 明确单列。

控制增益来自旧项目调试基线，不是厂家标定。相机附件等缺口未猜填；固定 frame 也未通过
虚构质量“补齐”。详见 `docs/robot_dynamics.md`。不构成真机机械安装或动力学认证。

## 复验

```bash
PXR_WORK_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES='' .venv/bin/pytest tests/test_robot_dynamics.py -q
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/robots/check_dynamics.py fr3_duo
```

复验只读取资产；结果写入 `outputs/robot_dynamics_audit/`，不会重新拼接或覆盖已接受资产。
