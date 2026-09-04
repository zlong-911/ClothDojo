# 机器人动力学

固定本体参数由工作站 USD 持有；任务配置不重新定义质量、惯量或驱动。
来源入口见 [reference_index.md](reference_index.md)。几何、安装、home、attach、相机和外观不因动力学校对而改变。

## 当前来源

| 本体 | 惯性参数 | 控制与限制 |
| --- | --- | --- |
| Piper | 原 USD 的质量、COM、惯量 | 旧项目 Piper retarget 基线；补齐 3 rad/s 速度字段，不称为厂家标定 |
| ALOHA2 | 固定 Menagerie MJCF；20 个刚体与 MuJoCo 3.11 的完整惯量 tensor 对齐 | 保留同源驱动、effort、armature 与摩擦；双侧 elbow/forearm-roll 被动阻尼采用项目仿真调参，其余被动项保留 |
| X5 | RoboDojo 同一 ARX.usd，18 个机械链接与附带 URDF 对齐 | 同 USD 的 RoboDojo 配置：臂 kp=4400、kd=40、armature=0.01、effort=100、velocity=5；effort 是仿真上限，不是电机额定值 |
| UR5e + 2F85 | 固定 Menagerie；两个安装座采用原 MJCF 经 MuJoCo 3.11 按网格和默认密度推导的质量/COM/惯量 | 保留源 stiffness/effort 和夹爪参数；臂 drive 阻尼采用项目仿真调参；修正角度单位，恢复夹爪被动弹簧 stiffness=0.05、springref=2.62 rad |
| FR3 Duo | 官方 fr3v2 description；夹爪复用 UR5e 已接受的 Menagerie Robotiq 模型；保留 VMKit 相机和转接件 | 臂保留 stiffness、effort 和被动项，调整 drive 阻尼；夹爪统一为每侧两个 2.5 Nm drive、完整闭环和同源惯性/被动参数，不把仿真限制称为厂家额定值 |

FR3 固定快照入口为 `assets/robots/fr3_duo/fr3_duo.usda`，不再从 `outputs/` 加载。
它已注册为 `fr3_duo`，与其余四款通过同一 USD 导入、控制与空载检查入口运行。
双臂共用一棵 articulation；每侧夹爪两个受控坐标共用一个 0–1 command，
其余连杆由与 UR5e 相同的闭环连接约束。原 VMKit 单驱动 mimic 链已停用。
`components/robotiq_2f85.usdc` 保存同源夹爪，`components/robotiq_installation.usdc`
保存安装转换和完整 q/qd/body 布局；保留 FR3 原相机、转接件、TCP 与 attach 标定。
两侧 driver 均为 kp=50、kd=5、effort=2.5 Nm、armature=0.005；被动转动关节
armature=0.001，弹簧与其余限制照搬 UR5e，不在 retarget 内增加特例。
重型网格/纹理/payload 继续按项目约定在 Git 外部署；Git 保存入口、动力学覆盖及来源记录。

## 单位与重建

ALOHA 双侧 elbow 和 forearm-roll 被动阻尼分别为 `10`、`2 Nm·s/rad`，原值分别为
`18.49`、`6.78`。这是针对 1× effort 下持续阻尼负载导致跟踪受限的项目仿真调整，
不是修改同源质量、惯量或力矩上限，也不称为厂家参数纠错；夹爪驱动与约束不变。

UR5e 每侧臂 drive 阻尼为 `[120, 125, 95, 18, 16, 15] Nm·s/rad`。以采样姿态
惯量估计阻尼后，通过空载响应和 1× action rollout 对照选择；不是厂家标定或源参数纠错。
数值由 USD 持有，retarget 不另改增益。

FR3 每侧臂 drive 阻尼为 `[180,170,145,130,20,23,5] Nm·s/rad`，同样基于惯量估计
和实际响应对照选择。保留臂 stiffness、官方 effort 和被动项；改善控制响应不等于
解决任务参考的时间连续性，各条 retarget 仍须独立验收。

- 质量 kg、COM m、完整 body-local 惯量 kg·m²。主惯量与主轴必须重建到完整 tensor 后对照，不能只比较三个特征值。
- Newton 1.5 的 USD 导入将旋转关节的角速度乘 π/180，将角驱动/被动阻尼/限位增益除 π/180。写入 USD 时必须进行逆变换。
- `mjc:stiffness`、`mjc:springref` 属于求解器扩展，采用 SI；不要再次做角度转换。
- ALOHA 仍通过现有 `_restore_exported_joint_dynamics` 处理 Newton 导出层的 SI 字段；UR5 已在顶层 USD 修正。两者不可重复转换。
- X5 的 `build_workstation.py` 已同步修正；X5/UR5/Piper 顶层 USD 保存参数覆盖。FR3 的已接受装配保存为自包含 payload，顶层 USD 固定有据惯性与关节驱动/状态，来源 tensor 单列在 `provenance/dynamics.json`。

## 适用边界

- X5 上游运行配置关闭重力；本项目没有随之关闭重力。参数同源不代表运行条件相同。
- X5 四个相机/支架刚体、FR3 部分相机/转接件及空 frame 尚缺独立可信的惯性依据；保留已有值，没有猜填或宣称全部校准。
- UR5e 已加入实体 PickNik/D415 腕部相机，但附件负载尚未计入：转接件无完整惯性，RealSense 示例 tensor 明确不可靠；不猜填，原42个刚体惯性保留。
- 当前 `SolverMuJoCo` 不执行 `joint_velocity_limit` 硬限速；字段存在不代表已实现速度控制器。
- MuJoCo Warp inverse-dynamics 适配会解码 Newton 对 velocity target actuator 使用的负号 DOF 编码；这用于读取 Piper 从动夹指所在 DOF 的 generalized force，不把 velocity actuator 改成 position actuator，也不改变其 drive。
- 普通 runtime 与 retarget 均直接使用 USD effort 的 1×，包括 arm 和 gripper；不在 retarget 内另设放大比例。历史 2× 轨迹不代表当前 1× 参数下可行，需要重新求解和回放验收。
- 基础保持/运动/夹爪空载检查不替代含布料、自碰撞、桌面接触的集成验证，也不构成真机动力学或安装安全认证。

## 复验

```bash
PXR_WORK_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES='' .venv/bin/pytest tests/test_robot_dynamics.py -q
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/robots/check_dynamics.py fr3_duo
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/robots/check_dynamics.py x5 --device cuda:0 --contacts
```

`check_dynamics.py` 只读资产，默认做 8 秒空载保持、小幅运动、夹爪开合；GPU 接触检查需显式加 `--contacts`。
逐属性审计和动态探针结果写入 `outputs/robot_dynamics_audit/`，不是运行时依赖。当前 ALOHA 的完整接触检查曾超过求解器容量，不能把其空载结果视为接触验证通过。
