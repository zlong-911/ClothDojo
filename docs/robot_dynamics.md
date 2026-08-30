# 机器人动力学

固定本体参数由工作站 USD 持有；任务配置不重新定义质量、惯量或驱动。
来源入口见 [reference_index.md](reference_index.md)。几何、安装、home、attach、相机和外观不因动力学校对而改变。

## 当前来源

| 本体 | 惯性参数 | 控制与限制 |
| --- | --- | --- |
| Piper | 原 USD 的质量、COM、惯量 | 旧项目 Piper retarget 基线；补齐 3 rad/s 速度字段，不称为厂家标定 |
| ALOHA2 | 固定 Menagerie MJCF；20 个刚体与 MuJoCo 3.11 的完整惯量 tensor 对齐 | 保留同源驱动、armature、被动阻尼与摩擦 |
| X5 | RoboDojo 同一 ARX.usd，18 个机械链接与附带 URDF 对齐 | 同 USD 的 RoboDojo 配置：臂 kp=4400、kd=40、armature=0.01、effort=100、velocity=5；effort 是仿真上限，不是电机额定值 |
| UR5e + 2F85 | 固定 Menagerie；两个安装座采用原 MJCF 经 MuJoCo 3.11 按网格和默认密度推导的质量/COM/惯量 | 保留源驱动；修正角度单位，恢复夹爪被动弹簧 stiffness=0.05、springref=2.62 rad |
| FR3 Duo | 官方 fr3v2 description；VMkit Robotiq 使用九类碰撞网格逐顶点一致的 PickNik 模型 | 臂采用旧项目调试 kp/kd，并纠正 USD 角度单位；官方关节限制与被动阻尼有独立来源，不把调试增益称为官方增益 |

FR3 固定快照入口为 `assets/robots/fr3_duo/fr3_duo.usda`，不再从 `outputs/` 加载。
它已注册为 `fr3_duo`，与其余四款通过同一 USD 导入、控制与空载检查入口运行。
双臂共用一棵 articulation；左右夹爪各一个受控坐标，其余连杆由原生 mimic 约束驱动。
重型网格/纹理/payload 继续按项目约定在 Git 外部署；Git 保存入口、动力学覆盖及来源记录。

## 单位与重建

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
- 普通 runtime 使用 USD effort 的 1×。现有 retarget 默认 2×，且同时影响 arm 和 gripper；本轮未修改。2× 下可行不保证 1× 下可行。
- 基础保持/运动/夹爪空载检查不替代含布料、自碰撞、桌面接触的集成验证，也不构成真机动力学或安装安全认证。

## 复验

```bash
PXR_WORK_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES='' .venv/bin/pytest tests/test_robot_dynamics.py -q
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/robots/check_dynamics.py fr3_duo
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/robots/check_dynamics.py x5 --device cuda:0 --contacts
```

`check_dynamics.py` 只读资产，默认做 8 秒空载保持、小幅运动、夹爪开合；GPU 接触检查需显式加 `--contacts`。
逐属性审计和动态探针结果写入 `outputs/robot_dynamics_audit/`，不是运行时依赖。当前 ALOHA 的完整接触检查曾超过求解器容量，不能把其空载结果视为接触验证通过。
