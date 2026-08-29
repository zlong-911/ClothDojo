# 机械臂 Retarget

Retarget 的公开边界是固定 robot profile，而不是一组零散参数。当前入口把 Piper
人工轨迹转换为 `aloha2` canonical trajectory collection；后续 Newton
重渲染、LeRobot 导出和 Isaac RTX 随机化都从目标轨迹的 `robot_id` 读取同一套
机械臂、关节投影、夹持语义和相机标定。当前正式目标为 ALOHA 2、ARX X5、Franka
Panda 与 UR5e + Robotiq 2F-85。ALOHA、Panda 和 UR5e/2F-85 使用同一固定 MuJoCo
Menagerie commit；X5 使用固定 RoboDojo X5A 模型，并以固定 `arx5-sdk` 真机参数替换
URDF 占位限位。

## 最短入口

```bash
.venv/bin/clothdojo data retarget \
  --source data/collections/shirt_flatten \
  --output data/retarget/shirt_flatten_aloha \
  --target-robot aloha2
```

retarget 默认只对齐左右 attach 中心。每次 attach 前的夹爪闭合阶段会额外对齐 Piper 与目标
机械臂 semantic attach frame 的接近轴；绕接近轴的旋转仍然自由。为避免闭合首帧才突然切换
姿态，系统会在闭合前增加一段三倍于源闭合时长的连续对齐过程，从目标机械臂原来的接近轴
平滑过渡到 Piper 接近轴。这整段是优化准备区，只有最后一个源闭合时长进入姿态硬门禁。
夹爪本身不生成额外的预闭合时序，而是直接映射 Piper 逐帧归一化开度。attach 成功后立即
恢复为纯位置目标，因此不会用完整 pose 约束后续布料操作。
该阶段完全由夹爪与 attach 事件推导，没有公开模式或固定秒数参数；最终释放后若不再抓取，
也不会施加接近轴约束。

Franka Panda 使用同一入口，只替换目标机器人：

```bash
.venv/bin/clothdojo data retarget \
  --source data/collections/shirt_flatten \
  --output data/retarget/shirt_flatten_panda \
  --target-robot panda
```

UR5e + Robotiq 2F-85 同样只替换目标机器人：

```bash
.venv/bin/clothdojo data retarget \
  --source data/collections/shirt_flatten \
  --output data/retarget/shirt_flatten_ur5e_2f85 \
  --target-robot ur5e_2f85
```

ARX X5 使用旧项目已经验证的双臂安装标定：

```bash
.venv/bin/clothdojo data retarget \
  --source data/collections/shirt_flatten \
  --output data/retarget/shirt_flatten_x5 \
  --target-robot x5
```

`--source` 可以是一条 canonical trajectory，也可以是带 `trajectories/` 的 collection。
正式训练使用 collection 输出。命令不会覆盖已有目录，失败时不会发布半成品。

目标 collection 可直接进入通用四视角 LeRobot 导出：

```bash
.venv/bin/clothdojo data export-lerobot \
  --source data/retarget/shirt_flatten_aloha \
  --output data/lerobot/shirt_flatten_aloha
```

同一入口添加 `--isaac-seed` 后改用 Isaac RTX，并按 episode 做确定性的材质、颜色和
光照随机化：

```bash
.venv/bin/clothdojo data export-lerobot \
  --source data/retarget/shirt_flatten_aloha \
  --output data/lerobot/shirt_flatten_aloha_isaac \
  --isaac-seed 17 \
  --randomize all
```

先用 `--plan-only` 检查 episode、frame、renderer 和 `training_eligible`，不会加载渲染器
或写数据。

## 数据语义

每条目标轨迹仍使用项目标准的 `manifest.json`、`states.npz` 和 `actions.npz`，普通
`TrajectoryReader`、LeRobot 导出和训练代码无需 retarget 专用 reader。目标机械臂的
`joint_q`、`body_q`、action 和夹爪状态存放在派生轨迹中；`particle_q` 与
`particle_qd` 透明读取源 Piper 轨迹。

派生 collection 因此依赖源轨迹的不可变路径和 manifest hash；归档或搬迁数据时必须同时
保留两者，reader 会拒绝读取被替换的源 manifest。

转换先用目标机械臂 FK/IK 生成逐帧几何参考，再用固定目标资产中的质量、惯量、关节阻尼、
actuator 增益和力矩限制执行刚体逆动力学与有界一步 shooting；retarget 环境统一把 arm 和
gripper actuator 的 effort limit 放宽到官方值的 2 倍，实际比例写入 validation。逆动力学只根据
IK 的 `q/qd` 产生 position
command seed；饱和 seed 会使用在整个 control frame 内保持同等饱和力矩的最小 target，
避免 physics substep 中途退出饱和，但对应关节仍可由后续 shooting 调整。

shooting 的 arm 物理目标只有逐帧 attach 中心和 pre-grasp 阶段的接近轴。IK
关节轨迹不进入逐帧 residual 或门禁，只通过很弱的 command-to-seed 正则为 position-only
任务选择连续的冗余姿态。因此实际关节轨迹可以偏离初始 IK，只要任务空间目标在同一 Newton
刚体环境中冷启动重放后仍成立。最终 `joint_q`、`joint_qd`、`body_q` 和 `body_qd` 都来自
这次 action-only 冷回放，不把下一帧 IK 状态冒充实际状态或 action。

源夹爪和目标夹爪都以 `0=机械全闭`、`1=机械全开` 表示归一化开度。retarget
逐帧保留这个开度，再由目标 robot profile 线性映射到其实际关节行程；attach/release
判定因此可以继续使用统一的归一化阈值。夹爪 action 使用目标 actuator 的增益、阻尼和
effort limit 反算，不参与 arm least-squares。

一步 shooting 的有限差分 Jacobian 会把基准命令和全部 arm command 扰动放入多个 Newton world
并行计算。批量化只合并相互独立的试算，不减少物理
substep、不改变源时间轴，也不复用前一候选的 solver 状态。

完整冷启动 replay 若仍未通过 arm 动力学门禁，只对最坏失败片段启用固定三步窗口：联合优化
三帧 arm command，并用弱 command seed 正则和相对修改前实际状态的末端连续性约束处理
动力学滞后。候选补丁必须降低 attach 中心与 pre-grasp 接近轴的归一化最坏值；已经失败的
指标也不得继续变差。窗口最多尝试三次，未被整条 action 冷回放接受的补丁会直接回滚。

retarget 不推进 Style3D 布料或机器人—布料接触。目标轨迹严格保留源轨迹的状态数、50 Hz
时间轴和逐帧布料状态，不插帧、减速、等待机械臂或重复布料帧。动力学 action 与对应的下一
状态位于同一控制边界，全程误差都会写入 validation。

manifest 中的门禁含义如下：

- `kinematic_valid`：几何参考的目标夹持中心相对语义目标的最大误差不超过 5 mm，pre-grasp
  阶段的接近轴误差不超过 0.05 rad；
- `dynamics_valid`：action-only 冷回放的所有帧 attach 中心误差不超过 1 cm、每次 attach
  transition 的夹爪开度已穿过统一的 0.1 attach 阈值，临近抓取的最后一个源闭合时长内
  接近轴误差不超过 0.05 rad，并且重复冷回放
  能复现发布状态；相对
  IK seed 的关节位置、速度差异只保留为诊断，不作为门禁，因为 IK 冗余姿态不是目标机械臂
  必须复刻的物理真值；
- `source_timeline_preserved`：输出状态与源轨迹逐帧对齐，没有 retiming；
- `collision_checked=false`：没有把刚体反事实回放误报成目标机器人—布料物理 rollout；
- collection 的 `training_eligible` 仅在所有轨迹的几何与动力学门禁都通过时为 true。

## Robot 与 task

机械臂选择与跨后端语义入口位于 `src/clothdojo/robots.py`。一个 `RobotProfile` 使用
固定 `robot_id`，同时拥有：

- Newton 运动学资产、Newton/Isaac 共用可视资产及固定安装架；
- 每个独立 articulation 的世界安装位姿；
- profile-sized 双臂投影、独立的 position-coordinate/velocity-DOF 映射、夹爪范围及
  home pose；
- 左右语义 attach frame；
- `cam_high`、`cam_low` 和左右腕部相机的 parent、外参和内参；
- Fabric replay 使用的刚体语义拓扑。

Piper 已采用完整工作站 USD：`assets/robots/piper/piper.usda` 组合双臂与安装结构，并直接
定义关节限位、position drive、armature、夹爪初态和 mimic、MuJoCo 重力补偿、碰撞接触
参数、左右 attach frame 及四个真实 `Camera` prim。Python 只选择关节组、解释资产已标注的
碰撞角色，并把 USD 语义适配到 Newton/Isaac；不再保存 Piper 的可调物理参数或相机矩阵。
任务配置继续拥有桌面、服装资产、布料参数、任务随机化和求解步进，不反向修改机器人本体。
Newton 1.5 会把 `ke=0, kd=80` 的从动夹指 drive 推断为 velocity mode；它与旧代码强制的
position mode 都产生零目标速度阻尼。非零夹指速度的一步 MuJoCo CPU 对照中，两者的下一步
`joint_q` 和 `joint_qd` 最大差值均为 0。

外部 MJCF 目标机器人的安装位姿只有 robot profile 一个事实来源；Piper 双臂的
相对安装已经固化在工作站 USD 中，profile 只使用整体 identity frame。Newton 按该定义建立
运动学模型，Isaac 按同一定义建立可视资产的父节点；replay bridge 把 Newton 的刚体世界变换转换为层级局部
变换，通过 USD xform 发布给 RTX，布料顶点则继续通过 Fabric 在 GPU 上更新。RTX 导出前
还会校验轨迹根刚体与 profile 一致，不一致时直接失败，避免同一份数值分别散落在 Newton
scene、USD stage 和 renderer 中。

当前目标资产来自固定 commit 的官方 MuJoCo Menagerie ALOHA 2。ClothDojo 保留现有
`1.262 × 1.4 m` 工作台、布料、地面与灯光，从官方 `scene.xml` 提取 30 个铝型材、角码、
相机支架和 D405 外壳作为 `frame.xml`；不导入官方桌面、桌腿或地面。官方原桌位于
桌面边缘的一根横梁，在 ClothDojo 更深的工作台上会落到布料操作区中部，因此不导入。
安装架是固定在当前
工作台上的 robot visual，不参与布料或刚体接触。ALOHA 2 资产内部的双臂间距与朝向
保持官方定义，robot profile 对双臂和安装架共同施加一次平移和绕 Z 轴 -90° 的安装变换，使它
与本项目工作台坐标系和 Piper 轨迹工作区对齐。最终左右基座分别为
`(0.18, 0.469, 0.82)` 和 `(0.18, -0.469, 0.82)`：宽度和朝向不变，沿桌面长度从中心
移到接近 Piper 的一端，同时保留当前轨迹的 5 mm 可达性门禁。

四路相机也属于同一 robot profile。`cam_high` 与 `cam_low` 分别使用安装架上的官方
`overhead_cam` 和 `worms_eye_cam` 位姿，左右腕部使用官方 D405 位姿；四路都采用官方
`1.93 mm / 3.896 mm` 水平成像参数。第四视角统一使用训练数据既有的 `cam_low` 键，
不另设 `cam_third_person`。相机外参与画面中可见的实体支架因此来自同一个安装事实。

任务选择位于 `src/clothdojo/tasks.py`，与机器人完全独立。`shirt_flatten` 已有场景资产与
随机化入口；`shirt_fold` 目前只有稳定身份，场景构建会明确拒绝执行，直到它自己的资产和
随机化契约完成。`build_scene(..., robot_id=..., task_name=...)` 是两者组合的统一入口。

轨迹只保存固定的 `robot_id` 和 `task_name`，不保存 robot profile hash。debug 阶段直接修改
同名 profile；流程成熟后把整套资产与定义固定下来。旧轨迹中的组合 `environment_id` 仅在
读取边界映射到 `robot_id`，新 manifest 不再写出。新增机械臂时应新增一个固定名称的 profile
和对应工作站资产，再为 retarget 求解器实现该机械臂的 IK/FK；不向公开 CLI 添加关节索引或
相机矩阵参数。

## 能力边界

当前源机器人是 `piper`，目标机器人是 `aloha2`、`x5`、`panda` 或 `ur5e_2f85`。
Newton 仿真与 Isaac 渲染都读取同一套固定工作站 USD；retarget/逆动力学仍可使用固定
MJCF 计算模型，但 MJCF 不再是第二套视觉真值。Isaac 只做目标状态回放与高保真外观
随机化，不会重新运行布料物理。

资产选择优先使用上游发布的完整双臂装配，而不是只看两个单臂型号。ALOHA 2 因此直接
保留 Menagerie 的整套双臂与安装架。ARX 已销售名为 OpenPI 的双臂数据采集整机，公开配置
包含双 6-DoF follower、leader arms、顶部和腕部相机；但截至当前固定版本，厂商公开仓库
只有两台 X5 的 bimanual 控制入口和单臂 URDF，没有整机底座 mesh、统一 URDF/MJCF 或基座
相对尺寸。当前 X5 profile 因而不能凭产品图片构造“官方套件”，仍将两台标准单臂
安装在 Kai0 CAD 测量位置 `(0.25, ±0.28625912, 0.81) m`。一旦厂商发布 OpenPI 整机描述，应在 debug
阶段更新同名 profile，并在整套流程验证成熟后统一固定。

X5 的完整双指和腕部相机几何来自固定 RoboDojo X5A；六轴真实位置限位来自固定
`real-stanford/arx5-sdk`。RoboDojo URDF 的 `joint1..5` 使用 `[-10, 10] rad` 占位值，不能
直接用于 IK。工作站 USD 左右各引用一份 RoboDojo 官方 `ARX.usd`，不再从 MJCF 生成
机械臂或将腕部 camera GLB 降面；`camera_base`、完整相机外壳及两个 fixed joint 均保留
上游定义。双臂安装位、固定底座 joint、末端 attach frame、drive、夹爪 home 和数据集相机
语义是 ClothDojo 的工作站覆盖层，不冒充 ARX 官方整机尺寸。MuJoCo 3.4 不能读取源 camera
GLB，因此 `x5.xml` 只用于 retarget/动力学，不能作为 Isaac 工作站的视觉真值。

Panda 保留 Menagerie 的官方几何、惯量、关节限位、equality constraint 和 tendon。
标准资产描述单臂，因此双臂在 robot profile 中分别安装到 `(0.25, ±0.60, 0.8) m`；该安装
位姿、语义 attach frame 与四路相机属于 ClothDojo，而不冒充 Franka 官方
工作站。Menagerie 使用 affine arm actuator 与 tendon gripper，Newton 1.5 不能直接把它们
写入 `joint_target_q`；profile 因此保存由官方 actuator 方程严格展开得到的等价标量 position
drive。上游 actuator metadata 在 Newton scene 中被替换，避免原 actuator 与等价 drive
重复施力；Isaac 外观仍读取未修改的上游 MJCF。

UR5e 与 Robotiq 2F-85 分别保留 Menagerie 的标准单臂和夹爪模型。项目固定的组合 MJCF
只通过 MuJoCo `MjSpec.attach` 把 2F-85 挂到 UR5e 的官方 `attachment_site`，并把
`MjSpec.to_xml()` 缩写的 `solref` 恢复成上游显式向量；双臂安装、home pose、attach frame
和相机仍只属于 robot profile。2F-85 的四杆机构包含两个 `connect` equality constraint，
Newton 将其表示成额外 quaternion position coordinates，而 MuJoCo velocity layout 不包含
这些辅助坐标，因此 profile 分别保存 `q` 与 `qd/effort` 索引，避免用 position index 访问
actuator 或 inverse-dynamics DOF。标准 tendon actuator 同样展开成等价标量 position drive，
但连杆、equality constraint 与 tendon 约束保持上游控制语义。

ALOHA 2 的归一化夹爪状态使用当前固定 visual mesh 的无穿透范围：`0` 对应两指关节
`(0.0079, 0.0079) m`，`1` 对应 `(0.037, 0.037) m`。官方 MJCF actuator 的软件下限
虽然是 `0.002 m`，但在该位置左右 custom-finger mesh 会相互穿透约 `11.65 mm`；mesh
刚好分离的位置约为 `0.00785 m`，因此闭合位保留约 `0.11 mm` 几何余量。两个坐标都为正
是因为左右滑块轴分别定义在各自局部坐标系中，并不表示两指向同一方向运动。Newton、Isaac
和 robot profile 使用同一组数值，避免训练动作、回放画面和部署命令各自采用不同的
开合范围。

通过 `training_eligible` 门禁的数据可以用于视觉策略训练、目标机器人实际状态监督和
position-control action 监督；它证明的是目标机械臂刚体 action 在固定仿真中的可重放性，
不能证明机器人—布料接触动力学成立或真机安全。训练后策略的目标环境物理 rollout 与真机
门禁是独立阶段；在 Style3D 行为被验证前，不把 retarget 扩展成隐式布料物理 rollout。
