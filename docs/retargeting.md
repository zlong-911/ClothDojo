# 机械臂 Retarget

Retarget 的公开边界是固定 robot profile，而不是一组零散参数。当前入口把 Piper
人工轨迹转换为 ALOHA2、X5、FR3 Duo 或 UR5e + 2F85 的 canonical trajectory collection；后续 Newton
重渲染、LeRobot 导出和 Isaac RTX 随机化都从目标轨迹的 `robot_id` 读取同一套
机械臂、关节投影、夹持语义和相机标定。后续本体选型为 ALOHA 2、ARX X5、FR3 Duo
与 UR5e + Robotiq 2F-85；FR3 Duo 已选定替代 Panda，Panda 不再用于新任务。
FR3 Duo 已以 `fr3_duo` 接入固定工作站与 retarget；旧 Panda 不再注册或接受为新目标。
四个目标均从固定工作站 USD 读取本体与驱动，动力学来源及运行边界见
[robot_dynamics.md](robot_dynamics.md)。入口检查实际 USD 引用闭包和 runtime 语义，
不再把未使用的旧 MJCF hash 当作资产门禁；输出记录实际 USD 路径，不冒充旧 XML 指纹。

## 最短入口

```bash
.venv/bin/clothdojo data retarget \
  --source data/collections/shirt_flatten \
  --output data/retarget/shirt_flatten_aloha \
  --target-robot aloha2
```

retarget 全程只约束左右 attach 中心的 position，包括抓取前和 attach 后，不追踪源轨迹的
朝向。几何参考会尝试绕工具当前局部 Z 轴小步旋转，让 +Y 尽可能朝世界上方；这只是服从位置
精度与关节连续性的软偏好，达不到不判失败。夹爪参考仍直接映射 Piper 逐帧归一化开度，
attach/release 事件规则与源时间轴不变，没有额外的抓取前姿态阶段或公开模式开关。

FR3 Duo 使用 `--target-robot fr3_duo`，保留已确认的镜像夹爪 home 和官方三相机。
双臂共用一棵 articulation，夹爪主动坐标和被动连杆约束直接读取当前工作站 profile。
来源记录固定 USD/payload 与惯性审计，不添加伪 MJCF 字段或新的资产 hash。
当前 FR3 已有独立虚拟 `cam_low`，满足四路 LeRobot 导出布局；不复制相机凑数。
控制布局已验证，但仍必须逐条通过 retarget 与接触门禁，不能据接入推断任务成功。

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
源轨迹必须至少包含一个 action；零动作记录会在资产/GPU 初始化前报错并列出路径，
不会静默跳过。输入位置容差必须是有限正数。

批量转换可显式指定多张空闲 GPU：

```bash
.venv/bin/clothdojo data retarget \
  --source data/collections/shirt_flatten \
  --output data/retarget/shirt_flatten_fr3 \
  --target-robot fr3_duo --devices cuda:0 cuda:1
```

`--devices` 与单卡 `--device` 互斥。默认每张卡一个独立进程，
`--workers-per-device N` 可以增加每卡常驻进程数；任务少时只创建实际需要的进程，
优先覆盖所有指定 GPU。按轨迹 action 数近似均衡工作量，并复用 IK 与刚体 solver；
同一 worker 内按源控制频率、substep 数分组复用，
不把不同源时间步强制转换成相同时间步。轨迹之间并行，单条内部始终按 action → 下一 state
顺序执行。最终 collection 保持源轨迹顺序，每条完成后输出耗时；所有 worker 成功后才发布
完整目录。源路径必须保持不可变，失败不会覆盖已有产物。

同卡多进程应配合 NVIDIA MPS，使不同 CUDA context 的小规模 IK/刚体内核可以重叠执行。
仅增加进程数而不开 MPS，可能主要变成 context 之间轮流运行。MPS 不改变 timestep、
solver、控制器或验收规则，也不会使单条轨迹内部并行。
它是外部 GPU 运行配置，retarget 不自动启动或关闭服务器、不改变 GPU compute mode。
已有 MPS 会话时直接继承它的 `CUDA_MPS_PIPE_DIRECTORY`；否则可在确认目标 GPU 无未知
compute process 后，为本次批处理创建独立会话。例如下面的 UUID 必须替换为目标卡：

```bash
(
  export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-VERIFIED-IDLE-GPU-UUID
  export CUDA_MPS_PIPE_DIRECTORY="$(mktemp -d /tmp/clothdojo-retarget-mps.XXXXXX)"
  export CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY"
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  nvidia-cuda-mps-control -d || exit
  trap 'printf "quit\n" | nvidia-cuda-mps-control' EXIT
  .venv/bin/clothdojo data retarget \
    --source data/collections/shirt_flatten \
    --output data/retarget/shirt_flatten_fr3 \
    --target-robot fr3_duo --device cuda:0 --workers-per-device 8
)
```

UUID 限制同时作用于服务器和子进程；命令中的 `cuda:0` 是限制可见设备后的逻辑编号。
多卡时在 `CUDA_VISIBLE_DEVICES` 中列出相应 UUID，并传 `--devices cuda:0 cuda:1 ...`。
退出清理只关闭这个独立 pipe directory 对应的 MPS，不操作其他作业的服务器。
进程数会增加 CPU 和显存占用，应按同卡整批吞吐选择；8 路是可测试的起点，
不是所有机器通用的最优值。MPS 只提高跨轨迹吞吐，不承诺降低单条完成延迟。

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
actuator 增益和力矩限制执行刚体逆动力学与有界一步 shooting；retarget 与普通 runtime
均直接使用 USD 的 arm 和 gripper effort limit（1×），不在求解、shooting 或冷回放中额外放大。
驱动调整由资产持有，validation 记录比例 1.0；并非所有本体的 USD effort 都是厂家额定值。逆动力学只根据
IK 的 `q/qd` 产生 position
command seed；饱和 seed 会使用在整个 control frame 内保持同等饱和力矩的最小 target，
避免 physics substep 中途退出饱和，但对应关节仍可由后续 shooting 调整。
臂的一步 shooting 以预测边界运动对应的正/负饱和 target 限定数值搜索区间，避免高刚度
驱动的大段饱和区让有限差分失去响应；该区间始终位于原关节指令限制内，不提高力矩上限。
该反事实只验证自由空间 actuator 动力学，因此关闭机器人与刚体、粒子 shape 的碰撞；质量、
惯量、关节被动项、drive 和 effort limit 保持启用。它不会把桌面或机器人自碰撞误报为已验证，
对应 manifest 仍明确写 `collision_checked=false`。

所有目标机器人共用同一套任务规则，不按 robot ID 分配姿态策略。每帧先从上一帧几何解出发
求 position-only IK，再试一次绕该解局部 Z 的小步 roll。理想 Y 是世界向上在垂直于当前 Z
平面上的归一化投影；Z 距竖直约 6° 内不扶正，约 6–12° 间平滑恢复扶正，避免退化投影翻转。
roll 请求角度和它额外引起的每个关节增量均按 1 rad/s × 控制帧时长限幅；这是偏好变化率，
不是提高 actuator 速度或力矩限制。候选再用 position-only IK 投影回位置目标，逐手检查
位置误差没有明显增加（最多 10 μm 且仍在几何容差内）、关节增量未超限、Y 高度不降低，
且当前 Z 方向向量变化不超过 0.001。未满足就保留最初的位置解，不强行追姿态。

所有目标的 dynamics shooting、窗口修复和验收均不约束朝向。持久化的 `pregrasp_mask` 与
`approach_validation_mask` 全为 false；它们不替代源 `grasp_attached` 事件标记。IK
关节轨迹不进入逐帧 residual 或门禁，只通过很弱的 command-to-seed 正则为 position-only
任务选择连续的冗余姿态。因此实际关节轨迹可以偏离初始 IK，只要任务空间目标在同一 Newton
刚体环境中冷启动重放后仍成立。最终 `joint_q`、`joint_qd`、`body_q` 和 `body_qd` 都来自
这次 action-only 冷回放，不把下一帧 IK 状态冒充实际状态或 action。

源夹爪和目标夹爪都以 `0=机械全闭`、`1=机械全开` 表示归一化开度。retarget
先将这个开度作为参考，由目标 robot profile 线性映射到其实际关节行程；attach/release
判定使用与 `ParticleGrasp` runtime 一致的主动指钳开度及端点修正。夹爪 action 使用目标 actuator 的增益、阻尼和
effort limit 反算 position command seed，再使用与 arm 相同的有界一步 shooting，
使下一控制边界的实际开度跟踪源开度。先求 arm command，再固定它求夹爪 command，避免
优化器为了改善受限的夹爪误差而晃动机械臂。随后固定 arm action，对源 `grasp_attached` 的每次
attach/release 转换执行事件预览前馈：在上一次同侧事件之后搜索夹爪 command 曲线的提前或延后量，
用真实刚体试算选择使主动指钳在目标帧首次跨过 runtime 阈值的命令。位移允许小数帧，插值的是
command，不是源布料或输出 state；不会直接把夹爪状态改写成目标开度。最后从初始状态整条冷回放，
重新验证事件序列及机械臂误差。全程开度曲线不要求逐帧相同，事件时序才是夹爪验收目标。

ALOHA 工作站 USD 保留原始 MJCF 的双指等式约束：每侧只有一个
主动驱动，另一指通过约束跟随。发布开度仍使用双指平均值；双指位置、速度的冷回放一致性
均参与验收，开度跟踪和双指同步误差继续保留为诊断，但从动指钳不独立决定 attach/release。

当前夹爪 action 仍遵循 `gripper_open_command ∈ [0, 1]`，因此动力学补偿后的 drive target
也受 profile 开合行程限制。事件预览不修改行程、增益、阻尼或 effort limit；如果仍不能在源
事件帧达到阈值，动力学门禁会标记为不合格，不延后源事件或伪造实际状态。

事件预览和验收共用主动指钳信号：将未经修正的归一化开度 `u` 按 runtime 的 2.5% 端点余量
换算为 `clip((u - 0.025) / 0.95, 0, 1)`，再重算带滞回的阈值状态：由 `>=0.1` 降至 `<0.1` 为 attach，
已 attach 时升至 `>0.7` 为 release，并与源标记逐帧比较，拒绝提前、延迟、漏触发和多触发。
相当于原始主动指钳开度的 `0.12 / 0.69` 阈值。初帧沿用源 attachment 状态，源事件帧和
零帧时序容差保持不变。`gripper_event_validation` 标明信号来源、margin 和控制帧采样范围，
逐侧记录 `source_frames` 与 `runtime_frames`；转换帧开度指标也使用这一修正后的主动指钳信号。
这只证明控制采样边界上的开度阈值事件对齐，不证明物理 substep 内的精确触发时刻或布料捕获。
输出的 `grasp_attached` 仍是源回放标记，不是目标夹爪重新捕获粒子的结果。
历史逐指钳、未经端点修正的门禁产物需要重新规划和验收，不能仅更新通过标记。

一步 shooting 的有限差分 Jacobian 会把基准命令和当前求解的 command 扰动放入多个 Newton world
并行计算。批量化只合并相互独立的试算，不减少物理
substep、不改变源时间轴，也不复用前一候选的 solver 状态。

完整冷启动 replay 若仍未通过 arm 动力学门禁，只对最坏失败片段启用固定三步窗口：联合优化
三帧 arm command，并用弱 command seed 正则和相对修改前实际状态的末端连续性约束处理
动力学滞后。候选补丁必须降低 attach 中心位置误差的归一化最坏值。
窗口最多尝试三次，未被整条 action
冷回放接受的补丁会直接回滚。夹爪事件预览在 arm 窗口修正完成之后执行。

retarget 不推进 Style3D 布料或机器人—布料接触。目标轨迹严格保留源轨迹的状态数、50 Hz
时间轴和逐帧布料状态，不插帧、减速、等待机械臂或重复布料帧。动力学 action 与对应的下一
状态位于同一控制边界，全程误差都会写入 validation。

刚体 retarget 场景不加载布料网格、不分配布料粒子，也不创建 Style3D 或粒子 grasp solver。
几何参考的最终 FK 结果复用于指标计算；逐帧反馈只保留当前关节 q/qd，
候选试算只回读 residual 所需字段，完整发布状态仍由独立 action-only replay 生成。
源布料继续由 trajectory overlay 回放。CUDA 上的单步 rollout、候选 shooting 和最终 replay
使用独立的刚体 CUDA Graph；控制步包含原有全部物理 substep，不降低 solver 迭代预算。
偶数 substep 保持双 state buffer 的闭环，奇数 substep 保留 eager 路径。几何 IK 同样缓存
CUDA Graph，并按 position／姿态目标及左右手 orientation mask 区分，避免捕获时冻结的权重
影响后续目标；迭代次数、位置约束与 roll 偏好不变。

刚体冷恢复同样复用 CUDA Graph，仍同步两份 state buffer、重算 FK 并清空 solver warm-start。
逆动力学在同一 MuJoCo Warp 后端缓存关节布局映射和 CUDA Graph，每次更新实际 state 与
期望加速度输入；不另建 CPU 动力学模型。奇数 substep 的 eager 路径保持不变。
候选残差按物理 world 批量计算；关闭的姿态项保持为零，不再逐候选计算朝向。已经固定 action
的事件预演和最终 replay 先上传整段命令，把每个实际控制边界留在 GPU，段末一次读回所需字段；
这不改变逐步动力学推进。生成 action 的反馈循环仍逐帧观察实际 state，再计算下一 action。

Shooting 与最终冷回放统一采用普通 runtime 的数值模式，不额外启用 Warp `RUN_TO_RUN`
确定性归约。最终回放仍使用独立 solver 从初态连续执行全部 action，并重复回放检查误差；
关闭确定性内核不等于取消回放验证，也不修改位置、事件或重复性门禁的原阈值。

manifest 中的门禁含义如下：

- `kinematic_valid`：几何参考的目标夹持中心相对语义目标的最大误差不超过 5 mm；
- `dynamics_valid`：action-only 冷回放的所有帧 attach 中心误差不超过 1 cm、按 runtime 主动指钳信号
  重算的 attach/release 阈值事件序列与源标记完全同帧，
  并且重复冷回放能复现机械臂和双指的发布状态。全程开度
  误差、同侧双指开度差以及相对 IK seed 的关节位置、速度差异只保留为诊断，不作为门禁；
- `up_axis_constrained=false`、`orientation_constraint_scope=none` 明确全程没有姿态门禁。
  `orientation_preference=bounded_local_z_y_up` 表示几何参考使用柔和 roll 偏好。
  `target_up_direction` 和方向误差保存在 `retarget_metrics.npz`，只用于诊断；旧的受约束
  Y/Z 最大误差字段在空 mask 下为 0，不代表全程方向误差为 0；
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
- Newton 刚体语义名称、状态顺序与运动学父节点；Isaac 使用语义名称映射可视节点。

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
运动学模型，Isaac 引用同一份工作站 USD。replay bridge 优先使用资产的 `clothdojo:bodyLabel`
映射刚体，未标注的 Piper 保留按节点名称映射；局部变换按 USD 实际父节点计算，不使用物理
关节父节点代替可视层级。写入保留已有尺度、visual/相机子节点的局部变换和资产原有的
`resetXformStack` 标记，不新增 reset 来绕过父节点。刚体通过 USD xform 发布给 RTX，
布料顶点则继续通过 Fabric 在 GPU 上更新。RTX 导出前
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

当前源机器人是 `piper`，目标机器人是 `aloha2`、`x5`、`fr3_duo` 或 `ur5e_2f85`。
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

第三人称相机复用旧项目 RoboDojo `camera_stand/00000` 的完整立柱、顶部转接件和 D455，
删除其独立 90 × 300 mm 底板及 Kai0 原相机底部支架，将 66.1 mm 脚座按中央孔阵外侧四孔
居中放置在 Kai0 一体化底板上。RoboDojo 54.1 mm 方形四孔与 Kai0 的六孔阵列并不机械
兼容；该固定位置是服务仿真的装配推断，不表示真机无需转接板即可安装。

X5 的完整双指和腕部相机几何来自固定 RoboDojo X5A；六轴真实位置限位来自固定
`real-stanford/arx5-sdk`。RoboDojo URDF 的 `joint1..5` 使用 `[-10, 10] rad` 占位值，不能
直接用于 IK。工作站 USD 左右各引用一份 RoboDojo 官方 `ARX.usd`，不再从 MJCF 生成
机械臂或将腕部 camera GLB 降面；`camera_base`、完整相机外壳及两个 fixed joint 均保留
上游定义。双臂安装位、固定底座 joint、末端 attach frame、drive、夹爪 home 和数据集相机
语义是 ClothDojo 的工作站覆盖层，不冒充 ARX 官方整机尺寸。MuJoCo 3.4 不能读取源 camera
GLB，因此 `x5.xml` 只用于 retarget/动力学，不能作为 Isaac 工作站的视觉真值。

FR3 Duo 保留官方手臂、VMKit 腕部/头部相机和转接件，安装恢复到原 payload 的桌边位置。
主臂惯性来自固定 fr3v2 description；Robotiq 整体替换为 UR5e 已接受的 Menagerie
双驱动闭环模型，包括几何、惯性、armature、被动弹簧与力矩限制。
每侧一个开合 action 控制两个 driver；q/qd 拓扑由 USD 同步定义，旧 FR3 state 布局
不能直接套用到新资产。安装、home、attach 和调试驱动属于项目覆盖，不冒充厂家标定。
固定工作站部署与渲染见 [robot_workstations.md](robot_workstations.md)。

UR5e 与 Robotiq 2F-85 分别保留 Menagerie 的标准单臂和夹爪模型。项目固定的组合 MJCF
只通过 MuJoCo `MjSpec.attach` 把 2F-85 挂到 UR5e 的官方 `attachment_site`，并把
`MjSpec.to_xml()` 缩写的 `solref` 恢复成上游显式向量；双臂安装、home pose、attach frame
和相机仍只属于 robot profile。2F-85 的四杆机构包含两个 `connect` equality constraint，
Newton 将其表示成额外 quaternion position coordinates，而 MuJoCo velocity layout 不包含
这些辅助坐标，因此 profile 分别保存 `q` 与 `qd/effort` 索引，避免用 position index 访问
actuator 或 inverse-dynamics DOF。标准 tendon actuator 在项目中适配为两个各限 2.5 Nm
的 position drive；保留连杆与闭环连接，但不宣称与原 tendon 控制器完全等价。

ALOHA 2 的归一化夹爪状态使用当前固定 visual mesh 的无穿透范围：`0` 对应两指关节
`(0.0079, 0.0079) m`，`1` 对应 `(0.037, 0.037) m`。官方 MJCF actuator 的软件下限
虽然是 `0.002 m`，但在该位置左右 custom-finger mesh 会相互穿透约 `11.65 mm`；mesh
刚好分离的位置约为 `0.00785 m`，因此闭合位保留约 `0.11 mm` 几何余量。两个坐标都为正
是因为左右滑块轴分别定义在各自局部坐标系中，并不表示两指向同一方向运动。Newton、Isaac
和 robot profile 使用同一组数值，避免训练动作、回放画面和部署命令各自采用不同的
开合范围。

ALOHA 工作站 payload 由 Newton `ModelBuilder` 导出，其中 `newton:damping`、limit gain 和
velocity limit 保存的是 SI 值；Newton 1.5 的 USD importer 会再次按 angular per-degree 单位换算。
场景适配层在 finalize 前恢复这些导出值，避免把 revolute joint 阻尼和 limit gain 放大
`180/π`，并由测试校验导入值与 USD 的 SI 定义一致。当前双侧 elbow/forearm-roll 被动
阻尼另采用项目 1× effort 调参值；它们不再宣称与迁移前 MJCF 相同，详见
[robot_dynamics.md](robot_dynamics.md)。

通过 `training_eligible` 门禁的数据可以用于视觉策略训练、目标机器人实际状态监督和
position-control action 监督；它证明的是目标机械臂刚体 action 在固定仿真中的可重放性，
不能证明机器人—布料接触动力学成立或真机安全。训练后策略的目标环境物理 rollout 与真机
门禁是独立阶段；在 Style3D 行为被验证前，不把 retarget 扩展成隐式布料物理 rollout。
