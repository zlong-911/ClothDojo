# shirt_fold 前后镜像增强

这个入口只处理双 Piper 的 `shirt_fold` canonical collection。每条通过门禁的源轨迹固定发布
四个 ID：原 ID、`<id>__front_back`、`<id>__mirror` 和
`<id>__front_back__mirror`；前后镜像被拒绝时仍保留原 ID 和 `<id>__mirror`。
没有 annotation 的完整源轨迹按 good 处理；有 annotation 时，原文件不改写并复制给实际发布的
变体。

## 使用入口

```bash
.venv/bin/clothdojo data augment-front-back \
  --source data/collections/shirt_fold \
  --output data/collections/shirt_fold_front_back_seed17 \
  --seed 17 \
  --device cuda:0 \
  --placement-candidates 64 \
  --max-placement-shift-x-m 0.15 \
  --max-placement-shift-y-m 0.05 \
  --min-base-clearance-m 0.10 \
  --min-inter-arm-clearance-m 0.03 \
  --table-margin-m 0.01
```

`--seed` 是必填的全局 seed；每条轨迹再由 seed 和源 ID 派生独立随机流，因此输入排序不改变
候选位置。命令拒绝覆盖已有输出，先在同级 staging 目录生成，完成后原子发布。输入只接受
`status=complete` 的轨迹；精确的 legacy EpisodeV1 manifest 会在内存中兼容读取，源文件不会
被改写。

## 空间与机器人语义

前后变换先以初始帧布料粒子的 XY 包围盒中心为反射中心，对 world X 做反射，再对整个 episode
施加一个固定的 seeded XY 微调；Z 不随机。零平移优先尝试，其他候选按离零平移的归一化距离
由近到远尝试。候选首先受 `--max-placement-shift-x-m` 和
`--max-placement-shift-y-m` 限制，默认分别不超过 15 cm 和 5 cm，再与保证所有布料回放帧都
位于源 manifest 桌面范围内且保留 `--table-margin-m` 的解析范围取交集。这样前后镜像可以为
可达性做有限调整，但不能靠把衣物大幅横移到单侧机器人附近来通过门禁。

若某个 coarse candidate 的 attach 误差不超过 10 mm 但仍有位置误差或关节分支跳变，或已通过 IK 但
最坏刚体 penetration 不超过 2 mm，入口会立刻在该位置周围按 `5 mm` 做确定性
XY 四轴向局部细化，并继续使用完全相同的严格门禁。这个细化只用于解决随机网格落在可行域边缘的
问题，不提高 5 mm IK 或 0.5 mm 碰撞容差；超出布料桌面解析范围的局部位置不会进入求解，
也不能越过上述 X/Y 微调上限。每个候选进入 IK 前还会检查初态布料
粒子到左右 `base_link` 原点的最小三维距离，默认不得小于
`--min-base-clearance-m=0.10`；这个门禁用于剔除布料几乎落到底座上的明显无效位置，并记录
最坏侧和实际 clearance。

若 home seed 得到的合法 IK 轨迹仍与工作台或自身碰撞，只对发生碰撞的一侧尝试固定的
`±1.5 rad` 腕部分支 seed。所有通过运动、碰撞和双臂净空门禁的分支都会按最大逐帧关节变化、
RMS 关节变化和偏离 home 的程度评分，选择更连续、较自然的一条；它不改变末端目标，也不扩大
X/Y 微调窗口，最终选择和全部尝试都会写入验证元数据。

布料的 `particle_q` 使用同一个反射和平移，`particle_qd` 只使用线性反射。布料不进入 Style3D
或接触 solver；因此最终折叠形状是源 good 轨迹的刚体空间等距变换，不会由机器人动力学重新
生成。前后镜像不交换手臂：源左臂目标仍由左臂执行，源右臂目标仍由右臂执行。

`grasp_attached` 为真的帧以镜像后的源 attach 中心为主目标，再根据记录的布料 grasp pins、
`grasp_pin_local` 和镜像后的夹爪 frame 重建局部布料 patch 中心；两者差值经过 21 帧平滑和
attach 段首尾 smoothstep 渐入/渐出后叠加，逐帧修正硬限 5 mm。这样局部动作可跟随布料 pose，
但软 pin 抖动不能直接驱动机械臂。

每只手的 maximal free 片段按上下文规划：初始 free 段从机器人 home TCP 开始，普通换抓使用
局部 lift-transfer-descend；同侧等待默认只侧移 5 cm。若等待点距离另一只仍在抓持的手小于
20 cm，或该手已经跨过工作区中线，则朝自身 home TCP 方向提前让行，平面位移（包括 yield）
硬限 25 cm，因桌面 clearance 需要的额外抬升硬限 8 cm。两手短暂并行的部分必须通过逐帧碰撞
和实体净空检查；轨迹末尾先松开、仍需给另一手让路的手会先抬升，再朝自身 home TCP
一侧的有界 standby 撤离并保持；最后松开的手在另一手已进入 terminal free 段后保持
release pose，避免在已无 attached 任务时制造多余的短时撤离动态。
撤离最多使用 60 个 state 间隔；若释放点已经满足高度净空，则跳过零距离抬升，避免停在另一只
手后续的 swept path 内。若释放点已经跨过工作区中线，则同时额外抬升并向远离机器人 base 的
`+X` 方向绕行，再向本侧横移，避免直接穿过另一只机械臂。自由路径在 release 端匹配上一抓持段的
速度；若另一只 free 手正接近下一次抓取，则把它的 approach 延后到 terminal retreat 已开始，
同时尽量保留到下一次 attach 的减速时域。attach/release 边界帧和所有 attached 主目标保持不变，
时间轴不增删。若 attached 帧已满足位置精度，但 free 帧的逐帧 IK 落到不连续分支，
先只平滑 detached 帧的关节路径，保留原避障形状与 attached 关节配置；若仍无法消除分支
跳变，再以两端 attached 配置为不可变锚点，对该手的 free 区间做零端点速度的五次
smoothstep 关节连续化。末尾 free 段只连续化到撤离路径首次稳定到 standby 的帧，之后
保持该配置；生成结果仍必须通过后续逐帧碰撞、净空和动力学门禁。

与 retarget 一致，IK 不把镜像后的源末端朝向作为硬约束，只保留求解器现有的 Y-up 柔和偏好；
镜像后的夹爪 frame 仅参与上述布料 patch 位置重建，不能牺牲位置精度或关节连续性。
几何参考随后进入 Piper 刚体 `RobotDynamicsRetargeter`，由 action
从初始状态完整冷启动重放。发布的
`joint_q/joint_qd/body_q/body_qd` 与 action 都来自这次 action-only replay，不把 IK 参考状态
冒充可执行状态。动力学跟踪门限作用于 attached 任务目标；detached staging target 是避障引导，
其 IK 与实际执行偏差允许存在，但最终实现出的关节轨迹必须重新通过碰撞与双臂实体间隙门禁。
几何 IK reference 的有限差分速度和加速度只记录为诊断；action-only replay 的实际轨迹仍严格执行
源 episode 运动包络门禁。

夹爪事件对齐只平移对应 phase 内的 command curve。若最优解耗尽 preview window，无论是正向提前
还是负向延后，都会继续向前扩展窗口；最终 attach/release 仍必须逐帧与源标注一致。

左右镜像沿用既有 `y=0` 双臂镜像代数。它分别作用于原轨迹和已经验收的前后镜像轨迹，所以
不会再运行 IK；前后镜像后左右手语义交换，两个空间反射组合后的 cloth winding 恢复原方向。

## 严格门禁

候选按顺序执行以下检查，任一失败即记录原因并尝试下一个 XY；不会放宽阈值或修改时间轴。
动力学包络失败但位置误差和关节限位仍正常时，也允许触发同一个 5 mm 局部细化；
细化优先沿 coarse candidate 已有的横向偏移方向搜索，再检查反向与另一轴：

1. 初态 cloth 与左右机器人底座满足配置的最小 clearance；
2. attached 目标与规划后的 free guidance 经过全帧 position-only IK；attached 误差不超过 5 mm，
   关节不越限；
3. 逐帧最大 arm 关节变化不超过 `max(0.25 rad, 2 × 源轨迹最大逐帧变化)`，用于剔除 IK 分支跳变；
4. 全轨迹左右臂互撞及机器人—工作台几何表面穿透不超过 home fixture baseline 加 0.5 mm；
   Newton 用于提前激活接触的人工 `shape_margin` 不计为实体穿透，但球体、胶囊等碰撞几何的
   有效半径仍完整计入；
5. 左右臂物理碰撞表面在控制帧和相邻帧中点均至少保留
   `--min-inter-arm-clearance-m`（默认 3 cm）；
6. action-only dynamics replay 的 attached 误差、夹爪事件时序和 action 重放一致性满足门禁；
   每个几何候选最多完整冷启动求解三次，每次都使用相同阈值；
7. dynamics 实际输出再次通过关节、跳变、刚体碰撞和 3 cm 双臂间隙检查，并且实际速度与
   加速度峰值不超过 `1.06 × max(源轨迹观测包络, manifest 配置上限)`；只有同一次重建同时
   通过第 6、7 项才可发布，三次均未通过则拒绝；
8. 变换后的全部 cloth 帧仍在桌面边界内。

Piper 导入器会保留同臂相邻连杆与固定基座的过滤，但不会过滤左右臂 collision proxy；否则双臂
碰撞门禁会被上游 articulation self-collision filter 静默架空。

源 manifest 的 arm velocity/acceleration 数值属于 planner 参数，真实采集状态本身可能超过它们；
因此门禁同时参考配置上限和源 episode 的实际峰值，并只保留 6% 的 cold replay 数值波动预算，防止
前后镜像产生明显的速度/加速度回退。峰值倍率、帧号和关节索引都会写入验证元数据。若所有候选
均失败，该源轨迹只发布原始与左右镜像两个安全变体，并在根目录 `augmentation.json` 的
`fallback_pairs` 和 `rejections` 中记录原因。成功轨迹的候选、变换、IK、动力学、碰撞和桌面验证
也写入 provenance。

## 存储边界

四变体使用 `derived_overlay_v1`。机器人字段与 action 存在增强目录中，cloth 字段由普通
`TrajectoryReader` 从 base trajectory 懒读，并在读取时应用正交空间变换。组合变体可以递归
引用前后镜像 overlay，reader 会逐层校验 base manifest SHA-256。

这避免把约 99 GB 的源布料状态复制四份，但意味着输出不是独立 collection：源 collection
路径与 manifest 必须保持不变，归档、迁移和训练时两者需要一起保留。源 manifest 被替换、
空间矩阵非正交或任一 archive 字段不符合 schema 时，reader 会直接拒绝读取。
