# ClothDojo 相关工作精读笔记

> 用途：从文献地图进入 Related Work 写作前的逐篇事实核验。本文档不替代 `related_work_landscape_zh.md` 的候选池，也不覆盖 `related_work_evidence_zh.md` 的声明矩阵。
>
> 核验时间：2026-09-06。优先依据论文正文、附录和作者项目页。下文的“安全写法”表示原文足以支持的概括，“避免写成”表示会扩大或扭曲原论文。页码按 PDF 首页为 P0；HTML 论文用章节定位。

## 1. 精读后的核心判断

ClothDojo 不能用以下任一单点建立新颖性：仿真平台、服装资产、多资产折叠数据、连续双臂策略、HIL、失败恢复、仿真特权信息、NOCS、成功筛选或 RGB 重渲染。每一项都已有直接前例。

真正需要建立的是一组更窄的关系：

1. **行为来源**：数据代理由人类示范训练，并在执行中根据中间观测继续产生动作；不是按外部脚本、拓扑图、关键点教师或示范片段顺序推进。
2. **数据闭环**：SFT 初始化后，代理在不同服装和初态上批量 rollout，任务指标自动筛查结果，HIL 只补充策略实际暴露的失败模式。
3. **特权信息的角色**：规范坐标以图像形式帮助生成代理理解跨资产结构，不改 VLA 的输入输出接口；它服务数据生成，而非构成下游策略的部署依赖。
4. **物理数据与视觉数据分开扩展**：代理 rollout 产生新的布料状态—动作过程，重渲染只扩展同一物理过程的外观覆盖。
5. **任务与规模**：在同一动态双臂环境中分别生成展平和折叠成功轨迹；预期覆盖 500 件服装，每件每项任务 10 条，共 10,000 条。该数字在数据审计完成前只能写作目标，不能写成既成结果。

最危险的二分法是“脚本 = 开环，学习 = 闭环”。FoldNet、Instant-Fold 和 SimWeaver 都能用规则生成数据并训练闭环策略；SimWeaver 的生成器本身也会重观测和重试。更可靠的比较轴是：**后续行为的结构由外部规则预先规定，还是由学习式代理根据执行后的状态继续决定。**

## 2. 横向事实表

| 工作 | 任务起点与终点 | 生成阶段的行为来源 | 最终策略输出 | 反馈与恢复 | 生成时特权信息 | 与 ClothDojo 最关键的边界 |
| --- | --- | --- | --- | --- | --- | --- |
| FlingBot | 严重褶皱到高 coverage | 自监督探索抓点；固定 pick–stretch–fling | 抓点 value map | primitive 间重决策；stretch 有视觉终止 | mesh/coverage 监督 | 学的是“在哪里抓”，不是连续行为生成 |
| Cloth Funnels | 任意状态到 canonicalized alignment；再用 heuristic 折叠 | 学习选择 fling/PnP 及参数 | spatial action maps | primitive 间闭环 | 仿真 mask、keypoint/目标状态 | 折叠由关键点 heuristic 完成 |
| SpeedFolding | 褶皱到平整，再折叠 | 学习 smoothing；折线编译为 folding primitive | primitive 与平面抓取位姿 | primitive 间闭环；fling 内有力阈值 | 人工标注与自监督结果 | 完整流程存在，但行为仍由 primitive 与指令组织 |
| UniFolding | 褶皱上衣到展平并折叠 | VR 人类动作、自监督探索、真机 HIL | primitive 类型与抓放点 | 每步重观测；真机在线标注 | 仿真 canonical/mesh 监督 | 已结合人类示范、HIL、展平和折叠 |
| FoldNet | 扰动/褶皱初态到预定义折叠 | 类别级关键点脚本；KG-DAgger 补抓取失败 | RGB Diffusion Policy 连续动作 | 最终策略闭环；教师自动重抓 | GT keypoint、gripper state | 最近的 privilege-to-RGB 与自动恢复先例，但任务结构预定义 |
| Instant-Fold | foldable manifold 到多种折叠；不从 crumpled 开始 | 19 类手工 folding programs | 视觉+单示范上下文到 8D action chunks | 最终策略闭环 | 粒子、可见性、语义 keypoint | 规模和连续策略都不是 ClothDojo 的独有点；行为库仍手工定义 |
| SoftMimicGen | 多类可变形任务；毛巾展平/折叠分开 | 选择、非刚性变换并执行示范片段 | 图像到末端/关节动作 | 每个 subtask 重新配准；成功后保留 | 完整 deformable nodes | 产生新物理 rollout，但 subtask 顺序和片段行为继承示范库 |
| SIM1 | 代表性验证为 T-shirt folding | 复用接触片段，diffusion 补全段间运动，再过滤 | RGB VLA 连续动作 | 生成后筛选，不是 on-policy 数据代理闭环 | 扫描 mesh、物理状态、粒子规则 | 同样是仿真数据引擎；生成主体不是闭环 VLA rollout |
| SimWeaver | 五项独立任务，含 T-shirt flatten/fold | 拓扑图、闭式谓词和规划器 | RGB π0.5 action chunks | 生成器重观测、验证抓取并重选 | canonical mesh、landmark、可见顶点 | 已有完整闭环合成与评测；区别只能落在学习式行为来源和 HIL |
| Flat’n’Fold | 真实 crumpled→flat→folded | 人直接操作或带动 Baxter | 数据集记录手/夹爪位姿 | 人自然调整；无自动数据闭环 | 无仿真特权信息 | 是自然人类长轨迹证据，不是可扩展策略数据代理 |
| π*0.6 / RECAP | 真实 laundry：取衣、展平、折叠、堆叠 | SFT 后自主 rollout 与人工纠正 | VLA action chunks | reward、value、advantage 与 intervention | 无仿真特权状态 | ClothDojo 流程的直接灵感；仿真增量是自动标签、并行度和可控资产/初态 |
| χ0 | 真实任意初态展平—折叠及更长任务 | 示范、标准/heuristic DAgger、增强 | π0/π0.5 action chunks | 失败状态纠正、stage advantage、控制对齐 | 人工阶段标签与任务规则 | 已覆盖连续任务和恢复；重点是资源高效真机训练，不是仿真数据生成 |
| DeMaVLA | 篮中随机衣物到铺平并折叠四类物体 | 大规模真实示范 + human-gated DAgger | 连续 flow-matching chunks | 人在失败附近接管再归还控制 | 无仿真特权状态 | 单一多类连续 VLA 已存在；ClothDojo 的贡献是仿真代理与可控数据扩展 |
| GarmentLab | unfold 与 fold 独立；前者随机 drop，后者平铺轻扰动 | benchmark、RL、视觉对应、teleop | 多种；视觉方法多预测点 | 依方法而定 | 粒子、关键点、完整状态 | 已是完整环境，但没有本文的数据代理闭环 |
| DexGarmentLab | 15 项服装任务；fold 只是其一 | 单示范任务序列经结构对应迁移 | HALO affordance + diffusion trajectory | 轨迹随几何调整；无 on-policy recovery 主线 | 点云对应、任务序列、抓点 | 资产规模和自动迁移已有更大先例，区别不在环境组件 |
| LeHome | Fling Garment 与 Fold Garment 独立 | 仿真遥操作；固定几何下重放并改外观 | ACT/DP/VLA 连续控制 | 成功检测用于筛选 replay | 仿真状态/几何判据 | 已有遥操—训练—评测链路；其 replay 主要扩展视觉而非行为 |
| NOCS / GarmentNets | 类别级刚体姿态 / 服装形态估计 | 不生成操作轨迹 | 稠密规范坐标/完整 mesh | 不适用 | 规范空间监督 | 证明 NOCS 及其服装化都不是新颖点 |
| UniGarmentManip | 展平、折叠、悬挂分别执行 | 将示范 manipulation points 对应到新资产 | fling 或 pick–place 参数 | 每个 primitive 后可重观测 | mesh tracing、skeleton correspondence | 跨资产结构先验和动作点迁移已有先例，但不生成开放连续行为 |
| SSFold | 真实任意褶皱到目标折叠 | 800 条真实人类示范 | 单次 pick/place 的像素分布 | 每次抓放后重观测 | 无仿真特权 | 已有“人类示范 + crumpled-to-folded”系统，但动作被压缩为顺序 PnP |
| BiFold | 平铺服装按语言完成折叠 | VR-Folding 人类动作，经解析和语言标注 | 双臂 pick/place | 用历史动作处理 refinement；低层 primitive 固定 | NOCS 区域标签 | 直接证明人类会偏离规定步骤并进行修正，也证明 NOCS 已用于行为数据加工 |

## 3. 逐篇证据卡

### 3.1 FlingBot（CoRL 2021）

**已核事实**

- 从任意褶皱状态提高布料 coverage，策略在顶视图上预测双臂抓点的 value map。
- 机器人随后执行固定的 pick–stretch–fling 运动；学习问题主要被压缩为抓取位置选择。
- 每次 fling 后重新观察并选择下一步。真机的 stretch 阶段还用前视视觉判断何时达到张紧状态，因此不能把整个系统称为“完全开环”。
- 训练先在仿真中自监督收集约 150K steps，再用少量真实经验微调；coverage 等监督依赖仿真状态或布料 mask。

**安全写法**

> FlingBot learns bimanual grasp selection while executing a predefined dynamic fling primitive, enabling rapid cloth unfolding with closed-loop re-selection between flings.

**避免写成**

- “FlingBot 没有反馈。”
- “FlingBot 学习完整双臂连续轨迹。”

**证据定位**：[论文](https://proceedings.mlr.press/v164/ha22a/ha22a.pdf)，P0 摘要；§III-B；§III-E；P4 的 simulation-to-real training。

### 3.2 Cloth Funnels（ICRA 2023）

**已核事实**

- 目标是把服装从任意状态转为 canonicalized-alignment，并通过 spatial action maps 在 fling 和 pick-and-place 之间选择动作及空间参数。
- 训练使用 12,500 个仿真 episodes；任务监督包含 coverage、canonicalization 和 alignment。
- 下游折叠不是自由学习的长程行为。论文先检测类别关键点，再按固定 heuristic 折袖、对折衣身；为新类别建立 detector 的成本被报告为约一小时。

**安全写法**

> Cloth Funnels learns multi-primitive canonicalization, after which a keypoint-parameterized heuristic performs folding from the aligned state.

**避免写成**

- “Cloth Funnels 学会了从褶皱状态连续折衣。”
- “它只靠固定脚本展平。”前半段 canonicalization 是学习得到的。

**证据定位**：[论文](https://arxiv.org/pdf/2210.09347)，§III、§IV-D，尤其是 Task Heuristics 和 folding evaluation。

### 3.3 SpeedFolding（IROS 2022）

**已核事实**

- 完整流程从 crumpled garment 开始，先 smoothing，再按用户给定 folding lines 完成折叠。
- BiMaMa-Net 在 smoothing 阶段选择 fling、drag 或 pick-and-place 并预测平面动作位姿；每项 primitive 的连续运动仍由预定义轨迹执行。
- 系统按视觉分类器判断是否足够平整，再进入 folding；折线被编译为双臂 pick-and-place 动作。
- 数据约含 4,300 个由人工标注或自监督获得的真实机器人 actions。Fling 在张紧阶段用力阈值，但速度和主要运动轨迹固定。

**安全写法**

> SpeedFolding couples learned primitive selection for smoothing with instruction-compiled folding primitives, yielding a closed-loop system at the primitive level.

**避免写成**

- “整个过程开环。”
- “折叠轨迹由网络连续生成。”

**证据定位**：[论文](https://arxiv.org/pdf/2208.10552)，§III-B Action Primitives、§III-E Self-supervised Learning、§III-F Folding。

### 3.4 UniFolding（CoRL 2023）

**已核事实**

- 同一 UFONet 根据逐步更新的 partial point cloud 预测 fling、fold1、fold2 及抓放点；必要时用规则化 drag/mop 处理不可达动作。
- 折叠阶段仍预先定义类别级 procedure，并连续执行两个 pick-and-place actions；网络学习动作点而非任意任务结构。
- 仿真 VR 数据包含 203 件短袖的 1,218 段视频和 315 件长袖的 1,575 段视频，平均每段约 5 个动作，采集约 16 小时。
- 真机 fine-tuning 并非人实时接管机器人，而是当前策略自动收集状态，标注者随后给出动作类型、动作点和候选抓点偏好；两类共 2,432 个真实样本，标注/采集约 20 小时。

**安全写法**

> UniFolding combines VR demonstrations, simulation self-supervision, and real-world human-in-the-loop annotation to train a unified primitive-level policy for unfolding and folding shirts.

**避免写成**

- “此前没有把 HIL 用于服装展平和折叠。”
- “UniFolding 的 HIL 是完整连续轨迹的人工接管。”原文描述的是在线采集后的动作与偏好标注。

**证据定位**：[论文 HTML](https://arxiv.org/html/2311.01267)，§3.1–3.3，尤其 §3.3 Data Collection & Network Training。

### 3.5 FoldNet（RA-L 2026）

**已核事实**

- 合成四类服装资产并自动附加 semantic keypoints。最终报告 15K trajectories、约 2M image–action pairs，真机成功率 75%。
- 每一类别的教师策略按预定义阶段执行。以 T-shirt 为例：旋转、拖动、依次折袖、折衣身；每阶段抓放点来自 GT keypoints，中间轨迹由插值获得。
- KG-DAgger rollout 当前视觉策略，利用 GT keypoints 和 gripper state 检测抓取失败，再由关键点策略接管重抓；最终 RGB policy 不需要显式 keypoint 或失败检测。
- 其恢复主要针对 grasp failure，不能等价为开放的人类式局部整理、动作重构或任意失败恢复。

**安全写法**

> FoldNet distills a keypoint-scripted folding teacher into an RGB closed-loop policy and augments training with keypoint-gated regrasp demonstrations collected on policy-induced failures.

**避免写成**

- “FoldNet 只生成理想成功轨迹，没有失败恢复。”
- “FoldNet 的关键点脚本不会适应执行结果。”KG-DAgger 会针对抓取失败自动纠正；更精确的限制是高层阶段与恢复类型仍由人工定义。

**证据定位**：[论文 HTML](https://arxiv.org/html/2505.09109)，§IV-A–C；摘要；§V 的 Data scale。

### 3.6 Instant-Fold（2026 preprint）

**已核事实**

- 在 360 个 Cloth3D tops 上用脚本化双臂 folding programs 生成约 80K trajectories；300 个 mesh 用于训练，60 个 held out。另有约 40K 只用于表示预训练的随机 deformation trajectories。
- 19 个 folding modes 由 sleeve primitive、body-fold primitive 和执行顺序组合；语义关键点由 canonical flattened mesh 上的几何规则检测，轨迹由 keypoint-conditioned PnP subactions 构成。
- 最终 flow-matching policy 根据当前视觉/本体状态和一条示范上下文输出 8D 双臂 action chunks，属于闭环连续策略。
- 作者明确列出限制：起点处于 foldable manifold，而非 crumpled configurations；只研究 top garments；context library 手工设计且规模有限。

**安全写法**

> Instant-Fold trains a closed-loop continuous policy on a large scripted corpus and conditions execution on a single demonstration, but its training modes are manually composed and exclude crumpled initial states.

**避免写成**

- “此前没有大规模、多资产、连续双臂折叠数据。”
- “80K 轨迹代表 80K 种自主发现的行为。”它们由 19 个手工 mode 及执行 variants 程序化展开。

**证据定位**：[论文 HTML](https://arxiv.org/html/2606.04269)，§4 Data Generation；Appendix A；§5 Limitations。

### 3.7 SoftMimicGen（ICRA 2026）

**已核事实**

- 每个任务实际收集 1–3 条 Apple Vision Pro 遥操作示范，再生成 1,000 条；论文一般问题设定允许 1–10 条源示范。
- 源轨迹先按 object-centric subtasks 切分。每个 subtask 开始前，生成器读取完整 deformable node positions，选择配准代价最低的源片段，用非刚性 warp 变换末端轨迹并重新执行；成功后才加入数据集。
- 因此它确实产生新的物理交互轨迹，不是重渲染。但 subtask 划分、执行顺序和局部行为结构来自源示范库。
- 下游 BC-RNN/DP 只从图像执行，不在 inference 时运行 non-rigid registration。

**安全写法**

> SoftMimicGen expands a few demonstrations into new physical rollouts by non-rigidly registering and re-executing source trajectory segments under privileged deformable state.

**避免写成**

- “SoftMimicGen 只是复制或重渲染示范。”
- “它由学习策略在新状态中自主决定新的高层行为顺序。”

**证据定位**：[论文 HTML](https://arxiv.org/html/2603.25725)，§III-C–D、§IV-C、§V-A–B。

### 3.8 SIM1（2026 preprint）

**已核事实**

- 以高精扫描、物理参数校准和稳定求解器建立 real-to-sim-to-real 数据引擎。代表性真实验证是 T-shirt folding。
- 仿真中先收集 200 条 teleoperated source demonstrations。生成时保留接触/抓取片段，用 diffusion 生成交互片段之间的平滑运动。
- 使用仿真粒子状态规则和 RGB video discriminator 双重过滤，再通过 Blender 外观随机化输出 LeRobot 格式数据。实验把生成数据扩展至 10K。
- 论文对片段顺序的描述存在轻微歧义：一处说保留原始 ordering，另一处说变化 sequence order。因此当前不能断言它严格保持或自由改变高层顺序。

**安全写法**

> SIM1 grounds a simulator in measured geometry and dynamics, then expands teleoperated seeds by recombining preserved interaction segments, synthesizing transitions, and filtering executed trajectories.

**避免写成**

- “SIM1 只改外观。”
- “SIM1 的 diffusion model 就是闭环策略代理。”它用于片段间 motion synthesis，生成流程仍有显式 decomposition 与 filtering。

**证据定位**：[论文 HTML](https://arxiv.org/html/2604.08544)，§3.1–3.3、§4.1。

### 3.9 SimWeaver（2026 preprint）

**已核事实**

- 每项任务从一个 task seed 生成 200 条 deterministic demonstrations，训练 RGB π0.5，并在五个真实可变形任务上零样本部署。
- 生成器用 canonical mesh 上的语义 landmarks 和拓扑邻接图限定有效双臂抓取对，用闭式状态谓词处理遮挡、分层、可达性和双臂冲突，再由 constrained planner 生成连续轨迹。
- 生成过程不是单次开环脚本：每次抓取后重新观测可见顶点，验证结果，固定已成功的手臂，并在失败时重选和重试。
- T-shirt flatten 与 T-shirt fold 是独立任务，不是从同一严重褶皱初态连续完成的统一 episode。

**安全写法**

> SimWeaver synthesizes continuous trajectories with topology-aware rules and closed-loop verification, removing teleoperation while retaining an explicitly programmed task and state abstraction.

**避免写成**

- “SimWeaver 没有反馈，只是固定脚本。”
- “现有规则生成器都无法处理抓取失败。”SimWeaver 是直接反例。

**证据定位**：[论文 HTML](https://arxiv.org/html/2606.15338)，§4，尤其 Closed-loop synthesis；§7.1。

### 3.10 π*0.6 / RECAP（2025 report）

**已核事实**

- 流程是预训练 VLA → 目标任务示范 SFT → 一轮或多轮 on-robot collection → value/advantage-conditioned offline RL。
- 部署数据同时包括 fully autonomous rollouts 和 human-gated corrections。每条 episode 有成功/失败标签；论文中的真实任务质量标签由人工评价多个维度后汇总，不是仿真自动指标。
- diverse laundry 从篮中 T-shirt 或 shorts 开始，包含 retrieval、flattening、folding 和 stacking，500 秒内完成才算成功。
- diverse laundry 收集 450 个 evaluation episodes 和 287 个 correction episodes。最终模型相对 offline RL + SFT 将该任务 throughput 提升超过一倍，并将失败率约减半；还在新家庭连续折叠未见衣物超过两小时。

**安全写法**

> Inspired by RECAP, we treat an SFT policy as a data-collecting agent and improve it with deployment experience and targeted human corrections; simulation additionally enables privileged observations, parallel rollouts, and automatic task evaluation.

**避免写成**

- “SFT—rollout—HIL 是 ClothDojo 首创。”
- “π*0.6 的所有成功标签自动得到。”
- “两小时 demo 是同一件衣服反复折叠。”论文说的是 novel laundry items in a new home。

**证据定位**：[论文 PDF](https://www.pi.website/download/pistar06.pdf)，P0–P1 overview；§VI-A task definition；P8 main results；Appendix G data quantities。

### 3.11 χ0 / Kai0（2026 preprint）

**已核事实**

- Task A 要求机器人从 arbitrary initial configuration 展平 T-shirt 并折叠，180 秒内把成品放在桌面中心；Task B/C 增加分类、交接和悬挂。
- 每项任务约采集 20 小时专家示范，使用 π0.5/π0 flow-matching action chunks。
- standard DAgger 等待当前策略自然进入失败状态再收集恢复；heuristic DAgger 直接把系统初始化为人工设计的失败状态，如错位抓取或部分掉落，再采集恢复示范。
- Stage Advantage 使用人工 stage labels；Train-Deploy Alignment 还包括左右臂镜像、跳帧速度增强和 temporal chunk-wise smoothing。

**安全写法**

> χ0 demonstrates long-horizon garment flattening and folding with continuous VLA control, while improving recovery through standard or heuristic DAgger and stage-aware supervision.

**避免写成**

- “χ0 只做平铺后的折叠。”
- “heuristic DAgger 是当前策略自主发现失败。”其中一条路径显式构造失败初态。

**证据定位**：[论文 HTML](https://arxiv.org/html/2602.09021)，§III-D–E、§IV-A–B。

### 3.12 DeMaVLA（2026 preprint）

**已核事实**

- 真实任务从随机掉入篮中的物品开始，机器人需取出、放桌、展开并铺到可折状态，再完成 shirt、skirt、pants 或 towel 的折叠。
- 单一 6.6B checkpoint 从多视角图像、语言和本体状态预测 flow-matching continuous action chunks；用 training-time RTC 处理异步推理延迟。
- 预训练约 5,000 小时真实双臂数据，四类 folding post-training 数据共 37 小时。
- Human-gated DAgger 在策略进入 unsafe、unstable 或 clearly suboptimal 状态时由人接管，修正到有效状态后把控制权交还策略；每轮 intervention 数据并入训练集。

**安全写法**

> DeMaVLA shows that one continuous VLA can execute complete multi-category household folding and improve policy-specific failures through real-robot human-gated DAgger.

**避免写成**

- “连续、多类别、从篮中取衣到折叠此前尚未实现。”
- 将其 37 小时 post-training 数据与 5,000 小时预训练数据混成同一 folding 数据规模。

**证据定位**：[论文 HTML](https://arxiv.org/html/2605.31286)，§3.1、§3.3–3.4、§4.2。

### 3.13 GarmentLab（NeurIPS 2024）

**已核事实**

- 提供基于 Isaac Sim 的服装环境，结合 PBD/FEM、多类机器人和传感器、ROS/MoveIt、11 类服装与 20 项任务。
- Benchmark 明确把 unfolding 和 folding 分开：unfolding 从随机 drop 的变形状态开始；folding 从平铺并轻微扰动的状态开始。
- folding 以最终粒子状态与 human demonstration 目标的距离判断；unfolding 以相对平铺目标的顶点范围判断。
- 视觉/RL baseline 主要学习操作点，连续轨迹多采用 scripted pick-place 或 fling；论文也明确说明直接用 RL 学轨迹不稳定。

**安全写法**

> GarmentLab broadens garment simulation and benchmarking across assets, physics, and tasks, while defining unfolding and folding as separate protocols with largely scripted low-level motions in its principal baselines.

**避免写成**

- “GarmentLab 只提供模拟器，没有采集、算法和真机接口。”
- “GarmentLab 已评测从严重褶皱连续展平再折叠。”

**证据定位**：[正式论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/15f80ec0fed53885d2ca6272edb96ede-Paper-Conference.pdf)，P1–P5；Appendix B.2–B.3（P16–P18）。

### 3.14 DexGarmentLab（NeurIPS 2025）

**已核事实**

- 包含 2,500+ garments、8 类服装和 15 项任务，强调双臂灵巧手、服装—自身与服装—环境交互。
- 自动采集从单条专家示范中取得 demo grasp points、hand poses 和 task sequence。其中抓点与 task sequence 实际由人工定义，hand poses 通过 LeapMotion 获取。
- GAM 用跨服装结构对应找到新资产上的抓点；系统随后按示范 task sequence 执行 IK/PD，并根据衣长、袖长或目标物位置调整抬升高度和放置位置。
- HALO 再用 affordance 和 Structure-Aware Diffusion Policy 学习跨形状轨迹。真机主要展示从该采集管线训练的策略，不应等价为完全零样本仿真迁移。

**安全写法**

> DexGarmentLab scales assets and demonstrations by transferring manually specified task sequences and grasp semantics across garment instances, with learned trajectory adaptation for geometric variation.

**避免写成**

- “DexGarmentLab 的轨迹完全固定。”其几何轨迹会随资产形状调整。
- “它从单条示范自主发现新的任务结构。”task sequence 仍由示范和人工定义给出。

**证据定位**：[论文 HTML](https://arxiv.org/html/2505.11032)，§4 Automated Data Collection、§5 HALO、§6.3。

### 3.15 LeHome（ICRA 2026）

**已核事实**

- 是带家庭场景、可变形物体、低成本 LeRobot 系列本体、遥操作、训练与 sim/real 评测的完整链路。
- Fling Garment 与 Fold Garment 是两个独立任务：前者从 wrinkled clothes 展平，后者从桌面服装开始折叠。
- 每项任务用 50 条仿真遥操作示范训练，在 100 个仿真 trials 上比较 ACT、DP、SmolVLA 和 π0。
- 轨迹 replay 固定物体位置以保持几何和接触，只改变 lighting、background texture 和 visual material，再用 task-specific success detector 过滤。因此 replay 主要增加视觉版本，不是新的操作行为。

**安全写法**

> LeHome integrates low-cost teleoperation, policy learning, and sim-to-real evaluation, while augmenting demonstrations mainly through appearance-randomized replay of fixed interaction trajectories.

**避免写成**

- “LeHome 只是环境。”
- “它的 replay 生成了新的初态—动作物理轨迹。”论文明确固定 object positions。

**证据定位**：[论文 HTML](https://arxiv.org/html/2604.22363)，§III-D–E、§IV-A–D。

### 3.16 Flat’n’Fold（ICRA 2025）

**已核事实**

- 真实数据覆盖 44 件服装、8 类，共 1,212 条 human demonstrations 和 887 条 human-controlled Baxter demonstrations；每条从 crumpled 到 flattened 再到 folded。
- flattening 不规定步骤，允许参与者在落桌后继续局部调整；folding 同时包含统一终态的 fixed strategy 和参与者日常使用的 daily-life strategy。
- 人类数据以 15 Hz 记录三视角 RGB-D、手背 tracker 位姿和脚踏板标注的抓放时刻；机器人数据由专家在 Baxter zero-G mode 下直接带动双臂采集。
- 采集明确约束人类避免屈指、只使用 pick-and-place、尽量不交叉手臂。因此它保留了自然调整与顺序差异，但不是未受约束的完整人手动作空间。
- 原论文只建立 grasp-point prediction 和 subtask decomposition benchmark，没有报告用该数据训练端到端闭环策略的成功率。

**关于“是否直接可用”的准确结论**

不能绝对写“Flat’n’Fold 不能直接使用”：robot subset 确实含 Baxter joints、wrist pose、gripper state 和同步视觉，理论上可训练匹配接口的策略。更准确的说法是：

> Flat’n’Fold is a real, embodiment-specific recording dataset rather than an automated robot-data generator; transferring it to another robot or VLA interface requires observation, action, calibration, and embodiment adaptation, and the paper does not validate end-to-end policy learning.

**证据定位**：[论文 HTML](https://arxiv.org/html/2409.18297)，§III-A、§III-C–D、§IV。

### 3.17 NOCS（CVPR 2019）

**已核事实**

- 原始 NOCS 为同类刚体实例定义共享的 unit-cube 规范空间，使不同实例具有一致方向、位置和尺度参照。
- 网络从 RGB 预测每个可见像素到 NOCS 的稠密对应，再结合 depth 估计 6D pose 和尺寸。
- RGB 色值是三维规范坐标的可视化编码；原论文用途是类别级姿态估计，不是机器人策略条件化。

**对 ClothDojo 的边界**

ClothDojo 可以把 NOCS 图像作为不改 VLA 结构的结构先验，但不能声称提出 NOCS、首次以 RGB 编码 NOCS，或首次让模型从 NOCS 图像学习。

**证据定位**：[论文 HTML](https://arxiv.org/html/1901.02970)，§1、§3。

### 3.18 GarmentNets（ICCV 2021）

**已核事实**

- 已把 NOCS 扩展到服装类别。输入是抓起服装的 partial RGB point cloud；模型先预测可见点 canonical coordinates，再在 canonical space 补全被遮挡的三维表面，并映回 task space。
- 规范姿态来自 CLOTH3D 中人体 T-pose 穿衣状态，并按类别统一缩放进 unit cube。
- 该工作是 perception/pose completion，不生成操作轨迹；其输出提供跨实例语义对应，可服务折叠等下游任务。

**安全写法**

> GarmentNets extends canonical coordinate prediction to highly deformed garments, using category-level shape completion to infer occluded surfaces from partial observations.

**避免写成**

- “NOCS 此前只用于刚体。”
- “GarmentNets 训练了服装控制策略。”

**证据定位**：[论文 HTML](https://arxiv.org/html/2104.05177)，§3.1–3.4。

### 3.19 UniGarmentManip（CVPR 2024）

**已核事实**

- 自监督学习同一服装跨形变、同类不同服装跨形状的 dense visual correspondence，再以少量示范适配 task-specific functional correspondence。
- 展平用双臂 fling：为遮挡情况设计 4 组候选抓点，最多执行 3 次；折叠把示范中的 pick/place sequence 映射到新服装对应点并逐步执行。
- 因此它迁移的是 manipulation points 和既有动作序列，不是由连续策略自主生成新行为。

**安全写法**

> UniGarmentManip transfers manipulation points and primitive sequences across garment instances through learned dense correspondence.

**避免写成**

- “跨资产结构先验此前没有用于服装数据或策略。”
- “UniGarmentManip 的 correspondence 等同于 ClothDojo 的 NOCS 图像输入。”前者是学习表示并用于点迁移，后者是生成代理的视觉条件。

**证据定位**：[论文 HTML](https://arxiv.org/html/2405.06903)，§3 Problem Formulation、§4、§5.2。

### 3.20 SSFold（T-ASE 2025）

**已核事实**

- 用低成本单目 hand tracking 采集 800 条真实人类示范，覆盖 towel、T-shirt、shorts，均从 crumpled initial state 开始；采集约 21 小时。
- 同一 two-stream network 用当前 RGB、pick-conditioned placing 和 visible connectivity graph 预测下一次 pick/place，宣称统一 smoothing 与 folding。
- 机器人为单台 UR5；动作被定义为一次抓取、轻抬、平行拖到 place point、释放，低层轨迹用 URScript 规划。
- 因而它保留了人类选择的抓放序列，但没有保留完整双手连续运动或双臂协调。

**安全写法**

> SSFold learns sequential pick-and-place decisions from real human demonstrations to fold several cloth categories from crumpled states, while abstracting away continuous bimanual motion.

**避免写成**

- “此前没有真实人类示范覆盖 crumpled-to-folded 全流程。”
- “SSFold 是双臂连续控制策略。”

**证据定位**：[论文 HTML](https://arxiv.org/html/2411.02608)，§III-A–C、§IV-A。

### 3.21 BiFold（ICRA 2025）

**已核事实**

- 使用 VR-Folding 的近 4K 人类双手折叠示范，解析出约 7K actions，并自动生成 1K+ language prompts。
- 原数据从 flattened T-pose 开始，志愿者被要求遵循类别级折叠步骤；但真实记录出现单手动作、最多 6 个 parsed actions 和数量不可忽略的 refinement actions，说明参与者没有机械复现指令。
- BiFold 用连续跟踪的 garment NOCS 将抓放点划分为袖、腿、腰等语义区域，再生成语言；这直接证明 NOCS 已用于服装行为数据加工。
- 模型预测双臂 pick/place；模拟执行仍采用固定七步 pick-and-place primitive。对于 crumpled initial state，论文建议先运行外部 flattening model。

**安全写法**

> BiFold shows that even instruction-guided human folding contains variable action counts and corrective refinements; it parses these behaviors into language-conditioned bimanual pick-and-place actions using NOCS-based semantic regions.

**避免写成**

- “VR-Folding 人类示范完全照固定脚本执行。”
- “BiFold 从任意褶皱状态端到端完成折叠。”

**证据定位**：[论文 HTML](https://arxiv.org/html/2501.16458)，§II-C、§III-B；Appendix A-C、A-D、B-C。

## 4. 可直接用于 Related Work 的组织逻辑

8 页 ICRA 正文不适合逐篇罗列。更自然的写法是三段，每段只建立一个对比关系。

### 第一段：服装策略如何压缩动作空间

先承认 primitive 路线的有效性：FlingBot、Cloth Funnels、SpeedFolding 和 UniFolding 通过学习抓点、放点或 primitive selection，把高维控制压缩为可学习问题。随后指出边界：连续运动及部分任务结构仍由预定义 primitive、关键点规则或 folding instructions 提供。再用 SSFold/BiFold 说明，人类示范即使受指令约束，也会出现额外步骤和 refinement；这给连续数据代理提供动机，但不能被写成已有方法“没有反馈”。

### 第二段：仿真怎样生成服装数据

按行为来源分组，而不是按论文年份堆叠：

- FoldNet、Instant-Fold、SimWeaver：外部规则、关键点或拓扑规划器定义行为结构；其中 FoldNet 和 SimWeaver 都有恢复，不能概括为开环脚本。
- DexGarmentLab、SoftMimicGen、SIM1：把少量示范迁移、重组或生成到新状态，并重新执行得到新物理 rollout。
- LeHome：轨迹 replay 主要固定物理过程并扩展外观。

收束句应落在：这些路线能高效产生数据，但高层行为仍由任务程序或源示范分解约束；ClothDojo 改用学习式连续策略作为数据代理，让后续动作直接依赖执行后的视觉状态。

### 第三段：数据闭环、平台与结构先验

先用 π*0.6、χ0、DeMaVLA 说明真实机器人上的示范—rollout—纠正闭环已经有效，不能宣称本文首创 HIL。再说明 GarmentLab、DexGarmentLab、LeHome 已把环境、资产、采集、训练和评测组合到不同程度，“完整平台”本身也不是差异。最后用 NOCS、GarmentNets、UniGarmentManip 和 BiFold 交代规范空间先验的来源，明确本文只把它作为兼容现有 VLA 的生成时视觉条件。

最终落点不是组件清单，而是：**ClothDojo 把真实机器人上验证过的策略数据闭环移入可并行、可自动评测的动态双臂服装仿真，以学习式代理扩展新的物理轨迹，再将成功交互转为普通 RGB 数据供下游策略学习。**

## 5. 当前可用与不可用的声明

### 可用，但需保留限定词

- “Representative garment systems learn primitive selection or grasp parameters while relying on predefined low-level motions.”
- “Recent simulators support large garment collections, continuous policy learning, and closed-loop evaluation, but differ substantially in task initialization, action interface, and data-generation logic.”
- “Rule- and demonstration-based generators can produce new physical trajectories, yet their behavior structure is typically specified by task programs, semantic keypoints, or source trajectory segments.”
- “Real-robot studies show that autonomous rollouts and targeted corrections improve long-horizon garment policies, motivating an analogous data loop in simulation.”
- “Canonical coordinates have been used for garment perception, correspondence, and behavior annotation; here they serve only as a privileged visual condition for the data-generating policy.”

### 暂不可用

- “首个大规模多资产服装数据集/平台/benchmark。”
- “首个从褶皱状态完成展平与折叠的连续策略。”
- “首个将 HIL、自动恢复或特权信息用于服装数据生成。”
- “现有仿真数据全部由开环脚本产生。”
- “Flat’n’Fold 数据不能用于机器人学习。”
- “现有规则方法无法根据失败重试。”
- “NOCS 只在刚体上使用过。”

### 仍需实验或项目事实才能成立

- 500 件资产、每件展平和折叠各 10 条、共 10,000 条成功轨迹。
- 代理 rollout 相对人工遥操作、关键点脚本或无 HIL SFT 的成功率、吞吐与行为多样性。
- NOCS 图像是否真的加快收敛、提高跨资产成功率，以及收益是否来自结构先验而非额外颜色捷径。
- HIL 的边际收益：补充多少次介入、解决哪些失败、自动筛选后保留率如何变化。
- 新物理 rollout 与重渲染的独立增益，以及二者是否存在互补。
- 下游只用普通 RGB 时，生成数据能否提高未见资产和未见初态上的闭环成功率。

## 6. 下一步写作前的剩余核验

1. 逐项核对 ClothDojo 的真实 action representation、观测频率、chunk 执行长度和双臂控制接口，避免把“连续动作”写成模糊口号。
2. 明确 10,000 条轨迹是最终发布数据、实验预算还是预设目标；展平和折叠是否各自从不同初态开始。
3. 固化自动成功判据，并区分 rollout 在线筛查、最终 benchmark metric 和人工质量复核。
4. 用实际 HIL 记录定义“介入”：整段接管、局部纠正、重启 episode，还是只补标签。
5. 对 FoldNet、Instant-Fold、SimWeaver 和 ClothDojo 做一张行为来源对照图或表；这是最容易让审稿人理解本文差异的证据。
6. 定稿前针对 2026 年后续 preprint 再检索一次完整交集；在此之前不使用“first”。

## 7. 第二轮复核：短会议 Related Work 的写法

### 7.1 篇幅约束

- ICRA 2025 的正式投稿要求是 6 页正文加不限页数参考文献，不是固定 8 页。即使项目按“约 8 页”规划，Related Work 仍应按六页会议正文的密度控制。
- 对 ClothDojo，合适的正文规模是两个小节、三至四个自然段。每段承担一个比较关系，引用服务于判断，不为覆盖论文数量单独增加段落。

### 7.2 优秀范例的共同结构

- **Cloth Funnels（ICRA 2023）**：Related Work 只有三个紧凑主题块，分别围绕启发式方法、学习式展平和目标条件操作。每块先概括共同假设，再指出与本文任务直接相关的边界，没有逐篇复述方法流程。
- **Mobile ALOHA（CoRL 2024）**：按 mobile manipulation 与 imitation learning 两条轴组织。论文很多，但名称和实现细节都退居引用；段末马上落到本文需要解决的数据或硬件缺口。
- **π*0.6/RECAP**：每段围绕一种策略改进机制，先承认已有路线，再用一句机制级差异界定本文。它不把“用过哪些组件”写成贡献，也不重复引言中的任务动机。
- **UniFolding**：按 unfolding 与 folding 任务分组，结构清楚；但与引言存在一定重复，不能照搬其篇幅。

### 7.3 对 ClothDojo 的具体约束

1. 以比较轴组织：第一节比较“策略直接决定什么”，第二节比较“仿真数据的行为结构由谁给出”。
2. 平台论文只承担一个事实：环境和资产已经存在。不要把 GarmentLab、DexGarmentLab、LeHome 的功能清单各写一遍。
3. primitive 工作的准确边界是“高层决策发生在 primitive 边界，内部反馈完成当前动作”；不要写成完全开环，也不要把局部反馈扩大成行为级重组。
4. 折叠和褶皱展平分开。大规模折叠数据只能证明结构化任务可以扩展，不能替代严重褶皱展平的证据。
5. SimWeaver 单独作为规则展平的直接近邻处理：承认遮挡判据和抓取重试，再指出受控初态、有限候选和固定 post-grasp behavior。
6. Instant-Fold 单独区分“数据生成方式”和“训练后策略接口”：数据来自人工定义的 19 种折叠模式，策略本身是连续闭环，不能归入运行时规则规划器。
7. 真机 SFT/rollout/HIL 只在一处出现，用来说明连续行为可以学习及其采集成本；不在仿真小节重复流程。
8. NOCS 不单列小节。若正文需要定位，只用一句话说明规范空间已有先例，本文把它用作数据代理的视觉先验。
9. Flat'n'Fold 不进入当前主线。它是绑定真实采集系统的离线记录，与本文最直接的“仿真数据如何生成”比较轴关系较弱；强行加入会拉长叙事并模糊近邻。

### 7.4 新稿的主线

建议使用两节四段：

1. **任务专用 primitive**：动作选择与连续执行的边界。
2. **连续策略与真机数据**：VLA 已验证行为能力，采集成本形成瓶颈。
3. **仿真平台与程序化数据**：环境规模不等于适应性行为规模；规则与示范变换主要在结构化折叠上扩展。
4. **最接近的展平方法与本文定位**：准确处理 SimWeaver，再用一到两句落到学习式数据代理和 NOCS 的使用方式。

这一结构不再沿用旧稿的“平台—数据生成—特权信息”三块并列。它顺着同一问题推进：**策略的行为由人写入 primitive，还是由数据学习；仿真扩展的是既定解法，还是从人类交互中学到的行为。**
