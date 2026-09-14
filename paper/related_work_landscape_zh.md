# ClothDojo 相关文献地图

> 用途：Related Work 写作前的大范围候选池。本文档先保证路线覆盖和比较口径，不等同于最终正文，也不默认每篇都进入参考文献。
>
> 调研截至：2026-09-06。优先采用正式论文、作者项目页、官方代码或数据页；2026 年尚未正式发表的工作统一标为 preprint，不把项目页宣传数字当作已独立复现的结论。

## 1. 统一比较口径

服装论文常用相同词语描述不同问题。后续阅读和写作至少同时核对以下六项：

1. **任务起点与终点**：平铺后折叠、轻度扰动后折叠、严重褶皱后展平，还是从褶皱连续完成展平与折叠。
2. **操作对象**：矩形布、毛巾、单类上衣，还是跨类别服装；训练和测试资产是否分离。
3. **行为表示**：策略预测 primitive/抓放点，还是直接预测关节或末端连续动作；低层轨迹由谁决定。
4. **反馈粒度**：primitive 之间重新观测、action chunk 之间重新推理、primitive 内预写反馈，还是策略直接生成连续闭环行为。
5. **数据怎样扩展**：人工新增采集、脚本或规划器执行、示范迁移、片段重组、策略自主 rollout，还是固定物理轨迹的视觉重渲染。
6. **特权信息归属**：特权状态供最终策略使用、供规划器/教师使用，还是仅用于数据生成和成功筛选。

这六项比“是否使用学习”“是否闭环”“是否在仿真中生成数据”更能区分 ClothDojo 与已有工作。

## 2. 直接近邻：正文必须逐项考虑

| 工作 | 状态 | 实际覆盖 | 数据与行为机制 | 对 ClothDojo 的约束 |
| --- | --- | --- | --- | --- |
| [UniFolding](https://proceedings.mlr.press/v229/xue23b.html) | CoRL 2023 | 随机褶皱上衣的展平与折叠；真机评测 | VR 人类示范、自监督仿真数据和真机 HIL；策略选择 primitive 及抓放点 | 已将人类示范、HIL、展平和折叠结合；差异不能只写“使用 HIL”或“覆盖两项任务” |
| [SSFold](https://arxiv.org/abs/2411.02608) | T-ASE 2025 / arXiv | 从任意褶皱状态完成 smoothing 与多种目标折叠 | 从真实人手视频提取 pick/place；图像与可见连接图预测下一次抓放 | 是“人类示范 + 褶皱到折叠”的直接近邻，但行为仍离散为逐次 pick–place，且不以大规模仿真数据生成器为目标 |
| [FoldNet](https://pku-epic.github.io/FoldNet/) | RA-L 2026 | 四类合成服装的多阶段折叠；初态含旋转、翻面和褶皱 | GT keypoint 教师生成动作；KG-DAgger 在策略失败状态自动接管重试；最终 RGB Diffusion Policy 输出连续动作 | 已实现“特权教师生成数据、失败恢复、普通视觉策略部署”；差异必须落在教师自由度、展平任务、多资产规模和数据代理形式 |
| [Instant-Fold](https://arxiv.org/abs/2606.04269) | 2026 preprint | 19 种折叠模式，360 个服装 mesh，约 80K 轨迹 | 程序化组合袖子/衣身 folding primitive；最终策略根据单条示范上下文输出连续 action chunks | 排除了“首个大规模、多资产、连续策略折叠数据”的宽泛声明；ClothDojo 应强调学习式代理自主组织行为，并同时生成展平和折叠轨迹 |
| [SoftMimicGen](https://softmimicgen.github.io/) | ICRA 2026 | 毛巾、绳索、软玩具、组织等多种可变形任务和多本体 | 1–3 条示范经分段、非刚性配准和片段变换，在新初态重新执行并筛选 | 已将少量示范扩展为新的物理 rollout，且生成时用完整节点、部署时用普通视觉；不能把示范迁移概括为重渲染 |
| [SIM1](https://arxiv.org/abs/2604.08544) | 2026 preprint | real-to-sim-to-real 的可变形物体策略学习 | 数字孪生与物理对齐；示范分段、diffusion 运动生成、轨迹优化、仿真执行和过滤 | 与 ClothDojo 同样把仿真作为数据引擎；差异在数据生成决策主体和是否依赖显式分段、优化与数字孪生 |
| [SimWeaver](https://simweaver.github.io/) | 2026 preprint | 五个独立可变形任务，含 T-shirt 展平与折叠 | 拓扑图、landmark、任务谓词和规划器合成轨迹；每任务 200 条；RGB VLA 零样本真机 | 已贯通仿真、连续轨迹、VLA 训练和真机验证；“完整平台”本身不足以区分，需强调学习式数据代理与人类纠正闭环 |
| [DexGarmentLab](https://wayrise.github.io/DexGarmentLab/) | NeurIPS 2025 | 2,500+ 资产、8 类服装、15 个灵巧操作任务 | HALO 利用结构对应，从单条专家示范迁移 affordance 与轨迹；策略输出双臂灵巧手连续动作 | 已有大规模服装资产、自动跨资产数据迁移和连续控制；差异不是资产数量，而是新状态上的学习式 rollout 及目标任务协议 |
| [LeHome](https://arxiv.org/abs/2604.22363) | ICRA 2026 | 家庭可变形物体环境；Garment Fling 与 Garment Fold 分别评测 | 仿真遥操作、轨迹重放与外观随机化、成功筛选；比较 ACT、DP 与多种 VLA | 已提供低成本双臂、视觉策略训练和 sim/real 验证；应比较数据来源、两项任务的起点及学习式数据扩展，而非只比较环境组件 |
| [Flat'n'Fold](https://eprints.gla.ac.uk/349229/) | ICRA 2025 | 44 件真实服装、8 类；1,212 条人类和 887 条机器人完整展平—折叠示范 | 真实多模态录制；主要 benchmark 为抓取点预测和子任务分解 | 已有自然、连续的人类行为，但并非可直接部署的机器人策略数据生成流程，也未以闭环策略成功率为主评测 |
| [π*₀.₆ / RECAP](https://www.pi.website/download/pistar06.pdf) | 2025 report | 真实机器人折叠多种衣物，并进行长时间连续运行 | 示范 SFT 后收集自主 rollout、奖励和人工纠正，再以离线 RL 训练 | 是 ClothDojo 数据闭环最直接的灵感来源；仿真版本的价值应落在特权信息、批量 rollout、自动判定和可控规模，而非把该闭环说成本文首创 |
| [χ₀ / Kai0](https://opendrivelab.com/kai0) | 2026 preprint | 多套真实双臂系统完成长程服装展平、折叠与悬挂 | 连续 action chunks；标准/heuristic DAgger、Stage Advantage 和模型组合 | 已证明真机连续服装行为与失败纠正可以规模化；同时说明不同平台和协议下的结果难直接复现与横比 |
| [DeMaVLA](https://arxiv.org/abs/2605.31286) | 2026 preprint | 单一模型覆盖 shirt、skirt、pants、towel 的真实折叠流程 | 大规模双臂预训练；human-gated DAgger 收集失败纠正；flow-matching action expert | 是“连续 VLA + 多类别服装 + HIL”的直接近邻；ClothDojo 的重点必须是仿真数据代理和可控数据实验，而不是连续控制本身 |
| [BiFold](https://arxiv.org/abs/2501.16458) | ICRA 2025 | 语言条件化的双臂折叠动作预测 | 从 VR-Folding 人类示范自动分段并生成语言标签；预测双臂 pick/place；完整折叠仍执行预定指令序列 | 数据明确包含人类 refinement action，并用 NOCS 划分服装区域；可支撑人类轨迹并非固定模板，但它不是连续机器人轨迹生成器 |

### 直接近邻形成的五条路线

1. **真实策略数据闭环**：π*₀.₆、χ₀、DeMaVLA、UniFolding。
2. **规则或规划驱动的仿真生成**：FoldNet、Instant-Fold、SimWeaver。
3. **示范迁移与生成模型扩展**：SoftMimicGen、SIM1、DexGarmentLab。
4. **真实或 VR 人类行为数据**：Flat'n'Fold、VR-Folding、BiFold、SSFold。
5. **环境与策略共同评测**：GarmentLab、DexGarmentLab、LeHome、SimWeaver。

ClothDojo 的定位必须同时跨过这五条路线，而不能只与早期 primitive 方法比较。

## 3. 服装操作方法：动作与任务演化

| 工作 | 代表性贡献 | 与本文关系 | 正文优先级 |
| --- | --- | --- | --- |
| [Learning Arbitrary-Goal Fabric Folding](https://arxiv.org/abs/2010.03209) | 用约一小时真实机器人自监督经验学习目标图像条件化 pick–place，主要是毛巾折叠 | 说明闭环学习和多目标折叠并非新概念，但对象和动作都较简化 | 背景 |
| [VisuoSpatial Foresight](https://arxiv.org/abs/2003.09044) | 从仿真随机交互学习视觉动力学，并通过规划完成多步 smoothing/folding | 代表模型学习与规划路线；不直接生成可供 VLA 训练的人类式轨迹 | 备选 |
| [FabricFlowNet](https://proceedings.mlr.press/v164/weng22a.html) | 用当前/目标深度图间的 flow 预测双臂抓放点 | 代表目标条件化 primitive，不直接预测机器人连续动作 | 建议引用 |
| [FlingBot](https://proceedings.mlr.press/v164/ha22a.html) | 学习选择双臂抓点，执行 pick–stretch–fling，从严重褶皱状态快速展平 | dynamic primitive 路线的奠基性近邻 | 必引 |
| [Learning Visible Connectivity Dynamics](https://proceedings.mlr.press/v164/lin22a.html) | 从部分点云推断可见连接图并学习动力学，用 MPC 展平布料 | 代表显式结构和 learned dynamics；与 SSFold/MEDOR 构成一条路线 | 备选 |
| [MEDOR](https://www.roboticsproceedings.org/rss18/p011.pdf) | 重建被遮挡的完整布料 mesh，再用 mesh dynamics 规划展平/规范化 | 与 NOCS、遮挡推理相关，但目标是状态估计和规划 | 建议引用 |
| [SpeedFolding](https://arxiv.org/abs/2208.10552) | 组合 fling、drag、pick–place 等 primitive，从褶皱上衣完成 smoothing 和折叠 | 是完整流水线和多 primitive 的强基线；折叠步骤仍由指令与 heuristic 决定 | 必引 |
| [DextAIRity](https://dextairity.cs.columbia.edu/) | 使用主动气流执行布料展开和袋子打开 | 表明动作库可以扩展到非接触操作，也说明“人类式连续行为”不是唯一有效路线 | 背景 |
| [Cloth Funnels](https://clothfunnels.cs.columbia.edu/) | 学习 canonicalized-alignment，再用关键点和固定 heuristic 折叠 | 清楚体现“学习规范化 + 脚本下游”的设计边界 | 必引 |
| [QDP](https://sites.google.com/view/qdp-srl) | 除抓放点外继续优化 primitive 的速度、轨迹等动态参数 | 用于避免把 primitive 误写成完全固定或完全开环 | 备选 |
| [UniFolding](https://proceedings.mlr.press/v229/xue23b.html) | 将展平和折叠决策统一到一个 primitive policy，并使用 VR 示范与 HIL | 与任务、数据和反馈三方面均直接相关 | 必引 |
| [APS-Net](https://proceedings.mlr.press/v267/zhou25q.html) | 将 coverage、朝向和关键点对齐统一为 garment standardization 目标 | 代表近期 multi-primitive 展平/规范化路线 | 建议引用 |
| [Learning to Fold Real Garments with One Arm](https://research.google/pubs/learning-to-fold-real-garments-with-one-arm-a-case-study-in-cloud-based-robotics-research/) | 通过云端真实机器人系统比较多类学习与解析算法，从褶皱 T-shirt 折叠 | 直接支撑真机实验难以复用、共享硬件可改善可比性的背景 | 备选 |
| [BiFold](https://arxiv.org/abs/2501.16458) | 语言条件化双臂 pick/place，并从人类 VR 轨迹自动生成指令标签 | 与自然人类行为、历史上下文和 NOCS 区域语义均有关 | 建议引用 |
| [SSFold](https://arxiv.org/abs/2411.02608) | 真实人类示范、图像与 graph dynamics 结合，从褶皱状态完成目标折叠 | 是完整任务的直接方法近邻 | 必引或重点核验 |

## 4. 环境、资产、数据集与 benchmark

| 工作 | 它主要提供什么 | 不能被误解为什么 | 正文优先级 |
| --- | --- | --- | --- |
| [CLOTH3D](https://arxiv.org/abs/1912.02792) | 大规模合成穿衣人体与多样服装 mesh，是许多后续资产的来源 | 不是机器人交互 benchmark，也不直接提供双臂操作轨迹 | 背景 |
| [SoftGym](https://proceedings.mlr.press/v155/lin21a.html) | 可变形物体 RL benchmark，包含 cloth spread/fold 等任务 | picker 是理想化动作抽象，不能等同于真实机器人连续控制平台 | 建议引用 |
| [DeformableRavens](https://arxiv.org/abs/2012.03385) | 12 个绳、布、袋任务；scripted oracle 生成 pick–place 示范 | 不提供多类别服装的连续双臂轨迹 | 备选 |
| [DaXBench](https://openreview.net/pdf?id=1NAzMofMnWl) | JAX 可微物理和多类可变形任务，比较规划、IL 与 RL | 核心是算法 benchmark，不是服装数据生成系统 | 背景 |
| [ThinShellLab](https://proceedings.iclr.cc/paper_files/paper/2024/file/bb01ae312268181758299e4ca75837e5-Paper-Conference.pdf) | 薄壳材料可微仿真、系统辨识与多种操作任务 | 对象包括纸张和薄片，不能直接代表多样服装任务 | 背景 |
| [ClothesNet](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_ClothesNet_An_Information-Rich_3D_Garment_Model_Repository_with_Simulated_Clothes_ICCV_2023_paper.pdf) | 4,400 个服装模型、丰富结构标注和模拟交互环境 | 其中约 3,051 个为处理后的可仿真资产；RL baseline 使用特权关键顶点与控制点，不是视觉双臂策略 | 建议引用 |
| [VR-Folding / GarmentTracking](https://openaccess.thecvf.com/content/CVPR2023/papers/Xue_GarmentTracking_Category-Level_Garment_Pose_Tracking_CVPR_2023_paper.pdf) | 约 9.7K 条 VR 人类展平/折叠视频和逐帧 mesh、手部、点云/NOCS | 主要用于姿态跟踪；折叠参与者遵循类别级规则，且原始数据不直接提供机器人接近轨迹 | 建议引用 |
| [GarmentLab](https://proceedings.neurips.cc/paper_files/paper/2024/file/15f80ec0fed53885d2ca6272edb96ede-Paper-Conference.pdf) | 服装专用模拟器、11 类服装和 20 个任务，并评测视觉、RL、IL 与 sim-to-real | unfold 与 fold 使用不同起点；环境广度不等于完整连续策略 benchmark | 必引 |
| [Cloth Sim-to-Real Benchmark](https://arxiv.org/abs/2310.09543) | 在真实与多个模拟器中复现规定 cloth 动作并比较动力学轨迹 | 评测模拟器 fidelity，不评测学习策略能否完成长程任务 | 建议引用 |
| [Flat'n'Fold](https://eprints.gla.ac.uk/349229/) | 真实人类与机器人完整展平—折叠多模态数据 | 主要是录制数据和离线任务，不是自动生成的可扩展成功 rollout | 必引 |
| [DexGarmentLab](https://arxiv.org/abs/2505.11032) | 大规模资产、灵巧手任务、结构对应迁移和连续策略 | 任务范围很广，但不是以展平和折叠数据闭环为唯一核心 | 必引 |
| [RGBench](https://ojs.aaai.org/index.php/AAAI/article/download/38894/42856) | 大规模服装资产、实测物性与真实—仿真动力学 fidelity | 规定动作的动力学复现不等同于闭环策略 benchmark | 建议引用 |
| [LeHome](https://arxiv.org/abs/2604.22363) | 家庭可变形物体、低成本本体、遥操作、视觉策略训练和 sim/real | 不能概括成“只提供环境”；其确实形成了数据—训练—评测链路 | 必引 |
| [LeHome Challenge 2026](https://2026.ieee-icra.org/program/competitions/) | 四类服装、统一低成本双臂平台、仿真与现场真机竞赛 | 与 LeHome 论文的六任务实验不是同一协议 | 备选 |
| [ICRA 2024 Cloth Competition Dataset](https://arxiv.org/abs/2508.16749) | 34 件服装、679 次真实悬挂展开抓取尝试 | 只评测一次 in-air grasp selection，不是桌面长程展平/折叠 | 背景 |

## 5. 可扩展数据生成与人类纠正：服装领域之外的基础

| 工作 | 核心机制 | 与 ClothDojo 的关系 | 正文优先级 |
| --- | --- | --- | --- |
| [DAgger](https://mlanthology.org/aistats/2011/ross2011aistats-reduction/) | 在学习策略自身访问的状态分布上聚合专家动作 | SFT 后 rollout、再对失败状态补充纠正的理论起点 | 建议引用 |
| [IWR](https://sites.google.com/stanford.edu/iwr) | 用远程遥操作收集干预，并对干预样本加权学习 | 代表从完整示范转向少量纠正的真实机器人方法 | 备选 |
| [HIL-SERL](https://hil-serl.github.io/) | 示范、奖励分类器、自主 RL 与人工 intervention 联合训练 | 与自动成功指标和必要人工介入的闭环相近，但优化算法与 ClothDojo 不同 | 建议引用 |
| [MimicGen](https://mimicgen.github.io/) | 把少量刚体任务示范分段并迁移到新场景，生成 50K+ 演示 | 自动数据生成的基础参照；其对象和接触结构主要是刚体 | 建议引用 |
| [DexMimicGen](https://dexmimicgen.github.io/) | 将 MimicGen 扩展到双臂与灵巧操作，从 60 条源示范生成 20K+ 演示 | 与双臂数据规模化直接相关，但仍依赖 object-centric 分段和轨迹变换 | 建议引用 |
| [IntervenGen](https://doi.org/10.1109/IROS58592.2024.10801523) | 从少量人工 intervention 自动合成更多纠正数据 | 与 HIL 数据扩展相关，但面向刚体姿态扰动，不是布料状态变化 | 备选 |
| [RoboCopilot](https://arxiv.org/abs/2503.07771) | 支持双臂策略与人类之间无缝切换的交互式模仿学习系统 | 与 HIL 遥操作基础设施相关；不是自主轨迹生成方法 | 备选 |
| [SoftMimicGen](https://arxiv.org/abs/2603.25725) | 对可变形物体示范做非刚性配准、片段选择和动作变换 | 从一般数据生成进入 ClothDojo 任务范围的最直接桥梁 | 必引 |
| [RECAP / π*₀.₆](https://www.pi.website/download/pistar06.pdf) | 示范、on-policy rollout、奖励、人工纠正和离线 RL 构成 VLA 数据闭环 | ClothDojo 流程的直接方法动机；必须准确写成“受启发”，不能宣称闭环形式首创 | 必引 |

这里应严格区分三种“自动生成”：

- **轨迹迁移**改变场景或对象后重新执行已有行为结构；
- **规划/脚本生成**在新状态执行新轨迹，但动作逻辑由外部程序规定；
- **策略 rollout**由已学习策略根据中间观测继续决定后续动作。

ClothDojo 的核心差异主要位于第三类，而不是笼统的“生成了新物理轨迹”。

## 6. 规范空间、稠密对应与特权信息

| 工作 | 特权或结构表示怎样使用 | 对 ClothDojo 的含义 | 正文优先级 |
| --- | --- | --- | --- |
| [NOCS](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.pdf) | 为同类刚体实例定义共享规范坐标，并预测像素到规范空间的对应 | NOCS 概念来源；ClothDojo 不拥有该表示本身的新颖性 | 必引 |
| [GarmentNets](https://openaccess.thecvf.com/content/ICCV2021/papers/Chi_GarmentNets_Category-Level_Pose_Estimation_for_Garments_via_Canonical_Space_Shape_ICCV_2021_paper.pdf) | 在服装类别共享的 canonical space 中补全被遮挡的三维表面 | 已把 NOCS 引入服装；本文差异只能落在对 VLA 数据代理的使用方式 | 必引 |
| [GarmentTracking](https://openaccess.thecvf.com/content/CVPR2023/papers/Xue_GarmentTracking_Category-Level_Garment_Pose_Tracking_CVPR_2023_paper.pdf) | 预测和跟踪服装逐点 NOCS，恢复跨帧完整形态 | 说明 NOCS 可用于动态服装状态，不只是静态姿态估计 | 建议引用 |
| [UniGarmentManip](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_UniGarmentManip_A_Unified_Framework_for_Category-Level_Garment_Manipulation_via_Dense_CVPR_2024_paper.html) | 自监督学习跨实例、跨形变的稠密对应，再迁移 manipulation point | 与 ClothDojo 都利用跨资产表面位置参照，但它迁移操作点而非训练自由连续的数据代理 | 必引 |
| [BiFold](https://arxiv.org/abs/2501.16458) | 在 VR-Folding 数据中用 NOCS 阈值划分袖口、衣身等区域并自动生成语言标签 | 说明 NOCS 已参与服装行为数据加工；本文重点不是首次用 NOCS 生成监督 | 建议引用 |
| [FoldNet](https://arxiv.org/abs/2505.09109) | GT keypoint 仅供教师和失败检测，最终策略从单目 RGB 执行 | 已有“生成器用特权信息、部署策略不用”的直接先例 | 必引 |
| [SoftMimicGen](https://arxiv.org/abs/2603.25725) | 完整 deformable nodes 用于非刚性配准，最终策略从图像执行 | 再次说明 privilege-to-RGB 范式不是新颖点 | 必引 |
| [Domain Randomization](https://arxiv.org/abs/1703.06907) | 通过随机化渲染训练视觉模型跨越 sim-to-real 外观差异 | 重渲染属于视觉覆盖扩展，不等同于新增物理行为 | 背景 |

ClothDojo 目前较准确的技术定位是：**不修改 VLA 结构，把规范表面坐标编码成图像条件，仅用于提高学习式数据代理在仿真中的轨迹生成能力；成功物理轨迹再重渲染为普通 RGB，服务下游策略训练。** 其中每个组成元素都有前例，潜在贡献来自它们围绕学习式数据代理形成的完整用途与实证，而不是任何单个组件的“首次”。

## 7. 面向 8 页正文的初步分级

### A 级：正文原则上不能漏

- 任务与 primitive：FlingBot、Cloth Funnels、SpeedFolding、UniFolding。
- 连续长程真实策略：π*₀.₆、χ₀、DeMaVLA。
- 可扩展服装数据：FoldNet、Instant-Fold、SoftMimicGen、SIM1、SimWeaver。
- 平台与真实数据：GarmentLab、DexGarmentLab、LeHome、Flat'n'Fold。
- 规范几何：NOCS、GarmentNets、UniGarmentManip。

这些工作不必各写一句。最终可以按路线聚合引用，但撰写前必须逐项完成事实核验。

### B 级：根据段落论点选用

- SSFold、BiFold、APS-Net、FabricFlowNet、MEDOR。
- SoftGym、ClothesNet、RGBench、Cloth Sim-to-Real Benchmark、VR-Folding。
- DAgger、MimicGen、DexMimicGen、HIL-SERL。

### C 级：只用于保证视野完整

- DaXBench、ThinShellLab、DeformableRavens、VisuoSpatial Foresight、Visible Connectivity Dynamics、DextAIRity、QDP、早期真实自监督折叠。

## 8. 当前调研对新颖性表述的直接影响

以下宽泛声明已经不安全：

- “首次从褶皱状态连续完成展平和折叠”；UniFolding、SpeedFolding、SSFold、π 系列和 χ₀ 均提供反例。
- “首次使用人类纠正改进服装策略”；UniFolding、π*₀.₆、χ₀、FoldNet、DeMaVLA 已覆盖不同形式的 HIL/DAgger。
- “首次使用仿真特权信息生成普通视觉策略数据”；FoldNet、SoftMimicGen、SIM1、SimWeaver 已采用这一范式。
- “首个大规模多资产折叠数据集”；Instant-Fold 已报告约 80K 轨迹和 360 个服装 mesh，DexGarmentLab 资产更多。
- “首次提供完整服装仿真研究平台”；GarmentLab、DexGarmentLab 和 LeHome 已覆盖环境、采集、训练和评测的不同组合。

目前更值得验证的窄声明是：

> ClothDojo 是否是首个在动态双臂服装仿真中，以从人类交互训练的连续 VLA 作为数据代理，通过批量 rollout、任务指标筛选和 HIL 自主扩展展平与折叠成功轨迹，并将这些轨迹重渲染为普通 RGB 下游训练数据的系统。

这只是候选声明，还不能直接写成“首个”。正式定稿前需要围绕这一完整交集再做一次针对性检索，并等待 500 件服装、每件各 10 条展平和 10 条折叠、共 10,000 条成功轨迹的最终数据审计。

## 9. 下一轮需要深挖的问题

1. FoldNet 的随机褶皱初态与 ClothDojo 初态在视觉严重程度和任务阶段上是否真正不同。
2. Instant-Fold 的程序化生成是否包含基于执行结果的局部调整，还是只改变预设 primitive 的参数和组合。
3. SSFold 的 800 条人类示范、服装类别和“full task”是否使用同一策略与同一评测协议。
4. SoftMimicGen 对 towel unfolding/folding 的片段选择和失败过滤怎样定义，是否会产生源示范未出现的行为顺序。
5. SIM1 和 SimWeaver 的数据生成器在多次接触失败后能否自主改变操作结构。
6. LeHome 正式论文、Challenge 和参赛方案中的数据规模、初态与自动成功判据必须分开记录。
7. π*₀.₆、χ₀、DeMaVLA 的“correction”“intervention”“DAgger”分别怎样进入训练，不能统称为相同算法。
8. ClothMate 正式全文中的多类别展平、局部修正和数据规模仍需补充核验。

