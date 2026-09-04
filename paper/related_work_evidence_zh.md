# ClothDojo 相关工作事实表

> 本文件是写作前的内部证据表，不是论文正文。更新时间：2026-09-03。只记录论文、正式项目页、官方数据页或竞赛仓库可以直接核验的事实；最后一列留给作者判断。`待核` 表示公开材料尚不足以支持更细的结论。

## 0. 比较口径

为了避免把不同问题混为一谈，后续概括至少区分以下四点：

1. **任务连续性**：展平与折叠是两个独立 task、由人工/规则衔接的 pipeline，还是从同一严重褶皱初态连续完成。
2. **动作接口**：策略只选择 primitive 及其参数，还是直接预测连续关节/末端动作；`closed-loop` 还要区分 primitive 间重规划、chunk 间重规划和 primitive 内反馈控制。
3. **数据扩展的实质**：新状态上的新物理 rollout、既有轨迹的几何迁移、动作片段重组、还是仅改变外观的重渲染。
4. **特权信息的使用者**：最终策略使用特权状态，还是仅由数据生成器/教师使用，最终策略仍从普通视觉执行。

## 1. 声明—文献矩阵

这里先回答“正文每句话由谁支持”。`直接` 表示论文的方法或实验可直接支撑；`综合` 表示由多篇协议对照得到，正文应使用“许多”“一类”等限定；`作者观察` 表示应由 ClothDojo 自身轨迹或实验展示，不能伪装成已有论文结论。

| ID | 拟写声明 | 相关原始文献 | 证据性质与写作边界 |
| --- | --- | --- | --- |
| P1 | 一类代表性服装方法将双臂控制压缩为少量可参数化 primitive。 | FlingBot；Cloth Funnels；SpeedFolding；UniFolding；APS-Net；DeformPAM | **直接/综合。** 各文均明确给出 action primitive；不能扩大成“所有现有方法”。 |
| P2 | 策略预测 primitive 类型、抓取点或放置点，连续轨迹由预定义控制器执行。 | FlingBot；Cloth Funnels；SpeedFolding；UniFolding；FabricFlowNet；DeformPAM | **直接。** 不同论文未必同时预测“类型”和“参数”，正文应用“通常选择……并/或预测……”。 |
| P3 | 这种抽象缩小了学习动作空间，并已用于展开、规范化和折叠。 | FlingBot；Cloth Funnels；SpeedFolding；UniFolding；APS-Net | **直接/综合。** 任务和指标不同，只能说明有效性，不能横向比较数值。 |
| P4 | 学习策略层面的反馈主要发生在 primitive 之间，primitive 内反馈通常是预写低层逻辑。 | FlingBot；SpeedFolding；UniFolding；Cloth Funnels | **直接/综合。** FlingBot 的拉伸有视觉终止，SpeedFolding/UniFolding 有力阈值；“primitive 内完全无反馈”不成立。 |
| P5 | 固定 primitive 库限定可表达的行为；加入新行为需继续设计动作及参数。 | SpeedFolding；UniFolding；DeformPAM | 前半句是接口定义的**直接推论**；“维护成本高”未被这些论文量化，应作为作者判断而非文献事实。 |
| P6 | 同一物理目标允许多条有效轨迹，具体动作依赖遮挡、接触结果和持续变化的中间状态；人类会组合难以枚举的恢复行为。 | Robot Olympics；ClothDojo 人类遥操作轨迹 | 博客支持“物理技能难以通过指令或程序完整表达、需要丰富行为数据”的**概念判断**；具体服装行为仍是**作者观察**，应由本文轨迹举证。不要把“随机”写成能力本身。 |
| G1 | 通用 visuomotor/VLA 已直接预测连续动作或 action chunks，并在衣物任务上评测。 | π₀；FAST；DexVLA；FoldNet；LeHome；π*₀.₆；χ₀/Kai0；DeMaVLA | **直接。** π₀.₅ 训练混合含折衣，但其论文没有独立的折衣主评测，不能单独作为此声明证据。 |
| G2 | 不依赖上述任务专用 primitive 的衣物控制是可行的。 | π₀；FAST；DexVLA；FoldNet；LeHome；SimWeaver；π*₀.₆；χ₀/Kai0；DeMaVLA | **直接/综合。** 这些方法的任务和本体不同，不构成公平性能排名。 |
| G3 | 不同通用策略的衣物结果难直接比较。 | π₀；FAST；DexVLA；FoldNet；LeHome；SimWeaver；π*₀.₆；χ₀/Kai0；DeMaVLA；标准化综述 | **协议综合。** 差异确实落在本体、初态、服装集、时限和成功定义上。 |
| S1 | 通用可变形物体平台提供布料、绳索、流体或薄壳任务。 | SoftGym；DeformableRavens；DaXBench；ThinShellLab | **直接。** 它们的物理表示和动作抽象不同，不能统称为服装 benchmark。 |
| S2 | 服装专用平台扩展了资产、机器人和交互场景。 | ClothesNet；GarmentLab；DexGarmentLab；LeHome | **直接。** ClothesNet 的核心还包括感知，DexGarmentLab 是 15 个任务/8 类服装。 |
| S3 | 已有工作专门测量真实—仿真布料动力学差距。 | Cloth Sim-to-Real Benchmark；RGBench | **直接。** 它们复现规定动作并比较几何轨迹，不等于评测学习策略的任务成功。 |
| S4 | 新近平台开始把仿真采集、连续视觉策略训练和 rollout 评测连起来。 | LeHome；FoldNet；DexGarmentLab；SIM1；SimWeaver | **直接。** LeHome 与 SimWeaver 是最适合正文概括的平台级证据。 |
| S5 | 各平台的资产、初态、动作接口与成功条件不统一。 | 本表第 3 节全部 benchmark；标准化综述 | **协议综合。** 这是 ClothDojo benchmark 动机的核心，而非单篇论文的自我声明。 |
| S6 | 许多服装 benchmark 将展平与折叠设为不同任务或不同起点，尚无共享的连续协议。 | ClothesNet；GarmentLab；DexGarmentLab；LeHome；SimWeaver | **有限支持。** 只能说“许多 benchmark”；SoftGym、SpeedFolding、UniFolding、π₀ laundry 和 χ₀ 提供反例，不能写“此前都分开评测”。 |
| D1 | 仿真粒子/顶点状态常用于 reward 或 success evaluator。 | SoftGym；ClothesNet；GarmentLab；LeHome；FoldNet；SimWeaver | **直接。** segmentation 与 NOCS 不应无依据地并列成“常用 reward”。 |
| D2 | GT 几何、关键点、稠密对应或完整节点状态已被用于教师和数据生成。 | UniFolding；UniGarmentManip；MetaFold；DexGarmentLab；FoldNet；SoftMimicGen；SIM1；SimWeaver | **直接。** 各自用法不同，正文应按用途归类而非堆名词。 |
| D3 | 特权状态可只服务生成器，最终策略仍从普通图像部署。 | FoldNet；SoftMimicGen；SIM1；SimWeaver；LeHome | **直接。** 因此前人已有该范式，ClothDojo 不能把它本身作为新颖性。 |
| D4 | 新物理 rollout、轨迹变换/重组与仅外观重渲染提供不同的数据增量。 | FoldNet；LeHome；SoftMimicGen；SIM1；SimWeaver | **机制综合。** 这是组织 2.3 的关键抽象，也直接支撑 ClothDojo 的“物理覆盖/视觉覆盖”区分。 |
| D5 | 恢复数据可来自当前策略访问到的失败状态。 | UniFolding；FoldNet；χ₀/Kai0；DeMaVLA | **直接。** 分别是 HIL、自动 KG-DAgger 或 human-gated/heuristic DAgger；不可混称为相同算法。 |

### 声明审计结论

- 可保留：P1–P3、G1–G3、S1–S5、D1–D5，但需要上表中的限定词。
- 必须改写：P4 从“没有反馈”改成“学习重决策与预写低层反馈的边界”；S6 从“展平折叠通常独立”改成“已有连续系统，但 benchmark 尚未形成共享协议”。
- 不应引用外部论文代替本文证据：P6 的具体人类行为，应由 ClothDojo 遥操作轨迹截图或轨迹分析支持。

## 2. 服装展平与折叠方法

| 工作 | 实际任务与初态 | 策略输出与机器人动作 | 在线反馈与恢复 | 数据与评测证据 | 核验来源 | 作者判断 |
| --- | --- | --- | --- | --- | --- | --- |
| FlingBot（CoRL 2021） | 任意褶皱状态到最大 coverage，核心任务是 unfolding | 顶视 RGB value map 选择双臂抓取位置；机器人执行预定义 pick–stretch–fling | 每个 fling 后重新观测；stretch 阶段利用前视布料轮廓决定终止，但 fling 轨迹不是策略连续生成 | 仿真矩形布训练并测试未见布料；真机包含矩形布和 T-shirt | [论文](https://proceedings.mlr.press/v164/ha22a/ha22a.pdf) | — |
| Cloth Funnels（ICRA 2023） | 任意服装状态到 canonicalized-alignment；折叠/熨烫作为下游展示 | spatial action maps 在 fling 与 pick-and-place primitive 间选择并预测空间参数；折叠使用关键点与固定 heuristic | 每个 primitive 后重新观测；学习部分不直接产生 primitive 内的连续轨迹 | 仿真按类别训练并测试未见 mesh；真机展示 canonicalization、折叠和熨烫 | [论文](https://openreview.net/pdf/fd71974e09c9c72e2b7ef8899444b115a99400a4.pdf)、[项目页](https://clothfunnels.cs.columbia.edu/) | — |
| SpeedFolding（IROS 2022） | 随机褶皱服装先 smoothing，再按用户折线 folding | 顶视 RGB-D；BiMaMa-Net 选择 fling、pick-and-place 或 drag，并预测双臂平面抓取位姿；fold/move 为 heuristic | primitive 之间闭环选择；fling 的速度和运动轨迹固定，stretch 使用力阈值 | 4,300 个真机人工标注或自监督动作；评测未见颜色、形状和刚度服装 | [论文](https://h2t.iar.kit.edu/pdf/Avigal2022.pdf) | — |
| UniFolding（CoRL 2023） | 随机褶皱初态到 unfolding 和 folding，覆盖长袖与短袖上衣 | partial point cloud；UFONet 预测 primitive 类型及双臂 pick/place points；包含 fling、pick-and-place，必要时 drag/mop | primitive 之间重新决策；通过真机 HIL 收集策略失败附近的数据 | 仿真 VR 示范、自监督数据和真机 HIL；真机测试 20 件服装 | [论文](https://proceedings.mlr.press/v229/xue23b/xue23b.pdf)、[项目页](https://unifolding.robotflow.ai/) | — |
| ClothMate（RA-L 2026） | 任意状态下的多类别服装展平 | 在 canonicalized-aligned 状态学习逐点 fling response，再投影至任意状态；静态 pick-and-stretch 局部修正 | 摘要能确认局部修正机制；primitive 内反馈、数据划分和失败恢复待全文核验 | 摘要报告五类服装上的联合训练与评测，并比较数据效率、coverage 和步骤数 | [IEEE 条目](https://ieeexplore.ieee.org/document/11248822)、[DOI](https://doi.org/10.1109/LRA.2025.3632758) | — |
| APS-Net（ICML 2025） | 褶皱服装的 unfolding 与 standardization；目标同时考虑 coverage、朝向和规范对齐 | 顶视 RGB-D spatial action maps 选择 dynamic fling 或 pick-and-place 及空间参数 | primitive 间闭环选择；不直接生成 primitive 内连续控制 | 仿真主要结果针对长袖；真机用后续 folding 说明 standardization 的用途 | [PMLR](https://proceedings.mlr.press/v267/zhou25q.html)、[论文](https://openreview.net/pdf?id=pT4gJ0PgGD) | — |
| FoldNet（RA-L 2026） | 从随机旋转、翻面和褶皱的初态完成预定义多阶段 folding；不包含先将严重褶皱服装展平的独立目标 | 最终 Diffusion Policy 从单目 RGB 直接输出双臂 8D 动作序列；生成教师按 GT keypoint 产生分阶段抓放与插值轨迹 | KG-DAgger 在策略 rollout 中用 GT keypoint 检测抓取失败并接管重试；最终部署不使用 keypoint | 4 类合成服装；15K 轨迹、约 2M 图像—动作对；报告仿真和零样本真机 folding | [论文](https://arxiv.org/html/2505.09109)、[项目页](https://pku-epic.github.io/FoldNet/) | — |
| χ₀ / Kai0（2026 preprint） | 真实机器人长程任务；Task A 连续 flattening→三次 folding，Task B/C 还包含取衣、整理、悬挂等阶段 | π₀.₅/π₀ 直接预测 action chunks；论文同时比较 absolute/delta joint control、RTC、temporal ensembling 和 chunk-wise smoothing | 使用标准/heuristic DAgger 收集失败恢复，Stage Advantage 训练与高频控制对齐；不是 primitive 策略 | 两套双臂平台；论文报告每个 task 约 20 h 示范、约 2.7K–3.5K episodes，并进行 24 h 压力测试；公开数据后续扩展到 20,909 episodes | [论文](https://arxiv.org/html/2602.09021)、[代码与数据](https://github.com/OpenDriveLab/kai0)、[数据说明](https://github.com/OpenDriveLab/kai0/blob/main/docs/dataset.md) | — |
| π₀（2024） | 一项 T-shirt fold 从基本平铺状态开始；laundry task 从衣篮中的随机褶皱衣物开始，连续完成抓取、展平、折叠与堆叠 | 多视角 RGB、语言、本体状态；flow matching action expert 生成连续 action chunks | 在 50 Hz 任务中每约 0.5 s 重新推理并执行下一段 25-step chunk，不是整条轨迹一次性开环生成 | 折衣评测为少量上衣、每件少量 trials；laundry 按抓取/展平/折叠/堆叠分项计分，说明可行性但不是标准 benchmark | [论文](https://www.pi.website/download/pi0.pdf) | 直接支持 G1/G2；也说明“连续展平—折叠此前不存在”是错误的。 |
| FAST（2025） | 沿用 π₀ 的 T-shirt folding 与 laundry 等高频任务，用来检验离散 action tokenizer | 将动作先做 DCT，再以 BPE token 化并自回归预测约 1 s action chunks | action chunk 间重新推理；论文比较 tokenization/训练效率，而非提出新的服装任务协议 | 服装结果证明 autoregressive VLA 也能执行高频连续控制，但不应与 π₀ 当作两个独立 benchmark 计数 | [论文](https://www.pi.website/download/fast.pdf) | 支持 G1；引用时最好写 π₀/FAST，而非暗示两个独立数据集。 |
| π₀.₅（2025） | 预训练任务混合包含 fold laundry/linen；论文的 open-world 主评测是家庭整理，其中 laundry 任务主要是把地面衣物放入篮中 | π₀ 系列 flow-matching action chunks，强调 heterogeneous co-training 与 high-level semantic knowledge | 具备闭环执行，但论文没有单列严重褶皱到折叠的评测协议 | 可说明通用策略训练数据覆盖衣物；**不能单独作为折衣成功的直接证据** | [论文](https://arxiv.org/abs/2504.16054) | 正文 2.1 不必单独引用；在 SimWeaver/SIM1 中注明其作为下游 backbone 即可。 |
| DexVLA（2025） | shirt folding 从基本平铺状态开始；laundry 从随机褶皱衣篮开始并连续抓取、展平、折叠、堆叠 | 视觉语言骨干加 diffusion action expert，直接生成连续动作并在 60 Hz 系统执行 | 通过 action chunk 周期性更新；不是 task-specific primitive policy | 100 h 跨本体预训练；服装只占评测的一部分，任务以阶段分数或少量 trials 报告 | [论文](https://arxiv.org/abs/2502.05855) | 直接支持 G1/G2，不能与 ClothDojo 的固定资产/初态闭环成功率直接比较。 |
| π*₀.₆（2025） | 标准 laundry 从篮中衣物开始，连续展平、折叠、堆叠；另有更多衣物类型及指定平铺 T-shirt 折叠设置 | 50 Hz 关节/夹爪 action chunks；在 π₀.₆ 基础上用 RECAP 离线 RL 学习成功与失败数据 | 自主 trials 与人工 correction 形成闭环数据飞轮，部署仍为连续 action policy | 标准 laundry 报告数百次评测，展示规模化真实闭环证据；协议仍由该系统独立定义 | [论文](https://arxiv.org/abs/2511.14759)、[项目页](https://www.pi.website/blog/pistar06) | 是 G1–G3 和 D5 的重要近邻。 |
| DeMaVLA（2026 preprint） | 一套真实双臂策略连续完成从篮中取出随机投放衣物、在桌面展开/铺平并折叠；统一 checkpoint 覆盖 shirt、skirt、pants、towel | VLM + 精简的 flow-matching action expert 直接生成 action chunks，并以 RTC 异步执行 | 在真实机器人上进行 human-gated DAgger，针对抓取、对齐和中间折叠状态的失败收集纠正轨迹 | 约 5,000 h 双臂预训练，后训练 folding 数据 37 h；四任务各 20 次真机评测，5 min 超时或落桌判失败，最终形态使用 task-specific completion criteria | [论文](https://arxiv.org/abs/2605.31286) | 当前最直接支持 G1–G3/D5 的专用 VLA 工作之一；协议仍是作者自建 benchmark，最终形态判据细节不足以和 ClothDojo 数值对齐。 |

### 方法层面的补充候选

| 工作 | 任务与方法 | 数据与评测 | 与正文声明的关系 | 核验来源 |
| --- | --- | --- | --- | --- |
| FabricFlowNet（CoRL 2021） | 输入当前/目标深度图，以 optical flow 建立对应并确定 place point，PickNet 再预测双臂 pick points，最后执行固定 pick-and-place | 用仿真随机动作生成训练对；真机主要为方布目标形状、trial 数有限，且假定折叠前已较平整 | 支持 P2；说明 primitive 不必显式预测“动作类型”，也可只学习其空间参数 | [论文](https://proceedings.mlr.press/v164/weng22a.html) |
| UniGarmentManip（CVPR 2024） | 学习跨实例、跨形变的服装稠密视觉对应，再迁移示范中的 manipulation points；展开、折叠、悬挂分别执行 | PyFlex 中约 500 tops、600 trousers、600 dresses，并做仿真与真机验证；各任务初态与成功指标不同 | 支持 D2；与 NOCS 都处理跨资产对应，但它主要迁移操作点，而不是训练可自由产生连续行为的数据代理 | [论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_UniGarmentManip_A_Unified_Framework_for_Category-Level_Garment_Manipulation_via_Dense_CVPR_2024_paper.pdf) |
| MetaFold（IROS 2025） | 从点云和语言生成后续 cloth trajectory，再由 ManiFoundation 推断接触点与力方向并迭代执行 | 基于 ClothesNetM 的 1,210 件服装；在 DiffClothAI 中由人工起终 keypoints 与 heuristic 生成约 3,376 条 point-cloud trajectories | 支持 D2/D4；是特权几何生成折叠轨迹的近邻，但不是从已训练策略失败分布收集 recovery | [论文](https://arxiv.org/abs/2503.08372)、[项目页](https://meta-fold.github.io/) |
| DeformPAM（2024 preprint） | 从分割 3D point cloud 预测 primitive 类型/参数，以偏好模型排序；连续轨迹仍由 OMPL/PyBullet 与手写规则规划 | 在三类长程可变形任务上做真机评测，其中 T-shirt unfolding 使用 FlingBot 类 primitive；论文也比较 primitive-free Diffusion Policy | 直接支持 P1/P2/P5，并由作者明确把降维搜索空间作为 primitive 的理由 | [论文](https://arxiv.org/abs/2410.11584) |
| Visuotactile Affordances（CoRL 2023） | 抓住布料后，策略利用视觉和触觉反馈沿边缘滑动，学习局部闭环控制 | 真机布料边缘跟随与重抓取实验；任务范围比完整展平—折叠窄 | primitive 内闭环反馈的边界反例；不必列入正文主引用，但用于避免 P4 的绝对表述 | [论文](https://proceedings.mlr.press/v205/sunil23a.html) |

## 3. 环境、数据集与 benchmark

| 工作 | 它实际 benchmark 什么 | 服装任务及连续性 | 策略/动作接口 | 数据、划分与评测事实 | 核验来源 | 作者判断 |
| --- | --- | --- | --- | --- | --- | --- |
| SoftGym（CoRL 2020） | 可变形物体 RL 算法；不是专门的服装通用策略 benchmark | 10 个环境含 SpreadCloth、FoldCloth、FoldCrumpledCloth；后者本身就是从褶皱状态折叠的反例，但仍是理想化单项环境 | 主要使用可自由移动并吸附最近粒子的理想化 picker，抽象低层机器人控制 | 800 个训练、200 个评测 variation；比较 ground-truth state、低维状态与视觉观测下多种 RL 算法，reward 由粒子状态计算 | [论文](https://proceedings.mlr.press/v155/lin21a/lin21a.pdf)、[代码](https://github.com/Xingyu-Lin/softgym) | 支持 S1/D1；约束 S6。 |
| DeformableRavens（2020） | PyBullet 中 12 个 1D/2D/3D 可变形任务的 imitation-learning benchmark | 布料任务含 flatten/cover 等，未提供多类别服装连续折叠协议 | UR5+suction；策略预测平面 pick/place pose，机器人执行预定义 primitive，每步后重观测 | 以 scripted expert 生成示范，主要检验 Transporter 类方法对柔性对象的泛化 | [论文](https://arxiv.org/abs/2012.03385) | 支持 S1，也再次说明“有机器人模型”不等于低层连续策略 benchmark。 |
| DaXBench（ICLR 2023） | JAX 可微物理与 9 个液体、绳索、布料任务 | FoldCloth、FoldTshirt 多从平铺状态开始；UnfoldCloth 从部分折叠状态开始，任务分别定义 | 同时含低层速度控制与降维 pick-place macro；不同 task 的 horizon 很短且不统一 | 比较 RL、imitation 与 differentiable planning；sim-to-real 仅展示 rope pushing | [论文](https://arxiv.org/abs/2210.13066) | 支持 S1；不应把其 T-shirt task 误写成严重褶皱连续展平—折叠。 |
| ThinShellLab（ICLR 2024） | 支持弯曲、塑性和摩擦接触的可微薄壳仿真与系统辨识 | 7 个任务含 lifting、separating、folding、forming，主要对象是纸张/薄片而非多样服装 | 状态常为降采样节点位置加 manipulator state，动作是 6/7-DoF manipulator control | 重点比较 trajectory optimization、RL 与实物参数辨识 | [论文](https://arxiv.org/abs/2404.00451) | 支持 S1 和仿真能力背景，不直接支持 garment benchmark 的覆盖声明。 |
| ClothesNet（ICCV 2023） | 服装资产、感知 benchmark 与模拟交互环境 | 总库约 4,400 个模型、11 类，其中 3,051 个经处理成为可仿真的 ClothesNetM；交互任务含 rearranging、folding、hanging、dressing | 基于 DiffClothAI；其 folding RL baseline 观察 20 个关键顶点的位置/速度，并输出 8 个控制顶点位移，不是机器人视觉连续控制 | 丰富的 feature、boundary line、keypoint 标注；folding success 由 20 个关键顶点是否进入目标阈值判断，机器人演示则用关键点检测加 scripted fold | [论文](https://arxiv.org/html/2308.09987) | 支持 S2/D1/D2；不能把 4,400 都写成 ready-to-sim 资产。 |
| GarmentLab（NeurIPS 2024） | 环境能力、视觉/RL/IL 方法和 sim-to-real | 11 类、20 个任务；unfold 从随机 drop 开始，fold 从平铺并轻微扰动开始，因此二者不是同一严重褶皱 episode 的连续阶段 | 多种机器人/末端执行器；视觉方法常预测操作点后走 scripted trajectory，也评测 RL/IL | 含 garment–garment、rigid、fluid、FEM object、avatar 交互；真机 T-shirt folding 为 15 次试验 | [论文](https://proceedings.nips.cc/paper_files/paper/2024/file/15f80ec0fed53885d2ca6272edb96ede-Paper-Conference.pdf)、[项目页](https://garmentlab.github.io/) | — |
| DexGarmentLab（NeurIPS 2025） | 灵巧服装任务环境、跨资产数据迁移与策略泛化 | 15 个任务、8 类服装，含 fling、fold、hang、wear、store；不是专门的端到端展平—折叠协议 | Isaac Sim 4.5；双臂灵巧手策略直接输出 60D 动作；HALO 用结构对应迁移示范中的 affordance 与轨迹 | 2,500+ 服装资产；依靠结构对应将单条专家示范迁移到多资产，仿真评测并选四项任务做真机展示 | [论文](https://arxiv.org/abs/2505.11032)、[项目页](https://wayrise.github.io/DexGarmentLab/) | — |
| **LeHome（ICRA 2026）** | 家庭场景中多类可变形物体的环境、数据采集、IL/VLA 训练和 sim+real 验证 | 论文将 Fling Garment 与 Fold Garment 列为两个独立 task；前者从 wrinkled clothes 展平，后者从桌面服装折叠，未证明二者在同一 episode 连续完成 | 支持 LeRobot/LeKiwi/XLeRobot 等低成本本体；键鼠、手柄及 leader–follower 遥操作；论文服装实验使用双臂连续控制，比较 ACT、DP、π₀、SmolVLA | 每 task 50 条遥操作示范、100 次仿真评测；真机用 10 条真实示范比较 real-only 与 sim+real co-training | [论文](https://arxiv.org/html/2604.22363)、[项目页](https://lehome-web.github.io/) | — |
| LeHome Challenge 2026（ICRA 竞赛） | 统一低成本双臂平台上的服装 folding 策略，而不是 LeHome 论文六任务实验的简单复述 | 四类服装随机加载且类别标签对参赛策略隐藏；官方仓库提供最多 600 steps 的完整 folding rollout | SO-ARM101 双臂、相机观测、连续关节动作；支持 ACT、DP、SmolVLA 及自定义策略 | 仿真 release set 与现场真机两阶段；官方提供四类示范和统一评测入口 | [ICRA 页面](https://2026.ieee-icra.org/program/competitions/)、[官方仓库](https://github.com/lehome-official/lehome-challenge) | — |
| Cloth Sim-to-Real Benchmark（ICRA 2024） | 在匹配的真实/仿真执行中比较 cloth simulator，而非训练任务策略 | 三块 50×70 cm 矩形布；规定 dynamic fling、quasi-static lowering/drag 等动作，抓取预先成功 | 两台 Franka 在真实与 MuJoCo/Bullet/Flex/SOFA 中复现相同动作 | 用 RGB-D point cloud 的 Chamfer/Hausdorff、稳定性和计算代价比较 simulator | [论文](https://arxiv.org/abs/2310.09543) | 直接支持 S3；对象和目标不同，不能用于声称已有大规模 garment strategy benchmark。 |
| RGBench（AAAI 2026） | 真实—仿真服装动力学差距和 simulator fidelity | 真实公开评测切片包含 9 件服装及 grasp、fold、fling 三类规定动作；不等同于学习策略完成长程任务 | 双臂真实/仿真执行相同动作；重点是动力学重现而非策略动作空间 | 总资产库超过 6,000 mesh；以实测真实服装动态比较 PyBullet、Isaac Sim 与 GarmentDynamics | [AAAI 论文](https://ojs.aaai.org/index.php/AAAI/article/download/38894/42856)、[项目页](https://rgbench.github.io/) | — |
| SimWeaver（2026 preprint） | 从仿真合成数据训练 RGB 策略并零样本真机执行 | 五项独立任务含 T-shirt flatten、T-shirt fold、silk grasp/flatten 和 bag；公开描述没有把 T-shirt flatten→fold 定义为同一 episode | Piper 双臂；SimWeaver-Syn 生成连续轨迹，π₀.₅ 从 RGB 学习并闭环执行 | 每 task 200 条合成示范；生成器使用拓扑图/landmark、闭式判据与规划器，并在仿真中闭环验证 | [论文](https://arxiv.org/html/2606.15338)、[项目页](https://simweaver.github.io/) | — |
| ICRA 2024 Cloth Competition / Dataset（IJRR 2025） | 共享真机上的单次悬挂布料抓取点选择 | 聚焦 in-air unfolding 的一次 grasp selection，不评测桌面长程展平或折叠 | 参赛者输出抓取位姿，执行流程由竞赛统一 | 11 支队伍现场对比；最终数据为 34 件服装、679 次真实 unfolding attempts | [论文](https://arxiv.org/abs/2508.16749)、[数据](https://zenodo.org/records/14621179) | — |
| Flat’n’Fold（ICRA 2025） | 真实长轨迹数据集，以及 grasp prediction / subtask decomposition 离线 benchmark | 数据覆盖完整 crumpled→flat→folded 过程；论文没有把闭环策略成功率作为主 benchmark | 同步多视角 RGB-D、点云及人手/机器人 gripper 位姿动作 | 1,212 条人类与 887 条机器人示范，44 件服装、8 类 | [论文与正式元数据](https://eprints.gla.ac.uk/349229/)、[项目页](https://cvas-ug.github.io/flat-n-fold) | — |

### 跨论文协议证据

| 工作 | 研究内容 | 能支持什么 | 不能替代什么 | 核验来源 |
| --- | --- | --- | --- | --- |
| Moravec's Paradox and the Robot Olympics（PI blog, 2025） | 论证日常物理技能包含难以通过语言和程序表达的隐性知识；通用模型需要足够丰富、多样的真实行为作为物理基础 | 支持 2.1 中“为什么要保留开放、连续的行为问题”和 2.3 中“为什么仅靠脚本不够”的概念动机 | 不是服装 benchmark 实验，也不直接证明 ClothDojo 的规模或效果；具体行为必须由本文轨迹与实验支撑 | [博客](https://www.pi.website/blog/olympics) |
| Unfolding the Literature（2024 review） | 系统整理 robotic cloth manipulation 的对象、状态表示、动作和评价方式，并指出实验协议长期缺少统一性 | 支持 G3/S5 的领域级背景，帮助检查是否漏掉方法族 | 不能替代对 FlingBot、LeHome 等具体方法动作接口的原文核验 | [论文](https://arxiv.org/abs/2407.01361) |
| Standardization of Cloth Objects and Tasks（2024） | 提议统一报告布料质量、形状、材料、刚度、弹性、摩擦及任务设置，讨论物性对结果的影响 | 支持“资产名字相同也未必构成可比实验”的判断 | 不直接证明 ClothDojo 的任务更难或物理更真实 | [论文](https://arxiv.org/abs/2403.04608) |

### LeHome 需要特别避免的混淆

- **LeHome 论文、LeHome Challenge 和参赛方案是三层证据。** 论文的六任务实验将 fling 与 fold 分开；Challenge 提供统一服装 folding 协议；参赛者技术报告可以说明某种解法，但不能反推为 LeHome 官方方法。
- LeHome 的 `trajectory replay` 明确固定物体位置来保持几何与接触，只随机 lighting、background texture 和 visual material，并用成功检测器过滤。因此它属于**已有轨迹的视觉扩展**，不能计作新的物理操作经验。
- LeHome 不应被概括成只说明“环境能模拟什么”：它确实训练并评测 ACT、DP、π₀ 和 SmolVLA，也做了 sim+real co-training。更准确的区别要落在它的任务分布、两个服装 task 是否连续、数据来源及诊断协议上。
- **LeHome 原论文中的 Fold 成功是怎样判断的：**公开 evaluator 读取固定的 6 个 mesh particle，将坐标乘 100 后检查三组欧氏距离是否分别不超过 10、16、10，并每隔固定检查周期判断一次。这是基于仿真 GT 顶点的 task-specific 几何判据，不是视觉分类器，也不是通用的“看起来折好”。Challenge 的四类别 evaluator 使用另一套按类别配置的边界条件，不能与原论文 Fold 判据混用。

## 4. 可变形物体数据获取与生成

| 工作 | 起始监督 | 如何扩展 | 是否得到新物理 rollout | 特权信息与最终策略 | 恢复数据 | 核验来源 | 作者判断 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UniFolding | 仿真 VR 人类 unfolding/folding 示范 | 自监督仿真数据；真机阶段用 HIL 继续聚合策略访问到的数据 | 是 | 训练/执行使用 RGB-D partial point cloud；仿真 mesh 信息参与数据构建 | 真机 HIL | [论文](https://proceedings.mlr.press/v229/xue23b/xue23b.pdf) | — |
| Flat’n’Fold | 人类及机器人直接采集 | 不自动扩展，主要增加真实采集规模与模态 | 是，但依赖继续人工/机器人采集 | 数据提供 RGB-D、点云和动作；不是“特权教师蒸馏到 RGB 策略”的方法 | 数据包含自然长轨迹，但未以 on-policy recovery 生成作为主方法 | [论文](https://eprints.gla.ac.uk/349229/) | — |
| MetaFold | ClothesNet mesh 与人工规定的 folding start/end keypoints | 在 DiffClothAI 中按 heuristic 生成分阶段折叠 mesh/point-cloud trajectories，并配语言 | 是，仿真执行会产生新的布料状态轨迹 | GT keypoint 用于生成；下游框架从点云与语言预测轨迹 | 有迭代 feedback，但不是基于策略失败的 DAgger 数据生成 | [论文](https://arxiv.org/abs/2503.08372)、[数据页](https://huggingface.co/datasets/chenhn02/MetaFold) | — |
| UniGarmentManip | CLOTH3D/PyFlex 中的 tops、trousers、dresses 与少量任务示范 | 用仿真 point tracing 学习跨实例稠密对应，再把示范中的 manipulation points 迁移到新服装 | 是，迁移后的动作在仿真/真机重新执行 | 对应监督来自 mesh；执行时由视觉对应网络定位目标点 | 不是 on-policy recovery；任务按 unfolding、folding、hanging 分别评测 | [论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_UniGarmentManip_A_Unified_Framework_for_Category-Level_Garment_Manipulation_via_Dense_CVPR_2024_paper.pdf) | 支持 D2；与 NOCS 都处理跨资产位置对应，但它迁移 manipulation points，不生成自由连续策略。 |
| FoldNet | 合成服装、自动 semantic keypoints 与 keypoint-based folding teacher | 教师从随机初态生成成功 folding；KG-DAgger 在模型 rollout 的失败状态接管并补充重试轨迹 | 是 | 生成器读取 GT keypoints；最终 Diffusion Policy 只看单目 RGB，不在部署时检测 keypoint | **是：自动生成 on-policy 抓取失败恢复** | [论文](https://arxiv.org/html/2505.09109) | — |
| χ₀ / Kai0 | 真实双臂人类示范 | 标准/heuristic DAgger、时空增强、不同数据子集训练后模型合并；当前公开数据含 base 与 dagger | DAgger 是；普通 augmentation 不是 | 最终 π₀.₅/π₀ 使用 RGB、本体状态和 action chunks；Stage Advantage 由阶段标签/进展训练 | **是：人工或 heuristic DAgger** | [论文](https://arxiv.org/html/2602.09021)、[数据说明](https://github.com/OpenDriveLab/kai0/blob/main/docs/dataset.md) | — |
| DeMaVLA | 约 5,000 h 真实双臂预训练和 37 h 多类别 folding 后训练数据 | 混合多类折叠示范；部署当前策略并由人只在预期失败时接管，聚合 correction 后继续训练 | 是，来自真机 rollout | 最终策略使用 RGB、语言与本体状态，flow matching 生成连续 action chunks | **是：human-gated DAgger** | [论文](https://arxiv.org/abs/2605.31286) | 与 ClothDojo 同样强调策略失败附近的数据；主要差异是其依赖大规模真实数据，ClothDojo 研究仿真代理生成与可控数据因素。 |
| LeHome | 每 task 仿真遥操作；真机实验另有少量真实示范 | 采集时随机初始位置和外观；采集后在固定物体位置下重放轨迹并随机外观，再按成功条件过滤 | 初次遥操作是；后处理 replay 主要不是新的行为 | 仿真状态用于 replay/filter；最终 ACT/DP/VLA 接收普通相机与机器人状态 | 论文没有以 on-policy failure recovery 为核心 | [论文](https://arxiv.org/html/2604.22363) | — |
| SoftMimicGen（ICRA 2026） | 每 task 1–3 条 Apple Vision Pro 遥操作示范 | 将示范切成 object-centric segments；根据完整 deformable nodes 做 non-rigid registration，选择并 warp 片段，在新初态执行 | **是**，变换后的动作在仿真重新执行并通过成功条件筛选 | 生成时读取完整 node positions；最终 BC-RNN/DP 从图像执行，不做 registration | 通过源片段选择覆盖新初态；不是 on-policy DAgger | [论文](https://arxiv.org/html/2603.25725)、[项目页](https://softmimicgen.github.io/) | — |
| SIM1（2026 preprint） | 少量示范与扫描得到的真实数字孪生 | 几何/物理对齐后，将示范分段，用 diffusion 生成运动并经过轨迹优化、仿真执行、质量过滤和外观随机化 | 是 | 仿真特权用于校准、规划和过滤；最终 π₀/π₀.₅ 从普通视觉执行 | 主要依靠生成与过滤；公开论述未把 on-policy recovery 作为核心 | [论文](https://arxiv.org/html/2604.08544) | — |
| SimWeaver | 手工构建的拓扑图、landmark、任务谓词和少量任务定义 | 规划器基于当前仿真状态合成连续轨迹，并以闭式成功/安全条件检查；每 task 生成 200 条 | 是 | 生成器访问拓扑与仿真几何；最终 π₀.₅ 使用 RGB | 生成过程闭环重规划，但不是从已训练策略的失败分布做 DAgger | [论文](https://arxiv.org/html/2606.15338) | — |

## 5. 与 ClothDojo 最接近、正文不能漏掉的四组关系

以下只是证据关系，不替作者填写最终结论：

1. **χ₀ / Kai0 与 DeMaVLA**：两者都已在真实机器人上研究长程 flatten–fold、连续 action chunks 和失败纠正数据；χ₀ 还系统讨论 advantage 与实时控制，DeMaVLA 则覆盖四类衣物的单 checkpoint 策略。ClothDojo 若讨论“通用策略为何需要新的 benchmark”，必须承认这些真实任务证据，再说明仿真可复现性、固定测试分布和规模化物理数据的增量。
2. **FoldNet**：与 NOCS 数据代理的技术关系最直接。两者都把仿真特权几何用于数据生成/恢复，并让最终视觉策略摆脱特权输入。真正的区别应从任务起点、教师表示、轨迹生成自由度、展平—折叠连续性和评测目标中找，不能写成“此前没有用特权信息生成普通视觉策略数据”。
3. **LeHome**：同时覆盖低成本双臂、连续控制、VLA benchmark、轨迹重放随机化和 sim+real，因此不能再笼统归入“只提供仿真能力”。可比较的是其论文中的服装 task 分离、每 task 数据协议，以及 ClothDojo 是否提供更严格的严重褶皱连续任务与诊断性测试分布。
4. **SimWeaver / SoftMimicGen / SIM1**：三者分别代表规则与拓扑驱动生成、非刚性轨迹迁移、物理对齐后的生成模型扩展。ClothDojo 的数据代理需要明确说明自己属于哪种机制，以及新增 rollout 与外观重渲染分别提供了什么。

## 6. 尚待作者核对或继续补证据

- ClothMate 全文与作者已有实验经验应补入表中，尤其是具体数据生成、投影失败条件与 primitive 参数控制边界。
- χ₀ 论文数据统计与后续公开 Kai0 数据集统计不是同一版本；正文引用数字时必须标明来源。
- FoldNet 的初态包含旋转、翻面和随机褶皱，但是否达到 ClothDojo 所称“严重褶皱”的同等分布，需要看资产/初态可视化或代码，不能只凭文字判定。
- LeHome Challenge 的详细初态生成和官方成功指标还应从 release evaluator 源码逐项核验；当前表只记录官方仓库明确暴露的协议。
- π₀ 系列服装任务的服装数量、初态与试验次数需按具体版本分别记录，不能把博客演示、论文任务和后续版本合并。
- DeMaVLA 的真机任务明确规定超时与落桌失败，但“task-specific completion criteria”没有在主文中给出可复现的几何阈值；可引用其连续任务证据，不宜与 ClothDojo 的最终形态指标直接做数值比较。
- `closed-loop` 一词不能直接填成“支持实时反馈”。需要进一步记录观测频率、action chunk 长度、每次实际执行长度、是否异步推理及低层控制频率。
