# 服装操作相关工作的公开仓库复核

复核日期：2026-09-06。本文档区分论文自述、公开实现和据此能够安全使用的论文表述。临时克隆目录不属于项目，关键结论与版本固定在本文件中。

## “连续行为”的工作定义

需要区分三种容易混淆的性质：

1. **时间连续**：控制轨迹包含稠密动作或经过插值，运动本身平滑。
2. **阶段反馈**：系统在预设动作之间重新观测，并调整抓点、目标点或重试当前阶段。
3. **行为连续**：系统根据布料的实时响应改变后续操作，包括动作次序、双臂配合、换抓和反复修正，而非始终推进预先写好的阶段图。

ClothDojo 应强调第三点。FoldNet、DexGarmentLab 和 SimWeaver 都可能同时满足前两点，因此不应使用“轨迹不连续”或“没有闭环”概括其局限。

## GarmentLab

- 官方仓库：https://github.com/GarmentLab/GarmentLab
- 复核提交：`6ee0620fddb8083d5eab554fa343cdc966c61d5f`
- `LearningBaseline/rl_fling/rl_utils/simulation_env.py:70-104`：策略动作只决定两个抓点；抓取后依次执行配置中的 `target_point`，随后释放并终止 episode。
- `LearningBaseline/unigarmentmanip/Inference/Fold.py:35-48`：模型给出 manipulation points，执行逻辑仍是写定的 grasp–move–release 序列。
- `demo/FlingDemo.py:24-29`：fling demo 的抓点和三个中间目标位置直接写在代码中。

**判断：** GarmentLab 确实提供 fold/unfold benchmark，但公开实现主要把学习限制在抓点或原子技能参数选择上，连续运动由固定控制流程完成。它的核心贡献是基础设施、资产和 benchmark，不是大规模、自适应的展平—折叠轨迹生成。

**安全表述：** “GarmentLab expands simulation assets and task interfaces, while its garment baselines select grasp points for predefined manipulation primitives.”

## FlingBot

- 官方仓库：https://github.com/columbia-ai-robotics/flingbot
- 复核提交：`c5f35c785c6bc5d49282fb10117aa66ca04bf144`
- `environment/simEnv.py:110-116` 将 `fling`、`stretchdrag`、`drag` 和 `place` 注册为独立 action handlers。
- `environment/simEnv.py:140-180` 会在抛甩前根据布料状态逐步拉伸，说明 primitive 内存在反馈，不能将系统概括为完全开环。
- `environment/simEnv.py:262-319` 的抛甩路径由固定 waypoint、速度和释放流程构成；学习动作主要提供抓点，控制器完成后续连续运动。

**判断：** FlingBot 是“学习抓点 + 预定义动态原语”的代表。它有反馈，但反馈用于完成既定 fling，而不是改变高层行为结构。

## UniFolding

- 官方仓库：https://github.com/xiaoxiaoxh/UniFolding
- 复核提交：`cbd0b374f1b7ab4d5ed3d66e7ad9e4736a42d81f`
- `common/datamodels.py:13-20` 将行为显式划分为 `FLING`、`FOLD_1`、`FOLD_2`、`PICK_AND_PLACE`、`DRAG` 等类型。
- `manipulation/statemachine_garment.py:250-270` 根据 `CRUMPLED → SMOOTHED → FOLDED_ONCE → FOLDED_TWICE` 状态依次选择 `FLING → FOLD_1 → FOLD_2 → DONE`。
- 同文件 `update_condition()` 中，fold 阶段是否完成直接由最近一次 action type 判断；代码注释还表明部分 drag 与 fling 成功条件未实现。
- `manipulation/experiment_real.py:220-280` 为不同 action type 检查和整理抓放位姿，再交给相应控制器执行。

**判断：** UniFolding 会逐步观测并学习原语类型和操作点，但公开状态机仍规定了整体展平—折叠顺序。它比单一 fling 更完整，却仍属于 primitive-level policy，而非自由连续的行为生成策略。

## DexGarmentLab

- 官方仓库：https://github.com/wayrise/DexGarmentLab
- 复核提交：`e4e298e696bae5d866ded3b31e0ae27becea5376`
- `Env_StandAlone/Fling_Tops_Env.py:219-324`：两阶段行为写在同一任务脚本中；GAM 按硬编码索引寻找抓点，随后执行固定抬升和前推动作。第一阶段只根据双手抓点距离改变横向间距；第二阶段只根据袖长改变拉伸距离，没有根据甩动结果调整速度、力度或动作结构。
- 同一文件在两阶段之间重新获取点云，因此它具有阶段反馈，但后续阶段和双臂配合方式不变。
- `Env_StandAlone/Fold_Tops_Env.py:210-335`：折叠固定为左袖、右袖、下摆三个阶段；每段独立 `record(..., stage_index=...)`，路径由关键点距离和固定 waypoint 计算。
- `README.md:198-270`：HALO 按 stage 单独准备数据和训练；README 明确列出 Fold Tops/Dress 为三阶段，Fling Tops/Dress 为两阶段，Fling Trousers 为单阶段。

**判断：** DexGarmentLab 会把抓点和部分几何参数迁移到新资产，因此不能说它完全没有扩展示范；但公开实现扩展的是任务专用原子技能及其参数，而不是完整的人类连续行为。论文所谓 single expert demonstration 不能掩盖任务序列、抓点索引和执行逻辑仍由人工编码这一事实。

**安全表述：** “DexGarmentLab retargets grasp points and geometric parameters across garments, but executes and learns separately defined task stages whose order and motion pattern are fixed.”

## LeHome

- 官方仓库：https://github.com/lehome-official/lehome
- 复核提交：`7613c74b4b85142d4d605f395b3156248feb7fbf`
- `source/lehome/lehome/tasks/bedroom/garment_bi.py:43-55` 与 `garment_fling_bi.py:42-54`：fold 和 fling 环境都硬编码为同一件 `TCLC_002` 上衣。
- `particle_garment_cfg.yaml:11-16`：fold 的位置和姿态范围退化为单点，即固定平铺初态。
- `particle_garment_fling_cfg.yaml:14-23`：fling 的位置固定，旋转仅在约 70–73 度之间变化。
- 仓库中的 `Assets/` 与 `Datasets/` 只有占位文件，需要另行下载；三个训练配置均默认指向 `Datasets/example/fold_garment/001`。
- 论文报告每任务 50 条遥操作示范和 100 次仿真评测；这证明数据可以训练策略，但不能支撑跨服装或复杂初态的数据规模结论。

**判断：** LeHome 确实实现了 fold 和 fling task，不能写成“没有展平”；但公开实验是单资产、窄初态、小样本的任务验证，服装操作只是其通用家庭可变形物体平台中的两个示例。

**安全表述：** “LeHome validates policy learning on one folding and one fling task with 50 demonstrations each; its released task definitions use a single garment and narrowly controlled initial states.”

## FoldNet

- 官方仓库：https://github.com/chen01yx/FoldNet_code
- 复核提交：`e98df529eac8ac9fdab1db365dcc9fc75b666200`
- `src/garmentds/foldenv/policy/state/tshirt.py:388-417`：高层流程由显式 stage counter 驱动，依次执行 rotate、align sleeves 和 fold。
- 每个阶段读取当前 mesh/keypoints 计算抓点和目标点，因此不是纯开环回放。
- `src/garmentds/foldenv/policy/state/base.py:134-153`：KG-DAgger 检测抓取失败后清空队列、将 stage counter 回退一位并重试；它没有根据折叠结果重组后续阶段。

**判断：** FoldNet 的轨迹在时间上连续，数据也能训练端到端 action-chunk policy；其局限是阶段性响应：任务结构固定，恢复主要针对抓取失败，无法表达随布料响应改变动作次序和双臂协作的行为。

**安全表述：** “FoldNet conditions keypoint-based motion on the current garment state and retries failed grasps, but advances through a fixed stage graph.”

## SimWeaver

- 官方仓库：https://github.com/simWeaver/SimWeaver
- 2026-09-06 时仓库只有一个 README，明确标注代码、资产和 checkpoint 尚未发布。因此无法通过公开实现复核其关键点可见性、遮挡处理、初始化分布和轨迹生成结果。
- 论文声称使用 topology graph、可见性与层叠判据筛选抓点，并在失败后重新观测和选择；不能直接写成“关键点遮挡导致方法失效”。
- 论文中的 T-shirt flatten 初态由 two-point hang-and-drop 生成，是受控且保持服装结构可辨认的状态族。更稳妥的质疑是：该设置没有证明规则系统能处理任意严重褶皱和自遮挡。
- 其高层行为仍由 task-specific graph、selector 和 post-grasp trajectory 定义；即使低层路径平滑且抓取可重试，也不等价于根据布料响应重组完整行为。

**判断：** SimWeaver 是规则方法中最强的直接相关工作，不能通过“无闭环”“动作不连续”或“没有展平”排除。ClothDojo 应与其比较行为结构、初态覆盖、跨资产轨迹分布和公开可复现性。

**安全表述：** “SimWeaver reselects topology-defined grasp pairs after failed attempts, while task semantics and post-grasp behaviors remain externally specified; its released repository does not yet expose code or assets for independent verification.”

## SoftMimicGen

- 官方仓库：https://github.com/NVlabs/SoftMimicGen
- 复核提交：`0cc4641b990b75d5e9fc920b332c7f4673cbdd68`
- `softmimicgen/datagen/data_generator.py:420-460` 读取并扰动预先标注的 subtask boundaries，同时强制各 subtask 保持顺序。
- 同文件 `generate_trajectory()` 从当前物体位姿或完整 soft-object node positions 出发，为每个 subtask 选择源示范片段；随后保留片段中的末端轨迹与 gripper actions，并根据当前物体状态进行变换后执行。
- 公开任务同时包含 towel fold 和 humanoid towel unfold，因此不能笼统声称该方法不能处理任何展平任务。

**判断：** SoftMimicGen 确实生成新的物理 rollout，并能利用完整节点状态适配新初态；其边界是 subtask 划分、顺序和局部行为来自源示范库。任务越难用稳定 subtask 表达，迁移与组合的成本越高。

**安全表述：** “SoftMimicGen generates new physical rollouts by selecting and transforming source demonstration segments under privileged deformable state, while preserving an annotated subtask structure.”

## Cloth Funnels 与 SpeedFolding

- Cloth Funnels 论文与公开实现均把策略输出限制为 `fling`、`pick-and-place` 等 primitive 及其空间参数；episode 最多运行若干次高层决策，折叠阶段则在规范化状态上执行基于关键点的固定步骤。
- SpeedFolding 的公开仓库明确把连续运动放在 `manipulation/experiment.py`，把非学习的折叠逻辑放在 `manipulation/heuristics/`。完整流程会在高层步骤重新推理；张紧阶段用力阈值和碰撞检查控制接触，但选定 fling 后，waypoint 与速度仍由 primitive 给定。

**判断：** “有闭环观测”与“连续运动由策略决定”是两件事。两项工作都会在 primitive 边界重新决策，也有局部执行反馈；反馈主要保证选定动作完成，没有在动作过程中重新选择行为类型、双臂分工或任务结构。

**安全表述：** “Cloth Funnels and SpeedFolding replan at primitive boundaries; local grasp or stretch feedback helps execute the selected primitive, while its motion pattern remains predefined.”

## SIM1

- 官方仓库：https://github.com/InternRobotics/SIM1
- 公开流水线为 generate → smooth → replay → filter；生成器从遥操作参考轨迹出发，分离接触段与移动段，再生成或平滑段间运动。
- `components/datagen/selector.py` 的当前 `SelectorFine` 按 `subtask_0, subtask_1, ...` 的位置依次选择源片段，没有在布料交互过程中根据观测改变子任务顺序。
- `components/datagen/splitter.py` 的当前细粒度路径要求 three-fold 轨迹具有固定数量的稳定段与移动段；这表明公开实现面向结构明确的折叠流程。
- README 的 10,000 条公开轨迹仍列在 TODO；仓库已提供生成、回放和过滤代码以及一个参考轨迹子集，不能把论文数据规模等同于当前公开数据规模。

**判断：** SIM1 扩展的是真实物理 rollout，不是简单图像增强；它保持接触行为并生成过渡运动，但当前公开实现仍沿固定分段位置重组折叠轨迹。它是结构化折叠的数据扩展方法，不是严重褶皱展平的直接先例。

**安全表述：** “SIM1 recombines interaction segments from teleoperated folding demonstrations and synthesizes transitions, while retaining an ordered subtask decomposition.”

## Instant-Fold

- 论文与项目页公开，未提供代码仓库；因此只能核对论文的方法与数据定义，不能独立复核生成器实现。
- 数据由 360 件上衣、19 种 folding modes 组成；每条轨迹的 context 由 sleeve primitive、body-fold primitive 和 execution order 定义，共约 80,000 条。
- 策略本身接收示范上下文与当前观测，直接输出双臂连续 action chunks，不能归为 primitive 执行器。
- 数据生成初态围绕平铺服装、cloth drop、袖部扰动和 wrinkle-like noise；论文的限制也将研究范围限定在 folding manifold，没有覆盖严重褶皱展平。

**判断：** Instant-Fold 证明了人工枚举的多模式折叠数据可以训练连续闭环策略。它应放在“大规模折叠数据仍由预设行为模式产生”的证据位置，而不是与 FoldNet、SimWeaver 一起写成运行时规则规划器。

**安全表述：** “Instant-Fold learns a reactive continuous policy from a large simulated corpus, while that corpus is generated from 19 predefined folding modes on structured garment states.”

## π*0.6、χ0 与 DeMaVLA

- π*0.6/RECAP 论文使用示范、自主 episode、结果标签和人工介入共同改进 VLA；洗衣任务从篮中取出衣物，包含展平、折叠和堆放。论文报告系统能在新家庭中连续运行两小时以上。
- `Physical-Intelligence/openpi` 公开 π0、π0.5 的训练与推理代码及 action-chunk 接口，但当前仓库没有完整公开 π*0.6/RECAP 的真实机器人数据闭环。因此 RECAP 的采集细节以论文为准，不能声称已由代码完全复现。
- χ0/Kai0 的论文与仓库都公开 FlattenFold 任务及 policy-in-the-loop 数据采集入口。公开数据说明列出数千条 flatten-fold episode、数十小时采集；不同版本中的小时数口径并不完全一致，正文不宜使用单个精确数字比较成本。
- DeMaVLA 论文与 README 声称采用 5,000 小时筛选后的双臂数据及 human-in-the-loop DAgger。当前仓库公开模型训练代码和示例配置，没有公开真实采集器与 DAgger 实现。

**判断：** 真机工作已证明连续 VLA 与 rollout/人工纠正能够处理长程衣物任务。它们支持 ClothDojo 的路线动机，也同时说明大规模真实行为依赖大量机器时间、复位与人工参与。相关工作不应宣称 ClothDojo 首创 SFT、HIL 或策略 rollout。

## NOCS、GarmentNets 与 UniGarmentManip

- NOCS 将类别内实例映射到统一单位立方体，并把规范坐标显示为 RGB 颜色；原工作面向刚体类别级姿态估计。
- GarmentNets 将规范空间用于不同形变服装的类别级形态估计；公开代码包含规范化网络、形态补全和对应数据。
- UniGarmentManip 学习跨服装形变的稠密对应，再由人工给定功能点和任务流程生成折叠、展平或悬挂操作；公开实现支持该感知与操作点迁移路线。

**判断：** 规范坐标和跨实例对应都不是 ClothDojo 的独立新颖点。差异在使用方式：NOCS 作为数据代理的图像条件，向不改结构的 VLA 提供规范位置参照；它不直接规定抓点、primitive 或任务阶段。

## 对 Related Work 的直接影响

1. GarmentLab、DexGarmentLab 和 LeHome 应首先被定位为基础设施或 task-specific skill 实现，而不是与 ClothDojo 等价的连续轨迹生成平台。
2. FoldNet、DexGarmentLab 和 SimWeaver 的共同局限不应写成“开环”或“动作不连续”，而应写成“根据当前状态调整预设阶段，但不能自由重组后续行为”。
3. Flat’n’Fold 是真实数据集，与仿真平台分开讨论；它的价值在于提供完整人类行为，对应的局限是逐条采集与本体绑定。
4. LeHome 的不足可以从公开实验的单资产、窄初态和 50 条示范陈述，不能用“没有做展平”概括。
5. SimWeaver 对遮挡作出了方法性回应。ClothDojo 更可靠的差异是学习式行为生成和更广的真实初态分布，而不是假设其关键点系统必然在遮挡下失败。
6. primitive 方法应写成“决策主要发生在 primitive 边界，内部反馈只完成当前动作”，不再笼统写成“每步重新观测”或“完全开环”。
7. 大规模折叠数据与大规模褶皱展平数据必须分开。FoldNet、SIM1 和 Instant-Fold 的规模不能用来暗示它们覆盖任意褶皱展平。
8. SimWeaver 是当前最直接的程序化展平近邻：它处理遮挡并重试抓取，但 T-shirt 初态来自受控的双点悬挂释放，抓取成功后执行预设 `tau_post`。
