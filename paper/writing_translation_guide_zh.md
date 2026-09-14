# ICRA 写作与中英翻译备忘

> 2026-09-09。面向 ClothDojo 的中文内容稿到英文 ICRA 稿转换。官方规则与编辑建议分开记录；本文件不替代当届投稿页面。

## 1. ICRA 2027 的硬约束

- 投稿 PDF 的全部内容共 8 页，包括正文、图表、致谢和参考文献；超页稿件不送审。
- 使用 ICRA/IEEE 双栏格式并执行双匿名评审。正文、图、补充视频和文件元数据都不能泄露作者身份。
- 不接受独立书面补充材料。可以提交最长 180 秒的视频，但审稿人也不需要访问正文中的外部链接。因此，支撑主要主张的任务定义、方法、对照和数字必须在八页 PDF 内自洽。
- 摘要应为单段、可独立理解；IEEE 的通用会议建议是不放引用、脚注和公式，术语应无歧义。ICRA 当届页面给出篇幅和匿名规则，IEEE Author Center 给出写作结构建议。

来源：[ICRA 2027 Call for Technical Papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)、[PaperCept manuscript preparation](https://ras.papercept.net/conferences/support/support.php)、[IEEE conference paper structure](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/)。

## 2. 适合本稿的会议写法

ICRA 没有强制规定“必须有几节”或统一叙事模板。下面是由八页限制、IEEE 的结构建议和机器人实验论文的可复现性要求推导出的编辑原则。

1. **按主张组织，不按系统组件组织。** 每个小节回答一个审稿问题：任务是什么、方法为什么这样设计、实验排除了什么其他解释。Newton、Style3D、RECAP 和 \(\pi_{0.5}\) 只解释它们在 ClothDojo 中承担的作用，不写成依赖库说明书。
2. **段首给判断，段内给机制或证据。** 一段优先采用“结论/设计 → 原因 → 必要边界”的顺序。不要以 “In this section, we introduce ...” 占据段首。
3. **引言形成收窄链条。** 具体任务困难 → 现有数据来源的限制 → 核心 insight → 方法概览 → 可由实验验证的贡献。贡献项必须能映射到正文结果，尚未完成的数据规模不能写成既成贡献。
4. **Method 解释可复现的因果关系。** 先定义输入、输出和任务边界，再说明 NOCS 为什么适合复用 VLA 图像接口，最后说明示范、自主经验、纠正和重渲染如何改变数据分布。实现参数集中放在实验设置，不在主线反复出现。
5. **实验标题直接写问题。** 例如 “Does NOCS improve trajectory generation?”。先交代被控制的因素，再报告结果和适用边界。成功率之外保留连续指标、配对差值和波动，避免只凭精选轨迹解释机制。
6. **图负责关系，表负责精确值。** Fig. 3 表达数据流，正文解释分支为何存在；主表给出比较数字。图注必须独立可读，但不复述整段 Method。
7. **机器人论文尤其要区分层级。** 记录频率不等于策略推理频率，仿真步频不等于数据吞吐；重渲染增加视觉观测但不增加物理轨迹；离线评分不等于闭环执行；仿真收益也不自动推出 sim-to-real。

IEEE 还建议方法描述应足以复现，结果部分应同时解释结果并承认限制，引用只保留直接支撑正文的工作。参见 [IEEE research reproducibility](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/research-reproducibility/) 和 [IEEE conference paper structure](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/)。

## 3. 中译英不是逐句替换

### 3.1 先冻结“意思”，再写英文

每段先标出三件事：本段唯一判断、支撑它的事实、不能越过的边界。英文按这三者重新组织句子，不保留中文原句的停顿和修饰顺序。一个中文长句常应拆成两句；连续几个短句若共享主语，也可以合并。

避免把中文省略的逻辑关系留给英文读者猜。例如：

- 中文：“固定轨迹后改变外观，可以分离两种覆盖。”
- 不完整直译：“Changing appearance after fixing trajectories separates two types of coverage.”
- 更准确：“Because re-rendering preserves states and actions, it changes visual coverage without adding physical interaction. We can therefore evaluate the two sources of data variation separately.”

### 3.2 保持主张强度

| 中文意图 | 推荐英文 | 避免 |
| --- | --- | --- |
| 实验直接显示差异 | `The paired comparison shows ...` | `proves` |
| 结果与解释一致，但未排除其他机制 | `supports` / `is consistent with` | `demonstrates that the only reason is ...` |
| 根据证据推断 | `suggests` / `indicates` | 无限定的 `shows` |
| 当前设置内有效 | `improves success under this protocol` | 空泛的 `is effective` |
| 尚未验证 | `we do not claim ...` / `remains to be evaluated` | 把计划写成现在时结果 |

`significant` 只在统计意义明确时使用；普通幅度差异写 `substantial`、`large` 或直接给百分点。`robust`、`general`、`realistic`、`scalable` 都需要紧邻相应协议或证据。

### 3.3 英文句子的技术责任要明确

- 优先使用明确主语和动作动词：`The policy predicts...`、`The simulator restores...`、`We evaluate...`。
- 可以使用 `we`；不要为了显得正式而大量改成被动语态。IEEE 建议清晰、简单、避免不必要的复杂措辞。
- 方法事实通常用一般现在时；已完成的实验过程用过去时；图表所呈现的结果可用现在时。不要在同一层级无理由切换时态。
- 注意冠词和可数性：第一次引入写 `a trajectory`, 指回特定对象写 `the trajectory`；`data` 在 IEEE 风格中通常按复数处理。
- 变量首次出现时定义，正文符号与公式完全一致。单位使用罗马体和标准缩写，如 `3 s`、`50 Hz`；范围、百分比和小数精度统一。
- 缩写首次出现时展开，此后固定一种形式。标题和摘要尽量少放缩写；摘要必须能脱离正文独立理解。

来源：[IEEE technical English](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/write-in-technical-english/)、[IEEE Editorial Style Manual for Authors](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE-Editorial-Style-Manual-for-Authors.pdf)。

## 4. ClothDojo 术语表

| 中文概念 | 建议固定英文 | 说明 |
| --- | --- | --- |
| 学习式数据代理 | `learned data-generating policy` | 首次出现可说明其作为 `trajectory agent` 运行；不要在 `data proxy`、`data agent`、`generator` 间无规则切换 |
| 轨迹生成策略 | `trajectory-generating policy` 或简写 `trajectory agent` | 选择一套作为主称谓 |
| 人工纠正 | `human correction` | 当前实现是恢复保存状态后继续 VR 操作，不写成实时 `takeover` |
| 人工介入状态 | `intervention state` | 指状态或时间点；不要用它代替完整 correction trajectory |
| 自主经验 | `autonomous rollout data` | 比泛化的 `experience` 更明确 |
| 规范表面位置参照 | `canonical surface-position cues` | NOCS 不是严格服装部位语义 |
| 物理轨迹 / 视觉副本 | `physical trajectory` / `visual rendering` | 不把重渲染版本计作新 trajectory |
| 曾成功率 / 终态成功率 | `success rate` / `terminal success rate` | 首次定义前者要求曾连续达标 3 s |
| 形变距离 / 朝向距离 | `deformable distance` / `orientation distance` | 若沿用旧英文 `rigid distance`，全文和图表必须统一 |
| 折叠质量评分 | `fold-quality score` \(Q\) | Q 是实数分数，不是概率或 reward value |

## 5. 建议的翻译流程

1. 先冻结中文每节的主张、术语和数字，再开始英文同步。
2. 以“段落”为单位重写英文，不逐句对齐；每段检查主语、因果连接和限定词。
3. 做一次术语审计：标题、摘要、正文、图注、表头和公式使用同一称谓。
4. 做一次证据审计：每个 `show/demonstrate/outperform/robust/generalize` 都能指向具体实验；每个数字与当前配置和数据版本一致。
5. 做一次匿名与版面审计：移除身份线索，确认八页内正文自洽，外链和视频只作补充。
6. 最后再润色句法和节奏。语言润色不能改变比较条件、成功定义或主张强度。

对当前仓库的直接提醒：英文稿仍保留旧的 `Benchmark and Environment` 与 `NOCS-Guided Data Engine` 分章结构，并包含早期折叠成功口径；应在中文 Method 与实验口径冻结后整体同步，不能只翻译新增段落。
