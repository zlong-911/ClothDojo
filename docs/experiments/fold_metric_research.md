# 折叠任务评估指标调研与专家轨迹基线

## 结论

正式 `shirt_fold` success 不应以单个 folded canonical target 为中心。折叠允许多种合法
路径和终态，逐粒子目标距离会把“与模板不同但同样正确”的折法判错；只看 footprint、
矩形度或 silhouette 又会把紧凑的揉团误判为成功。

当前最合适的主方案是 **template-free semantic relation evaluator**：从每件衣服的展平
网格自动取得左右袖、领口、左右下摆和衣身等语义区域，再用尺寸归一化的区域距离、
接触/覆盖和分离关系定义一个合法终态集合。规则只描述终态语义，不规定折线、动作顺序
或唯一层叠，因此可以同时接受多种折法，也不需要人工采集任何 folded template。

这不是完全“零语义先验”：仍需一次性确定 `Fold the shirt` 到底要求哪些部位收进衣身、
紧凑到什么程度，以及哪些退化状态必须排除。但这部分规则和阈值是 task-level 的，不是
per-asset 的。ClothDojo 已保存固定粒子对应和由展平网格生成的 NOCS，因而可以把自动
提取出的语义区域稳定映射到整个 rollout；当前 converted USD 只保留几何与 face-varying
UV，没有现成的袖口/领口标签，区域提取器仍需实现和验证。

现有对齐后对应粒子 3D 距离仍有价值，但只应作为有模板子集上的诊断项和专家一致性
分析，不应作为开放折法的正式 success。

## 文献中的常用指标

- Hoque 等在真实 T-shirt 折叠中使用与目标模板做旋转、平移配准后的 silhouette IoU，
  并用衣服内部 Canny edge 像素比例作为 wrinkle penalty。优点是无需完整 3D 状态，
  缺点是依赖相机、分割和图像配准，而且相同轮廓可能对应不同层叠。
- UniFolding 的展开阶段报告 IoU 与 normalized coverage，端到端折叠仍以人工定义的
  folding rules 和 success rate 为主，没有提供连续的折叠质量分。
- GarmentLab 对折叠采用目标与结果的 IoU，并要求成功状态保持至少 5 秒。这一稳定性
  要求适合 ClothDojo，但现有专家数据没有统一的“释放后静置 5 秒”尾段，当前不能
  公平复算。
- AdaFold 计算折叠上、下层投影面积的 IoU，直接检查 layer overlap；这比整个外轮廓
  IoU 更接近折叠本身，但需要事先知道折线和上下层语义。
- 有粒子真值的近期仿真工作直接计算结果与目标状态的 mean particle distance。这与
  ClothDojo 已有 flattening metric 的对应顶点思想一致，也是当前数据条件下信息损失
  最小的目标匹配方案，但不能容纳开放的多种合法折法。
- LeHome 的官方 evaluator 没有使用完整终态模板，而是检查少量服装语义点对的距离：
  长/短袖上衣同时包含“应靠近”和“应保持分离”的约束，并按衣服初始尺度调整阈值。这直接
  证明关系式 success 可以接受一组终态，同时用分离约束避免简单揉团。
- AdaFold 的点云表示额外加入 Upper/Bottom 语义描述，用于区分几何外观相似但折叠状态
  不同的点云；这支持“几何紧凑度不够，必须保留部位/层语义”的判断。
- DefFusionNet 将合法目标学习为条件多峰分布，避免把多个合法终态平均成一个不存在的
  目标。这适合未来用已有专家终态学习 soft score，但依赖示范覆盖、hard negatives 和
  校准，不宜直接作为第一版 benchmark 的黑盒 success。
- 基于 distinctive points 的拓扑衣物表示和 CloSE 等 shape-agnostic 表示说明，粗粒度
  拓扑/部位关系可以跨形状描述衣物状态；后者较新，现阶段更适合作为后续研究方向。

参考入口：

- [Learning to Fold Real Garments with One Arm](https://autolab.berkeley.edu/assets/publications/media/IROS22_2254_FI%20%281%29.pdf)
- [UniFolding](https://proceedings.mlr.press/v229/xue23b.html)
- [GarmentLab](https://openreview.net/pdf?id=bIRcf8i1kp)
- [AdaFold](https://arxiv.org/abs/2403.06210)
- [Learning Efficient Robotic Garment Manipulation with Standardization](https://arxiv.org/abs/2506.22769)
- [LeHome official success checker](https://github.com/lehome-official/lehome-challenge/blob/main/source/lehome/lehome/utils/success_checker_chanllege.py)
- [DefFusionNet / DiffDef](https://arxiv.org/abs/2506.18779)
- [A topological representation of cloth state](https://link.springer.com/article/10.1007/s10514-021-09968-7)
- [CloSE](https://close-representation.github.io/)

## 推荐的正式指标结构

### 1. 二值 success：语义关系全部成立

以展平网格面积平方根 `s` 归一化所有距离。语义区域优先使用一小片 boundary region，
区域间距离取低分位数或对称 Chamfer，而不是依赖单个顶点，以降低网格离散和局部褶皱
的敏感性。第一版关系应保持少而可解释：

- 左右袖区均进入或贴近衣身核心区；不规定先折哪一侧，也不要求落到某个固定顶点。
- 下摆区与上衣身充分接近/覆盖，或整体达到等价的纵向收纳关系。
- 终态 footprint 足够紧凑，主要布料位于桌面上，机器人已经释放。
- 至少若干语义区域仍保持必要分离，且局部应变受限，用于拒绝揉团、卷成细条和异常拉伸。
- 上述条件在释放后的稳定窗口内持续满足；参考 GarmentLab，可先采用 5 秒而不是终态单帧。

若数据表明确实存在语义不同、无法由同一组宽松关系覆盖的合法类别，可以加入少量
task-level rule branches，并对分支取 OR。分支表达“合格终态类型”，不绑定动作轨迹，也
不随测试 rollout 更新。

### 2. 连续分数：同一组约束的 margin

每条关系输出相对阈值的归一化 margin，主分数取 soft minimum 或低分位聚合，使最差的
必要条件决定整体质量。这样连续分数与二值 success 语义一致，避免另造一套任意加权的
IoU/厚度/矩形度总分。各关系值仍应单独落盘，便于解释失败原因。

### 3. 诊断项：不参与主 success

继续报告 `convex_footprint_ratio`、`convex_rectangularity`、厚度、速度，以及有模板子集
上的 `aligned_fold_distance`。这些指标适合排序、debug 和回归分析，但任何一项都不能
单独代表“折好了”。

### 4. 人工成本

- 不采集逐资产 folded template。
- 一次性审查自动提取的语义区域；若几何启发式在少数异形资产失败，只修区域映射，不
  录制折叠终态。
- 用现有人工复核专家终态，加上自动生成的未折、半折、揉团、卷条、悬空和高应变 hard
  negatives，按 asset 分组校准阈值和 false-positive rate。
- 第二阶段可以在上述透明 feature space 上拟合 mixture/one-class density，吸收专家的
  多峰分布；模型分数先作辅助项，验证跨资产泛化后再考虑升级为正式指标。

## 本轮指标

主误差 `loo_consensus_fold_error` 的计算为：

1. 按 `asset_id` 分组，取每条轨迹的 terminal `particle_q`。
2. 对目标状态做桌面内刚体 Procrustes 配准：连续旋转和平移可变，z 坐标保留。
3. 对所有对应顶点的 3D 欧氏距离取平均。
4. 用该资产初始展平状态 footprint 面积平方根归一化，消除衣服尺寸影响。
5. 对每条专家轨迹，从其余专家中选择 medoid 目标，避免自比较得到零误差。

同时报告以下诊断项，但不把它们用未经验证的权重压成一个总分：

| 字段 | 含义 | 方向与边界 |
| --- | --- | --- |
| `canonical_fold_error` | 到全体专家 medoid 的归一化对应粒子距离 | 越低越好；medoid 自身为 0，不能用于评价训练内专家 |
| `loo_consensus_fold_error` | 到 leave-one-out 专家 medoid 的距离 | 越低越一致；单例资产为 `null` |
| `expert_consistency_percentile` | 在全部可比较专家中的误差反向百分位 | 越高越一致；不是成功概率 |
| `convex_footprint_ratio` | 终态/初态的 XY 凸包面积比 | 只表示紧凑度；过大可能没折完，过小也可能是揉成团 |
| `convex_rectangularity` | 终态凸包面积/最小外接矩形面积 | 越高越像矩形；不能识别内部层叠 |
| `footprint_aspect_ratio` | 最小外接矩形短边/长边 | 描述形状，不预设单一最佳值 |
| `normalized_thickness_p90` | `(z95-z05)/sqrt(initial area)` | 越低通常越平整，但正常多层厚度不应被罚为零 |
| `speed_rms_m_s` | 终态粒子 RMS 速度 | 越低越稳定；单帧不能替代持续稳定条件 |

外轮廓目前使用凸包而非三角面片 union，所以相关字段明确以 `convex_` 命名。它们只做
快速诊断，不冒充文献中的精确 silhouette IoU。

## 数据审计与评分结果

输入为 `data/collections/shirt_fold/trajectories`：

- 539 条完整 canonical 轨迹；
- 156 个资产；
- 210 条有 `review_status=complete` 且无 bad segment 的人工复核轨迹；
- 490 条来自至少有两条专家轨迹的资产，可计算 leave-one-out 共识误差；
- 49 条属于单例资产，只能计算 target-free 诊断。

全量专家结果：

| 指标 | P05 | P25 | Median | P75 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `loo_consensus_fold_error` ↓ | 0.0530 | 0.0681 | 0.0822 | 0.1017 | 0.1620 |
| `convex_footprint_ratio` | 0.1571 | 0.1745 | 0.1908 | 0.2182 | 0.2859 |
| `convex_rectangularity` ↑ | 0.7863 | 0.8329 | 0.8652 | 0.8881 | 0.9191 |
| `normalized_thickness_p90` ↓ | 0.1083 | 0.1230 | 0.1364 | 0.1503 | 0.1741 |
| `speed_rms_m_s` ↓ | 0.00478 | 0.00679 | 0.00933 | 0.01229 | 0.01952 |

人工复核子集的共识误差中位数为 0.0767，未复核子集为 0.0866。两个子集的资产和采集
时间组成不同，因此该差异只能作为数据检查线索，不能解释为人工复核提高了质量。

共识误差最大的两条轨迹是：

- `20260813T021837.728146Z`：0.6868，终态仍明显悬空，归一化厚度 0.7159；
- `20260812T093338.629104Z`：0.3766，footprint ratio 0.8107，明显没有达到常见折叠终态。

这说明对应粒子距离能够发现 silhouette 形状指标容易漏掉的严重失败。另一方面，少量
只有两条示范的资产会因合法折法或层叠顺序不同而同时获得较大误差。这说明正式目标
不能直接由稀疏专家数据自动拍脑袋确定，也进一步支持正式 success 改用 task-level
语义关系；模板距离只保留为分析工具。

结果文件位于：

- `outputs/evaluations/fold_expert_metrics_v1/results.jsonl`
- `outputs/evaluations/fold_expert_metrics_v1/summary.json`
- `outputs/evaluations/fold_expert_metrics_v1/per_asset.json`

复现命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/data/score_fold_experts.py
```

## 正式 benchmark 前还缺什么

1. 在展平网格上实现并可视化自动语义区域提取，抽样覆盖不同袖长、领型和版型，确认
   左右袖、领口、下摆、衣身的识别可靠性。
2. 用少量可解释关系实现离线 prototype，并在现有专家终态、不同折法和自动 hard
   negatives 上画出各项 margin；先观察可分性，再冻结阈值。
3. 对边界样本做人审，按资产分组校准 threshold 和 false-positive rate。专家 P95 只能
   定义“专家包络”，不能证明揉团等失败不会通过。
4. 统一在机器人释放后继续仿真 5 秒，success 必须在窗口内持续满足语义关系、速度和
   桌面接触条件；终态单帧只用于离线连续分数。
5. 用按资产分组的交叉验证检查阈值和诊断项，避免轨迹数最多的
   `rand_ZZCWITH5HY` 支配全局校准。
