# 折叠任务评估指标调研与专家轨迹基线

## 结论

ClothDojo 的仿真轨迹保存了有固定顶点对应的逐帧布料状态，因此折叠任务应以
**对齐后的对应粒子 3D 平均距离**为主指标，而不是只用 top-down silhouette IoU。
评分时去除桌面内的连续旋转和平移，但不允许反射、顶点重排或 z 方向平移；这样不会
惩罚衣服在桌面上的摆放位置，同时仍能区分错误的层叠顺序、局部翻面、悬空和褶皱。

当前还没有独立定义的 `shirt_fold` canonical target，也没有失败 rollout 用来校准成功
阈值。本轮因此采用同资产专家终态的 leave-one-out medoid 作为临时目标，得到的是
**专家间一致性分**，不是正式成功率。对只有一条专家轨迹的资产，目标距离不可识别，
只报告不依赖目标的几何诊断。

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
  最小的方案。

参考入口：

- [Learning to Fold Real Garments with One Arm](https://autolab.berkeley.edu/assets/publications/media/IROS22_2254_FI%20%281%29.pdf)
- [UniFolding](https://proceedings.mlr.press/v229/xue23b.html)
- [GarmentLab](https://openreview.net/pdf?id=bIRcf8i1kp)
- [AdaFold](https://arxiv.org/abs/2403.06210)
- [Learning Efficient Robotic Garment Manipulation with Standardization](https://arxiv.org/abs/2506.22769)

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
不能直接由稀疏专家数据自动拍脑袋确定：应人工确认每个任务允许的 fold convention，
必要时冻结多个等价 target，并对这些 target 取最小距离。

结果文件位于：

- `outputs/evaluations/fold_expert_metrics_v1/results.jsonl`
- `outputs/evaluations/fold_expert_metrics_v1/summary.json`
- `outputs/evaluations/fold_expert_metrics_v1/per_asset.json`

复现命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/data/score_fold_experts.py
```

## 正式 benchmark 前还缺什么

1. 明确折叠规则：折几次、折线、袖子与衣身的目标层叠，以及允许哪些 180°/正反面等价。
2. 每个资产冻结一个经人工审查的 folded canonical target；若确有多种合法终态，则冻结
   有明确语义的有限 target 集，而不是用测试 rollout 更新目标。
3. 采集失败和 near-miss 轨迹，结合人工标签校准 distance threshold。专家 P95 只能定义
   “专家包络”，不能验证 false-positive rate。
4. 统一在机器人释放后继续仿真 5 秒，成功必须在这段窗口持续满足距离、速度和桌面接触
   条件；终态单帧只作为连续分数，不作为正式 success。
5. 用按资产分组的交叉验证检查阈值和诊断项，避免轨迹数最多的
   `rand_ZZCWITH5HY` 支配全局校准。
