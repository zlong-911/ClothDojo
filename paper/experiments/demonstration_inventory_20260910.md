# 示范规模与折叠初态来源核验（2026-09-10）

本记录核对中文主稿现有生成策略主表使用的专家训练数据。只读取 dataset sidecar、episode 列表、collection manifest 与现有审计，不读取状态／动作数组，不使用 GPU。下列路径均相对 ClothDojo 仓库根目录。

## 计数结论

| 口径 | 展平 | 折叠 |
| --- | ---: | ---: |
| 当前 canonical collection 中完整原始轨迹 | 890 | 537 |
| 上述完整轨迹覆盖的唯一资产 | 182 | 156 |
| 实际进入专家训练集的原始轨迹（去重） | 889 | 536 |
| 专家训练片段（不含镜像） | 919 | 536 |
| 实际训练覆盖的唯一资产 | 182 | 156 |
| 不含镜像的训练状态—动作帧 | 1,576,387 | 714,211 |
| 含镜像的专家训练片段 | 1,838 | 1,072 |

因此，1,455 是两个任务的**非镜像训练片段数**，不是独立采集轨迹数；实际贡献训练样本的原始轨迹合计为 1,425。当前保留的完整原始轨迹合计为 1,427，不能据此推断历史所有采集尝试总量。资产按 `metadata.asset_id` 去重，镜像不增加资产。

## 展平：890 个来源如何变成 919 个片段

- 入口：`data/datasets/shirt_flatten_reviewed_good_white/meta/clothdojo.json`。
- 来源：该文件的 `source.trajectory` 指向 `data/collections/shirt_flatten_mirrored`；其 `selection.augmentation` 为 `original_only`。读取 `episodes[].source_trajectory`，再读取对应 `trajectories/<id>/manifest.json` 的 `metadata.asset_id`。
- `counts.source_trajectories=890`、`represented_source_trajectories=889`、`output_episodes=919`；919 个片段中，844 个覆盖完整源轨迹，75 个为局部片段。坏段的补集按最少 65 帧导出，一条来源可以产生多个训练片段。
- `dropped_short_source_trajectories=1`。890 个完整非镜像来源与 889 个实际来源的集合差仅为 `20260730T083252.099327Z`，资产为 `rand_ZZCWITH5HY`。该源本身有 1,763 个 action，不能称为“原轨迹过短”；准确说法是剔除坏段后没有满足最短长度的训练片段。
- 当前 `data/collections/shirt_flatten/trajectories` 有 892 份 manifest，其中 890 份状态为 `complete`，资产总数为 182。完整源轨迹口径不包括另两份未完成记录。

White 与 NOCS 919 的 episode 身份及全部状态／动作已由既有审计逐一核对一致，记录见 `outputs/evaluations/nocs919_noaug_20260831/README.md` 和 `docs/experiments/evaluation_results.md` 第八节。本轮使用可直接读取的 White sidecar 恢复相同训练来源；NOCS checkpoint 的 launch 记录为 `data/checkpoints/downloaded/nocs919_noaug_b128_s43_30k_20260831/clothdojo_launch.json`，其中记录 919 episodes、1,576,387 frames。

## 折叠：536 条示范、156 个资产

- 入口：`data/datasets/shirt_fold_lr_allgood_runtime_target_nocs_20260903/meta/clothdojo.json`。
- 来源：`data/collections/shirt_fold_lr_allgood_runtime_target_20260903/trajectories`。
- Sidecar 共 1,072 个完整训练 episode；过滤 `source_trajectory` 以 `__mirror` 结尾的副本，剩 536 个 episode、536 个唯一原始来源、156 个唯一资产，总长 714,211 帧。
- 该 dataset 的 `meta/episodes.jsonl` SHA256 为 `da6270949e7492599dfaa4c633d1dab3ad46066cfdce762d5ec6842aa9f3f661`，sidecar SHA256 为 `0775152689ae5ef5140324f1835b75bbfcdbe11abb2b8712853e5cd87db89894`；两者均与 `data/checkpoints/downloaded/fold_clean_mirrored_40k_20260908/clothdojo_launch.json` 记录完全一致。远端 dataset 名称虽为 `shirt_fold_clean`，本地文件确为该训练输入。
- 当前 `data/collections/shirt_fold/trajectories` 有 537 条完整原始轨迹、156 个资产。唯一未进入上述训练集的是 `20260813T033442.680990Z`，仅 40 个 action，低于 65 帧最短长度。正文的 537 条专家质量参考与 536 条专家训练示范应分别统计。

## 折叠初态确实来自展平示范终态

536 条非镜像折叠训练轨迹的 manifest 全部记录：

```text
metadata.initialization.protocol = terminal_episode_restore_v1
metadata.source = task_continuation
metadata.source_task_name = shirt_flatten
metadata.source_profile_id = shirt_flatten_v1
metadata.control_source = vr_teleop
```

这些记录指向 535 个不同展平源 episode。534 条可在当前 `data/collections/shirt_flatten/trajectories` 找到源 manifest，且全部满足 `source_sample_index == source.state_count - 1`，直接证实恢复的是末帧；未发现恢复到中间帧的反例。另两条历史源不在当前 collection，不能做相同的交叉核对，其折叠 manifest 仍明确记录终态恢复协议：

| 折叠 episode | 展平源 episode | source_sample_index |
| --- | --- | ---: |
| `20260813T084141.977060Z` | `20260730T024251.580081Z` | 3,119 |
| `20260813T092804.892322Z` | `20260730T033558.580808Z` | 3,570 |

示例直接证据：`data/collections/shirt_fold_lr_allgood_runtime_target_20260903/trajectories/20260812T093554.198397Z/manifest.json` 记录展平源 `20260804T062344.540781Z` 与末帧索引 1,987。当前恢复实现也将 continuation 限定为终态：`src/clothdojo/teleop/restoration.py` 的 `_load_continuation` 调用 `_readers(..., require_terminal=True)`，并校验末帧索引。

适合正文的表述是“折叠示范从展平示范的终态恢复仿真，再由操作者接续折叠”。这是跨任务的初态复用；两任务的数据仍按独立 episode 训练，不能称为连续展平—折叠轨迹训练。

## 历史训练配置备注

`data/checkpoints/downloaded/nocs919_noaug_b128_s43_30k_20260831/clothdojo_launch.json` 与 `outputs/training/pi05_nocs919_noaug_promptfix_b128_s43_20260831/preflight.json` 均记录 `enable_gradient_accumulation=true`、`global_batch_size=128`、`micro_batch_size=8`、`cluster.num_nodes=1`、`component_placement.actor="0-7"`。这是指向 8 个 actor 的历史启动配置；本轮未找到实际进程日志来独立确认运行时 world size，不能仅凭配置断言实际累积次数。足以确认正文不应无证据声称历史模型“均不使用梯度累积”。当前仓库禁止新训练使用梯度累积的规则不受此历史核验影响。

## 可复核的集合指纹

以下指纹对去重排序后的 ID 逐行连接、末尾保留一个换行，再计算 SHA256。

| 集合 | SHA256 |
| --- | --- |
| 展平实际训练的 889 个原始来源 ID | `5a7d51ffd6bb6a3636609d20c7cab3e6582150894dbaf8ac54fc22f9f7574153` |
| 展平实际训练的 182 个资产 ID | `a88286590ca1ae5cdda7eb3abd6c2a443f80a858f71324da5d46551ac92d6ddc` |
| 折叠实际训练的 536 个原始来源 ID | `ae45ef0aa526f263bea4e4747af96526b2f40736f14963dbbfcf4847a2050c2b` |
| 折叠实际训练的 156 个资产 ID | `90da865c5fe26a53314c944f05ddbc140163730d6679e074ca58e6cb557ee015` |
