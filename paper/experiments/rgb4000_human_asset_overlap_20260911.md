# RGB 4000 与人工数据的资产对照

2026-09-11 只读核验。资产按 asset_id 去重，镜像不增加资产数。

导出：`data/datasets/rgb_flatten_wrist_opaque_train400_v1_20260910/meta/clothdojo.json`，export_status=complete，training_eligible=true。4000 条展平轨迹覆盖 400 件资产，每件恰好 10 条。来源为 `data/rollouts/dagger40k_release_window_train400_v1_20260910`。

专家 SFT 采用实际训练来源的资产集合：展平 `shirt_flatten_reviewed_good_white`，折叠 `shirt_fold_lr_allgood_runtime_target_nocs_20260903`；本次重新读取来源 manifest，确认与前次资产表一致。

HIL 读取 `data/collections/shirt_flatten/hil/trajectories` 与 `shirt_fold/hil/trajectories`：分别 204/214 条记录、88/125 件资产（不包含 rejected 目录）。已整理的展平 201 条原始纠正及折叠 213 条纠正分别覆盖同样的 88/125 件资产，因此本表资产交集不受这部分轨迹筛选影响。不将这些数量解释成所有历史采集尝试。

| 对照 | 展平人工数据 | 折叠人工数据 |
| --- | ---: | ---: |
| 专家 SFT 资产总数 | 182 | 156 |
| HIL 资产总数 | 88 | 125 |
| 400 件中与 SFT 重合 | 120 | 113 |
| 400 件中与 HIL 重合 | 67 | 102 |
| 400 件中同时与 SFT、HIL 重合 | 40 | 32 |
| 400 件中未出现在该任务 SFT 或 HIL | 253 | 217 |
| SFT 中未进入这 400 件的资产 | 62 | 43 |
| HIL 中未进入这 400 件的资产 | 21 | 23 |

按展平人工数据划分这 400 件：SFT 与 HIL 都见过 40 件，仅 SFT 见过 80 件，仅 HIL 见过 27 件，两者都没见过 253 件。对应 rollout 数为 400、800、270、2530 条。四组互斥且合计 400 件、4000 条。

若同时考虑展平和折叠的 SFT、HIL 四个人工资产集合，400 件中有 192 件从未出现在这些集合中。跨任务交集不能解释成这批 rollout 包含折叠轨迹。

实验含义：G 不只是原专家资产的轨迹扩充，也扩展了资产覆盖；E+G 对 E 检验的是两者合并的整体收益。253 件是相对于本次核验的展平 SFT/HIL 集合未见，不自动意味着对任何历史训练、所有模型或任意测试协议都未见。400 件也不是人工资产的超集，不能将其描述为覆盖了全部专家服装。
