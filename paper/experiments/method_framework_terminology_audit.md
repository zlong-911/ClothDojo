# 方法框架图术语核对

2026-09-10。当前图依据中文主稿、人工纠正数据边界和聚合 SFT 配置核对。策略更新采用专家示范与人工纠正聚合后的监督微调。图内模块名说明操作，正文说明输入或机制。

| 位置 | 最终表达 | 含义 |
| --- | --- | --- |
| 初始训练 | Supervised fine-tuning / Pretrained VLA | 专家示范对预训练 VLA 做监督微调。 |
| 策略输入 | Policy inputs / NOCS images / Robot state · instruction | 区分视觉、机器人状态和语言指令。 |
| 任务评价 | Task metrics / Shape · orientation / Fold quality | 评价 rollout，并输出成功或失败轨迹。 |
| 人工纠正 | Rollout correction / Restore rollout state / Record VR corrections | 明确从已记录 rollout 恢复状态，而非在线接管。 |
| 轨迹筛选 | Rollout screening / Success / failure | 使用任务结果筛选成功数据并辅助失败审查，不表示优势标签。 |
| 纠正状态选择 | State selection / Review failures | 审查失败 rollout，选择待恢复的状态，沿底部回路进入人工纠正。 |
| 策略更新 | Policy update with human corrections / Supervised fine-tuning | 专家示范与人工纠正聚合后进行监督训练。 |
| 视觉扩展 | Domain randomization / Trajectory re-rendering | 改变材质、场景和光照；记录的运动与动作保持不变。 |
| 下游训练 | RGB policy training / RGB observations / Recorded actions | 重渲染图像和原动作形成下游训练数据。 |

## 数据流约束

- 展平与折叠采用相同训练流程。折叠使用五特征学习式质量评分，状态阈值固定为 $Q\geq0.1422866$；主成功率要求曾连续达标至少三秒，终态成功率要求最后三秒全部达标。
- $\mathcal D_E$ 初始化 $\theta_{\mathrm{init}}$。
- $\mathcal D_H$ 从选定 rollout 状态采集，直接进入策略更新分支。
- Updated policy 箭头表示训练得到的新策略，不限定从上一轮 checkpoint 继续优化；已有 strict-40k 聚合 SFT 从预训练模型重新训练。
- 成功 rollout 进入轨迹重渲染；自主动作不进入 NOCS 策略的监督更新。图中不使用正、负优势符号。
- 状态恢复与 VR 纠正合并在左下模块，只有一条从状态选择返回的连线，不再返回初始示范采集模块。
- 图注保留整体流程、符号及“重渲染保持运动与动作不变”，实现细节放在正文。

## 项目证据

- [中文方法稿](../method_draft_zh.md)
- [人工纠正数据边界](../../src/clothdojo/training/hil_sft.py)
- [聚合 SFT 配置](../../configs/training/pi05_aloha_openpi_rlinf_hil800_aggregate_fullonly_wd001_40k.yaml)
- [strict-40k 配置](../../configs/training/pi05_aloha_openpi_rlinf_hil800_aggregate_fullonly_wd001_b256_strict_40k.yaml)
- [人工纠正入口](../../src/clothdojo/collection/hil.py)
- [遥操作恢复](../../src/clothdojo/teleop/isaac_app.py)
- [轨迹重渲染](../../src/clothdojo/data/isaac_render.py)

保留既有 $\pi_{0.5}$、人工纠正和 Domain Randomization 术语来源；链接保存在图的 `.sources.json` 中。历史相关工作中对其他方法的描述不表示本文采用其训练算法。
