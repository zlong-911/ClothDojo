# STEAM 评价器

训练入口是 `clothdojo train steam --config … --member-index 0`，三个成员分别使用
`0/1/2`，最后通过 `clothdojo steam merge --member … --member … --member … --output …`
合并。`--check` 校验数据、归一化、batch、placement 和精度，不分配 GPU。
项目扩展由 `rlinf_steam_state.py` 在进程内安装，不修改 RLINF 的 site-packages。

## 观测与几何特征

标准相机为 `observation.images.cam_high`、`observation.images.cam_low`。
原有评价器使用 14 维 `observation.state`；几何版本使用 NOCS 图像，并按顺序拼接：

1. `observation.state`：14 维机器人状态，原数据不变。
2. `observation.cloth_metrics`：`[deformable_distance, rigid_distance]`，均为越小越好。

几何指标只作为评价器的当前帧输入。它们不改变 STEAM 的正反时间对标签，也不进入
policy 的 14 维机器人 state。指标依赖仿真粒子状态和对应资产的 canonical target；
因此这是 privileged critic，后续 rollout 打标必须能够获得同样的指标，不能缺失补零。

`steam_geometry.py` 复用 evaluation 的 factorized metric。镜像轨迹先逆反射当前
粒子的 y 坐标，再与原 canonical target 比较，避免把镜像误当作变形。帧与 action
index 对齐，不读取未来帧或用专家终态充当目标。

## 数据准备

已有 NOCS 完整 Good 镜像数据可使用：

```bash
.venv/bin/python -m clothdojo.training.steam_geometry \
  --dataset <NOCS-LeRobot数据> \
  --trajectories <镜像collection的trajectories目录> \
  --templates <canonical-template集合> \
  --protocol configs/evaluation/shirt_flatten.yaml \
  --selection <原始轨迹ID到Good确认依据的JSON映射> \
  --output <新的LeRobot数据目录>
```

此入口要求每条入选原轨迹都有完整镜像，不接受局部分割片段；不修改源数据或伪造
人工 review 状态。只重写数值数据和元数据，视频使用相对链接复用，迁移时需要一起
保留其源数据目录。原/镜像指标逐帧一致性校验后才发布结果。

`meta/steam_state.json` 保存训练集的 16 维 mean/std 和特征定义；训练 gate 会重新
从 parquet 核对。`steam_value_shirt_flatten_geometry_bs2048.yaml` 从该文件读取统计量。
checkpoint 保存 `state_keys`、归一化和 `geometry_contract`，打标时要求数据合同相同，
合并成员时也拒绝不一致的 state 定义。geometry contract 采用 JSON 字符串，以适配
固定版本 RLINF 的扁平 model config。

原轨迹和它的镜像不是独立样本。做留出集比较时应按原始轨迹分组；不同 batch 的比较
应同时报告 step、已处理 pair 数和独立 rollout 指标，不能仅凭训练 loss 判断改善。
