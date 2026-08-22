# ClothDojo

ClothDojo 是一个可复现的布料操作项目，基于 Newton 1.5 构建仿真环境，
使用 RLINF 训练 π0.5 策略，并通过 RTC rollout 生成离线数据。

主要流程：

```text
专家数据 → SFT → RTC rollout → STEAM 标注 → CFG 训练
                                      │
冻结策略 → 场景随机化 → 基准数据生成与评测
```

## 主要功能

- Newton 1.5 双臂布料仿真与轨迹记录
- 专家轨迹清洗和 LeRobot v2.1 导出
- 通过 RLINF 启动 π0.5 SFT、STEAM 和 CFG 训练
- RTC 异步 action chunk 执行
- 三相机渲染及 rollout LeRobot 数据发布
- CFG 训练前的数据兼容性和 advantage 覆盖检查

## 安装

项目使用 Python 3.11 和一个 `requirements.txt`：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e .
.venv/bin/clothdojo env check --training --smoke
```

LeRobot 视频读取需要系统提供 FFmpeg 4～7 的动态库。

## 常用命令

验证完整 Newton 场景：

```bash
.venv/bin/clothdojo sim check \
  --assets assets \
  --device cuda:0 \
  --worlds 1
```

验证 RTC、三相机渲染、轨迹和 LeRobot 发布：

```bash
.venv/bin/clothdojo rollout check-rtc \
  --scenario configs/rollout/shirt_flatten_rtc_gate.yaml \
  --assets assets \
  --trajectory-output /path/to/trajectory \
  --lerobot-output /path/to/lerobot \
  --device cuda:0
```

该命令使用保持当前位置的验收策略，输出会标记为
`training_eligible: false`，不能直接用于 CFG 训练。

检查并启动 SFT：

```bash
.venv/bin/clothdojo train pi05 \
  --config configs/training/pi05_aloha.yaml \
  --check

.venv/bin/clothdojo train pi05 \
  --config configs/training/pi05_aloha.yaml
```

检查 CFG 输入并启动训练：

```bash
.venv/bin/clothdojo data check-cfg \
  --config configs/training/cfg_pi05_aloha.yaml

.venv/bin/clothdojo train cfg \
  --config configs/training/cfg_pi05_aloha.yaml \
  --check
```

导出清洗后的专家数据或重新计算 normalization statistics：

```bash
.venv/bin/clothdojo data export-lerobot \
  --source /path/to/reviewed_trajectory \
  --output /path/to/lerobot_v21

.venv/bin/clothdojo data compute-norm-stats \
  --dataset /path/to/lerobot_v21 \
  --output /path/to/norm_stats
```

## 数据位置

- `assets/` 保存项目所需的机器人、场景和服装资产，不提交到 Git。
- 数据集、checkpoint、缓存和训练输出必须放在项目目录之外。
- `newton-cloth-rl` 只作为迁移行为的参考，不是运行依赖。

## 文档

- [项目入口](docs/project_index.md)
- [架构](docs/architecture.md)
- [迁移进度](docs/migration.md)
- [轨迹格式](docs/trajectory.md)
- [RTC rollout](docs/rtc_rollout.md)
- [离线 CFG](docs/offline_cfg.md)
