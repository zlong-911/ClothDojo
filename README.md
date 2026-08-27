# ClothDojo

ClothDojo 是基于 Newton 1.5 的可复现布料操作项目，覆盖数据采集、π0.5
训练、并行 rollout 和多资产模型评估。

```text
专家数据 → SFT → rollout → STEAM 标注 → CFG 训练
                    │
固定初态池 ─────────┴→ checkpoint × 推理模式评估
```

项目的核心运行路径保持统一：一张 GPU 上的同资产初态组成 Newton world
batch，物理、三相机 tiled rendering 和策略推理各批量执行一次；`batch_size=1`
仍走同一套代码。普通 chunk 与 RTC 也共用一个 rollout loop，RTC 历史只由策略
会话维护。仿真和渲染使用同一个 Newton model，视觉 shape 不参与碰撞。

## 安装

需要 Python 3.11、支持 Newton 1.5 的 NVIDIA GPU；LeRobot 视频读取还需要系统
提供 FFmpeg 4～7 动态库。

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e .
.venv/bin/clothdojo env check --training --smoke
```

项目只使用这一套 Python 环境。`newton-cloth-rl` 是只读迁移参考，不是运行依赖。

## 数据位置

- `assets/`：机器人、场景和服装资产。
- `data/`：正式数据集、初态池、rollout、评估结果、checkpoint 和缓存。
- `outputs/`：可删除的日志、内部验收结果和临时可视化。
- `tests/`：长期回归测试。

`data/` 和 `outputs/` 均不提交 Git。

## 数据采集

首次使用先复制并修改采集配置：

```bash
cp configs/collection.yaml.example configs/collection.yaml
.venv/bin/clothdojo collect serve --config configs/collection.yaml
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。工作台统一提供
遥操、轨迹审查与 bad 区间标注、HIL 介入和资产清洗；设备连接、按键和数据布局见
[采集文档](docs/collection.md)。

## 并行 rollout

先从已接受资产生成可复用的初态池：

```bash
export CLOTHDOJO_EXPERT_TRAJECTORIES=/path/to/expert/trajectories

.venv/bin/clothdojo rollout make-states \
  --seen-assets 20 --unseen-assets 30 \
  --states-per-asset 32 \
  --gpus 0,1,2,3 \
  --output data/states
```

在策略 GPU 上启动 checkpoint：

```bash
.venv/bin/clothdojo policy serve-sft \
  --checkpoint /path/to/checkpoint \
  --norm-stats /path/to/norm_stats \
  --device cuda:1
```

在仿真 GPU 上批量消费初态：

```bash
.venv/bin/clothdojo rollout run \
  --scenario configs/rollout/shirt_flatten.yaml \
  --states data/states \
  --policy ws://127.0.0.1:8080 \
  --gpus 0 \
  --batch-size 4 \
  --output data/rollouts/step40000_chunk
```

加 `--execution rtc` 启用同步、零推理延迟的 guided RTC 对照。多 GPU 时每个
worker 必须使用独占策略端点，例如：

```text
--gpus 0,1 --policy ws://policy-a:8080,ws://policy-b:8080
```

正式输出始终是一层平坦的 canonical trajectory；内部 worker 和 batch 不进入数据
目录结构。随后可精确回放状态并发布 LeRobot v2.1：

```bash
.venv/bin/clothdojo data export-lerobot \
  --source data/rollouts/step40000_chunk \
  --device cuda:0
```

```text
data/
├── states/<asset_id>/states.npz
└── rollouts/<run>/
    ├── meta.json
    ├── trajectories/<asset_id>_<initial_state>/
    └── lerobot/
```

调度、固定 batch、断点恢复和 RTC 语义见 [rollout 文档](docs/rtc_rollout.md)。

## 模型评估

评估复用相同的批量环境和策略边界，但只写紧凑指标，不生成训练轨迹。先为初态池
中的每个资产生成无随机、自由沉降的模板，再校验协议：

```bash
.venv/bin/clothdojo benchmark make-templates \
  --states data/states \
  --protocol configs/evaluation/shirt_flatten.yaml \
  --gpus 2,3,4,5,6,7 \
  --output data/benchmarks/shirt_flatten_calibration

.venv/bin/clothdojo benchmark check-protocol \
  --protocol configs/evaluation/shirt_flatten.yaml
```

生成时机器人仍在场景中但不参与布料碰撞，工作台碰撞保持启用；一个模板由指标
围绕桌面中心生成 `0°/+180°` 两个等价目标，无需保存两份。

每个 checkpoint × 推理模式写入独立 condition 目录；chunk 与 RTC 做归因比较时
使用相同初态、case 集合和 `exec_horizon`：

```bash
.venv/bin/clothdojo benchmark run \
  --condition-id step40000_chunk_h10 \
  --scenario configs/rollout/shirt_flatten.yaml \
  --protocol configs/evaluation/shirt_flatten.yaml \
  --states data/states \
  --templates data/benchmarks/shirt_flatten_calibration/templates \
  --policy ws://127.0.0.1:8080 \
  --execution chunk --exec-horizon 10 \
  --gpus 0 --batch-size 4 \
  --output data/evaluations/step40000_chunk_h10

.venv/bin/clothdojo benchmark summarize \
  --condition data/evaluations/step40000_chunk_h10 \
  --condition data/evaluations/step40000_rtc_h10
```

主指标是固定时限内的稳定成功率和首次确认成功时间。成功一旦确认即保留；终态
距离、终态布料速度、best-settled 距离和 `success_at_end` 用于解释“时限到达时
仍在操作”的失败，但不会事后把 timeout 改判为成功。详见
[评估协议](docs/evaluation.md)。

## 训练

STEAM critic 使用三个独立的 8-GPU member 训练；`--member-index` 决定
value-head seed、run seed 和 Ray actor 名称。正式配置固定使用 reviewed-good
数据、state、全量 VLM、FP32 master、BF16 forward 和 global batch 512。

```bash
.venv/bin/clothdojo train steam \
  --config configs/training/steam_value_shirt_flatten.yaml \
  --member-index 0 --check
.venv/bin/clothdojo train steam \
  --config configs/training/steam_value_shirt_flatten.yaml \
  --member-index 0

.venv/bin/clothdojo steam merge \
  --member /path/to/member{0,1,2}/actor \
  --output /path/to/steam_ensemble/actor
```

```bash
# SFT 预检与启动
.venv/bin/clothdojo train pi05 --config configs/training/pi05_aloha.yaml --check
.venv/bin/clothdojo train pi05 --config configs/training/pi05_aloha.yaml

# CFG 数据预检与启动
.venv/bin/clothdojo data check-cfg --config configs/training/cfg_pi05_aloha.yaml
.venv/bin/clothdojo train cfg --config configs/training/cfg_pi05_aloha.yaml --check
```

## 验证与文档

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

- [项目入口](docs/project_index.md)
- [架构与依赖方向](docs/architecture.md)
- [迁移边界](docs/migration.md)
- [数据采集](docs/collection.md)
- [并行 rollout 与 RTC](docs/rtc_rollout.md)
- [模型评估](docs/evaluation.md)
- [轨迹格式](docs/trajectory.md)
- [LeRobot 导出](docs/lerobot_export.md)
- [离线 CFG](docs/offline_cfg.md)
