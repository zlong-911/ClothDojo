# ClothDojo

ClothDojo 是基于 Newton 1.5 的布料操作研究项目，统一管理仿真、策略 rollout、轨迹数据、训练和评测。

## 环境

默认环境需要 Linux、Python 3.11 和支持 CUDA 的 NVIDIA GPU：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e .
```

Newton、训练、评测和默认数据导出都使用 `requirements.txt`。可选 Isaac RTX 离线渲染使用独立环境和 `requirements-isaac-render.txt`；H100/A100 可运行 Newton 与训练，但不用于需要 RT Core 的 Isaac 渲染。

## 统一入口

所有正式功能都从一个 `clothdojo` CLI 进入：

```bash
.venv/bin/clothdojo --help
.venv/bin/clothdojo <command> --help
```

主要命令只有以下职责：

| 命令 | 职责 |
| --- | --- |
| `rollout` | 生成初态和策略完整轨迹 |
| `benchmark` | 规划评测 condition、离线打分和汇总 |
| `policy` | 转换或启动 JAX、SFT、CFG 策略服务 |
| `data` | 轨迹审查、LeRobot 导出和数据检查 |
| `train` | OpenPI SFT、STEAM 和 CFG 训练 |
| `collect` | 遥操、HIL 审查和资产管理网页 |

项目只有一个 rollout runner：单 world 是 batch=1，多 world 使用 Newton Worlds、整步 CUDA Graph 和 `SensorTiledCamera`。`rollout run` 与 `benchmark run` 共用相同的 worker、batch、轨迹恢复和事务发布逻辑。
策略观测、相机标定和传输尺寸会写入轨迹 provenance；离线 LeRobot/Isaac
导出只读取 canonical state，不反向进入 rollout 执行链。

## Rollout 与评测

先准备固定初态和自由沉降 canonical target：

```bash
.venv/bin/clothdojo rollout make-states \
  --states-per-asset 8 --unseen-assets 10 \
  --gpus 0,1 --output data/initial_states

.venv/bin/clothdojo benchmark make-templates \
  --states data/initial_states \
  --protocol configs/evaluation/shirt_flatten.yaml \
  --gpus 0,1 --output data/templates
```

策略服务启动后，同一入口可运行 chunk 或 RTC：

```bash
.venv/bin/clothdojo rollout run \
  --scenario configs/rollout/shirt_flatten.yaml \
  --states data/initial_states --policy ws://127.0.0.1:8080 \
  --execution rtc --exec-horizon 10 --rtc-delay-steps 0 \
  --gpus 0 --batch-size 4 --output data/rollouts/example

.venv/bin/clothdojo benchmark run --condition-id example_chunk \
  --scenario configs/rollout/shirt_flatten.yaml \
  --protocol configs/evaluation/shirt_flatten.yaml \
  --states data/initial_states --templates data/templates \
  --policy ws://127.0.0.1:8080 --gpus 0 --batch-size 4 \
  --output data/evaluations/example_chunk
```

轨迹保存完整的 50 Hz 状态与动作；评测指标随后从轨迹离线计算，不依赖策略服务或 GPU。结果同时保留终态、best-observed、时域平均距离和成功判据。重新打分与配对汇总分别使用：

```bash
.venv/bin/clothdojo benchmark score --condition <condition> \
  --protocol configs/evaluation/shirt_flatten.yaml --templates data/templates
.venv/bin/clothdojo benchmark summarize --condition <condition-a> --condition <condition-b>
```

`benchmark run` 产物固定标记为 `training_eligible=false`；训练 rollout 使用 `rollout run`。RTC 的 `rtc-delay-steps` 是确定性的逻辑延迟，不随机器推理 wall time 改变。

## 审查与 LeRobot

轻量 HIL 审查只回放主视角，不创建 LeRobot 数据集；回放直接恢复 Newton state，不初始化物理 solver：

```bash
.venv/bin/clothdojo data render-rollout-review \
  --source data/rollouts/example --device cuda:0
.venv/bin/clothdojo collect serve --config configs/collection.yaml
```

需要训练数据或完整回放时，从同一 canonical trajectory 导出 LeRobot v2.1：

```bash
.venv/bin/clothdojo data export-lerobot \
  --source data/rollouts/example --output data/datasets/example \
  --repo-id local/example --device cuda:0
```

Isaac RTX 外观随机化仍使用这个 `data export-lerobot` 入口。首次使用先准备固定的视觉资产集合：

```bash
.venv/bin/clothdojo data prepare-isaac-assets
```

随后通过 `--isaac-seed`、`--randomize` 和 `--visual-assets` 选择可选后端；需要 RoboDojo 房间时，在准备和导出命令中都加 `--include-room`。

## 训练

OpenPI SFT 只保留两个入口：

```bash
.venv/bin/clothdojo train openpi-jax \
  --config configs/training/pi05_aloha_jax.yaml --check
.venv/bin/clothdojo train openpi-rlinf \
  --config configs/training/pi05_aloha_openpi_rlinf_lr2x_60k.yaml --check
```

JAX EMA checkpoint 通过 `policy convert-jax` 做一次带数值对齐门禁的 PyTorch 转换，之后 chunk 与 RTC 都走同一策略协议和 rollout runner。STEAM critic 与 CFG 是后续算法训练，仍分别使用 `train steam`、`steam merge` 和 `train cfg`，不是额外的 OpenPI SFT 后端。
用于框架比较的 `openpi-rlinf` 配置显式引用对应 JAX trial；启动前会从该
reference 推导并核对 batch、seed、optimizer、LR schedule、EMA、数据和初始化，
不在训练实现中硬编码某一组实验超参数。

## 开发与文档

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

- [项目索引](docs/project_index.md)：稳定代码、配置和文档入口。
- [开发指南](docs/development.md)：环境、工具和目录约定。
- [资料索引](docs/reference_index.md)：固定版本的官方文档和源码。
- [计算节点](docs/compute_hosts.md)：远端资源和连接方式，不含凭据。
