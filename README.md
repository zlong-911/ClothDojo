# ClothDojo

ClothDojo 是基于 Newton 1.5 的布料操作研究项目，统一管理仿真、策略 rollout、轨迹数据、训练和评测。

## 环境

默认环境需要 Linux、Python 3.11 和支持 CUDA 的 NVIDIA GPU：

```bash
./scripts/setup-environment.sh
```

项目只维护一个 `requirements.txt` 和一个 `.venv`。安装脚本先解析主依赖，再在同一环境中补齐 Isaac Sim 5 的 RTX、Replicator 和离线 extension cache；Isaac 的包元数据与训练栈存在硬版本冲突，因此不能用一条裸 `pip install -r requirements.txt` 正确完成两部分安装。H100/A100 可运行 Newton 与训练，但不用于需要 RT Core 的 Isaac 渲染。

## 统一入口

机器人固定工作站为 `piper`、`aloha2`、`x5`、`ur5e_2f85`、`fr3_duo`；Panda 已停用。
安装、home、attach、物理与相机由各自同名 USD 持有，不从任务配置重新拼装。
资产部署与整体/attach 渲染见 [工作站说明](docs/robot_workstations.md)。

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

任意规模的数据生成先准备初态；它不属于正式评测：

```bash
.venv/bin/clothdojo rollout make-states \
  --states-per-asset 8 --unseen-assets 10 \
  --gpus 0,1 --output data/initial_states
```

普通 rollout 在策略服务启动后运行；它保留 endpoint、GPU 和采样范围等开发参数：

```bash
.venv/bin/clothdojo rollout run \
  --scenario configs/rollout/shirt_flatten.yaml \
  --states data/initial_states --policy ws://127.0.0.1:8080 \
  --execution rtc --exec-horizon 10 --rtc-delay-steps 0 \
  --gpus 0 --batch-size 4 --output data/rollouts/example

.venv/bin/clothdojo benchmark run \
  --config configs/evaluation/runs/cfg_expert822_balanced200.yaml
```

折叠使用同一个 runner 和同一套 Piper、桌面、布料物理场景，但以独立的
`shirt_fold` task identity 写入 rollout 与 trajectory provenance。折叠初态必须由
显式的已展平 state pool 提供；`rollout make-states` 的随机散乱初态不作为折叠入口：

```bash
.venv/bin/clothdojo rollout run \
  --scenario configs/rollout/shirt_fold.yaml \
  --states /path/to/reviewed_flattened_states \
  --policy ws://127.0.0.1:8080 \
  --execution chunk --exec-horizon 10 \
  --gpus 0 --batch-size 4 \
  --output data/collections/shirt_fold/rollouts/<run>
```

`benchmark run` 固定使用 `balanced200_20260901` 的 100 个 seen、100 个 unseen
资产及每资产 4 个初态，共 800 个配对 case；资产、初态、模板、协议和划分均由
冻结 hash 约束。正式入口只接受一份严格 run config，由它引用模型和 deployment；
Coordinator 自动启动独占策略服务、本机或远端仿真 worker、补齐缺失轨迹、离线评分并
清理所属进程。固定面板路径和 endpoint 不再由用户传入。旧 96-case 面板和小规模
链路检查使用 `rollout run` 或读取已有历史结果，不能进入正式评测汇总。

轨迹保存完整的 50 Hz 状态与动作；评测指标随后从轨迹离线计算，不依赖策略服务或 GPU。默认结果表包含 Peak、终态、AUC distance、曾成功率、终态成功率和抬起时间（`airborne_seconds_mean`，每条轨迹平均累计秒数）。抬起时间不参与成功判据，定义见 [`docs/benchmark.md`](docs/benchmark.md)。重新打分与配对汇总分别使用：

```bash
.venv/bin/clothdojo benchmark score --condition <condition> \
  --protocol configs/evaluation/shirt_flatten.yaml \
  --templates data/benchmarks/balanced200_20260901/templates
.venv/bin/clothdojo benchmark summarize --condition <condition-a> --condition <condition-b>
```

更改指标或成功阈值后，仍使用 `benchmark score`，额外传入
`--rescore-from <原采集协议.yaml>`。这只重新评分：原始轨迹和采集协议保留，
不允许借此更改任务、评测时长或模板生成条件。

`benchmark run` 产物固定标记为 `training_eligible=false`；训练 rollout 使用 `rollout run`。RTC 的 `rtc-delay-steps` 是确定性的逻辑延迟，不随机器推理 wall time 改变。

## 审查与 LeRobot

轻量审查只回放主视角，不创建 LeRobot 数据集；重复执行时只渲染新增或变化的 trajectory。它同时适用于连续采集目录和 rollout，且直接恢复 Newton state，不初始化物理 solver：

```bash
.venv/bin/clothdojo data render-review \
  --source data/collections/shirt_flatten --device cuda:0
.venv/bin/clothdojo collect serve --config configs/collection.yaml
```

网页 HIL 审查在 rollout 视频上分别记录介入点和任务完成点；完成点保存审核人员
判定成功时的精确帧号，再次点击 `Good` 会取消该点。两类标注都会立即原子写入对应
rollout 的 `hil_review.json`，不依赖浏览器缓存。审核阶段可跨 rollout、轨迹和
资产连续标记介入点。已采集的 HIL 结果可在审查区与来源 rollout 切换；“一键导出
全部 Review”覆盖配置中的普通采集、HIL 介入结果和 rollout，排除
`teleoperation.gpu_ids` 后分配到本机其余 NVIDIA GPU。启动阶段在 GPU worker
选择“服装展平 HIL”或“服装折叠 HIL”
及目标资产，该资产的全部介入点会组成恢复队列。HIL 与普通折叠共用恢复采集流程：
恢复后先等待 trigger 对齐，按 `A` 开始录制，再按 `A` 停止并保存；保存后会自动恢复
当前点，允许继续对齐并重新录制。按 `X` 重置并切换到下一个恢复点；如果正在录制或
保存，则先保存本次结果再切换。同一个恢复点只保留最后一次成功录制，先前版本会从
`trajectories/` 原子移动到相邻的 `rejected/` 目录，并记录
`superseded_by_rerecording`。普通折叠的恢复点来自展平轨迹终态，HIL 的恢复点来自
审核员标记的 rollout 介入帧。

网页删除 HIL 结果时，轨迹会从正式 `trajectories/` 原子归档到相邻的 `rejected/`，
清除其派生 Review 视频，并将来源 rollout 介入点恢复为待采状态。

每个采集 profile 的三类轨迹使用独立根目录：`dataset_root` 保存普通遥操轨迹，
`hil_dataset_root` 保存 HIL 介入轨迹，`rollout_root` 保存 policy rollout。HIL manifest
仍保留来源 rollout、episode 和 sample index，用于追溯，但三类数据不会物理混放。
普通遥操与 HIL 位于同一任务目录，例如 `data/collections/shirt_flatten/trajectories`
和 `data/collections/shirt_flatten/hil/trajectories`；对应 rollout 位于
`data/collections/shirt_flatten/rollouts/<run>`。

普通服装展平遥操可为每个长期运行实例配置一对 GPU：`teleoperation.gpu_ids`
列出主 GPU，等长的 `teleoperation.standby_gpu_ids` 列出备用 GPU。操作者停止录制并
选中另一件衣服后，可在同一个控制台实例中请求切换；系统冻结当前画面，在备用 GPU
构建并验证新环境，成功后才切换并释放旧环境。Kit、CloudXR 和头显连接在此过程中
保持运行。HIL 队列按资产依次推进，按 `X` 走完当前资产后，会使用备用 GPU
自动切换到下一资产；跨资产 HIL 必须配置备用 GPU。当前配置为主 GPU 0、1
分别配对备用 GPU 2、3。普通折叠连续采集仍不支持跨资产切换。

新生成的 rollout `meta.json` 在 policy identity 中保存 checkpoint SHA-256、训练数据集
根目录及其 info/sidecar SHA-256；训练数据集 sidecar 再指向普通遥操 collection。
每条 HIL manifest 保存来源 rollout 根目录、rollout `meta.json` SHA-256、episode、
介入 sample 和源 trajectory manifest SHA-256，形成可校验的完整 lineage。

Piper 原始 collection 可以离线扩充为“原轨迹 + 双臂中轴镜像轨迹”的独立
canonical collection。转换完整复制原轨迹和 annotation，并对状态与动作做确定性
代数变换，不调用 FK、IK 或物理 solver：

```bash
.venv/bin/clothdojo data augment-mirror \
  --source data/collections/shirt_flatten \
  --output data/collections/shirt_flatten_mirrored
```

历史 collection 中裁剪前的 action 可以复制成一套完整的新 collection，并把 action
替换为 runtime 实际采用的关节限位 target。源轨迹始终只读；新 collection 独立复制
`states.npz`，轨迹格式不变，只在 manifest 中标记 `action_semantics=runtime_target`。
这个转换不处理力矩限幅；训练数据集和 norm stats 由训练配置显式选择：

```bash
.venv/bin/clothdojo data canonicalize-actions \
  --source /path/to/shirt_flatten/000004 --plan-only
.venv/bin/clothdojo data canonicalize-actions \
  --source /path/to/shirt_flatten/000004 \
  --output /path/to/shirt_flatten/000005
```

需要训练数据时，同一入口直接读取 canonical trajectory 和审查标注，回放四路视角并导出标准 LeRobot v2.1；审查视频不是训练输入：

```bash
.venv/bin/clothdojo data export-lerobot \
  --source data/collections/shirt_flatten --device cuda:0
```

同一批轨迹可只替换衣物纹理为纯白，用于与 NOCS 做严格配对对照：

```bash
.venv/bin/clothdojo data export-lerobot \
  --source data/collections/shirt_flatten \
  --output data/datasets/shirt_flatten_reviewed_good_white \
  --cloth-visual white --device cuda:0
```

canonical rollout 和历史 trajectory collection 都可以用
`--shard-count N --shard-index I` 做确定性的互斥分片：rollout 按 asset 分组，
trajectory collection 按原轨迹及其镜像配对分组。每个进程输出一个完整的
LeRobot shard，适合绑定到不同 GPU 并行导出。trajectory shards 完成后可以无视频重编码合并：

```bash
.venv/bin/clothdojo data merge-lerobot-shards \
  --shard data/datasets/shard_00 --shard data/datasets/shard_01 \
  --output data/datasets/shirt_flatten_mirrored_reviewed_good
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
新 OpenPI_RLinf SFT/CFG 默认文本为 `flatten the shirt`，关闭图像随机增强及固定左上裁剪；新模型部署使用同一文本与 `--image-preprocessing none`。详见 [训练边界](docs/development.md#openpi_rlinf-训练边界)。
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
