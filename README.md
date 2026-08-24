# ClothDojo

ClothDojo 是一个可复现的布料操作项目，基于 Newton 1.5 构建仿真环境，
使用 RLINF 训练 π0.5 策略，并通过 policy rollout 生成离线数据。

主要流程：

```text
专家数据 → SFT → policy rollout → STEAM 标注 → CFG 训练
                                      │
冻结策略 → 场景随机化 → 基准数据生成与评测
```

## 主要功能

- Newton 1.5 双臂布料仿真与轨迹记录
- 遥操、轨迹审查、HIL 和资产清洗网页工作台
- 专家轨迹清洗和 LeRobot v2.1 导出
- 通过 RLINF 启动 π0.5 SFT、STEAM 和 CFG 训练
- 同步 action chunk rollout，并保留可选 RTC 对照
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

## 网页工作台

网页工作台是数据采集与管理的统一入口，保留原项目的四面板设计，提供遥操、
轨迹视频审查与 bad 区间标注、HIL 介入点规划和服装资产筛选。前后端都随
ClothDojo 安装，不需要 Node 或另一套 Python 环境。

首次使用时复制配置样例；已有配置不要覆盖：

```bash
cp configs/collection.yaml.example configs/collection.yaml
```

主要需要核对 `profiles.<任务>.dataset_root`。审核数据应采用以下布局：

```text
data/collections/<任务>/
├── trajectories/
│   ├── <episode_id>/manifest.json
│   └── annotations/<episode_id>.segments.json
└── reviews/review-*/
```

启动服务：

```bash
.venv/bin/clothdojo collect serve --config configs/collection.yaml
```

样例配置监听 `0.0.0.0:8000`。本机访问
[http://127.0.0.1:8000](http://127.0.0.1:8000)，内网其他机器访问
`http://<服务器内网 IP>:8000`。启动命令默认在前台运行，终端退出或按
`Ctrl-C` 后服务会停止。可用下面的命令检查服务和数据索引：

```bash
curl http://127.0.0.1:8000/api/snapshot
```

页面中的常用操作：

- **遥操**：选择任务、服装和空闲 GPU，点击“启动”，再用 Pico 4 Ultra 扫描页面给出的
  CloudXR 二维码。双手 `squeeze` 是 clutch，`trigger` 控制夹爪，右手 `A`
  开始或结束录制，左手 `X` 重置场景；`Y`/`B` 分别让左/右臂回 Home，松开后
  再握紧对应手柄才恢复跟手。夹爪进入可抓取区域时对应手柄持续震动；同时按下
  两侧摇杆重置 XR 视点。布料按静止网格 NOCS 着色，形变时颜色身份保持不变。
- **审查**：选择轨迹和相机；空格播放/暂停，左右方向键逐帧移动。点击两次
  “标记 bad 片段”确定区间，没有 bad 区间时可点击 `Good`。标注会自动写入
  `trajectories/annotations/`。
- **HIL**：在 rollout 视频上添加介入点并启动遥操。运行时校验 rollout、轨迹和
  资产哈希，冷恢复完整机器人、布料、夹爪与抓取生命周期；安全接管后按 `A`
  录制，结束后按 `X` 进入下一个介入点。
- **续采**：`shirt_fold` 从尚未消费的 `shirt_flatten` 完整轨迹终态恢复；来源按
  会话租约防重，按 `A` 录制，按 `X` 保存并进入下一个终态。
- **资产清洗**：查看服装正反面预览并接受或拒绝；资产决定保存于
  `assets/garments/selection.json`。

网页服务只负责会话管理和正式数据索引。实时 XR 输入、Newton 物理和轨迹写入
由固定的 NVIDIA Isaac Teleop + CloudXR/ClothDojo 进程负责。GPU 遥操默认并
强制捕获 Newton 物理与 IK CUDA Graph；会话状态中的 `physics_cuda_graph` 和
`ik_cuda_graph` 可用于确认实际启用情况。性能排查可同时查看 `loop_fps`、
`frame_ms`、`app_ms`、`physics_submit_ms` 和 `ik_updates`；正常松开 clutch 时 IK
不会重复求解。CloudXR 固定使用 20 FPS 的低帧率应用 pacing；这是给 72/90 Hz
头显的调度提示，不会改变头显刷新率或 Newton 的仿真时间步。

## Rollout 离线数据

先从已接受并完成转换的资产中选择 seen/unseen 资产，并生成初态。seen 表示
专家轨迹中出现过明确 `asset_id` 的资产：

```bash
export CLOTHDOJO_EXPERT_TRAJECTORIES=/path/to/expert/trajectories

.venv/bin/clothdojo rollout make-states \
  --seen-assets 20 \
  --unseen-assets 30 \
  --states-per-asset 32 \
  --gpus 0,1,2,3 \
  --output data/states
```

每个资产的全部初态保存在 `states/<asset_id>/states.npz`，根目录只有一个
`meta.json`。命令中断后可以原样重跑，已经完成的资产会自动跳过。

在策略 GPU 上启动训练好的 SFT checkpoint：

```bash
.venv/bin/clothdojo policy serve-sft \
  --checkpoint /outputs/pi05-good-v2/checkpoints/global_step_40000 \
  --norm-stats /data/norm_stats/pi05_aloha_shirt_flatten_reviewed_good_v2 \
  --device cuda:1
```

`serve-sft` 会在 checkpoint 的祖先目录中查找训练时写入的
`pi05_launch_manifest.json`，并在加载模型前同时核对 `norm_stats.json` 和
`clothdojo.json` 的 SHA-256 与变换 provenance。因此移动 checkpoint 时应保留其
run 目录结构；good-v2 checkpoint 与 v1 normalization statistics 的混搭会直接失败。

在仿真 GPU 上消费初态池，输出多条与遥操相同格式的物理轨迹：

```bash
.venv/bin/clothdojo rollout run \
  --scenario configs/rollout/shirt_flatten.yaml \
  --states data/states \
  --policy ws://127.0.0.1:8080 \
  --gpus 0 \
  --output data/rollouts/sft_step10000_chunk
```

checkpoint 和 normalization statistics 的身份会自动从服务读取并记录。如需
正式运行前强制核对，可额外传入 `--expected-checkpoint-sha256`。
多 GPU rollout 必须为每个活跃 worker 提供一个独占策略端点，例如
`--gpus 0,1 --policy ws://policy-a:8080,ws://policy-b:8080`；端点不会在 worker
之间共享，以免不同 episode 的随机数流互相干扰。

最后离线回放轨迹、渲染三相机画面并导出 LeRobot v2.1。输出默认放在同一
rollout 目录的 `lerobot/`：

```bash
.venv/bin/clothdojo data export-lerobot \
  --source data/rollouts/sft_step10000_chunk \
  --device cuda:0
```

最终目录保持简单：

```text
data/
├── states/
│   ├── meta.json
│   └── <asset_id>/states.npz
└── rollouts/<模型或推理名称>/
    ├── meta.json
    ├── trajectories/<asset_id>_<初态序号>/
    │   ├── manifest.json
    │   ├── states.npz
    │   └── actions.npz
    └── lerobot/
```

`trajectories/` 与遥操使用相同的轨迹格式，是主数据；`lerobot/` 是可重新生成
的训练视图。多 GPU 队列和内部批处理不会增加输出目录层级。

## 其他常用命令

验证完整 Newton 场景：

```bash
.venv/bin/clothdojo sim check \
  --assets assets \
  --device cuda:0 \
  --worlds 1
```

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
- `data/` 保存正式数据：数据集、初态池、rollout、checkpoint 和缓存。
- `outputs/` 保存可随时删除的日志、内部验收结果和临时可视化。
- `tests/` 只放长期回归测试，不放一次性开发或诊断脚本。
- `data/` 和 `outputs/` 都由 Git 忽略。
- `newton-cloth-rl` 只作为迁移行为的参考，不是运行依赖。

## 文档

- [项目入口](docs/project_index.md)
- [架构](docs/architecture.md)
- [迁移进度](docs/migration.md)
- [轨迹格式](docs/trajectory.md)
- [网页工作台与 XR 架构](docs/collection.md)
- [审核数据导出](docs/lerobot_export.md)
- [RTC rollout](docs/rtc_rollout.md)
- [离线 CFG](docs/offline_cfg.md)
