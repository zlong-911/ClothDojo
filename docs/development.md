# 开发指南

本文只说明开发工具和稳定约定。具体任务如何实现、运行和验收，由当前需求、代码与配置决定。

## Python 环境

项目使用 Python 3.11，只维护根目录 `requirements.txt` 和 `.venv`。完整环境通过统一入口创建：

```bash
./scripts/setup-environment.sh
```

脚本先按 `requirements.txt` 安装 Newton、训练、数据和服务依赖，再向同一个环境添加固定版本的 Isaac Sim 5 RTX、Replicator 与离线 extension cache，最后安装 ClothDojo。Isaac 的包元数据精确固定了 `torch==2.7.0`、`fsspec==2024.6.1`、`scipy==1.15.3` 和 `numpy==1.26.0` 等版本，与项目主栈存在不可同时解析的约束；因此 Isaac binary distributions 必须在第二步以 `--no-deps` 安装，主栈版本仍由唯一的 `requirements.txt` 管理。pip 的 requirements 格式不能只对部分条目设置 `--no-deps`，所以不要用裸 `pip install -r requirements.txt` 代替统一脚本来构建完整环境。

不在仓库中增加第二套 venv、Conda 环境、requirements 文件、安装期 patch 或生成式 lock 文件，除非用户针对具体问题另有决定。

Isaac RTX 离线导出在这个主环境内运行。它在同一进程中直接读取 Newton trajectory state，通过 Fabric bridge 交给 Isaac Replicator；导出时不重建或推进物理场景。安装的 extension cache 使 RTX 与 Replicator 扩展不依赖首次运行时再从 registry 下载。Isaac 6 的 Python 3.12 应用环境仍是现有 OpenXR 遥操作的外部应用运行时，不属于项目 `.venv`。

离线 RTX 导出还要求具备 RT Core 且 Vulkan 可用的 RTX GPU。H100/A100 没有 RT Core，不满足 Isaac Sim 官方要求；这些节点继续用于 Newton、训练和策略推理，不作为 Isaac 离线渲染节点。

## 命令行

`clothdojo` 是统一 CLI，注册在 `pyproject.toml`，分发逻辑位于 `src/clothdojo/cli.py`：

```bash
.venv/bin/clothdojo --help
.venv/bin/clothdojo <command> --help
```

新增用户入口时优先扩展该 CLI，不另外增加平行 launcher。一次性诊断可先放在 `outputs/`，确认需要长期维护后再进入源码。

## 代码工具

- **pytest**：测试入口为 `.venv/bin/pytest`，默认读取 `tests/`。
- **Ruff**：lint 入口为 `.venv/bin/ruff check src tests`；规则与 Python 目标版本在 `pyproject.toml`。
- **Git**：开始修改前查看 `git status --short`；用户已有改动保持原样。
- **文本检索**：优先 `rg` 和 `rg --files`；环境没有 ripgrep 时使用 `find`、`grep`。
- **文件修改**：Agent 使用 `apply_patch`，避免通过临时脚本批量改写文本。

格式、测试范围和运行参数不在文档中追加固定流程，需要时由当前任务决定。

## 宿主机命令

Codex shell 建议具备以下通用 CLI：

| 命令 | 用途 | 当前状态 |
| --- | --- | --- |
| `rg` | 快速全文检索 | 缺失 |
| `fdfind` / `fd` | 快速文件定位 | 缺失 |
| `jq` | JSON 查询与格式化 | 缺失 |
| `rsync` | 本地或 SSH 增量同步 | 缺失 |
| `tmux` | 保持远端交互会话 | 缺失 |
| `tree` | 查看紧凑目录结构 | 缺失 |
| `shellcheck` | Shell 静态检查 | 缺失 |
| `git`、`ssh`、`curl`、`wget` | 版本控制、远程连接和下载 | 已安装 |
| `cmake`、`ninja`、`gcc`、`g++` | 本地 C/C++ 构建 | 已安装 |

Ubuntu 22.04 可安装建议的基础集合：

```bash
sudo apt-get update
sudo apt-get install ripgrep fd-find jq rsync tmux tree shellcheck
```

Ubuntu 的可执行文件名通常是 `fdfind`；如需 `fd`，可在用户自己的 `~/.local/bin/` 建立短名，不要修改系统文件。`gh`、`uv`、`yq`、`shfmt` 和 CUDA Toolkit 属于按需工具，不作为 Codex 基础依赖。

当前未激活项目 venv，裸 `python` 与 `pip` 指向 `/opt/isaacsim-venv`。在仓库内始终显式使用：

```bash
.venv/bin/python
.venv/bin/pip
.venv/bin/pytest
.venv/bin/ruff
```

`nvcc` 缺失不等于 CUDA runtime 不可用；项目使用预编译 PyTorch/Warp wheel 时通常只需要兼容的 NVIDIA driver。只有编译 CUDA extension 或从源码构建时才需要 CUDA Toolkit。

## 主要依赖

| 类别 | 工具 | 用途 |
| --- | --- | --- |
| 仿真 | Newton、Warp、MuJoCo、MuJoCo Warp | 布料、机器人与 GPU 仿真 |
| 场景 | OpenUSD、Newton USD Schemas | USD 资产读取 |
| 训练 | PyTorch、JAX/OpenPI、RLINF | 模型训练与推理 |
| 数据与网格 | NumPy、PyArrow、LeRobot、PyAV、Trimesh | 数值、表格、轨迹、视频和 MJCF visual mesh |
| 服务 | FastAPI、Uvicorn | 本地控制与服务入口 |
| 配置 | PyYAML、Hydra、OmegaConf | 项目与上游配置 |

共享 Python 依赖的精确版本和安装来源看 `requirements.txt`；Isaac binary distributions 的固定版本和特殊安装边界看 `scripts/setup-environment.sh`。第三方行为以固定版本的官方文档与源码为准，入口集中在 [`reference_index.md`](reference_index.md)。

## Rollout 部署

正式评测只从 `benchmark run --config <run.yaml>` 进入。run config 记录策略及执行条件，
deployment config 只记录可替换的主机、GPU、端口与节点路径；冻结面板、协议、初态和
模板由 `configs/evaluation/balanced200_20260901.yaml` 统一持有，实际冻结数据位于
`data/benchmarks/balanced200_20260901/`，不依赖可删除的实验输出。`cloth_visual: white`
使用与 white 训练数据相同的纯白布料外观；默认仍为 `nocs`。外观同时写入 condition
与 trajectory 身份，续跑拒绝混用，不改变固定初态、物理或评分。

Coordinator 为每个仿真 worker 启动一个独占策略服务；节点可以是 `local` 或 SSH host。
远端节点必须给出绝对项目路径和 GPU，policy 节点另外给出仿真可访问的 `connect_host`。
每个节点 worker 只写计划分配给自己的
trajectory，只有 Coordinator 写 condition `meta.json`、建立完整 index 和评分。多机不依赖
Ray、NCCL 或跨机任务队列，只依赖互斥分片和各节点可见的共享项目路径。Coordinator
对同一输出持有排他锁，策略服务和仿真节点必须匹配同一项目源码指纹；任一节点退出时
终止其余所属进程，已经原子发布的轨迹可在下次运行中复用。

deployment 中 policy 与 simulation 的 GPU 总数必须相同。`host` 使用已有 SSH alias，
`connect_host` 是仿真节点访问 WebSocket 服务的地址；本机节点只写 `host: local`：

```yaml
schema: clothdojo_evaluation_deployment
policy:
  nodes:
    - host: h100-policy-a
      connect_host: 10.0.0.10
      project_root: /shared/ClothDojo
      python: /shared/env/bin/python
      gpus: [0, 1]
      port_start: 28000
simulation:
  nodes:
    - host: rtx-sim-a
      project_root: /shared/ClothDojo
      python: /shared/ClothDojo/.venv/bin/python
      gpus: [0, 1]
score_workers: 4
```

chunk 在每个 worker 内依次执行观测、推理和仿真，不重叠计算。显存足够时，可以在
deployment 中将策略服务与仿真绑定到同一张 GPU；否则将 policy 和 simulation nodes 分开。
仍需每个 worker 一个独立 endpoint，不能为省卡共享策略随机数状态。RTC 的异步执行应
另行测量，不能直接沿用 chunk 的部署结论。

重复执行同一 run config 会校验 condition 身份并补跑缺失轨迹；迁移时先确认旧 Coordinator
及其 worker 全部退出，再更换 deployment。保持 batch size、初态、policy seed 和模型身份
不变，GPU 数量与主机位置可以调整。隐藏的 `.tmp` 轨迹尚未发布，不计入完成数。

## OpenPI_RLinf 训练边界

SFT（OpenPI/JAX 与 OpenPI_RLinf）不续训。中断后使用新的输出目录，从配置的初始权重和
第 0 步重新训练，不覆盖旧产物。OpenPI_RLinf 的 `runner.resume_dir` 必须缺省或为 `null`，
`CLOTHDOJO_TRAIN_OUTPUT` 必须指向尚不存在的目录；JAX 重训使用新的 `logging.run_name`。
checkpoint 继续保留 EMA 推理权重及 online/optimizer 状态用于分析，不提供 SFT 恢复入口。
此约定不改变下述 CFG 的恢复能力。

新 OpenPI_RLinf SFT/CFG 默认文本为 `flatten the shirt`，数据入口在 tokenize 前应用 `actor.model.openpi_data.default_prompt`，覆盖旧数据中的大小写/句号，不修改数据集文件。launch manifest 的 `training_prompt` 记录实际训练文本，dataset 的 `prompt` 保留数据原文。state/action norm stats 与文本无关，可复用；其余归一化契约仍严格校验。JAX 配方和历史实验记录不变，新默认不再自动绑定旧 JAX 输入对齐实验。

SFT/CFG 默认不做图像随机增强，也绕过上游 `train=True, rng=None` 的错误固定左上裁剪；flow noise/time、数据 shuffle 和模型训练模式不变。SFT 的 `clothdojo.image_augmentation` 缺省为 `false`，显式设为 `true` 才复用固定版 RLinf 的随机裁剪、旋转和颜色扰动，使用已播种、可保存的 CPU RNG。修正只作用于模型实例，不修改安装包或 checkpoint 布局。新模型部署使用相同文本与 `--image-preprocessing none`；模型数值精度及 resize 逻辑不变。

对确认使用上游 `train=True, rng=None` 训练的历史原生 RLinf 权重，开发用
`policy serve-sft` / `serve-cfg` 显式传 `--image-preprocessing training_fixed_crop`；正式评测在
run config 的 `execution.image_preprocessing` 中记录同一值。推理会在 CUDA 上先将图像
转为训练 FSDP 使用的 BF16，再复用固定版上游 `preprocess_observation(train=True, rng=None)`。
它包含主相机左上 212×212 裁剪、resize、零角度旋转及所有相机的数值范围往返；wrist 不做几何裁剪，
但保留 BF16 舍入。chunk/RTC/CFG 共用一次预处理，模型保持 eval，不启用随机增强。
此前 `legacy_fixed_crop` 的 FP32 诊断实现仅保留在旧实验冻结目录，不与此选项混用。
默认 `none` 保持普通确定性推理，JAX 及 JAX 转换权重不套用这项历史修正；新 noaug 训练也使用 `none`。
策略身份及轨迹记录预处理选项，服务连接、采集和离线评分拒绝混用。

CFG 的 `actor.seed + rank` 控制训练随机数，`data.seed` 控制 mixture 采样。每个采样索引携带 epoch，常驻 worker 不依赖父进程里的 epoch 状态；gradient accumulation 跨 epoch 时以实际消费的 batch 为准。

CFG 精确续训要求 checkpoint 同时包含模型、优化器、scheduler，以及各 rank 的数据位置和 RNG 状态，且数据/训练配置与 rank 拓扑匹配。恢复时直接跳过已消费的索引，不解码旧 batch。缺少运行时状态的旧 checkpoint 仍可用于推理或初始化新训练，但不能作为精确 resume。

对应 CPU 回归测试：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q \
  tests/test_cfg_worker_runtime.py tests/test_training_augmentation.py \
  tests/test_cfg_training.py tests/test_rlinf_training.py
```

## 目录

- `configs/`：可读配置，不保存凭据。
- `assets/`：项目资产，由外部存储管理，不提交 Git。
- `data/`：正式数据集、checkpoint、缓存和可复用运行产物。
- `outputs/`：日志、调试结果、临时报告和可删除可视化。
- `tests/`：需要长期保留的回归测试。

`data/` 与 `outputs/` 均被 Git 忽略。正式路径由配置显式引用，不使用含义不明的 catch-all runtime 目录。

## 外部应用

Isaac Sim 6/CloudXR 遥操作作为独立应用运行。Isaac Sim 5 离线渲染使用项目主 `.venv`，并通过统一 `clothdojo data export-lerobot` 入口调用。

远端训练节点、共享路径和 Codex bootstrap 见 [`compute_hosts.md`](compute_hosts.md)。凭据不得写入仓库。
