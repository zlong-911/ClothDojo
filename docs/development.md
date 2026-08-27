# 开发指南

本文只说明开发工具和稳定约定。具体任务如何实现、运行和验收，由当前需求、代码与配置决定。

## Python 环境

项目使用 Python 3.11，Newton 仿真、训练和默认导出的依赖由根目录 `requirements.txt` 定义：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e .
```

不在仓库中增加第二套 venv、Conda 环境、安装期 patch 或生成式 lock 文件，除非用户针对具体问题另有决定。

Isaac RTX 离线导出是明确的可选运行时。它仍使用 Python 3.11，在同一进程中直接读取 Newton trajectory state，通过 Fabric bridge 交给 Isaac Replicator；导出时不重建或推进物理场景。独立 venv 先安装 `requirements.txt`，再用 `pip install --no-deps -r requirements-isaac-render.txt` 添加最小 Kit 运行时。`--no-deps` 是必要边界：Isaac 的完整 Python 依赖集与训练栈存在精确版本冲突，而导出路径不使用那些上层工具包。这样不需要 Python 3.11/3.12 IPC，也不把 Isaac 包安装到普通训练节点。Isaac 6 的 Python 3.12 应用环境仍只服务现有 OpenXR 遥操作。

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
| 数据 | NumPy、PyArrow、LeRobot、PyAV | 数值、表格、轨迹和视频 |
| 服务 | FastAPI、Uvicorn | 本地控制与服务入口 |
| 配置 | PyYAML、Hydra、OmegaConf | 项目与上游配置 |

默认环境的精确版本和安装来源看 `requirements.txt`；可选 Isaac 渲染额外看 `requirements-isaac-render.txt`。第三方行为以固定版本的官方文档与源码为准，入口集中在 [`reference_index.md`](reference_index.md)。

## 目录

- `configs/`：可读配置，不保存凭据。
- `assets/`：项目资产，由外部存储管理，不提交 Git。
- `data/`：正式数据集、checkpoint、缓存和可复用运行产物。
- `outputs/`：日志、调试结果、临时报告和可删除可视化。
- `tests/`：需要长期保留的回归测试。

`data/` 与 `outputs/` 均被 Git 忽略。正式路径由配置显式引用，不使用含义不明的 catch-all runtime 目录。

## 外部应用

Isaac Sim 6/CloudXR 遥操作作为独立应用运行。Isaac Sim 5 离线渲染使用上文的可选 Python 3.11 环境，并通过统一 `clothdojo data export-lerobot` 入口调用。

远端训练节点、共享路径和 Codex bootstrap 见 [`compute_hosts.md`](compute_hosts.md)。凭据不得写入仓库。
