# 项目索引

本文件只用于定位稳定入口，不解释具体功能或实验参数。

## 顶层文件

| 位置 | 用途 |
| --- | --- |
| `README.md` | 项目总览和最短使用入口 |
| `AGENTS.md` | Agent 的协作、表达、实现与工具约定 |
| `requirements.txt` | 唯一的共享 Python 依赖定义 |
| `scripts/setup-environment.sh` | 创建唯一 `.venv` 并补齐 Isaac Sim 5 binary runtime |
| `scripts/manage-gpu-keepalive.sh` | H100 空闲、任务使用和任务退出之间的保活生命周期入口 |
| `scripts/gpu-keepalive.py` | 单 GPU 有界矩阵保活 worker，仅由管理脚本启动 |
| `pyproject.toml` | 包、CLI、pytest 和 Ruff 配置 |
| `paper/` | 论文工程；中文初稿为 `paper/draft_zh.md`，LaTeX 入口为 `paper/main.tex` |
| `configs/` | 运行与实验配置 |
| `assets/` | 项目资产 |
| `assets/robots/` | Piper、ALOHA2、X5、UR5e_2F85、FR3 Duo 的固定源资产、许可证与完整工作站 USD；Panda 停用 |
| `assets/robots/<robot>/<robot>.usda` | 每种机器人唯一的双臂工作站入口；固定安装、物理、home/attach、拓扑和原生相机 |
| `assets/robots/fr3_duo/fr3_duo.usda` | FR3 Duo 桌边安装、原生三相机加独立虚拟 cam_low、同源 Robotiq 和动力学快照，正式 profile 为 fr3_duo |
| `assets/robots/fr3_duo/build_robotiq.py` | 离线复制 UR5e 同源 Robotiq 组件及 FR3 安装/拓扑覆盖，保留原 payload、相机和 TCP |
| `scripts/robots/check_dynamics.py` | 只读机器人 USD 的空载/接触动态复验；结果写入 outputs |
| `scripts/robots/check_cameras.py` | 五套本体的原生 USD 相机 home 出图检查，保留双轴视场、roll 与 clipping |
| `configs/rendering/materials.json` | 高质量视觉材质与 HDRI 的固定选择清单 |
| `data/` | 正式数据与模型产物 |
| `data/visual-assets/polyhaven/` | Poly Haven CC0 材质的本地 2K 工作副本 |
| `outputs/` | 可删除的运行输出 |
| `tests/` | 长期回归测试 |

## 源码

| 位置 | 职责 |
| --- | --- |
| `src/clothdojo/cli.py` | 用户命令入口 |
| `src/clothdojo/diagnostics.py` | 环境诊断 |
| `src/clothdojo/assets/` | 资产目录与元数据 |
| `src/clothdojo/assets/workstations.py` | 固定工作站部署、USD/纹理引用闭包检查；由 assets prepare/check-workstations 调用 |
| `src/clothdojo/assets/materials.py` | Poly Haven 固定清单、下载校验与物理纹理尺度 |
| `src/clothdojo/assets/mount_features.py` | STL/OBJ 安装孔截面拟合与 mesh 证据报告 |
| `src/clothdojo/assets/mount_alignment.py` | 安装孔型配准、位姿歧义和紧固件余量门禁 |
| `src/clothdojo/sim/` | 仿真场景与运行时 |
| `src/clothdojo/sim/workstation_review.py` | sim render-workstations 的统一 Newton 整体/home/attach 审查渲染 |
| `src/clothdojo/rendering.py` | 渲染视角、策略观测尺寸与审查/策略共用的 USD 外观读取补齐 |
| `src/clothdojo/robots.py` | 从五套工作站 USD 读取控制布局、安装根、attach、相机与回放拓扑 |
| `src/clothdojo/tasks.py` | 与机器人独立的任务身份及场景可用性 |
| `src/clothdojo/trajectory/` | 遥操、rollout 与评测共用的轨迹格式、读写和 Newton capture |
| `src/clothdojo/data/` | 数据转换与导出 |
| `src/clothdojo/data/retarget.py` | Piper 到目标 robot profile 的几何参考、门禁与多 GPU／同卡常驻 worker 调度 |
| `src/clothdojo/data/retarget_dynamics.py` | 共用 RobotDynamicsRetargeter：刚体逆动力学、一步 shooting、夹爪事件前馈与 action-only replay |
| `src/clothdojo/data/rollout_review.py` | canonical trajectory 的增量无 solver 轻量审查视频 |
| `src/clothdojo/collection/` | 采集控制面 |
| `src/clothdojo/teleop/` | 遥操运行时 |
| `src/clothdojo/training/` | 训练与数据处理入口 |
| `src/clothdojo/training/jax_sft.py` | `openpi-jax` 单次与矩阵训练入口 |
| `src/clothdojo/training/openpi_rlinf_sft.py` | `openpi-rlinf` 的 deterministic loader、FP32 master 与 EMA checkpoint worker 边界 |
| `src/clothdojo/training/augmentation.py` | SFT/CFG 默认关闭图像增强与固定左上裁剪；SFT 显式开启时提供正确 RNG，不修改上游安装包 |
| `configs/training/pi05_aloha_openpi_rlinf_noaug_bs128_30k.yaml` | 非镜像 NOCS、关闭图像增强的 B128/30k SFT 对照配置 |
| `configs/training/pi05_aloha_openpi_rlinf_mirrored_noaug_bs256_wd001_40k.yaml` | 四机32卡、镜像 NOCS、统一图像/文本输入并启用 weight decay 的 B256 SFT 配置；30k衰减后低学习率训练至40k |
| `src/clothdojo/training/steam.py` | STEAM critic 数据/超参数门禁、launch manifest 与三成员 checkpoint 合并 |
| `src/clothdojo/training/rlinf_steam.py` | RLINF STEAM 的确定性 DataLoader、state/full-model 与 FP32-master 运行时适配 |
| `src/clothdojo/training/worker_environment.py` | SFT/STEAM 在 worker 内应用通信配置，覆盖容器默认网卡与 RoCE 设置 |
| `src/clothdojo/training/rlinf_steam_state.py` | 当前项目拥有的 STEAM state-aware 训练/推理扩展与严格 checkpoint 加载门禁 |
| `src/clothdojo/training/steam_geometry.py` | NOCS 完整 Good 镜像数据筛选、逐帧几何特征及原/镜像一致性检查；不改 policy state |
| `configs/training/steam_value_shirt_flatten.yaml` | 三个独立 8-GPU STEAM evaluator member 共用的正式训练配置 |
| `configs/training/steam_value_shirt_flatten_geometry_bs2048.yaml` | 四机单成员、三成员串行的 16 维 geometry-state STEAM 配置 |
| `src/clothdojo/training/cfg.py` | 多 shard CFG 数据、标签 identity、采样权重和初始化 checkpoint 门禁 |
| `src/clothdojo/training/cfg_model.py` | 当前 `openpi_rlinf` core 上的 CFG condition routing、训练 forward 与 guided inference |
| `src/clothdojo/training/rlinf_cfg.py` | 项目拥有的 CFG worker、向量化 advantage loader、严格 checkpoint 加载与保存合同 |
| `src/clothdojo/training/cfg_data_runtime.py` | CFG 的 epoch-tagged 采样、无解码跳过已消费 batch 与续训状态校验 |
| `configs/training/cfg_pi05_aloha_rollout400.yaml` | 719 条受审专家轨迹与 400 条 rollout 的正式 STEAM/CFGRL 续训配置 |
| `src/clothdojo/rollout/` | 策略执行、共用 trajectory collection runner 与轨迹生成 |
| `src/clothdojo/rollout/policy.py` | SFT/CFG 服务、远端协议与 CFG+RTC 组合 |
| `src/clothdojo/rollout/guided_rtc.py` | Physical Intelligence guided RTC 数学 |
| `src/clothdojo/rollout/checkpoint.py` | OpenPI-RLinf SFT/CFG 权重格式、hash、layout 与 strict reload 门禁 |
| `src/clothdojo/rollout/conversion.py` | JAX EMA 到 rollout checkpoint 的一次性受检转换 |
| `src/clothdojo/rollout/parity.py` | JAX/PyTorch 同观测、同噪声的数值对齐门禁 |
| `src/clothdojo/rollout/trajectory_collection.py` | rollout 与 evaluation 共用的 worker、batch、轨迹恢复和事务发布 |
| `src/clothdojo/data/rollout.py` | canonical rollout/trajectory 的 LeRobot 四视角发布与确定性多 GPU 分片 |
| `src/clothdojo/data/lerobot.py` | 标准 LeRobot v2.1 读写、视频编码与 trajectory shard 无重编码合并 |
| `src/clothdojo/data/mirror.py` | Piper canonical collection 的双臂中轴镜像增强与完整独立发布 |
| `src/clothdojo/data/isaac_render.py` | 同一导出入口的 Isaac RTX 与逐 episode 外观随机化后端 |
| `src/clothdojo/training/cfg_pipeline.py` | STEAM advantage、数据 gate 与 CFG launch |
| `src/clothdojo/evaluation/collection.py` | 冻结 balanced-200 / 800-case 正式评测 condition 规划、离线 scoring 与发布 |
| `src/clothdojo/evaluation/managed.py` | 单一正式入口的策略服务、多机仿真、恢复、评分与进程清理 Coordinator |
| `src/clothdojo/evaluation/run_config.py` | 正式 run、冻结 benchmark 和 deployment 的严格配置边界 |
| `src/clothdojo/evaluation/worker.py` | 只写互斥 trajectory shard 的内部节点 Worker |
| `src/clothdojo/runtime_identity.py` | 多机推理与仿真节点共用的项目 Python 源码指纹 |
| `src/clothdojo/evaluation/scoring.py` | canonical trajectory 的离线指标计算 |
| `src/clothdojo/evaluation/results.py` | condition 读取与配对汇总 |
| `configs/evaluation/balanced200_20260901.yaml` | 正式800-case面板的稳定输入入口 |
| `configs/evaluation/runs/` | 模型、执行条件与输出身份的正式 run config |
| `configs/evaluation/deployments/` | 可替换的本机或SSH计算资源配置 |

新增代码前先检查现有目录是否已经拥有对应职责。具体命令和参数使用：

```bash
.venv/bin/clothdojo --help
.venv/bin/clothdojo <command> --help
```

## 文档

- [`development.md`](development.md)：环境、工具和目录约定。
- [`benchmark.md`](benchmark.md)：严重褶皱服装展平 benchmark 的数据路径与实验规划。
- [`steam.md`](steam.md)：STEAM 的 state/geometry 输入、NOCS 数据准备和 checkpoint 合同。
- [`retargeting.md`](retargeting.md)：机械臂 retarget 的使用入口、产物语义与验证边界。
- [`mirroring.md`](mirroring.md)：Piper 原始轨迹镜像增强的数据变换、产物和验证边界。
- [`mount_alignment.md`](mount_alignment.md)：机械臂底座孔型配准的证据等级、计算入口与验收关系。
- [`robot_dynamics.md`](robot_dynamics.md)：机器人 USD 动力学来源、单位与运行时适用边界。
- [`robot_workstations.md`](robot_workstations.md)：五套固定工作站、外部 payload 部署与统一渲染入口。
- [`reference_index.md`](reference_index.md)：官方文档与源码入口。
- [`compute_hosts.md`](compute_hosts.md)：远端计算节点。
- [`experiments/rapid_iteration_evaluation.md`](experiments/rapid_iteration_evaluation.md)：
  固定 12 seen + 12 unseen、每资产 4 初态的 checkpoint、后端、RTC 与
  exec-horizon 配对评测结果。
- [`todo.md`](todo.md)：用户明确保留的长期事项。

旧项目位于 `/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/repos/newton-cloth-rl`，只用于查找历史行为和证据。
