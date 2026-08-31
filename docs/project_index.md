# 项目索引

本文件只用于定位稳定入口，不解释具体功能或实验参数。

## 顶层文件

| 位置 | 用途 |
| --- | --- |
| `README.md` | 项目总览和最短使用入口 |
| `AGENTS.md` | Agent 的协作、表达、实现与工具约定 |
| `requirements.txt` | 默认仿真、训练和数据环境定义 |
| `requirements-isaac-render.txt` | 可选 Isaac Sim 5 离线渲染依赖 |
| `pyproject.toml` | 包、CLI、pytest 和 Ruff 配置 |
| `paper/` | 论文工程；中文初稿为 `paper/draft_zh.md`，LaTeX 入口为 `paper/main.tex` |
| `configs/` | 运行与实验配置 |
| `assets/` | 项目资产 |
| `assets/robots/` | Piper、ALOHA2、X5、UR5e_2F85、FR3 Duo 的固定源资产、许可证与完整工作站 USD；Panda 停用 |
| `assets/robots/<robot>/<robot>.usda` | 每种机器人唯一的双臂工作站入口；固定安装、物理、home/attach、拓扑和原生相机 |
| `assets/robots/fr3_duo/fr3_duo.usda` | 已接受 FR3 Duo 桌面装配、原生三相机加独立虚拟 cam_low 和动力学快照，正式 profile 为 fr3_duo |
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
| `src/clothdojo/sim/` | 仿真场景与运行时 |
| `src/clothdojo/sim/workstation_review.py` | sim render-workstations 的统一 Newton 整体/home/attach 审查渲染 |
| `src/clothdojo/rendering.py` | 渲染视角接口与策略观测尺寸 |
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
| `src/clothdojo/training/steam.py` | STEAM critic 数据/超参数门禁、launch manifest 与三成员 checkpoint 合并 |
| `src/clothdojo/training/rlinf_steam.py` | RLINF STEAM 的确定性 DataLoader、state/full-model 与 FP32-master 运行时适配 |
| `configs/training/steam_value_shirt_flatten.yaml` | 三个独立 8-GPU STEAM evaluator member 共用的正式训练配置 |
| `src/clothdojo/rollout/` | 策略执行、共用 trajectory collection runner 与轨迹生成 |
| `src/clothdojo/rollout/policy.py` | SFT/CFG 服务、远端协议与 CFG+RTC 组合 |
| `src/clothdojo/rollout/guided_rtc.py` | Physical Intelligence guided RTC 数学 |
| `src/clothdojo/rollout/conversion.py` | JAX EMA 到 rollout checkpoint 的一次性受检转换 |
| `src/clothdojo/rollout/parity.py` | JAX/PyTorch 同观测、同噪声的数值对齐门禁 |
| `src/clothdojo/rollout/trajectory_collection.py` | rollout 与 evaluation 共用的 worker、batch、轨迹恢复和事务发布 |
| `src/clothdojo/data/rollout.py` | canonical rollout 的 LeRobot 四视角发布 |
| `src/clothdojo/data/isaac_render.py` | 同一导出入口的 Isaac RTX 与逐 episode 外观随机化后端 |
| `src/clothdojo/training/cfg_pipeline.py` | STEAM advantage、数据 gate 与 CFG launch |
| `src/clothdojo/evaluation/collection.py` | 评测 condition 规划、离线 scoring 与发布 |
| `src/clothdojo/evaluation/scoring.py` | canonical trajectory 的离线指标计算 |
| `src/clothdojo/evaluation/results.py` | condition 读取与配对汇总 |

新增代码前先检查现有目录是否已经拥有对应职责。具体命令和参数使用：

```bash
.venv/bin/clothdojo --help
.venv/bin/clothdojo <command> --help
```

## 文档

- [`development.md`](development.md)：环境、工具和目录约定。
- [`benchmark.md`](benchmark.md)：严重褶皱服装展平 benchmark 的数据路径与实验规划。
- [`retargeting.md`](retargeting.md)：机械臂 retarget 的使用入口、产物语义与验证边界。
- [`robot_dynamics.md`](robot_dynamics.md)：机器人 USD 动力学来源、单位与运行时适用边界。
- [`robot_workstations.md`](robot_workstations.md)：五套固定工作站、外部 payload 部署与统一渲染入口。
- [`reference_index.md`](reference_index.md)：官方文档与源码入口。
- [`compute_hosts.md`](compute_hosts.md)：远端计算节点。
- [`experiments/rapid_iteration_evaluation.md`](experiments/rapid_iteration_evaluation.md)：
  checkpoint chunk 对比与 RTC 参数初步评测结果。
- [`todo.md`](todo.md)：用户明确保留的长期事项。

旧项目位于 `/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/repos/newton-cloth-rl`，只用于查找历史行为和证据。
