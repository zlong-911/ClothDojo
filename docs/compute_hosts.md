# 计算节点

当前按用户 2026-09-04 提供的完整清单登记 5 台 GPU worker：4 台 H100 和
1 台 RTX 5090。`33.229.141.59` 已连接确认有 8 × RTX 5090 D v2 24GB；
4 台 H100 均已连接确认有 8 × H100 80GB，并已验证项目自动化 key、共享路径、
`br0` 和 InfiniBand 设备。未列入本次清单的旧 worker 不计入当前资源。

下列 H100 节点适合 Newton、训练和策略推理，不用于 Isaac RTX 离线导出。H100 没有 RT Core，不满足 Isaac Sim 的渲染要求；离线导出应调度到 Vulkan 可用的 RTX GPU。

## H100 占用与保活规范

项目当前持有且确认空闲的 H100 应运行项目保活负载；任务使用 GPU 前停止对应
保活，任务结束且 GPU 仍由项目持有时恢复保活。这个约定用于维护已经分配的
worker，不表示可以占用未授权资源，也不能用低利用率替代进程归属判断。

每次调度按以下顺序处理：

1. 用 `nvidia-smi` 同时检查利用率、显存和 compute process。只有没有未知
   compute process 的卡才能视为空闲；仅仅利用率为 0% 不足以判为空闲。
2. 对确认空闲且仍由项目持有的 H100 启动保活。已有项目保活的卡保持不动。
3. 提交任务前只停止目标卡上由项目入口管理的保活，再次检查显存和未知进程，
   然后启动任务。
4. 任务退出后，无论成功还是失败，都检查目标卡；仍空闲且继续持有时重新启动
   保活。
5. 不停止无法确认归属的 PID，不在正在训练、推理或被其他任务占用的卡上叠加
   保活。主机不可达时只报告状态，不把它推断为空闲。

稳定入口为 `scripts/manage-gpu-keepalive.sh`。它把 PID 和日志保存在每台主机
本地的 `/tmp/clothdojo-gpu-keepalive-$UID`，启动前拒绝已有 compute process
的 GPU，停止时也只接受 PID 文件和完整命令同时匹配的进程。共享训练节点需要
显式指定包含 PyTorch 的 Python：

```bash
export CLOTHDOJO_PYTHON=/mnt/dolphinfs-serving/hldy/user/hadoop-aipnlp/huangzilong09/envs/clothdojo-py311-cu128/bin/python
scripts/manage-gpu-keepalive.sh status
scripts/manage-gpu-keepalive.sh start       # 全部可见卡都确认空闲时
scripts/manage-gpu-keepalive.sh stop 0 1 2  # 使用前只释放目标 GPU
scripts/manage-gpu-keepalive.sh start 0 1 2 # 任务结束后恢复
```

不传 GPU 编号时目标是当前主机的全部可见卡；主机只有部分卡空闲时，应显式传入
已确认空闲的 GPU 编号。

不要把 PID、瞬时 GPU 状态或某次实验目录写进长期文档。需要调整负载时使用
`CLOTHDOJO_KEEPALIVE_MATRIX_SIZE` 和 `CLOTHDOJO_KEEPALIVE_DUTY_CYCLE`；默认
值分别为 4096 和 1.0。

## 当前 H100 资源池

以下 4 台由用户于 2026-09-04 指定为当前 H100 worker，SSH 端口均为 8022。
四台均已验证 `id_ed25519_clothdojo_h100` key、共享 DolphinFS 项目与 Python
环境，各有 8 张 H100 80GB，登录 shell 的 memlock 为 `unlimited`。

| SSH 地址 | 端口 | 用户 | 容器 hostname | GPU |
| --- | ---: | --- | --- | --- |
| `10.68.0.138` | 8022 | `hadoop-aipnlp` | `hldy-data-k8s-gpu-h800-node0037.mt` | 8 × H100 80GB |
| `10.68.56.130` | 8022 | `hadoop-aipnlp` | `hldy-data-k8s-gpu-h800-node0390.mt` | 8 × H100 80GB |
| `10.68.59.186` | 8022 | `hadoop-aipnlp` | `hldy-data-k8s-gpu-h800-node0773.mt` | 8 × H100 80GB |
| `10.68.39.34` | 8022 | `hadoop-aipnlp` | `hldy-data-k8s-gpu-h800-node1018.mt` | 8 × H100 80GB |

自动化使用 key：

```bash
ssh -i ~/.ssh/id_ed25519_clothdojo_h100 -p 8022 \
  hadoop-aipnlp@10.68.0.138
```

四台均已授权并验证上述自动化 key。密码登录只作为人工恢复路径：

```bash
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -p 8022 hadoop-aipnlp@10.68.0.138
```

密码不得写入 Git、shell 命令、环境文件或本文档，应从用户的外部凭据记录获取。

## RTX 5090 资源池

RTX 5090 节点适合 Newton 仿真、策略推理和 Isaac RTX 渲染，均已授权同一
自动化 key：

| 地址 | 端口 | 用户 | GPU |
| --- | ---: | --- | --- |
| `33.229.141.59` | 8022 | `hadoop-aipnlp` | 8 × RTX 5090 D v2 24GB |

```bash
ssh -i ~/.ssh/id_ed25519_clothdojo_h100 -p 8022 \
  hadoop-aipnlp@33.229.141.59
```

该节点已验证 `id_ed25519_clothdojo_h100` 密钥。

5090 worker 共享项目路径：

- 项目：`/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/repos/ClothDojo`
- Python：`/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/repos/ClothDojo/.venv/bin/python`
- OpenPI cache：`/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/repos/ClothDojo/data/cache/openpi`（该主机的 `~/.cache/openpi` 已指向该共享目录）

H100 worker 共享路径：

- 存储根：`/mnt/dolphinfs-serving/hldy/user/hadoop-aipnlp/huangzilong09`
- 项目镜像：`/mnt/dolphinfs-serving/hldy/user/hadoop-aipnlp/huangzilong09/ClothDojo`
- Python：`/mnt/dolphinfs-serving/hldy/user/hadoop-aipnlp/huangzilong09/envs/clothdojo-py311-cu128/bin/python`
- OpenPI cache：`/mnt/dolphinfs-serving/hldy/user/hadoop-aipnlp/huangzilong09/openpi_cache`

## 多机通信

当前四台 H100 均已确认 `br0` 使用表中的 SSH 地址。每台有八条状态为 ACTIVE
的 400 Gbit/s 原生 InfiniBand rail：`mlx5_0`、`mlx5_1`、`mlx5_2`、
`mlx5_3`、`mlx5_6`、`mlx5_7`、`mlx5_8`、`mlx5_9`，端口均为 `1`；另有
四条 ACTIVE 的 100 Gbit/s 设备 `mlx5_4`、`mlx5_5`、`mlx5_10`、
`mlx5_11`。四机跨节点 NCCL 尚未复测。

集群环境预设的 `NCCL_IB_GID_INDEX=7` 面向 RoCE；使用上述原生 IB rail 时
应 unset，并显式设置：

```bash
unset NCCL_IB_GID_INDEX
export NCCL_SOCKET_IFNAME=br0
export NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1'
```

W&B 凭据按 host 保存在 `~/.netrc`；权重不得作为 W&B artifact 上传。

替换 worker 上配置 Codex App 时，运行持久化 bootstrap，不要 symlink 整个 `.codex`：

```bash
/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/tools/setup-codex-home.sh
```
