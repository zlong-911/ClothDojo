# 计算节点

当前登记了两个 H100 资源池：一组四机互联 worker（4 × 8 GPU），以及三台
独立 8-GPU worker；另有两台 RTX 5090 worker。每台 GPU 数量和型号以最近一次
连接检查为准，实际可用数量仍需要在每次实验前确认。

下列 H100 节点适合 Newton、训练和策略推理，不用于 Isaac RTX 离线导出。H100 没有 RT Core，不满足 Isaac Sim 的渲染要求；离线导出应调度到 Vulkan 可用的 RTX GPU。

## H100 四机互联资源池

Codex session `01a037bf-548b-7031-b658-2051a3f83fa3` 当前持有或可访问这组
4 × 8 H100 worker。该 session 已完成 32-rank NCCL all-reduce 校验；session
关联不代表 GPU 正在计算，调度前仍须逐机检查 `nvidia-smi` 和现有进程。

| SSH alias | 地址 | 端口 | 用户 | GPU |
| --- | --- | ---: | --- | --- |
| `clothdojo-h100-a` | `10.68.21.26` | 8022 | `hadoop-aipnlp` | 8 × H100 80GB |
| `clothdojo-h100-b` | `10.68.22.54` | 8022 | `hadoop-aipnlp` | 8 × H100 80GB |
| `clothdojo-h100-c` | `10.68.38.178` | 8022 | `hadoop-aipnlp` | 8 × H100 80GB |
| `clothdojo-h100-d` | `10.68.49.26` | 8022 | `hadoop-aipnlp` | 8 × H100 80GB |

自动化使用 key：

```bash
ssh -i ~/.ssh/id_ed25519_clothdojo_h100 -p 8022 \
  hadoop-aipnlp@10.68.21.26
```

四台 worker 均已授权同一自动化 key。密码登录只作为人工恢复路径：

```bash
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -p 8022 hadoop-aipnlp@10.68.21.26
```

密码不得写入 Git、shell 命令、环境文件或本文档，应从用户的外部凭据记录获取。

## H100 独立单机资源池

这三台机器各自提供 8 × H100 80GB，SSH 端口均为 `8416`：

| 地址 | 端口 | 用户 | GPU | 自动化访问状态 |
| --- | ---: | --- | --- | --- |
| `10.238.4.136` | 8416 | `hadoop-aipnlp` | 8 × H100 80GB | key 已验证 |
| `10.245.92.191` | 8416 | `hadoop-aipnlp` | 8 × H100 80GB | key 已验证 |
| `10.245.115.39` | 8416 | `hadoop-aipnlp` | 8 × H100 80GB | key 已验证 |

```bash
ssh -i ~/.ssh/id_ed25519_clothdojo_h100 -p 8416 \
  hadoop-aipnlp@10.238.4.136
```

独立单机不应被当作上述四机 NCCL 资源池的额外节点；是否具备相同的 IB 拓扑和
跨机通信能力需要另行验证。

## RTX 5090 资源池

RTX 5090 节点适合 Newton 仿真、策略推理和 Isaac RTX 渲染，均已授权同一
自动化 key：

| SSH alias | 地址 | 端口 | 用户 | GPU |
| --- | --- | ---: | --- | --- |
| `clothdojo-5090-a` | `33.229.142.91` | 8022 | `hadoop-aipnlp` | 8 × RTX 5090 D v2 24GB |
| `clothdojo-5090-b` | `33.229.219.50` | 8022 | `hadoop-aipnlp` | 8 × RTX 5090 D v2 24GB |

```bash
ssh clothdojo-5090-a
ssh clothdojo-5090-b
```

两台 5090 worker 共享项目路径：

- 项目：`/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/repos/ClothDojo`
- Python：`/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/repos/ClothDojo/.venv/bin/python`
- OpenPI cache：`/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/repos/ClothDojo/data/cache/openpi`（两台主机的 `~/.cache/openpi` 已指向该共享目录）

共享路径：

- 存储根：`/mnt/dolphinfs-serving/hldy/user/hadoop-aipnlp/huangzilong09`
- 项目镜像：`/mnt/dolphinfs-serving/hldy/user/hadoop-aipnlp/huangzilong09/ClothDojo`
- Python：`/mnt/dolphinfs-serving/hldy/user/hadoop-aipnlp/huangzilong09/envs/clothdojo-py311-cu128/bin/python`
- OpenPI cache：`/mnt/dolphinfs-serving/hldy/user/hadoop-aipnlp/huangzilong09/openpi_cache`

## 多机通信

当前四台 worker 使用 `br0` 上表中的容器 IP 做 Ray、torch distributed 和
NCCL 的带外通信。每台有八条 400 Gbit/s 原生 InfiniBand rail：
`mlx5_0`、`mlx5_1`、`mlx5_2`、`mlx5_3`、`mlx5_6`、`mlx5_7`、
`mlx5_8`、`mlx5_9`，端口均为 `1`。登录 shell 的 memlock 必须保持
`unlimited`。

本次集群环境预设的 `NCCL_IB_GID_INDEX=7` 面向 RoCE；使用上述原生 IB
rail 时应 unset，并显式设置：

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
