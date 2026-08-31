# Piper 轨迹镜像增强

镜像增强作用于 Piper canonical trajectory collection，与 `flatten`、`fold` 等任务
语义无关。输出是可独立搬运和归档的新 collection：每条完整源轨迹按原 ID 逐文件复制，
并生成一条 `<source_id>__mirror` 镜像轨迹；删除或移动源 collection 不影响输出读取。

## 使用入口

```bash
.venv/bin/clothdojo data augment-mirror \
  --source data/collections/shirt_flatten \
  --output data/collections/shirt_flatten_mirrored
```

入口接受带 `trajectories/` 的 canonical collection，以及字段形状受检的 legacy
EpisodeV1 collection，且不会覆盖已有输出。legacy manifest 只在输出副本中规范化为当前
`format_version`，原 manifest hash 写入 provenance；state/action archive 不改写。生成过程
先写入同目录临时目录，全部轨迹通过检查后才发布；失败时不留下正式输出。

## 变换语义

镜像平面固定为 world-local `y=0`。左右臂交换后，Piper 六个关节的符号变换为
`[-1, +1, +1, -1, +1, -1]`；`joint_q`、`joint_qd` 和 `joint_target` 使用同一规则。
夹爪状态和命令按左右语义交换。

布料位置与速度的 `y` 分量取反。刚体位置按普通向量反射，body linear velocity 使用
`[+x, -y, +z]`，angular velocity 作为轴向量使用 `[-x, +y, -z]`。左右 body 分组、
抓取状态和 pin 分组同步交换，但保持每个 link 的语义身份，包括 `link7`/`link8` 指钳
body。body orientation 不直接反射绝对 quaternion；它先转换为相对 Piper 对应 home body
frame 的旋转，对该相对 quaternion `xyzw` 应用 `[-x, +y, -z, +w]`，再乘回目标侧同名
home frame。这样不会把坐标系反射误当成网格或装配结构的反射。Piper `link6` home frame
的局部 `x` 轴对应世界 `y` 轴，因此 `grasp_pin_local` 的局部 `x` 分量取反；该规则与
body orientation 一致，保持 pin 重建后的世界位置镜像对齐。粒子编号和 mesh 顶点顺序不变。

空间反射会反转曲面朝向，但轨迹本身不复制 garment topology。镜像 manifest 因此记录
`surface_winding_reversed=true`；状态审核和后续 LeRobot 渲染据此按反向 winding 计算布料
法线，使 NOCS 顶点语义保持不变，同时保持正确的衣服里外和光照。

该转换是纯代数状态变换，不调用 FK、IK、轨迹规划或物理 solver，也不声称镜像状态满足
Piper articulation FK 或可复现布料接触动力学。状态回放渲染用于检查镜像语义；review
视频按需另行生成，不属于增强命令的默认产物。可以重复 `--trajectory` 精确选择一组镜像
轨迹，其余已有 review 记录不会被移除：

```bash
.venv/bin/clothdojo data render-review \
  --source data/collections/shirt_flatten_mirrored \
  --trajectory 20260807T081925.075214Z__mirror
```

## 产物与标注

canonical 原轨迹的 manifest、state/action archive 和其他轨迹内文件保持原字节与 hash；
legacy 原轨迹仅规范化 manifest，原 manifest hash 与规范化后的 hash 都会保留。镜像
manifest 在 `metadata.augmentation` 中记录源 ID、源 manifest hash、镜像平面、关节符号
以及未使用 kinematics/physics 的事实。collection 根目录的 `augmentation.json` 保存确定性
轨迹配对与源快照 hash。

annotation 文件同时复制给原轨迹和镜像轨迹，包括原有 `review_status` 和完整时间区间。
增强阶段不裁剪 bad 区间；后续 `export-lerobot` 继续按现有 annotation 逻辑导出 good 数据。
