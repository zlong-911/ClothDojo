# 机械臂安装接口验证

安装位姿不通过目测或手调坐标确定。ClothDojo 将安装接口表示为同一接触面上的孔轴、
接触面外法向和可选的正前方基准，使用刚体最小二乘配准求解机械臂到底座的变换，并输出
逐孔残差、对称解数量和紧固件余量。

## 证据边界

输入必须声明来源：

- `drawing`：带标注尺寸的机械图；
- `brep`：STEP 等 B-Rep 中的解析平面和圆柱面；
- `mesh`：STL/OBJ 截面拟合，只用于筛选和交叉检查；
- `physical`：实物测量。

STL 不记录单位，而且圆柱已经三角化。`mount_features` 要求显式传入单位比例、截面和
直径范围，输出始终标为 `mesh`；它拒绝 STEP/SLDPRT，避免把网格近似冒充原生 CAD
测量。制造前应以厂商机械图、STEP B-Rep 或实物测量替换 mesh-derived 输入。

## 使用入口

从 X5 底座 STL 的指定截面拟合四个圆形孔位：

```bash
.venv/bin/python -m clothdojo.assets.mount_features \
  assets/robots/x5/meshes/base_link.STL \
  --plane-origin 0 0 0.005 \
  --plane-normal 0 0 1 \
  --unit-scale-mm 1000 \
  --diameter-mm 2 3 \
  --max-radial-error-mm 0.08 \
  --output outputs/mount-alignment/x5-features.json
```

根据显式安装基准求解变换和残差：

```bash
.venv/bin/python -m clothdojo.assets.mount_alignment \
  configs/hardware/kai0_x5_mount.json \
  --require-compatible \
  --output outputs/mount-alignment/kai0-x5-report.json
```

`--require-compatible` 只检查孔型残差是否满足配置的筛选门槛。只有输入同时提供底板孔径、
紧固件直径、两侧测量不确定度和装配余量后，才可使用 `--require-mechanical-fit` 作为机械
装配门禁。

## 验收关系

孔位对应关系通过穷举小型螺栓阵列的排列确定；每个排列都求解满足接触面法向相反的最佳
刚体变换。报告使用最大孔心残差检查孔型，并按以下保守关系计算剩余径向余量：

```text
(底板孔径 - 紧固件直径) / 2
- 最大孔心残差
- 两侧测量不确定度之和
- 装配余量
```

结果非负才是 `mechanical_fit: pass`。方形四孔通常仍有多个等价旋转解；只有两侧都提供
`datum_direction`（例如机械臂正前方、出线口或 `base_link` 的前向）后，位姿才会从
`ambiguous` 变为 `resolved`。
基准方向作为软观测参与拟合，不保证方向误差为零；具体偏差见报告的
`datum_direction_error_degrees`。

## Kai0 当前结论

- Kai0 HangCloth 底板与 X5：最大孔心残差约 `0.679 mm`，通过 `1.0 mm` 孔型筛选；
- 同一底板与 Piper：最大孔心残差约 `17.677 mm`，孔型不兼容；
- X5 方孔阵仍有四个等价朝向，且紧固件规格和测量不确定度尚未补齐，因此当前结果不是
  加工批准。

左右臂输入分别为 `configs/hardware/kai0_x5_mount.json`、
`configs/hardware/kai0_x5_right_mount.json`；Piper 排除对照为
`configs/hardware/kai0_piper_rejection.json`。
