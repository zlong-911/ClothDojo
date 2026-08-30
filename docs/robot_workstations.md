# 固定机器人工作站

正式本体只有 `piper`、`aloha2`、`x5`、`ur5e_2f85`、`fr3_duo`。Panda 不再注册，
保留的历史文件不代表第六个入口。每套本体从 `assets/robots/<robot>/<robot>.usda`
加载；机器人、安装件、home、attach、控制布局、动力学与相机均由该固定入口持有。
任务仍独立选择；`shirt_flatten` 已有场景，`shirt_fold` 暂不可运行。

FR3 Duo 使用已认可的桌面安装、后移位置、镜像夹爪 home 与官方 VMkit；保留头部 ZED Mini
和两个腕部 D405，不再挂额外第三人称支架。X5 保留 Kai0 一体化底板、整台 RoboDojo X5
及其腕部相机，第三人称立柱为已接受的旧项目组件。
UR5e 双腕采用 PickNik 法兰贴装转接件与官方 D415 网格；保留原 UR→Robotiq 连接座，
夹爪及 attach 因 7 mm 转接件沿工具轴外移，机械臂 home 关节角不变。
这里固定的是仿真装配，不是加工装配认证。

## 部署

Git 保存轻量入口、语义覆盖和来源记录，重型 mesh/texture/payload 在 Git 外管理。
`--source` 必须是包含五套完整资产的已有 **assets 根目录**，不是旧项目根目录，也不是 CAD 源目录。

```bash
.venv/bin/clothdojo assets prepare-workstations --source /path/to/complete/assets --output assets
.venv/bin/clothdojo assets check-workstations --assets assets
```

部署复制固定目录、来源/许可证和共同组件，不重建 CAD、不重新推断安装，不产生 hash/v* 入口。
递归检查 USD reference/payload/sublayer 与纹理引用；缺失文件、越界引用和内容不同的已存在目标
都会报错，同内容重跑不重复复制。`OmniPBR.mdl` 等渲染器内置模块单独列出，不冒充已打包文件。
仅克隆 Git 不能替代 payload 部署；此命令也不下载或重新生成丢失的外部原件。

## 整体、home 和 attach 检查

```bash
.venv/bin/clothdojo sim render-workstations --robot all --assets assets \
  --output outputs/workstation_review --device cpu --egl-device 1
```

`--robot` 可换为单个正式本体。FK 使用指定 Warp device；EGL 是实际显示 GPU 序号，
使用前检查占用。渲染器为 Newton ViewerGL，不是 AI 图片或软件示意图。
每台输出整体、左右夹爪近景以及 JSON，标出 attach 的红 X/绿 Y/蓝 Z，记录完整 home、
local/world attach 和 USD 默认姿态与 Newton FK 的位置误差。
查看腕部相机安装时，同一命令追加 `--views left_wrist right_wrist`；从该本体的
原生相机坐标系选择镜头正面的近景，不改变相机或机器人的装配。

几何与绑定材质来自固定 USD；Newton 不读取的 uniform displayColor 仅在无绑定颜色时补读。
不重新配色，不覆盖资产。桌面、桌腿与地面是只读审查背景；FR3 桌面孔板来自已认可的
`tabletop_context_mm.stl`，随外部资产部署，不修改任务桌面碰撞体。
ViewerGL 不等价于 Isaac RTX/MDL 的最终光照与材质效果。

## 相机与运行边界

- 五套入口已统一资产加载、场景控制布局、retarget 与检查渲染；现有策略 rollout、
  teleop 录制仍使用 Piper 的 14 维动作/记录合同，不代表这些上层流程已支持任意本体切换。
- Piper、ALOHA2、X5、UR5e 使用四路相机；FR3 使用原生三路 `cam_high`、左右 wrist。
  USD 相机 prim 不复制或重命名，通过语义属性映射接口名，保留两轴 aperture。
- FR3 的 Newton 三路策略观测按原始双轴视场生成射线；640×480 是输出采样尺寸，
  不把原生 aperture 强改为 4:3。相机随所属刚体更新，包括头部相机。
- 当前 LeRobot 数据集合同要求四路，因此 FR3 在创建数据集前明确报缺少 `cam_low`。
  新增三路数据集格式需要独立决定，不能靠复制图像绕过。
- 旧四款 Newton 策略相机保留历史像素路径：水平 FOV 被用作垂直 FOV，与 USD 标定不一致。
  本次未改动，以免混入已有训练/评测观测变化；这不是已完成的跨渲染器标定一致性。
- FR3 retarget 已接入同一控制布局，左右夹爪各一个受控坐标，被动连杆保持原约束。
  仍需逐条轨迹通过 retarget 门禁；不能据静态渲染推断任务成功。
- 抓布开闭状态按本体的闭合/张开端点及 articulation 布局读取；释放后的短暂碰撞宽限
  覆盖末个机械臂关节以下的整套腕部机构，包括其固定相机分支。
- 动力学来源与尚未解决的惯性/接触边界见 [robot_dynamics.md](robot_dynamics.md)。
  UR5e 新腕部相机/转接件未计入附加载荷：PickNik 无完整惯性，RealSense 明确警告其示例
  COM/tensor 不适用于建模。保留原刚体惯性，不用密度猜填或把几何接入冒充负载校准。
