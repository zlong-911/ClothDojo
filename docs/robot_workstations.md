# 固定机器人工作站

正式本体只有 `piper`、`aloha2`、`x5`、`ur5e_2f85`、`fr3_duo`。Panda 不再注册，
保留的历史文件不代表第六个入口。每套本体从 `assets/robots/<robot>/<robot>.usda`
加载；机器人、安装件、home、attach、控制布局、动力学与相机均由该固定入口持有。
任务仍独立选择；`shirt_flatten` 已有场景，`shirt_fold` 暂不可运行。

FR3 Duo 使用已认可的桌面安装、后移位置、镜像夹爪 home 与官方 VMkit；保留头部 ZED Mini
和两个腕部 D405，不再挂额外第三人称支架。X5 保留 Kai0 一体化底板、整台 RoboDojo X5
及其腕部相机，第三人称立柱为已接受的旧项目组件。
FR3 另有与 Piper 同世界位姿及投影的独立虚拟 `cam_low`，不对应新增实体或支架。
ALOHA2 保留既有手臂 home，四指初态使用已有张开端点 0.037 m；USD 默认变换、
关节初态和驱动目标一致，不再沿用源 neutral pose 的近闭合夹爪状态。
UR5e 双腕采用 PickNik 法兰贴装转接件与官方 D415 网格；保留原 UR→Robotiq 连接座，
夹爪及 attach 因 7 mm 转接件沿工具轴外移。home 两侧末个腕关节各减 π，工具
绕自身 Z 转 180°；attach 局部定义也绕自身 Z 转 180°，保留旋转前的 home
抓取世界坐标系。腕部相机随工具旋转，局部标定与第三人称相机不变。
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

几何与绑定材质来自固定 USD；Newton 不读取的 uniform displayColor 仅在无绑定颜色及
贴图时补读，并复用固定 Newton 的颜色空间转换，避免把 USD 默认线性颜色直接当作 sRGB。
不重新配色，不覆盖资产。桌面、桌腿与地面是只读审查背景；FR3 桌面孔板来自已认可的
`tabletop_context_mm.stl`，随外部资产部署，不修改任务桌面碰撞体。
ViewerGL 不等价于 Isaac RTX/MDL 的最终光照与材质效果。
UR5e 腕部 D415 源 STL 无贴图；经用户同意，按原 CAD 遮罩窗口烘焙前盖贴图，
补充银色外壳与深色面板，转接件复用 UR5 原黑色材质。颜色和玻璃呈现为项目近似，
不是厂家原生纹理或完整镜片光学；安装、相机标定与全部原网格不变。

## 材质与纹理边界

五套固定本体的实际绑定材质（包括 instance proxies 与 face subsets）主要使用常量颜色、
roughness 和 metallic。当前唯一使用的图像贴图是 UR5e 双腕共用的 `d415_front.png`：
`UsdUVTexture.outputs:rgb → UsdPreviewSurface.inputs:diffuseColor`，标记 `sRGB`，
使用 `st` UV；不把它用作 normal、roughness、metallic 或 alpha。没有图像的原生参数
材质不等于丢失纹理。`tests/test_ur5_wrist_camera.py` 检查实际绑定、通道、UV 和读取路径；
`tests/test_workstation_appearance.py` 检查颜色空间补读及不覆盖已有材质/贴图。

ViewerGL 是简化预览：支持 albedo 图和标量 PBR，不执行完整 MDL，不读取 normal/MR 图，
也不呈现 FR3 LED 的 emissive 效果。当前五套本体没有绑定 normal/MR 图，无须翻转不存在的
法线绿通道或重排 ORM 通道；以后新增此类材质时不能据当前检查推断 ViewerGL 已支持。
D415 原纹理要求 clamp，固定 ViewerGL sampler 使用 repeat，存在边缘过滤差异；
现有 UV 在 `[0,1]` 内，不是大范围平铺错位。保留源 UV/纹理，不通过改图补偿渲染器限制。

ViewerGL 的近似 gamma、金属提亮和环境反射也不同于 Isaac RTX，因此不能为追求两端
截图同色而反向修改源材质。X5 的 link3/相机使用既有的有效常量外观回退，不宣称完全
恢复了上游异常绑定所指的原材质；其它深色臂节和夹爪仍使用原 RoboDojo 参数。

## 原生相机出图检查

```bash
PXR_WORK_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/robots/check_cameras.py \
  --robot all --assets assets --output outputs/native_camera_review --device cpu --egl-device 1
```

从五套工作站的全部 20 个资产内 Camera prim 出图，不是上述外部腕部近景。
以 home FK 更新相机世界姿态，通过 USD frustum 保留 roll、双轴 aperture、offset
和 clipping，在 640×480 网格采样，不自动扶正或按画幅修改视场；使用同一 ViewerGL
资产材质与只读桌面背景，不运行物理、不加布料、不改相机。每套输出逐相机 PNG、
`overview.jpg` 和 `report.json`；报告包含完整投影、attach 像素位置与画面统计。
非黑图只说明渲染执行成功，不能单独证明视野正确；需检查画面和抓取区域的覆盖。

X5 双腕按用户决定使用完整 RoboDojo **仿真相机预设**，不再使用厂家名义 RGB 光心。
相对各自 `camera` body：位置 `(0,0,0.001)` m、intrinsic XYZ 欧拉角 `(0,-80,-90)`°；
USD focal/aperture 为 `13 / 20.955 / 15.71625`，640×480 视场约77.735°×62.304°，
裁剪范围0.0001–10m。原预设的 home 画面已经朝上且抓取中心居中，不再额外旋转180°。
来源记录在 `assets/robots/x5/provenance/robodojo_wrist_camera.json`，只用于离线生成；
运行时仍读固定 USD，不依赖旧项目。机械实体、安装、home、attach、材质和第三人称相机不变。

仿真预设与物理标定须明确区分：X5 当前虚拟光心在壳体中部附近，不宣称对齐真实 RGB
镜头，也不把其投影或裁剪范围称为厂家测量值。两臂使用相同局部预设，没有图像镜像或
居中裁剪。相机预设切换不代表其支架已通过：X5 后孔距45mm、支架38.4mm的源接口
错配仍待处理。UR5e 的8°支架与6°相机装配也仍待对孔修正；其偏轴 RGB 视角不因本轮改变。
原生检查图也不能冒充当前策略观测的逐像素等价输出。

## 相机与运行边界

- 五套入口已统一资产加载、场景控制布局、retarget 与检查渲染；现有策略 rollout、
  teleop 录制仍使用 Piper 的 14 维动作/记录合同，不代表这些上层流程已支持任意本体切换。
- 五套本体均有四路相机；FR3 保留原生 `cam_high`、左右 wrist，新增与 Piper 同定义的
  独立虚拟 `cam_low`。原生 VMkit prim 不重命名，通过语义属性映射接口名，保留两轴 aperture。
- FR3 的 Newton 策略观测按原始双轴视场生成射线；640×480 是输出采样尺寸，
  不把原生 aperture 强改为 4:3。相机随所属刚体更新，包括头部相机。
- 当前 LeRobot 数据集合同要求四路；FR3 已具备四个独立 Camera，不再缺少 `cam_low`。
  这不代表历史三路轨迹自动补全，也不以复制图像绕过缺路检查。
- Newton 策略渲染在所选相机全部是旧 4:3 aperture 时仍把水平 FOV 用作垂直 FOV。
  X5 恢复 RoboDojo 预设后四路均为4:3，重新走该历史路径；Piper、ALOHA2、UR5e
  的旧路径未改动。这是尚未修正的策略投影边界，不等于上述原生 USD 视场。相机读取层尚未
  传递 USD clipping，不能据原生出图宣称策略渲染的近裁剪或跨渲染器标定已经一致。
- FR3 retarget 已接入同一控制布局，左右夹爪各一个受控坐标，被动连杆保持原约束。
  仍需逐条轨迹通过 retarget 门禁；不能据静态渲染推断任务成功。
- 抓布开闭状态按本体的闭合/张开端点及 articulation 布局读取；释放后的短暂碰撞宽限
  覆盖末个机械臂关节以下的整套腕部机构，包括其固定相机分支。
- 动力学来源与尚未解决的惯性/接触边界见 [robot_dynamics.md](robot_dynamics.md)。
  UR5e 新腕部相机/转接件未计入附加载荷：PickNik 无完整惯性，RealSense 明确警告其示例
  COM/tensor 不适用于建模。保留原刚体惯性，不用密度猜填或把几何接入冒充负载校准。
