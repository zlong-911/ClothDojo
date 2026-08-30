# ClothDojo provenance

本目录的 `ur5e.xml` 与 `2f85.xml` 分别取自 Google DeepMind MuJoCo
Menagerie 的 `universal_robots_ur5e` 和 `robotiq_2f85`，固定 revision：

`da76818e269b82289eba39808e2fb91d679d6994`

`ur5e_2f85.xml` 由项目固定的 MuJoCo 3.11.0 使用 `MjSpec.attach` 将标准 2F-85
挂到标准
UR5e 的 `attachment_site` 后导出。组合时保留 UR5e 顶层的 `cone` 与
`impratio` compiler/option 设置，并将 `MjSpec.to_xml()` 缩写的一项 `solref`
向量恢复成上游的显式两项写法，以便 Newton 1.5 正确导入。机器人和夹爪的
body、joint、geom、tendon 与 actuator 定义均来自上述两个标准模型。组合文件与
两份许可证的 SHA-256 为：

- `ur5e_2f85.xml`: `d0399d65466c742fc5a6241a48feb12e034b47c77803c12b0021d560e02870bb`
- `LICENSE.ur5e`: `5ec71ccf66c8d03261448f2441a586765b97f2248c860a1ae19689fb1c45cee6`
- `LICENSE.robotiq_2f85`: `3d5edd18d1e22a17666e9bb84270171ed5ac90b192d24e0177fd171708ea2316`

双臂安装、相机、attach frame 和 home pose 属于 ClothDojo 工作站，固定在
`ur5e_2f85.usda` 及其 payload。既有 tendon 到 position-drive 的适配是项目的
仿真选择，不应视为与源控制器完全等价。

动力学校对在顶层 USD 修正 Newton 导入的旋转关节被动阻尼、限位增益和速度
单位，并恢复四个 Robotiq 弹簧 `mjc:stiffness=0.05`、`mjc:springref=2.62 rad`。
两个 `gripper_base_mount` 没有源显式惯性，使用原 MJCF 经 MuJoCo 3.11.0 按
网格和默认密度推导的质量 `0.150002855 kg`、COM 与完整惯量；不是厂家实测。
来源及求解器限制见 `docs/robot_dynamics.md`，复现校对见
`outputs/robot_dynamics_audit/others_audit.py`。

## 固定腕部相机装配

两侧末端现采用 PickNik 公布的贴合法兰安装方式：7 mm 相机转接件、保留的
Robotiq coupling、官方 RealSense D415 高模。转接件网格、D415 的全部
425,160 个三角形和来源碰撞模型均保留。按用户要求，相机转接件复用 UR5 的
`Material_source_00_black`；原 coupling 和机械臂材质不变。
D415 源 STL 没有贴图；经用户同意，按原 CAD 前盖的 308 面遮罩烘焙三个圆窗
和投射器窗口，补充银色外壳、深色面板外观。UV/颜色为项目近似，不冒充原厂纹理，
不增加镜头几何，不改变安装和标定。外部组件旁的 `d415_front.png` 随部署复制。
内部组件为 `components/wrist_camera.usdc`，唯一工作站入口仍是
`ur5e_2f85.usda`。详见 [完整来源、装配链及边界](provenance/wrist_camera/README.md)。

底座、桌面安装和第三人称相机不变；相机板位于原 coupling
之前，所以夹爪及 attach 随末端轴向外移 7 mm。两侧 home 的最后一个腕关节
相对 payload 各减去 π，令工具绕自身 Z 转 180°；attach 局部定义再绕自身 Z
转 180°，因此 home 下 attach 世界位置和方向不因这两次旋转改变。
关节 home、USD 默认 body 姿态、关节初态与驱动目标同步固定在顶层 USD。
腕部相机随工具旋转，安装标定不变。两个原有 wrist Camera prim
改用真实 D415 nominal optical chain，镜头比例来自 PickNik 的
`horizontal_fov=1.047 rad`、640×480，不再是夹爪内部的虚拟观察点。

新增外观/碰撞形状融合在既有 wrist body，不增加 body、joint 或控制坐标。
**相机和支架的附加载荷尚未计入动力学**：PickNik 未给支架惯性，RealSense
明确警告其示例 COM/tensor 不可靠。本次不猜质量或密度；原 42 个 body
mass/COM/inertia 保持不变，因此不能称末端完整动力学校准已经完成。

复验：`PXR_WORK_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES='' .venv/bin/pytest tests/test_ur5_wrist_camera.py tests/test_robot_dynamics.py -q`。
