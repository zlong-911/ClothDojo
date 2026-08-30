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
