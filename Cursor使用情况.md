# Cursor 使用量报告

**统计范围**：Cursor 项目下的 Agent 对话（含 `home-nvidia-isaac-ros-dev`、`home-nvidia-u4-tools-sdk-test`、`trajectory_tracking_mpc` 等）  
**时间范围**：约 2026-01 至 2026-02  

---

## 按业务的使用情况

### 二维码（AprilTag 检测与可视化）

**涉及代码与代码量**：根目录下 `qr_pose_estimator.py`（约 832 行，二维码检测 + 双相机可视化与位姿到 base 的映射）、`pose_comparison_statistics.py`（约 329 行，检测成功率统计与对比）、`gripper_controller.py`（约 235 行）；`robot_tools/basket_place/` 中与 marker 可视化及位姿相关的脚本如 `visualize_two_markers_relative.py`（约 419 行）、`compute_gripper_to_aruco.py`（约 434 行）、`move_right_to_marker.py`（约 463 行）等，合计与二维码/可视化强相关的 Python 约 2.7k+ 行。  
涉及从既有抓取流程中拆出「只做二维码读取 + 双相机可视化」的独立能力、理清位姿到 base 的映射、以及增加检测成功率统计与单相机模式。使用 Cursor 主要是为了在现有实现上快速拆出和扩展，减少通读整条调用链和手写重复逻辑。

---


### 遥操控制（头部、夹爪、关键点录制）

**涉及代码与代码量**：SDK 侧以 `unt_sdk/unt_sdk/robot.py` 为核心（约 3210 行，内含头部/夹爪/末端位姿等接口）；`unt_sdk/examples/arm_hand/` 下如 `h5_arm_cartesian_control.py`（约 648 行）、`move_and_return_pose.py`（约 627 行）、`keyboard_incremental_control.py`（约 605 行）、`hand_control.py`（约 456 行）、`record.py` / `record_circle_trajectory.py` 等；`unt_sdk/examples/body/` 下 `whole_body_contrl.py`（约 1068 行）、`robot_state.py`（约 549 行）、`head_control.py`、`reset_ee_head.py` 等。上述示例与 robot 合计约 5.9k+ 行。流程与策略侧涉及 `robot_tools/controller/`（约 4.2k 行）与 `robot_tools/basket_place/`（约 11.6k 行）中与遥操、关键点、夹爪串联的部分。  
涉及头部控制、夹爪遥控、以及遥控场景下的关键点录制（末端位姿与夹爪状态按序落盘），需要与 SDK 的位姿接口、控制接口保持一致。使用 Cursor 主要是为了对齐 SDK 用法并设计录制数据结构与集成方式，减少翻 API 和手写样板。

---

### 相机与深度图

**涉及代码与代码量**：`unt_sdk/unt_sdk/robot.py` 中相机初始化、话题订阅与图像获取逻辑；`unt_sdk/unt_sdk/cpp_camera_bridge/` 下 C++ 桥接与解码：`cosine_camera_bridge.cpp`（约 718 行）、`jetson_jpeg_decoder.cpp`（约 188 行）、`jetson_jpeg_decoder_module.cpp`、`jetson_jpeg_decoder.h`（约 32 行），合计约 954 行；Python 侧通过 `_camera_bridge` 调用。配置与 QoS 涉及 `unt_sdk/unt_sdk/config.json`（约 48 行）。深度图格式（compressedDepth、16 位）与解码兼容、以及「拿不到深度图」的排查会牵涉上述桥接与 robot 内多段逻辑。使用 Cursor 主要是为了梳理调用链与最小改动点、协助定位失败环节与补充调试手段。

---

### 脚本与外部依赖

**涉及代码与代码量**：`scripts/` 下 Python 脚本合计约 8.6k+ 行，包括 `genie_client_rtc_grasp.py`（约 2307 行）、`genie_client_rtc_place.py`（约 2323 行）、`genie_client.py`（约 1460 行）、`genie2lerobot.py`（约 753 行）、`calibrate_tool_mass.py`（约 638 行）、`pose_data_recorde.py`（约 350 行）、`analyze_h5_structure.py`、各类 monitor 脚本等；根目录下 `start_gripper_node.sh`、`record_bag.sh`、`setup.bash` 等 shell；以及 `unt_sdk/tools/check_ros2_env.sh` 等。涉及相机/服务启动（检查、杀冗余进程再启动）、FoundationPose 与 colcon 构建等外部依赖的报错排查，需结合终端日志与构建流程判断原因与改法。使用 Cursor 主要是为了快速得到可执行的脚本修改或排错方向，减少手写 shell 与盲目试错。

---


### FoundationPose 与 SDK 集成（u4_tools / unt_sdk）

**涉及代码与代码量**：`unt_sdk/unt_sdk/robot.py`（约 3210 行）中的 `start_foundationpose`、`foundationpose_get_pose` 及 `to_link`（base/camera）坐标变换（外参 + 头部姿态）；`unt_sdk/unt_sdk/foundationpose_manager.py`（约 276 行）负责启动与参数；`unt_sdk/unt_sdk/config.json`（约 48 行）中 FoundationPose 与相机外参配置；`unt_sdk/README.md`（约 833 行）中 FoundationPose 结构/原理/用法与 0.4 版本说明；测试入口如 `unt_sdk/tests/test_foundationpose_launch.py`。合计核心修改与文档约 4.4k+ 行。涉及启动链路（点击窗口与随机 mask 行为）、to_link 与外参/头部姿态变换、以及 use_camera、tf_qos_depth、hand_pose_states/whole_body_status 的 debug、opencv 默认解码等说明。依赖 Docker 镜像与 isaac_ros-dev，容器内需参考 isaac_ros-dev/docs。使用 Cursor 主要是为了对照 isaac_ros 文档与现有 Robot/FoundationPose 接口做多文件一致修改，保证与现有策略和配置一致、一次性落地上层接口与文档。

---


### Git 与分支管理（unt_sdk）

**涉及代码与范围**：操作对象为 `unt_sdk` 仓库（含 `unt_sdk/robot.py`、`unt_sdk/foundationpose_manager.py`、`config.json`、`tests/h5_arm_cartesian_control.py`、`tests/reset_ee_head.py`、`tests/test_foundationpose_launch.py` 等）；对话中涉及对 `h5_arm_cartesian_control.py`（约 648 行）step 模式是否做位置到达检查的代码阅读与修改建议。涉及分支情况检查、当前 commit 压缩后 merge 到 v0.3、以及在不改动当前分支的前提下创建新分支并将所有未提交改动应用过去。使用 Cursor 主要是为了在保证「当前分支不变、改动不丢失」的前提下一次性给出正确的 git 操作序列，避免误操作导致代码或分支丢失。

---

### 轨迹跟踪 MPC（ProbMPC / 标准 MPC）

**涉及代码与代码量**：根目录下 `main.py`（约 211 行，argparse、run_mpc/run_prob_mpc 入口）、`main_prob_mpc_original.py`（ProbMPC 主循环、预计算参考轨迹、replan 逻辑、观测/终点噪声）、`main_mpc_original.py`（标准 MPC 主循环）、`reference_trajectory.py`、`h5_trajectory_loader.py`；`prob_mpc/` 下 `template_prob_mpc.py`（约 404 行，`update_mpc_gp_trajectory`、`update_mpc_from_precomputed_ref`）、`template_prob_model.py`、`template_prob_simulator.py`、`gp_trajectory.py`；`mpc/` 下 `template_mpc.py`。合计与轨迹跟踪 MPC 强相关的 Python 约 4k+ 行。  
涉及 ProbMPC 与标准 MPC 的参数语义（num_steps / num_replanning）、每步是否重查 GP、方差是否随步更新；生成轨迹下终点到不了的 bug 修复（t_normalized 用 sim_time 而非固定 trajectory_duration）；预计算整条 GP mean 参考、replan 时仅重观测并沿用同一轨迹剩余段、不重查 GP；replan 步序改为「先规划→执行 N 步→再重规划→再执行 N 步」；路径进度限制在 [0,100%]；观测噪声（当前位置 + 每次 replan 时的终点）统一为单一参数 `--observation-noise-std`，误差仍按真实终点与停止位置计算。使用 Cursor 主要是为了在多文件间一致地改参数与逻辑、修 bug 并统一观测噪声语义，减少反复试跑与手工对齐。

---

## 小结

| 业务                     | 涉及范围与代码量概要                                                                 | 使用 Cursor 的侧重点                     |
|--------------------------|--------------------------------------------------------------------------------------|------------------------------------------|
| 二维码                   | `qr_pose_estimator.py`(832 行)、`pose_comparison_statistics.py`(329)、`gripper_controller.py`(235)、`robot_tools/basket_place` 中可视化/aruco 等，约 2.7k+ 行 | 快速拆出与扩展，减少通读与重复           |
| 抓取流程                 | 策略模块与流程脚本多环节串联（`robot_tools/controller`、`basket_place` 等）          | **流程对齐与多环节一次性落地**           |
| 遥操控制                 | `unt_sdk/robot.py`(3210)、`examples/arm_hand`+`body`(约 5.9k 行)、`robot_tools/controller`(约 4.2k) | 对齐 SDK、设计录制与集成                 |
| 相机与深度图             | `robot.py` 订阅与解码、`cpp_camera_bridge`(约 954 行 C++)、`config.json`(48)         | 梳理链路、兼容与排错                     |
| 脚本与依赖               | `scripts/*.py`(约 8.6k 行)、`start_gripper_node.sh`、`record_bag.sh`、`setup.bash` 等 | 脚本健壮性与排错方向                     |
| Isaac ROS / FP 与迁移   | isaac_ros-dev 内 FP 节点/滤波、迁移脚本、Docker、Git 初始化                          | 定位实现、对照脚本给出命令与迁移流程     |
| FoundationPose 与 SDK   | `robot.py`(3210)、`foundationpose_manager.py`(276)、`config.json`(48)、`README.md`(833)，约 4.4k+ 行 | 对照文档与接口多文件一致修改、一次落地   |
| Docker 内 FP / topic 频率 | `run_dev.sh`、`test_docker_sdk_launch_fp.py`，环境与 DDS 差异                        | 梳理环境与 DDS 差异、定位频率问题        |
| Git 与分支管理           | unt_sdk 仓库（robot、foundationpose_manager、tests/h5_arm_cartesian_control 等）     | 安全 git 操作序列、避免改动丢失          |
| 轨迹跟踪 MPC             | `main.py`(211)、`main_prob_mpc_original.py`、`prob_mpc/template_prob_mpc.py`(404)、`gp_trajectory.py`、`h5_trajectory_loader.py` 等，约 4k+ 行 | 参数/逻辑多文件一致、bug 修复、观测噪声统一 |

**使用 Cursor 的必要性（核心）**：多业务线涉及策略模块、流程脚本、SDK 接口与外部依赖的串联修改；使用 Cursor 主要是为了**对照策略做流程梳理与多环节一次性落地**，保证与现有策略一致，同时减少通读整条调用链、手写重复逻辑与盲目试错，在脚本/环境差异大时快速得到可执行方案与排错方向。

