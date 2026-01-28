"""
轨迹跟踪 MPC 主程序
实现基于 do-mpc 的参考轨迹跟踪控制

核心思想：
- 使用 MPC 在每个控制周期预测未来并强制状态贴近参考轨迹
- 参考轨迹是时间参数化的连续函数（如螺旋轨迹）
- 支持目标更新和连续重规划
- 轨迹形状在整个预测过程中被保持
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import do_mpc

# 导入自定义模块
from mpc.template_model import template_model
from mpc.template_mpc import template_mpc, update_mpc_reference_trajectory
from mpc.template_simulator import template_simulator
from reference_trajectory import ReferenceTrajectoryGenerator


def main(num_replanning: int = 20, trajectory_type: str = 'spiral', dimension: int = 3, estimate_velocity: bool = True, use_real_trajectory: bool = False, dataset_path: str = None):
    """
    主函数：轨迹跟踪 MPC 仿真（仅位置控制）
    
    Args:
        num_replanning: 重新规划次数（默认20次）
        trajectory_type: 轨迹类型 ('spiral' 或 'rollercoaster')
        dimension: 状态空间维度（默认3，表示3D空间）
        estimate_velocity: 已废弃（保留以兼容接口，但不再使用）
        use_real_trajectory: 是否使用真实轨迹（从H5数据集加载）
        dataset_path: 数据集路径（当使用真实轨迹时必需）
    
    注意：系统现在只观测和控制位置，速度和加速度都不可观测或控制
    """
    print("=" * 60)
    print("轨迹跟踪 MPC 仿真（一阶系统）")
    print("=" * 60)
    print(f"重新规划次数: {num_replanning}")
    print(f"状态空间维度: {dimension}")
    print(f"控制模式: 一阶系统（状态=位置，控制输入=期望速度，p_dot=u）")

    # ========== 1. 创建系统模型 ==========
    print("\n[1/6] 创建系统模型...")
    model = template_model('SX', dimension=dimension)
    print(f"   状态维度: {model.n_x}")
    print(f"   输入维度: {model.n_u}")

    # ========== 2. 生成或加载参考轨迹 ==========
    print("\n[2/6] 生成或加载参考轨迹...")
    
    if use_real_trajectory:
        # 使用真实轨迹
        if dataset_path is None:
            raise ValueError("使用真实轨迹时必须提供 dataset_path 参数")
        
        # 尝试导入 h5_trajectory_loader
        try:
            from h5_trajectory_loader import H5TrajectoryLoader
            HAS_H5PY = True
        except ImportError:
            HAS_H5PY = False
            raise ImportError(
                "使用真实轨迹需要安装 h5py 模块。请运行: pip install h5py 或 conda install h5py"
            )
        
        print(f"   从数据集加载真实轨迹: {dataset_path}")
        print(f"   指定维度: {dimension}")
        loader = H5TrajectoryLoader(dataset_path, max_trajectories=1, dimension=dimension)
        all_trajectories = loader.load_all_trajectories()
        
        if len(all_trajectories) == 0:
            raise ValueError(f"未能从数据集加载任何轨迹: {dataset_path}")
        
        # 选择第一条轨迹
        selected_traj, selected_ts = all_trajectories[0]
        
        if len(selected_traj) == 0:
            raise ValueError("加载的轨迹为空")
        
        # 检查真实轨迹的维度
        real_dimension = selected_traj.shape[1]
        if dimension != real_dimension:
            print(f"   警告: 真实轨迹维度 ({real_dimension}) 与指定维度 ({dimension}) 不匹配")
            print(f"   将使用加载后的轨迹维度: {real_dimension}")
            dimension = real_dimension
            # 需要重新创建模型
            print("\n[1/6] 重新创建系统模型（使用正确的维度）...")
            model = template_model('SX', dimension=dimension)
            print(f"   状态维度: {model.n_x}")
            print(f"   输入维度: {model.n_u}")
        
        # 使用真实轨迹
        trajectory = selected_traj
        time_stamps = selected_ts
        trajectory_duration = time_stamps[-1] - time_stamps[0] if len(time_stamps) > 1 else len(trajectory) * 0.1
        
        print(f"   参考轨迹长度: {len(trajectory)} 点")
        print(f"   轨迹持续时间: {trajectory_duration:.2f} 秒")
        
        # 从真实轨迹获取起点和终点
        start_mean = trajectory[0].copy()
        end_mean = trajectory[-1].copy()
        
        # 为了兼容后续代码，定义一些默认值（虽然真实轨迹不需要这些）
        start_std = 0.05
        start_cov = start_std**2
        end_std = 0.05
        end_cov = end_std**2
        num_trajectories = 1  # 真实轨迹只有1条
        num_points_per_traj = len(trajectory)  # 使用真实轨迹的点数
        
        # 创建轨迹生成器，传入真实轨迹
        traj_gen = ReferenceTrajectoryGenerator(trajectory_type=trajectory_type, real_trajectory=(trajectory, time_stamps))
        all_trajectories = [(trajectory, time_stamps)]  # 为了兼容后续代码
    else:
        # 使用生成的轨迹
        traj_gen = ReferenceTrajectoryGenerator(trajectory_type=trajectory_type)
        
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        # 定义起点和终点的多维高斯分布
        start_mean = np.zeros(dimension)
        start_mean[:3] = [0.0, 0.0, 0.0]  # 前3维用于3D空间
        start_std = 0.05  # 标准差 0.05m
        start_cov = start_std**2  # 协方差 = 标准差^2 = 0.0025
        end_mean = np.zeros(dimension)
        end_mean[:3] = [1.0, 1.0, 1.0]  # 前3维用于3D空间
        end_std = 0.05  # 标准差 0.05m
        end_cov = end_std**2  # 协方差 = 标准差^2 = 0.0025
        
        # 生成多条参考轨迹（20条）
        num_trajectories = 20
        trajectory_duration = 10.0  # 秒
        num_points_per_traj = 200  # 增加点数确保连续性
        
        print(f"   生成 {num_trajectories} 条参考轨迹...")
        print(f"   起点分布: 均值={start_mean}, 标准差={start_std}m, 协方差={start_cov}")
        print(f"   终点分布: 均值={end_mean}, 标准差={end_std}m, 协方差={end_cov}")
        
        # 根据轨迹类型生成轨迹
        if trajectory_type == 'rollercoaster':
            print(f"   轨迹类型: 过山车轨迹（平稳曲线，方差小）")
            all_trajectories = traj_gen.generate_multiple_trajectories(
                num_trajectories=num_trajectories,
                start_mean=start_mean,
                start_cov=start_cov,
                end_mean=end_mean,
                end_cov=end_cov,
                duration=trajectory_duration,
                num_points=num_points_per_traj,
                trajectory_type='rollercoaster',
                circle_radius=0.3,
                circle_plane='vertical',
                circle_ratio=0.6,
                dimension=dimension
            )
        else:  # 'spiral'
            print(f"   轨迹类型: 螺旋轨迹")
            all_trajectories = traj_gen.generate_multiple_trajectories(
                num_trajectories=num_trajectories,
                start_mean=start_mean,
                start_cov=start_cov,
                end_mean=end_mean,
                end_cov=end_cov,
                duration=trajectory_duration,
                num_points=num_points_per_traj,
                trajectory_type='spiral',
                base_spiral_radius=0.2,
                base_num_turns=2.0,
                noise_scale=0.3,
                convergence_start=0.8,  # 在80%时间后开始收敛
                convergence_length=0.2,  # 最后20%的时间用于收敛
                dimension=dimension
            )
        
        # 选择第一条轨迹用于 MPC 跟踪（可以随机选择或选择特定的）
        selected_idx = 0
        trajectory, time_stamps = all_trajectories[selected_idx]
        print(f"   选择轨迹 {selected_idx+1}/{num_trajectories} 用于 MPC 跟踪")
        print(f"   参考轨迹长度: {len(trajectory)} 点")
        print(f"   轨迹持续时间: {trajectory_duration} 秒")

    # ========== 3. 创建 MPC 控制器 ==========
    print("\n[3/6] 创建 MPC 控制器...")
    mpc = template_mpc(model, traj_gen, silence_solver=True)
    print(f"   预测时域: {mpc.settings.n_horizon} 步")
    print(f"   采样时间: {mpc.settings.t_step} 秒")

    # ========== 4. 创建仿真器 ==========
    print("\n[4/6] 创建仿真器...")
    # 创建可更新的 TVP 数据字典
    simulator_tvp_data = {
        'traj_gen': traj_gen,
        'trajectory': trajectory,
        'time_stamps': time_stamps
    }
    simulator = template_simulator(model, tvp_data_dict=simulator_tvp_data)

    # ========== 5. 初始化状态 ==========
    print("\n[5/6] 初始化状态...")
    # 初始状态：使用起点分布的均值（与轨迹生成一致）
    # 注意：虽然轨迹起点是从分布中采样的，但为了确保初始状态在数据集起点附近，
    # 我们使用起点分布的均值作为初始状态
    # 构建初始状态向量：只有位置
    x0_list = []
    for i in range(dimension):
        if i < len(start_mean):
            x0_list.append(start_mean[i])  # 使用起点分布的均值
        else:
            x0_list.append(0.0)
    x0 = np.array(x0_list).reshape(-1, 1)
    
    # 验证初始状态与轨迹起点的距离
    start_point = trajectory[0]
    start_dist = np.linalg.norm(x0.flatten()[:min(dimension, len(start_point))] - start_point[:min(dimension, len(start_point))])
    print(f"   初始状态: {x0.flatten()}")
    print(f"   轨迹起点: {start_point}")
    print(f"   初始状态与轨迹起点距离: {start_dist:.4f}m")

    # 设置初始状态
    mpc.x0 = x0
    simulator.x0 = x0

    # 设置初始猜测
    mpc.set_initial_guess()

    # ========== 6. 仿真循环 ==========
    print("\n[6/6] 开始仿真...")

    # 仿真参数
    # 确保仿真时间足够长，能够到达参考轨迹的终点
    # 使用轨迹的持续时间，确保能够完整跟踪参考轨迹
    sim_time = trajectory_duration  # 仿真总时间等于轨迹持续时间
    n_steps = int(sim_time / mpc.settings.t_step)

    # 存储数据
    actual_trajectory = []
    reference_trajectory_actual = []
    control_inputs = []
    time_history = []

    # 当前状态
    x_current = x0.copy()
    current_time = 0.0
    
    # 初始化前一个控制输入（用于计算控制输入变化率）
    u_prev = np.zeros((dimension, 1))  # 初始控制输入为0
    
    # 先记录初始状态（用于可视化起点）
    actual_trajectory.append(x0.flatten())
    reference_trajectory_actual.append(trajectory[0])
    control_inputs.append(np.zeros(dimension))
    time_history.append(0.0)

    # 计算重新规划的时间点（均匀分布在仿真过程中）
    # 避免在第一步和最后几步更新，确保有足够的仿真时间到达终点
    if num_replanning > 0:
        # 在 [1, n_steps-5] 范围内均匀分布更新点
        # 最后5步不重新规划，确保有足够时间到达终点
        last_replanning_step = max(1, n_steps - 5)
        if last_replanning_step > 1:
            replanning_steps = np.linspace(1,
                                           last_replanning_step,
                                           num_replanning,
                                           dtype=int)
            replanning_steps = np.unique(replanning_steps)  # 确保唯一性
            # 确保不超过最后一步
            replanning_steps = replanning_steps[replanning_steps < n_steps - 1]
        else:
            replanning_steps = []
        num_replanning = len(replanning_steps)  # 实际更新次数
        if num_replanning > 0:
            print(f"   重新规划时间点: {replanning_steps}")
            print(f"   实际重新规划次数: {num_replanning}")
        else:
            print(f"   警告: 无法进行重新规划（仿真时间太短）")
            replanning_steps = []
    else:
        replanning_steps = []

    # 重新规划计数器
    replanning_count = 0

    print(f"   仿真步数: {n_steps}")
    print(f"   开始仿真循环...\n")

    for k in range(n_steps):
        # ========== 目标重新规划 ==========
        # 在预定的时间点更新目标，测试重规划能力
        if num_replanning > 0 and k in replanning_steps:
            replanning_count += 1
            print(
                f"   [步骤 {k}, 时间={current_time:.2f}s] 重新规划 #{replanning_count}/{num_replanning}"
            )

            # 从参考轨迹终点的分布中采样新目标
            # 参考轨迹的终点是从 end_mean 和 end_cov 的高斯分布中采样的
            # 为了保持一致性，新目标也应该从相同的分布中采样
            new_target = np.random.multivariate_normal(end_mean,
                                                       np.eye(dimension) * end_cov)

            print(
                f"     从终点分布采样新目标 (均值={end_mean}, 标准差={np.sqrt(end_cov):.3f}m)"
            )
            print(f"     新目标: {new_target}")

            # 从 all_trajectories 中选择一条轨迹，提取其螺旋参数
            # 然后从当前位置到新目标重新生成轨迹，使用相似的螺旋参数
            # 提取所有维度的位置（现在状态就是位置）
            current_position = x_current.flatten()

            # 方法：选择终点最接近新目标的轨迹（用于提取螺旋参数）
            best_end_dist = float('inf')
            best_idx = 0

            for idx, (traj, _) in enumerate(all_trajectories):
                traj_end = traj[-1]
                dist_end = np.linalg.norm(traj_end - new_target)
                if dist_end < best_end_dist:
                    best_end_dist = dist_end
                    best_idx = idx

            # 从选择的轨迹中提取螺旋参数（通过分析轨迹形状）
            selected_traj, _ = all_trajectories[best_idx]

            # 估算螺旋参数：通过分析轨迹的螺旋特征
            # 计算轨迹的"螺旋度"（偏离直线的程度）
            traj_direction = selected_traj[-1] - selected_traj[0]
            traj_length = np.linalg.norm(traj_direction)
            traj_path_length = np.sum(
                np.linalg.norm(np.diff(selected_traj, axis=0), axis=1))
            spiral_ratio = traj_path_length / (traj_length + 1e-6)  # 路径长度/直线距离

            # 根据螺旋比估算螺旋参数
            # 螺旋比越大，说明螺旋越明显
            estimated_radius = 0.2 * spiral_ratio  # 估算半径
            estimated_turns = 2.0 * spiral_ratio  # 估算圈数

            # 添加一些随机性，但保持合理的范围
            spiral_radius = max(
                0.1, estimated_radius * (1 + 0.2 * np.random.randn()))
            num_turns = max(0.5,
                            estimated_turns * (1 + 0.2 * np.random.randn()))

            # 关键修复：从当前位置到新目标重新生成轨迹
            # 计算剩余仿真时间，确保新轨迹的终点时间正好是仿真结束时间
            remaining_sim_time = sim_time - current_time
            # 使用剩余时间作为新轨迹的持续时间，确保能够到达终点
            new_trajectory_duration = max(0.1, remaining_sim_time)  # 至少0.1秒
            
            # 根据剩余时间调整点数，保持相同的采样密度
            new_num_points = max(10, int(num_points_per_traj * (new_trajectory_duration / trajectory_duration)))
            
            trajectory, time_stamps_new = traj_gen.generate_spiral_trajectory(
                start=current_position,  # 从当前位置开始
                target=new_target,  # 到新目标
                duration=new_trajectory_duration,  # 使用剩余时间
                num_points=new_num_points,  # 根据剩余时间调整点数
                spiral_radius=spiral_radius,
                num_turns=num_turns,
                dimension=dimension)

            # 关键修复：时间戳从当前时间开始，确保连续性
            # 新轨迹的终点时间应该是 current_time + new_trajectory_duration = sim_time
            time_stamps = time_stamps_new + current_time

            print(f"     当前位置: {current_position}")
            print(f"     新目标: {new_target}")
            print(
                f"     从 {len(all_trajectories)} 条轨迹中选择轨迹 {best_idx+1} 提取螺旋参数")
            print(
                f"     提取的螺旋参数: radius={spiral_radius:.3f}, turns={num_turns:.3f}"
            )
            print(
                f"     新轨迹起点距离: {np.linalg.norm(trajectory[0] - current_position):.4f}m (应该≈0)"
            )
            new_end_dist = np.linalg.norm(trajectory[-1] - new_target)
            print(f"     新轨迹终点距离: {new_end_dist:.4f}m (应该≈0)")
            print(
                f"     剩余仿真时间: {remaining_sim_time:.2f}s"
            )
            print(
                f"     新轨迹时间范围: [{time_stamps[0]:.2f}, {time_stamps[-1]:.2f}]s (终点时间应该={sim_time:.2f}s)"
            )

            # 更新仿真器的 TVP 数据（通过字典引用自动更新）
            simulator_tvp_data['trajectory'] = trajectory
            simulator_tvp_data['time_stamps'] = time_stamps
            
            # 关键修复：重新规划时，根据新轨迹的初始方向重置前一个控制输入
            # 这样可以避免控制输入变化率惩罚导致的不匹配和锐利转折
            # 计算新轨迹初始方向的速度（用于设置前一个控制输入）
            if len(trajectory) > 1 and len(time_stamps) > 1:
                # 计算从当前位置到轨迹下一个点的时间差
                dt_traj = time_stamps[1] - time_stamps[0]
                if dt_traj > 1e-6:
                    # 计算从当前位置到轨迹下一个点的方向速度
                    next_point = trajectory[1]
                    direction = next_point - current_position
                    # 计算方向速度（位置差 / 时间差）
                    u_prev = (direction / dt_traj).reshape(-1, 1)
                    # 限制速度大小，避免过大
                    max_u_prev = 3.0  # 最大速度限制
                    u_prev_norm = np.linalg.norm(u_prev)
                    if u_prev_norm > max_u_prev:
                        u_prev = u_prev / u_prev_norm * max_u_prev
                else:
                    u_prev = np.zeros((dimension, 1))
            else:
                u_prev = np.zeros((dimension, 1))
            
            print(f"     重新规划时重置前一个控制输入: {u_prev.flatten()}")
            
            # 重新规划时，更新MPC的初始猜测，帮助优化器更快收敛
            # 使用新轨迹的前几个点作为初始猜测
            mpc.set_initial_guess()

        # ========== 更新 MPC 参考轨迹 ==========
        # 关键步骤：在每个控制周期更新预测时域内的参考轨迹
        remaining_time = sim_time - current_time
        remaining_steps = int(round(remaining_time / mpc.settings.t_step))
        terminal_index = None
        if remaining_steps <= mpc.settings.n_horizon:
            terminal_index = max(0, remaining_steps - 1)
        update_mpc_reference_trajectory(mpc, traj_gen, current_time,
                                        trajectory, time_stamps,
                                        terminal_index=terminal_index)
        
        # ========== 设置前一个控制输入参数 ==========
        # 用于计算控制输入变化率，提高平滑性
        # 更新MPC和仿真器内部的参数数据字典（参数函数会从中读取）
        for i, dim_name in enumerate(mpc.model.dim_names):
            if i < dimension:
                mpc._p_data[f'u_{dim_name}_prev'] = float(u_prev[i, 0])
                # 同时更新仿真器的参数（如果存在）
                if hasattr(simulator, '_p_data'):
                    simulator._p_data[f'u_{dim_name}_prev'] = float(u_prev[i, 0])
        
        # ========== MPC 求解 ==========
        # MPC 基于当前状态和参考轨迹计算最优控制输入
        u0 = mpc.make_step(x_current)
        
        # 更新前一个控制输入（用于下一步）
        u_prev = u0.copy()

        # ========== 仿真器步进 ==========
        # 应用控制输入，仿真系统响应
        y_next = simulator.make_step(u0)

        # ========== 状态更新 ==========
        # 真实场景：只能观测到位置，状态就是位置
        # 从仿真器获取状态（现在只有位置）
        x_current = y_next
        
        # ========== 数据记录 ==========
        actual_trajectory.append(x_current.flatten())  # 位置（所有维度，现在状态就是位置）

        # 获取参考点（确保在轨迹范围内）
        # 如果超出轨迹范围，使用轨迹的最后一个点（目标终点）
        if current_time >= time_stamps[-1]:
            # 超出或到达轨迹终点，使用轨迹的最后一个点（目标终点）
            ref_point = trajectory[-1]
        else:
            # 正常范围内，使用插值
            ref_point = traj_gen.get_reference_at_time(trajectory, time_stamps,
                                                       current_time)
        
        # 在接近终点时（最后1秒），强制使用目标终点，确保终点对齐
        if current_time >= time_stamps[-1] - 1.0:
            ref_point = trajectory[-1]

        reference_trajectory_actual.append(ref_point)
        control_inputs.append(u0.flatten())
        time_history.append(current_time)

        # 时间更新
        current_time += mpc.settings.t_step

        # 进度显示
        if (k + 1) % 20 == 0:
            pos_error = np.linalg.norm(x_current.flatten() - ref_point[:dimension])
            print(f"   步骤 {k+1}/{n_steps}: 时间={current_time:.2f}s, "
                  f"位置误差={pos_error:.3f}m")

    print("\n仿真完成！")

    # 检查终点位置
    actual_end = np.array(actual_trajectory[-1])
    # 参考轨迹的目标终点（这是真正的目标）
    reference_target_end = trajectory[-1]
    # 参考轨迹跟踪的最后一个点（应该等于目标终点）
    ref_end = np.array(reference_trajectory_actual[-1])

    end_error = np.linalg.norm(actual_end - ref_end)
    print(f"\n终点位置检查:")
    print(f"   实际轨迹终点: {actual_end}")
    print(f"   参考轨迹目标终点: {reference_target_end}")
    print(f"   参考轨迹跟踪终点: {ref_end}")
    print(f"   终点误差: {end_error:.4f}m")

    # 检查 MPC 预测终点（用于验证终端约束是否在优化层生效）
    try:
        pred_p_x = float(np.array(mpc.data.prediction(('_x', 'p_x'), t_ind=-1)).reshape(-1)[-1])
        pred_p_y = float(np.array(mpc.data.prediction(('_x', 'p_y'), t_ind=-1)).reshape(-1)[-1])
        pred_p_z = float(np.array(mpc.data.prediction(('_x', 'p_z'), t_ind=-1)).reshape(-1)[-1])
        print(f"   MPC 预测终点: [{pred_p_x:.6f} {pred_p_y:.6f} {pred_p_z:.6f}]")
    except Exception as exc:
        print(f"   MPC 预测终点获取失败: {exc}")

    # ========== 7. 可视化 ==========
    print("\n生成可视化...")

    visualize_results(np.array(actual_trajectory),
                      np.array(reference_trajectory_actual),
                      np.array(control_inputs),
                      np.array(time_history),
                      trajectory,
                      all_trajectories,
                      reference_target_end=reference_target_end,
                      dimension=dimension)

    print("\n" + "=" * 60)
    print("轨迹跟踪 MPC 仿真完成！")
    print("=" * 60)


def visualize_results(actual_traj: np.ndarray,
                      ref_traj_actual: np.ndarray,
                      control_inputs: np.ndarray,
                      time_history: np.ndarray,
                      full_ref_traj: np.ndarray,
                      all_trajectories: List[Tuple[np.ndarray,
                                                   np.ndarray]] = None,
                      reference_target_end: np.ndarray = None,
                      dimension: int = 3):
    """
    可视化仿真结果
    
    Args:
        actual_traj: 实际执行轨迹 [N, dimension]
        ref_traj_actual: 实际跟踪的参考轨迹 [N, dimension]
        control_inputs: 控制输入历史 [N, dimension]
        time_history: 时间历史 [N]
        full_ref_traj: 完整参考轨迹 [M, dimension]（用于跟踪的轨迹）
        all_trajectories: 所有生成的参考轨迹列表
        reference_target_end: 参考目标终点 [dimension]
        dimension: 状态空间维度（默认3）
    """
    # 计算需要的子图数量
    # 基础图：3D轨迹图、X-Y投影、位置误差、控制输入（前3维）
    # 额外维度图：每个额外维度一个图
    num_extra_dims = max(0, dimension - 3)
    
    # 基础布局：2x2，如果有额外维度，增加行数
    if num_extra_dims > 0:
        n_rows = 2 + (num_extra_dims + 1) // 2  # 每行2个图
        n_cols = 2
        fig = plt.figure(figsize=(16, 5 * n_rows))
    else:
        n_rows = 2
        n_cols = 2
        fig = plt.figure(figsize=(16, 10))

    # ========== 3D Trajectory Plot (只显示前3维 x, y, z) ==========
    ax1 = fig.add_subplot(n_rows, n_cols, 1, projection='3d')

    # Plot all reference trajectories
    if all_trajectories is not None:
        for i, (traj, _) in enumerate(all_trajectories):
            if i == 0:
                # 第一条轨迹（用于跟踪的）用不同颜色和标签
                ax1.plot(traj[:, 0],
                         traj[:, 1],
                         traj[:, 2],
                         'b-',
                         linewidth=2,
                         alpha=0.7,
                         label=f'Tracked Reference Trajectory')
            else:
                # 其他轨迹用浅色显示
                ax1.plot(traj[:, 0],
                         traj[:, 1],
                         traj[:, 2],
                         'gray',
                         linewidth=1,
                         alpha=0.3)
        # 添加图例说明
        if len(all_trajectories) > 1:
            ax1.plot(
                [], [],
                'gray',
                linewidth=1,
                alpha=0.3,
                label=
                f'Other Reference Trajectories ({len(all_trajectories)-1})')
    else:
        # 如果没有多条轨迹，只显示当前跟踪的轨迹
        ax1.plot(full_ref_traj[:, 0],
                 full_ref_traj[:, 1],
                 full_ref_traj[:, 2],
                 'b--',
                 linewidth=2,
                 alpha=0.5,
                 label='Full Reference Trajectory (Spiral)')

    # Actual executed trajectory
    ax1.plot(actual_traj[:, 0],
             actual_traj[:, 1],
             actual_traj[:, 2],
             'r-',
             linewidth=2,
             label='MPC Actual Trajectory')

    # Reference trajectory points - 绘制完整的连续参考轨迹
    # 清理重复点（超出轨迹范围时会出现重复的终点）
    ref_traj_clean = []
    prev_point = None
    for i, point in enumerate(ref_traj_actual):
        if prev_point is None or np.linalg.norm(point - prev_point) > 1e-5:
            ref_traj_clean.append(point)
            prev_point = point

    ref_traj_clean = np.array(ref_traj_clean)

    # 确保参考轨迹的最后一个点就是目标终点
    # 如果参考轨迹目标终点存在，将最后一个点替换为目标终点
    if reference_target_end is not None and len(ref_traj_clean) > 0:
        ref_traj_clean[-1] = reference_target_end

    # 使用细线绘制参考轨迹，确保连续性
    if len(ref_traj_clean) > 1:
        ax1.plot(ref_traj_clean[:, 0],
                 ref_traj_clean[:, 1],
                 ref_traj_clean[:, 2],
                 'b-',
                 linewidth=1.5,
                 alpha=0.6,
                 label='Reference Trajectory (tracked)')
    else:
        # 如果只有单个点，使用scatter
        ax1.scatter(ref_traj_actual[:, 0],
                    ref_traj_actual[:, 1],
                    ref_traj_actual[:, 2],
                    c='blue',
                    s=20,
                    alpha=0.6,
                    label='Reference Points')

    # Start and end points
    ax1.scatter(actual_traj[0, 0],
                actual_traj[0, 1],
                actual_traj[0, 2],
                c='green',
                s=100,
                marker='o',
                label='Start (Actual)')

    # 参考轨迹的目标终点（当前跟踪的参考轨迹的终点）
    # 这是所有三个终点应该对齐的位置
    if reference_target_end is not None:
        ax1.scatter(reference_target_end[0],
                    reference_target_end[1],
                    reference_target_end[2],
                    c='blue',
                    s=150,
                    marker='*',
                    label='End (Reference Target)',
                    edgecolors='black',
                    linewidths=2,
                    zorder=12)

    # 实际跟踪的参考轨迹的最后一个点（应该和reference_target_end相同）
    # 如果不同，说明有问题，但我们仍然显示它
    if len(ref_traj_clean) > 0:
        ref_end = ref_traj_clean[-1]
        if reference_target_end is not None and np.linalg.norm(
                ref_end - reference_target_end) > 1e-3:
            # 如果不相同，显示警告标记
            ax1.scatter(ref_end[0],
                        ref_end[1],
                        ref_end[2],
                        c='orange',
                        s=120,
                        marker='^',
                        label='End (Reference Tracked - WARNING)',
                        edgecolors='red',
                        linewidths=2,
                        zorder=11,
                        alpha=0.8)
        else:
            # 如果相同，不单独显示（因为已经显示了目标终点）
            pass

    # 实际轨迹的终点（应该接近reference_target_end）
    ax1.scatter(actual_traj[-1, 0],
                actual_traj[-1, 1],
                actual_traj[-1, 2],
                c='red',
                s=100,
                marker='s',
                label='End (Actual)',
                zorder=10)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Tracking (Spiral)')
    ax1.legend()
    ax1.grid(True)

    # ========== X-Y Plane Projection (只显示前2维 x, y) ==========
    ax2 = fig.add_subplot(n_rows, n_cols, 2)

    # Plot all reference trajectories in X-Y plane
    if all_trajectories is not None:
        for i, (traj, _) in enumerate(all_trajectories):
            if i == 0:
                ax2.plot(traj[:, 0],
                         traj[:, 1],
                         'b-',
                         linewidth=2,
                         alpha=0.7,
                         label='Full Reference Trajectory')
            else:
                ax2.plot(traj[:, 0],
                         traj[:, 1],
                         'gray',
                         linewidth=1,
                         alpha=0.3)
        if len(all_trajectories) > 1:
            ax2.plot(
                [], [],
                'gray',
                linewidth=1,
                alpha=0.3,
                label=
                f'Other Reference Trajectories ({len(all_trajectories)-1})')
    else:
        ax2.plot(full_ref_traj[:, 0],
                 full_ref_traj[:, 1],
                 'b--',
                 linewidth=2,
                 alpha=0.5,
                 label='Reference Trajectory')

    # Plot tracked reference trajectory (continuous line, cleaned)
    if len(ref_traj_clean) > 1:
        ax2.plot(ref_traj_clean[:, 0],
                 ref_traj_clean[:, 1],
                 'b-',
                 linewidth=1.5,
                 alpha=0.6,
                 label='Reference Trajectory (tracked)')

    ax2.plot(actual_traj[:, 0],
             actual_traj[:, 1],
             'r-',
             linewidth=2,
             label='Actual Trajectory')
    ax2.scatter(actual_traj[0, 0],
                actual_traj[0, 1],
                c='green',
                s=100,
                marker='o',
                label='Start (Actual)')
    # 参考轨迹的目标终点（所有终点应该对齐的位置）
    if reference_target_end is not None:
        ax2.scatter(reference_target_end[0],
                    reference_target_end[1],
                    c='blue',
                    s=150,
                    marker='*',
                    label='End (Reference Target)',
                    edgecolors='black',
                    linewidths=2,
                    zorder=12)
    # 实际轨迹的终点（应该接近reference_target_end）
    ax2.scatter(actual_traj[-1, 0],
                actual_traj[-1, 1],
                c='red',
                s=100,
                marker='s',
                label='End (Actual)',
                zorder=10)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y Plane Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # ========== Position Tracking Error ==========
    ax3 = fig.add_subplot(n_rows, n_cols, 3)
    position_error = np.linalg.norm(actual_traj - ref_traj_actual, axis=1)
    ax3.plot(time_history, position_error, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Tracking Error (All Dimensions)')
    ax3.grid(True)

    # ========== Control Input (前3维) ==========
    ax4 = fig.add_subplot(n_rows, n_cols, 4)
    # 只显示前3维的控制输入（期望速度）
    dim_names = ['x', 'y', 'z'] + [chr(ord('a') + i) for i in range(max(0, dimension - 3))]
    if dimension > 26:
        dim_names = ['x', 'y', 'z'] + [f'd{i}' for i in range(3, dimension)]
    
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'pink']
    for i in range(min(3, dimension)):
        if i < control_inputs.shape[1]:
            ax4.plot(time_history,
                     control_inputs[:, i],
                     colors[i % len(colors)] + '-',
                     label=f'u_{dim_names[i]}',
                     linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Desired Velocity (m/s)')
    ax4.set_title('Control Input (First 3 Dimensions)')
    ax4.legend()
    ax4.grid(True)
    
    # ========== 额外维度的可视化 ==========
    if num_extra_dims > 0:
        plot_idx = 5  # 从第5个位置开始
        for dim_idx in range(3, dimension):
            row = (plot_idx - 1) // n_cols + 1
            col = (plot_idx - 1) % n_cols + 1
            ax = fig.add_subplot(n_rows, n_cols, plot_idx)
            
            # 绘制该维度的轨迹
            if dim_idx < actual_traj.shape[1]:
                ax.plot(time_history, actual_traj[:, dim_idx], 
                       'r-', linewidth=2, label=f'Actual (dim {dim_idx})')
            if dim_idx < ref_traj_actual.shape[1]:
                ax.plot(time_history, ref_traj_actual[:, dim_idx], 
                       'b--', linewidth=1.5, alpha=0.7, label=f'Reference (dim {dim_idx})')
            
            # 绘制该维度的控制输入（期望速度）
            if dim_idx < control_inputs.shape[1]:
                ax2_twin = ax.twinx()
                ax2_twin.plot(time_history, control_inputs[:, dim_idx], 
                             'g-', linewidth=1.5, alpha=0.6, label=f'Control (dim {dim_idx})')
                ax2_twin.set_ylabel('Desired Velocity (m/s)', color='g')
                ax2_twin.tick_params(axis='y', labelcolor='g')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Position (dim {dim_idx})', color='b')
            ax.set_title(f'Dimension {dim_idx} ({dim_names[dim_idx] if dim_idx < len(dim_names) else f"d{dim_idx}"})')
            ax.tick_params(axis='y', labelcolor='b')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            if dim_idx < control_inputs.shape[1]:
                ax2_twin.legend(loc='upper right')
            
            plot_idx += 1

    plt.tight_layout()
    plt.savefig('trajectory_tracking_result.png', dpi=150, bbox_inches='tight')
    print("   Results saved to: trajectory_tracking_result.png")
    plt.show()
