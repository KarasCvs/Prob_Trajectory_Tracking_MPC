"""
概率模型轨迹跟踪 MPC 主程序
实现基于高斯过程概率轨迹的MPC控制
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import do_mpc

# 导入自定义模块
from prob_mpc.template_prob_model import template_prob_model
from prob_mpc.template_prob_mpc import template_prob_mpc, update_mpc_gp_trajectory
from prob_mpc.template_prob_simulator import template_prob_simulator
from reference_trajectory import ReferenceTrajectoryGenerator
from prob_mpc.gp_trajectory import GaussianProcessTrajectory


def main(num_replanning: int = 0, debug: bool = False, trajectory_type: str = 'rollercoaster'):
    """
    主函数：概率模型轨迹跟踪 MPC 仿真
    
    Args:
        num_replanning: 重新规划次数（概率模型中，重规划仅体现为时间推进，GP自动提供连续轨迹）
        debug: 是否使用debug模式（减少GP训练迭代次数）
        trajectory_type: 轨迹类型 ('spiral' 或 'rollercoaster')
    """
    print("=" * 60)
    print("概率模型轨迹跟踪 MPC 仿真")
    print("=" * 60)
    print(f"重新规划次数: {num_replanning}")

    # ========== 1. 创建系统模型 ==========
    print("\n[1/7] 创建概率系统模型...")
    model = template_prob_model('SX')
    print(f"   状态维度: {model.n_x}")
    print(f"   输入维度: {model.n_u}")

    # ========== 2. 生成多条示例轨迹 ==========
    print("\n[2/7] 生成多条示例轨迹...")
    traj_gen = ReferenceTrajectoryGenerator(trajectory_type='spiral')

    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 定义起点和终点的多维高斯分布
    start_mean = np.array([0.0, 0.0, 0.0])
    start_std = 0.05  # 标准差 0.05m
    start_cov = start_std**2
    end_mean = np.array([1.0, 1.0, 1.0])
    end_std = 0.05
    end_cov = end_std**2

    # 生成多条参考轨迹（用于GP训练）
    num_trajectories = 20
    trajectory_duration = 10.0  # 秒
    num_points_per_traj = 200

    print(f"   生成 {num_trajectories} 条示例轨迹...")
    print(f"   起点分布: 均值={start_mean}, 标准差={start_std}m")
    print(f"   终点分布: 均值={end_mean}, 标准差={end_std}m")

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
            circle_ratio=0.6
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
            convergence_start=0.8,
            convergence_length=0.2
        )

    # ========== 3. 时间归一化 ==========
    print("\n[3/7] 时间归一化...")
    # GP训练需要统一的时间尺度，将所有轨迹归一化到[0,1]
    normalized_trajectories = []
    for traj, time_stamps in all_trajectories:
        # 归一化时间到[0,1]
        t_min, t_max = time_stamps.min(), time_stamps.max()
        if t_max - t_min > 1e-6:
            t_norm = (time_stamps - t_min) / (t_max - t_min)
        else:
            t_norm = np.zeros_like(time_stamps)
        normalized_trajectories.append((traj, t_norm))
    
    print(f"   时间归一化完成: [0, {trajectory_duration}]s -> [0, 1]")

    # ========== 4. 训练GP模型 ==========
    print("\n[4/7] 训练高斯过程模型...")
    gp_traj = GaussianProcessTrajectory(normalize_time=True)
    max_iters = 10 if debug else 200
    if debug:
        print(f"   Debug模式: GP训练仅进行 {max_iters} 次迭代")
    # 设置合理的inducing points数量（性能与精度平衡）
    # 对于20条轨迹，50-100个inducing points通常足够
    # 太多会导致预测变慢，太少会低估方差
    num_inducing = 50 if debug else 100
    gp_traj.fit(
        normalized_trajectories,
        optimize=True,
        kernel_type='RBF',
        noise_variance=1e-4,
        max_iters=max_iters,
        num_inducing=num_inducing
    )
    print("   GP模型训练完成")
    
    # 获取目标均值和方差
    goal_mean = gp_traj.get_goal_mean()
    goal_variance = gp_traj.get_goal_variance()
    print(f"   目标位置均值: {goal_mean}")
    print(f"   目标位置方差: {goal_variance}")

    # ========== 5. 创建概率模型MPC控制器 ==========
    print("\n[5/7] 创建概率模型MPC控制器...")
    mpc = template_prob_mpc(model, gp_trajectory=gp_traj, silence_solver=True)
    print(f"   预测时域: {mpc.settings.n_horizon} 步")
    print(f"   采样时间: {mpc.settings.t_step} 秒")

    # ========== 6. 创建仿真器 ==========
    print("\n[6/7] 创建概率模型仿真器...")
    simulator = template_prob_simulator(model, gp_trajectory=gp_traj, trajectory_duration=trajectory_duration)

    # ========== 7. 初始化状态 ==========
    print("\n[7/7] 初始化状态...")
    # 初始状态：使用实际轨迹起点的均值（确保与训练数据一致）
    # 注意：GP在t=0时的预测可能不准确（特别是对于过山车轨迹），
    # 因此直接使用训练数据的起点均值更可靠
    actual_start_mean = np.mean([traj[0] for traj, _ in normalized_trajectories], axis=0)
    actual_end_mean = np.mean([traj[-1] for traj, _ in normalized_trajectories], axis=0)
    
    # 验证GP预测的起点和终点（用于调试）
    gp_start_mean = gp_traj.predict_mean(0.0)
    gp_end_mean_pred = gp_traj.predict_mean(1.0)
    start_error = np.linalg.norm(gp_start_mean - actual_start_mean)
    end_error = np.linalg.norm(gp_end_mean_pred - actual_end_mean)
    if start_error > 0.1:  # 如果误差较大，给出警告
        print(f"   注意: GP预测起点与实际起点均值差异较大 ({start_error:.4f}m)")
        print(f"   GP预测起点: {gp_start_mean}")
        print(f"   实际起点均值: {actual_start_mean}")
        print(f"   使用实际起点均值作为初始状态（更可靠）")
    if end_error > 0.1:  # 如果误差较大，给出警告
        print(f"   注意: GP预测终点与实际终点均值差异较大 ({end_error:.4f}m)")
        print(f"   GP预测终点: {gp_end_mean_pred}")
        print(f"   实际终点均值: {actual_end_mean}")
        print(f"   将在最后一个时间步使用实际终点均值（更可靠）")
    
    x0 = np.array([
        actual_start_mean[0],  # p_x
        actual_start_mean[1],  # p_y
        actual_start_mean[2],  # p_z
        0.0,  # v_x
        0.0,  # v_y
        0.0  # v_z
    ]).reshape(-1, 1)

    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    # ========== 8. 仿真循环 ==========
    print("\n[8/8] 开始仿真...")

    # 仿真参数
    sim_time = trajectory_duration
    n_steps = int(sim_time / mpc.settings.t_step)

    # 存储数据
    actual_trajectory = []
    gp_mean_trajectory = []  # GP均值轨迹
    gp_variance_history = []  # GP方差历史
    alpha_history = []  # 权重历史
    control_inputs = []
    time_history = []

    x_current = x0.copy()
    current_time = 0.0

    print(f"   仿真步数: {n_steps}")
    print(f"   开始仿真循环...\n")
    
    # 性能统计
    import time
    import torch
    gp_times = []
    mpc_times = []
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"   计算设备: {device_info}")
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"   GP Inducing Points: {num_inducing}")

    for k in range(n_steps):
        # ========== 更新 MPC GP轨迹 ==========
        remaining_time = sim_time - current_time
        remaining_steps = int(round(remaining_time / mpc.settings.t_step))
        terminal_index = None
        if remaining_steps <= mpc.settings.n_horizon:
            terminal_index = max(0, remaining_steps - 1)
        
        # 注意：在概率模型中，时间需要映射到归一化时间[0,1]
        t_normalized = current_time / trajectory_duration
        
        # 统计GP预测时间
        t_gp_start = time.time()
        update_mpc_gp_trajectory(
            mpc, 
            gp_traj, 
            t_normalized,  # 使用归一化时间
            terminal_index=terminal_index,
            trajectory_duration=trajectory_duration,
            actual_end_mean=actual_end_mean  # 使用实际终点均值，确保终端约束准确
        )
        t_gp_end = time.time()
        gp_times.append(t_gp_end - t_gp_start)
        
        # ========== MPC 求解 ==========
        t_mpc_start = time.time()
        u0 = mpc.make_step(x_current)
        t_mpc_end = time.time()
        mpc_times.append(t_mpc_end - t_mpc_start)

        # ========== 仿真器步进 ==========
        y_next = simulator.make_step(u0)

        # ========== 状态更新 ==========
        x_current = y_next
        
        # ========== 数据记录 ==========
        actual_trajectory.append(x_current[:3].flatten())
        
        # 查询GP均值和方差（用于可视化）
        # 注意：在t=0和t=1时，GP预测可能不准确，使用实际起点/终点均值更可靠
        if k == 0 and t_normalized < 1e-6:
            # 第一个时间步，使用实际起点均值（与初始状态一致）
            gp_mean = actual_start_mean.copy()
        elif k == n_steps - 1 or t_normalized >= 1.0 - 1e-6:
            # 最后一个时间步，使用实际终点均值（与数据集一致）
            gp_mean = actual_end_mean.copy()
        else:
            gp_mean = gp_traj.predict_mean(t_normalized)
        gp_variance = gp_traj.predict_variance(t_normalized)
        gp_mean_trajectory.append(gp_mean)
        gp_variance_history.append(gp_variance)
        
        # 计算权重（用于可视化，与MPC中的公式一致）
        trace_sigma = np.sum(gp_variance)
        threshold = 0.1  # 与template_prob_mpc.py中的阈值一致
        alpha_k = threshold / (threshold + trace_sigma)
        alpha_k = np.clip(alpha_k, 0.01, 1.0)
        alpha_history.append(alpha_k)
        
        control_inputs.append(u0.flatten())
        time_history.append(current_time)

        # 时间更新
        current_time += mpc.settings.t_step

        # 进度显示
        if (k + 1) % 20 == 0:
            pos_error = np.linalg.norm(x_current[:3].flatten() - gp_mean)
            avg_gp_time = np.mean(gp_times[-20:]) * 1000  # 最近20步的平均GP时间（ms）
            avg_mpc_time = np.mean(mpc_times[-20:]) * 1000  # 最近20步的平均MPC时间（ms）
            # 显示当前alpha值（用于调试）
            current_alpha = alpha_history[-1] if len(alpha_history) > 0 else 0.0
            trace_var = np.sum(gp_variance)
            print(f"   步骤 {k+1}/{n_steps}: 时间={current_time:.2f}s, "
                  f"位置误差={pos_error:.3f}m, "
                  f"GP={avg_gp_time:.2f}ms, MPC={avg_mpc_time:.2f}ms, "
                  f"α={current_alpha:.3f}, trace_var={trace_var:.4f}")

    print("\n仿真完成！")
    
    # 打印性能统计
    print("\n性能统计:")
    print(f"   GP预测时间: 平均={np.mean(gp_times)*1000:.2f}ms, "
          f"最小={np.min(gp_times)*1000:.2f}ms, "
          f"最大={np.max(gp_times)*1000:.2f}ms, "
          f"总计={np.sum(gp_times):.3f}s")
    print(f"   MPC求解时间: 平均={np.mean(mpc_times)*1000:.2f}ms, "
          f"最小={np.min(mpc_times)*1000:.2f}ms, "
          f"最大={np.max(mpc_times)*1000:.2f}ms, "
          f"总计={np.sum(mpc_times):.3f}s")
    print(f"   总计算时间: {np.sum(gp_times) + np.sum(mpc_times):.3f}s")
    print(f"   GP占比: {np.sum(gp_times) / (np.sum(gp_times) + np.sum(mpc_times)) * 100:.1f}%")
    print(f"   MPC占比: {np.sum(mpc_times) / (np.sum(gp_times) + np.sum(mpc_times)) * 100:.1f}%")

    # 检查终点位置
    actual_end = np.array(actual_trajectory[-1])
    # 使用实际终点均值（与数据集一致），而不是GP预测值
    gp_end_mean = actual_end_mean  # 使用实际终点均值，确保与数据集一致
    
    end_error = np.linalg.norm(actual_end - gp_end_mean)
    
    print(f"\n终点位置检查:")
    print(f"   实际轨迹终点: {actual_end}")
    print(f"   数据集终点均值: {actual_end_mean}")
    print(f"   GP目标均值 (使用数据集终点均值): {gp_end_mean}")
    print(f"   终点误差: {end_error:.4f}m")

    # ========== 9. 可视化 ==========
    print("\n生成可视化...")
    visualize_prob_results(
        np.array(actual_trajectory),
        np.array(gp_mean_trajectory),
        np.array(gp_variance_history),
        np.array(alpha_history),
        np.array(control_inputs),
        np.array(time_history),
        all_trajectories,
        gp_traj,
        trajectory_duration,
        reference_target_end=gp_end_mean
    )

    print("\n" + "=" * 60)
    print("概率模型轨迹跟踪 MPC 仿真完成！")
    print("=" * 60)


def visualize_prob_results(
    actual_traj: np.ndarray,
    gp_mean_traj: np.ndarray,
    gp_variance_history: np.ndarray,
    alpha_history: np.ndarray,
    control_inputs: np.ndarray,
    time_history: np.ndarray,
    all_trajectories: List[Tuple[np.ndarray, np.ndarray]],
    gp_trajectory: GaussianProcessTrajectory,
    trajectory_duration: float,
    reference_target_end: np.ndarray = None
):
    """
    可视化概率模型仿真结果
    
    Args:
        actual_traj: 实际执行轨迹 [N, 3]
        gp_mean_traj: GP均值轨迹 [N, 3]
        gp_variance_history: GP方差历史 [N, 3]
        alpha_history: 权重历史 [N]
        control_inputs: 控制输入历史 [N, 3]
        time_history: 时间历史 [N]
        all_trajectories: 所有生成的参考轨迹列表
        gp_trajectory: GP轨迹模型
        trajectory_duration: 轨迹持续时间
        reference_target_end: 参考目标终点
    """
    fig = plt.figure(figsize=(18, 12))

    # ========== 3D Trajectory Plot ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # Plot all reference trajectories (示例轨迹)
    if all_trajectories is not None:
        for i, (traj, _) in enumerate(all_trajectories):
            if i == 0:
                ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                        'gray', linewidth=1, alpha=0.3, label='Example Trajectories')
            else:
                ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                        'gray', linewidth=1, alpha=0.3)
        if len(all_trajectories) > 1:
            ax1.plot([], [], 'gray', linewidth=1, alpha=0.3,
                    label=f'Example Trajectories ({len(all_trajectories)})')

    # GP均值轨迹（带不确定性区间）
    ax1.plot(gp_mean_traj[:, 0], gp_mean_traj[:, 1], gp_mean_traj[:, 2],
            'b-', linewidth=2, alpha=0.8, label='GP Mean Trajectory')
    
    # 绘制不确定性区间（±2σ）
    std_traj = np.sqrt(gp_variance_history)
    # 在x方向的不确定性
    ax1.plot(gp_mean_traj[:, 0] + 2*std_traj[:, 0], 
            gp_mean_traj[:, 1], gp_mean_traj[:, 2],
            'b--', linewidth=1, alpha=0.3, label='±2σ Uncertainty')
    ax1.plot(gp_mean_traj[:, 0] - 2*std_traj[:, 0], 
            gp_mean_traj[:, 1], gp_mean_traj[:, 2],
            'b--', linewidth=1, alpha=0.3)

    # Actual executed trajectory
    ax1.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2],
            'r-', linewidth=2, label='MPC Actual Trajectory')

    # Start and end points
    ax1.scatter(actual_traj[0, 0], actual_traj[0, 1], actual_traj[0, 2],
               c='green', s=100, marker='o', label='Start')
    
    if reference_target_end is not None:
        ax1.scatter(reference_target_end[0], reference_target_end[1], reference_target_end[2],
                   c='blue', s=150, marker='*', label='End (GP Goal Mean)',
                   edgecolors='black', linewidths=2, zorder=12)
    
    ax1.scatter(actual_traj[-1, 0], actual_traj[-1, 1], actual_traj[-1, 2],
               c='red', s=100, marker='s', label='End (Actual)', zorder=10)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Tracking (Probabilistic)')
    ax1.legend()
    ax1.grid(True)

    # ========== X-Y Plane Projection ==========
    ax2 = fig.add_subplot(2, 3, 2)
    
    if all_trajectories is not None:
        for i, (traj, _) in enumerate(all_trajectories):
            ax2.plot(traj[:, 0], traj[:, 1], 'gray', linewidth=1, alpha=0.3)
    
    ax2.plot(gp_mean_traj[:, 0], gp_mean_traj[:, 1],
            'b-', linewidth=2, alpha=0.8, label='GP Mean')
    ax2.plot(actual_traj[:, 0], actual_traj[:, 1],
            'r-', linewidth=2, label='Actual')
    
    ax2.scatter(actual_traj[0, 0], actual_traj[0, 1],
               c='green', s=100, marker='o', label='Start')
    if reference_target_end is not None:
        ax2.scatter(reference_target_end[0], reference_target_end[1],
                   c='blue', s=150, marker='*', label='End (GP Goal)',
                   edgecolors='black', linewidths=2, zorder=12)
    ax2.scatter(actual_traj[-1, 0], actual_traj[-1, 1],
               c='red', s=100, marker='s', label='End (Actual)', zorder=10)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y Plane Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # ========== Position Tracking Error ==========
    ax3 = fig.add_subplot(2, 3, 3)
    position_error = np.linalg.norm(actual_traj - gp_mean_traj, axis=1)
    ax3.plot(time_history, position_error, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Tracking Error (vs GP Mean)')
    ax3.grid(True)

    # ========== GP Variance (Uncertainty) ==========
    ax4 = fig.add_subplot(2, 3, 4)
    total_variance = np.sum(gp_variance_history, axis=1)
    ax4.plot(time_history, total_variance, 'b-', linewidth=2, label='Total Variance')
    ax4.plot(time_history, gp_variance_history[:, 0], 'r--', linewidth=1, alpha=0.7, label='σ²_x')
    ax4.plot(time_history, gp_variance_history[:, 1], 'g--', linewidth=1, alpha=0.7, label='σ²_y')
    ax4.plot(time_history, gp_variance_history[:, 2], 'm--', linewidth=1, alpha=0.7, label='σ²_z')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Variance')
    ax4.set_title('GP Variance (Uncertainty)')
    ax4.legend()
    ax4.grid(True)

    # ========== Weight α(t) ==========
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(time_history, alpha_history, 'g-', linewidth=2, label='α(t)')
    # 添加参考线：alpha=0.5（平衡点）
    if len(time_history) > 0:
        ax5.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Balance (0.5)')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Weight α(t)')
    ax5.set_title('Trajectory Weight α(t)\n(High=Track Traj, Low=Track Goal)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.1])
    # 如果alpha有变化，自动调整y轴范围以突出变化
    if len(alpha_history) > 0:
        alpha_min, alpha_max = np.min(alpha_history), np.max(alpha_history)
        if alpha_max - alpha_min > 0.1:  # 如果有明显变化
            ax5.set_ylim([max(0, alpha_min - 0.1), min(1.1, alpha_max + 0.1)])
    ax5.legend()

    # ========== Control Input ==========
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(time_history, control_inputs[:, 0], 'r-', label='a_x', linewidth=2)
    ax6.plot(time_history, control_inputs[:, 1], 'g-', label='a_y', linewidth=2)
    ax6.plot(time_history, control_inputs[:, 2], 'b-', label='a_z', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Acceleration (m/s²)')
    ax6.set_title('Control Input')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig('probabilistic_trajectory_tracking_result.png', dpi=150, bbox_inches='tight')
    print("   Results saved to: probabilistic_trajectory_tracking_result.png")
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='概率模型轨迹跟踪 MPC 仿真')
    parser.add_argument('--num-replanning', type=int, default=0,
                       help='重新规划次数（概率模型中，重规划仅体现为时间推进）')
    parser.add_argument('--debug', action='store_true',
                       help='Debug模式：GP训练仅进行10次迭代')
    parser.add_argument('--trajectory-type', type=str, default='rollercoaster',
                       choices=['spiral', 'rollercoaster'],
                       help='轨迹类型: spiral (螺旋轨迹) 或 rollercoaster (过山车轨迹)')

    args = parser.parse_args()
    main(num_replanning=args.num_replanning, debug=args.debug, trajectory_type=args.trajectory_type)
