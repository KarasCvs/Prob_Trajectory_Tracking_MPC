"""
概率模型轨迹跟踪 MPC 主程序
实现基于高斯过程概率轨迹的MPC控制
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import do_mpc

# 导入自定义模块
from prob_mpc.template_prob_model import template_prob_model
from prob_mpc.template_prob_mpc import (
    template_prob_mpc,
    update_mpc_gp_trajectory,
    update_mpc_from_precomputed_ref,
)
from prob_mpc.template_prob_simulator import template_prob_simulator
from reference_trajectory import ReferenceTrajectoryGenerator
from prob_mpc.gp_trajectory import GaussianProcessTrajectory

# 延迟导入 h5_trajectory_loader（仅在需要时导入，避免在没有 h5py 时出错）
try:
    from h5_trajectory_loader import H5TrajectoryLoader, load_trajectories_from_dataset
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    H5TrajectoryLoader = None
    load_trajectories_from_dataset = None


def main(num_replanning: int = 0,
         debug: bool = False,
         trajectory_type: str = 'rollercoaster',
         dimension: int = 3,
         estimate_velocity: bool = True,
         alpha_threshold: float = 0.0001,
         use_real_trajectory: bool = False,
         dataset_path: Optional[str] = None,
         num_steps: int = 100,
         enable_terminal_constraint: bool = False,
         observation_noise_std: float = 0.0):
    """
    主函数：概率模型轨迹跟踪 MPC 仿真
    
    Args:
        num_replanning: 重新规划次数（默认0，不重新规划）
        debug: 是否启用调试模式
        trajectory_type: 轨迹类型 ('spiral' 或 'rollercoaster')
        dimension: 状态空间维度（默认3，表示3D空间）
        estimate_velocity: 已废弃（保留以兼容接口，但不再使用）
        alpha_threshold: Alpha权重计算的阈值参数（默认0.0001）
                       - 较小的值：alpha更小，更强调目标项，终点误差更小
                       - 较大的值：alpha更大，更强调轨迹跟踪
        use_real_trajectory: 是否使用真实轨迹（从H5数据集加载）
        dataset_path: 数据集路径（当使用真实轨迹时必需）
        num_steps: 仿真步数（默认100）
        observation_noise_std: 观测噪声标准差（同时作用于当前位置与终点；每次 replan 时对观测终点加噪；误差仍按真实终点算；0表示无噪声）。
                               扰动仅加在 MPC 观测到的位置上，真实状态/仿真不扰动。
    
    Args:
        num_replanning: 重新规划次数（概率模型中，重规划仅体现为时间推进，GP自动提供连续轨迹）
        debug: 是否使用debug模式（减少GP训练迭代次数）
        trajectory_type: 轨迹类型 ('spiral' 或 'rollercoaster')
        dimension: 状态空间维度（默认3，表示3D空间）
        estimate_velocity: 是否估计速度（默认True。True=从位置估计，False=假设完全可观测）
    """
    print("=" * 60)
    print("概率模型轨迹跟踪 MPC 仿真（一阶系统）")
    print("=" * 60)
    print(f"重新规划次数: {num_replanning}")
    print(f"状态空间维度: {dimension}")
    print(f"控制模式: 一阶系统（状态=位置，控制输入=期望速度，p_dot=u）")
    print(f"Alpha阈值参数: {alpha_threshold}")
    print(f"注意: estimate_velocity参数已废弃，系统现在是一阶系统，不会产生旋转")
    if observation_noise_std > 0:
        print(f"观测噪声: 标准差={observation_noise_std}（作用于当前位置与终点观测，真实轨迹与误差仍按真实值算）")

    # ========== 1. 创建系统模型 ==========
    # 注意：如果使用真实轨迹，维度可能会在加载轨迹后更新
    # 所以先不创建模型，等加载轨迹后再创建
    if not use_real_trajectory:
        print("\n[1/7] 创建概率系统模型...")
        model = template_prob_model('SX', dimension=dimension)
        print(f"   状态维度: {model.n_x}")
        print(f"   输入维度: {model.n_u}")
    else:
        model = None  # 稍后创建

    # ========== 2. 生成或加载多条示例轨迹 ==========
    print("\n[2/7] 生成或加载多条示例轨迹...")

    if use_real_trajectory:
        # 使用真实轨迹
        if dataset_path is None:
            raise ValueError("使用真实轨迹时必须提供 dataset_path 参数")

        if not HAS_H5PY:
            raise ImportError(
                "使用真实轨迹需要安装 h5py 模块。请运行: pip install h5py 或 conda install h5py"
            )

        print(f"   从数据集加载真实轨迹: {dataset_path}")
        print(f"   指定维度: {dimension}")
        loader = H5TrajectoryLoader(dataset_path,
                                    max_trajectories=5,
                                    dimension=dimension)
        all_trajectories = loader.load_all_trajectories()

        if len(all_trajectories) == 0:
            raise ValueError(f"未能从数据集加载任何轨迹: {dataset_path}")

        # 从随机的一条轨迹中选择起点和终点
        # 注意：真实轨迹格式是 (trajectory, time_stamps)，其中 trajectory 是 [N, dimension]
        if len(all_trajectories) == 0:
            raise ValueError("真实轨迹数据为空")

        # 随机选择一条轨迹
        np.random.seed(42)  # 确保可重复性
        selected_idx = np.random.randint(0, len(all_trajectories))
        selected_traj, selected_ts = all_trajectories[selected_idx]

        if len(selected_traj) == 0:
            raise ValueError("选中的轨迹为空")

        # 从选中的轨迹中选择起点和终点
        start_mean = selected_traj[0].copy()  # 第一条轨迹的起点
        end_mean = selected_traj[-1].copy()  # 第一条轨迹的终点

        # 检查真实轨迹的维度（现在应该与指定维度匹配，因为加载时已经选择了维度）
        real_dimension = start_mean.shape[0]
        if dimension != real_dimension:
            print(f"   警告: 真实轨迹维度 ({real_dimension}) 与指定维度 ({dimension}) 不匹配")
            print(f"   将使用加载后的轨迹维度: {real_dimension}")
            dimension = real_dimension
        else:
            print(f"   轨迹维度匹配: {dimension}")

        # 现在创建系统模型（使用正确的维度）
        print("\n[1/7] 创建概率系统模型...")
        model = template_prob_model('SX', dimension=dimension)
        print(f"   状态维度: {model.n_x}")
        print(f"   输入维度: {model.n_u}")

        # 计算起点和终点的统计信息（用于显示）
        start_points = [
            traj[0] for traj, _ in all_trajectories if len(traj) > 0
        ]
        end_points = [
            traj[-1] for traj, _ in all_trajectories if len(traj) > 0
        ]
        start_std = np.std(start_points,
                           axis=0).mean() if len(start_points) > 0 else 0.05
        end_std = np.std(end_points,
                         axis=0).mean() if len(end_points) > 0 else 0.05
        start_cov = start_std**2
        end_cov = end_std**2

        print(f"   从轨迹 {selected_idx+1}/{len(all_trajectories)} 中选择起点和终点")
        print(f"   起点: {start_mean[:min(3, len(start_mean))]}")
        print(f"   终点: {end_mean[:min(3, len(end_mean))]}")

        # 获取轨迹持续时间（从选中的轨迹计算）
        if len(selected_ts) > 1:
            trajectory_duration = selected_ts[-1] - selected_ts[0]
        else:
            # 如果无法从时间戳计算，使用轨迹长度估算
            trajectory_duration = len(selected_traj) * 0.1  # 假设每个点间隔0.1秒

        if trajectory_duration <= 0:
            trajectory_duration = num_steps * mpc.settings.t_step  # 使用指定的步数计算

        print(f"   加载了 {len(all_trajectories)} 条真实轨迹")
        print(
            f"   起点分布: 均值={start_mean[:min(3, len(start_mean))]}, 标准差={start_std:.4f}"
        )
        print(
            f"   终点分布: 均值={end_mean[:min(3, len(end_mean))]}, 标准差={end_std:.4f}"
        )
        print(f"   轨迹持续时间: {trajectory_duration:.2f}秒")

        # 创建轨迹生成器（用于插值，但使用真实轨迹数据）
        # 注意：真实轨迹格式是 (trajectory, time_stamps)，其中 trajectory 是 [N, dimension]
        # 对于GP训练，我们需要多条轨迹，所以直接使用 all_trajectories
        # 但对于参考轨迹生成器，我们使用选中的轨迹作为参考
        reference_trajectory = selected_traj
        reference_time_stamps = selected_ts
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type='real',
            real_trajectory=(reference_trajectory, reference_time_stamps))
    else:
        # 使用生成的轨迹
        traj_gen = ReferenceTrajectoryGenerator(trajectory_type='spiral')

        # 设置随机种子以确保可重复性
        np.random.seed(42)

        # 定义起点和终点的多维高斯分布
        start_mean = np.zeros(dimension)
        start_mean[:3] = [0.0, 0.0, 0.0]  # 前3维用于3D空间
        start_std = 0.05  # 标准差 0.05m
        start_cov = start_std**2
        end_mean = np.zeros(dimension)
        end_mean[:3] = [1.0, 1.0, 1.0]  # 前3维用于3D空间
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
                circle_ratio=0.6,
                dimension=dimension)
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
                convergence_length=0.2,
                dimension=dimension)

    # ========== 3. 时间归一化 ==========
    print("\n[3/7] 时间归一化...")
    # GP训练需要统一的时间尺度，将所有轨迹归一化到[0,1]
    normalized_trajectories = []
    for traj, time_stamps in all_trajectories:
        # 确保轨迹是2D数组
        if len(traj.shape) == 1:
            traj = traj.reshape(1, -1)

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
    gp_traj = GaussianProcessTrajectory(normalize_time=True,
                                        dimension=dimension)
    max_iters = 10 if debug else 200
    if debug:
        print(f"   Debug模式: GP训练仅进行 {max_iters} 次迭代")
    # 设置合理的inducing points数量（性能与精度平衡）
    # 对于20条轨迹，50-100个inducing points通常足够
    # 太多会导致预测变慢，太少会低估方差
    num_inducing = 10 if debug else 50
    gp_traj.fit(normalized_trajectories,
                optimize=True,
                kernel_type='RBF',
                noise_variance=1e-4,
                max_iters=max_iters,
                num_inducing=num_inducing)
    print("   GP模型训练完成")

    # 获取目标均值和方差
    goal_mean = gp_traj.get_goal_mean()
    goal_variance = gp_traj.get_goal_variance()
    print(f"   目标位置均值: {goal_mean}")
    print(f"   目标位置方差: {goal_variance}")

    # ========== 4.5. 根据GP轨迹采样密度动态计算步长 ==========
    # 计算GP mean轨迹的总路径长度（用于估算速度）
    num_gp_samples = 100  # 采样点数
    gp_mean_samples = []
    for i in range(num_gp_samples + 1):
        t_sample = i / num_gp_samples  # [0, 1]
        gp_mean_sample = gp_traj.predict_mean(t_sample)[:dimension]
        gp_mean_samples.append(gp_mean_sample)
    
    # 计算GP mean轨迹的总路径长度
    gp_mean_path_length = 0.0
    for i in range(len(gp_mean_samples) - 1):
        segment_length = np.linalg.norm(gp_mean_samples[i + 1] - gp_mean_samples[i])
        gp_mean_path_length += segment_length
    
    print(f"   GP mean轨迹总路径长度: {gp_mean_path_length:.4f}m")
    
    # 计算原始轨迹的平均速度（从训练数据估算）
    avg_speed = 0.0
    total_path_length = 0.0
    total_time = 0.0
    num_trajectories_used = 0
    
    for traj, time_stamps in normalized_trajectories:
        if len(traj) > 1 and len(time_stamps) > 1:
            # 计算路径长度（基于归一化轨迹）
            traj_path_length = 0.0
            for i in range(len(traj) - 1):
                traj_path_length += np.linalg.norm(traj[i+1] - traj[i])
            
            # 归一化时间[0,1]对应原始时间trajectory_duration
            traj_time_span = time_stamps[-1] - time_stamps[0] if len(time_stamps) > 1 else 1.0
            traj_time_original = traj_time_span * trajectory_duration
            
            total_path_length += traj_path_length
            total_time += traj_time_original
            num_trajectories_used += 1
    
    # 如果无法计算，使用GP路径长度和轨迹持续时间
    if total_time > 1e-6 and num_trajectories_used > 0:
        avg_speed = total_path_length / total_time
    else:
        avg_speed = gp_mean_path_length / trajectory_duration if trajectory_duration > 1e-6 else 0.1
    
    print(f"   估算平均速度: {avg_speed:.4f}m/s (基于{num_trajectories_used}条轨迹)")
    
    # 根据步数和平均速度计算合适的步长
    if num_steps > 0:
        # 计算完成GP轨迹所需的总时间
        if avg_speed > 1e-6:
            required_time = gp_mean_path_length / avg_speed
        else:
            required_time = trajectory_duration
        
        # 根据步数计算步长
        dynamic_t_step = required_time / num_steps
        print(f"   动态步长: {dynamic_t_step:.4f}s (基于{num_steps}步和平均速度{avg_speed:.4f}m/s)")
    else:
        # 如果未指定步数，使用默认步长
        dynamic_t_step = 0.1

    # ========== 5. 创建概率模型MPC控制器 ==========
    print("\n[5/7] 创建概率模型MPC控制器...")
    mpc = template_prob_mpc(
        model,
        gp_trajectory=gp_traj,
        silence_solver=True,
        alpha_threshold=alpha_threshold,
        enable_terminal_constraint=enable_terminal_constraint)
    # 更新MPC的步长（可以在setup后修改settings）
    mpc.settings.t_step = dynamic_t_step
    print(f"   预测时域: {mpc.settings.n_horizon} 步")
    print(f"   采样时间: {mpc.settings.t_step} 秒")
    print(f"   终点约束: {'启用' if enable_terminal_constraint else '禁用'}")

    # ========== 6. 创建仿真器 ==========
    print("\n[6/7] 创建概率模型仿真器...")
    simulator = template_prob_simulator(
        model,
        gp_trajectory=gp_traj,
        trajectory_duration=trajectory_duration,
        alpha_threshold=alpha_threshold,
        t_step=dynamic_t_step)  # 传入动态计算的步长

    # ========== 7. 初始化状态 ==========
    print("\n[7/7] 初始化状态...")
    # 初始状态：使用实际轨迹起点的均值（确保与训练数据一致）
    # 注意：GP在t=0时的预测可能不准确（特别是对于过山车轨迹），
    # 因此直接使用训练数据的起点均值更可靠
    actual_start_mean = np.mean(
        [traj[0] for traj, _ in normalized_trajectories], axis=0)
    actual_end_mean = np.mean(
        [traj[-1] for traj, _ in normalized_trajectories], axis=0)

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

    # 构建初始状态向量：只有位置（一阶系统）
    x0_list = []
    for i in range(dimension):
        if i < len(actual_start_mean):
            x0_list.append(actual_start_mean[i])  # 位置
        else:
            x0_list.append(0.0)
    x0 = np.array(x0_list).reshape(-1, 1)

    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    # ========== 8. 仿真循环 ==========
    print("\n[8/8] 开始仿真...")

    # 仿真参数
    # 如果指定了步数，使用指定的步数；否则使用轨迹持续时间
    if num_steps > 0:
        n_steps = num_steps
        sim_time = n_steps * dynamic_t_step  # 使用动态计算的步长
        print(f"   总仿真时间: {sim_time:.2f}s")
    else:
        sim_time = trajectory_duration
        n_steps = int(sim_time / mpc.settings.t_step)

    # 存储数据
    actual_trajectory = []
    gp_mean_trajectory = []  # GP均值轨迹
    gp_variance_history = []  # GP方差历史
    alpha_history = []  # 权重历史
    control_inputs = []
    time_history = []

    # 跟踪实际轨迹的累积路径长度
    cumulative_path_length = 0.0
    previous_position = None

    # 初始化状态和时间
    x_current = x0.copy()
    current_time = 0.0
    previous_position = x0[:dimension].flatten()  # 用于计算路径长度

    # 先记录初始状态（用于可视化起点）
    actual_trajectory.append(x0.flatten())
    # 记录初始GP均值和方差
    gp_mean_trajectory.append(actual_start_mean.copy())
    gp_variance_initial = gp_traj.predict_variance(0.0)
    if len(gp_variance_initial) < dimension:
        gp_variance_initial = np.pad(gp_variance_initial,
                                     (0, dimension - len(gp_variance_initial)),
                                     'constant',
                                     constant_values=1e-6)
    gp_variance_history.append(gp_variance_initial[:dimension])
    # 计算初始权重
    trace_sigma_initial = np.sum(gp_variance_initial[:dimension])
    # 使用传入的 alpha_threshold 参数
    alpha_initial = alpha_threshold / (alpha_threshold + trace_sigma_initial)
    alpha_initial = np.clip(alpha_initial, 0.01, 1.0)
    alpha_history.append(alpha_initial)
    # 初始控制输入为0
    control_inputs.append(np.zeros(dimension))
    time_history.append(0.0)

    # ========== 预计算 GP mean 整条参考轨迹（仅此一次，训练数据不变 GP 不变）==========
    # num_steps = 一条轨迹的总步数 = 从 GP mean 采样的参考点数。
    # 整段仿真共用这一条轨迹，不再按步或按 replan 重查 GP。
    times_ref = np.linspace(0.0, 1.0, num_steps + 1)
    ref_mean_traj, ref_var_traj = gp_traj.predict_mean_and_variance_batch(times_ref)
    ref_mean_traj = np.asarray(ref_mean_traj)
    ref_var_traj = np.asarray(ref_var_traj)
    if ref_mean_traj.ndim == 1:
        ref_mean_traj = ref_mean_traj.reshape(-1, 1)
    if ref_var_traj.ndim == 1:
        ref_var_traj = ref_var_traj.reshape(-1, 1)
    d = ref_mean_traj.shape[1]
    ref_mean_traj[0] = np.pad(np.asarray(actual_start_mean).flatten(), (0, max(0, d - len(np.asarray(actual_start_mean).flatten()))), 'constant')[:d]
    ref_mean_traj[-1] = np.pad(np.asarray(actual_end_mean).flatten(), (0, max(0, d - len(np.asarray(actual_end_mean).flatten()))), 'constant')[:d]
    goal_variance = np.asarray(gp_traj.get_goal_variance()).flatten()
    if len(goal_variance) < dimension:
        goal_variance = np.pad(goal_variance, (0, dimension - len(goal_variance)),
                              'constant', constant_values=1e-6)
    print(f"   预计算参考点数: {num_steps + 1}（仅预测一次 GP 轨迹，replan 时不再查 GP）")

    # 重规划步数：先规划一次 → 执行 N 步 → 再重规划 → 再执行 N 步 …；N = num_steps / replanning
    # 例如 num_steps=100, replanning=5 → 在步 0,20,40,60,80 做 replan，每段执行 20 步
    # 例如 num_steps=100, replanning=20 → 在步 0,5,10,...,95 做 replan，每段执行 5 步
    if num_replanning > 0:
        step_per_phase = max(1, n_steps // num_replanning)
        replan_steps = np.arange(0, n_steps, step_per_phase)
        if len(replan_steps) > 0:
            print(f"   重规划次数: {len(replan_steps)}，每段 {step_per_phase} 步（先规划→执行→再重规划→再执行…）")
            print(f"   重规划步: {replan_steps}")
    else:
        replan_steps = np.array([], dtype=int)

    # 终点观测噪声：每次 replan 时对「观测到的终点」加噪，MPC 跟踪该观测终点；误差仍用真实终点 vs 停止位置
    observed_goal = np.asarray(actual_end_mean).flatten().copy()
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

    replan_count = 0
    for k in range(n_steps):
        # ========== 更新 MPC 参考（预计算轨迹 + 可选重规划）==========
        remaining_time = sim_time - current_time
        remaining_steps = int(round(remaining_time / mpc.settings.t_step))

        # 计算实际轨迹的累积路径长度
        current_position = x_current[:dimension].flatten()
        if previous_position is not None:
            step_length = np.linalg.norm(current_position - previous_position)
            cumulative_path_length += step_length

        # 路径进度：实际已走路径长 / GP mean 路径长，表示「沿参考路径的完成度」，限制在 [0, 100%]
        raw = cumulative_path_length / gp_mean_path_length if gp_mean_path_length > 1e-6 else 0.0
        path_progress = min(1.0, raw)

        # 激活终端约束的条件：路径进度 >= 90% 或 距离终点很近（< 0.02m）
        terminal_index = None
        dist_to_goal = np.linalg.norm(current_position - actual_end_mean[:dimension])
        if enable_terminal_constraint and (path_progress >= 0.9 or dist_to_goal < 0.02) and remaining_steps <= mpc.settings.n_horizon:
            terminal_index = max(0, mpc.settings.n_horizon - remaining_steps)

        # 更新previous_position用于下一步
        previous_position = current_position.copy()

        # 重规划：在预定步数仅「重观测当前状态」并用同一 GP 轨迹的剩余段 ref[k:]；
        # 训练数据不变故不重新查 GP，预测一次 GP 轨迹即可。
        t_gp_start = time.time()
        if num_replanning > 0 and k in replan_steps:
            replan_count += 1
            remaining = num_steps - k
            if observation_noise_std > 0:
                observed_goal = np.asarray(actual_end_mean).flatten().copy()
                observed_goal[:dimension] += np.random.randn(dimension) * observation_noise_std
            if replan_count <= 3 or (k + 1) % max(1, n_steps // 5) == 0:
                print(f"   [步骤 {k}] 重规划 #{replan_count}：重观测当前状态，沿用同一 GP 轨迹剩余 {remaining} 步")

        # 用预计算参考更新 TVP；MPC 的终点/终端约束用 observed_goal（可能带噪），误差仍按 actual_end_mean 算
        update_mpc_from_precomputed_ref(
            mpc,
            ref_mean_traj,
            ref_var_traj,
            current_step=k,
            num_steps=num_steps,
            goal_mean=observed_goal,
            goal_variance=goal_variance,
            actual_end_mean=observed_goal,
            alpha_threshold=alpha_threshold,
            terminal_index=terminal_index,
            enable_terminal_constraint=enable_terminal_constraint,
        )
        gp_times.append(time.time() - t_gp_start)

        # ========== MPC 求解 ==========
        # 观测噪声：仅对 MPC 看到的“观测”加噪，真实状态 x_current 用于仿真与记录，不扰动
        x_observed = x_current.copy()
        if observation_noise_std > 0:
            noise = np.random.randn(dimension) * observation_noise_std
            pos = x_current[:dimension].flatten() + noise
            x_observed[:dimension] = pos.reshape(x_current[:dimension].shape)
        t_mpc_start = time.time()
        u0 = mpc.make_step(x_observed)
        t_mpc_end = time.time()
        mpc_times.append(t_mpc_end - t_mpc_start)

        # ========== 仿真器步进 ==========
        y_next = simulator.make_step(u0)

        # ========== 状态更新 ==========
        # 真实场景：只能观测到位置，状态就是位置（一阶系统）
        # 从仿真器获取状态（现在只有位置）
        x_current = y_next

        # ========== 数据记录 ==========
        actual_trajectory.append(
            x_current.flatten())  # 位置（所有维度，现在状态就是位置）

        # 使用预计算参考轨迹中当前步对应的 GP 均值/方差（用于可视化，与 MPC 一致）
        idx_ref = min(k + 1, num_steps)  # 步 k 结束后对应参考点 k+1
        gp_mean = np.asarray(ref_mean_traj[idx_ref]).flatten()
        gp_variance = np.asarray(ref_var_traj[idx_ref]).flatten()
        if len(gp_mean) < dimension:
            gp_mean = np.pad(gp_mean, (0, dimension - len(gp_mean)), 'constant')
        if len(gp_variance) < dimension:
            gp_variance = np.pad(gp_variance, (0, dimension - len(gp_variance)),
                                 'constant', constant_values=1e-6)
        gp_mean_trajectory.append(gp_mean)
        gp_variance_history.append(gp_variance)

        # 计算权重（用于可视化，与MPC中的公式一致）
        trace_sigma = np.sum(gp_variance)
        # 使用传入的 alpha_threshold 参数
        alpha_k = alpha_threshold / (alpha_threshold + trace_sigma)

        # 添加基于距离终点的权重因子（平方关系，类似重力）
        # 当接近终点时，减小alpha（更强调目标项）
        # 计算当前位置（GP均值）到终点的欧式距离
        distance_to_goal = np.linalg.norm(gp_mean[:dimension] -
                                          actual_end_mean[:dimension])

        # 改进的距离因子：使用平滑的过渡函数
        # distance_scale 控制影响范围（距离小于此值时开始显著影响）
        # transition_zone 控制过渡区域的宽度（在此距离内alpha平滑过渡到0）
        distance_scale = 0.02  # 可调参数：开始影响的距离阈值（单位：米）
        transition_zone = 0.01  # 可调参数：过渡区域宽度（单位：米），在此距离内alpha平滑过渡到0

        # 使用改进的过渡函数：
        # - 当 distance > distance_scale 时，alpha基本不变（影响很小）
        # - 当 distance_scale - transition_zone < distance <= distance_scale 时，alpha开始减小
        # - 当 distance <= distance_scale - transition_zone 时，alpha = 0（完全强调目标项）
        if distance_to_goal > distance_scale:
            # 距离较远，影响很小
            distance_factor = 0.0
        elif distance_to_goal <= (distance_scale - transition_zone):
            # 距离很近，alpha完全为0
            distance_factor = 1.0
        else:
            # 过渡区域：使用平滑的插值函数
            # 归一化距离：[0, 1]，0表示在distance_scale处，1表示在distance_scale-transition_zone处
            normalized_dist = (distance_scale -
                               distance_to_goal) / transition_zone
            # 使用平方函数实现平滑过渡（类似重力）
            distance_factor = normalized_dist**2

        # 应用距离因子：alpha_new = alpha * (1 - distance_factor)
        # 当distance_factor=1时，alpha_k=0（完全强调目标项）
        # 当distance_factor=0时，alpha_k不变（保持原值）
        alpha_k = alpha_k * (1.0 - distance_factor)

        alpha_k = np.clip(alpha_k, 0.00, 1.0)
        alpha_history.append(alpha_k)

        control_inputs.append(u0.flatten())
        time_history.append(current_time)

        # 时间更新
        current_time += mpc.settings.t_step

        # 进度显示
        if (k + 1) % 20 == 0:
            # 确保维度匹配
            gp_mean_aligned = gp_mean[:dimension] if len(
                gp_mean) >= dimension else np.pad(gp_mean,
                                                  (0, dimension -
                                                   len(gp_mean)), 'constant')
            pos_error = np.linalg.norm(x_current[:dimension].flatten() -
                                       gp_mean_aligned)
            avg_gp_time = np.mean(gp_times[-20:]) * 1000  # 最近20步的平均GP时间（ms）
            avg_mpc_time = np.mean(mpc_times[-20:]) * 1000  # 最近20步的平均MPC时间（ms）
            # 显示当前alpha值（用于调试）
            current_alpha = alpha_history[-1] if len(
                alpha_history) > 0 else 0.0
            trace_var = np.sum(gp_variance)
            # 计算实际轨迹与GP均值轨迹的偏差
            if len(actual_trajectory) >= 2:
                # 计算最近几步的轨迹方向
                recent_positions = np.array(
                    actual_trajectory[-min(5, len(actual_trajectory)):])
                if len(recent_positions) >= 2:
                    trajectory_direction = recent_positions[
                        -1] - recent_positions[0]
                    straight_direction = actual_end_mean[:
                                                         dimension] - recent_positions[
                                                             0][:dimension]
                    # 计算方向相似度（cosine similarity）
                    if np.linalg.norm(
                            trajectory_direction[:dimension]
                    ) > 1e-6 and np.linalg.norm(straight_direction) > 1e-6:
                        direction_similarity = np.dot(
                            trajectory_direction[:dimension],
                            straight_direction
                        ) / (np.linalg.norm(trajectory_direction[:dimension]) *
                             np.linalg.norm(straight_direction))
                    else:
                        direction_similarity = 1.0
                else:
                    direction_similarity = 1.0
            else:
                direction_similarity = 1.0

            print(f"   步骤 {k+1}/{n_steps}: 时间={current_time:.2f}s, "
                  f"位置误差={pos_error:.3f}m, "
                  f"GP={avg_gp_time:.2f}ms, MPC={avg_mpc_time:.2f}ms, "
                  f"α={current_alpha:.3f}, trace_var={trace_var:.4f}, "
                  f"路径进度={path_progress:.2%} (>=90%激活终点约束)")

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
    print(
        f"   GP占比: {np.sum(gp_times) / (np.sum(gp_times) + np.sum(mpc_times)) * 100:.1f}%"
    )
    print(
        f"   MPC占比: {np.sum(mpc_times) / (np.sum(gp_times) + np.sum(mpc_times)) * 100:.1f}%"
    )

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
    # 确保所有数组长度一致（它们都应该有 n_steps+1 个点，包括初始状态）
    # 如果长度不一致，取最小长度
    min_length = min(len(actual_trajectory), len(gp_mean_trajectory),
                     len(time_history), len(gp_variance_history),
                     len(alpha_history), len(control_inputs))
    if min_length < len(actual_trajectory):
        actual_trajectory = actual_trajectory[:min_length]
    if min_length < len(gp_mean_trajectory):
        gp_mean_trajectory = gp_mean_trajectory[:min_length]
    if min_length < len(gp_variance_history):
        gp_variance_history = gp_variance_history[:min_length]
    if min_length < len(alpha_history):
        alpha_history = alpha_history[:min_length]
    if min_length < len(control_inputs):
        control_inputs = control_inputs[:min_length]
    if min_length < len(time_history):
        time_history = time_history[:min_length]

    visualize_prob_results(np.array(actual_trajectory),
                           np.array(gp_mean_trajectory),
                           np.array(gp_variance_history),
                           np.array(alpha_history),
                           np.array(control_inputs),
                           np.array(time_history),
                           all_trajectories,
                           gp_traj,
                           trajectory_duration,
                           reference_target_end=gp_end_mean,
                           dimension=dimension)

    print("\n" + "=" * 60)
    print("概率模型轨迹跟踪 MPC 仿真完成！")
    print("=" * 60)


def visualize_prob_results(actual_traj: np.ndarray,
                           gp_mean_traj: np.ndarray,
                           gp_variance_history: np.ndarray,
                           alpha_history: np.ndarray,
                           control_inputs: np.ndarray,
                           time_history: np.ndarray,
                           all_trajectories: List[Tuple[np.ndarray,
                                                        np.ndarray]],
                           gp_trajectory: GaussianProcessTrajectory,
                           trajectory_duration: float,
                           reference_target_end: np.ndarray = None,
                           dimension: int = 3):
    """
    可视化概率模型仿真结果
    
    Args:
        actual_traj: 实际执行轨迹 [N, dimension]
        gp_mean_traj: GP均值轨迹 [N, dimension]
        gp_variance_history: GP方差历史 [N, dimension]
        alpha_history: 权重历史 [N]
        control_inputs: 控制输入历史 [N, dimension]
        time_history: 时间历史 [N]
        all_trajectories: 所有生成的参考轨迹列表
        gp_trajectory: GP轨迹模型
        trajectory_duration: 轨迹持续时间
        reference_target_end: 参考目标终点 [dimension]
        dimension: 状态空间维度（默认3）
    """
    # 创建两个figure：第一个只有3D交互图，第二个包含所有其他图
    num_extra_dims = max(0, dimension - 3)

    # Figure 1: 只有3D交互图
    fig1 = plt.figure(figsize=(12, 10))

    # Figure 2: 所有其他图（XY投影、位置误差、GP方差、权重α、控制输入、额外维度）
    # 计算需要的子图数量：5个固定图（XY投影、误差、方差、alpha、控制输入）+ 额外维度
    num_plots_fig2 = 5 + num_extra_dims
    n_rows2 = (num_plots_fig2 + 2) // 3  # 每行3个图
    n_cols2 = 3
    fig2 = plt.figure(figsize=(18, 6 * n_rows2))

    # ========== Figure 1: 只有3D交互图 ==========
    # ========== 3D Trajectory Plot (只显示前3维 x, y, z) ==========
    ax1 = fig1.add_subplot(1, 1, 1, projection='3d')

    # Plot all reference trajectories (示例轨迹)
    if all_trajectories is not None:
        for i, (traj, _) in enumerate(all_trajectories):
            if i == 0:
                ax1.plot(traj[:, 0],
                         traj[:, 1],
                         traj[:, 2],
                         'gray',
                         linewidth=1,
                         alpha=0.3,
                         label='Example Trajectories')
            else:
                ax1.plot(traj[:, 0],
                         traj[:, 1],
                         traj[:, 2],
                         'gray',
                         linewidth=1,
                         alpha=0.3)
        if len(all_trajectories) > 1:
            ax1.plot([], [],
                     'gray',
                     linewidth=1,
                     alpha=0.3,
                     label=f'Example Trajectories ({len(all_trajectories)})')

    # GP均值轨迹（带不确定性区间）
    ax1.plot(gp_mean_traj[:, 0],
             gp_mean_traj[:, 1],
             gp_mean_traj[:, 2],
             'b-',
             linewidth=2,
             alpha=0.8,
             label='GP Mean Trajectory')

    # 绘制不确定性区间（±2σ）
    std_traj = np.sqrt(gp_variance_history)
    # 在x方向的不确定性
    ax1.plot(gp_mean_traj[:, 0] + 2 * std_traj[:, 0],
             gp_mean_traj[:, 1],
             gp_mean_traj[:, 2],
             'b--',
             linewidth=1,
             alpha=0.3,
             label='±2σ Uncertainty')
    ax1.plot(gp_mean_traj[:, 0] - 2 * std_traj[:, 0],
             gp_mean_traj[:, 1],
             gp_mean_traj[:, 2],
             'b--',
             linewidth=1,
             alpha=0.3)

    # Actual executed trajectory
    ax1.plot(actual_traj[:, 0],
             actual_traj[:, 1],
             actual_traj[:, 2],
             'r-',
             linewidth=2,
             label='MPC Actual Trajectory')

    # Start and end points
    ax1.scatter(actual_traj[0, 0],
                actual_traj[0, 1],
                actual_traj[0, 2],
                c='green',
                s=100,
                marker='o',
                label='Start')

    if reference_target_end is not None:
        ax1.scatter(reference_target_end[0],
                    reference_target_end[1],
                    reference_target_end[2],
                    c='blue',
                    s=150,
                    marker='*',
                    label='End (GP Goal Mean)',
                    edgecolors='black',
                    linewidths=2,
                    zorder=12)

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
    ax1.set_title('3D Trajectory Tracking (Probabilistic)')
    ax1.legend()
    ax1.grid(True)

    # ========== Figure 2: 所有其他图 ==========
    # ========== X-Y Plane Projection (只显示前2维 x, y) ==========
    ax2 = fig2.add_subplot(n_rows2, n_cols2, 1)

    if all_trajectories is not None:
        for i, (traj, _) in enumerate(all_trajectories):
            ax2.plot(traj[:, 0], traj[:, 1], 'gray', linewidth=1, alpha=0.3)

    ax2.plot(gp_mean_traj[:, 0],
             gp_mean_traj[:, 1],
             'b-',
             linewidth=2,
             alpha=0.8,
             label='GP Mean')
    ax2.plot(actual_traj[:, 0],
             actual_traj[:, 1],
             'r-',
             linewidth=2,
             label='Actual')

    ax2.scatter(actual_traj[0, 0],
                actual_traj[0, 1],
                c='green',
                s=100,
                marker='o',
                label='Start')
    if reference_target_end is not None:
        ax2.scatter(reference_target_end[0],
                    reference_target_end[1],
                    c='blue',
                    s=150,
                    marker='*',
                    label='End (GP Goal)',
                    edgecolors='black',
                    linewidths=2,
                    zorder=12)
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
    ax3 = fig2.add_subplot(n_rows2, n_cols2, 2)
    position_error = np.linalg.norm(actual_traj - gp_mean_traj, axis=1)
    ax3.plot(time_history, position_error, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Tracking Error (All Dimensions)')
    ax3.grid(True)

    # ========== GP Variance (Uncertainty) ==========
    ax4 = fig2.add_subplot(n_rows2, n_cols2, 3)
    total_variance = np.sum(gp_variance_history[:, :dimension], axis=1)
    ax4.plot(time_history,
             total_variance,
             'b-',
             linewidth=2,
             label='Total Variance')
    # 只显示前3维的方差
    dim_names = ['x', 'y', 'z'
                 ] + [chr(ord('a') + i) for i in range(max(0, dimension - 3))]
    if dimension > 26:
        dim_names = ['x', 'y', 'z'] + [f'd{i}' for i in range(3, dimension)]
    colors = ['r', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'pink']
    for i in range(min(3, dimension)):
        if i < gp_variance_history.shape[1]:
            ax4.plot(time_history,
                     gp_variance_history[:, i],
                     colors[i % len(colors)] + '--',
                     linewidth=1,
                     alpha=0.7,
                     label=f'σ²_{dim_names[i]}')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Variance')
    ax4.set_title('GP Variance (Uncertainty) - First 3 Dims')
    ax4.legend()
    ax4.grid(True)

    # ========== Weight α(t) ==========
    ax5 = fig2.add_subplot(n_rows2, n_cols2, 4)
    ax5.plot(time_history, alpha_history, 'g-', linewidth=2, label='α(t)')
    # 添加参考线：alpha=0.5（平衡点）
    if len(time_history) > 0:
        ax5.axhline(y=0.5,
                    color='r',
                    linestyle='--',
                    linewidth=1,
                    alpha=0.5,
                    label='Balance (0.5)')
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

    # ========== Control Input (前3维) ==========
    ax6 = fig2.add_subplot(n_rows2, n_cols2, 5)
    colors_ctrl = [
        'r', 'g', 'b', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'pink'
    ]
    for i in range(min(3, dimension)):
        if i < control_inputs.shape[1]:
            ax6.plot(time_history,
                     control_inputs[:, i],
                     colors_ctrl[i % len(colors_ctrl)] + '-',
                     label=f'u_{dim_names[i]}',
                     linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Desired Velocity (m/s)')
    ax6.set_title('Control Input (First 3 Dimensions)')
    ax6.legend()
    ax6.grid(True)

    # ========== Figure 2: 额外维度的可视化 ==========
    if num_extra_dims > 0:
        plot_idx = 6  # 从第6个位置开始（前5个位置已被占用）
        for dim_idx in range(3, dimension):
            row = (plot_idx - 1) // n_cols2 + 1
            col = (plot_idx - 1) % n_cols2 + 1
            ax = fig2.add_subplot(n_rows2, n_cols2, plot_idx)

            # 绘制该维度的轨迹
            if dim_idx < actual_traj.shape[1]:
                ax.plot(time_history,
                        actual_traj[:, dim_idx],
                        'r-',
                        linewidth=2,
                        label=f'Actual (dim {dim_idx})')
            if dim_idx < gp_mean_traj.shape[1]:
                ax.plot(time_history,
                        gp_mean_traj[:, dim_idx],
                        'b--',
                        linewidth=1.5,
                        alpha=0.7,
                        label=f'GP Mean (dim {dim_idx})')

            # 绘制该维度的方差
            if dim_idx < gp_variance_history.shape[1]:
                ax2_twin = ax.twinx()
                ax2_twin.plot(time_history,
                              gp_variance_history[:, dim_idx],
                              'g-',
                              linewidth=1.5,
                              alpha=0.6,
                              label=f'Variance (dim {dim_idx})')
                ax2_twin.set_ylabel('Variance', color='g')
                ax2_twin.tick_params(axis='y', labelcolor='g')

            # 绘制该维度的控制输入
            if dim_idx < control_inputs.shape[1]:
                if dim_idx < gp_variance_history.shape[1]:
                    # 如果已经有twinx，使用另一个twinx
                    ax3_twin = ax.twinx()
                    ax3_twin.spines['right'].set_position(('outward', 60))
                    ax3_twin.plot(time_history,
                                  control_inputs[:, dim_idx],
                                  'm-',
                                  linewidth=1.5,
                                  alpha=0.6,
                                  label=f'Control (dim {dim_idx})')
                    ax3_twin.set_ylabel('Desired Velocity (m/s)', color='m')
                    ax3_twin.tick_params(axis='y', labelcolor='m')
                else:
                    ax2_twin = ax.twinx()
                    ax2_twin.plot(time_history,
                                  control_inputs[:, dim_idx],
                                  'm-',
                                  linewidth=1.5,
                                  alpha=0.6,
                                  label=f'Control (dim {dim_idx})')
                    ax2_twin.set_ylabel('Desired Velocity (m/s)', color='m')
                    ax2_twin.tick_params(axis='y', labelcolor='m')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Position (dim {dim_idx})', color='b')
            ax.set_title(
                f'Dimension {dim_idx} ({dim_names[dim_idx] if dim_idx < len(dim_names) else f"d{dim_idx}"})'
            )
            ax.tick_params(axis='y', labelcolor='b')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            if dim_idx < gp_variance_history.shape[1]:
                ax2_twin.legend(loc='upper right')
            if dim_idx < control_inputs.shape[
                    1] and dim_idx < gp_variance_history.shape[1]:
                ax3_twin.legend(loc='lower right')
            elif dim_idx < control_inputs.shape[1]:
                ax2_twin.legend(loc='upper right')

            plot_idx += 1

    # 保存和显示Figure 1（3D交互图）
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig('probabilistic_trajectory_tracking_result_3d.png',
                dpi=150,
                bbox_inches='tight')
    print(
        "   Results (3D interactive plot) saved to: probabilistic_trajectory_tracking_result_3d.png"
    )
    plt.show()

    # 保存和显示Figure 2（所有其他图）
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig('probabilistic_trajectory_tracking_result_all_plots.png',
                dpi=150,
                bbox_inches='tight')
    print(
        "   Results (all other plots) saved to: probabilistic_trajectory_tracking_result_all_plots.png"
    )
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='概率模型轨迹跟踪 MPC 仿真')
    parser.add_argument('--num-replanning',
                        type=int,
                        default=0,
                        help='重新规划次数（概率模型中，重规划仅体现为时间推进）')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Debug模式：GP训练仅进行10次迭代')
    parser.add_argument('--trajectory-type',
                        type=str,
                        default='rollercoaster',
                        choices=['spiral', 'rollercoaster'],
                        help='轨迹类型: spiral (螺旋轨迹) 或 rollercoaster (过山车轨迹)')
    parser.add_argument('--dimension',
                        type=int,
                        default=3,
                        help='状态空间维度（默认3，表示3D空间）')
    parser.add_argument('--estimate-velocity',
                        action='store_true',
                        help='是否估计速度（已废弃，保留以兼容接口）')
    parser.add_argument(
        '--alpha-threshold',
        type=float,
        default=0.0001,
        help=
        'Alpha权重计算的阈值参数（默认0.0001）。较小的值：alpha更小，更强调目标项，终点误差更小；较大的值：alpha更大，更强调轨迹跟踪'
    )
    parser.add_argument('--use-real-trajectory',
                        action='store_true',
                        help='使用真实轨迹（从H5数据集加载）而不是生成轨迹')
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='数据集路径（当使用真实轨迹时必需），例如: datasets/0122/20260122_182659')
    parser.add_argument('--num-steps',
                        type=int,
                        default=100,
                        help='仿真步数（默认100）')
    parser.add_argument('--observation-noise-std',
                        type=float,
                        default=0.0,
                        help='观测噪声标准差（同时作用于当前位置与终点；每次replan时对观测终点加噪；误差仍按真实终点算；0表示无噪声）')

    args = parser.parse_args()
    main(num_replanning=args.num_replanning,
         debug=args.debug,
         trajectory_type=args.trajectory_type,
         dimension=args.dimension,
         estimate_velocity=args.estimate_velocity,
         alpha_threshold=args.alpha_threshold,
         use_real_trajectory=args.use_real_trajectory,
         dataset_path=args.dataset_path,
         num_steps=args.num_steps,
         observation_noise_std=args.observation_noise_std)
