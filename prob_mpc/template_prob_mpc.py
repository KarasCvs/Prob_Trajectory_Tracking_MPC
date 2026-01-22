"""
概率模型MPC控制器配置
实现基于高斯过程概率轨迹的MPC控制器
核心：使用Mahalanobis距离和权重调度实现自适应轨迹跟踪
"""
import numpy as np
import casadi as ca
import do_mpc


def template_prob_mpc(model, gp_trajectory=None, silence_solver=False):
    """
    概率模型MPC配置
    
    - **Mahalanobis距离** cost：基于GP均值和方差的轨迹跟踪
    - **权重调度**：根据不确定性动态平衡轨迹项和目标项
    - **Reachability guarantee**：终点等式约束强制 p(N)=μ_T，v(N)=0
    
    Args:
        model: 概率模型（template_prob_model）
        gp_trajectory: GaussianProcessTrajectory实例（可选，用于获取目标信息）
        silence_solver: 是否静默求解器输出
    """
    mpc = do_mpc.controller.MPC(model)

    # === MPC 设置 ===
    mpc.settings.n_robust = 0
    mpc.settings.n_horizon = 30
    mpc.settings.t_step = 0.1
    mpc.settings.store_full_solution = True

    if silence_solver:
        mpc.settings.supress_ipopt_output()

    # === Mahalanobis距离 Cost ===
    # 轨迹项：ℓ_traj = (x_p - μ(t))^T Σ(t)^(-1) (x_p - μ(t))
    mahalanobis_traj_x = model.aux['mahalanobis_traj_x']
    mahalanobis_traj_y = model.aux['mahalanobis_traj_y']
    mahalanobis_traj_z = model.aux['mahalanobis_traj_z']
    mahalanobis_traj = (
        mahalanobis_traj_x + 
        mahalanobis_traj_y + 
        mahalanobis_traj_z
    )
    
    # 目标项：ℓ_goal = (x_p - μ_T)^T Σ_T^(-1) (x_p - μ_T)
    mahalanobis_goal_x = model.aux['mahalanobis_goal_x']
    mahalanobis_goal_y = model.aux['mahalanobis_goal_y']
    mahalanobis_goal_z = model.aux['mahalanobis_goal_z']
    mahalanobis_goal = (
        mahalanobis_goal_x + 
        mahalanobis_goal_y + 
        mahalanobis_goal_z
    )
    
    # 权重调度：α(t) 从TVP获取
    alpha = model.tvp['alpha_weight']
    
    # 综合Stage Cost：ℓ(t) = α(t) * ℓ_traj + (1-α(t)) * ℓ_goal
    lterm = alpha * mahalanobis_traj + (1 - alpha) * mahalanobis_goal
    
    # 添加速度跟踪项（可选，用于平滑性）
    vel_error_x = model.aux['vel_error_x']
    vel_error_y = model.aux['vel_error_y']
    vel_error_z = model.aux['vel_error_z']
    Q_vel = 0.5  # 速度权重（相对较小，主要靠位置项）
    lterm += Q_vel * (vel_error_x**2 + vel_error_y**2 + vel_error_z**2)

    # do-mpc 要求 mterm 形状为 (1,1)，这里设为 0
    mterm = 0 * lterm
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(a_x=0.1, a_y=0.1, a_z=0.1)

    # === state/input bounds ===
    max_pos = 10.0
    mpc.bounds['lower', '_x', 'p_x'] = -max_pos
    mpc.bounds['upper', '_x', 'p_x'] = max_pos
    mpc.bounds['lower', '_x', 'p_y'] = -max_pos
    mpc.bounds['upper', '_x', 'p_y'] = max_pos
    mpc.bounds['lower', '_x', 'p_z'] = -max_pos
    mpc.bounds['upper', '_x', 'p_z'] = max_pos
    max_vel = 5.0
    mpc.bounds['lower', '_x', 'v_x'] = -max_vel
    mpc.bounds['upper', '_x', 'v_x'] = max_vel
    mpc.bounds['lower', '_x', 'v_y'] = -max_vel
    mpc.bounds['upper', '_x', 'v_y'] = max_vel
    mpc.bounds['lower', '_x', 'v_z'] = -max_vel
    mpc.bounds['upper', '_x', 'v_z'] = max_vel
    max_acc = 2.0
    mpc.bounds['lower', '_u', 'a_x'] = -max_acc
    mpc.bounds['upper', '_u', 'a_x'] = max_acc
    mpc.bounds['lower', '_u', 'a_y'] = -max_acc
    mpc.bounds['upper', '_u', 'a_y'] = max_acc
    mpc.bounds['lower', '_u', 'a_z'] = -max_acc
    mpc.bounds['upper', '_u', 'a_z'] = max_acc

    # TVP 模板（主循环填充具体参考）
    mpc._tvp_template = mpc.get_tvp_template()
    mpc.set_tvp_fun(lambda t_now: mpc._tvp_template)

    # === Terminal equality constraint（硬约束） ===
    def _active(expr, ref):
        return expr * (ref < 1e19)

    terminal_expr = _active(model.x['p_x'] - model.tvp['p_x_ref_terminal'],
                            model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_p_x_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_p_x_neg', -terminal_expr, ub=0.0, soft_constraint=False)

    terminal_expr = _active(model.x['p_y'] - model.tvp['p_y_ref_terminal'],
                            model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_p_y_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_p_y_neg', -terminal_expr, ub=0.0, soft_constraint=False)

    terminal_expr = _active(model.x['p_z'] - model.tvp['p_z_ref_terminal'],
                            model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_p_z_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_p_z_neg', -terminal_expr, ub=0.0, soft_constraint=False)

    terminal_expr = _active(model.x['v_x'], model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_v_x_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_v_x_neg', -terminal_expr, ub=0.0, soft_constraint=False)
    terminal_expr = _active(model.x['v_y'], model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_v_y_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_v_y_neg', -terminal_expr, ub=0.0, soft_constraint=False)
    terminal_expr = _active(model.x['v_z'], model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_v_z_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_v_z_neg', -terminal_expr, ub=0.0, soft_constraint=False)

    mpc.setup()
    return mpc


def update_mpc_gp_trajectory(
    mpc,
    gp_trajectory,
    current_time_normalized: float,
    terminal_index: int = None,
    epsilon: float = 1e-6,
    trajectory_duration: float = 1.0,
    actual_end_mean: np.ndarray = None
):
    """
    更新 MPC 的GP轨迹（时变参数）
    
    在每个控制周期调用，为 MPC 预测时域内的每一步查询GP得到均值/方差，
    计算权重α，填充TVP模板
    
    Args:
        mpc: MPC 控制器对象
        gp_trajectory: GaussianProcessTrajectory实例
        current_time_normalized: 当前归一化时间 [0,1]
        terminal_index: 终端约束生效的索引（如果为None，则不激活终端约束）
        epsilon: 权重计算的小量（避免除零）
        trajectory_duration: 轨迹持续时间（用于将归一化时间转换回原始时间，用于速度计算）
        actual_end_mean: 实际终点均值（如果提供，将用于终端约束和目标项，而不是GP预测值）
    """
    # 获取时变参数模板
    tvp_template = mpc._tvp_template
    
    horizon = mpc.settings.n_horizon
    
    # 获取目标均值和方差（用于目标项和终端约束）
    # 如果提供了actual_end_mean，使用它（更准确）；否则使用GP预测值
    if actual_end_mean is not None:
        goal_mean = actual_end_mean
    else:
        goal_mean = gp_trajectory.get_goal_mean()
    goal_variance = gp_trajectory.get_goal_variance()
    
    # 计算归一化时间步长
    dt_normalized = (mpc.settings.t_step / trajectory_duration) if trajectory_duration > 1e-6 else mpc.settings.t_step
    
    # ========== 性能优化：批量预测所有时间点 ==========
    # 准备所有预测时间点（horizon + 1个，因为需要计算速度）
    times_normalized = []
    for k in range(horizon + 1):
        t_k = current_time_normalized + k * dt_normalized
        t_k = np.clip(t_k, 0.0, 1.0)
        times_normalized.append(t_k)
    
    times_normalized = np.array(times_normalized)
    
    # 批量预测所有时间点的均值和方差（一次性完成，大幅提升性能）
    means, variances = gp_trajectory.predict_mean_and_variance_batch(times_normalized)
    
    # 填充TVP模板
    for k in range(horizon):
        mean_k = means[k]
        variance_k = variances[k]
        
        # 设置参考位置（GP均值）
        tvp_template['_tvp', k, 'p_x_ref'] = mean_k[0]
        tvp_template['_tvp', k, 'p_y_ref'] = mean_k[1]
        tvp_template['_tvp', k, 'p_z_ref'] = mean_k[2]
        
        # 设置GP方差（对角）
        # 确保方差有下界，避免除零
        tvp_template['_tvp', k, 'sigma_x_sq'] = max(variance_k[0], 1e-6)
        tvp_template['_tvp', k, 'sigma_y_sq'] = max(variance_k[1], 1e-6)
        tvp_template['_tvp', k, 'sigma_z_sq'] = max(variance_k[2], 1e-6)
        
        # 计算权重 α(t) = threshold / (threshold + trace(Σ(t)))
        # 方差小（不确定性低）→ alpha接近1（强调轨迹跟踪）
        # 方差大（不确定性高）→ alpha接近0（强调目标吸引）
        trace_sigma = np.sum(variance_k)
        # 使用合理的阈值（基于数据变异性，约0.1-0.3）
        # 当trace_sigma = threshold时，alpha = 0.5（平衡点）
        threshold = 0.1  # 可调参数：控制alpha的敏感度
        alpha_k = threshold / (threshold + trace_sigma)
        # Clamp权重到合理范围
        alpha_k = np.clip(alpha_k, 0.01, 1.0)
        tvp_template['_tvp', k, 'alpha_weight'] = alpha_k
        
        # 设置目标均值和方差（用于目标项）
        tvp_template['_tvp', k, 'mu_goal_x'] = goal_mean[0]
        tvp_template['_tvp', k, 'mu_goal_y'] = goal_mean[1]
        tvp_template['_tvp', k, 'mu_goal_z'] = goal_mean[2]
        tvp_template['_tvp', k, 'sigma_goal_x_sq'] = max(goal_variance[0], 1e-6)
        tvp_template['_tvp', k, 'sigma_goal_y_sq'] = max(goal_variance[1], 1e-6)
        tvp_template['_tvp', k, 'sigma_goal_z_sq'] = max(goal_variance[2], 1e-6)
        
        # 计算参考速度（通过数值微分，使用批量预测的结果）
        if k < horizon - 1:
            mean_k_next = means[k + 1]
            # 速度 = 位置差 / 时间差（使用原始时间单位）
            v_ref = (mean_k_next - mean_k) / mpc.settings.t_step
        else:
            # 最后一步，速度设为0（接近终点）
            v_ref = np.zeros(3)
        
        tvp_template['_tvp', k, 'v_x_ref'] = v_ref[0]
        tvp_template['_tvp', k, 'v_y_ref'] = v_ref[1]
        tvp_template['_tvp', k, 'v_z_ref'] = v_ref[2]
        
        # 终端等式约束只在对齐"仿真终点"的那一步生效：其余步填哨兵值（不生效）
        if terminal_index is not None and k == terminal_index:
            tvp_template['_tvp', k, 'p_x_ref_terminal'] = goal_mean[0]
            tvp_template['_tvp', k, 'p_y_ref_terminal'] = goal_mean[1]
            tvp_template['_tvp', k, 'p_z_ref_terminal'] = goal_mean[2]
        else:
            tvp_template['_tvp', k, 'p_x_ref_terminal'] = 1e20
            tvp_template['_tvp', k, 'p_y_ref_terminal'] = 1e20
            tvp_template['_tvp', k, 'p_z_ref_terminal'] = 1e20
    
    # TVP 函数已在初始化时设置，这里只更新模板内容
