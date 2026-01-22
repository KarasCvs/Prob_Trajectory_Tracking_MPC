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
    概率模型MPC配置（一阶系统）
    
    - **Mahalanobis距离** cost：基于GP均值和方差的轨迹跟踪
    - **权重调度**：根据不确定性动态平衡轨迹项和目标项
    - **控制输入惩罚**：u的惩罚用于平滑性（λ||u||²）
    - **Reachability guarantee**：终点等式约束强制 p(N)=μ_T（只约束位置）
    
    注意：系统是一阶的（p_dot = u），天然是梯度下降，不会产生旋转
    
    Args:
        model: 概率模型（template_prob_model）
        gp_trajectory: GaussianProcessTrajectory实例（可选，用于获取目标信息）
        silence_solver: 是否静默求解器输出
    """
    # 获取维度信息
    dimension = getattr(model, 'dimension', 3)
    dim_names = getattr(model, 'dim_names', ['x', 'y', 'z'])
    
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
    mahalanobis_traj_terms = []
    for dim_name in dim_names:
        mahalanobis_traj_terms.append(model.aux[f'mahalanobis_traj_{dim_name}'])
    mahalanobis_traj = sum(mahalanobis_traj_terms)
    
    # 目标项：ℓ_goal = (x_p - μ_T)^T Σ_T^(-1) (x_p - μ_T)
    mahalanobis_goal_terms = []
    for dim_name in dim_names:
        mahalanobis_goal_terms.append(model.aux[f'mahalanobis_goal_{dim_name}'])
    mahalanobis_goal = sum(mahalanobis_goal_terms)
    
    # 权重调度：α(t) 从TVP获取
    alpha = model.tvp['alpha_weight']
    
    # 综合Stage Cost：ℓ(t) = α(t) * ℓ_traj + (1-α(t)) * ℓ_goal
    lterm = alpha * mahalanobis_traj + (1 - alpha) * mahalanobis_goal
    
    # 注意：不再添加速度跟踪项，因为系统是一阶的，没有速度状态
    # 控制输入u的惩罚用于平滑性

    # do-mpc 要求 mterm 形状为 (1,1)，这里设为 0
    mterm = 0 * lterm
    mpc.set_objective(mterm=mterm, lterm=lterm)
    
    # 设置输入惩罚项（控制输入u的惩罚用于平滑性）
    rterm_dict = {}
    for dim_name in dim_names:
        rterm_dict[f'u_{dim_name}'] = 0.1  # 期望速度控制输入的惩罚（平滑性）
    mpc.set_rterm(**rterm_dict)

    # === state/input bounds ===
    max_pos = 10.0
    for dim_name in dim_names:
        mpc.bounds['lower', '_x', f'p_{dim_name}'] = -max_pos
        mpc.bounds['upper', '_x', f'p_{dim_name}'] = max_pos
    
    # 控制输入是期望速度（m/s）
    max_vel = 5.0
    for dim_name in dim_names:
        mpc.bounds['lower', '_u', f'u_{dim_name}'] = -max_vel
        mpc.bounds['upper', '_u', f'u_{dim_name}'] = max_vel

    # TVP 模板（主循环填充具体参考）
    mpc._tvp_template = mpc.get_tvp_template()
    mpc.set_tvp_fun(lambda t_now: mpc._tvp_template)

    # === Terminal equality constraint（硬约束） ===
    def _active(expr, ref):
        return expr * (ref < 1e19)

    # 位置终端约束（只约束位置，不约束速度，因为系统是一阶的）
    for dim_name in dim_names:
        terminal_expr = _active(model.x[f'p_{dim_name}'] - model.tvp[f'p_{dim_name}_ref_terminal'],
                                model.tvp[f'p_{dim_name}_ref_terminal'])
        mpc.set_nl_cons(f'terminal_p_{dim_name}_pos', terminal_expr, ub=0.0, soft_constraint=False)
        mpc.set_nl_cons(f'terminal_p_{dim_name}_neg', -terminal_expr, ub=0.0, soft_constraint=False)

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
    # 获取维度信息
    model = mpc.model
    dimension = getattr(model, 'dimension', 3)
    dim_names = getattr(model, 'dim_names', ['x', 'y', 'z'])
    
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
    
    # 确保维度匹配
    if len(goal_mean) < dimension:
        # 如果GP返回的维度不足，用0填充
        goal_mean = np.pad(goal_mean, (0, dimension - len(goal_mean)), 'constant')
    if len(goal_variance) < dimension:
        goal_variance = np.pad(goal_variance, (0, dimension - len(goal_variance)), 'constant', constant_values=1e-6)
    
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
        
        # 确保维度匹配
        if len(mean_k) < dimension:
            mean_k = np.pad(mean_k, (0, dimension - len(mean_k)), 'constant')
        if len(variance_k) < dimension:
            variance_k = np.pad(variance_k, (0, dimension - len(variance_k)), 'constant', constant_values=1e-6)
        
        # 设置参考位置（GP均值）
        for i, dim_name in enumerate(dim_names):
            if i < len(mean_k):
                tvp_template['_tvp', k, f'p_{dim_name}_ref'] = mean_k[i]
            else:
                tvp_template['_tvp', k, f'p_{dim_name}_ref'] = 0.0
        
        # 设置GP方差（对角）
        # 确保方差有下界，避免除零
        for i, dim_name in enumerate(dim_names):
            if i < len(variance_k):
                tvp_template['_tvp', k, f'sigma_{dim_name}_sq'] = max(variance_k[i], 1e-6)
            else:
                tvp_template['_tvp', k, f'sigma_{dim_name}_sq'] = 1e-6
        
        # 计算权重 α(t) = threshold / (threshold + trace(Σ(t)))
        # 方差小（不确定性低）→ alpha接近1（强调轨迹跟踪）
        # 方差大（不确定性高）→ alpha接近0（强调目标吸引）
        trace_sigma = np.sum(variance_k[:dimension])  # 只计算实际维度的方差
        # 使用合理的阈值（基于数据变异性，约0.1-0.3）
        # 当trace_sigma = threshold时，alpha = 0.5（平衡点）
        threshold = 0.1  # 可调参数：控制alpha的敏感度
        alpha_k = threshold / (threshold + trace_sigma)
        # Clamp权重到合理范围
        alpha_k = np.clip(alpha_k, 0.01, 1.0)
        tvp_template['_tvp', k, 'alpha_weight'] = alpha_k
        
        # 设置目标均值和方差（用于目标项）
        for i, dim_name in enumerate(dim_names):
            if i < len(goal_mean):
                tvp_template['_tvp', k, f'mu_goal_{dim_name}'] = goal_mean[i]
            else:
                tvp_template['_tvp', k, f'mu_goal_{dim_name}'] = 0.0
            
            if i < len(goal_variance):
                tvp_template['_tvp', k, f'sigma_goal_{dim_name}_sq'] = max(goal_variance[i], 1e-6)
            else:
                tvp_template['_tvp', k, f'sigma_goal_{dim_name}_sq'] = 1e-6
        
        # 注意：不再需要计算参考速度，因为系统是一阶的，没有速度状态
        
        # 终端等式约束只在对齐"仿真终点"的那一步生效：其余步填哨兵值（不生效）
        if terminal_index is not None and k == terminal_index:
            for i, dim_name in enumerate(dim_names):
                if i < len(goal_mean):
                    tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = goal_mean[i]
                else:
                    tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = 0.0
        else:
            for dim_name in dim_names:
                tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = 1e20
    
    # TVP 函数已在初始化时设置，这里只更新模板内容
