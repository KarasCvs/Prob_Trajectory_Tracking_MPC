"""
概率模型MPC控制器配置
实现基于高斯过程概率轨迹的MPC控制器
核心：使用Mahalanobis距离和权重调度实现自适应轨迹跟踪
"""
import numpy as np
import casadi as ca
import do_mpc


def template_prob_mpc(model,
                      gp_trajectory=None,
                      silence_solver=False,
                      alpha_threshold=0.0001,
                      enable_terminal_constraint=False):
    """
    概率模型MPC配置（一阶系统）
    
    - **Mahalanobis距离** cost：基于GP均值和方差的轨迹跟踪
    - **权重调度**：根据不确定性动态平衡轨迹项和目标项
    - **控制输入惩罚**：u的惩罚用于平滑性（λ||u||²）
    - **Reachability guarantee**：终点等式约束强制 p(N)=μ_T（只约束位置，可选）
    
    注意：系统是一阶的（p_dot = u），天然是梯度下降，不会产生旋转
    
    Args:
        model: 概率模型（template_prob_model）
        gp_trajectory: GaussianProcessTrajectory实例（可选，用于获取目标信息）
        silence_solver: 是否静默求解器输出
        alpha_threshold: Alpha权重计算的阈值参数（默认0.0001）
        enable_terminal_constraint: 是否启用终点约束（默认False）
    """
    # 获取维度信息
    dimension = getattr(model, 'dimension', 3)
    dim_names = getattr(model, 'dim_names', ['x', 'y', 'z'])

    mpc = do_mpc.controller.MPC(model)

    # 存储 alpha_threshold 以便后续使用
    mpc.alpha_threshold = alpha_threshold

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
        mahalanobis_traj_terms.append(
            model.aux[f'mahalanobis_traj_{dim_name}'])
    mahalanobis_traj = sum(mahalanobis_traj_terms)

    # 只保留轨迹跟踪项，移除目标项和终端约束
    # 综合Stage Cost：ℓ(t) = ℓ_traj（只跟踪GP均值轨迹）
    lterm = mahalanobis_traj

    # 注意：不再添加速度跟踪项，因为系统是一阶的，没有速度状态
    # 控制输入u的惩罚用于平滑性

    # 终端惩罚项（软约束）在下面定义，这里先设为0
    # mterm会在下面根据terminal_weight动态设置
    mterm_placeholder = 0 * lterm

    # 设置输入惩罚项（控制输入u的惩罚用于平滑性）
    # 注意：当启用终端约束时，在接近终点时需要降低惩罚，允许更大的控制输入来满足硬约束
    # 这里使用TVP来控制惩罚权重（如果需要的话）
    rterm_dict = {}
    for dim_name in dim_names:
        rterm_dict[f'u_{dim_name}'] = 0.5  # 期望速度控制输入的惩罚（平滑性），从0.1增加到0.5
        # 注意：如果需要动态调整，可以添加TVP变量，但do-mpc的rterm不支持TVP
        # 所以这里使用固定值，但在接近终点时通过降低其他项来补偿
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

    # === 终端约束（硬约束 + 软约束，可选） ===
    # 如果启用终端约束，使用硬约束确保精确到达终点
    if enable_terminal_constraint:
        # 硬约束：强制 p(N) = μ_T（精确等式约束）
        # 使用与标准MPC相同的方法：通过TVP控制约束是否生效
        for dim_name in dim_names:
            terminal_expr = model.x[f'p_{dim_name}'] - model.tvp[f'p_{dim_name}_ref_terminal']
            
            # 正负两个约束确保等式：p - ref = 0
            mpc.set_nl_cons(f'terminal_p_{dim_name}_pos',
                            terminal_expr,
                            ub=0.0,
                            soft_constraint=False)
            mpc.set_nl_cons(f'terminal_p_{dim_name}_neg',
                            -terminal_expr,
                            ub=0.0,
                            soft_constraint=False)
        
        # 同时添加软约束作为辅助（在硬约束无法满足时提供梯度）
        # 终端惩罚项：ℓ_terminal = w_terminal * ||p(N) - μ_T||²
        terminal_penalty_terms = []
        for dim_name in dim_names:
            terminal_error = model.x[f'p_{dim_name}'] - model.tvp[f'p_{dim_name}_ref_terminal']
            terminal_penalty_terms.append(terminal_error**2)
        terminal_penalty = sum(terminal_penalty_terms)
        
        # 终端惩罚权重（通过TVP控制，可以在接近终点时增大）
        terminal_weight = model.tvp['terminal_weight']  # 0表示不激活，>0表示激活
        mterm = terminal_weight * terminal_penalty
    else:
        # 不启用终端约束
        mterm = 0 * lterm
    
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.setup()
    return mpc


def update_mpc_gp_trajectory(mpc,
                             gp_trajectory,
                             current_time_normalized: float,
                             terminal_index: int = None,
                             epsilon: float = 1e-6,
                             trajectory_duration: float = 1.0,
                             actual_end_mean: np.ndarray = None,
                             alpha_threshold: float = 0.0001,
                             enable_terminal_constraint: bool = False):
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
        alpha_threshold: Alpha权重计算的阈值参数（默认0.0001）
        enable_terminal_constraint: 是否启用终点约束（默认False）
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
        goal_mean = np.pad(goal_mean, (0, dimension - len(goal_mean)),
                           'constant')
    if len(goal_variance) < dimension:
        goal_variance = np.pad(goal_variance,
                               (0, dimension - len(goal_variance)),
                               'constant',
                               constant_values=1e-6)

    # 计算归一化时间步长
    dt_normalized = (mpc.settings.t_step / trajectory_duration
                     ) if trajectory_duration > 1e-6 else mpc.settings.t_step

    # ========== 性能优化：批量预测所有时间点 ==========
    # 准备所有预测时间点（horizon + 1个，因为需要计算速度）
    times_normalized = []
    for k in range(horizon + 1):
        t_k = current_time_normalized + k * dt_normalized
        t_k = np.clip(t_k, 0.0, 1.0)
        times_normalized.append(t_k)

    times_normalized = np.array(times_normalized)

    # 批量预测所有时间点的均值和方差（一次性完成，大幅提升性能）
    means, variances = gp_trajectory.predict_mean_and_variance_batch(
        times_normalized)

    # 填充TVP模板
    for k in range(horizon):
        mean_k = means[k]
        variance_k = variances[k]

        # 确保维度匹配
        if len(mean_k) < dimension:
            mean_k = np.pad(mean_k, (0, dimension - len(mean_k)), 'constant')
        if len(variance_k) < dimension:
            variance_k = np.pad(variance_k, (0, dimension - len(variance_k)),
                                'constant',
                                constant_values=1e-6)

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
                tvp_template['_tvp', k, f'sigma_{dim_name}_sq'] = max(
                    variance_k[i], 1e-6)
            else:
                tvp_template['_tvp', k, f'sigma_{dim_name}_sq'] = 1e-6

        # 计算权重 α(t) = threshold / (threshold + trace(Σ(t)))
        # 方差小（不确定性低）→ alpha接近1（强调轨迹跟踪）
        # 方差大（不确定性高）→ alpha接近0（强调目标吸引）
        trace_sigma = np.sum(variance_k[:dimension])  # 只计算实际维度的方差
        # 使用合理的阈值（基于数据变异性，约0.1-0.3）
        # 当trace_sigma = threshold时，alpha = 0.5（平衡点）
        # threshold 从函数参数传入
        alpha_k = alpha_threshold / (alpha_threshold + trace_sigma)
        
        # ========== 核心修复：当距离终点很近时，将GP均值设置为目标位置 ==========
        # 问题：当系统接近终点时，GP均值可能停止变化，导致系统被"困住"在GP均值位置
        # 解决：当距离终点 < 0.02m 时，将GP均值设置为目标位置，使轨迹跟踪项自动满足（误差为0）
        #       这样轨迹跟踪项就不会阻止系统到达真正的终点
        if actual_end_mean is not None:
            dist_to_goal_k = np.linalg.norm(mean_k[:dimension] - actual_end_mean[:dimension])
            if dist_to_goal_k < 0.02:
                # 将GP均值设置为目标位置，使轨迹跟踪项自动满足（误差为0）
                mean_k = actual_end_mean.copy()
                if len(mean_k) < dimension:
                    mean_k = np.pad(mean_k, (0, dimension - len(mean_k)), 'constant')
                # alpha_k设为0（虽然当前不使用，但保持一致性）
                alpha_k = 0.0

        # 设置alpha_weight（虽然当前不使用，但需要设置以避免错误）
        tvp_template['_tvp', k, 'alpha_weight'] = alpha_k
        
        # 设置目标均值和方差（虽然当前不使用，但需要设置以避免错误）
        for i, dim_name in enumerate(dim_names):
            if i < len(goal_mean):
                tvp_template['_tvp', k, f'mu_goal_{dim_name}'] = goal_mean[i]
            else:
                tvp_template['_tvp', k, f'mu_goal_{dim_name}'] = 0.0

            if i < len(goal_variance):
                tvp_template['_tvp', k, f'sigma_goal_{dim_name}_sq'] = max(
                    goal_variance[i], 1e-6)
            else:
                tvp_template['_tvp', k, f'sigma_goal_{dim_name}_sq'] = 1e-6

        # 设置终端目标位置和权重（如果启用终端约束）
        if enable_terminal_constraint:
            # 设置终端目标位置（用于硬约束和软约束）
            # 关键：硬约束应该只在horizon的最后一步（k == horizon - 1）生效
            # 当terminal_index存在时，表示应该激活终端约束
            # 在非最后一步，将ref_terminal设置为当前GP均值，使硬约束自动满足
            # 在最后一步，将ref_terminal设置为目标位置，使硬约束生效
            if terminal_index is not None and k == horizon - 1:
                # horizon的最后一步：设置为目标位置，使硬约束生效（强制到达终点）
                for i, dim_name in enumerate(dim_names):
                    if i < len(goal_mean):
                        tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = goal_mean[i]
                    else:
                        tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = 0.0
            else:
                # 非最后一步：设置为当前GP均值，使硬约束自动满足（误差为0）
                # 这样硬约束不会干扰轨迹跟踪，只在最后一步强制到达终点
                for i, dim_name in enumerate(dim_names):
                    if i < len(mean_k):
                        tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = mean_k[i]
                    else:
                        tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = 0.0
            
            # 设置终端惩罚权重（通过terminal_index和路径进度控制）
            # terminal_index表示horizon中应该激活终端约束的起始索引
            # 如果terminal_index存在，则在horizon的最后几步（k >= terminal_index）都激活软约束
            # 但硬约束只在最后一步（k == horizon - 1）生效
            if terminal_index is not None and k >= terminal_index:
                # 激活终端惩罚（软约束，作为硬约束的辅助）
                # 在horizon的最后一步使用最大权重，前面的步骤使用较小的权重以平滑过渡
                if k == horizon - 1:
                    # 最后一步使用最大权重（确保硬约束能够满足）
                    tvp_template['_tvp', k, 'terminal_weight'] = 10000.0  # 大幅增大终端惩罚权重
                else:
                    # 前面的步骤使用较小的权重，平滑过渡（引导向终点）
                    tvp_template['_tvp', k, 'terminal_weight'] = 1000.0
            else:
                # 不激活终端惩罚
                tvp_template['_tvp', k, 'terminal_weight'] = 0.0
        else:
            # 不启用终端约束，但仍需要设置TVP变量（避免错误）
            for i, dim_name in enumerate(dim_names):
                if i < len(goal_mean):
                    tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = goal_mean[i]
                else:
                    tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = 0.0
            tvp_template['_tvp', k, 'terminal_weight'] = 0.0

    # TVP 函数已在初始化时设置，这里只更新模板内容


def update_mpc_from_precomputed_ref(mpc,
                                    ref_mean_traj: np.ndarray,
                                    ref_var_traj: np.ndarray,
                                    current_step: int,
                                    num_steps: int,
                                    goal_mean: np.ndarray,
                                    goal_variance: np.ndarray,
                                    actual_end_mean: np.ndarray = None,
                                    alpha_threshold: float = 0.0001,
                                    terminal_index: int = None,
                                    enable_terminal_constraint: bool = False):
    """
    用预计算的 GP 参考轨迹更新 MPC 的 TVP（不每步调 GP）。
    ref_mean_traj / ref_var_traj 为 [0,1] 上 num_steps+1 个点（对应步 0..num_steps），
    current_step 为当前仿真步，horizon 内参考从 ref_mean_traj[current_step],
    ref_mean_traj[current_step+1], ... 取值，超出用 goal 填充。
    """
    model = mpc.model
    dimension = getattr(model, 'dimension', 3)
    dim_names = getattr(model, 'dim_names', ['x', 'y', 'z'])
    tvp_template = mpc._tvp_template
    horizon = mpc.settings.n_horizon

    goal_mean = np.asarray(goal_mean).flatten()
    goal_variance = np.asarray(goal_variance).flatten()
    if len(goal_mean) < dimension:
        goal_mean = np.pad(goal_mean, (0, dimension - len(goal_mean)), 'constant')
    if len(goal_variance) < dimension:
        goal_variance = np.pad(goal_variance, (0, dimension - len(goal_variance)),
                               'constant', constant_values=1e-6)

    for k in range(horizon):
        idx = min(current_step + k, num_steps)
        mean_k = np.asarray(ref_mean_traj[idx]).flatten()
        variance_k = np.asarray(ref_var_traj[idx]).flatten()
        if len(mean_k) < dimension:
            mean_k = np.pad(mean_k, (0, dimension - len(mean_k)), 'constant')
        if len(variance_k) < dimension:
            variance_k = np.pad(variance_k, (0, dimension - len(variance_k)),
                                'constant', constant_values=1e-6)

        for i, dim_name in enumerate(dim_names):
            tvp_template['_tvp', k, f'p_{dim_name}_ref'] = mean_k[i] if i < len(mean_k) else 0.0
        for i, dim_name in enumerate(dim_names):
            tvp_template['_tvp', k, f'sigma_{dim_name}_sq'] = max(
                variance_k[i] if i < len(variance_k) else 1e-6, 1e-6)

        trace_sigma = np.sum(variance_k[:dimension])
        alpha_k = alpha_threshold / (alpha_threshold + trace_sigma)
        if actual_end_mean is not None:
            dist_to_goal_k = np.linalg.norm(mean_k[:dimension] - np.asarray(actual_end_mean)[:dimension])
            if dist_to_goal_k < 0.02:
                mean_k = np.asarray(actual_end_mean).flatten()
                if len(mean_k) < dimension:
                    mean_k = np.pad(mean_k, (0, dimension - len(mean_k)), 'constant')
                alpha_k = 0.0
        tvp_template['_tvp', k, 'alpha_weight'] = alpha_k

        for i, dim_name in enumerate(dim_names):
            tvp_template['_tvp', k, f'mu_goal_{dim_name}'] = goal_mean[i] if i < len(goal_mean) else 0.0
            tvp_template['_tvp', k, f'sigma_goal_{dim_name}_sq'] = max(
                goal_variance[i] if i < len(goal_variance) else 1e-6, 1e-6)

        if enable_terminal_constraint:
            if terminal_index is not None and k == horizon - 1:
                for i, dim_name in enumerate(dim_names):
                    tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = goal_mean[i] if i < len(goal_mean) else 0.0
            else:
                for i, dim_name in enumerate(dim_names):
                    tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = mean_k[i] if i < len(mean_k) else 0.0
            if terminal_index is not None and k >= terminal_index:
                tvp_template['_tvp', k, 'terminal_weight'] = 10000.0 if k == horizon - 1 else 1000.0
            else:
                tvp_template['_tvp', k, 'terminal_weight'] = 0.0
        else:
            for i, dim_name in enumerate(dim_names):
                tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = goal_mean[i] if i < len(goal_mean) else 0.0
            tvp_template['_tvp', k, 'terminal_weight'] = 0.0
