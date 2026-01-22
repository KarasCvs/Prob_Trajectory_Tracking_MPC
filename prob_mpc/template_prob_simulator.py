"""
概率模型仿真器配置
支持GP概率轨迹的仿真器
"""
import numpy as np
import do_mpc


def template_prob_simulator(model, gp_trajectory=None, trajectory_duration=1.0):
    """
    配置概率模型仿真器
    
    注意：概率模型的TVP由MPC控制，simulator只需要填充基本TVP
    
    Args:
        model: 概率模型对象
        gp_trajectory: GaussianProcessTrajectory实例（可选，用于填充TVP）
        trajectory_duration: 轨迹持续时间（用于时间归一化）
        
    Returns:
        simulator: 配置好的仿真器
    """
    # 获取维度信息
    dimension = getattr(model, 'dimension', 3)
    dim_names = getattr(model, 'dim_names', ['x', 'y', 'z'])
    
    simulator = do_mpc.simulator.Simulator(model)
    
    # 设置仿真参数
    simulator.set_param(t_step=0.1)  # 与 MPC 采样时间一致
    
    # 获取 TVP 模板
    tvp_template = simulator.get_tvp_template()
    
    def tvp_fun(t_now):
        """
        时变参数函数
        对于概率模型，simulator的TVP主要用于记录，实际控制由MPC处理
        """
        tvp_num = simulator.get_tvp_template()
        
        # 如果提供了GP轨迹，查询GP填充TVP
        if gp_trajectory is not None:
            # 归一化时间（确保是标量）
            t_now_scalar = float(t_now)
            t_normalized = t_now_scalar / trajectory_duration if trajectory_duration > 1e-6 else t_now_scalar
            t_normalized = float(np.clip(t_normalized, 0.0, 1.0))
            
            # 查询GP
            mean = gp_trajectory.predict_mean(t_normalized)
            variance = gp_trajectory.predict_variance(t_normalized)
            goal_mean = gp_trajectory.get_goal_mean()
            goal_variance = gp_trajectory.get_goal_variance()
            
            # 确保维度匹配
            if len(mean) < dimension:
                mean = np.pad(mean, (0, dimension - len(mean)), 'constant')
            if len(variance) < dimension:
                variance = np.pad(variance, (0, dimension - len(variance)), 'constant', constant_values=1e-6)
            if len(goal_mean) < dimension:
                goal_mean = np.pad(goal_mean, (0, dimension - len(goal_mean)), 'constant')
            if len(goal_variance) < dimension:
                goal_variance = np.pad(goal_variance, (0, dimension - len(goal_variance)), 'constant', constant_values=1e-6)
            
            # 填充参考位置（GP均值）
            for i, dim_name in enumerate(dim_names):
                if i < len(mean):
                    tvp_num[f'p_{dim_name}_ref'] = float(mean[i])
                else:
                    tvp_num[f'p_{dim_name}_ref'] = 0.0
            
            # 填充GP方差
            for i, dim_name in enumerate(dim_names):
                if i < len(variance):
                    tvp_num[f'sigma_{dim_name}_sq'] = float(max(variance[i], 1e-6))
                else:
                    tvp_num[f'sigma_{dim_name}_sq'] = 1e-6
            
            # 计算权重（与MPC中的公式一致）
            trace_sigma = np.sum(variance[:dimension])
            threshold = 0.1  # 与template_prob_mpc.py中的阈值一致
            alpha = threshold / (threshold + trace_sigma)
            alpha = np.clip(alpha, 0.01, 1.0)
            tvp_num['alpha_weight'] = float(alpha)
            
            # 填充目标均值和方差
            for i, dim_name in enumerate(dim_names):
                if i < len(goal_mean):
                    tvp_num[f'mu_goal_{dim_name}'] = float(goal_mean[i])
                else:
                    tvp_num[f'mu_goal_{dim_name}'] = 0.0
                
                if i < len(goal_variance):
                    tvp_num[f'sigma_goal_{dim_name}_sq'] = float(max(goal_variance[i], 1e-6))
                else:
                    tvp_num[f'sigma_goal_{dim_name}_sq'] = 1e-6
            
            # 填充终端目标（使用GP目标均值）
            for i, dim_name in enumerate(dim_names):
                if i < len(goal_mean):
                    tvp_num[f'p_{dim_name}_ref_terminal'] = float(goal_mean[i])
                else:
                    tvp_num[f'p_{dim_name}_ref_terminal'] = 0.0
            
            # 注意：不再需要计算参考速度，因为系统是一阶的，没有速度状态
        else:
            # 默认值（如果未提供GP）
            for dim_name in dim_names:
                tvp_num[f'p_{dim_name}_ref'] = 0.0
                tvp_num[f'sigma_{dim_name}_sq'] = 1e-6
                tvp_num[f'mu_goal_{dim_name}'] = 0.0
                tvp_num[f'sigma_goal_{dim_name}_sq'] = 1e-6
                tvp_num[f'p_{dim_name}_ref_terminal'] = 0.0
            tvp_num['alpha_weight'] = 1.0
        
        return tvp_num
    
    simulator.set_tvp_fun(tvp_fun)
    simulator.setup()
    
    return simulator
