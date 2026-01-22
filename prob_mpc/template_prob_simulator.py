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
            
            # 填充参考位置（GP均值）
            tvp_num['p_x_ref'] = float(mean[0])
            tvp_num['p_y_ref'] = float(mean[1])
            tvp_num['p_z_ref'] = float(mean[2])
            
            # 填充GP方差
            tvp_num['sigma_x_sq'] = float(max(variance[0], 1e-6))
            tvp_num['sigma_y_sq'] = float(max(variance[1], 1e-6))
            tvp_num['sigma_z_sq'] = float(max(variance[2], 1e-6))
            
            # 计算权重（与MPC中的公式一致）
            trace_sigma = np.sum(variance)
            threshold = 0.1  # 与template_prob_mpc.py中的阈值一致
            alpha = threshold / (threshold + trace_sigma)
            alpha = np.clip(alpha, 0.01, 1.0)
            tvp_num['alpha_weight'] = float(alpha)
            
            # 填充目标均值和方差
            tvp_num['mu_goal_x'] = float(goal_mean[0])
            tvp_num['mu_goal_y'] = float(goal_mean[1])
            tvp_num['mu_goal_z'] = float(goal_mean[2])
            tvp_num['sigma_goal_x_sq'] = float(max(goal_variance[0], 1e-6))
            tvp_num['sigma_goal_y_sq'] = float(max(goal_variance[1], 1e-6))
            tvp_num['sigma_goal_z_sq'] = float(max(goal_variance[2], 1e-6))
            
            # 填充终端目标（使用GP目标均值）
            tvp_num['p_x_ref_terminal'] = float(goal_mean[0])
            tvp_num['p_y_ref_terminal'] = float(goal_mean[1])
            tvp_num['p_z_ref_terminal'] = float(goal_mean[2])
            
            # 计算参考速度（数值微分）
            if t_normalized < 0.99:
                t_next = float(min(1.0, t_normalized + 0.01))  # 小步长用于微分
                mean_next = gp_trajectory.predict_mean(t_next)
                dt_actual = 0.01 * trajectory_duration
                v_ref = (mean_next - mean) / dt_actual
            else:
                v_ref = np.zeros(3)
            
            tvp_num['v_x_ref'] = float(v_ref[0])
            tvp_num['v_y_ref'] = float(v_ref[1])
            tvp_num['v_z_ref'] = float(v_ref[2])
        else:
            # 默认值（如果未提供GP）
            tvp_num['p_x_ref'] = 0.0
            tvp_num['p_y_ref'] = 0.0
            tvp_num['p_z_ref'] = 0.0
            tvp_num['v_x_ref'] = 0.0
            tvp_num['v_y_ref'] = 0.0
            tvp_num['v_z_ref'] = 0.0
            tvp_num['sigma_x_sq'] = 1e-6
            tvp_num['sigma_y_sq'] = 1e-6
            tvp_num['sigma_z_sq'] = 1e-6
            tvp_num['alpha_weight'] = 1.0
            tvp_num['mu_goal_x'] = 0.0
            tvp_num['mu_goal_y'] = 0.0
            tvp_num['mu_goal_z'] = 0.0
            tvp_num['sigma_goal_x_sq'] = 1e-6
            tvp_num['sigma_goal_y_sq'] = 1e-6
            tvp_num['sigma_goal_z_sq'] = 1e-6
            tvp_num['p_x_ref_terminal'] = 0.0
            tvp_num['p_y_ref_terminal'] = 0.0
            tvp_num['p_z_ref_terminal'] = 0.0
        
        return tvp_num
    
    simulator.set_tvp_fun(tvp_fun)
    simulator.setup()
    
    return simulator
