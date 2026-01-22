"""
概率模型系统定义
扩展的3D位置+速度动力学模型，支持GP概率轨迹跟踪
"""
import numpy as np
import do_mpc


def template_prob_model(symvar_type='SX'):
    """
    定义支持概率轨迹跟踪的3D系统模型
    
    状态: x = [p_x, p_y, p_z, v_x, v_y, v_z]
    输入: u = [a_x, a_y, a_z] (加速度)
    
    动力学:
        p_dot = v
        v_dot = u
    
    Args:
        symvar_type: 符号类型 ('SX' 或 'MX')
        
    Returns:
        model: do-mpc 模型对象
    """
    model_type = 'discrete'  # 离散时间系统（Euler）
    model = do_mpc.model.Model(model_type, symvar_type)
    
    # ========== 状态变量 ==========
    # 位置
    p_x = model.set_variable('_x', 'p_x')
    p_y = model.set_variable('_x', 'p_y')
    p_z = model.set_variable('_x', 'p_z')
    
    # 速度
    v_x = model.set_variable('_x', 'v_x')
    v_y = model.set_variable('_x', 'v_y')
    v_z = model.set_variable('_x', 'v_z')
    
    # ========== 控制输入 ==========
    # 加速度输入
    a_x = model.set_variable('_u', 'a_x')
    a_y = model.set_variable('_u', 'a_y')
    a_z = model.set_variable('_u', 'a_z')
    
    # ========== 时变参数（TVP）==========
    # 参考位置（GP均值轨迹）
    p_x_ref = model.set_variable('_tvp', 'p_x_ref')
    p_y_ref = model.set_variable('_tvp', 'p_y_ref')
    p_z_ref = model.set_variable('_tvp', 'p_z_ref')
    
    # 参考速度（可选，用于更平滑的跟踪）
    v_x_ref = model.set_variable('_tvp', 'v_x_ref')
    v_y_ref = model.set_variable('_tvp', 'v_y_ref')
    v_z_ref = model.set_variable('_tvp', 'v_z_ref')
    
    # GP方差（对角协方差）
    sigma_x_sq = model.set_variable('_tvp', 'sigma_x_sq')
    sigma_y_sq = model.set_variable('_tvp', 'sigma_y_sq')
    sigma_z_sq = model.set_variable('_tvp', 'sigma_z_sq')
    
    # 权重调度因子 α(t)
    alpha_weight = model.set_variable('_tvp', 'alpha_weight')
    
    # 目标位置均值（用于目标项）
    mu_goal_x = model.set_variable('_tvp', 'mu_goal_x')
    mu_goal_y = model.set_variable('_tvp', 'mu_goal_y')
    mu_goal_z = model.set_variable('_tvp', 'mu_goal_z')
    
    # 目标位置方差
    sigma_goal_x_sq = model.set_variable('_tvp', 'sigma_goal_x_sq')
    sigma_goal_y_sq = model.set_variable('_tvp', 'sigma_goal_y_sq')
    sigma_goal_z_sq = model.set_variable('_tvp', 'sigma_goal_z_sq')
    
    # 终端目标（用于终端等式约束）
    p_x_ref_terminal = model.set_variable('_tvp', 'p_x_ref_terminal')
    p_y_ref_terminal = model.set_variable('_tvp', 'p_y_ref_terminal')
    p_z_ref_terminal = model.set_variable('_tvp', 'p_z_ref_terminal')
    
    # ========== 动力学方程（Euler 离散） ==========
    dt = 0.1
    # 位置更新
    model.set_rhs('p_x', p_x + v_x * dt)
    model.set_rhs('p_y', p_y + v_y * dt)
    model.set_rhs('p_z', p_z + v_z * dt)
    # 速度更新
    model.set_rhs('v_x', v_x + a_x * dt)
    model.set_rhs('v_y', v_y + a_y * dt)
    model.set_rhs('v_z', v_z + a_z * dt)
    
    # ========== 辅助表达式（用于代价函数）==========
    # 位置跟踪误差（用于轨迹项）
    pos_error_x = p_x - p_x_ref
    pos_error_y = p_y - p_y_ref
    pos_error_z = p_z - p_z_ref
    
    model.set_expression('pos_error_x', pos_error_x)
    model.set_expression('pos_error_y', pos_error_y)
    model.set_expression('pos_error_z', pos_error_z)
    
    # 速度跟踪误差
    vel_error_x = v_x - v_x_ref
    vel_error_y = v_y - v_y_ref
    vel_error_z = v_z - v_z_ref
    
    model.set_expression('vel_error_x', vel_error_x)
    model.set_expression('vel_error_y', vel_error_y)
    model.set_expression('vel_error_z', vel_error_z)
    
    # Mahalanobis距离 - 轨迹项
    # ℓ_traj = (x_p - μ(t))^T Σ(t)^(-1) (x_p - μ(t))
    # 使用对角协方差：Σ^(-1) = diag(1/σ²)
    mahalanobis_traj_x = (pos_error_x**2) / sigma_x_sq
    mahalanobis_traj_y = (pos_error_y**2) / sigma_y_sq
    mahalanobis_traj_z = (pos_error_z**2) / sigma_z_sq
    
    model.set_expression('mahalanobis_traj_x', mahalanobis_traj_x)
    model.set_expression('mahalanobis_traj_y', mahalanobis_traj_y)
    model.set_expression('mahalanobis_traj_z', mahalanobis_traj_z)
    
    # 目标位置误差（用于目标项）
    goal_error_x = p_x - mu_goal_x
    goal_error_y = p_y - mu_goal_y
    goal_error_z = p_z - mu_goal_z
    
    model.set_expression('goal_error_x', goal_error_x)
    model.set_expression('goal_error_y', goal_error_y)
    model.set_expression('goal_error_z', goal_error_z)
    
    # Mahalanobis距离 - 目标项
    # ℓ_goal = (x_p - μ_T)^T Σ_T^(-1) (x_p - μ_T)
    mahalanobis_goal_x = (goal_error_x**2) / sigma_goal_x_sq
    mahalanobis_goal_y = (goal_error_y**2) / sigma_goal_y_sq
    mahalanobis_goal_z = (goal_error_z**2) / sigma_goal_z_sq
    
    model.set_expression('mahalanobis_goal_x', mahalanobis_goal_x)
    model.set_expression('mahalanobis_goal_y', mahalanobis_goal_y)
    model.set_expression('mahalanobis_goal_z', mahalanobis_goal_z)
    
    # 终端位置误差（用于终端约束）
    model.set_expression('terminal_pos_error_x', p_x - p_x_ref_terminal)
    model.set_expression('terminal_pos_error_y', p_y - p_y_ref_terminal)
    model.set_expression('terminal_pos_error_z', p_z - p_z_ref_terminal)
    
    # 完成模型设置
    model.setup()
    
    return model
