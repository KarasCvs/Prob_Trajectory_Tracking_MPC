"""
概率模型系统定义
一阶系统：只有位置状态，控制输入是期望速度
支持GP概率轨迹跟踪
"""
import numpy as np
import do_mpc


def template_prob_model(symvar_type='SX', dimension=3):
    """
    定义支持概率轨迹跟踪的N维系统模型（一阶系统）
    
    状态: x = [p_x, p_y, p_z, ..., p_dim] (只有位置)
    输入: u = [u_x, u_y, u_z, ..., u_dim] (期望速度，控制输入)
    
    动力学:
        p_dot = u  (连续时间)
        p_{k+1} = p_k + u_k * dt  (离散时间，Euler积分)
    
    注意：一阶系统在势场中天然是梯度下降，不会产生旋转（无curl）
    这解决了之前二阶系统导致的螺旋问题
    
    Args:
        symvar_type: 符号类型 ('SX' 或 'MX')
        dimension: 状态空间维度（默认3，表示3D空间）
        
    Returns:
        model: do-mpc 模型对象
    """
    if dimension < 3:
        raise ValueError("维度必须至少为3（前3维用于笛卡尔空间可视化）")
    
    # 维度名称列表：x, y, z, w, ...
    dim_names = ['x', 'y', 'z'] + [chr(ord('a') + i) for i in range(dimension - 3)]
    if dimension > 26:
        # 如果维度超过26，使用数字后缀
        dim_names = ['x', 'y', 'z'] + [f'd{i}' for i in range(3, dimension)]
    
    model_type = 'discrete'  # 离散时间系统（Euler）
    model = do_mpc.model.Model(model_type, symvar_type)
    
    # ========== 状态变量 ==========
    # 位置变量（只有位置，没有速度）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_x', f'p_{dim_name}')
    
    # ========== 控制输入 ==========
    # 期望速度输入（控制输入u，理解为期望速度）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_u', f'u_{dim_name}')
    
    # ========== 时变参数（TVP）==========
    # 参考位置（GP均值轨迹）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_tvp', f'p_{dim_name}_ref')
    
    # GP方差（对角协方差）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_tvp', f'sigma_{dim_name}_sq')
    
    # 权重调度因子 α(t)
    alpha_weight = model.set_variable('_tvp', 'alpha_weight')
    
    # 目标位置均值（用于目标项）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_tvp', f'mu_goal_{dim_name}')
    
    # 目标位置方差
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_tvp', f'sigma_goal_{dim_name}_sq')
    
    # 终端目标（用于终端软约束惩罚）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_tvp', f'p_{dim_name}_ref_terminal')
    
    # 终端惩罚权重（0表示不激活，>0表示激活）
    model.set_variable('_tvp', 'terminal_weight')
    
    # ========== 动力学方程（Euler 离散） ==========
    dt = 0.1
    # 位置更新：p_{k+1} = p_k + u_k * dt
    # 其中 u_k 是期望速度控制输入
    for i, dim_name in enumerate(dim_names):
        p_var = model.x[f'p_{dim_name}']
        u_var = model.u[f'u_{dim_name}']  # 控制输入是期望速度
        model.set_rhs(f'p_{dim_name}', p_var + u_var * dt)
    
    # ========== 辅助表达式（用于代价函数）==========
    # 位置跟踪误差（用于轨迹项）
    for i, dim_name in enumerate(dim_names):
        p_var = model.x[f'p_{dim_name}']
        p_ref_var = model.tvp[f'p_{dim_name}_ref']
        pos_error = p_var - p_ref_var
        model.set_expression(f'pos_error_{dim_name}', pos_error)
    
    # Mahalanobis距离 - 轨迹项
    # ℓ_traj = (x_p - μ(t))^T Σ(t)^(-1) (x_p - μ(t))
    # 使用对角协方差：Σ^(-1) = diag(1/σ²)
    for i, dim_name in enumerate(dim_names):
        pos_error = model.aux[f'pos_error_{dim_name}']
        sigma_sq = model.tvp[f'sigma_{dim_name}_sq']
        mahalanobis_traj = (pos_error**2) / sigma_sq
        model.set_expression(f'mahalanobis_traj_{dim_name}', mahalanobis_traj)
    
    # 目标位置误差（用于目标项）
    for i, dim_name in enumerate(dim_names):
        p_var = model.x[f'p_{dim_name}']
        mu_goal_var = model.tvp[f'mu_goal_{dim_name}']
        goal_error = p_var - mu_goal_var
        model.set_expression(f'goal_error_{dim_name}', goal_error)
    
    # Mahalanobis距离 - 目标项
    # ℓ_goal = (x_p - μ_T)^T Σ_T^(-1) (x_p - μ_T)
    for i, dim_name in enumerate(dim_names):
        goal_error = model.aux[f'goal_error_{dim_name}']
        sigma_goal_sq = model.tvp[f'sigma_goal_{dim_name}_sq']
        mahalanobis_goal = (goal_error**2) / sigma_goal_sq
        model.set_expression(f'mahalanobis_goal_{dim_name}', mahalanobis_goal)
    
    # 终端位置误差（用于终端约束）
    for i, dim_name in enumerate(dim_names):
        p_var = model.x[f'p_{dim_name}']
        p_ref_terminal_var = model.tvp[f'p_{dim_name}_ref_terminal']
        model.set_expression(f'terminal_pos_error_{dim_name}', p_var - p_ref_terminal_var)
    
    # 完成模型设置
    model.setup()
    
    # 存储维度信息供后续使用
    model.dimension = dimension
    model.dim_names = dim_names
    
    return model
