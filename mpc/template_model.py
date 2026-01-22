"""
系统模型定义
定义 N维 位置控制模型（只观测和控制位置）
用于 MPC 轨迹跟踪控制
"""
import numpy as np
import do_mpc


def template_model(symvar_type='SX', dimension=3):
    """
    定义 N维 轨迹跟踪系统模型（一阶系统，仅位置控制）
    
    状态: x = [p_x, p_y, p_z, ..., p_dim] (只有位置)
    输入: u = [u_x, u_y, u_z, ..., u_dim] (期望速度，控制输入)
    
    动力学:
        p_dot = u  (连续时间)
        p_{k+1} = p_k + u_k * dt  (离散时间，Euler积分)
    
    注意：真实场景下只能观测位置，只能控制期望速度
    控制输入u理解为"期望速度"，系统是一阶系统，天然是梯度下降，不会产生旋转
    
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
    
    model_type = 'discrete'  # 离散时间系统
    model = do_mpc.model.Model(model_type, symvar_type)
    
    # ========== 状态变量 ==========
    # 位置变量（只有位置，没有速度）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_x', f'p_{dim_name}')
    
    # ========== 控制输入 ==========
    # 期望速度输入（控制输入u，理解为期望速度）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_u', f'u_{dim_name}')
    
    # ========== 时变参数（参考轨迹）==========
    # 参考位置（在每个预测步由 MPC 提供）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_tvp', f'p_{dim_name}_ref')
    
    # 终端目标（用于终端等式约束，避免和 tracking 引用混用）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_tvp', f'p_{dim_name}_ref_terminal')
    
    # 前一个控制输入（用于计算控制输入变化率，提高平滑性）
    for i, dim_name in enumerate(dim_names):
        model.set_variable('_p', f'u_{dim_name}_prev')
    
    # ========== 动力学方程 ==========
    dt = 0.1  # 采样时间
    # 位置更新：p_{k+1} = p_k + u_k * dt
    # 其中 u_k 是期望速度控制输入
    for i, dim_name in enumerate(dim_names):
        p_var = model.x[f'p_{dim_name}']
        u_var = model.u[f'u_{dim_name}']  # 控制输入是期望速度
        model.set_rhs(f'p_{dim_name}', p_var + u_var * dt)
    
    # ========== 辅助表达式（用于代价函数）==========
    # 位置跟踪误差
    for i, dim_name in enumerate(dim_names):
        p_var = model.x[f'p_{dim_name}']
        p_ref_var = model.tvp[f'p_{dim_name}_ref']
        model.set_expression(f'pos_error_{dim_name}', p_var - p_ref_var)
    
    # 控制输入变化率（用于平滑性惩罚）
    for i, dim_name in enumerate(dim_names):
        u_var = model.u[f'u_{dim_name}']
        u_prev_var = model.p[f'u_{dim_name}_prev']
        model.set_expression(f'u_change_{dim_name}', u_var - u_prev_var)
    
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
