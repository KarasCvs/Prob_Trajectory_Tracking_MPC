"""
系统模型定义
定义 3D 位置+速度动力学模型
用于 MPC 轨迹跟踪控制
"""
import numpy as np
import do_mpc


def template_model(symvar_type='SX'):
    """
    定义 3D 轨迹跟踪系统模型
    
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
    
    # ========== 时变参数（参考轨迹）==========
    # 参考位置（在每个预测步由 MPC 提供）
    p_x_ref = model.set_variable('_tvp', 'p_x_ref')
    p_y_ref = model.set_variable('_tvp', 'p_y_ref')
    p_z_ref = model.set_variable('_tvp', 'p_z_ref')
    
    # 参考速度（可选，用于更平滑的跟踪）
    v_x_ref = model.set_variable('_tvp', 'v_x_ref')
    v_y_ref = model.set_variable('_tvp', 'v_y_ref')
    v_z_ref = model.set_variable('_tvp', 'v_z_ref')
    # 终端目标（用于终端等式约束，避免和 tracking 引用混用）
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
    # 位置跟踪误差
    model.set_expression('pos_error_x', p_x - p_x_ref)
    model.set_expression('pos_error_y', p_y - p_y_ref)
    model.set_expression('pos_error_z', p_z - p_z_ref)
    
    # 速度跟踪误差
    model.set_expression('vel_error_x', v_x - v_x_ref)
    model.set_expression('vel_error_y', v_y - v_y_ref)
    model.set_expression('vel_error_z', v_z - v_z_ref)
    
    # 终端位置误差（用于终端约束）
    model.set_expression('terminal_pos_error_x', p_x - p_x_ref)
    model.set_expression('terminal_pos_error_y', p_y - p_y_ref)
    model.set_expression('terminal_pos_error_z', p_z - p_z_ref)
    
    # 完成模型设置
    model.setup()
    
    return model
