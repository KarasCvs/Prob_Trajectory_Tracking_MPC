"""
仿真器配置
模拟真实系统的动力学响应
"""
import numpy as np
import do_mpc


def template_simulator(model, tvp_data_dict=None):
    """
    配置仿真器
    
    注意：由于模型定义了时变参数（TVP），仿真器也需要设置 TVP 函数
    
    Args:
        model: do-mpc 模型对象
        tvp_data_dict: 可选的字典，用于存储参考轨迹数据（可在运行时更新）
        
    Returns:
        simulator: 配置好的仿真器
    """
    simulator = do_mpc.simulator.Simulator(model)
    
    # 设置仿真参数
    simulator.set_param(t_step=0.1)  # 与 MPC 采样时间一致
    
    # 设置时变参数（TVP）函数
    # 使用可更新的数据字典（如果提供）
    if tvp_data_dict is None:
        tvp_data_dict = {
            'traj_gen': None,
            'trajectory': None,
            'time_stamps': None
        }
    
    # 获取 TVP 模板（用于创建数值结构）
    tvp_template = simulator.get_tvp_template()
    
    def tvp_fun(t_now):
        """
        时变参数函数
        从数据字典中获取参考轨迹（如果可用）
        注意：每次调用都需要创建新的数值结构
        """
        # 创建新的 TVP 数值结构
        tvp_num = simulator.get_tvp_template()
        
        # 如果参考轨迹数据可用，使用它
        if (tvp_data_dict['traj_gen'] is not None and 
            tvp_data_dict['trajectory'] is not None and
            tvp_data_dict['time_stamps'] is not None):
            
            traj_gen = tvp_data_dict['traj_gen']
            trajectory = tvp_data_dict['trajectory']
            time_stamps = tvp_data_dict['time_stamps']
            
            # 获取当前时刻的参考轨迹点
            ref_point = traj_gen.get_reference_at_time(trajectory, time_stamps, t_now)
            
            # 确保 ref_point 是 numpy 数组并展平，然后转换为 Python 标量
            ref_point = np.array(ref_point).flatten()
            
            # 设置 TVP 值（确保是标量）
            tvp_num['p_x_ref'] = float(ref_point[0])
            tvp_num['p_y_ref'] = float(ref_point[1])
            tvp_num['p_z_ref'] = float(ref_point[2])
            
            # 速度参考（简化处理）
            tvp_num['v_x_ref'] = 0.0
            tvp_num['v_y_ref'] = 0.0
            tvp_num['v_z_ref'] = 0.0
        else:
            # 默认值
            tvp_num['p_x_ref'] = 0.0
            tvp_num['p_y_ref'] = 0.0
            tvp_num['p_z_ref'] = 0.0
            tvp_num['v_x_ref'] = 0.0
            tvp_num['v_y_ref'] = 0.0
            tvp_num['v_z_ref'] = 0.0
        
        return tvp_num
    
    simulator.set_tvp_fun(tvp_fun)
    
    # 将数据字典附加到仿真器对象（便于后续更新）
    simulator._tvp_data = tvp_data_dict
    
    # 完成仿真器设置
    simulator.setup()
    
    return simulator
