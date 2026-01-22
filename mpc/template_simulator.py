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
    # 获取维度信息
    dimension = getattr(model, 'dimension', 3)
    dim_names = getattr(model, 'dim_names', ['x', 'y', 'z'])
    
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
            
            # 设置 TVP 值（确保是标量，支持可变维度）
            # 只设置位置参考，不需要速度参考
            for i, dim_name in enumerate(dim_names):
                if i < len(ref_point):
                    tvp_num[f'p_{dim_name}_ref'] = float(ref_point[i])
                else:
                    tvp_num[f'p_{dim_name}_ref'] = 0.0
        else:
            # 默认值
            for dim_name in dim_names:
                tvp_num[f'p_{dim_name}_ref'] = 0.0
        
        return tvp_num
    
    simulator.set_tvp_fun(tvp_fun)
    
    # 设置参数函数（用于前一个控制输入）
    # 仿真器的 get_p_template() 不需要 n_combinations 参数
    # 使用可更新的字典存储前一个控制输入（初始值为0）
    simulator._p_data = {}
    for dim_name in dim_names:
        simulator._p_data[f'u_{dim_name}_prev'] = 0.0  # 初始值
    
    def p_fun(t_now):
        p_num = simulator.get_p_template()  # 仿真器不需要 n_combinations 参数
        for dim_name in dim_names:
            p_num[f'u_{dim_name}_prev'] = simulator._p_data[f'u_{dim_name}_prev']
        return p_num
    simulator.set_p_fun(p_fun)
    
    # 将数据字典附加到仿真器对象（便于后续更新）
    simulator._tvp_data = tvp_data_dict
    
    # 完成仿真器设置
    simulator.setup()
    
    return simulator
