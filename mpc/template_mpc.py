"""
MPC 控制器配置
实现参考轨迹跟踪的 MPC 控制器
核心：通过 cost function 强制状态贴近参考轨迹
"""
import numpy as np
import casadi as ca
import do_mpc


def template_mpc(model,
                 reference_trajectory_generator=None,
                 silence_solver=False):
    """
    MPC 配置（只做 tracking，终点靠硬约束）
    
    注意：系统现在只观测和控制位置，速度和加速度都不可观测或控制

    - **Tracking** cost：stage cost 覆盖每一步，确保 MPC 跟随 TVP（只跟踪位置）；
    - **Reachability guarantee**：终点等式约束（set_nl_cons）强制 p(N)=p_ref(N)；
    - 所有终端约束在 setup 前定义，不在主循环中动态改。
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

    # === Tracking cost ===
    # 位置误差项（只跟踪位置，不跟踪速度）
    pos_error_terms = []
    for dim_name in dim_names:
        pos_error_terms.append(model.aux[f'pos_error_{dim_name}']**2)

    Q_pos = 10.0
    lterm = Q_pos * sum(pos_error_terms)

    # 控制输入变化率惩罚（提高平滑性，减少抖动）
    u_change_terms = []
    for dim_name in dim_names:
        u_change_terms.append(model.aux[f'u_change_{dim_name}']**2)

    Q_u_change = 0.5  # 控制输入变化率权重（平滑性，降低权重以避免在重新规划时过度约束）
    lterm += Q_u_change * sum(u_change_terms)

    # do-mpc 要求 mterm 形状为 (1,1)，这里设为 0 以弱化终端项
    mterm = 0 * lterm
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # 设置输入惩罚项（控制输入是期望速度，惩罚用于平滑性）
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
    max_vel = 5.0  # 最大期望速度（m/s）
    for dim_name in dim_names:
        mpc.bounds['lower', '_u', f'u_{dim_name}'] = -max_vel
        mpc.bounds['upper', '_u', f'u_{dim_name}'] = max_vel

    # TVP 模板（主循环填充具体参考）
    mpc._tvp_template = mpc.get_tvp_template()
    mpc.set_tvp_fun(lambda t_now: mpc._tvp_template)

    # 参数模板（用于前一个控制输入）
    # 对于非鲁棒MPC（n_robust=0），n_combinations=1
    # 手动设置 n_combinations（do-mpc 在 setup 前需要这个属性）
    mpc.n_combinations = 1  # n_robust=0 时，n_combinations=1

    # 使用可更新的字典存储前一个控制输入
    mpc._p_data = {}
    for dim_name in dim_names:
        mpc._p_data[f'u_{dim_name}_prev'] = 0.0  # 初始值

    # 设置参数函数（从可更新的字典中获取值）
    def p_fun(t_now):
        p_num = mpc.get_p_template(n_combinations=mpc.n_combinations)
        for dim_name in dim_names:
            p_num['_p', 0,
                  f'u_{dim_name}_prev'] = mpc._p_data[f'u_{dim_name}_prev']
        return p_num

    mpc.set_p_fun(p_fun)

    # === Terminal equality constraint（硬约束） ===
    # 关键修复：使用更可靠的方法来激活/禁用约束
    # 在非终端步，将 ref_terminal 设置为当前参考位置，使约束自动满足
    # 在终端步，将 ref_terminal 设置为目标位置，使约束生效
    # 这里直接使用约束表达式，通过 TVP 更新来控制约束是否生效
    
    # 位置终端约束（只约束位置，不约束速度）
    for dim_name in dim_names:
        # 直接使用位置误差作为约束表达式
        # 在 TVP 更新时，非终端步的 ref_terminal 会被设置为当前参考位置
        # 终端步的 ref_terminal 会被设置为目标位置
        terminal_expr = model.x[f'p_{dim_name}'] - model.tvp[f'p_{dim_name}_ref_terminal']
        
        mpc.set_nl_cons(f'terminal_p_{dim_name}_pos',
                        terminal_expr,
                        ub=0.0,
                        soft_constraint=False)
        mpc.set_nl_cons(f'terminal_p_{dim_name}_neg',
                        -terminal_expr,
                        ub=0.0,
                        soft_constraint=False)

    mpc.setup()
    return mpc


def update_mpc_reference_trajectory(mpc,
                                    reference_trajectory_generator,
                                    current_time: float,
                                    trajectory: np.ndarray,
                                    time_stamps: np.ndarray,
                                    terminal_index: int = None):
    """
    更新 MPC 的参考轨迹（时变参数）
    
    在每个控制周期调用，为 MPC 预测时域内的每一步提供参考轨迹点
    
    Args:
        mpc: MPC 控制器对象
        reference_trajectory_generator: 参考轨迹生成器
        current_time: 当前时间
        trajectory: 完整参考轨迹 [N, dimension]
        time_stamps: 参考轨迹时间戳 [N]
        terminal_index: 终端约束生效的预测步索引
    """
    # 获取维度信息
    model = mpc.model
    dimension = getattr(model, 'dimension', 3)
    dim_names = getattr(model, 'dim_names', ['x', 'y', 'z'])

    # 获取时变参数模板
    tvp_template = mpc._tvp_template

    horizon = mpc.settings.n_horizon
    for k in range(horizon):
        t_k = current_time + k * mpc.settings.t_step
        ref_point = reference_trajectory_generator.get_reference_at_time(
            trajectory, time_stamps, t_k)

        # 设置参考位置（只跟踪位置，不需要参考速度）
        for i, dim_name in enumerate(dim_names):
            if i < len(ref_point):
                tvp_template['_tvp', k, f'p_{dim_name}_ref'] = ref_point[i]
            else:
                # 如果轨迹维度不足，使用0
                tvp_template['_tvp', k, f'p_{dim_name}_ref'] = 0.0

        # 终端等式约束：在终端步设置为目标位置，在非终端步设置为当前参考位置
        # 这样在非终端步，约束自动满足（x - ref = 0），在终端步约束生效（x - target = 0）
        if terminal_index is not None and k == terminal_index:
            # 终端步：设置为目标位置，使约束生效
            for i, dim_name in enumerate(dim_names):
                if i < len(trajectory[-1]):
                    tvp_template[
                        '_tvp', k,
                        f'p_{dim_name}_ref_terminal'] = trajectory[-1][i]
                else:
                    tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = 0.0
        else:
            # 非终端步：设置为当前参考位置，使约束自动满足（误差为0）
            for i, dim_name in enumerate(dim_names):
                if i < len(ref_point):
                    tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = ref_point[i]
                else:
                    tvp_template['_tvp', k, f'p_{dim_name}_ref_terminal'] = 0.0

    # TVP 函数已在初始化时设置，这里只更新模板内容
