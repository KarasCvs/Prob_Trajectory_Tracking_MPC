"""
MPC 控制器配置
实现参考轨迹跟踪的 MPC 控制器
核心：通过 cost function 强制状态贴近参考轨迹
"""
import numpy as np
import casadi as ca
import do_mpc


def template_mpc(model, reference_trajectory_generator=None, silence_solver=False):
    """
    MPC 配置（只做 tracking，终点靠硬约束）

    - **Tracking** cost：stage cost 覆盖每一步，确保 MPC 跟随 TVP；
    - **Reachability guarantee**：终点等式约束（set_nl_cons）强制 p(N)=p_ref(N)，v(N)=0；
    - 所有终端约束在 setup 前定义，不在主循环中动态改。
    """
    mpc = do_mpc.controller.MPC(model)

    # === MPC 设置 ===
    mpc.settings.n_robust = 0
    mpc.settings.n_horizon = 30
    mpc.settings.t_step = 0.1
    mpc.settings.store_full_solution = True

    if silence_solver:
        mpc.settings.supress_ipopt_output()

    # === Tracking cost ===
    pos_error_x = model.aux['pos_error_x']
    pos_error_y = model.aux['pos_error_y']
    pos_error_z = model.aux['pos_error_z']
    vel_error_x = model.aux['vel_error_x']
    vel_error_y = model.aux['vel_error_y']
    vel_error_z = model.aux['vel_error_z']

    Q_pos = 10.0
    Q_vel = 1.0
    lterm = (
        Q_pos * (pos_error_x**2 + pos_error_y**2 + pos_error_z**2) +
        Q_vel * (vel_error_x**2 + vel_error_y**2 + vel_error_z**2)
    )

    # do-mpc 要求 mterm 形状为 (1,1)，这里设为 0 以弱化终端项
    mterm = 0 * lterm
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(a_x=0.1, a_y=0.1, a_z=0.1)

    # === state/input bounds ===
    max_pos = 10.0
    mpc.bounds['lower', '_x', 'p_x'] = -max_pos
    mpc.bounds['upper', '_x', 'p_x'] = max_pos
    mpc.bounds['lower', '_x', 'p_y'] = -max_pos
    mpc.bounds['upper', '_x', 'p_y'] = max_pos
    mpc.bounds['lower', '_x', 'p_z'] = -max_pos
    mpc.bounds['upper', '_x', 'p_z'] = max_pos
    max_vel = 5.0
    mpc.bounds['lower', '_x', 'v_x'] = -max_vel
    mpc.bounds['upper', '_x', 'v_x'] = max_vel
    mpc.bounds['lower', '_x', 'v_y'] = -max_vel
    mpc.bounds['upper', '_x', 'v_y'] = max_vel
    mpc.bounds['lower', '_x', 'v_z'] = -max_vel
    mpc.bounds['upper', '_x', 'v_z'] = max_vel
    max_acc = 2.0
    mpc.bounds['lower', '_u', 'a_x'] = -max_acc
    mpc.bounds['upper', '_u', 'a_x'] = max_acc
    mpc.bounds['lower', '_u', 'a_y'] = -max_acc
    mpc.bounds['upper', '_u', 'a_y'] = max_acc
    mpc.bounds['lower', '_u', 'a_z'] = -max_acc
    mpc.bounds['upper', '_u', 'a_z'] = max_acc

    # TVP 模板（主循环填充具体参考）
    mpc._tvp_template = mpc.get_tvp_template()
    mpc.set_tvp_fun(lambda t_now: mpc._tvp_template)

    # === Terminal equality constraint（硬约束） ===
    def _active(expr, ref):
        return expr * (ref < 1e19)

    terminal_expr = _active(model.x['p_x'] - model.tvp['p_x_ref_terminal'],
                            model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_p_x_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_p_x_neg', -terminal_expr, ub=0.0, soft_constraint=False)

    terminal_expr = _active(model.x['p_y'] - model.tvp['p_y_ref_terminal'],
                            model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_p_y_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_p_y_neg', -terminal_expr, ub=0.0, soft_constraint=False)

    terminal_expr = _active(model.x['p_z'] - model.tvp['p_z_ref_terminal'],
                            model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_p_z_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_p_z_neg', -terminal_expr, ub=0.0, soft_constraint=False)

    terminal_expr = _active(model.x['v_x'], model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_v_x_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_v_x_neg', -terminal_expr, ub=0.0, soft_constraint=False)
    terminal_expr = _active(model.x['v_y'], model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_v_y_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_v_y_neg', -terminal_expr, ub=0.0, soft_constraint=False)
    terminal_expr = _active(model.x['v_z'], model.tvp['p_x_ref_terminal'])
    mpc.set_nl_cons('terminal_v_z_pos', terminal_expr, ub=0.0, soft_constraint=False)
    mpc.set_nl_cons('terminal_v_z_neg', -terminal_expr, ub=0.0, soft_constraint=False)

    mpc.setup()
    return mpc


def update_mpc_reference_trajectory(
    mpc,
    reference_trajectory_generator,
    current_time: float,
    trajectory: np.ndarray,
    time_stamps: np.ndarray,
    terminal_index: int = None
):
    """
    更新 MPC 的参考轨迹（时变参数）
    
    在每个控制周期调用，为 MPC 预测时域内的每一步提供参考轨迹点
    
    Args:
        mpc: MPC 控制器对象
        reference_trajectory_generator: 参考轨迹生成器
        current_time: 当前时间
        trajectory: 完整参考轨迹 [N, 3]
        time_stamps: 参考轨迹时间戳 [N]
    """
    # 获取时变参数模板
    tvp_template = mpc._tvp_template
    
    horizon = mpc.settings.n_horizon
    for k in range(horizon):
        t_k = current_time + k * mpc.settings.t_step
        ref_point = reference_trajectory_generator.get_reference_at_time(
            trajectory, time_stamps, t_k
        )

        tvp_template['_tvp', k, 'p_x_ref'] = ref_point[0]
        tvp_template['_tvp', k, 'p_y_ref'] = ref_point[1]
        tvp_template['_tvp', k, 'p_z_ref'] = ref_point[2]

        t_k_next = current_time + (k + 1) * mpc.settings.t_step
        ref_point_next = reference_trajectory_generator.get_reference_at_time(
            trajectory, time_stamps, t_k_next
        )
        v_ref = (ref_point_next - ref_point) / mpc.settings.t_step

        tvp_template['_tvp', k, 'v_x_ref'] = v_ref[0]
        tvp_template['_tvp', k, 'v_y_ref'] = v_ref[1]
        tvp_template['_tvp', k, 'v_z_ref'] = v_ref[2]

        # 终端等式约束只在对齐"仿真终点"的那一步生效：其余步填哨兵值（不生效）
        if terminal_index is not None and k == terminal_index:
            tvp_template['_tvp', k, 'p_x_ref_terminal'] = trajectory[-1][0]
            tvp_template['_tvp', k, 'p_y_ref_terminal'] = trajectory[-1][1]
            tvp_template['_tvp', k, 'p_z_ref_terminal'] = trajectory[-1][2]
        else:
            tvp_template['_tvp', k, 'p_x_ref_terminal'] = 1e20
            tvp_template['_tvp', k, 'p_y_ref_terminal'] = 1e20
            tvp_template['_tvp', k, 'p_z_ref_terminal'] = 1e20
    
    # TVP 函数已在初始化时设置，这里只更新模板内容
