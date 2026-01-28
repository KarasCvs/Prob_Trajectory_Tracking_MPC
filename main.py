"""
统一的轨迹跟踪 MPC 主程序
支持标准MPC和概率MPC两种方法
"""
import argparse
import sys
import os

# 添加当前目录到路径，以便导入公共模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_mpc(num_replanning: int = 20,
            trajectory_type: str = 'spiral',
            dimension: int = 3,
            estimate_velocity: bool = True,
            use_real_trajectory: bool = False,
            dataset_path: str = None):
    """运行标准MPC方法"""
    # 导入原标准MPC实现
    import importlib.util

    main_mpc_path = os.path.join(os.path.dirname(__file__),
                                 'main_mpc_original.py')

    if not os.path.exists(main_mpc_path):
        raise FileNotFoundError(f"找不到标准MPC的原始实现文件: {main_mpc_path}")

    spec = importlib.util.spec_from_file_location("main_mpc", main_mpc_path)
    main_mpc_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_mpc_module)

    # 调用main函数
    main_mpc_module.main(num_replanning=num_replanning,
                         trajectory_type=trajectory_type,
                         dimension=dimension,
                         estimate_velocity=estimate_velocity,
                         use_real_trajectory=use_real_trajectory,
                         dataset_path=dataset_path)


def run_prob_mpc(num_replanning: int = 0,
                 debug: bool = False,
                 trajectory_type: str = 'rollercoaster',
                 dimension: int = 3,
                 estimate_velocity: bool = True,
                 alpha_threshold: float = 0.0001,
                 use_real_trajectory: bool = False,
                 dataset_path: str = None,
                 num_steps: int = 100,
                 enable_terminal_constraint: bool = False,
                 observation_noise_std: float = 0.0):
    """运行概率MPC方法"""
    # 导入原概率MPC实现
    import importlib.util

    main_prob_mpc_path = os.path.join(os.path.dirname(__file__),
                                      'main_prob_mpc_original.py')

    if not os.path.exists(main_prob_mpc_path):
        raise FileNotFoundError(f"找不到概率MPC的原始实现文件: {main_prob_mpc_path}")

    spec = importlib.util.spec_from_file_location("main_prob_mpc",
                                                  main_prob_mpc_path)
    main_prob_mpc_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_prob_mpc_module)

    # 调用main函数
    main_prob_mpc_module.main(
        num_replanning=num_replanning,
        debug=debug,
        trajectory_type=trajectory_type,
        dimension=dimension,
        estimate_velocity=estimate_velocity,
        alpha_threshold=alpha_threshold,
        use_real_trajectory=use_real_trajectory,
        dataset_path=dataset_path,
        num_steps=num_steps,
        enable_terminal_constraint=enable_terminal_constraint,
        observation_noise_std=observation_noise_std)


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description='轨迹跟踪 MPC 仿真（支持标准MPC和概率MPC）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行标准MPC（螺旋轨迹）
  python main.py --method mpc --trajectory-type spiral --num-replanning 20
  
  # 运行概率MPC（过山车轨迹）
  python main.py --method prob_mpc --trajectory-type rollercoaster --debug
  
  # 运行标准MPC（过山车轨迹）
  python main.py --method mpc --trajectory-type rollercoaster
        """)

    parser.add_argument('--method',
                        type=str,
                        default='prob_mpc',
                        choices=['mpc', 'prob_mpc'],
                        help='选择MPC方法: mpc (标准MPC) 或 prob_mpc (概率MPC)')

    parser.add_argument('--trajectory-type',
                        type=str,
                        default='rollercoaster',
                        choices=['spiral', 'rollercoaster'],
                        help='轨迹类型: spiral (螺旋轨迹) 或 rollercoaster (过山车轨迹)')

    parser.add_argument(
        '--num-replanning',
        type=int,
        default=5,
        help='重新规划次数。标准MPC: 更新目标的次数；概率MPC: 重观测+沿用同一GP轨迹剩余段的次数，不重查GP，默认 0')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Debug模式（仅用于概率MPC：GP训练仅进行10次迭代）')

    parser.add_argument('--dimension',
                        type=int,
                        default=3,
                        help='状态空间维度（默认: 3，表示3D空间。前3维永远是x, y, z用于可视化）')

    parser.add_argument('--perfect-velocity',
                        action='store_true',
                        default=False,
                        help='假设速度完全可观测（默认: False。如果设置，则禁用速度估计）')

    parser.add_argument(
        '--alpha-threshold',
        type=float,
        default=0.1,
        help=
        'Alpha权重计算的阈值参数（仅用于概率MPC，默认0.01）。较小的值：alpha更小，更强调目标项，终点误差更小；较大的值：alpha更大，更强调轨迹跟踪'
    )

    parser.add_argument('--use-real-trajectory',
                        action='store_true',
                        help='使用真实轨迹（从H5数据集加载）而不是生成轨迹')

    parser.add_argument('--dataset-path',
                        type=str,
                        default=None,
                        help='数据集路径（当使用真实轨迹时必需），例如: datasets/0122')

    parser.add_argument('--num-steps',
                        type=int,
                        default=200,
                        help='轨迹总步数 / 从 GP mean 采样的参考点数（仅概率MPC，越大越平滑）')

    parser.add_argument('--enable-terminal-constraint',
                        action='store_true',
                        help='启用终点约束（仅用于概率MPC，默认关闭）')

    parser.add_argument('--observation-noise-std',
                        type=float,
                        default=0.0,
                        help='观测位置的高斯噪声标准差（仅ProbMPC，单位与位置一致；0表示无噪声）')

    args = parser.parse_args()

    # 处理速度估计选项：默认启用，除非指定--perfect-velocity
    estimate_velocity = not args.perfect_velocity

    # 验证维度参数
    if args.dimension < 3:
        parser.error("维度必须至少为3（前3维用于笛卡尔空间可视化）")

    # 根据方法选择运行
    if args.method == 'mpc':
        num_replanning = args.num_replanning if args.num_replanning is not None else 20
        print("=" * 60)
        print("运行标准MPC方法")
        print(f"状态空间维度: {args.dimension}")
        print(f"控制模式: 一阶系统（状态=位置，控制输入=期望速度，p_dot=u）")
        print("=" * 60)
        run_mpc(num_replanning=num_replanning,
                trajectory_type=args.trajectory_type,
                dimension=args.dimension,
                estimate_velocity=estimate_velocity,
                use_real_trajectory=args.use_real_trajectory,
                dataset_path=args.dataset_path)
    elif args.method == 'prob_mpc':
        num_replanning = args.num_replanning if args.num_replanning is not None else 0  # 默认不重采样
        print("=" * 60)
        print("运行概率MPC方法")
        print(f"状态空间维度: {args.dimension}")
        print(f"控制模式: 一阶系统（状态=位置，控制输入=期望速度，p_dot=u）")
        print("=" * 60)
        run_prob_mpc(
            num_replanning=num_replanning,
            debug=args.debug,
            trajectory_type=args.trajectory_type,
            dimension=args.dimension,
            estimate_velocity=estimate_velocity,
            alpha_threshold=args.alpha_threshold,
            use_real_trajectory=args.use_real_trajectory,
            dataset_path=args.dataset_path,
            num_steps=args.num_steps,
            enable_terminal_constraint=args.enable_terminal_constraint,
            observation_noise_std=args.observation_noise_std)
    else:
        parser.error(f"未知的方法: {args.method}")


if __name__ == '__main__':
    main()
