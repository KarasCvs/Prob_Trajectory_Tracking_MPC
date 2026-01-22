"""
统一的轨迹跟踪 MPC 主程序
支持标准MPC和概率MPC两种方法
"""
import argparse
import sys
import os

# 添加当前目录到路径，以便导入公共模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_mpc(num_replanning: int = 20, trajectory_type: str = 'spiral'):
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
                         trajectory_type=trajectory_type)


def run_prob_mpc(num_replanning: int = 0,
                 debug: bool = False,
                 trajectory_type: str = 'rollercoaster'):
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
    main_prob_mpc_module.main(num_replanning=num_replanning,
                              debug=debug,
                              trajectory_type=trajectory_type)


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
                        default='mpc',
                        choices=['mpc', 'prob_mpc'],
                        help='选择MPC方法: mpc (标准MPC) 或 prob_mpc (概率MPC)')

    parser.add_argument('--trajectory-type',
                        type=str,
                        default='rollercoaster',
                        choices=['spiral', 'rollercoaster'],
                        help='轨迹类型: spiral (螺旋轨迹) 或 rollercoaster (过山车轨迹)')

    parser.add_argument('--num-replanning',
                        type=int,
                        default=None,
                        help='重新规划次数（标准MPC默认: 20，概率MPC默认: 0）')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Debug模式（仅用于概率MPC：GP训练仅进行10次迭代）')

    args = parser.parse_args()

    # 根据方法选择运行
    if args.method == 'mpc':
        num_replanning = args.num_replanning if args.num_replanning is not None else 20
        print("=" * 60)
        print("运行标准MPC方法")
        print("=" * 60)
        run_mpc(num_replanning=num_replanning,
                trajectory_type=args.trajectory_type)
    elif args.method == 'prob_mpc':
        num_replanning = args.num_replanning if args.num_replanning is not None else 0
        print("=" * 60)
        print("运行概率MPC方法")
        print("=" * 60)
        run_prob_mpc(num_replanning=num_replanning,
                     debug=args.debug,
                     trajectory_type=args.trajectory_type)
    else:
        parser.error(f"未知的方法: {args.method}")


if __name__ == '__main__':
    main()
