"""
H5轨迹加载器
从H5文件中加载真实轨迹数据
"""
import os
import h5py
import numpy as np
from typing import List, Tuple, Optional
import glob


class H5TrajectoryLoader:
    """
    H5轨迹加载器
    
    从H5文件中加载轨迹数据，每条轨迹包含位置和姿态信息
    """

    def __init__(self,
                 dataset_path: str,
                 max_trajectories: Optional[int] = None,
                 dimension: Optional[int] = None):
        """
        初始化H5轨迹加载器
        
        Args:
            dataset_path: 数据集路径，例如 '/path/to/datasets/0122/20260122_182659'
            max_trajectories: 最大加载轨迹数量，如果为None则加载所有轨迹
            dimension: 要加载的维度数量，如果为None则加载所有维度（位置3维+姿态4维=7维）
                       如果dimension <= 3，只加载位置的前dimension个维度
                       如果dimension > 3，加载位置的前3个维度 + 姿态的前(dimension-3)个维度
        """
        # 转换为绝对路径
        if not os.path.isabs(dataset_path):
            # 相对路径，尝试从当前工作目录或脚本目录解析
            current_dir = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 先尝试当前目录
            abs_path = os.path.join(current_dir, dataset_path)
            if not os.path.exists(abs_path):
                # 再尝试脚本目录
                abs_path = os.path.join(script_dir, dataset_path)
            dataset_path = abs_path

        self.dataset_path = dataset_path
        self.max_trajectories = max_trajectories
        self.dimension = dimension
        if not os.path.exists(dataset_path):
            raise ValueError(f"数据集路径不存在: {dataset_path}")

    def load_trajectory_from_h5(
            self, h5_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从单个H5文件中加载轨迹
        
        Args:
            h5_file_path: H5文件路径
            
        Returns:
            trajectory: 轨迹数组 [N, dimension]，包含位置和姿态（根据self.dimension选择）
            time_stamps: 时间戳数组 [N]
        """
        with h5py.File(h5_file_path, 'r') as f:
            # 读取位置: state/end/position，形状为 [N, 6]
            if 'state/end/position' not in f:
                raise KeyError(f"H5文件中缺少 'state/end/position': {h5_file_path}")
            # 先读取完整数据到内存
            position_full = np.array(f['state/end/position'])  # [N, 6]

            # 读取姿态: state/end/orientation，形状为 [N, 8]
            if 'state/end/orientation' not in f:
                raise KeyError(
                    f"H5文件中缺少 'state/end/orientation': {h5_file_path}")
            # 先读取完整数据到内存
            orientation_full = np.array(f['state/end/orientation'])  # [N, 8]

            # 根据dimension参数选择要加载的维度
            if self.dimension is None:
                # 默认：加载位置的前3维 + 姿态的前4维 = 7维
                if position_full.shape[1] >= 3:
                    position = position_full[:, 3:6]  # [N, 3]
                else:
                    position = position_full  # 如果不足3维，使用全部

                if orientation_full.shape[1] >= 4:
                    orientation = orientation_full[:, 4:8]  # [N, 4]
                else:
                    orientation = orientation_full  # 如果不足4维，使用全部

                # 组合位置和姿态: [N, 3] + [N, 4] = [N, 7]
                trajectory = np.concatenate([position, orientation],
                                            axis=1)  # [N, 7]
            else:
                # dimension > 3: 加载位置的前3维 + 姿态的前(dimension-3)维
                if position_full.shape[1] >= 3:
                    position = position_full[:, 3:6]  # [N, 3]
                else:
                    # 如果位置数据不足，用零填充
                    position = np.zeros((position_full.shape[0], 3))
                    position[:, :position_full.shape[1]] = position_full

                orientation_dim = self.dimension - 3
                if orientation_full.shape[1] >= orientation_dim:
                    orientation = orientation_full[:, 0:
                                                   orientation_dim]  # [N, orientation_dim]
                else:
                    # 如果姿态数据不足，用零填充
                    orientation = np.zeros(
                        (orientation_full.shape[0], orientation_dim))
                    orientation[:, :orientation_full.
                                shape[1]] = orientation_full

                # 组合位置和姿态: [N, 3] + [N, orientation_dim] = [N, dimension]
                trajectory = np.concatenate([position, orientation],
                                            axis=1)  # [N, dimension]

            # 读取时间戳
            if 'timestamp' in f:
                time_stamps = np.array(f['timestamp'])  # 转换为numpy数组
                if time_stamps.ndim == 0:
                    # 如果是标量，创建与轨迹长度相同的数组
                    time_stamps = np.full(trajectory.shape[0],
                                          float(time_stamps))
                elif len(time_stamps) != trajectory.shape[0]:
                    # 如果长度不匹配，使用索引作为时间
                    time_stamps = np.arange(trajectory.shape[0]) * 0.1
            else:
                # 如果没有时间戳，使用索引作为时间（假设每个点间隔0.1秒）
                time_stamps = np.arange(trajectory.shape[0]) * 0.1

        return trajectory, time_stamps

    def load_all_trajectories(
        self,
        trajectory_dirs: Optional[List[str]] = None,
        h5_filename: str = 'aligned_ee.h5'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        加载所有轨迹
        
        Args:
            trajectory_dirs: 轨迹目录列表（如 ['0000', '0001', ...]），如果为None则自动扫描
            h5_filename: H5文件名（默认 'aligned_ee.h5'）
            
        Returns:
            trajectories: 轨迹列表，每个元素是 (trajectory, time_stamps)
                         其中 trajectory 是 [N, dimension] 数组，time_stamps 是 [N] 数组
        """
        trajectories = []

        # 如果没有指定目录，自动扫描数据集目录
        if trajectory_dirs is None:
            # 查找所有数字命名的目录（如 0000, 0001, ...）
            all_dirs = sorted([
                d for d in os.listdir(self.dataset_path)
                if os.path.isdir(os.path.join(self.dataset_path, d))
                and d.isdigit()
            ])
            # 如果设置了最大轨迹数量，则限制加载数量
            if self.max_trajectories is not None and self.max_trajectories > 0:
                all_dirs = all_dirs[:self.max_trajectories]
            trajectory_dirs = all_dirs

        print(f"   找到 {len(trajectory_dirs)} 个轨迹目录")
        if self.max_trajectories is not None:
            print(f"   限制加载数量: {self.max_trajectories}")

        for traj_dir in trajectory_dirs:
            traj_path = os.path.join(self.dataset_path, traj_dir)
            h5_path = os.path.join(traj_path, h5_filename)

            if not os.path.exists(h5_path):
                print(f"   警告: 跳过 {traj_dir}，找不到文件 {h5_filename}")
                continue

            try:
                # 加载轨迹（每个h5文件包含一条完整轨迹）
                trajectory, time_stamps = self.load_trajectory_from_h5(h5_path)
                trajectories.append((trajectory, time_stamps))
            except Exception as e:
                print(f"   警告: 加载 {traj_dir} 失败: {e}")
                continue

        print(f"   成功加载 {len(trajectories)} 条轨迹")
        return trajectories

    def load_trajectory_sequence(
            self,
            start_idx: int = 0,
            end_idx: Optional[int] = None,
            h5_filename: str = 'aligned_ee.h5'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载连续的轨迹序列（将多个h5文件组合成一条完整轨迹）
        
        Args:
            start_idx: 起始索引（轨迹目录编号）
            end_idx: 结束索引（不包含），如果为None则加载到最后一个
            h5_filename: H5文件名（默认 'aligned_ee.h5'）
            
        Returns:
            trajectory: 完整轨迹 [N, dimension]
            time_stamps: 时间戳数组 [N]
        """
        # 获取所有轨迹目录
        all_dirs = sorted([
            d for d in os.listdir(self.dataset_path) if
            os.path.isdir(os.path.join(self.dataset_path, d)) and d.isdigit()
        ])

        if end_idx is None:
            end_idx = len(all_dirs)

        # 加载指定范围的轨迹点
        trajectory_points = []
        time_stamps = []

        for idx in range(start_idx, min(end_idx, len(all_dirs))):
            traj_dir = all_dirs[idx]
            traj_path = os.path.join(self.dataset_path, traj_dir)
            h5_path = os.path.join(traj_path, h5_filename)

            if not os.path.exists(h5_path):
                print(f"   警告: 跳过 {traj_dir}，找不到文件 {h5_filename}")
                continue

            try:
                trajectory, time_stamps_single = self.load_trajectory_from_h5(
                    h5_path)
                # 每个h5文件是一条完整轨迹，直接添加
                trajectory_points.append(trajectory)
                time_stamps.append(time_stamps_single)
            except Exception as e:
                print(f"   警告: 加载 {traj_dir} 失败: {e}")
                continue

        if len(trajectory_points) == 0:
            raise ValueError(f"未能加载任何轨迹点（索引范围: {start_idx} 到 {end_idx}）")

        # 将所有轨迹点组合成一条连续轨迹
        # trajectory_points 是列表，每个元素是一个 [N_i, dimension] 的数组
        # 需要将它们拼接成一条长轨迹
        all_trajectory_points = []
        all_time_stamps = []
        current_time = 0.0

        for traj, ts in zip(trajectory_points, time_stamps):
            all_trajectory_points.append(traj)
            # 调整时间戳，使它们连续
            if len(ts) > 0:
                ts_adjusted = ts - ts[0] + current_time
                all_time_stamps.append(ts_adjusted)
                current_time = ts_adjusted[-1] + (ts[-1] - ts[0]) / len(
                    ts) if len(ts) > 1 else current_time + 0.1

        # 拼接所有轨迹点
        trajectory = np.concatenate(all_trajectory_points, axis=0)
        time_stamps = np.concatenate(all_time_stamps, axis=0)

        return trajectory, time_stamps


def load_trajectories_from_dataset(
        dataset_path: str,
        h5_filename: str = 'aligned_ee.h5',
        load_mode: str = 'all',
        max_trajectories: Optional[int] = None,
        dimension: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    便捷函数：从数据集加载轨迹
    
    Args:
        dataset_path: 数据集路径
        h5_filename: H5文件名
        load_mode: 加载模式
                   - 'all': 加载所有轨迹（每个目录一条轨迹）
                   - 'sequence': 将所有目录组合成一条连续轨迹
        max_trajectories: 最大加载轨迹数量，如果为None则加载所有轨迹
        dimension: 要加载的维度数量，如果为None则加载所有维度（位置3维+姿态4维=7维）
                   
    Returns:
        trajectories: 轨迹列表
    """
    loader = H5TrajectoryLoader(dataset_path,
                                max_trajectories=max_trajectories,
                                dimension=dimension)

    if load_mode == 'all':
        return loader.load_all_trajectories(h5_filename=h5_filename)
    elif load_mode == 'sequence':
        traj, time_stamps = loader.load_trajectory_sequence(
            h5_filename=h5_filename)
        return [(traj, time_stamps)]
    else:
        raise ValueError(f"未知的加载模式: {load_mode}")
