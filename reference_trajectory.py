"""
参考轨迹生成器
生成时间参数化的连续参考轨迹（如螺旋轨迹）
用于 MPC 轨迹跟踪控制
"""
import numpy as np
from typing import Tuple, Optional, List


class ReferenceTrajectoryGenerator:
    """
    参考轨迹生成器
    
    生成从起点到终点的连续参考轨迹，支持多种轨迹形状：
    - 螺旋轨迹
    - 直线轨迹（用于对比）
    - 自定义曲线
    """
    
    def __init__(self, trajectory_type: str = 'spiral'):
        """
        初始化轨迹生成器
        
        Args:
            trajectory_type: 轨迹类型 ('spiral', 'straight', 'custom')
        """
        self.trajectory_type = trajectory_type
    
    def generate_spiral_trajectory(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        num_points: int,
        spiral_radius: float = 1.0,
        num_turns: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成螺旋轨迹
        
        从起点螺旋式到达目标点，保持轨迹形状约束
        
        Args:
            start: 起点 [x, y, z]
            target: 目标点 [x, y, z]
            duration: 轨迹持续时间（秒）
            num_points: 轨迹点数量
            spiral_radius: 螺旋半径
            num_turns: 螺旋圈数
            
        Returns:
            trajectory: 轨迹点数组 [num_points, 3]
            time_stamps: 时间戳数组 [num_points]
        """
        start = np.array(start).flatten()
        target = np.array(target).flatten()
        
        # 时间参数
        t = np.linspace(0, duration, num_points)
        
        # 计算起点到终点的方向向量
        direction = target - start
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # 起点和终点相同，返回静态轨迹
            trajectory = np.tile(start, (num_points, 1))
            return trajectory, t
        
        # 归一化方向向量
        direction_unit = direction / distance
        
        # 生成螺旋轨迹
        trajectory = np.zeros((num_points, 3))
        
        for i, t_i in enumerate(t):
            # 归一化时间 [0, 1]
            s = t_i / duration
            
            # 沿主方向的线性进展
            linear_progress = start + s * direction
            
            # 螺旋参数
            theta = 2 * np.pi * num_turns * s  # 角度
            
            # 螺旋半径衰减（开始时为0确保起点精确，结束时也为0）
            # 使用平滑的衰减函数，在s=0和s=1时都为0
            if s <= 0.0:
                radius = 0.0  # 起点精确对齐
            elif s >= 1.0:
                radius = 0.0  # 终点精确对齐
            else:
                # 使用平滑的衰减函数：sin(pi * s)，在[0,1]区间内从0到1再到0
                radius = spiral_radius * np.sin(np.pi * s)
            
            # 计算垂直于主方向的平面
            # 使用两个正交向量构建旋转平面
            if abs(direction_unit[2]) < 0.9:
                # 使用 z 轴作为参考
                v1 = np.array([0, 0, 1])
            else:
                # 使用 x 轴作为参考
                v1 = np.array([1, 0, 0])
            
            # 计算垂直于 direction_unit 的向量
            v1 = v1 - np.dot(v1, direction_unit) * direction_unit
            v1 = v1 / (np.linalg.norm(v1) + 1e-10)
            
            # 第二个垂直向量（叉积）
            v2 = np.cross(direction_unit, v1)
            v2 = v2 / (np.linalg.norm(v2) + 1e-10)
            
            # 螺旋偏移
            spiral_offset = radius * (np.cos(theta) * v1 + np.sin(theta) * v2)
            
            # 最终轨迹点
            trajectory[i] = linear_progress + spiral_offset
        
        return trajectory, t
    
    def generate_rollercoaster_trajectory(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        num_points: int,
        circle_radius: float = 0.3,
        circle_plane: str = 'vertical',
        circle_ratio: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成过山车轨迹
        
        轨迹分为两部分：
        1. 前半部分：在空间中画一个圆（垂直或水平）
        2. 后半部分：从圆的终点平滑过渡到目标点
        
        所有轨迹走同一条平稳曲线，方差较小
        
        Args:
            start: 起点 [x, y, z]
            target: 目标点 [x, y, z]
            duration: 轨迹持续时间（秒）
            num_points: 轨迹点数量
            circle_radius: 圆的半径
            circle_plane: 圆所在平面 ('vertical' 或 'horizontal')
            circle_ratio: 圆部分占整个轨迹的比例 [0, 1]，例如0.6表示60%时间用于画圆
            
        Returns:
            trajectory: 轨迹点数组 [num_points, 3]
            time_stamps: 时间戳数组 [num_points]
        """
        start = np.array(start).flatten()
        target = np.array(target).flatten()
        
        # 时间参数
        t = np.linspace(0, duration, num_points)
        
        # 计算起点到终点的方向向量
        direction = target - start
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # 起点和终点相同，返回静态轨迹
            trajectory = np.tile(start, (num_points, 1))
            return trajectory, t
        
        # 归一化方向向量
        direction_unit = direction / distance
        
        # 计算圆的中心点（在起点和终点的中点）
        midpoint = (start + target) / 2
        
        # 确定圆的法向量（垂直于圆所在平面）
        if circle_plane == 'vertical':
            # 垂直圆：法向量与主方向垂直，使用y轴作为参考
            if abs(direction_unit[1]) < 0.9:
                normal = np.array([0, 1, 0])
            else:
                normal = np.array([1, 0, 0])
            # 确保法向量垂直于主方向
            normal = normal - np.dot(normal, direction_unit) * direction_unit
            normal = normal / (np.linalg.norm(normal) + 1e-10)
        else:  # horizontal
            # 水平圆：法向量就是主方向（z轴方向）
            normal = direction_unit
        
        # 计算圆平面内的两个正交向量
        if abs(normal[2]) < 0.9:
            v1 = np.array([0, 0, 1])
        else:
            v1 = np.array([1, 0, 0])
        v1 = v1 - np.dot(v1, normal) * normal
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        v2 = np.cross(normal, v1)
        v2 = v2 / (np.linalg.norm(v2) + 1e-10)
        
        # 计算圆的起点（从实际起点开始）
        # 圆的起点应该在从起点指向圆心的方向上
        start_to_center = midpoint - start
        # 在圆平面内找到最接近start_to_center的方向
        start_dir_in_plane = start_to_center - np.dot(start_to_center, normal) * normal
        if np.linalg.norm(start_dir_in_plane) > 1e-6:
            start_dir_in_plane = start_dir_in_plane / np.linalg.norm(start_dir_in_plane)
        else:
            # 如果起点就在圆心上，使用v1方向
            start_dir_in_plane = v1
        
        # 圆的起点角度（使得起点在圆上）
        start_angle = np.arctan2(np.dot(start_dir_in_plane, v2), np.dot(start_dir_in_plane, v1))
        
        # 生成轨迹
        trajectory = np.zeros((num_points, 3))
        circle_end_idx = int(circle_ratio * num_points)
        
        for i, t_i in enumerate(t):
            s = t_i / duration  # 归一化时间 [0, 1]
            
            if s <= circle_ratio:
                # 第一部分：画圆
                # 归一化到 [0, 1]
                s_circle = s / circle_ratio
                # 角度从start_angle到start_angle+2π
                theta = start_angle + 2 * np.pi * s_circle
                
                # 圆上的点
                circle_point = midpoint + circle_radius * (np.cos(theta) * v1 + np.sin(theta) * v2)
                trajectory[i] = circle_point
            else:
                # 第二部分：从圆的终点平滑过渡到目标点
                s_transition = (s - circle_ratio) / (1 - circle_ratio)  # [0, 1]
                
                # 使用平滑过渡函数（三次函数）
                smooth_s = s_transition ** 2 * (3 - 2 * s_transition)  # smoothstep函数
                
                # 圆的终点（在circle_ratio处，即theta = start_angle + 2π）
                circle_end_angle = start_angle + 2 * np.pi
                circle_end_point = midpoint + circle_radius * (np.cos(circle_end_angle) * v1 + np.sin(circle_end_angle) * v2)
                
                # 从圆的终点平滑过渡到目标点
                trajectory[i] = circle_end_point + smooth_s * (target - circle_end_point)
        
        # 确保起点和终点精确对齐
        trajectory[0] = start
        trajectory[-1] = target
        
        return trajectory, t
    
    def generate_straight_trajectory(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成直线轨迹（用于对比）
        
        Args:
            start: 起点 [x, y, z]
            target: 目标点 [x, y, z]
            duration: 轨迹持续时间
            num_points: 轨迹点数量
            
        Returns:
            trajectory: 轨迹点数组
            time_stamps: 时间戳数组
        """
        start = np.array(start).flatten()
        target = np.array(target).flatten()
        
        t = np.linspace(0, duration, num_points)
        trajectory = np.zeros((num_points, 3))
        
        for i, t_i in enumerate(t):
            s = t_i / duration
            trajectory[i] = start + s * (target - start)
        
        return trajectory, t
    
    def get_reference_at_time(
        self,
        trajectory: np.ndarray,
        time_stamps: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        在指定时间获取参考轨迹点（插值）
        
        Args:
            trajectory: 完整轨迹 [N, 3]
            time_stamps: 时间戳 [N]
            t: 查询时间
            
        Returns:
            ref_point: 参考点 [3]
        """
        # 边界检查
        if t <= time_stamps[0]:
            return trajectory[0]
        if t >= time_stamps[-1]:
            return trajectory[-1]
        
        # 线性插值
        idx = np.searchsorted(time_stamps, t)
        t1, t2 = time_stamps[idx-1], time_stamps[idx]
        p1, p2 = trajectory[idx-1], trajectory[idx]
        
        alpha = (t - t1) / (t2 - t1 + 1e-10)
        ref_point = (1 - alpha) * p1 + alpha * p2
        
        return ref_point
    
    def update_target(
        self,
        current_target: np.ndarray,
        new_target: np.ndarray,
        current_time: float,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新目标点并重新生成轨迹
        
        Args:
            current_target: 当前目标点
            new_target: 新目标点
            current_time: 当前时间
            duration: 新轨迹持续时间
            
        Returns:
            trajectory: 新轨迹
            time_stamps: 新时间戳
        """
        # 从当前位置（当前目标）到新目标生成轨迹
        return self.generate_spiral_trajectory(
            start=current_target,
            target=new_target,
            duration=duration,
            num_points=int(duration * 10)  # 10 Hz 采样
        )
    
    def generate_multiple_trajectories(
        self,
        num_trajectories: int,
        start_mean: np.ndarray,
        start_cov: np.ndarray,
        end_mean: np.ndarray,
        end_cov: np.ndarray,
        duration: float,
        num_points: int,
        trajectory_type: str = 'spiral',
        base_spiral_radius: float = 0.3,
        base_num_turns: float = 2.0,
        noise_scale: float = 0.2,
        convergence_start: float = 0.8,
        convergence_length: float = 0.2,
        # 过山车轨迹参数
        circle_radius: float = 0.3,
        circle_plane: str = 'vertical',
        circle_ratio: float = 0.6
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成多条参考轨迹
        
        支持两种轨迹类型：
        - 'spiral': 螺旋轨迹（起点和终点从高斯分布采样，每条轨迹使用不同的螺旋参数）
        - 'rollercoaster': 过山车轨迹（所有轨迹走同一条平稳曲线，方差较小）
        
        Args:
            num_trajectories: 轨迹数量
            start_mean: 起点均值 [3]
            start_cov: 起点协方差矩阵 [3, 3] 或标量（各向同性）
            end_mean: 终点均值 [3]
            end_cov: 终点协方差矩阵 [3, 3] 或标量（各向同性）
            duration: 每条轨迹的持续时间
            num_points: 每条轨迹的点数
            trajectory_type: 轨迹类型 ('spiral' 或 'rollercoaster')
            base_spiral_radius: 基础螺旋半径（仅用于spiral）
            base_num_turns: 基础螺旋圈数（仅用于spiral）
            noise_scale: 螺旋参数噪音尺度（仅用于spiral）
            convergence_start: 收敛开始时间比例 [0, 1]（仅用于spiral）
            convergence_length: 收敛段长度比例 [0, 1]（仅用于spiral）
            circle_radius: 圆的半径（仅用于rollercoaster）
            circle_plane: 圆所在平面 ('vertical' 或 'horizontal')（仅用于rollercoaster）
            circle_ratio: 圆部分占整个轨迹的比例 [0, 1]（仅用于rollercoaster）
            
        Returns:
            trajectories: 轨迹列表，每个元素是 (trajectory, time_stamps) 元组
        """
        start_mean = np.array(start_mean).flatten()
        end_mean = np.array(end_mean).flatten()
        
        # 处理协方差矩阵
        if np.isscalar(start_cov):
            start_cov = np.eye(3) * start_cov
        else:
            start_cov = np.array(start_cov)
        
        if np.isscalar(end_cov):
            end_cov = np.eye(3) * end_cov
        else:
            end_cov = np.array(end_cov)
        
        trajectories = []
        
        if trajectory_type == 'rollercoaster':
            # 过山车轨迹：所有轨迹走同一条平稳曲线，只在起点和终点有小的变化
            # 使用均值起点和终点生成基准轨迹
            base_trajectory, time_stamps = self.generate_rollercoaster_trajectory(
                start=start_mean,
                target=end_mean,
                duration=duration,
                num_points=num_points,
                circle_radius=circle_radius,
                circle_plane=circle_plane,
                circle_ratio=circle_ratio
            )
            
            for i in range(num_trajectories):
                # 从高斯分布采样起点和终点（方差很小，保持轨迹相似）
                start = np.random.multivariate_normal(start_mean, start_cov)
                end = np.random.multivariate_normal(end_mean, end_cov)
                
                # 计算起点和终点的偏移
                start_offset = start - start_mean
                end_offset = end - end_mean
                
                # 生成基于采样起点和终点的轨迹
                trajectory, _ = self.generate_rollercoaster_trajectory(
                    start=start,
                    target=end,
                    duration=duration,
                    num_points=num_points,
                    circle_radius=circle_radius,
                    circle_plane=circle_plane,
                    circle_ratio=circle_ratio
                )
                
                trajectories.append((trajectory, time_stamps))
        
        else:  # 'spiral' 或其他类型，默认使用螺旋
            # 生成统一的收敛路径（从均值起点到均值终点，使用标准螺旋参数）
            # 这条路径将作为所有轨迹在接近终点时的收敛目标
            convergence_trajectory, convergence_time_stamps = self.generate_spiral_trajectory(
                start=start_mean,
                target=end_mean,
                duration=duration,
                num_points=num_points,
                spiral_radius=base_spiral_radius,
                num_turns=base_num_turns
            )
            
            # 找到收敛段的起始索引
            convergence_start_idx = int(convergence_start * num_points)
            convergence_end_idx = num_points
            
            for i in range(num_trajectories):
                # 从高斯分布采样起点和终点
                start = np.random.multivariate_normal(start_mean, start_cov)
                end = np.random.multivariate_normal(end_mean, end_cov)
                
                # 为每条轨迹生成不同的螺旋参数（加入噪音）
                spiral_radius = base_spiral_radius * (1 + noise_scale * np.random.randn())
                spiral_radius = max(0.1, spiral_radius)  # 确保半径为正
                
                num_turns = base_num_turns * (1 + noise_scale * np.random.randn())
                num_turns = max(0.5, num_turns)  # 确保圈数合理
                
                # 生成随机螺旋轨迹（前半部分）
                random_trajectory, time_stamps = self.generate_spiral_trajectory(
                    start=start,
                    target=end,
                    duration=duration,
                    num_points=num_points,
                    spiral_radius=spiral_radius,
                    num_turns=num_turns
                )
                
                # 创建混合轨迹：前半部分使用随机螺旋，后半部分收敛到统一路径
                mixed_trajectory = random_trajectory.copy()
                
                # 计算收敛段的连接点：在收敛开始时，随机轨迹的位置
                connection_point = random_trajectory[convergence_start_idx]
                
                # 计算收敛路径上对应的连接点（在收敛开始时）
                convergence_connection_point = convergence_trajectory[convergence_start_idx]
                
                # 在收敛段，从随机轨迹平滑过渡到各自的采样终点
                # 关键：每条轨迹收敛到各自的采样终点（end），而不是统一的end_mean
                # 这样可以保持终点的多样性，符合高斯分布
                for idx in range(convergence_start_idx, convergence_end_idx):
                    # 计算在收敛段内的归一化位置 [0, 1]
                    s_converge = (idx - convergence_start_idx) / (convergence_end_idx - convergence_start_idx)
                    
                    # 使用平滑过渡函数（三次函数使过渡更平滑）
                    # alpha从0（随机轨迹）平滑过渡到1（收敛到各自的终点）
                    alpha = s_converge ** 3  # 使用三次函数使过渡更平滑
                    
                    # 计算从连接点到各自采样终点的线性插值
                    # 在收敛段开始时，使用随机轨迹的连接点
                    # 在收敛段结束时，使用各自的采样终点（end）
                    if idx < convergence_end_idx - 1:
                        # 从连接点线性插值到各自的采样终点
                        convergence_target = connection_point + s_converge * (end - connection_point)
                        
                        # 混合：从随机轨迹的连接点平滑过渡到各自的终点
                        # 在收敛段开始时，使用随机轨迹的点
                        # 在收敛段结束时，使用各自的采样终点（保持多样性）
                        mixed_trajectory[idx] = (1 - alpha) * random_trajectory[idx] + alpha * convergence_target
                    else:
                        # 最后一个点：精确收敛到各自的采样终点（保持分布多样性）
                        mixed_trajectory[idx] = end
                
                trajectories.append((mixed_trajectory, time_stamps))
        
        return trajectories
