"""
速度估计器
从位置观测估计速度（用于真实场景，速度不可直接观测）
"""
import numpy as np
from typing import Optional


class VelocityEstimator:
    """
    速度估计器
    
    从位置观测通过数值微分估计速度
    支持多种估计方法：
    - 简单差分：v = (p[k] - p[k-1]) / dt
    - 滑动平均：使用多个历史点平滑估计
    - 低通滤波：减少噪声影响
    """
    
    def __init__(self, 
                 dimension: int = 3,
                 dt: float = 0.1,
                 method: str = 'simple',
                 filter_alpha: float = 0.3):
        """
        初始化速度估计器
        
        Args:
            dimension: 状态空间维度
            dt: 采样时间（秒）
            method: 估计方法 ('simple', 'moving_average', 'lowpass')
            filter_alpha: 低通滤波系数（0-1，越小越平滑）
        """
        self.dimension = dimension
        self.dt = dt
        self.method = method
        self.filter_alpha = filter_alpha
        
        # 存储历史位置（用于估计）
        self.position_history = []
        self.velocity_estimate = np.zeros(dimension)
        
        # 滑动平均参数
        self.window_size = 3  # 使用最近3个点
        
    def estimate(self, position_observed: np.ndarray) -> np.ndarray:
        """
        从位置观测估计速度
        
        Args:
            position_observed: 观测到的位置 [dimension]
            
        Returns:
            velocity_estimated: 估计的速度 [dimension]
        """
        position_observed = np.array(position_observed).flatten()
        
        # 确保维度匹配
        if len(position_observed) < self.dimension:
            position_observed = np.pad(position_observed, 
                                      (0, self.dimension - len(position_observed)), 
                                      'constant')
        elif len(position_observed) > self.dimension:
            position_observed = position_observed[:self.dimension]
        
        # 添加到历史记录
        self.position_history.append(position_observed.copy())
        
        # 如果历史记录太少，返回零速度
        if len(self.position_history) < 2:
            return np.zeros(self.dimension)
        
        # 根据方法估计速度
        if self.method == 'simple':
            # 简单差分：v = (p[k] - p[k-1]) / dt
            if len(self.position_history) >= 2:
                v_est = (self.position_history[-1] - self.position_history[-2]) / self.dt
            else:
                v_est = np.zeros(self.dimension)
                
        elif self.method == 'moving_average':
            # 滑动平均：使用多个点平滑估计
            if len(self.position_history) >= self.window_size:
                # 使用最近window_size个点进行线性拟合
                positions = np.array(self.position_history[-self.window_size:])
                # 简单线性回归：v = mean(diff) / dt
                diffs = np.diff(positions, axis=0)
                v_est = np.mean(diffs, axis=0) / self.dt
            elif len(self.position_history) >= 2:
                v_est = (self.position_history[-1] - self.position_history[-2]) / self.dt
            else:
                v_est = np.zeros(self.dimension)
                
        elif self.method == 'lowpass':
            # 低通滤波：v[k] = alpha * v_new + (1-alpha) * v[k-1]
            if len(self.position_history) >= 2:
                v_new = (self.position_history[-1] - self.position_history[-2]) / self.dt
                self.velocity_estimate = (self.filter_alpha * v_new + 
                                         (1 - self.filter_alpha) * self.velocity_estimate)
                v_est = self.velocity_estimate
            else:
                v_est = np.zeros(self.dimension)
        else:
            raise ValueError(f"未知的估计方法: {self.method}")
        
        # 限制历史记录长度（避免内存无限增长）
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-50:]
        
        return v_est
    
    def reset(self):
        """重置估计器（清除历史记录）"""
        self.position_history = []
        self.velocity_estimate = np.zeros(self.dimension)
    
    def get_current_velocity(self) -> np.ndarray:
        """
        获取当前估计的速度
        
        Returns:
            velocity: 当前速度估计 [dimension]
        """
        if len(self.position_history) < 2:
            return np.zeros(self.dimension)
        
        if self.method == 'lowpass':
            return self.velocity_estimate
        else:
            # 对于其他方法，返回最后一次估计
            if len(self.position_history) >= 2:
                return (self.position_history[-1] - self.position_history[-2]) / self.dt
            else:
                return np.zeros(self.dimension)
