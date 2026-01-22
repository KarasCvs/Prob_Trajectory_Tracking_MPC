"""
高斯过程轨迹学习模块
使用GPyTorch的Sparse GP从多条示例轨迹学习轨迹概率分布
"""
import numpy as np
import torch
import gpytorch
from typing import List, Tuple, Optional
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


class SparseGPRegressionModel(ExactGP):
    """
    Sparse GP回归模型（使用InducingPointKernel）
    """

    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 num_inducing=50,
                 kernel_type='RBF'):
        super(SparseGPRegressionModel, self).__init__(train_x, train_y,
                                                      likelihood)
        self.mean_module = ConstantMean()

        # 基础核
        if kernel_type == 'RBF':
            base_kernel = RBFKernel()
            # 设置合理的初始lengthscale（归一化时间[0,1]的合理范围）
            # lengthscale=0.3意味着在时间距离0.3内相关性较高
            base_kernel.lengthscale = 0.3
        elif kernel_type == 'Matern':
            base_kernel = MaternKernel(nu=2.5)  # Matern52
            base_kernel.lengthscale = 0.3
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        # ScaleKernel控制输出尺度（方差）
        # 初始outputscale应该反映数据的实际变异性
        # 对于位置数据，合理的初始值在0.1-1.0之间
        self.base_covar_module = ScaleKernel(base_kernel)
        # 设置初始outputscale（会在优化中调整）
        # 这个值应该接近数据的标准差
        self.base_covar_module.outputscale = 0.1

        # 使用InducingPointKernel实现Sparse GP
        # 选择前num_inducing个点作为inducing points
        num_inducing = min(num_inducing, train_x.size(0))
        inducing_points = train_x[:num_inducing].clone()

        self.covar_module = InducingPointKernel(
            base_kernel=self.base_covar_module,
            inducing_points=inducing_points,
            likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GaussianProcessTrajectory:
    """
    高斯过程轨迹模型（使用GPyTorch Sparse GP）
    
    从多条示例轨迹中学习统一的轨迹概率分布
    对每个空间维度（x, y, z）独立建模GP
    """

    def __init__(self,
                 normalize_time: bool = True,
                 device: Optional[torch.device] = None):
        """
        初始化GP轨迹模型
        
        Args:
            normalize_time: 是否将时间归一化到[0,1]（推荐，GP更稳定）
            device: torch设备（None则自动检测CPU/GPU）
        """
        self.normalize_time = normalize_time

        # 设备管理
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dtype = torch.float64  # 使用double精度提高数值稳定性

        self.gp_models = {}  # 存储三个维度的GP模型
        self.likelihoods = {}  # 存储三个维度的likelihood
        self.is_fitted = False
        # 存储每个维度的数据变异性（用于方差校正）
        self.data_std = {}  # 每个维度的标准差

    def fit(self,
            trajectories: List[Tuple[np.ndarray, np.ndarray]],
            optimize: bool = True,
            kernel_type: str = 'RBF',
            noise_variance: float = 1e-4,
            max_iters: int = 200,
            num_inducing: Optional[int] = None):
        """
        从多条示例轨迹拟合GP模型
        
        Args:
            trajectories: 轨迹列表，每个元素是 (trajectory, time_stamps) 元组
                trajectory: [N, 3] 位置数组
                time_stamps: [N] 时间戳数组
            optimize: 是否优化GP超参数
            kernel_type: 核函数类型 ('RBF' 或 'Matern')
            noise_variance: 初始噪声方差
            max_iters: 最大优化迭代次数
            num_inducing: Inducing points数量（None则自动确定）
        """
        if len(trajectories) == 0:
            raise ValueError("至少需要一条轨迹用于训练")

        # 收集所有轨迹数据
        all_times = []
        all_positions = {'x': [], 'y': [], 'z': []}

        for traj, time_stamps in trajectories:
            traj = np.array(traj)
            time_stamps = np.array(time_stamps)

            if traj.shape[1] != 3:
                raise ValueError(f"轨迹维度应为3，得到{traj.shape[1]}")

            all_times.append(time_stamps)
            all_positions['x'].append(traj[:, 0])
            all_positions['y'].append(traj[:, 1])
            all_positions['z'].append(traj[:, 2])

        # 归一化时间到[0,1]（如果启用）
        if self.normalize_time:
            normalized_times = []
            for time_stamps in all_times:
                t_min, t_max = time_stamps.min(), time_stamps.max()
                if t_max - t_min > 1e-6:
                    t_norm = (time_stamps - t_min) / (t_max - t_min)
                else:
                    t_norm = np.zeros_like(time_stamps)
                normalized_times.append(t_norm)
        else:
            normalized_times = all_times

        # 对每个维度独立训练GP
        for dim in ['x', 'y', 'z']:
            # 准备训练数据
            X_list = []
            Y_list = []

            for i, t_norm in enumerate(normalized_times):
                X_list.append(t_norm[:, None])  # [N, 1]
                Y_list.append(all_positions[dim][i][:, None])  # [N, 1]

            # 合并所有轨迹数据
            X = np.vstack(X_list)  # [total_N, 1]
            Y = np.vstack(Y_list)  # [total_N, 1]

            # 转换为torch.Tensor并移到设备
            train_x = torch.tensor(X, dtype=self.dtype, device=self.device)
            train_y = torch.tensor(Y, dtype=self.dtype,
                                   device=self.device).squeeze(-1)  # [total_N]

            # 确定inducing points数量
            if num_inducing is None:
                # 默认：min(500, 数据量的50%)
                num_inducing = min(500, max(10, int(0.5 * len(X))))
            else:
                num_inducing = min(num_inducing, len(X))
                num_inducing = max(10, num_inducing)  # 至少10个
            
            # 计算数据的实际变异性（标准差）
            # 这反映了轨迹间的真实差异，用于方差校正
            Y_std = float(np.std(Y))
            self.data_std[dim] = Y_std
            
            # 根据数据变异性设置初始outputscale
            # 确保outputscale在合理范围内（不能太小）
            initial_outputscale = max(0.01, min(1.0, Y_std))

            # 创建likelihood
            likelihood = GaussianLikelihood()
            likelihood.noise = noise_variance
            likelihood = likelihood.to(device=self.device, dtype=self.dtype)

            # 创建模型
            model = SparseGPRegressionModel(train_x,
                                            train_y,
                                            likelihood,
                                            num_inducing=num_inducing,
                                            kernel_type=kernel_type)
            model = model.to(device=self.device, dtype=self.dtype)
            
            # 设置初始outputscale（基于数据变异性）
            model.base_covar_module.outputscale = initial_outputscale

            # 优化超参数
            if optimize:
                model.train()
                likelihood.train()

                # 使用Adam优化器（同时优化模型和likelihood参数）
                # 收集所有参数，使用id去重（因为InducingPointKernel可能已经包含likelihood参数）
                model_param_ids = {id(p) for p in model.parameters()}
                likelihood_params = [p for p in likelihood.parameters() if id(p) not in model_param_ids]
                all_params = list(model.parameters()) + likelihood_params
                optimizer = torch.optim.Adam(all_params, lr=0.01)

                # 使用ExactMarginalLogLikelihood作为损失
                mll = ExactMarginalLogLikelihood(likelihood, model)

                # 训练循环
                try:
                    for i in range(max_iters):
                        optimizer.zero_grad()
                        output = model(train_x)
                        loss = -mll(output, train_y)
                        loss.backward()
                        optimizer.step()
                except Exception as e:
                    print(f"警告: GP优化失败 ({dim}维度), 使用默认参数: {e}")

            # 设置为评估模式
            model.eval()
            likelihood.eval()

            self.gp_models[dim] = model
            self.likelihoods[dim] = likelihood

        self.is_fitted = True

    def predict_mean(self, t: float) -> np.ndarray:
        """
        预测指定时间的均值轨迹
        
        Args:
            t: 时间
                - 如果normalize_time=True: 期望接收归一化时间[0,1]
                - 如果normalize_time=False: 期望接收原始时间
            
        Returns:
            mean: 均值位置 [3] (x, y, z)
        """
        if not self.is_fitted:
            raise ValueError("GP模型尚未训练，请先调用fit()")

        # 确保t是标量
        t = float(t)

        # 如果使用归一化时间，确保在[0,1]范围内
        if self.normalize_time:
            t_norm = float(np.clip(t, 0.0, 1.0))
        else:
            t_norm = t

        # 查询每个维度的GP
        mean = np.zeros(3)
        for i, dim in enumerate(['x', 'y', 'z']):
            model = self.gp_models[dim]
            likelihood = self.likelihoods[dim]

            # 准备输入（torch.Tensor）
            test_x = torch.tensor([[t_norm]],
                                  dtype=self.dtype,
                                  device=self.device)

            # 预测（禁用梯度计算，启用SGPR方差校正）
            with torch.no_grad(), gpytorch.settings.sgpr_diagonal_correction(
                    True):
                pred = model(test_x)
                pred_likelihood = likelihood(pred)
                mean_pred = pred_likelihood.mean

            # 转换为numpy
            mean[i] = float(mean_pred.cpu().numpy()[0])

        return mean

    def predict_variance(self, t: float) -> np.ndarray:
        """
        预测指定时间的方差（对角协方差）
        
        Args:
            t: 时间
                - 如果normalize_time=True: 期望接收归一化时间[0,1]
                - 如果normalize_time=False: 期望接收原始时间
            
        Returns:
            variance: 方差 [3] (σ²_x, σ²_y, σ²_z)
        """
        if not self.is_fitted:
            raise ValueError("GP模型尚未训练，请先调用fit()")

        # 确保t是标量
        t = float(t)

        # 如果使用归一化时间，确保在[0,1]范围内
        if self.normalize_time:
            t_norm = float(np.clip(t, 0.0, 1.0))
        else:
            t_norm = t

        # 查询每个维度的GP方差
        variance = np.zeros(3)
        for i, dim in enumerate(['x', 'y', 'z']):
            model = self.gp_models[dim]
            likelihood = self.likelihoods[dim]

            # 准备输入（torch.Tensor）
            test_x = torch.tensor([[t_norm]],
                                  dtype=self.dtype,
                                  device=self.device)

            # 预测（禁用梯度计算，启用SGPR方差校正）
            with torch.no_grad(), gpytorch.settings.sgpr_diagonal_correction(
                    True):
                pred = model(test_x)
                pred_likelihood = likelihood(pred)
                var_pred = pred_likelihood.variance

            # 转换为numpy
            var_pred_np = float(var_pred.cpu().numpy()[0])
            
            # 方差校正：Sparse GP会低估方差，特别是对于轨迹间的变异性
            # 添加一个基于数据实际变异性的最小方差项
            data_var_min = (self.data_std[dim] * 0.3) ** 2  # 使用30%的数据标准差作为最小方差
            variance[i] = max(var_pred_np, data_var_min)

        return variance

    def predict_covariance(self, t: float) -> np.ndarray:
        """
        预测指定时间的完整协方差矩阵
        
        注意：当前实现假设各维度独立，返回对角协方差矩阵
        
        Args:
            t: 时间（归一化或原始，取决于normalize_time设置）
            
        Returns:
            covariance: 协方差矩阵 [3, 3]
        """
        variance = self.predict_variance(t)
        return np.diag(variance)

    def predict_mean_and_variance(self,
                                  t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时预测均值和方差（更高效）
        
        Args:
            t: 时间（归一化或原始，取决于normalize_time设置）
            
        Returns:
            mean: 均值位置 [3]
            variance: 方差 [3]
        """
        if not self.is_fitted:
            raise ValueError("GP模型尚未训练，请先调用fit()")

        # 确保t是标量
        t = float(t)

        # 如果使用归一化时间，确保在[0,1]范围内
        if self.normalize_time:
            t_norm = float(np.clip(t, 0.0, 1.0))
        else:
            t_norm = t

        # 准备输入（torch.Tensor）
        test_x = torch.tensor([[t_norm]], dtype=self.dtype, device=self.device)

        # 同时查询所有维度（更高效）
        mean = np.zeros(3)
        variance = np.zeros(3)

        for i, dim in enumerate(['x', 'y', 'z']):
            model = self.gp_models[dim]
            likelihood = self.likelihoods[dim]

            # 预测（禁用梯度计算，启用SGPR方差校正）
            with torch.no_grad(), gpytorch.settings.sgpr_diagonal_correction(
                    True):
                pred = model(test_x)
                pred_likelihood = likelihood(pred)
                mean_pred = pred_likelihood.mean
                var_pred = pred_likelihood.variance

            # 转换为numpy
            mean[i] = float(mean_pred.cpu().numpy()[0])
            var_pred_np = float(var_pred.cpu().numpy()[0])
            
            # 方差校正：Sparse GP会低估方差，特别是对于轨迹间的变异性
            # 添加一个基于数据实际变异性的最小方差项
            # 使用数据标准差的平方作为最小方差（但不要完全覆盖GP的预测）
            data_var_min = (self.data_std[dim] * 0.3) ** 2  # 使用30%的数据标准差作为最小方差
            var_pred_np = max(var_pred_np, data_var_min)
            variance[i] = var_pred_np

        return mean, variance

    def get_goal_mean(self) -> np.ndarray:
        """
        获取目标位置均值（在归一化时间t=1.0）
        
        Returns:
            goal_mean: 目标位置均值 [3]
        """
        # 归一化时间下，t=1.0对应终点
        return self.predict_mean(1.0)

    def get_goal_variance(self) -> np.ndarray:
        """
        获取目标位置方差（在归一化时间t=1.0）
        
        Returns:
            goal_variance: 目标位置方差 [3]
        """
        # 归一化时间下，t=1.0对应终点
        return self.predict_variance(1.0)
    
    def predict_mean_and_variance_batch(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测多个时间点的均值和方差（高效版本）
        
        这是性能优化的关键方法：一次性预测所有时间点，避免循环调用
        
        Args:
            times: 时间数组 [N]（归一化或原始，取决于normalize_time设置）
            
        Returns:
            mean: 均值位置 [N, 3]
            variance: 方差 [N, 3]
        """
        if not self.is_fitted:
            raise ValueError("GP模型尚未训练，请先调用fit()")
        
        # 确保times是1D数组
        times = np.asarray(times).flatten()
        N = len(times)
        
        # 如果使用归一化时间，确保在[0,1]范围内
        if self.normalize_time:
            times_norm = np.clip(times, 0.0, 1.0)
        else:
            times_norm = times
        
        # 转换为torch.Tensor [N, 1]
        test_x = torch.tensor(times_norm[:, None], dtype=self.dtype, device=self.device)
        
        # 批量预测所有维度
        mean = np.zeros((N, 3))
        variance = np.zeros((N, 3))
        
        for i, dim in enumerate(['x', 'y', 'z']):
            model = self.gp_models[dim]
            likelihood = self.likelihoods[dim]
            
            # 批量预测（禁用梯度计算，启用SGPR方差校正）
            with torch.no_grad(), gpytorch.settings.sgpr_diagonal_correction(True):
                pred = model(test_x)
                pred_likelihood = likelihood(pred)
                mean_pred = pred_likelihood.mean  # [N]
                var_pred = pred_likelihood.variance  # [N]
            
            # 转换为numpy
            mean[:, i] = mean_pred.cpu().numpy()
            var_pred_np = var_pred.cpu().numpy()
            
            # 方差校正：Sparse GP会低估方差，特别是对于轨迹间的变异性
            # 添加一个基于数据实际变异性的最小方差项
            data_var_min = (self.data_std[dim] * 0.3) ** 2  # 使用30%的数据标准差作为最小方差
            variance[:, i] = np.maximum(var_pred_np, data_var_min)
        
        return mean, variance
