# 轨迹跟踪 MPC

一个全面的模型预测控制（MPC）框架，用于三维空间中的轨迹跟踪，支持确定性和概率性参考轨迹。本项目展示了 MPC 在严格约束系统沿参考轨迹形状（如螺旋、过山车）演化方面的能力，而不仅仅是到达终点。

## 目录

- [概述](#概述)
- [项目结构](#项目结构)
- [数学公式](#数学公式)
- [安装](#安装)
- [使用方法](#使用方法)
- [方法对比](#方法对比)
- [关键特性](#关键特性)

## 概述

本项目实现了两种 MPC 轨迹跟踪方法：

1. **标准 MPC**：使用欧几里得距离的确定性参考轨迹跟踪
2. **概率 MPC**：使用高斯过程（GP）回归和马氏距离的概率性轨迹跟踪

**核心思想**：轨迹不是被"学出来的"，而是通过 MPC "实时优化并跟踪"的。

**重要说明**：
- 系统采用**一阶模型**（只观测和控制位置），控制输入为期望速度（增量步进方式）
- 标准 MPC 在终端步强制执行等式约束以保证目标位置处的误差为零
- 概率 MPC 已移除终端约束（避免轨迹变直线问题），仅通过轨迹跟踪项实现跟踪

## 项目结构

```
trajectory_tracking_mpc/
├── main.py                          # 统一入口点（参数解析与路由）
├── main_mpc_original.py             # 标准 MPC 实现（679 行）
├── main_prob_mpc_original.py         # 概率 MPC 实现（526 行）
│
├── mpc/                              # 标准 MPC 模块
│   ├── __init__.py
│   ├── template_model.py            # 系统模型定义
│   ├── template_mpc.py              # MPC 控制器配置
│   └── template_simulator.py        # 仿真器配置
│
├── prob_mpc/                         # 概率 MPC 模块
│   ├── __init__.py
│   ├── template_prob_model.py       # 概率系统模型
│   ├── template_prob_mpc.py        # 概率 MPC 控制器
│   ├── template_prob_simulator.py   # 概率仿真器
│   └── gp_trajectory.py             # GP 轨迹学习（GPyTorch SGPR）
│
├── reference_trajectory.py          # 公共模块：轨迹生成
│                                     # 支持：spiral（螺旋）、rollercoaster（过山车）
│
└── README.md                         # 本文档
```

### 模块依赖

- **公共模块**：`reference_trajectory.py`（两种方法共享）
- **标准 MPC**：`mpc/template_*.py` 模块
- **概率 MPC**：`prob_mpc/template_prob_*.py` 和 `prob_mpc/gp_trajectory.py`
- **外部依赖**：`do_mpc`、`numpy`、`casadi`、`matplotlib`、`gpytorch`、`torch`

## 数学公式

### 系统模型

两种方法使用相同的**一阶系统模型**（增量步进控制）：

**状态向量**（只包含位置）：
\[
\mathbf{x} = [p_x, p_y, p_z, ..., p_n]^T
\]
其中 \(n\) 是状态空间维度（默认3，可通过 `--dimension` 参数自定义）。

**控制输入**（期望速度，增量步进方式）：
\[
\mathbf{u} = [u_x, u_y, u_z, ..., u_n]^T
\]
其中 \(\mathbf{u}\) 表示期望速度，系统通过控制期望速度来实现位置控制。

**状态观测**：
- **位置**：完全可观测（笛卡尔空间位置）
- **速度**：系统不观测或控制速度状态，速度通过控制输入（期望速度）间接控制

**离散时间动力学**（欧拉积分，\(dt = 0.1\) s）：
\[
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{u}_k \cdot dt
\]

**注意**：这是一阶系统（\(\dot{\mathbf{p}} = \mathbf{u}\)），控制输入 \(\mathbf{u}\) 直接作为期望速度，系统通过增量步进的方式实现位置控制。这种设计避免了二阶系统可能产生的旋转问题，在势场中天然是梯度下降。

### 方法一：标准 MPC

#### 代价函数

预测步 \(k\) 的阶段代价为：

\[
\ell_k = Q_{\text{pos}} \|\mathbf{p}_k - \mathbf{p}_{\text{ref},k}\|^2 + Q_{\Delta u} \|\Delta \mathbf{u}_k\|^2 + \mathbf{u}_k^T R \mathbf{u}_k
\]

其中：
- \(\mathbf{p}_{\text{ref},k}\)：第 \(k\) 步的参考位置（通过 TVP 提供）
- \(\Delta \mathbf{u}_k = \mathbf{u}_k - \mathbf{u}_{k-1}\)：控制输入变化率（用于平滑性）
- \(Q_{\text{pos}} = 10.0\)：位置跟踪权重
- \(Q_{\Delta u} = 0.5\)：控制输入变化率惩罚权重（提高平滑性，减少抖动）
- \(R = 0.1 \cdot \mathbf{I}_n\)：控制输入惩罚矩阵（期望速度的平滑性惩罚）

**总代价**：
\[
J = \sum_{k=0}^{N-1} \ell_k
\]

终端代价 \(m_N = 0\)（弱终端代价，使用严格约束代替）。

**注意**：由于系统是一阶的（只有位置状态），代价函数中不包含速度跟踪项。

#### 终端约束

在预测步 \(N-1\) 处强制执行等式约束（确保到达目标位置）：

\[
\mathbf{p}_{N-1} = \mathbf{p}_{\text{target}}
\]

使用 `mpc.set_nl_cons()` 实现，设置 `lb = ub = 0.0` 和 `soft_constraint = False`。

**注意**：由于系统是一阶的（没有速度状态），终端约束只约束位置，不约束速度。

#### MPC 参数

- **预测时域**：\(N = 30\) 步
- **采样时间**：\(dt = 0.1\) s
- **总预测时间**：\(T = 3.0\) s

### 方法二：概率 MPC

#### 高斯过程轨迹模型

从多个示例轨迹中，GP 学习一个连续的轨迹分布：

\[
\mathbf{x}(t) \sim \mathcal{GP}(\boldsymbol{\mu}(t), \boldsymbol{\Sigma}(t))
\]

其中：
- \(\boldsymbol{\mu}(t) \in \mathbb{R}^3\)：均值轨迹（GP 预测）
- \(\boldsymbol{\Sigma}(t) = \text{diag}(\sigma_x^2(t), \sigma_y^2(t), \sigma_z^2(t))\)：对角协方差矩阵

**GP 实现**：使用 GPyTorch 的稀疏 GP 回归（SGPR），采用诱导点以提高可扩展性。

#### 代价函数

阶段代价使用马氏距离进行轨迹跟踪：

**轨迹项**（到 GP 均值的马氏距离）：
\[
\ell_{\text{traj},k} = (\mathbf{p}_k - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{p}_k - \boldsymbol{\mu}_k)
\]

其中 \(\boldsymbol{\Sigma}_k^{-1} = \text{diag}(1/\sigma_x^2, 1/\sigma_y^2, 1/\sigma_z^2, ...)\) 是对角协方差矩阵的逆。

**阶段代价**：
\[
\ell_k = \ell_{\text{traj},k} + \mathbf{u}_k^T R \mathbf{u}_k
\]

其中 \(R = 0.5 \cdot \mathbf{I}_n\) 是控制输入惩罚矩阵（期望速度的平滑性惩罚）。

**注意**：
- 由于系统是一阶的（只有位置状态），代价函数中不包含速度跟踪项
- 已移除目标项和终端约束，仅通过轨迹跟踪项实现跟踪（避免终端约束导致的轨迹变直线问题）

#### 终端约束

**当前实现**：概率 MPC 已**移除终端约束**，仅通过轨迹跟踪项实现跟踪。

**原因**：如果增加终点位置作为约束（硬约束或软约束），即使权重非常小，轨迹也会变成直线，这是代码实现的问题，等待修复。

**替代方案**：当前通过调整控制输入惩罚权重（\(R = 0.5\)）和轨迹跟踪项来间接影响终点行为。

## 安装

### 前置要求

- Python 3.8+
- Conda（推荐）

### 安装步骤

1. **创建 conda 环境**：
```bash
conda env create -f .conda.yml
conda activate do-mpc
```

2. **安装 do-mpc**（可编辑模式）：
```bash
cd /path/to/do-mpc
pip install -e .
```

3. **安装额外依赖**（用于概率 MPC）：
```bash
pip install gpytorch torch matplotlib
```

## 使用方法

### 命令行接口

统一入口点 `main.py` 支持两种方法：

```bash
python main.py [OPTIONS]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--method` | `str` | `mpc` | MPC 方法：`mpc`（标准）或 `prob_mpc`（概率） |
| `--trajectory-type` | `str` | `rollercoaster` | 轨迹类型：`spiral`（螺旋）或 `rollercoaster`（过山车） |
| `--num-replanning` | `int` | `None` | 重规划事件次数（默认：MPC 为 20，prob_mpc 为 0） |
| `--debug` | `flag` | `False` | 调试模式（仅用于 prob_mpc：将 GP 训练限制为 10 次迭代） |
| `--dimension` | `int` | `3` | 状态空间维度（默认3，表示3D空间。前3维永远是x, y, z用于可视化） |
| `--perfect-velocity` | `flag` | `False` | 假设速度完全可观测（默认：False。注意：当前系统是一阶的，不观测速度，此选项保留用于兼容性） |

#### 使用示例

**标准 MPC，螺旋轨迹**：
```bash
python main.py --method mpc --trajectory-type spiral --num-replanning 20
```

**概率 MPC，过山车轨迹**：
```bash
python main.py --method prob_mpc --trajectory-type rollercoaster --debug
```

**标准 MPC，过山车轨迹**：
```bash
python main.py --method mpc --trajectory-type rollercoaster
```

**标准 MPC，螺旋轨迹（默认行为）**：
```bash
python main.py --method mpc --trajectory-type spiral
```

**概率 MPC，5维空间**：
```bash
python main.py --method prob_mpc --trajectory-type rollercoaster --dimension 5
```

**概率 MPC，自定义 alpha_threshold**：
```bash
python main.py --method prob_mpc --trajectory-type rollercoaster --alpha-threshold 0.01
```

### 编程接口

#### 标准 MPC

```python
from mpc.template_model import template_model
from mpc.template_mpc import template_mpc, update_mpc_reference_trajectory
from mpc.template_simulator import template_simulator
from reference_trajectory import ReferenceTrajectoryGenerator

# 创建模型、MPC、仿真器
# 注意：model 是一阶系统，状态只有位置，控制输入是期望速度
model = template_model('SX', dimension=3)
traj_gen = ReferenceTrajectoryGenerator(trajectory_type='spiral')
mpc = template_mpc(model, traj_gen)
simulator = template_simulator(model)

# 仿真循环
# x_current 只包含位置：[p_x, p_y, p_z, ...]
# u0 是期望速度控制输入：[u_x, u_y, u_z, ...]
for k in range(n_steps):
    update_mpc_reference_trajectory(mpc, traj_gen, current_time, trajectory, time_stamps)
    u0 = mpc.make_step(x_current)
    y_next = simulator.make_step(u0)
    x_current = y_next
```

#### 概率 MPC

```python
from prob_mpc.template_prob_model import template_prob_model
from prob_mpc.template_prob_mpc import template_prob_mpc, update_mpc_gp_trajectory
from prob_mpc.template_prob_simulator import template_prob_simulator
from prob_mpc.gp_trajectory import GaussianProcessTrajectory
from reference_trajectory import ReferenceTrajectoryGenerator

# 从示例轨迹训练 GP
gp_traj = GaussianProcessTrajectory(normalize_time=True)
gp_traj.fit(normalized_trajectories, optimize=True, max_iters=200, num_inducing=100)

# 创建模型、MPC、仿真器
# 注意：model 是一阶系统，状态只有位置，控制输入是期望速度
model = template_prob_model('SX', dimension=3)
mpc = template_prob_mpc(model, gp_trajectory=gp_traj)
simulator = template_prob_simulator(model, gp_trajectory=gp_traj, trajectory_duration=10.0)

# 仿真循环
# x_current 只包含位置：[p_x, p_y, p_z, ...]
# u0 是期望速度控制输入：[u_x, u_y, u_z, ...]
# 注意：概率 MPC 已移除终端约束，仅通过轨迹跟踪项实现跟踪
for k in range(n_steps):
    t_normalized = current_time / trajectory_duration
    update_mpc_gp_trajectory(mpc, gp_traj, t_normalized, terminal_index=None, 
                             trajectory_duration=trajectory_duration, actual_end_mean=None)
    u0 = mpc.make_step(x_current)
    y_next = simulator.make_step(u0)
    x_current = y_next
```

## 方法对比

| 特性 | 标准 MPC | 概率 MPC |
|------|---------|---------|
| **系统模型** | 一阶系统（位置状态，期望速度控制） | 一阶系统（位置状态，期望速度控制） |
| **参考轨迹** | 确定性轨迹 | GP 学习的轨迹分布 |
| **代价度量** | 欧几里得距离 | 马氏距离 |
| **代价函数** | 位置跟踪 + 控制输入变化率惩罚 + 控制输入惩罚 | 轨迹跟踪（马氏距离）+ 控制输入惩罚 |
| **不确定性** | 未建模 | 通过 GP 方差显式建模 |
| **自适应权重** | 固定权重 | 已移除（当前仅使用轨迹跟踪项） |
| **终端约束** | 等式约束（位置） | 已移除（避免轨迹变直线问题） |
| **轨迹学习** | 手动生成 | 从多个示例中学习 |
| **计算成本** | 低 | 较高（GP 预测 + 优化） |
| **使用场景** | 已知参考轨迹 | 不确定/学习的轨迹模式 |

## 关键特性

### 1. 一阶系统模型（增量步进控制）

系统采用**一阶模型**，只观测和控制位置：

- **状态**：只有位置 `x = [p_x, p_y, p_z, ..., p_n]`
- **控制输入**：期望速度 `u = [u_x, u_y, u_z, ..., u_n]`（增量步进方式）
- **动力学**：`p_{k+1} = p_k + u_k * dt`

这种设计的优势：
- 避免二阶系统可能产生的旋转问题
- 在势场中天然是梯度下降（无curl）
- 更符合实际应用中只能观测位置的场景

**注意**：系统不观测或控制速度状态，速度通过控制输入（期望速度）间接控制。

### 2. 终端约束

**标准 MPC**：在终端步强制执行**数学等式约束**（而非软惩罚）：

\[
\mathbf{p}(T) = \mathbf{p}_{\text{target}}
\]

这保证了即使在多次重规划事件后，目标位置处的误差也为零。

**概率 MPC**：当前实现已移除终端约束，仅通过轨迹跟踪项实现跟踪。原因：如果增加终点位置作为约束，即使权重非常小，轨迹也会变成直线（代码问题，等待修复）。

### 3. 参考轨迹类型

#### 螺旋轨迹

- **数学形式**：参数化螺旋，半径衰减
- **参数**：`spiral_radius`、`num_turns`、`noise_scale`
- **收敛性**：轨迹在端点附近收敛到相同段

#### 过山车轨迹

- **数学形式**：圆弧 + 平滑过渡到终点
- **参数**：`circle_radius`、`circle_plane`、`circle_ratio`
- **特性**：方差更小，曲线更平滑

### 4. 实时重规划

标准 MPC 支持仿真过程中的动态目标更新：
- 目标终点从与参考轨迹相同的分布中采样
- 从当前位置到新目标重新生成轨迹
- MPC 平滑适应目标变化

### 5. 性能优化

概率 MPC 包括：
- **批量 GP 预测**：一次调用预测所有时域点
- **稀疏 GP (SGPR)**：使用诱导点以提高可扩展性
- **GPU 加速**：通过 GPyTorch 自动支持 CUDA

## 可视化输出

两种方法都生成全面的可视化：

1. **3D 轨迹图**：完整参考轨迹、实际执行轨迹、起点/终点
2. **X-Y 平面投影**：轨迹跟踪的 2D 视图
3. **位置跟踪误差**：误差随时间的变化
4. **控制输入历史**：期望速度命令随时间的变化

**概率 MPC 额外显示**：
5. **GP 方差（不确定性）**：不确定性随时间的演化
6. **轨迹权重 \(\alpha(t)\)**：基于不确定性的动态权重（当前实现中主要用于记录，代价函数仅使用轨迹跟踪项）

## 理论背景

本项目实现了：

1. **滚动时域控制 (RHC)**：滚动时域优化
2. **参考轨迹跟踪**：时间参数化轨迹跟随
3. **时变参数 (TVP)**：动态参考更新
4. **高斯过程回归**：概率性轨迹学习（概率 MPC）
5. **马氏距离**：不确定性感知的代价函数（概率 MPC）

## 参数调优

### 标准 MPC

| 参数 | 默认值 | 效果 |
|------|--------|------|
| `Q_pos` | 10.0 | 位置跟踪权重（增大以获得更紧密的跟踪） |
| `Q_Δu` | 0.5 | 控制输入变化率惩罚权重（增大以提高平滑性，减少抖动） |
| `R` | 0.1 | 控制输入惩罚（期望速度的平滑性惩罚，减小以允许更大的控制动作） |
| `n_horizon` | 30 | 预测时域（增大以更好地保持轨迹形状） |

**注意**：由于系统是一阶的，不再有速度跟踪权重参数。

### 概率 MPC

| 参数 | 默认值 | 效果 |
|------|--------|------|
| `alpha_threshold` | 0.0001 | Alpha权重计算的阈值参数（较小的值：更强调目标项；较大的值：更强调轨迹跟踪） |
| `R` | 0.5 | 控制输入惩罚（期望速度的平滑性惩罚，从0.1增加到0.5以减少"直接冲向目标"的倾向） |
| `num_inducing` | 100 | GP 诱导点（增大以提高精度，减小以提高速度） |
| `max_iters` | 200 | GP 优化迭代次数 |

**注意**：概率 MPC 已移除目标项和终端约束，仅通过轨迹跟踪项实现跟踪。

## 已知问题

### 标准 MPC

- **速度改变后的不确定性问题**：在应用速度改变后，标准 MPC 可能存在不确定的问题，可能需要修正。具体表现和原因待进一步调查。

### 概率 MPC

- **终点约束问题**：如果增加终点位置作为约束（硬约束或软约束），即使权重非常小（如 \(10^{-6}\)），轨迹也会变成直线。这是代码实现的问题，等待修复。当前实现已移除终端约束，仅通过轨迹跟踪项实现跟踪。

## 参考文献

- **do-mpc**： [https://www.do-mpc.com/](https://www.do-mpc.com/)
- **GPyTorch**： [https://gpytorch.ai/](https://gpytorch.ai/)
- **CasADi**： [https://web.casadi.org/](https://web.casadi.org/)

## 许可证

参见父项目许可证。
