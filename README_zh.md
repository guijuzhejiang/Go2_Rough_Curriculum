# 复杂地形速度跟随步态训练

> 📄 [English Documentation](README.md)

## 项目概述

本项目基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) v2.3.2，使用 PPO 算法（[RSL-RL](https://github.com/leggedrobotics/rsl_rl) 框架）训练宇树 Go2 四足机器人在复杂地形中的速度跟随运动控制策略。

项目的核心目标是训练出可以泛化到真实世界各种地形（平地、斜坡、台阶、粗糙地面等）的通用步态控制器，而非仅针对单一理想地形的过拟合策略。

### 训练演示

![](assets/demo.gif)

---

## 核心特性

### 🏔️ 混合地形训练 · 提升真实世界泛化能力

智能体在程序化生成的复合地形（`COBBLESTONE_ROAD_CFG`）中训练，七种地形类型在同一场景中混合出现：

| 地形类型 | 占比 | 说明 |
|---|---|---|
| 平坦地面 | 20% | 基础步行 |
| 平滑斜坡 | 15% | 上坡 / 下坡 |
| 反向斜坡 | 10% | 倒置金字塔斜面 |
| 随机粗糙面 | 15% | 随机高度场，模拟碎石路 |
| 金字塔台阶 | 20% | 规则台阶攀爬 |
| 反向台阶 | 20% | 下行台阶地形 |
| 离散障碍物 | 可选 | 散布障碍物 |

在多样化地形中同时训练，可以引导策略学习出鲁棒的、可迁移的运动行为，从而在实际部署中取得更好的泛化效果。

---

### 📈 课程学习 · 逐步提升训练效率

为了防止策略被过难任务淹没，同时充分挖掘学习潜力，项目采用了两级课程学习机制：

#### 1. 地形难度课程（`terrain_levels_vel`）

机器人根据每个 episode 内的行走距离动态调整地形难度：
- **走得足够远** → 晋升到更难地形
- **走得不足一半** → 降级到更简单地形

这种机制使训练初期从简单地形开始积累基础能力，再逐步接触高难度场景。

#### 2. 速度指令课程（`lin_vel_cmd_levels`）

速度指令范围随策略表现动态扩展：
- 初始阶段速度范围保守（如 0.2 ~ 1.2 m/s）
- 当跟踪奖励达到阈值后，自动扩大速度范围（最高可达 6.0 m/s）

这使策略能够循序渐进地学习从慢走到快跑的全速域控制。

---

### 🔄 迁移学习 · 加载平地预训练模型

在复杂地形上从零训练难度极大。本项目采用**两阶段训练策略**：

- **第一阶段**：在平坦地形上训练，让策略先掌握基础步态节律、平衡控制和速度跟踪能力。
- **第二阶段**：将平地训练的 checkpoint 作为预训练权重加载，在混合地形上微调，快速迁移已有能力并适应复杂场景。

加载预训练模型的训练命令：

```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096 \
    --pretrained=/path/to/flat_terrain/model_XXXX.pt
```

两阶段策略相比从零训练可以显著缩短收敛时间，并降低策略陷入局部最优的风险。

---

### 🔧 RSL-RL v5.x 兼容适配层

RSL-RL v5.0 对配置 API 和 checkpoint 格式进行了重大重构，与旧版不兼容。为了在不修改 Isaac Lab 官方模板的前提下支持新版本，本项目在 `scripts/rsl_rl/train.py` 中实现了一套本地兼容适配层。

#### 配置格式适配（`handle_deprecated_rsl_rl_cfg`）

自动将旧版 `policy` 字段格式转换为 v5.x 所需的 `actor` / `critic` 分离格式：

```python
# 旧版格式（< v5.0）：
cfg["policy"] = {
    "actor_hidden_dims": [512, 256, 128],
    "init_noise_std": 1.0,
    ...
}

# 新版格式（v5.x）：
cfg["actor"] = {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], ...}
cfg["critic"] = {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], ...}
```

同时自动处理 `GaussianDistribution` / `HeteroscedasticGaussianDistribution` 等分布配置的映射。

#### Checkpoint 格式适配（`handle_deprecated_rsl_rl_checkpoint`）

自动将旧版扁平权重文件（单一 `model_state_dict`）转换为 v5.x 所需的 `actor_state_dict` / `critic_state_dict` 分离格式，并对 `MLPModel` 内部键名做精确重映射：

| 旧版权重键名 | v5.x MLPModel 期望键名 |
|---|---|
| `"0.weight"` | `"mlp.0.weight"` |
| `"0.bias"` | `"mlp.0.bias"` |
| `"std"` | `"distribution.std_param"` |
| `"actor.X.weight"` | `"mlp.X.weight"` |

以上适配均在运行时自动完成，无需手动转换权重文件。

---

## AER 奖励设计

本项目实现了论文中的 **AER（Adaptive Exponential Reward，自适应指数奖励）** 结构。有别于线性加权求和，负向惩罚项通过指数门控乘到正向奖励上：

```
R_总奖励 = R_正向 × exp(R_负向 / σ)
```

- **正向项**：速度跟踪（线速度、角速度）、能量效率奖励
- **负向项**：关节力矩过大、关节速度过大、关节加速度过大、动作变化率过大、足部打滑、躯干姿态偏差、机身高度偏差等

这种设计在任务完成度和运动质量之间形成自然的权衡机制，而非简单的线性累加，能更有效地引导策略同时优化运动效果和物理合理性。

---

## 安装

**前置条件**：已安装 Isaac Lab v2.3.2，推荐使用 conda 环境。

```bash
# 1. 克隆项目（需在 IsaacLab 目录之外）
git clone <repo-url>
cd <project-dir>

# 2. 激活 conda 环境并安装项目扩展
conda activate <your-conda-env>
python -m pip install -e source/go2_demo

# 3. 验证安装
python scripts/list_envs.py
```

---

## 训练与评估

### 标准训练（混合地形）

```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096
```

### 基于预训练模型的迁移学习训练

```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096 \
    --pretrained=/path/to/flat_terrain/model_XXXX.pt
```

### 加载 checkpoint 继续训练（--resume）

```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --resume
```

### 评估 / 可视化推理

```bash
python scripts/rsl_rl/play.py \
    --task=<TASK_NAME> \
    --num_envs=16
```

### 调试用 Dummy Agent

```bash
# 零动作（验证环境配置是否正常）
python scripts/zero_agent.py --task=<TASK_NAME>

# 随机动作
python scripts/random_agent.py --task=<TASK_NAME>
```

---

## 项目结构

```
<project-root>/
├── README.md                  # 英文文档
├── README_zh.md               # 中文文档（本文件）
├── scripts/
│   ├── list_envs.py
│   ├── zero_agent.py
│   ├── random_agent.py
│   └── rsl_rl/
│       ├── train.py           # 训练入口（含 RSL-RL v5.x 兼容层）
│       └── play.py            # 评估推理脚本
└── source/
    └── go2_demo/
        └── go2_demo/
            ├── assets/
            │   └── robot/unitree.py          # Go2 机器人资产配置
            └── tasks/manager_based/go2_demo/
                ├── go2_demo_velocity.py      # 主环境配置（地形/观测/奖励）
                ├── aer_env.py                # AER 奖励管理器
                └── mdp/
                    ├── curriculums.py        # 地形与速度课程
                    └── rewards.py            # 自定义奖励函数
```
