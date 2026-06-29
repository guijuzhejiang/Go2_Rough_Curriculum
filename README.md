# Adaptive Gait — Velocity Tracking on Complex Terrain

> 📄 [中文文档](README_zh.md)

## Overview

This is a reinforcement learning project for training robust velocity-tracking locomotion policies on the Unitree Go2 quadruped robot, built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) v2.3.2 and trained with [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (PPO).

The core objective is to train a generalizable gait controller that can follow velocity commands across diverse real-world terrains — flat ground, slopes, stairs, and rough surfaces — rather than overfitting to a single idealized terrain.

---

## Key Features

### 🏔️ Mixed Terrain Training for Real-World Generalization

The agent is trained on a procedurally-generated composite terrain (`COBBLESTONE_ROAD_CFG`) that combines 7 terrain types in a single environment:

| Terrain Type | Proportion | Description |
|---|---|---|
| Flat plane | 20% | Baseline locomotion |
| Smooth slope | 15% | Uphill / downhill |
| Inverted slope | 10% | Reverse pyramid slope |
| Random rough | 15% | Stochastic height field |
| Pyramid stairs | 20% | Regular stair climbing |
| Inverted stairs | 20% | Descending step terrain |
| Discrete obstacles| (optional) | Scattered obstacles |

Training on diverse terrains simultaneously encourages the policy to learn robust, generalizable behaviors that transfer well to real-world deployment.

---

### 📈 Curriculum Learning

To progressively increase training difficulty and improve learning efficiency, two curriculum strategies are applied:

1. **Terrain Curriculum** (`terrain_levels_vel`): Robots that successfully traverse a terrain sub-tile are promoted to harder difficulty levels; robots that fail are demoted. This prevents the policy from being overwhelmed early in training.

2. **Velocity Command Curriculum** (`lin_vel_cmd_levels`): The commanded velocity range starts conservatively and expands as the policy's tracking performance improves, allowing the agent to incrementally learn to run faster.

---

### 🔄 Transfer Learning via Pretrained Models

Rather than training from scratch on complex terrain, the policy is initialized from a pretrained model obtained by training on flat ground. This **two-stage curriculum**:

- **Stage 1**: Train on flat terrain to acquire basic locomotion primitives (gait rhythm, balance, velocity tracking).
- **Stage 2**: Load the flat-terrain checkpoint as a pretrained model and fine-tune on the mixed terrain environment.

To load a pretrained checkpoint during training:

```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --pretrained=/path/to/flat_terrain_checkpoint.pt
```

---

### 🔧 RSL-RL v5.x Compatibility Layer

RSL-RL v5.0+ introduced significant breaking changes to its configuration API and checkpoint format. Since the original Isaac Lab template was written against older versions, this project includes a local compatibility shim in `scripts/rsl_rl/train.py`.

#### Configuration Compatibility (`handle_deprecated_rsl_rl_cfg`)

Automatically remaps the old `policy`-based agent configuration dict into the new `actor`/`critic` split format expected by `rsl-rl-lib >= 5.0`:

```python
# Old format (< v5.0):
cfg["policy"] = {"actor_hidden_dims": [...], "init_noise_std": 1.0, ...}

# New format (v5.x):
cfg["actor"] = {"class_name": "MLPModel", "hidden_dims": [...], ...}
cfg["critic"] = {"class_name": "MLPModel", "hidden_dims": [...], ...}
```

#### Checkpoint Compatibility (`handle_deprecated_rsl_rl_checkpoint`)

Automatically converts old-format `.pt` checkpoint files (single `model_state_dict`) into the v5.x format (separate `actor_state_dict` / `critic_state_dict`), including key remapping for `MLPModel`'s internal structure:

```
Old key                 →  New key (MLPModel)
────────────────────────────────────────────────
"0.weight"              →  "mlp.0.weight"
"std"                   →  "distribution.std_param"
```

These adapters run automatically and require no manual intervention.

---

## AER Reward Shaping

This project implements the **AER (Adaptive Exponential Reward)** reward structure from the locomotion research literature. Rather than summing all reward terms linearly, penalties are applied multiplicatively via an exponential gate:

```
R_total = R_positive × exp(R_negative / σ)
```

- **Positive terms**: `track_lin_vel_xy`, `track_ang_vel_z`, `energy_efficiency`
- **Negative terms**: joint torque, velocity, acceleration penalties; foot slip; orientation error; body height deviation

This design encourages the robot to prioritize task completion while softly penalizing physically undesirable behaviors.

---

## Installation

1. Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. Clone this repository **outside** the IsaacLab directory:
   ```bash
   git clone <repo-url>
   cd <project-dir>
   ```

3. Install the project extension in editable mode:
   ```bash
   python -m pip install -e source/go2_demo
   ```

4. Verify installation by listing available tasks:
   ```bash
   python scripts/list_envs.py
   ```

---

## Training

### Standard Training (Mixed Terrain)
```bash
python scripts/rsl_rl/train.py --task=<TASK_NAME> --num_envs=4096
```

### Training with Pretrained Flat-Terrain Model
```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096 \
    --pretrained=/path/to/flat_terrain/model_XXXX.pt
```

### Resume Training from Checkpoint
```bash
python scripts/rsl_rl/train.py --task=<TASK_NAME> --resume
```

### Playback / Evaluation
```bash
python scripts/rsl_rl/play.py --task=<TASK_NAME> --num_envs=16
```

### Dummy Agent Testing
```bash
# Zero-action agent (verify environment setup)
python scripts/zero_agent.py --task=<TASK_NAME>

# Random-action agent
python scripts/random_agent.py --task=<TASK_NAME>
```

---

## Project Structure

```
<project-root>/
├── scripts/
│   └── rsl_rl/
│       ├── train.py          # Training entry point (+ RSL-RL v5.x compat layer)
│       └── play.py           # Playback / evaluation script
└── source/
    └── go2_demo/
        └── go2_demo/
            ├── assets/
            │   └── robot/unitree.py       # Go2 robot asset config
            └── tasks/manager_based/go2_demo/
                ├── go2_demo_velocity.py   # Main env config (terrains, obs, rewards)
                ├── aer_env.py             # AER reward manager
                └── mdp/
                    ├── curriculums.py     # Terrain & velocity curriculum
                    └── rewards.py         # Custom reward functions
```
