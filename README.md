<div align="center">
  <h1 align="center">Unitree GO2 GYM --harryma</h1>
  <p align="center">
 
  </p>
</div>
 
---
 
## 📌 算法与任务总览

### Panda 机器人任务

Panda 使用 `resources/robots/panda/urdf/panda3v2.urdf`，初始配置参考对应 Go2 任务，并使用 Panda 专用 PD 参数。任务按功能分别放在 `PANDA_Stand/`、`PANDA_Flip/` 和 `Panda_MoB/` 下；每个任务都拥有完整、独立的环境与配置文件，后续调整无需查看或修改 Go2 配置。

| task 名 | 功能 |
|---|---|
| `panda_handstand` | 前肢支撑、后腿抬起 |
| `panda_leggedstand` | 后肢支撑、前腿抬起 |
| `panda_jump` | 原地或带速度指令跳跃 |
| `panda_spring_jump` | 蓄力式弹跳 |
| `panda_backflip` | 后空翻及落地控制 |

```bash
python legged_gym/scripts/train.py --task=panda_handstand --headless
python legged_gym/scripts/train.py --task=panda_leggedstand --headless
python legged_gym/scripts/train.py --task=panda_jump --headless
python legged_gym/scripts/train.py --task=panda_spring_jump --headless
python legged_gym/scripts/train.py --task=panda_backflip --headless
```

训练完成后可用同一个标准 PPO Play 脚本验证并导出策略，例如：

```bash
python legged_gym/scripts/play.py --task=panda_handstand
python legged_gym/scripts/play.py --task=panda_leggedstand
python legged_gym/scripts/play.py --task=panda_jump
python legged_gym/scripts/play.py --task=panda_spring_jump
python legged_gym/scripts/play.py --task=panda_backflip
```
 
### 算法简介
- **标准 PPO**：ETH legged_gym 原版 PPO，用于各类特技动作（trot/jump/backflip/handstand 等）。
- **DreamwaQ**：基于隐式地形想象（CENet/VAE）的鲁棒四足运动，从历史观测编码出隐式地形 latent + 显式线速度估计。参考论文 [Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602)。
- **AMP（Adversarial Motion Priors）**：用判别器引导策略学习参考动作（mocap 数据），让运动风格更自然。参考 [rl_amp](https://github.com/fan-ziqi/rl_amp)。
- **CTS（Concurrent Teacher-Student）**：师生并行训练 + 隐变量蒸馏，teacher 吃 privileged obs，student 吃 history obs。参考论文 [arxiv 2405.10830](https://arxiv.org/abs/2405.10830)。
 
---
 
## 🔁 流程说明
 
强化学习实现运动控制的基本流程为：
 
`Train` → `Play` → `Sim2Sim` → `Sim2Real`
 
- **Train**: 通过 Gym 仿真环境，让机器人与环境互动，找到最满足奖励设计的策略。通常不推荐实时查看效果，以免降低训练效率。
- **Play**: 通过 Play 命令查看训练后的策略效果，确保策略符合预期。
- **Sim2Sim**: 将 Gym 训练完成的策略部署到其他仿真器，避免策略小众于 Gym 特性。
- **Sim2Real**: 将策略部署到实物机器人，实现运动控制。
 
## 🛠️ 使用指南
### 0. 安装依赖
python环境：3.8
#### Isaacgym 安装
 
```bash
pip install -e ~/isaacgym/python
```
#### rsl_rl 算法库（含 PPO / DreamwaQ / AMP / CTS 全部算法）
```bash
pip install -e ./rsl_rl
```
#### legged_gym
```bash
pip install -e .
```

> 如果环境使用为 RTX 50 系显卡定制编译的 PyTorch（例如包含 `sm_120` 的 CUDA 12.8 构建），请加 `--no-deps`，避免 pip 用官方旧架构 wheel 覆盖定制版本：
> `pip install -e ./rsl_rl --no-deps && pip install -e . --no-deps`
#### AMP 额外依赖（仅 AMP 系列任务需要）
```bash
pip install pybullet   # AMP 动作数据加载（pybullet_utils.transformations）
```
 
### 1. 训练
 
运行以下命令进行训练（所有命令需在仓库根目录下执行，AMP 任务依赖 `datasets/` 相对路径）：
 
#### 标准 PPO 任务
```bash
python legged_gym/scripts/train.py --task=go2_trot --headless
python legged_gym/scripts/train.py --task=go2_jump --headless
python legged_gym/scripts/train.py --task=go2_handstand --headless
python legged_gym/scripts/train.py --task=go2_leggedstand --headless
python legged_gym/scripts/train.py --task=go2_spring_jump --headless
python legged_gym/scripts/train.py --task=go2_backflip --headless
```
 
#### DreamwaQ 任务
```bash
python legged_gym/scripts/train.py --task=go2_stairs_dreamwaq --headless
```
 
#### AMP + DreamwaQ 任务
```bash
python legged_gym/scripts/train.py --task=go2_amp_dreamwaq --headless
```
 
#### CTS 任务
```bash
python legged_gym/scripts/train.py --task=go2_cts --headless
```
 
#### AMP + CTS 任务
```bash
python legged_gym/scripts/train.py --task=go2_amp_cts --headless

#### AMP Teacher-Student 任务（先训 teacher 再蒸馏 student）
python legged_gym/scripts/train.py --task=go2_amp_ts --headless          # teacher（AMP 特权策略）
python legged_gym/scripts/train.py --task=go2_amp_ts_student --headless  # student（从 teacher 蒸馏 LSTM）
```
 
 
#### ⚙️  参数说明
- `--task`: 必选参数，可选值见上方任务总览表
- `--headless`: 默认启动图形界面，设为 true 时不渲染图形界面（效率更高）
- `--resume`: 从日志中选择 checkpoint 继续训练
- `--experiment_name`: 运行/加载的 experiment 名称
- `--run_name`: 运行/加载的 run 名称
- `--load_run`: 加载运行的名称，默认加载最后一次运行
- `--checkpoint`: checkpoint 编号，默认加载最新一次文件
- `--num_envs`: 并行训练的环境个数
- `--seed`: 随机种子
- `--max_iterations`: 训练的最大迭代次数
- `--sim_device`: 仿真计算设备，指定 CPU 为 `--sim_device=cpu`
- `--rl_device`: 强化学习计算设备，指定 CPU 为 `--rl_device=cpu`
 
**默认保存训练结果**：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`
 
> 各算法的训练日志已按算法分类存放在 `logs/` 下的子目录（`dreamwaq/`、`amp_dreamwaq/`、`amp_cts/`），便于回溯。
 
---
 
### 2. Play
 
不同算法使用对应的 play 脚本（因为各算法的推理/导出逻辑不同）：
 
#### 标准 PPO 任务
```bash
python legged_gym/scripts/play.py --task=go2_trot
python legged_gym/scripts/play.py --task=go2_jump
python legged_gym/scripts/play.py --task=go2_handstand
python legged_gym/scripts/play.py --task=go2_leggedstand
python legged_gym/scripts/play.py --task=go2_spring_jump
python legged_gym/scripts/play.py --task=go2_backflip
```
 
#### DreamwaQ / AMP + DreamwaQ 任务
```bash
python legged_gym/scripts/play_dreamwaq.py --task=go2_stairs_dreamwaq
python legged_gym/scripts/play_dreamwaq.py --task=go2_amp_dreamwaq
```
 
#### CTS 任务
```bash
python legged_gym/scripts/play_cts.py --task=go2_cts
```
 
#### AMP + CTS 任务
```bash
python legged_gym/scripts/play_amp_cts.py --task=go2_amp_cts

#### AMP Teacher-Student 任务
python legged_gym/scripts/play_amp_ts.py --task=go2_amp_ts
python legged_gym/scripts/play_amp_ts_student.py --task=go2_amp_ts_student
```
 
#### AMP 动作数据回放（查看 mocap 参考动作）
```bash
python legged_gym/scripts/replay_amp_data.py --task=go2_amp_cts
```
 
**说明**：
 
- Play 启动参数与 Train 相同。
- 默认加载实验文件夹上次运行的最后一个模型。
- 可通过 `load_run` 和 `checkpoint` 指定其他模型。
 
#### 💾 导出网络
 
Play 会导出 Actor 网络，保存于 `logs/{experiment_name}/exported/policies` 中：
- 普通网络（MLP）导出为 `policy_1.pt`
- DreamwaQ 网络通过 `export_policy_as_dwaq` 导出为 `policy_dwaq.pt`（含 VAE/CENet）
- CTS 网络通过 `export_policy_as_cts` 导出为 `policy_cts.pt`（含 student encoder）

---

### 3. Sim2Sim（MuJoCo 部署验证）

将训练好的策略部署到 MuJoCo 仿真器验证效果。不同算法使用对应的 sim2sim 脚本（位于 `deploy_mujoco/`）：

#### 标准 PPO 任务
```bash
python deploy_mujoco/sim2sim_GO2_trot_viewer.py
python deploy_mujoco/sim2sim_GO2_jump_viewer.py
python deploy_mujoco/sim2sim_GO2_spring_jump_viewer.py
python deploy_mujoco/sim2sim_GO2_backflip_viewer.py
python deploy_mujoco/sim2sim_handstand_viewer.py
python deploy_mujoco/sim2sim_legstand_viewer.py
```

#### DreamwaQ 任务（加载 policy_dwaq.pt）
```bash
python deploy_mujoco/sim2sim_GO2_stairs_dreamwaq_viewer.py
```

#### AMP + DreamwaQ 任务（加载 policy_dwaq.pt）
```bash
python deploy_mujoco/sim2sim_GO2_amp_dreamwaq_viewer.py
```

#### CTS 任务（加载 policy_cts.pt）
```bash
python deploy_mujoco/sim2sim_GO2_cts_viewer.py
```

#### AMP + CTS 任务（加载 policy_cts.pt）
```bash
python deploy_mujoco/sim2sim_GO2_amp_cts_viewer.py
```

#### AMP Teacher-Student 任务（student LSTM 部署，加载 policy_amp_ts.pt）
```bash
python deploy_mujoco/sim2sim_GO2_amp_ts_viewer.py
```

**说明**：
- 各脚本默认从 `logs/<experiment_name>/exported/policies/` 加载导出的策略（需先运行对应 Play 脚本导出）。
- 可通过 `--load_model` 参数指定其他策略文件路径。
 
---
 
## 📂 仓库结构
 
```
.
├── legged_gym/
│   ├── envs/
│   │   ├── base/                 # 基类（BaseTask / LeggedRobotCfg 等）
│   │   ├── Go2_MoB/              # 标准 PPO: trot / stairs / jump
│   │   ├── GO2_Flip/             # 标准 PPO: backflip / spring_jump
│   │   ├── GO2_Stand/            # 标准 PPO: handstand / leggedstand
│   │   ├── GO2_DreamWaQ/         # DreamwaQ 任务
│   │   ├── GO2_AMP_DreamWaQ/     # AMP + DreamwaQ 任务
│   │   ├── GO2_CTS/              # CTS 任务
│   │   ├── GO2_AMP_CTS/          # AMP + CTS 任务
│   │   ├── GO2_AMP_TS/           # AMP Teacher-Student 任务（teacher + student 配置）
│   │   └── __init__.py           # 注册全部 12 个 task
│   ├── scripts/                  # train.py + 各算法的 play 脚本
│   └── utils/                    # task_registry（动态 runner 调度）/ helpers / terrain
├── rsl_rl/                       # 算法库（PPO / DreamwaQ / AMP / CTS 全部共存）
│   └── rsl_rl/
│       ├── algorithms/           # PPO, PPO_DreamWaQ, PPO_DreamWaQ_AMP, CTS, AMPCTS, AMPDiscriminator
│       ├── modules/              # ActorCritic, ActorCriticDreamWaQ, ActorCriticCTS, VAE
│       ├── runners/              # OnPolicyRunner, DreamWaQRunner, OnPolicyRunnerCTS 等
│       ├── storage/              # RolloutStorage, RolloutStorageDreamWaQ, RolloutStorageCTS, ReplayBuffer
│       └── datasets/             # AMPLoader（AMP 动作数据加载）
├── datasets/                     # AMP 动作捕捉数据（mocap_motions_go2 / mocap_motions_a1）
├── deploy_mujoco/                # Sim2Sim 部署脚本（11 个：标准 PPO 6 + CTS 2 + DreamwaQ/AMP 2 + AMP_TS 1）+ 手柄/箭头工具
├── resources/                    # 机器人资产（urdf / mjcf / mesh）
└── logs/                         # 训练日志（按算法分类）
```
 
---
 
# TODO List
- [ ] jump任务原地转圈不行
- [ ] stand 任务关节抖动，静止时后退，没有下落控制，粗糙地面没有加入训练
- [ ] 跳远要修改为可以在走路的时候直接切换以及连跳，以及可以控制下落位置和起跳高度
- [ ] DreamwaQ 下楼梯不稳定、平地走歪
- [ ] AMP + DreamwaQ  不稳定
- [ ] CTS / AMP + CTS 不稳定，开发中
- [ ] parkour (PIE)
- [ ] backflip / sideflip
- [ ] 部署代码

# 参考文章
https://arxiv.org/pdf/2205.02824
 
https://arxiv.org/pdf/2309.05665
 
https://arxiv.org/abs/2212.03238
 
https://arxiv.org/abs/2409.15755
 
https://arxiv.org/abs/2401.16337
 
https://arxiv.org/abs/2301.10602
 
https://arxiv.org/abs/2312.11460
