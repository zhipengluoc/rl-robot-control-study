# 12 周学习路线图

> 总体策略：**先把 RL 算法吃透，再叠加机器人仿真，最后做精一个项目用来求职**。
> 每阶段都有一个"硬性产出"——做不完不进入下一阶段。

---

## 阶段 1 ｜ W1-W2 ｜ RL 基础复盘与重写

**为什么从这里开始**：你说有 CartPole 级别的基础，但具身智能面试里 PPO/SAC/TD3 的细节是必考点（GAE、advantage normalization、entropy bonus、target network、policy collapse 等）。这两周把自己写过的算法变成"能讲清楚每一行代码为什么这么写"。

### 学习目标

- 重新看懂 PPO 论文和 SAC 论文，能默写出 loss function
- 在 Gymnasium 经典控制（CartPole、Pendulum、LunarLander）上手写 PPO 跑通
- 用 Stable-Baselines3 在 MuJoCo `HalfCheetah-v4`、`Ant-v4` 上训出能跑的策略
- 读完 CleanRL 的 ppo.py 和 sac.py，理解"工业级单文件实现"长啥样

### 推荐资料

- **必读论文**：PPO（Schulman 2017）、SAC（Haarnoja 2018）、GAE（Schulman 2016）
- **代码**：[CleanRL](https://github.com/vwxyzjn/cleanrl)（每个算法一个文件，几百行，最佳教学代码）
- **课程**（任选）：UC Berkeley CS285（Sergey Levine）、Spinning Up in Deep RL（OpenAI）
- **博客**：[The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)（**强烈推荐**，求职被问 PPO 细节就靠这篇）

### 硬性产出

- [ ] `week01_02_foundations/ppo_minimal.py` —— 自己写的 PPO，能在 `Pendulum-v1` 收敛
- [ ] `week01_02_foundations/sb3_mujoco.py` —— SB3 在 HalfCheetah 上的训练脚本，附带学习曲线截图
- [ ] `week01_02_foundations/notes.md` —— 200-500 字总结：PPO 与 SAC 的区别 / GAE 怎么算 / 为什么要 advantage normalization

### 时间分配（按 15 小时/周）

| 任务 | 周 1 | 周 2 |
|---|---|---|
| 看论文 + 笔记 | 4h | 2h |
| CleanRL 代码精读 | 3h | 2h |
| 手写 PPO | 6h | 4h |
| SB3 + MuJoCo 实验 | - | 5h |
| 周报 / 整理 | 2h | 2h |

---

## 阶段 2 ｜ W3-W4 ｜ MuJoCo + robosuite 操作任务

**为什么**：操作（manipulation）和移动（locomotion）是机器人 RL 的两条主线。先做操作，因为 robosuite 任务清晰、可视化直观、训练时间短，适合建立"我能训出一个机器人策略"的信心。

### 学习目标

- 理解 MuJoCo 的 XML 模型描述（body / joint / actuator / sensor）
- 能自己写一个简单的 MuJoCo 环境并和 Gymnasium 接口对接
- 用 robosuite 训出 `Lift`（机械臂把方块举起）任务
- 第一次接触**域随机化（Domain Randomization）**的概念（friction、mass、camera pose 扰动）

### 推荐资料

- **官方文档**：[MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)、[robosuite docs](https://robosuite.ai/docs/)
- **教程**：[MuJoCo tutorial Colab](https://github.com/google-deepmind/mujoco/tree/main/python)（DeepMind 官方 notebook）
- **论文**：Domain Randomization（Tobin 2017）、Asymmetric Actor-Critic（Pinto 2018）

### 硬性产出

- [ ] `week03_04_mujoco_robosuite/mujoco_hello.py` —— 加载一个简单 MJCF 模型，渲染一帧
- [ ] `week03_04_mujoco_robosuite/robosuite_lift_sac.py` —— SAC 在 Lift 上训练
- [ ] `week03_04_mujoco_robosuite/dr_experiment.md` —— 一个小实验：开 vs 不开域随机化，对比泛化性
- [ ] 学习曲线 + 训练后视频（mp4 或 gif），保存到 `week03_04_mujoco_robosuite/results/`

### 时间分配

| 任务 | 周 3 | 周 4 |
|---|---|---|
| MuJoCo XML + Python API | 5h | - |
| robosuite 上手 | 4h | 3h |
| Lift 任务训练 | 4h | 5h |
| 域随机化小实验 | - | 5h |
| 论文 / 笔记 | 2h | 2h |

---

## 阶段 3 ｜ W5-W7 ｜ MJX + 四足 Locomotion（**重头戏**）

**为什么**：四足 locomotion 是过去 3 年（2022-2025）整个领域最火的方向，几乎所有主流公司（Unitree、Boston Dynamics、字节、银河通用、星动纪元、宇树等）都在做。MJX（MuJoCo XLA）+ JAX 是 2024-2025 年新的"事实标准"，能在 Mac 的 GPU 上跑大规模并行环境训练。

### 学习目标

- 入门 JAX：纯函数 / `jit` / `vmap` / `pmap` / PyTree
- 用 MuJoCo Playground 跑通官方四足任务（Go1 / Go2 / Spot）
- 看懂 locomotion 训练里的关键技巧：privileged learning、actuator network、 observation noise、curriculum
- 自己改一个任务（比如改地形 / 改奖励函数 / 改观测）

### 推荐资料

- **JAX 入门**：[JAX 官方 quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)、[The Annotated S4](https://srush.github.io/annotated-s4/)（不为学 S4，看 JAX 范式）
- **代码**：[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)（DeepMind 出，2025 年最新教学项目）
- **论文**：
  - "Learning to walk in minutes using massively parallel deep reinforcement learning"（Rudin 2022，IsaacGym 那篇，但思想通用）
  - "Learning quadrupedal locomotion over challenging terrain"（Lee 2020）
  - "RMA: Rapid Motor Adaptation for Legged Robots"（Kumar 2021，sim2real 经典）
  - "Walk These Ways"（Margolis 2022，行为多样性）

### 硬性产出（重要！求职项目候选）

- [ ] `week05_07_locomotion_mjx/jax_basics.ipynb` —— 自己整理的 JAX 入门 notebook
- [ ] `week05_07_locomotion_mjx/playground_go2_baseline.py` —— 跑通 Go2 平地行走
- [ ] `week05_07_locomotion_mjx/custom_task/` —— **自己改一个任务**（改地形 / 改奖励 / 加扰动），跑出能用的策略，**这是简历最强亮点之一**
- [ ] 训练视频（最好 60 秒以内，剪辑好）

### 时间分配

| 任务 | 周 5 | 周 6 | 周 7 |
|---|---|---|---|
| JAX 学习 | 6h | - | - |
| Playground 上手 | 5h | 3h | - |
| Go2 baseline 训练 | 3h | 5h | - |
| 自定义任务 | - | 6h | 10h |
| 论文 / 笔记 | 2h | 2h | 4h |

> 这阶段最耗时间在"等训练"。提前规划，让训练在你睡觉时跑。

---

## 阶段 4 ｜ W8-W9 ｜ 模仿学习与 LeRobot

**为什么**：纯 RL 已经不再是机器人控制的全部。2024-2025 年具身智能岗位会问 Diffusion Policy、ACT、Octo、π0、RT-2 这些名词，背后都是模仿学习 / VLA 模型。这两周不是要你掌握 VLA，而是把模仿学习 baseline 跑通，建立认知。

### 学习目标

- 理解 Behavior Cloning 的局限（distribution shift）和 DAgger 的修复思路
- 跑通 [LeRobot](https://github.com/huggingface/lerobot) 一个示例任务
- 看懂 Diffusion Policy 的核心思想（用扩散模型生成动作序列）
- 看懂 ACT（Action Chunking Transformer）思想

### 推荐资料

- **代码**：[Hugging Face LeRobot](https://github.com/huggingface/lerobot)（社区活跃、文档对新手友好）
- **论文**：
  - "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"（Chi 2023）
  - "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"（Zhao 2023，ACT 这篇）
  - "Octo: An Open-Source Generalist Robot Policy"（2024）
  - "π0: A Vision-Language-Action Flow Model"（2024）

### 硬性产出

- [ ] `week08_09_manipulation_il/lerobot_demo/` —— LeRobot 跑通一个 demo（推车、抓取等）
- [ ] `week08_09_manipulation_il/comparison.md` —— BC / Diffusion Policy / ACT 三者对比表
- [ ] 读 π0 / Octo 论文笔记，能口头讲清楚架构

### 时间分配

| 任务 | 周 8 | 周 9 |
|---|---|---|
| LeRobot 上手 | 5h | 3h |
| Diffusion Policy 论文+代码 | 4h | 3h |
| ACT / VLA 论文阅读 | 3h | 4h |
| 笔记整理 | 3h | 5h |

---

## 阶段 5 ｜ W10-W12 ｜ 求职作品集精修

**这是最关键的阶段。前 9 周是建认知，这 3 周是把认知转化为简历亮点**。

### 选一个主打项目

从前 4 阶段的产出里挑一个深做。推荐方向：

- **方向 A（推荐多数人）**：把 W5-W7 的 MJX 四足项目做精——完整 README、消融实验（学习率 / 域随机化幅度 / 奖励权重）、训练曲线、视频、博客文章。
- **方向 B（动手能力强者）**：把 W3-W4 的 robosuite 项目扩展成多任务——同一份策略代码跑 Lift、Stack、PickPlace，加一个 ablation table。
- **方向 C（理论倾向）**：把 W1-W2 的 PPO 实现升级为 mini-RL-zoo，对标 CleanRL，附带详细的 implementation tricks 文档（可作为博客发出去）。

### 求职动作

- [ ] 把整个 `RL/` 仓库整理干净，推到 GitHub（README 写好，加 badges、demo gif）
- [ ] 写 1-2 篇技术博客（中英文皆可，发知乎 / 个人站 / Medium）
- [ ] 录主打项目的演示视频（60-90 秒，剪辑过的）
- [ ] 准备 STAR 法则的项目讲解（Situation - Task - Action - Result）
- [ ] 刷一遍 `interviews/job_prep.md` 里的常考点
- [ ] 投简历（具身智能岗位 / 机器人 RL 工程师 / 仿真工程师）

### 硬性产出

- [ ] 主打项目独立仓库（≥50 stars 是个好目标，但不强求）
- [ ] 至少 1 篇博客
- [ ] 1 段演示视频
- [ ] 简历 + 自我介绍稿

---

## 学习节奏检查表

每周末问自己：

1. 这周硬性产出完成了吗？没完成就**不要**进入下周。
2. `progress.md` 写了吗？卡点和收获都记下来了吗？
3. GitHub 推了吗？commit 是不是有意义的（不要堆 "update"）？
4. 论文笔记有没有补？哪怕 100 字也好。

如果连续两周拖延，停下来问：是路线不合适、节奏太赶、还是状态问题？必要时把 12 周延长到 16 周，**比放弃强**。
