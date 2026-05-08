# RL Robot Control 学习项目

> 目标：3 个月内系统掌握 RL 机器人控制，产出 1-2 个可上 GitHub 简历的项目，瞄准机器人 / 具身智能岗位。
> 硬件：MacBook Pro / Air with Apple Silicon **M4**。
> 学习者：有 Python 和深度学习基础，懂一点 RL（CartPole 级别），未做过机器人仿真。
> 投入：10-20 小时 / 周（认真模式）。

---

## 快速开始

```bash
# 1. 配环境（一次性，约 30 分钟）
cd setup && bash install.sh

# 2. 验证环境（重要，确认 MPS / Metal 都通了）
python setup/verify_env.py

# 3. 进入第一阶段
cd week01_02_foundations && cat README.md
```

每周做完事情后，在 `progress.md` 里打卡。

---

## 项目导航

| 文件 / 目录 | 用途 |
|---|---|
| `ROADMAP.md` | **先读这个**。12 周详细学习路线，每周目标、产出、推荐资料 |
| `setup/` | 环境安装脚本和验证工具 |
| `week01_02_foundations/` | 阶段 1：RL 算法复习与重写（PPO/SAC） |
| `week03_04_mujoco_robosuite/` | 阶段 2：MuJoCo 基础 + robosuite 机械臂 |
| `week05_07_locomotion_mjx/` | 阶段 3：MuJoCo Playground / MJX 四足训练 |
| `week08_09_manipulation_il/` | 阶段 4：模仿学习（Diffusion Policy / ACT / LeRobot） |
| `week10_12_portfolio/` | 阶段 5：精修一个作品集项目 + 求职准备 |
| `papers/reading_list.md` | 按阶段匹配的论文清单（每周 1-2 篇） |
| `interviews/job_prep.md` | 具身智能岗位常考点 + 项目话术 |
| `progress.md` | 每周打卡日志（自己填） |

---

## 双技术栈

M4 上两条路都要走。

**PyTorch + MPS（基础栈）**
- 用于：算法理解、机械臂操作、模仿学习
- 库：`torch`、`stable-baselines3`、`gymnasium`、`robosuite`、`mujoco`、`lerobot`
- 速度：CPU/MPS，单环境训练偏慢但够用

**JAX + Metal（性能栈）**
- 用于：四足 locomotion、大规模并行训练
- 库：`jax`、`jaxlib`、`mujoco`（含 `mjx`）、`brax`、`mujoco_playground`
- 速度：M4 的 GPU 能用上，比 CPU 快几十倍，单机训四足策略可行

> 注意：`jax-metal` 历史上有兼容性坑（不支持某些 op，会回退到 CPU）。`verify_env.py` 会做一次健康检查。如果 Metal 后端有问题，可降级为 JAX CPU 也够第一阶段用，到第三阶段再细致调。

---

## 心态与节奏建议

1. **先跑通，再深入**。每周第一目标是把当周的代码跑出第一个学习曲线，而不是把论文读完。
2. **Git 全程开仓**。从第一天起就把这个文件夹推到 GitHub，每周 push 一次。求职时，commit history 本身就是最好的简历。
3. **每周写 200 字小结**。在 `progress.md` 记录这周学了啥、卡在哪、下周目标。求职面试时这是最好的话术原料。
4. **不要陷入工具链调试地狱**。环境装不上、某个仿真器跑不动，先绕开，用 PyBullet / Gymnasium 兜底。学习节奏比追求最优栈重要。

---

## 阶段总览（详细见 ROADMAP.md）

| 阶段 | 周次 | 核心目标 | 主要技术栈 |
|---|---|---|---|
| 1 | W1-W2 | 把 PPO/SAC 自己重写一遍，跑通 MuJoCo HalfCheetah | PyTorch + SB3 + CleanRL |
| 2 | W3-W4 | MuJoCo 仿真原理 + robosuite 机械臂抓取 | PyTorch + MuJoCo + robosuite |
| 3 | W5-W7 | MJX 四足 locomotion 训练，从 Go1/Go2 到自定义任务 | **JAX** + MuJoCo Playground |
| 4 | W8-W9 | LeRobot + Diffusion Policy / ACT，模仿学习 | PyTorch + LeRobot |
| 5 | W10-W12 | 选一个方向做精，写 README + 录视频 + 博客 | 任选 |

---

## 求职导向的产出清单

12 周后，你的 GitHub 上应该有：

- [ ] 一个 RL 算法库（PPO/SAC 自己实现，对标 CleanRL，代码干净有注释）
- [ ] 一个 robosuite 操作任务训练 demo（视频 + 学习曲线 + 域随机化实验）
- [ ] 一个 MJX 四足 locomotion 项目（最少能在平地走稳，加分项：上不平地形）
- [ ] 一个模仿学习 demo（用 LeRobot 跑一个机械臂模仿任务）
- [ ] 一个精修的"主打项目"（从上面挑一个深做：消融实验、博客文章、bilibili / YouTube 视频）
- [ ] 一份针对具身智能岗位的项目讲解材料（`interviews/` 下）

按这个清单来，简历就有东西写。
