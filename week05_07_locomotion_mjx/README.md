# 阶段 3 ｜ W5-W7 ｜ MJX + 四足 Locomotion

**激活环境**：`conda activate rl-jax`

> **这是整个学习路线的"最终 boss"。三周的成果是简历最大亮点之一。**

## 为什么用 JAX + MJX

四足 locomotion 训练需要"成千上万个并行环境"才能在合理时间内收敛。传统 PyTorch+CPU 仿真单环境是不够的。MJX 把 MuJoCo 物理引擎用 JAX 重写，可以：

- `jit` 编译加速
- `vmap` 实现批量并行（一次仿真 4096 个机器人）
- 在 M4 的 GPU 上跑（通过 jax-metal）
- 在 NVIDIA GPU 上能比 IsaacGym 快或相当

## 任务清单

### W5：JAX 基础（**只学一周，不要花更久**）

JAX 一个最大的难点是"纯函数式"编程范式（不能改 state，要返回 new state）。学的时候不要试图"精通"，能看懂 MJX 代码即可。

**学习路径**：

1. JAX 官方 [Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)（1 小时）
2. [JAX 101](https://jax.readthedocs.io/en/latest/jax-101/index.html)（半天）
3. 关键概念：`jit`、`vmap`、`grad`、`PyTree`、`Pure Functions`、`PRNGKey` 的工作原理

写一个 `jax_basics.ipynb` notebook，内容包括：

- 一个简单的线性回归（用 `grad` + `jit`）
- 一个 vmap 批处理的例子
- `flax.linen` 写一个 MLP 并训练

### W5-W6：MuJoCo Playground 上手

```bash
# 已在 rl-jax 环境装好
python -c "import mujoco_playground; print(mujoco_playground.locomotion.ALL_ENVS)"
```

跑通官方 Go1/Go2 训练例子。Playground 自带 PPO 实现（基于 brax）：

```python
from mujoco_playground import locomotion, wrapper
from mujoco_playground.config import locomotion_params

env_name = "Go1JoystickFlatTerrain"
env = locomotion.load(env_name)
ppo_params = locomotion_params.brax_ppo_config(env_name)
# 然后用 brax.training.agents.ppo 训练
```

参考：[MuJoCo Playground 官方文档](https://playground.mujoco.org/) 和 [README](https://github.com/google-deepmind/mujoco_playground)。

文件名：`playground_go2_baseline.py`。

**预期训练时间**（M4 + jax-metal，仅供参考）：

- Flat terrain Go2：30-60 分钟
- Rough terrain Go2：1-2 小时
- 如果 jax-metal 跑不动，回退 CPU：2-4 小时（接受）

每跑完一个 baseline，导出训练后视频，存 `results/`。

### W7：自定义任务（**简历金矿**）

从下面三选一：

**选项 A：自定义地形**
- 改地形为楼梯 / 斜坡 / 障碍物
- 调整 reward，让策略在地形变化下也稳定
- 要点：观察 sim2real gap 的来源

**选项 B：自定义奖励**
- 现有任务用的是 "tracking velocity"
- 你改成"侧向行走" / "后退" / "跳跃"
- 要点：理解 reward shaping

**选项 C：自定义机器人**
- 把 Go2 换成 Spot 或者 Unitree A1（Playground 都有）
- 看不同机器人的 morphology 怎么影响策略

**强烈建议选 A 或 C**——简历更好讲故事。

输出：

- `custom_task/` 下完整代码
- `results/` 下训练前后对比视频（剪成 30-60 秒一段）
- `report.md` 写实验设计、超参、踩坑、最终效果

### 全程：论文阅读

每周 1-2 篇，配合实验进度：

- W5：[Learning to walk in minutes (Rudin 2022)](https://arxiv.org/abs/2109.11978) —— 大规模并行 RL 训练范式的开山之作
- W6：[RMA: Rapid Motor Adaptation (Kumar 2021)](https://arxiv.org/abs/2107.04034) —— privileged learning + adaptation module，sim2real 经典
- W7：[Walk These Ways (Margolis 2022)](https://arxiv.org/abs/2212.03238) —— behavior diversity，求职面试加分项

---

## 推荐资料

### 代码

- [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
- [google/brax](https://github.com/google/brax)
- [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)

### 论文（按重要性）

1. Rudin et al. 2022, "Learning to walk in minutes using massively parallel deep reinforcement learning"
2. Lee et al. 2020, "Learning quadrupedal locomotion over challenging terrain" (Science Robotics)
3. Kumar et al. 2021, "RMA: Rapid Motor Adaptation for Legged Robots"
4. Margolis et al. 2022, "Walk These Ways"
5. Hwangbo et al. 2019, "Learning agile and dynamic motor skills for legged robots"（actuator network 来源）

### 视频 / 课程

- ETH Zurich Marco Hutter 组的相关 lecture（YouTube 搜 "legged robot reinforcement learning"）
- Berkeley Pieter Abbeel "Foundations of Deep RL" 系列

---

## 硬性产出 checklist

- [ ] `jax_basics.ipynb` 完整跑通
- [ ] `playground_go2_baseline.py` Go2 平地行走训练成功
- [ ] `custom_task/` 选 A/B/C 之一，跑出有效策略
- [ ] `results/` 至少 3 个训练视频
- [ ] `report.md` 自定义任务实验报告（≥500 字 + 图）
- [ ] 论文笔记 3 篇
- [ ] git push（**这周的 commit 要质量高，求职用**）
