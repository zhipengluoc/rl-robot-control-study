# 阶段 1 ｜ W1-W2 ｜ RL 基础复盘与重写

**激活环境**：`conda activate rl-torch`

## 这两周要做的事（按顺序）

### W1 D1-D2：把 PPO 论文重新读一遍

打印或开 PDF，做 marginalia。重点搞清楚：

1. 为什么需要 importance sampling？
2. clip 的作用是什么？为什么是 1±ε？
3. GAE 怎么算？λ 趋近 0 / 1 各代表什么？
4. value function 是 share 还是 separate？
5. PPO 论文 Appendix 里的工程实现细节（advantage normalization、orthogonal init 等）

**强烈推荐配套读**：[The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)。这一篇博客在面试中救过无数人。

笔记记到 `notes.md`。

### W1 D3-D4：精读 CleanRL 的 ppo_continuous_action.py

```bash
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl/cleanrl
# 重点看 ppo.py 和 ppo_continuous_action.py
```

**逐行打注释**。一行 1 句，写完一句问自己"为什么这么做"。如果回答不出来，就去查、去问。

把带注释的版本另存到 `cleanrl_ppo_annotated.py`。

### W1 D5-W2 D2：手写 PPO

不用 SB3，不用 CleanRL，从空白文件开始写一个能跑 `Pendulum-v1` 的 PPO。约束：

- 单文件 ≤ 400 行
- 用 PyTorch
- 用 MPS 加速（如果可行）
- 学习曲线用 tensorboard 或者直接 matplotlib 存图

文件名：`ppo_minimal.py`。期望产出：

```
$ python ppo_minimal.py
[step 0]    return = -1200
[step 50k]  return = -800
[step 200k] return = -200   <- 收敛到这里就算过关
```

写完后跑一下 `LunarLanderContinuous-v3`，验证泛化（可能需要调超参）。

### W2 D3-D4：用 SB3 在 MuJoCo 跑

```python
from stable_baselines3 import PPO, SAC
import gymnasium as gym

env = gym.make("HalfCheetah-v4")
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tb/")
model.learn(total_timesteps=1_000_000)
model.save("sac_halfcheetah")
```

跑两组对比：

- PPO vs SAC 在 HalfCheetah 上的样本效率
- HalfCheetah vs Ant 的训练曲线

文件名：`sb3_mujoco.py`。

### W2 D5：整理 + 写笔记 + Push

把这两周所有的代码、笔记、学习曲线整理成一个能给别人看的状态。

写 `notes.md`：

- PPO 和 SAC 的关键区别
- on-policy vs off-policy 的实战体感
- 你踩的坑（这个最值钱，求职被问"你踩过什么坑"时直接讲）

最后 `git commit -am "week 1-2 done"` 推到 GitHub。

---

## 推荐资料

### 论文（按优先级）

1. **PPO**：Schulman et al. 2017, "Proximal Policy Optimization Algorithms"
2. **SAC**：Haarnoja et al. 2018, "Soft Actor-Critic"
3. **GAE**：Schulman et al. 2016, "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
4. **TD3**（看个大意即可）：Fujimoto et al. 2018

### 课程 / 教材

- Spinning Up in Deep RL（OpenAI）：[https://spinningup.openai.com/](https://spinningup.openai.com/)
- CS285 Berkeley Sergey Levine（重点看 PG / Actor-Critic 几节课）

### 必看代码

- CleanRL：[https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)
- SB3 文档：[https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)

### 必看博客

- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)（**面试救命**）
- Lilian Weng 的 RL 系列（[lilianweng.github.io](https://lilianweng.github.io/)）

---

## 硬性产出 checklist

- [ ] `ppo_minimal.py` 在 Pendulum 上收敛
- [ ] `cleanrl_ppo_annotated.py` 逐行注释
- [ ] `sb3_mujoco.py` 在 HalfCheetah / Ant 上各跑出一条学习曲线
- [ ] `notes.md` 写完（≥ 300 字）
- [ ] 学习曲线截图存到 `results/`
- [ ] git push

不达标不进入 W3。
