# 阶段 2 ｜ W3-W4 ｜ MuJoCo + robosuite 操作任务

**激活环境**：`conda activate rl-torch`

## 这两周的目标

把"训出一个机械臂能完成任务"这件事跑通，并且第一次接触域随机化。

## 任务清单（按顺序）

### W3 D1：MuJoCo 基础

跟着官方 Colab 走一遍：[mujoco_python_tutorial](https://github.com/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)

重点理解：

- MJCF（XML 模型）的 body / joint / geom / actuator
- `MjModel` 和 `MjData` 的区别
- `mj_step` 在干什么
- 渲染（`mujoco.Renderer` 或 `mujoco.viewer`）

写一个 `mujoco_hello.py`：加载一个简单模型（盒子从空中掉落），渲染成 gif。

### W3 D2-D3：自己写一个最简 Gymnasium 环境

不用 robosuite，自己包一个最简的 `MujocoPushBlockEnv`：

- 一个机械臂（直接用 MuJoCo 自带 model 也行）
- 一个目标方块
- observation = 关节角度 + 方块位置
- action = 关节速度
- reward = 负的方块到目标距离

目的：理解"环境"在 RL 里是怎么写出来的。这个理解在 W5 自定义 locomotion 任务时极其关键。

文件名：`custom_push_env.py`。能用 SB3 PPO 训 100k 步看看效果即可，**不要求完美**。

### W3 D4-D5：robosuite 上手

```python
import robosuite as suite

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
env.reset()
for _ in range(500):
    action = env.action_spec[0]  # zeros
    env.step(action)
    env.render()
```

熟悉 robosuite 的 controller、observation space、reward function。

### W4 D1-D3：训练 SAC on Lift

robosuite 有 Gym wrapper：

```python
from robosuite.wrappers import GymWrapper
import robosuite as suite
from stable_baselines3 import SAC

env = GymWrapper(
    suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )
)

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tb/", device="mps")
model.learn(total_timesteps=500_000)
```

期望：500k 步内能学会把方块举起来。如果不收敛，先检查 reward_shaping 是否开启。

文件名：`robosuite_lift_sac.py`。训练后存视频到 `results/lift_demo.mp4`。

### W4 D4-D5：域随机化小实验

修改 `Lift` 任务，加入：

- 摩擦系数 ±20%
- 物块质量 ±30%
- 初始位置噪声

跑两组：开 DR / 不开 DR，看在"测试时改变物理参数"下哪个鲁棒。

写 `dr_experiment.md`，1 张表 + 2 条曲线 + 200 字结论。

---

## 推荐资料

### 论文

- **Domain Randomization**：Tobin et al. 2017, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
- **Asymmetric Actor-Critic**：Pinto et al. 2018（理解 privileged info 的最早思想之一）
- **robosuite paper**：Zhu et al. 2020

### 文档 / 教程

- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [MuJoCo Python tutorial Colab](https://github.com/google-deepmind/mujoco/tree/main/python)
- [robosuite docs](https://robosuite.ai/docs/)

---

## 硬性产出 checklist

- [ ] `mujoco_hello.py` 能加载并渲染一个模型
- [ ] `custom_push_env.py` 自己实现一个 Gymnasium 接口的 MuJoCo 任务
- [ ] `robosuite_lift_sac.py` 训出能举起方块的策略
- [ ] `dr_experiment.md` 域随机化对比实验
- [ ] `results/` 下至少 1 个视频 + 2 张曲线图
- [ ] `notes.md` 总结 MuJoCo 的设计理念、robosuite 的层次结构
- [ ] git push
