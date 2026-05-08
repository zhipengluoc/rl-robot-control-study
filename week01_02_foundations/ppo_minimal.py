"""手写 PPO（最简版本，可在 Pendulum-v1 上收敛）。

学习目标：
    - 不依赖 SB3 / CleanRL，从 PyTorch 写一个能跑的 PPO
    - 理解每一行代码在干什么
    - 学完后用它对比 CleanRL 的实现，找差异

使用方法：
    conda activate rl-torch
    python ppo_minimal.py

预期结果（Pendulum-v1）：
    [step 0]    return ≈ -1200
    [step 50k]  return ≈ -800
    [step 200k] return > -300  ← 基本算收敛

TODO（你要补的部分用 raise NotImplementedError 标出来）：
    1. compute_gae() —— GAE 计算
    2. update() 里 clipped surrogate objective 部分
    3. value loss 和 entropy bonus
    4. 把这些拼起来，跑通 Pendulum

提示：
    - 论文公式在头脑里要清晰
    - 第一次 ratio 应该等于 1（确认你的 logprob 计算正确）
    - advantage normalize 用 (a - mean) / (std + 1e-8)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------


@dataclass
class Config:
    env_id: str = "Pendulum-v1"
    seed: int = 0
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    # rollout
    num_envs: int = 4
    num_steps: int = 2048      # 每个 env 收集多少 step 后做一次更新
    total_timesteps: int = 200_000

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 256
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # optimization
    learning_rate: float = 3e-4

    # logging
    log_interval: int = 1


# -----------------------------------------------------------------------------
# 网络
# -----------------------------------------------------------------------------


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    """正交初始化（PPO 实施细节里很重要的一条）。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # 共享 feature 还是分开？这里用分开（PPO 经典做法）
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),  # std 小一点，初始 action 不要太大
        )
        # 用 state-independent log_std（标准做法）
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        log_std = self.actor_logstd.expand_as(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


# -----------------------------------------------------------------------------
# GAE
# -----------------------------------------------------------------------------


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """计算 GAE advantages 和 returns。

    输入 shape: (T, num_envs)
    rewards[t], values[t], dones[t]
    last_value: (num_envs,) —— 最后一步之后的 value bootstrap

    输出:
        advantages: (T, num_envs)
        returns: (T, num_envs)  —— = advantages + values

    TODO: 实现 GAE 公式。
        delta_t = r_t + γ * V_{t+1} * (1 - done_{t+1}) - V_t
        A_t = delta_t + γ λ (1 - done_{t+1}) A_{t+1}
    """
    raise NotImplementedError("写 GAE 是这个文件最重要的练习之一，自己实现！")


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


def train(cfg: Config):
    print(f"device: {cfg.device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 多环境（用 SyncVectorEnv，方便起步；性能不重要）
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed + i) for i in range(cfg.num_envs)]
    )
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    agent = ActorCritic(obs_dim, action_dim).to(cfg.device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # rollout buffer
    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=cfg.device)
    act_buf = torch.zeros((cfg.num_steps, cfg.num_envs, action_dim), device=cfg.device)
    logp_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=cfg.device)
    val_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=cfg.device)
    rew_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=cfg.device)
    done_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=cfg.device)

    obs, _ = envs.reset(seed=cfg.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=cfg.device)
    done = torch.zeros(cfg.num_envs, device=cfg.device)

    num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
    global_step = 0
    start_time = time.time()
    ep_returns = []

    for update in range(1, num_updates + 1):
        # ------ Rollout ------
        for t in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[t] = obs
            done_buf[t] = done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)

            act_buf[t] = action
            logp_buf[t] = logprob
            val_buf[t] = value

            next_obs_np, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            done_np = np.logical_or(terminated, truncated)
            rew_buf[t] = torch.tensor(reward, dtype=torch.float32, device=cfg.device)
            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=cfg.device)
            done = torch.tensor(done_np, dtype=torch.float32, device=cfg.device)

            # 收集 episode return（仅展示）
            if "episode" in info:
                # gymnasium 0.29+ 用 info["episode"]["r"] / info["_episode"] (mask)
                if "_episode" in info:
                    for i in range(cfg.num_envs):
                        if info["_episode"][i]:
                            ep_returns.append(float(info["episode"]["r"][i]))

        # ------ Compute advantages ------
        with torch.no_grad():
            last_value = agent.get_value(obs)
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, last_value,
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        )

        # ------ Update ------
        # flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, action_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        # advantage normalization
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = cfg.num_steps * cfg.num_envs
        indices = np.arange(batch_size)
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb = indices[start:end]

                _, new_logp, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb], b_act[mb]
                )

                # TODO: 实现 PPO 的核心 loss
                # 1. ratio = exp(new_logp - b_logp[mb])
                # 2. surr1 = ratio * advantages
                # 3. surr2 = clamp(ratio, 1-clip, 1+clip) * advantages
                # 4. policy_loss = -min(surr1, surr2).mean()
                # 5. value_loss = (new_value - returns)^2 .mean() * 0.5
                # 6. entropy_loss = -entropy.mean()
                # 7. loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                raise NotImplementedError("写 PPO loss！这是练习核心")

                optimizer.zero_grad()
                loss.backward()  # noqa: F821
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        if update % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            mean_ret = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(
                f"[update {update:4d}] step={global_step:7d}  "
                f"sps={sps:5d}  ep_return(last20)={mean_ret:8.2f}"
            )

    envs.close()
    print(f"训练完成。耗时 {(time.time() - start_time):.1f}s")

    # 保存模型
    os.makedirs("results", exist_ok=True)
    torch.save(agent.state_dict(), "results/ppo_minimal.pt")
    print("model saved -> results/ppo_minimal.pt")


if __name__ == "__main__":
    train(Config())
