"""用 Stable-Baselines3 在 MuJoCo 上训练 PPO 和 SAC，对比样本效率。

学习目标：
    - 把"调库党"的水平从 90 分降到 60 分（笑），但能快速验证 baseline
    - 学会读 tensorboard 曲线
    - 体感 PPO（on-policy）和 SAC（off-policy）的差异

使用方法：
    conda activate rl-torch
    python sb3_mujoco.py --algo sac --env HalfCheetah-v4
    python sb3_mujoco.py --algo ppo --env HalfCheetah-v4

    # 看曲线
    tensorboard --logdir tb/

提前提示：
    - HalfCheetah PPO 需要 ~5M steps，SAC 需要 ~1M steps（粗略）
    - M4 上单 env 训练偏慢，可以减小 total_timesteps 先看趋势
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env


ALGOS = {"ppo": PPO, "sac": SAC}


def train(args):
    os.makedirs("results", exist_ok=True)

    # PPO 喜欢多并行环境，SAC 单 env 即可
    if args.algo == "ppo":
        env = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed)
    else:
        env = gym.make(args.env)

    eval_env = gym.make(args.env)

    cls = ALGOS[args.algo]
    if args.algo == "ppo":
        model = cls(
            "MlpPolicy", env,
            verbose=1,
            seed=args.seed,
            tensorboard_log="./tb/",
            device=args.device,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
        )
    else:  # SAC
        model = cls(
            "MlpPolicy", env,
            verbose=1,
            seed=args.seed,
            tensorboard_log="./tb/",
            device=args.device,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"results/{args.algo}_{args.env}/",
        log_path=f"results/{args.algo}_{args.env}/",
        eval_freq=10_000,
        n_eval_episodes=5,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        tb_log_name=f"{args.algo}_{args.env}",
        progress_bar=True,
    )

    model.save(f"results/{args.algo}_{args.env}_final")
    print(f"模型已保存到 results/{args.algo}_{args.env}_final.zip")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(ALGOS), default="sac")
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--n_envs", type=int, default=4, help="PPO 并行 env 数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="mps",
        help="mps / cpu / cuda（M4 用 mps，但小网络 cpu 可能更快）",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
