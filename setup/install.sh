#!/usr/bin/env bash
# RL Robot Control 学习项目 - M4 环境安装脚本
# 创建两个 conda 环境：rl-torch（PyTorch 栈）和 rl-jax（JAX 栈）

set -e  # 出错即退出

echo "================================================================"
echo "  RL Robot Control 学习项目 - 环境安装"
echo "  目标平台：Apple Silicon (M1/M2/M3/M4)"
echo "================================================================"
echo ""

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "错误：未找到 conda。请先安装 Miniforge："
    echo "  brew install --cask miniforge"
    exit 1
fi

# 检查芯片架构
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "警告：当前架构是 $ARCH，本脚本针对 Apple Silicon (arm64) 优化"
    read -p "继续吗？(y/N) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

echo ">>> 系统依赖（Homebrew）"
brew install glfw cmake pkg-config || true

# ============================================================
# 环境 1：PyTorch 栈
# ============================================================
echo ""
echo ">>> [1/2] 创建 rl-torch 环境（PyTorch + MuJoCo + SB3 + robosuite）"
echo ""

conda create -n rl-torch python=3.11 -y

# 进入环境（在脚本内激活需要 source）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl-torch

pip install --upgrade pip

# PyTorch（M 系列原生支持 MPS）
pip install torch torchvision

# RL 基础
pip install gymnasium[all]
pip install stable-baselines3[extra]

# MuJoCo
pip install mujoco
pip install mujoco-mjx  # MJX 也装到这边便于对比

# 可视化 / 实用工具
pip install matplotlib tensorboard wandb tqdm rich pyyaml imageio[ffmpeg]

# robosuite（可能较慢）
pip install robosuite

# LeRobot（W8-W9 用）
pip install lerobot || echo "LeRobot 安装失败，可后续手动从源码装：pip install git+https://github.com/huggingface/lerobot.git"

conda deactivate

# ============================================================
# 环境 2：JAX 栈
# ============================================================
echo ""
echo ">>> [2/2] 创建 rl-jax 环境（JAX + Metal + MJX + Brax + Playground）"
echo ""

conda create -n rl-jax python=3.11 -y
conda activate rl-jax

pip install --upgrade pip

# JAX + Metal（Apple Silicon 加速）
pip install jax jaxlib
pip install jax-metal || echo "jax-metal 装不上时可降级，详见 setup/README.md"

# MuJoCo MJX
pip install mujoco mujoco-mjx

# Brax（DeepMind JAX 物理引擎）
pip install brax

# MuJoCo Playground（W5-W7 主战场）
pip install playground || pip install git+https://github.com/google-deepmind/mujoco_playground.git

# 通用工具
pip install matplotlib tensorboard wandb tqdm rich pyyaml imageio[ffmpeg]
pip install flax optax  # JAX 生态的神经网络 + 优化器库
pip install gymnasium

conda deactivate

# ============================================================
# 完成
# ============================================================
echo ""
echo "================================================================"
echo "  安装完成"
echo "================================================================"
echo ""
echo "下一步：验证环境"
echo "  conda activate rl-torch && python verify_env.py"
echo "  conda activate rl-jax   && python verify_env.py"
echo ""
echo "如果验证失败，请查看 setup/README.md 的'常见坑'部分。"
