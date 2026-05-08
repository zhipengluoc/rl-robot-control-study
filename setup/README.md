# 环境配置

针对 **Apple Silicon M4** 优化。两套环境分开装，避免相互干扰。

## 前置条件

```bash
# 1. 安装 Homebrew（如果还没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装 Miniforge（M 系列芯片推荐，比 Anaconda 干净）
brew install --cask miniforge
conda init zsh  # 或 bash

# 3. 关掉默认 base 自动激活（推荐）
conda config --set auto_activate_base false
```

## 一键安装（推荐）

```bash
cd setup
bash install.sh
```

这会创建两个 conda 环境：

- `rl-torch`（Python 3.11）—— PyTorch + MuJoCo + SB3 + robosuite + LeRobot
- `rl-jax`（Python 3.11）—— JAX + jax-metal + MuJoCo + MJX + Brax + Playground

## 验证

```bash
conda activate rl-torch
python verify_env.py

conda activate rl-jax
python verify_env.py
```

`verify_env.py` 会检查：

- PyTorch MPS 是否可用 / 能跑前向反向
- JAX 后端 / Metal 加速是否启用
- MuJoCo 能否加载模型并 step
- Gymnasium 能否创建 MuJoCo 环境

## 常见坑

### jax-metal 兼容性

`jax-metal` 是 Apple 官方插件，但**不支持所有 JAX op**，遇到不支持的会报错或回退 CPU。如果在 W5-W7 训练 MJX 时遇到 Metal 相关错误：

```bash
# 临时强制 CPU 后端
JAX_PLATFORMS=cpu python your_script.py
```

CPU 后端跑小规模训练（< 1024 个并行环境）也是够用的，只是慢一些。

### MuJoCo 首次启动黑屏 / 闪退

```bash
brew install glfw
```

### robosuite 安装慢 / 失败

robosuite 依赖较多。可以先单独装：

```bash
pip install robosuite==1.5
```

或源码安装最新版：

```bash
pip install git+https://github.com/ARISE-Initiative/robosuite.git
```

### Python 版本

不要用 Python 3.13。截至 2026 年初，部分 RL 库还没适配。**3.11 最稳**。
