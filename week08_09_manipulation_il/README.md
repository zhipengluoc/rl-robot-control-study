# 阶段 4 ｜ W8-W9 ｜ 模仿学习与 LeRobot

**激活环境**：`conda activate rl-torch`

> 这两周不是要你"训出 SOTA 模仿学习模型"，而是把概念吃透、跑通最小 demo，搭建认知。
> **目标是面试时能说："我知道 Diffusion Policy / ACT 在干什么，跑过 LeRobot 的 demo"**。

## 为什么要学模仿学习

2024-2025 年具身智能领域发生了范式转变：

- 单纯 RL 在复杂操作任务上样本效率太低
- 大量数据 + 模仿学习成为新范式（RT-2、Octo、π0 等 VLA 模型）
- 但 RL 仍然在 locomotion、低维度控制、需要超过人类 demo 性能的场景里不可替代

求职被问"你怎么看 RL vs IL"——你要能给出有 nuance 的答案。

## 任务清单

### W8 D1-D2：模仿学习理论

读以下两篇博客 / 章节，建立认知：

- Sergey Levine 的 [Imitation Learning lecture notes](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf)
- Lilian Weng [Robot Learning blog](https://lilianweng.github.io/)（任何相关篇章）

笔记答这几个问题：

1. 为什么 Behavior Cloning（BC）会有 distribution shift / compounding error 问题？
2. DAgger 怎么修复？局限是什么？
3. 为什么近年大家转向 "scale up data" 而不是 "fix the algorithm"？

### W8 D3-W9 D2：跑通 LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

跟着 [官方 Getting Started](https://github.com/huggingface/lerobot)：

1. 加载一个公开数据集（PushT 或 Aloha）
2. 训一个 Diffusion Policy 或 ACT 模型
3. 在仿真里 evaluation

期望产出：

```
lerobot_demo/
├── train.py        # 配置好的训练脚本
├── eval.py         # 评估脚本
├── checkpoints/    # 训好的 ckpt
└── eval_video.mp4  # 评估视频
```

> 提示：M4 上某些 GPU op 可能慢，可以先用小数据集 / 短训练验证流程通了，再决定要不要全量训。

### W9 D3-D4：精读 Diffusion Policy 与 ACT

**Diffusion Policy**（Chi et al. 2023）：

- 论文：[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- 代码：[real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- 看懂：为什么用 diffusion model 生成 action sequence？相比 BC + Gaussian head 优势在哪？

**ACT**（Zhao et al. 2023）：

- 论文：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/)
- 看懂：action chunking 是什么？temporal ensemble 解决了什么问题？

写 `comparison.md`：BC、Diffusion Policy、ACT 三者对比表（一栏一栏写：思路、优势、劣势、适用场景）。

### W9 D5：VLA 模型扫盲

**不要试图深入**，扫一下论文了解大概：

- [RT-2](https://robotics-transformer2.github.io/)（Google 2023）
- [Octo](https://octo-models.github.io/)（Berkeley 2024）
- [π0](https://www.physicalintelligence.company/blog/pi0)（Physical Intelligence 2024）
- [GR00T](https://research.nvidia.com/labs/gear/gr00t/)（NVIDIA 2024）

要能在面试时口头讲清楚：

- 这些模型的输入输出是什么？
- VLA 和传统 BC 区别？
- 为什么 2024 年开始 robot foundation model 概念火起来？

笔记到 `vla_overview.md`。

---

## 推荐资料

### 论文（按优先级）

1. **Diffusion Policy**（Chi 2023）—— 必读
2. **ACT**（Zhao 2023）—— 必读
3. **Octo**（2024）—— 重要
4. **π0**（2024）—— 重要
5. **RT-2**（2023）—— 了解
6. DAgger（Ross 2011）—— 经典背景

### 代码

- [LeRobot](https://github.com/huggingface/lerobot)
- [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- [tonyzhaozh/act](https://github.com/tonyzhaozh/act)

---

## 硬性产出 checklist

- [ ] `lerobot_demo/` 跑通一个完整训练 + 评估流程
- [ ] `comparison.md` BC/DP/ACT 对比表
- [ ] `vla_overview.md` 主流 VLA 模型简介
- [ ] 至少读完 Diffusion Policy + ACT 两篇论文
- [ ] git push
