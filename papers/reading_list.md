# 论文阅读清单

> 12 周读完这个清单的"必读"部分（约 15-18 篇），具身智能岗位面试相关的论文基本就覆盖了。
> 优先级标注：⭐⭐⭐ 必读 / ⭐⭐ 推荐 / ⭐ 选读

每读完一篇，在对应位置打勾，并在 `papers/notes/` 下建一个同名 .md 文件写笔记（200-500 字够）。

---

## 阶段 1（W1-W2）：RL 算法基础

- [ ] ⭐⭐⭐ **PPO** — Schulman et al. 2017, *Proximal Policy Optimization Algorithms*
  - https://arxiv.org/abs/1707.06347
  - 必备：clip 机制、ratio、value loss、entropy
- [ ] ⭐⭐⭐ **SAC** — Haarnoja et al. 2018, *Soft Actor-Critic*
  - https://arxiv.org/abs/1801.01290
  - 必备：max entropy framework、temperature 自动调节
- [ ] ⭐⭐⭐ **GAE** — Schulman et al. 2016, *Generalized Advantage Estimation*
  - https://arxiv.org/abs/1506.02438
  - 必备：bias-variance tradeoff、λ 含义
- [ ] ⭐⭐ **TD3** — Fujimoto et al. 2018
  - https://arxiv.org/abs/1802.09477
  - 看个大意，知道为什么要 twin Q + delayed policy update + target smoothing
- [ ] ⭐⭐ **DDPG** — Lillicrap et al. 2015
  - 历史背景，知道 SAC/TD3 是怎么演化过来的

---

## 阶段 2（W3-W4）：MuJoCo + 操作 + 域随机化

- [ ] ⭐⭐⭐ **Domain Randomization** — Tobin et al. 2017
  - https://arxiv.org/abs/1703.06907
  - 必备：DR 的动机、参数选择、为什么有效
- [ ] ⭐⭐ **Asymmetric Actor-Critic** — Pinto et al. 2018
  - https://arxiv.org/abs/1710.06542
  - privileged learning 思想的早期版本
- [ ] ⭐ **robosuite paper** — Zhu et al. 2020
  - 了解 robosuite 设计哲学

---

## 阶段 3（W5-W7）：四足 Locomotion + Sim2Real

- [ ] ⭐⭐⭐ **Massively Parallel RL** — Rudin et al. 2022
  - https://arxiv.org/abs/2109.11978
  - "Learning to walk in minutes"，IsaacGym 那篇但思想通用
- [ ] ⭐⭐⭐ **RMA** — Kumar et al. 2021
  - https://arxiv.org/abs/2107.04034
  - sim2real 经典：teacher-student、adaptation module
- [ ] ⭐⭐⭐ **Lee et al. 2020** — Science Robotics
  - *Learning quadrupedal locomotion over challenging terrain*
  - 四足 RL 工业落地的里程碑
- [ ] ⭐⭐ **Walk These Ways** — Margolis et al. 2022
  - https://arxiv.org/abs/2212.03238
  - behavior diversity，求职被问"怎么让策略多样化"时用
- [ ] ⭐⭐ **Hwangbo et al. 2019** — Science Robotics
  - actuator network 的来源
- [ ] ⭐ **Extreme Parkour** — Cheng et al. 2024
  - 看 demo 视频也行，了解 SOTA 在做什么
- [ ] ⭐ **Eureka** — Ma et al. 2023
  - LLM-augmented reward design，前沿话题

---

## 阶段 4（W8-W9）：模仿学习 + VLA

- [ ] ⭐⭐⭐ **Diffusion Policy** — Chi et al. 2023
  - https://arxiv.org/abs/2303.04137
  - 必读，2024-2025 manipulation 的事实标准
- [ ] ⭐⭐⭐ **ACT** — Zhao et al. 2023
  - https://arxiv.org/abs/2304.13705
  - action chunking、ALOHA、低成本双臂
- [ ] ⭐⭐ **Octo** — 2024
  - https://arxiv.org/abs/2405.12213
  - 第一个开源 generalist robot policy
- [ ] ⭐⭐ **π0** — Physical Intelligence 2024
  - https://www.physicalintelligence.company/blog/pi0
  - flow matching + VLM
- [ ] ⭐⭐ **RT-2** — Google 2023
  - https://arxiv.org/abs/2307.15818
  - VLA 概念奠基
- [ ] ⭐ **OpenVLA** — 2024
  - 开源 VLA，可上手
- [ ] ⭐ **GR00T N1** — NVIDIA 2024
  - 工业关注度高，看博客 + 论文摘要即可

---

## 阶段 5（W10-W12）：求职阶段补充阅读

针对你最终选的方向再补 2-3 篇综述 / 最新工作。

### 综述类（任选 1-2 篇通读）

- [ ] **Sim-to-Real survey** — 任意一篇近期综述（2023-2025）
- [ ] **Foundation models for robotics survey** — 选 2024 后的

### 求职话术准备

- [ ] **The 37 Implementation Details of PPO**（博客）
  - https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
  - **被问 PPO 时救命**，必须熟读

---

## 笔记建议

在 `papers/notes/` 下，每篇论文一个 `<paper_short_name>.md`，结构建议：

```markdown
# 论文标题

**作者 / 年份 / 会议**：
**链接**：

## 一句话总结

## 解决了什么问题

## 核心方法
- key idea 1
- key idea 2

## 实验亮点

## 我学到了什么 / 怎么用到自己的项目

## 我的疑问 / 不理解的点
```

求职面试时翻出来快速复习就有用。
