# 具身智能 / 机器人 RL 求职准备

> 这份材料整理常见考点，自己学完一个章节就在 checklist 打勾，能口头讲清楚再算过。
> **能"讲清楚"的标准**：找一个不懂 RL 的朋友，向 ta 解释，对方能听懂。

---

## 一、RL 算法基础（高频）

### 1.1 PPO 系列

- [ ] PPO 的核心目标函数怎么写？clip 在干什么？
- [ ] importance sampling ratio 是什么？为什么 PPO 要用它？
- [ ] PPO 和 TRPO 的关系？为什么 PPO 取代了 TRPO？
- [ ] GAE 怎么算？λ=0 和 λ=1 各退化为什么？
- [ ] advantage normalization 为什么有效？
- [ ] 为什么 PPO 的 entropy bonus 系数要调大调小？
- [ ] PPO 在 on-policy 数据上为什么可以多 epoch 重复利用？
- [ ] orthogonal initialization 为什么对 PPO 重要？

📌 救命资源：[The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

### 1.2 SAC

- [ ] SAC 的目标函数？entropy 项的物理意义？
- [ ] temperature α 怎么自动调节？
- [ ] SAC vs DDPG vs TD3，区别在哪？
- [ ] SAC 的 sample efficiency 为什么比 PPO 高？

### 1.3 通用

- [ ] on-policy vs off-policy 的区别和取舍？
- [ ] replay buffer 在 on-policy 算法里能用吗？为什么？
- [ ] 什么时候用 PPO，什么时候用 SAC？
- [ ] Q-learning 的 max 操作为什么会造成 over-estimation？怎么修？
- [ ] policy gradient 公式的推导（log trick）

---

## 二、机器人 RL 工程实践（中高频）

### 2.1 仿真器

- [ ] MuJoCo 和 Isaac Gym / Isaac Sim 的区别？
- [ ] MJX 是什么？相比传统 MuJoCo 优势在哪？
- [ ] 为什么大规模并行环境对训练 locomotion 重要？
- [ ] 仿真器的"contact dynamics"为什么是 sim2real gap 的主要来源之一？

### 2.2 训练技巧

- [ ] 大规模并行 RL（4096+ env）会带来什么新问题？（如 reward 信号被平均掉）
- [ ] 什么是 privileged learning / teacher-student？为什么有效？
- [ ] reward shaping vs sparse reward 的取舍？
- [ ] curriculum learning 在 locomotion 里怎么设计？
- [ ] action smoothing / regularization 为什么对实物部署重要？

### 2.3 Sim2Real

- [ ] 域随机化（DR）通常随机什么参数？
- [ ] DR 太猛会怎样？太保守会怎样？
- [ ] 系统辨识（system identification）和 DR 的关系？
- [ ] actuator network 是什么？为什么需要？
- [ ] 为什么观测噪声 / 延迟 / 滤波在 sim 里也要建模？
- [ ] RMA 的 adaptation module 是怎么做的？

---

## 三、模仿学习与 VLA（2024+ 高频）

### 3.1 模仿学习基础

- [ ] Behavior Cloning 的 distribution shift 问题？
- [ ] DAgger 怎么修复？局限是什么？
- [ ] 为什么 IL 在数据量大时反而比 RL 实用？
- [ ] IL + RL 的混合方法（如 RLPD、residual policy）

### 3.2 现代 IL 方法

- [ ] Diffusion Policy 为什么用 diffusion model 生成 action？相比 Gaussian policy 优势？
- [ ] Action chunking（ACT）解决了什么？为什么 chunking 比单步预测好？
- [ ] temporal ensemble 的作用？
- [ ] Diffusion Policy 在推理时怎么加速？

### 3.3 VLA 概念

- [ ] 什么是 VLA（Vision-Language-Action）模型？
- [ ] RT-2、Octo、π0、OpenVLA 各自的特点？
- [ ] VLA 和 LLM 的关系？
- [ ] VLA 的 tokenization 怎么处理 action？

---

## 四、概念辨析（容易混的）

- [ ] model-based RL vs model-free RL
- [ ] online RL vs offline RL
- [ ] policy gradient vs value-based
- [ ] deterministic vs stochastic policy
- [ ] reward 是 dense 还是 sparse
- [ ] Behavior Cloning vs Inverse RL
- [ ] domain adaptation vs domain randomization
- [ ] zero-shot vs few-shot transfer

---

## 五、项目讲解（最重要！）

每个项目准备一份 STAR 法则讲解。

### 模板

```
[Situation]  我做这个项目的背景是什么。
             "我看到 2024 年具身智能赛道兴起，想系统训一遍 RL locomotion..."

[Task]       具体目标是什么，量化。
             "训出一个能在不平地形稳定行走的 Go2 策略，且在 M4 上单机训练 < 2 小时"

[Action]     你怎么做的，用了什么技术，**最重要的是 trade-off**。
             "选 MJX 而非 IsaacGym 因为 Mac 限制；
              用 PPO 而非 SAC 因为大规模并行 PPO 更稳；
              用 teacher-student + DR 增强鲁棒性..."

[Result]     结果，最好可视化。
             "训练 90 分钟在 4096 并行环境下收敛；
              在 5 种未见地形上 stand-up 成功率 87%；
              视频在这里 → [链接]"
```

### 常见追问

- "为什么不用 SAC？"
- "DR 你具体随机了哪些参数？范围怎么定的？"
- "如果给你真机，最大的挑战会在哪？"
- "你有没有跑过 ablation？最关键的 component 是什么？"
- "你这个工作和 [某 SOTA 论文] 比怎么样？"

提前准备好答案。

---

## 六、行为面试常见问题

- [ ] "为什么对机器人 / 具身智能感兴趣？"
- [ ] "你做过最难的 debug 是什么？"（仿真环境一定有故事，提前想好）
- [ ] "你有没有做过的事情失败的经历？"
- [ ] "未来 3-5 年你看好这个领域哪个方向？"
- [ ] "我们公司为什么选你？"

---

## 七、刷算法题（基础但必要）

如果应聘的是工程岗（不是纯研究岗），LeetCode 还是要刷。建议：

- 中等难度 100 题
- 重点：DP、图、二分、堆
- 一周 2-3 小时

---

## 八、求职渠道清单

### 招聘网站
- 国内：BOSS 直聘、拉勾、内推 Plus、北森智库的具身智能人才库
- 海外：LinkedIn、AngelList Talent (Wellfound)、Levels.fyi

### Twitter / 微信群
- 关注的 Twitter 账号：@svlevine、@chelseafinn、@DrJimFan、@andyzeng__、@cloneofsimo（Diffusion 相关）等
- 知乎关键词：具身智能 / 机器人学习 / 强化学习 招聘

### 内推
- LinkedIn 找在目标公司工作的校友
- 参加 RSS / CoRL / NeurIPS RL workshop（线上也行）
- GitHub 给目标公司开源项目提 PR，刷脸熟

---

## 九、心态

- 12 周不够"成为专家"，但**够"建立壁垒"**——别人没跑过 MJX、没读过 RMA、没写过 PPO，你都做了。
- 第一份 offer 可能不是最理想的，**先上车再说**。在职可以慢慢看更好的机会。
- 被拒了别气馁。机器人岗位 hire 周期长，每家有自己的 timing。
