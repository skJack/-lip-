# 第 41 期 · 具身智能中的强化学习（一 · 上集）

**视频标题**：强化学习从零讲清楚：机器狗在仿真里练完，直接上真机（上）

新系列「具身智能中的强化学习」第一期「仿真里练」的上集。先把强化学习从零讲清楚：监督、模仿、强化三种学法差在哪，一个「状态 → 动作 → 奖励 → 更新策略」的循环，机器人的三层（大脑 / 操作 / 运动控制）里 RL 各成熟到哪一步，三期的路线图。然后是一张 **PPO 训练全景图**（策略网络 + 价值网络，五步一轮），五步逐页讲：1 采样（策略出的是概率，按概率抽就是探索；每一步打一个分）→ 2 回报（之后所有奖励加起来，越远打折越多）→ 3 价值（这个状态平时能拿多少）→ 4 优势（这一次比平时好多少）→ 5 更新（按优势改概率，每次最多改 20%；数据用完就扔还是存起来反复用），再用一个新骑手送外卖的例子把五步串一遍。接着讲放到机器人身上的难点：和大模型的 RL 有什么不一样、为什么非得去仿真里练、仿真和真机差在哪。后半段是 **ETH 四足这条线**：2019 年之前靠模型控制、之后 RL 慢慢成了标配的故事；**Hwangbo 2019** 照开源代码详讲 14 页（四步配方：刚体模型 → 用不到 4 分钟的真机数据学一个执行器网络 → 在仿真里练三个任务 → 上真机；97 个数进、12 个目标角出，奖励、课程、代码里的一步、价值网络、TRPO 训练流程）；**Lee 2020** 师生蒸馏，只靠关节和 IMU 走野外；**legged_gym** 一张 GPU 上 4096 只、二十来分钟学会走，游戏式地形课程、为什么都用 PPO、人手写的 9 项奖励；最后是 ETH 之后的工作（一个策略走废墟、羽毛球、爬梯子、人形荡单杠），收在一句话：策略只在仿真里练，仿真靠真机数据调准。

## 这期的材料从哪来

这期讲的是方法和一条研究线，不是单篇论文：RL 基础的数字来自 PPO 原文和 legged_gym 的开源配置，ETH 段按 Hwangbo 2019 的论文和开源代码详讲。下表是 deck 里用到的论文（arXiv 链接）、代码，和相关的演示视频（deck 里用的是论文图，原片在这里）；每页数字的出处在 `notes/技术事实核查.md`。

| 类型 | 材料 | 链接 |
|---|---|---|
| RL 基础 | PPO：Proximal Policy Optimization Algorithms（OpenAI 2017） | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| RL 基础 | TRPO：Trust Region Policy Optimization（Hwangbo 2019 用的训练算法） | [arXiv:1502.05477](https://arxiv.org/abs/1502.05477) |
| RL 基础 | GAE：Generalized Advantage Estimation（优势怎么估） | [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) |
| RL 基础 | FastTD3（off-policy 的例子） | [arXiv:2505.22642](https://arxiv.org/abs/2505.22642) |
| RL 基础 | FastSAC：Learning Sim-to-Real Humanoid Locomotion in 15 Minutes（off-policy 的例子） | [arXiv:2512.01996](https://arxiv.org/abs/2512.01996) |
| 三层 · 大脑 | Cosmos-Reason1 | [arXiv:2503.15558](https://arxiv.org/abs/2503.15558) |
| 三层 · 大脑 | Embodied-R1 | [arXiv:2508.13998](https://arxiv.org/abs/2508.13998) |
| 三层 · 操作 | π*0.6（第 23 期讲过） | [arXiv:2511.14759](https://arxiv.org/abs/2511.14759) |
| 三层 · 操作 | GR-RL | [arXiv:2512.01801](https://arxiv.org/abs/2512.01801) |
| 路线图 · 第二期 | SimpleVLA-RL | [arXiv:2509.09674](https://arxiv.org/abs/2509.09674) |
| 路线图 · 第二期 | πRL | [arXiv:2510.25889](https://arxiv.org/abs/2510.25889) |
| 路线图 · 第二期 | RISE | [arXiv:2602.11075](https://arxiv.org/abs/2602.11075) |
| 路线图 · 第三期 | QT-Opt | [arXiv:1806.10293](https://arxiv.org/abs/1806.10293) |
| 路线图 · 第三期 | HIL-SERL | [arXiv:2410.21845](https://arxiv.org/abs/2410.21845) |
| 为什么去仿真 | Dactyl：Learning Dexterous In-Hand Manipulation（仿真里练了相当于 100 年） | [arXiv:1808.00177](https://arxiv.org/abs/1808.00177) |
| ETH 故事 | Sim-to-Real: Learning Agile Locomotion For Quadruped Robots（Google 2018，Minitaur） | [arXiv:1804.10332](https://arxiv.org/abs/1804.10332) |
| ETH 故事 | Rapid Locomotion via Reinforcement Learning（Mini Cheetah 跑到 3.9 m/s） | [arXiv:2205.02824](https://arxiv.org/abs/2205.02824) |
| ETH · 详讲 | Hwangbo 2019：Learning agile and dynamic motor skills for legged robots（执行器网络） | [arXiv:1901.08652](https://arxiv.org/abs/1901.08652) |
| ETH | Lee 2020：Learning Quadrupedal Locomotion over Challenging Terrain（师生蒸馏） | [arXiv:2010.11251](https://arxiv.org/abs/2010.11251) |
| ETH | Rudin 2021：Learning to Walk in Minutes Using Massively Parallel Deep RL（legged_gym） | [arXiv:2109.11978](https://arxiv.org/abs/2109.11978) |
| ETH | Miki 2022：Learning robust perceptive locomotion for quadrupedal robots in the wild | [arXiv:2201.08117](https://arxiv.org/abs/2201.08117) |
| ETH | ANYmal Parkour（Hoeller 2024） | [arXiv:2306.14874](https://arxiv.org/abs/2306.14874) |
| ETH 之后 | Parkour in the Wild（一个策略走废墟） | [arXiv:2505.11164](https://arxiv.org/abs/2505.11164) |
| ETH 之后 | Robust Ladder Climbing with a Quadrupedal Robot（爬梯子） | [arXiv:2409.17731](https://arxiv.org/abs/2409.17731) |
| ETH 之后 | Learning Agile Perceptive Traversal of Sparse 3D Structures for Humanoids（人形荡单杠） | [arXiv:2608.29769](https://arxiv.org/abs/2608.29769) |
| ETH 之后 | Learning coordinated badminton skills for legged manipulators（羽毛球，Science Robotics 2025） | <https://doi.org/10.1126/scirobotics.adu3922> |
| 综述 | Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes（Tang 2024） | [arXiv:2408.03539](https://arxiv.org/abs/2408.03539) |
| 封面图 | BeyondMimic（封面的树林侧手翻，下集详讲） | [arXiv:2508.08241](https://arxiv.org/abs/2508.08241) |
| 代码 | legged_gym | <https://github.com/leggedrobotics/legged_gym> |
| 代码 | rsl_rl（legged_gym 用的 PPO 实现） | <https://github.com/leggedrobotics/rsl_rl> |
| 视频 | Hwangbo 2019 Movie S7：摔倒后自己翻身 | <https://youtu.be/bbp2vcNb7jg> |
| 视频 | Hwangbo 2019 Movie S6：高速奔跑 | <https://youtu.be/wR3xnK0ZCNs> |
| 视频 | Hwangbo 2019 Movie S2：执行器网络训出来的策略 | <https://youtu.be/23mBeaGmQ2o> |
| 视频 | Lee 2020 主视频：野外地形 | <https://youtu.be/9j2a1oAHDL8> |
| 视频 | Lee 2020 Movie S3：绊脚反射 | <https://youtu.be/tPixnjLbTvE> |
| 视频 | legged_gym 主视频 | <https://www.youtube.com/watch?v=8sO7VS3q8d0> |
| 视频 | 羽毛球主视频 | <https://youtu.be/zYuxOVQXVt8> |
| 视频 | 爬梯子项目页 | <https://sites.google.com/leggedrobotics.com/climbingladders> |
| 视频 | 人形荡单杠项目页 | <https://nemantor.github.io/sparse-3d-traversal-website/> |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：视频和 deck 里每个数字的出处，按页排，每条标「已核 / 精读已核 / 推算 / 判断」；ETH 段开头列了几处口径不一致的地方（比如 legged_gym 的奖励权重，论文附录表和开源代码不一样），还核了一个常听到的说法：ETH 之前波士顿动力是不是无敌、机器狗控制是不是已经被解决。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 上集没有视频页；分步页（第 7、17、33、37、38 页）在这一页按 `G` 或 `J` 走一步、`K` 退一步。论文里的演示视频没有放进 deck，原片链接见上表的「视频」几行。
