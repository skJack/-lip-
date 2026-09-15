# Dream to Control: Learning Behaviors by Latent Imagination（Dreamer，即 DreamerV1）

> 对应「可lip」第 27 期视频（世界模型系列第一期）。论文：[arXiv:1912.01603](https://arxiv.org/abs/1912.01603)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 1912.01603（v1 2019-12，具体日期未核实；本笔记依据 v3 2020-03-17，页眉"Published as a conference paper at ICLR 2020"）· 机构: University of Toronto、Google Brain、DeepMind（Hafner、Lillicrap、Ba、Norouzi，按 main.tex 作者栏）· 代码/权重: 论文只给 https://danijar.com/dreamer（称 source code 和视频已公开）；据我所知对应 GitHub danijar/dreamer（TensorFlow 2），许可证 MIT 未核实；无预训练权重——在线 RL，每个任务从零 · 项目页: https://danijar.com/dreamer
- 一句话: 在 PlaNet 的 RSSM latent world model 上，把测试时的 CEM 在线规划换成"在想象轨迹（imagination rollout：只在 latent 里滚动 transition model、不渲染图像）里训练 actor-critic"：actor 的梯度沿可微的 latent dynamics 解析地反传（reparameterization / backprop through the world model，不是 REINFORCE），critic 用 λ-return V_λ 做 bootstrapping 把 credit assignment 推到有限 imagination horizon 之外。20 个 DeepMind Control Suite 视觉任务 5e6 步平均 823，超过 D4PG 1e8 步的 786 和 PlaNet 的 332（Fig 6 / 附录 G）。它定下了 Dreamer 系列（V2、V3、DayDreamer）沿用至今的"latent WM + imagination actor-critic"模板，把 latent world model 的用法从"在线规划"推向"在模型里训 policy"。

## 1. 要解决的问题
- latent dynamics model 相比像素空间预测内存小、可并行想象数千条轨迹（Intro）；但从 WM 里"取出行为"的已有路线，无论是参数化 policy（Dyna、World Models、SOLAR）还是在线规划（PETS、PlaNet），都只最大化固定 imagination horizon 内的想象 reward，会短视（引 Wang et al. 2019 的 benchmark）。
- 为了对模型误差鲁棒，先前工作普遍用 derivative-free 优化（CEM：VisualMPC、PETS、PIPPS），浪费了神经网络 dynamics 自带的解析梯度；planning by backprop（Schmidhuber 1990、Henaff）只在小问题上跑通、难扩展。
- PlaNet 每个环境步都要跑一遍 CEM，慢（1e6 步 11 h vs 本文 3 h），且没有可直接部署的 policy。
- 目标：一个纯在 latent 想象里训练的 actor-critic，既通过 value 顾及 horizon 之外的 reward，又高效利用 dynamics 梯度；全部超参跨任务固定。

## 2. 方法
**预测空间：state-space（latent），沿用 PlaNet 的 RSSM。** 三个组件（式 1）：representation model p_θ(s_t | s_{t-1}, a_{t-1}, o_t)、transition model q_θ(s_t | s_{t-1}, a_{t-1})、reward model q_θ(r_t | s_t)；p 记真实环境侧、q 记想象侧。latent 是 30 维对角高斯（附录 A）——与 V2/V3 的 categorical latent 不同；RSSM 的 GRU + 随机 latent 结构见 PlaNet / Dreamer-v3 笔记。CNN encoder / decoder 用 Ha & Schmidhuber World Models 的架构，其余全是 3 层 300 单元 ELU 的 dense 网络。
- **WM 训练目标（Sec 4）：** 默认 PlaNet 的重建式 ELBO / VIB（式 9–10）：J_O 像素重建 + J_R reward 预测 − β·KL[p(s_t | ·, o_t) ‖ q(s_t | s_{t-1}, a_{t-1})]，β = 1，KL 低于 3 free nats 不再优化（附录 A），不需要 latent overshooting。
- 表示学习目标被当作与 Dreamer 正交的可换件，另给两种：contrastive（式 11–12：把 observation model 换成 state model q(s_t | o_t)（CNN），用 InfoNCE mini-batch bound 估计 ln Σ_{o'} q(s_t | o') 防塌缩）和 reward-only；附录 B 从 information bottleneck 推导两个 bound。
- **动作怎么进出：** low-level 连续动作 a_{t-1}（1–12 维）作为 RSSM 输入；出口是 action model（actor）q_φ(a_τ | s_τ)——tanh 变换的高斯（SAC 式），均值 tanh 后 ×5（让动作能饱和）、标准差 softplus，重参数化 a_τ = tanh(μ_φ(s_τ) + σ_φ(s_τ)·ε)（式 3）；value model（critic）v_ψ(s_τ)（式 2）。
- 部署时只跑 representation model 做滤波 + actor 一次前向，不做 lookahead 规划（Alg 1 "Environment interaction"）。
- **想象环境（Sec 3）：** model state s_τ 是 Markov 的，所以想象出的 MDP 全观测；每次从 replay 采一批真实序列，以其中每个真实 model state s_t 为起点，actor 采动作、transition 采 s_τ、reward model 出 r_τ，滚 H = 15 步（连续任务；离散任务 10 步）；行为学习期间 WM 参数固定。
- **critic 目标 = λ-return（式 4–6）：** V_R 只累加 horizon 内 reward（"No value"消融）；V_N^k 用 k 步 reward + γ^k·v_ψ(s_h) bootstrap；V_λ 是各 V_N^n 的指数加权平均（γ = 0.99，λ = 0.95）。critic 用 MSE 回归 V_λ（式 8），目标 stop-gradient，没有 target network。
- **actor 目标 = 解析梯度（式 7，本文主贡献）：** max_φ E[Σ_τ V_λ(s_τ)]，∇_φ 直接沿"动作 → 下一 latent → reward / value"这条全由神经网络组成的链反传（stochastic backpropagation；latent 和连续动作用 reparameterization，离散动作用 straight-through）。没有 entropy bonus。
- 与其他 actor-critic 的区别（Sec 3 末段）：REINFORCE 类（A3C / PPO）只把 value 当 baseline 降方差，这里 value 本身被求导；DPG / DDPG / SAC 只对即时 Q 求梯度、不穿过 transition；MVE / STEVE 用学到的 dynamics 做多步 Q 目标，本文只需 state value、不需 Q，因为梯度已穿过 dynamics。
- **early termination：** WM 另有 discount 预测头（二分类，soft label 0 / γ），式 7–8 各项按预测 discount 的累乘加权（离散任务用）。
- **训练数据 / 流程（Alg 1，附录 A）：** 在线 RL：5 条随机 episode 起步；循环"100 个梯度步 → 用 actor 众数动作 + N(0, 0.3) 噪声采 1 条 episode"；batch 50 序列 × 50 步；Adam，lr WM 6e-4、actor / critic 8e-5，grad-norm clip 100；action repeat 固定 R = 2（Fig 12：R = 2 跨任务最好，不再像 PlaNet / SLAC 逐环境调）；预算 5e6 环境步；所有连续任务一套超参。
- 离散任务：categorical actor + straight-through，ε-greedy 0.4→0.1（前 20 万梯度步），H = 10，β = 0.1，reward 过 tanh。
- **开销（Sec 6 Implementation）：** 单张 V100 + 10 CPU 核；约 3 h / 1e6 环境步，PlaNet 在线规划 11 h，D4PG 达到相近性能用 24 h。测试时每步只有一次 RSSM 更新 + MLP 前向；wall-clock 延迟和参数量论文未报。
- **V1 vs V3（对照 Dreamer-v3 笔记，只列不同）：** latent 高斯 30 维（V2 起 categorical）；actor 用 dynamics backprop（V2 在 Atari 上已改 Reinforce，V3 全部 Reinforce + 回报百分位归一化 + 固定熵系数；V2 部分据其论文、未在本篇核实）。
- KL 是 β = 1 + 3 free nats（V3 KL balancing + 1 nat free bits）；critic MSE、无 target network（V3 symlog twohot + EMA 正则）；主文只做连续控制、离散另调超参（V3 单一配置跨领域）；探索靠固定高斯噪声（V3 直接从 policy 采样）；网络是几百万参数量级的小模型（未核实）而非 V3 的 12M–400M。


## 3. 实验
Setting：DMC 20 个视觉任务（64×64×3，动作 1–12 维，reward ∈ [0, 1]，episode 1000 步，随机初态，R = 2；选的是 Tassa et al. 2018 报告像素输入有非零分的任务）；5 seeds；baseline D4PG（像素，1e8 步）和 A3C（本体感知输入，1e8 步）直接取 Tassa et al. 的报告值，PlaNet 用 R = 2 重跑（5e6 步），附录曲线另画 SLAC。
- Fig 6 / 附录 G 表（5e6 步均分）：Dreamer 823.39 vs D4PG 786.32（1e8）、PlaNet 332.97、A3C 243.70。
- 长程 / 稀疏任务差距最大：Acrobot Swingup 365.26 vs D4PG 91.70 / PlaNet 3.21；Cartpole Swingup Sparse 812.22 vs 482.00 / 0.64；Hopper Hop 368.97 vs 242.00 / 0.37；Pendulum Swingup 833.00 vs 680.90 / 3.27；Cheetah Run 894.56 vs 523.80 / 496.12；Walker Run 824.67 vs 567.20 / 626.25；Quadruped Run / Walk 888.39 / 931.61（PlaNet 280.45 / 238.90，D4PG 无）。
- 输给 D4PG 的是细粒度接触 / 小目标任务：Finger Spin 498.88 vs 985.70、Finger Turn Easy 825.86 vs 971.40、Reacher Hard 817.05 vs 957.10。
- 最有信息量的消融 1——imagination horizon（Fig 4，4 任务，H = 3–40，估读）：Dreamer 在 H ≥ 10 时几乎不随 H 变（Cartpole Swingup 约 800、Cheetah Run 约 800、Quadruped Walk 约 900），H = 3–5 时会掉（Quadruped Walk 约 460–580）；No value 强依赖 H（Cartpole 始终 ≤ 350，Cheetah 从 140 爬到 H = 40 的 830，Quadruped 要 H ≥ 15 才到 700–940）；PlaNet 在 Quadruped Walk 上 ≤ 190、Cartpole ≤ 490。Walker Walk 是反应式任务，三者在 H = 10–30 都约 900–960，但到 H = 40 三者都掉（Dreamer 约 530、No value 约 740、PlaNet 约 590）——长程想象本身会漂。
- Fig 7（2e6 步曲线）：Acrobot、Cartpole Swingup Sparse、Hopper Hop / Stand、Pendulum 上 No value 与 PlaNet 几乎为 0，critic 是长程 credit assignment 的关键；附录 D（H = 20 全任务曲线，Fig 10）：Dreamer 20 任务中 16 胜 4 平（原文一处写 19 任务一处写 20）。
- 消融 2——表示学习目标（Fig 8 / 附录 E，8 任务，2e6 步）：像素重建在多数任务最好；contrastive 解决约一半（估读：Cup Catch、Walker Stand 约 950 接近重建，Finger Spin 约 750 反而高于重建的约 550，Hopper Stand 约 700 vs 930，Acrobot / Pendulum 明显差）；reward-only 基本学不到（估读 < 100）。作者据此说"表示学习的进步会直接转化为 Dreamer 的性能"。
- 离散动作 + early termination（Fig 9，附录 C）：14 个 Atari 游戏 + 2 个 DMLab 关卡（Collect Good Objects、Watermaze），sticky actions 协议；能学到有效行为（估读：Boxing 约 100 超过 DQN、Kangaroo 约 1 万接近 DQN），但多数游戏不及 Rainbow / DQN 2e8 步，作者明说"纯靠 WM 学习的 agent 在这些领域还不具竞争力"。
- Fig 5：RSSM 在 hold-out 轨迹上给 5 帧上下文后开环预测 45 帧仍准确。

## 4. 局限
- 作者承认：reward-only 表示不够、contrastive 弱于重建，视觉更复杂的环境要等表示学习进步（Conclusion）；Atari / DMLab 上不具竞争力（附录 C）；action repeat 实验用的是旧超参、仅 2 seeds（Fig 12）。
- 我读出来的 (1)：单任务在线 RL，必须有 reward，latent 被 reward 头绑成 task-specific，每个任务从零跑 5e6 步——真机不可承受，DayDreamer 后来专门处理。
- (2) 解析梯度依赖 dynamics 光滑、准确，actor 会利用 WM 误差，这正是 V3 改 Reinforce 的动机（见 Dreamer-v3 笔记）；DINO-WM 也发现测试时直接对动作序列做梯度规划不如 CEM（其 Table 8）——说明"梯度穿过 WM"适合训练 amortized policy，不适合直接优化动作，这是 Dreamer 独有的定位。
- (3) 30 维高斯 latent + 64×64 + 从零训的 CNN 表达力有限，V2 换 categorical 才在 Atari 上翻盘；(4) 探索只是固定 0.3 的高斯噪声、无熵项，稀疏 / 硬探索任务靠运气。
- (5) 想象只从 replay 状态出发、只在分布内可信，Fig 4 中 H = 40 时连 Dreamer 自己也掉分；(6) D4PG / A3C 是他人报告值而非同协议重跑；(7) 无目标 / 语言条件，无多任务。

## 5. 复现要点
- 代码：danijar.com/dreamer → GitHub danijar/dreamer（TF2 + TensorFlow Probability；许可证据我所知 MIT，未核实）；后续 danijar/dreamerv2、dreamerv3 已取代它，V3 的 PyTorch 移植（NM512/dreamerv3-torch）见 Dreamer-v3 笔记；第三方 PyTorch 复现多个，质量未核实。没有预训练权重。
- 算力：单 V100 + 10 CPU 核，约 3 h / 1e6 步 → 一个 5e6 步 run 约 15 h；20 任务 × 5 seeds 约 1500 V100 小时。8 张 H100 一类的机器绰绰有余，瓶颈是 MuJoCo 渲染的 CPU 和 replay ratio 而非 GPU；DMC 无头渲染要配 EGL / OSMesa。
- 关键超参（附录 A）：latent 30 维高斯、dense 3×300 ELU、batch 50×50、H = 15、γ 0.99、λ 0.95、lr 6e-4 / 8e-5 / 8e-5、grad clip 100、free nats 3、β 1、R = 2、探索噪声 0.3、S = 5、C = 100。
- 坑：action repeat 影响大（Fig 12），别照搬 PlaNet 逐环境的值；离散任务必须换 β = 0.1、H = 10、tanh reward、ε-greedy；行为学习时要 stop WM 梯度、critic 目标要 stop-gradient。
- 结果方差大（Fig 7 / 8 阴影），至少 5 seeds；Finger / Reacher 类任务本来就弱，别拿它们当 sanity check；想象 horizon 别超过 30（Fig 4）。

## 6. 关键引用链
- 建立在：PlaNet（同一个 RSSM + 重建目标，本文把它的 CEM 规划换成 actor-critic；见 PlaNet）；World Models（Ha & Schmidhuber 2018：CNN 编解码架构，以及"在想象里学控制器"的两阶段前身，本文改为联合训练 + 梯度学习；见 World-Models）；Dyna（Sutton 1991）的"学模型 / 学行为 / 交互"三件套。
- 方法上的近亲：SVG（Heess 2015）与 DPG / DDPG / SAC 的 value 梯度和 tanh 高斯 policy；MVE / STEVE 的多步想象目标；Schmidhuber 1990 / Henaff 的 planning by backprop；同期 IVG（Byravan 2019，确定性模型的 latent imagination）；MuZero（2019-11，value-equivalent 模型 + MCTS，另一条路，本文说它"需要大量经验"；见 muzero-value-equivalent-planning）。
- 后续：DreamerV2（categorical latent + KL balancing，Atari；见 Dreamer-v2）→ DreamerV3（Reinforce actor + robustness 技巧，跨领域固定超参；见 Dreamer-v3）→ DayDreamer（真机；见 DayDreamer）。
- 对照与引用者：TD-MPC2 走"latent + 规划 + value、不重建"的路线（见 tdmpc2-scalable-latent-mpc）；DINO-WM / V-JEPA 2 把 Dreamer 系列当离线 / 仿真 baseline（见 dino-wm、vjepa2）；tutorial-wm-to-wam、survey-world-model-robot-learning 把这条线归为 state-space latent WM。
