# Learning Latent Dynamics for Planning from Pixels（PlaNet）

> 对应「可lip」第 27 期视频（世界模型系列第一期）。论文：[arXiv:1811.04551](https://arxiv.org/abs/1811.04551)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 1811.04551（v1 2018-11，精确到日未核实；本地 PDF 为 v5 2019-06-04；ICML 2019, PMLR 97）· 机构: Google Brain、DeepMind、Google Research、University of Toronto、University of Michigan（按作者标注）· 代码/权重: 论文只给 https://danijar.com/planet（TensorFlow + TensorFlow Probability 实现）；据我所知代码仓库是 github.com/google-research/planet（Apache-2.0，未核实）；没有预训练权重，每个任务在线从零学 · 项目页: https://danijar.com/planet
- 一句话: PlaNet（Deep Planning Network）是一个"纯 model-based"agent：从 64×64 像素学一个 RSSM（Recurrent State-Space Model：确定性 GRU 状态 h_t + 随机高斯 latent s_t 的序列 VAE）latent 动力学模型，然后每一步在 latent 空间里用 CEM 做 MPC 选动作，没有 policy 也没有 value 网络；在 DeepMind Control Suite 6 个像素任务上用 1,000 episodes 达到 D4PG 用 100,000 episodes 的水平（Table 1，平均约 200× 数据效率）。在世界模型史上它做了三件事：把 World Models 的"分开训 VAE + MDN-RNN、进化搜索 controller"改成"端到端序列 VAE + reward 头 + 在线规划"；首次提出 RSSM，此后 Dreamer v1/v2/v3、DayDreamer 全部沿用；提出 latent overshooting（多步 latent 一致性正则），但 v5 版自己就不再用、Dreamer 系列也没有继承。它还开启了"latent 空间 MPC"这条线——Dreamer v1 用 actor-critic 取代了它的 CEM，TD-MPC2、DINO-WM、V-JEPA 2-AC 又回到了在 latent 里做采样式规划。

## 1. 要解决的问题
- 已知动力学时规划很强（MPC、AlphaGo），未知环境要先从交互里学 dynamics；从像素学到"准到能规划"的模型一直是难题，Introduction 列出四个难点：模型误差、多步预测误差累积、无法表达多个可能的未来、训练分布外过度自信。
- 之前能规划的学习模型要么假设能拿到底层状态和 reward function（PILCO、PETS、Henaff 2018），要么 latent 模型只能做 cartpole 平衡和 2-link arm 且要 dense reward（E2C、RCE，还假设单帧即状态的 Markov 性）。
- 机器人界的 Visual Foresight 一类在像素空间逐帧生成视频来规划（Related Work）：能应付真实世界的视觉复杂度，但太贵，评估不了上千条候选序列，任务也限于简单夹爪的抓取 / 推动。
- 目标：在紧凑 latent 空间里快速规划，处理 contact dynamics（finger、cheetah、walker）、partial observability（cartpole 相机固定，车会出视野）和 sparse reward（reacher、cup），同时大幅减少环境交互；顺带验证 model-based 的三个承诺——数据效率、算力换性能、dynamics 与任务无关可迁移。

## 2. 方法
**预测空间：state-space（latent）。** 问题建模为 POMDP（部分可观测 MDP：单帧图像不含完整状态，式 1）。模型四件套（Section 3）：transition p(s_t | s_{t-1}, a_{t-1})、observation p(o_t | s_t)（deconv 解码器，单位方差高斯 = MSE）、reward p(r_t | s_t)（标量高斯）、encoder q(s_t | o_{≤t}, a_{<t})（CNN + MLP 的 filtering 后验，只看过去不看未来）。observation model 只提供训练信号，规划时不用。
- **RSSM（式 4）：** h_t = f(h_{t-1}, s_{t-1}, a_{t-1})（GRU，200 维）；s_t ~ p(s_t | h_t)（30 维对角高斯）；o_t、r_t 由 (h_t, s_t) 解码；后验 q(s_t | h_t, o_t)。作者说可以理解为非线性 Kalman filter 或 sequential VAE。DreamerV3 里的 RSSM 就是这个结构，只是 s_t 换成了离散 categorical（见 Dreamer-v3 笔记）。
- **为什么确定性路径 h 和随机路径 s 必须并存**（Section 3 "Deterministic path"、Fig 2 说明、Section 5 "Model designs"）：纯随机 SSM 每一步都要过一次采样，信息要跨多步保留就得反复穿过噪声——理论上模型可以把某些维度的方差学成 0 来记忆，但优化器实际找不到这个解，于是"记不住多步之前的信息"（比如出视野的 cart 在哪）。纯确定性 GRU 只能输出一个未来，表达不了 partial observability 带来的多模态（从 agent 视角环境是随机的，初始状态看不全），而且 planner 会去利用模型的确定性错误（exploit inaccuracies）——CEM 会找到让模型幻觉出高 reward 的动作序列。
- 实验上 GRU 几乎完全学不会，作者原话是没有随机成分 agent 就不学习，并猜测噪声还给规划目标加了 safety margin，让选出的动作序列更鲁棒；h 则负责跨多步记忆。关键实现细节：观测的全部信息必须经过 encoder 的采样步进入 s_t，不能有从 o_t 到重建的确定性捷径（否则随机路径会被绕过）。
- **损失（式 3，附录 F 推导）：** 标准 sequential ELBO（变分下界）= Σ_t [ E_q ln p(o_t | s_t)（重建）− KL[ q(s_t | o_{≤t}, a_{<t}) ‖ p(s_t | s_{t-1}, a_{t-1}) ]（complexity）]，reward 项同理；用单个 reparameterized 样本估计。KL 不加权重，但给 3 free nats（KL 低于 3 不再惩罚；DreamerV3 是 1 nat 并加 KL balancing，见 Dreamer-v3 笔记）。附录 A 说"不对 KL 相对缩放"却又把 KL 尺度 β 列为重要超参，原文略有不一致。
- **Latent overshooting（Section 4，本文提出）：** 标准 ELBO 里随机 transition 只通过一步 KL 训练，梯度从不穿过多步 p 链；但容量有限的模型"一步预测最优"不等于"多步预测最优"，而规划恰恰需要多步。做法：对每个距离 d = 1..D，从 t−d 的后验出发把 prior 往前 roll d 步得到多步 prior p(s_t | s_{t−d})，再对后验 q(s_t | o_{≤t}) 做 KL（式 6–7），按 β_d 加权平均；d > 1 时对后验 stop-gradient，只让多步预测去追后验。全程在 latent 里算，不像 observation overshooting（Amos 2018）那样要解码额外图像；作者猜想它仍是一步似然的下界（data processing inequality，附录 F）。
- 结果（附录 D，Fig 8）：对弱模型（DRNN 等）显著有帮助，对 RSSM 反而略降；附录 A 承认早期版本用了 overshooting + 固定全局 prior，最终 agent 不用。Dreamer v1 起再没用过这个正则；"多步 latent 一致性"的思路以另一种形式（rollout 上逐步的 consistency loss）回到了 TD-MPC 系列（见 TD-MPC2（arXiv:2310.16828） 笔记）。
- **动作怎么进、怎么出：** a_{t-1} 直接作为 GRU 输入（式 4）。输出侧没有 policy：每个决策步在 latent 里跑 CEM（cross-entropy method，采样式优化：维护动作序列的对角高斯 N(μ_{t:t+H}, σ²I)，采 J 条候选，用模型 roll 出 latent 轨迹并把 reward 头的均值求和当回报，取最好的 K 条重新拟合 μ、σ，迭代 I 次），返回当前步的均值 μ_t（Algorithm 2）。
- MPC（model predictive control）= 每收到新观测就重新规划，只执行第一个动作；每步都从 N(0, I) 重启以避免局部最优（不 warm-start）。每条候选只采一条 latent 轨迹，把算力花在更多候选上。默认 H = 12、I = 10、J = 1000、K = 100，即每个决策步 10^4 条 12 步 latent rollout（附录 A）。因为 reward 是 latent 的函数，规划全程不解码图像，这是能大批量评估的关键。
- **训练循环（Algorithm 1）：** S = 5 条随机动作 seed episodes 起步；每 C = 100 步梯度更新后用当前模型规划采 1 条新 episode（动作加 N(0, 0.3) 噪声）；action repeat R（每个动作重复 R 步，缩短规划 horizon 并让学习信号更清晰）按任务 2–8；batch B = 50 × chunk L = 50；Adam 1e-3、ε = 1e-4、梯度裁剪 1000；图像降到 5 bit；conv/deconv 编解码器直接取自 World Models（附录 A）。
- **训练数据 / 开销：** 全部在线自采，每任务 1,000 episodes；单张 V100 训 10–20 小时（Section 5）；决策步的 wall-clock 延迟论文未报告。
- **控制用途：** reward 条件的在线规划——任务完全由环境 reward 定义，没有目标图像或语言接口；同一模型可以同时服务多个任务（附录 C），但每个任务仍要自己的 reward。


## 3. 实验
Setting：DM Control 6 任务（cartpole swingup、reacher easy、cheetah run、finger spin、cup catch、walker walk），观测只有 64×64×3 第三人称图像；每任务 5 seeds × 10 条测试轨迹；除 action repeat 外全任务共用超参；A3C / D4PG 数字取自 Tassa et al. 2018 的报告（未重跑，A3C 还是用 proprioceptive 状态训的）；数据效率倍数是从 D4PG 训练曲线估的（Table 1 说明）。
- Table 1（1,000 episodes 后最终均分，顺序同上）：PlaNet 821 / 832 / 662 / 700 / 930 / 951；D4PG（像素，100,000 episodes）862 / 967 / 524 / 985 / 980 / 968；A3C（proprio，100,000 episodes）558 / 285 / 214 / 129 / 105 / 311；CEM + 真实模拟器（同样 H=12, I=10, J=1000, K=100，作为上界）850 / 964 / 656 / 825 / 993 / 994。数据效率倍数 250× / 40× / 500+× / 300× / 100× / 90×，平均约 200×。cheetah 上比 D4PG 高 26% 且与真模拟器规划持平（662 vs 656）；finger 是唯一 500 episodes 后仍明显落后 D4PG 的任务。
- 学习曲线（Fig 4）：100 episodes 内所有任务超过 A3C，500 episodes 接近 D4PG（finger 除外）。
- 最有信息量的消融——模型结构（Fig 4，估读 1,000 episodes 时中位数）：纯确定性 GRU 几乎不学（cheetah ≈ 30、walker ≈ 40、cartpole ≈ 170、finger / cup ≈ 0）；纯随机 SSM 学得慢且方差极大（cheetah ≈ 400 且 800 episodes 后才升起来、walker ≈ 350–400、cartpole ≈ 550、finger ≈ 630、cup 在 0 和 900 之间跳）；RSSM 最好且稳定。Fig 4 说明提到 GRU 版不用 latent overshooting（因为没有随机 latent），暗示该图的 SSM 对照可能用了 overshooting，原文未明确。
- Agent 设计（Fig 5，估读）：random collection（数据全靠随机动作采、只规划不用规划去采）与 random shooting（每步从 1,000 条随机序列里挑最好、不迭代）都更差：cheetah 上 PlaNet ≈ 660、random collection ≈ 470、random shooting ≈ 380；cartpole 820 / 370 / 580。在线规划采集对 cartpole、finger、walker 是必需的；CEM 迭代在所有任务上都优于 random shooting。
- Latent overshooting（附录 D，Fig 8，估读）：RSSM 加不加曲线基本重合（cheetah ≈ 660 vs ≈ 640），DRNN（两个 RNN 夹一个随机状态序列）从 ≈ 120 提到 ≈ 250；walker 上 DRNN 两版都只有 100–150。
- 多任务（附录 C，Fig 6 / 7）：单个 agent 同时学 6 任务，不给任务 id、靠图像推断，全部能解但更慢（估读 6 任务平均在 1,000 episodes 时 ≈ 650 vs 分开训 ≈ 780）——"共享 dynamics 做多任务"的最早演示之一。
- 其他附录：Fig 9 激活函数——ELU 帮 SSM / GRU，RSSM 对此不敏感；Fig 10 开环视频预测——5 帧上下文后 cheetah 50 步像素级准确；Fig 11 状态诊断——冻结模型后小网络能从 latent 开环解出真实位置、速度、reward，且准到超过规划 horizon。
- 规划器参数扫描（附录 J，Fig 12，用真模拟器、cheetah，分数 132–837）：horizon 6 不够、8 附近最好、更长反而因搜索空间变大而差，proposals / iterations 越多越好，拟合比例 K/J 取 0.05–0.1 最好；注意主实验仍用 H = 12。

## 4. 局限
- 作者承认（Discussion、Section 5）：固定 action repeat 而不是学到的时间抽象；没有 value function，看不到 horizon 之外的回报（finger 上不及 D4PG）；CEM 采样规划算力大，gradient-based planning 可能更省；像素重建让视觉多样性高的任务困难，需要"不重建的表征"；多任务只是初步。这几条分别成了 Dreamer（value + 解析梯度）和 TD-MPC / MuZero / JEPA 系（不重建）的出发点。
- 我读出来的：(1) 全部仿真、64×64 单视角固定相机、5 bit 色深；(2) 任务只能由环境 reward 定义，latent 被 reward 头绑成 task-specific，没有目标图像 / 语言接口；(3) 每任务从零在线交互 1,000 episodes，真机上仍是几十小时的自主试错；(4) 规划只看 H=12 × R 步，稀疏 reward 只有落在 horizon 内才能被规划到；(5) 每个决策步 10^4 条 rollout，延迟未报告，真机实时性存疑；(6) 30 维单峰高斯 latent 的多模态表达力有限（DreamerV2 改离散）；(7) baseline 数字取自他人论文、A3C 输入还是 proprio，"200×"是估读曲线得到的；(8) 稀疏奖励任务 5–95 分位带极宽，结论依赖 5 seeds 中位数；(9) Introduction 点名的"分布外过度自信"并未被方法解决，只靠随机路径的噪声当 safety margin，规划器不知道模型什么时候错。

## 5. 复现要点
- 代码：论文只给 danijar.com/planet，实现是 TensorFlow + TensorFlow Probability（Section 5）；据我所知开源仓库为 github.com/google-research/planet（Apache-2.0，未核实），社区有 PyTorch 复现（未核实）。没有预训练权重。实际复现更建议拿 danijar/dreamerv3 里的 RSSM 把 actor 换成 CEM——原版 TF1 代码年代久远。
- 模型很小：GRU 200 维、latent 30 维、两层 200 维 MLP、World Models 的 conv/deconv；参数量论文未报告（估计百万量级，未核实）。
- 算力：单张 V100 每任务 10–20 h；5 seeds × 6 tasks = 30 个 run。8 张 H100 一类的机器绰绰有余，一张卡可并行多个 run；瓶颈在 dm_control（MuJoCo）渲染的 CPU 和每决策步 10^4 条 CEM rollout。
- 数据：全在线；S=5 随机 seed episodes，之后每 C=100 步更新采 1 条 episode，动作噪声 N(0, 0.3)；1,000 episodes 内出结果；batch 50 × chunk 50。
- 坑（模型 / 超参）：action repeat 是唯一逐任务调的超参也最敏感（cartpole 8、reacher 4、cheetah 4、finger 2、cup 4、walker 2），作者点名的重要超参是 action repeat、KL 尺度、学习率；free nats = 3；图像降 5 bit；观测信息必须经过采样步（别给 h 加从 o 的直连）；RSSM 不需要 latent overshooting（加了略降）；Adam ε=1e-4、梯度裁剪 1000 是原文设置。
- 坑（规划 / 评估）：CEM 每步从 N(0, I) 重启；horizon 与 K/J 比例见 Fig 12；稀疏奖励任务方差极大，必须跑 5 seeds 看中位数；A3C / D4PG 对照数字直接沿用 Tassa 2018，自己重跑 D4PG 要 100,000 episodes 量级的交互。

## 6. 关键引用链
- 建立在：World Models（Ha & Schmidhuber 2018，slug World-Models）——直接沿用其 conv/deconv 编解码器和"VAE latent + RNN 动力学"骨架，但把分开训练的 VAE / MDN-RNN + 进化 controller 改为端到端 sequential VAE + reward 头 + 在线 CEM 规划；PlaNet 自己把 World Models 归入"hybrid agent（在想象经验上加速 policy 学习）"一类。
- 建立在：VRNN（Chung 2015）、DSSM（Buesing 2018）给出确定性 + 随机状态的组合（VRNN 把生成的观测喂回模型，前向预测贵；Buesing 的模型相近但用于 hybrid agent 而非显式规划）；DVBF（Karl 2016）、PR-SSM（Doerr 2018）是同期 latent 序列模型；PETS（Chua 2018）是状态空间里的 CEM 规划；Amos 2018 的 observation overshooting 是 latent overshooting 的前身；E2C / RCE 是 latent 规划的早期尝试（局部线性 + LQR）；Visual Foresight（Finn & Levine 2017、Ebert 2018，slug arXiv:1812.00568）是被对比的像素空间规划路线；任务与 baseline 来自 DM Control（Tassa 2018）。
- 后续（Dreamer 线）：Dreamer v1（Dreamer-v1）沿用 RSSM，用在想象轨迹里训的 actor-critic 取代 CEM（critic 解决 horizon 之外的回报，actor 用穿过 latent dynamics 的解析梯度训练——正是本文 Discussion 里的两条 future work），并以 PlaNet 为 baseline；Dreamer v2（Dreamer-v2）把 s_t 改离散、加 KL balancing；DreamerV3（Dreamer-v3）、DayDreamer（DayDreamer）把同一 RSSM 推到 150+ 任务和真机。
- 后续（latent 规划线）：MuZero（arXiv:1911.08265）是平行的"不重建、value-equivalent latent + MCTS（蒙特卡洛树搜索）"规划；TD-MPC2（arXiv:2310.16828）回到 latent MPC，但用 MPPI（与 CEM 同类的采样式 MPC，softmax 加权而非 top-K 重拟合）+ 终端 value + policy prior 采样、latent 不重建像素（据我所知还会 warm-start 上一步的计划，未核实）；DINO-WM（dino-wm）、V-JEPA 2-AC（vjepa2）把同样的 CEM-MPC 搬到预训练视觉特征空间并用目标图像距离代替 reward；LeCun 2022（LeCun-2022-立场文）的"JEPA 世界模型 + cost + MPC"蓝图可以看作 PlaNet 配方去掉重建后的通用化；tutorial（WM-to-WAM-教程）把这一整支归为 state-space / latent state WM。
