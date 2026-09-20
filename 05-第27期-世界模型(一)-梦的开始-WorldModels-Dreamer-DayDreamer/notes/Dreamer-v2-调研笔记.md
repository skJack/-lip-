# Mastering Atari with Discrete World Models（DreamerV2）

> 对应「可lip」第 27 期视频（世界模型系列第一期）。论文：[arXiv:2010.02193](https://arxiv.org/abs/2010.02193)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2010.02193（v1 2020-10-05，v4 2022-02-12；ICLR 2021）· 机构: Google Research、DeepMind、University of Toronto（Hafner、Lillicrap、Norouzi、Ba）· 代码/权重: 论文脚注给出项目页含源码与 JSON 训练曲线；GitHub 据我所知为 danijar/dreamerv2（TensorFlow 2，MIT），未核实；没有预训练权重，每个游戏从零在线 RL · 项目页: https://danijar.com/dreamerv2
- 一句话: 在 PlaNet / DreamerV1 的 RSSM 上做两处小改——把对角高斯 latent 换成 32×32 的 categorical latent（straight-through 梯度），用 KL balancing 替换 free nats——外加 Atari 离散动作改用 Reinforce，让"只在 world model 想象里训 actor-critic"的 agent 第一次在 Atari 55 游戏 200M 帧上达到人类水平、超过单 GPU 的 model-free 冠军 Rainbow / IQN（Table 1）。它是 latent world model 从"连续控制小任务"走向"通用 RL benchmark"的转折点：离散 latent + KL balancing 从此成为 Dreamer 系（Dreamer-v3、DayDreamer）的标配，"表征主要靠图像重建而不是 reward"这一后来支撑通用 WM 预训练路线的证据也首次在这里给出（Table 2）。

## 1. 要解决的问题
- DreamerV1 只在 DM Control 连续控制上验证过；Atari 一直是 model-free 的地盘（DQN、A3C、Rainbow），此前的 Atari 世界模型（Oh 2015、Chiappa 2017、SimPLe）都没到有竞争力的水平（Intro）。SimPLe 只在较易的 36 个游戏、少量帧上评测，且继续训练也不涨（Fig 1 说明）。
- MuZero 证明"学模型 + planning"可以很强，但不开源、模型只拟合 reward / value 不学图像表征、单 GPU 要训 2 个月以上（Table 3：20B 帧、80 加速器天）。
- 目标：用"分开训练的 world model + policy 纯靠想象学习"证明 WM 能把 Atari 学准；约束是单 GPU、单环境实例、10 天内 200M 帧，与 Rainbow 同预算。
- 附带议题：Atari 常用汇总指标会被少数离群游戏主导，提出 clipped record-normalized mean（按人类世界纪录归一化、截断到 1 再对游戏取均值）。

## 2. 方法
**预测空间：state-space（latent）。** 世界模型 = CNN encoder + RSSM + image / reward / discount 三个预测头（式 1，Fig 2）；RSSM（确定性 GRU 状态 h_t + 随机 latent z_t 的序列模型）的基本结构见 Dreamer-v3 笔记，下面只写 v2 首次引入或与后续不同的部分。
- **Categorical latent（首次）：** z_t 是 32 个 categorical 变量、每个 32 类，展开成 1024 维、恰好 32 位为 1 的稀疏二值向量（3.2）；posterior q(z_t | h_t, x_t) 和 prior p(ẑ_t | h_t) 都是 categorical。采样不可导，用 straight-through（Alg 1：`sample + probs − stop_grad(probs)`，前向传 one-hot 样本、反向拿 softmax 概率的梯度）。试过更多的 binary latent 代替 categorical，更差（App C）。
- **为什么离散更好——作者只给四条假设（3.2）：** (a) categorical 的混合仍是 categorical，prior 能拟合 aggregate posterior，而高斯 prior 拟合不了混合高斯，所以更适合一帧到下一帧的多模态转移；(b) 稀疏性利于泛化；(c) ST 估计丢掉了一个缩放梯度的项，反而少了梯度爆炸 / 消失；(d) 对 Atari 的非平滑变化（换房间、物体消失）是更好的归纳偏置。
- **KL balancing（首次）：** 损失式 (2) 是动作条件 HMM 的 ELBO：image / reward / discount 的负对数似然 + β·KL[q ‖ p]，β = 0.1（Atari）、1.0（连续控制）。KL 一边把 prior（transition predictor）训向 posterior，一边把 posterior 正则向 prior；prior 难学，不想让表征被拉向一个差 prior，于是 Alg 2：`kl = α·KL[sg(q) ‖ p] + (1−α)·KL[q ‖ sg(p)]`，α = 0.8——同一个 KL 对 prior 和 posterior 用两个学习率，让模型靠改进 prior 而不是靠抬高 posterior 熵来降 KL。这就是 Dreamer-v3 笔记里 L_dyn / L_rep 两项的雏形。
- **与 free nats 的关系：** v2 没有 free bits——App C 明说 KL balancing 是"instead of using free nats"，即 PlaNet / DreamerV1 的 free nats 被拿掉了；v3 又把 free bits（1 nat）加回来，并把 α / (1−α) 拆成独立权重 β_dyn / β_rep。
- **网络与参数量：** 84×84 灰度降到 64×64 以复用 DreamerV1 的 CNN；ELU；GRU 600 单元；WM 20M 参数（2.1），actor / critic 各 1M（2.2），合计 22M（Table 3；App C 说从 13M 加到 22M 有帮助）。
- **reward / discount 头：** reward 先做 tanh 变换（Table D.1）再用单位方差高斯似然（v3 才换成 symlog + twohot）；discount 头是 Bernoulli，episode 内 γ = 0.999、终止步为 0（2.1；Table D.1 又写 γ = 0.995，原文不一致）。
- **数据管线：** replay FIFO 2M 步，batch 50 条 × 长 50 步，起点在 episode 内均匀采样再裁剪以保证见到足够多的结束步；每 4 个 policy 步一次梯度更新，WM 学习率 2e-4；不用 frame stacking，历史由 RSSM 自己累积（3）。
- **动作怎么进出：** 上一步动作 a_{t−1} 直接是 GRU 输入（低层 Atari 离散动作，full action space）。Behavior learning（2.2，Fig 3）：从 WM 训练时算出的 posterior 状态出发，actor 在 latent 里想象 H = 15 步（imagination MDP），reward 头的均值当奖励、discount 头做折扣并给 actor / critic 损失加权；WM 在这一阶段冻结，actor / critic 梯度不回传给表征；一个 batch 并行想象 2500 条 latent 轨迹。
- **Critic：** 回归 λ-return（式 4–5，λ = 0.95，偏向长 horizon 目标），平方损失，目标网络每 100 步同步一次（v3 改成 EMA 自正则）。
- **Actor 梯度（首次 Reinforce）：** 式 (6) = ρ·Reinforce（baseline 为 critic 值 v(ẑ_t)）+ (1−ρ)·dynamics backprop（用 ST 穿过离散动作和离散状态回传价值梯度，即 DreamerV1 的做法）+ η·熵正则。Atari：ρ = 1、η = 1e-3，Reinforce 远好于 backprop；连续控制：ρ = 0、η = 1e-4，反过来（2.2）。探索靠熵正则，不再像 v1 那样在采集时加外部动作噪声（App C）。actor 对离散动作输出 categorical，连续任务改 truncated normal（App A）。
- **v2 → v3 改了什么（对照 Dreamer-v3 笔记）：** free bits 加回；α = 0.8 变成 β_dyn / β_rep = 1 / 0.1；categorical 混入 1% uniform（unimix）；tanh reward + 高斯似然换成 symlog + twohot；critic 目标网络换成 EMA 自正则 + replay 辅助损失；Reinforce 从"仅 Atari"推广到所有领域并配回报归一化；β、ρ、η 按领域切换被一套固定超参取代；GRU 改成 block-diagonal（8 块）；layer norm 在 v2 里试过但 App C 说收益不明显，v3 是否默认启用以代码为准（未核实）。
- **训练数据：** 纯在线 RL，单环境实例，200M 帧（action repeat 4 后 50M 个 agent 步），期间想象了 468B 个 latent 状态，是真实输入的 10,000×（3）。
- **推理开销：** 部署时 actor 是 4 层 400 单元 MLP（Table D.1）直接出动作，不做 lookahead planning；训练单张 V100 不到 10 天到 200M 帧（3）。


## 3. 实验
Setting：Atari 55 游戏（作者挑各实验室共同使用的集合），Machado 2018 协议——sticky actions、200M 帧、action repeat 4、每局 108k 步上限、full action space、不用生命信息；每游戏单独训一个 agent，主结果 5 seeds（Fig F.1），消融 2 seeds（Fig G.1）；IQN / Rainbow / C51 / DQN 分数取自 Dopamine 的 sticky-actions 版本（与原论文的确定性 Atari 结果不同）。
- 四种汇总指标（3.1）：gamer median 对近半游戏得零也不敏感；gamer mean 被 gamer 打得差的 Crazy Climber、James Bond、Video Pinball 主导；record mean 仍被轻松超纪录的游戏主导；clipped record mean 截断到 1，作者推荐用它。
- Table 1（200M 帧，gamer median / gamer mean / record mean / clipped record mean）：DreamerV2 2.15 / 11.33 / 0.44 / 0.28；DreamerV2 + schedules（熵系数与 ρ 退火）2.64 / 10.45 / 0.43 / 0.28；IQN 1.29 / 8.85 / 0.21 / 0.21；Rainbow 1.47 / 9.12 / 0.17 / 0.17；C51 1.09 / 7.70 / 0.15 / 0.15；DQN 0.65 / 2.84 / 0.12 / 0.12。四种汇总全赢；Rainbow 与 IQN 谁强取决于指标。
- Table K.1 / Fig E.1（单游戏）：赢最多的 James Bond 40445 vs Rainbow 1097 / IQN 3166、Up N Down 653662 vs 34888 / 59944、Assault 23625 vs 3229 / 4885；输的 Video Pinball 41860 vs 466895 / 415833（作者解释：球只占一个像素，重建损失学不到它）、Montezuma 81 vs 500 / 500、Venture 2 vs 1529 / 1313、Star Gunner 7800 vs 57909 / 80003、Private Eye 2198 vs 21334 / 4181、Hero 21868 vs 46675 / 36058。55 个游戏里 2 个没有世界纪录，作者填了"合理值"。
- Table 3（概念对比）：DreamerV2 建模 reward + 图像 + latent 转移、单 GPU、22M 参数、200M 帧、10 加速器天；SimPLe 74M / 4M 帧 / 40 天、无 latent 转移；MuZero 40M / 20B 帧 / 80 天、不建模图像、非单 GPU；MuZero Reanalyze 200M 帧 / 80 天。
- 最有信息量的消融（Table 2、Fig 5，用"稍早版本"的 agent，clipped record mean）：完整 0.25；No Layer Norm 0.25；No Reward Gradients 0.24；No Discrete Latents（换回高斯）0.19；No KL Balancing 0.16；No Policy Reinforce（只用 ST）0.15；No Image Gradients 0.01。
- 同一组消融按游戏计胜 / 负 / 平（3.2；平 = 差距 5% 内）：categorical vs 高斯 42 / 8 / 5（Fig G.1）；KL balancing vs 标准 KL 44 / 6 / 5；停 image 梯度 3 / 51 / 1——表征几乎全靠图像；停 reward 梯度 15 / 22 / 18（Fig H.1，差距很小且部分游戏反而更好，作者据此说不绑 reward 的表征可能泛化更好，与 MuZero 只学 reward / value 的路线相反）；只用 Reinforce vs 混合 18 / 24 / 13；只用 ST 5 / 44 / 6（Fig I.1，混合只在 Seaquest、James Bond 上明显更好）。Fig J.1 还对比了"用均匀随机策略采数据"，用来标出哪些游戏真正需要探索。
- App A（DMC Humanoid Walk，21 维连续动作，纯像素）：ρ = 0、η = 1e-5、β = 2，4e7 步内回报升到约 800（Fig A.2，估读）；作者称是首个只用像素解 humanoid 的公开结果。
- App B（Montezuma）：把 γ 改成 0.99 后 200M 帧约 2500 分（Fig B.2，估读），与 Rainbow + ICM 的约 2800（估读）相当，远超 Rainbow / IQN 的 500；默认超参只有 81（Table K.1）。
- App C：帮了的改动——categorical、KL balancing、Reinforce、模型 13M → 22M、熵正则替代动作噪声；没帮的——binary latent、long-term entropy、混合 actor 梯度、各种 schedule、GRU layer norm。完整消融每项要 55 任务 × 5 seeds × 10 天 > 60,000 GPU 小时，所以只做了主消融。

## 4. 局限
- 作者承认：categorical 为什么好只有四条假设；Video Pinball 这类关键物体极小的游戏重建损失失效；Montezuma 要单独调 γ；算力不够做完整消融，且消融用的是稍早版本、只 2 seeds；MuZero 式 planning 是互补的但没做；Table 3 承认 MCTS 难并行。
- 我读出来的：(1) 仍是 task-specific 在线 RL——WM 带 reward / discount 头、每个游戏从零、55 个 agent 各占 10 天 V100，真机上意味着大量试错；(2) 64×64 灰度 + 20M 参数的重建式表征对小物体 / 细节不敏感，真机彩色多物体场景会更严重；(3) β 在 Atari 0.1、连续控制 1.0、Humanoid 2，ρ / η 也按领域切换，"一套超参跨领域"要等 Dreamer-v3；(4) 想象只有 15 步、单环境实例、replay 2M 步，稀疏奖励与硬探索仍弱（Montezuma 81、Venture 2、Pitfall 0）；(5) 只有单任务 Atari 加一个 Humanoid，无真机、无多任务、无语言 / 目标条件，WM 对外没有任何接口；(6) Table 2 的完整版 0.25 与 Table 1 的 0.28 不是同一版本代码，消融结论只能看相对大小。

## 5. 复现要点
- 开源：项目页 danijar.com/dreamerv2 提供源码 + 训练曲线 JSON（论文脚注）；GitHub danijar/dreamerv2 据我所知是 TensorFlow 2、MIT（未核实）；没有预训练权重；PyTorch 社区移植据我所知存在，未核实。
- 算力：单张 V100、单环境实例、不到 10 天跑完 200M 帧（3）；55 游戏 × 5 seeds ≈ 275 个"10 天单卡" run。8×H100 上：模型只有 22M，每张卡可并行塞多个 run，单 run 受串行环境交互限制、换 H100 也只是几倍提速（我的估计）；整套 Atari 主结果要数周，只复现几个游戏加 Humanoid 一两天足够。
- 超参（Table D.1）：32×32 categorical、GRU 600、β = 0.1、α = 0.8、H = 15、λ = 0.95、ρ = 1、η = 1e-3、actor 学习率 4e-5、critic 1e-4、目标网络 100 步、梯度裁剪 100、Adam ε = 1e-5、weight decay 1e-6、每 4 个 policy 步训一次。作者建议新任务搜 β ∈ {0.1, 0.3, 1, 3}、η ∈ {3e-5, 1e-4, 3e-4, 1e-3}、γ ∈ {0.99, 0.999}；要更省数据就提高更新频率。
- 实现坑：ST 梯度按 Alg 1 写（sample + probs − sg(probs)），KL balancing 按 Alg 2 两次 stop_grad；不要 frame stacking；连续控制必须切 ρ = 0 并调大 β（Humanoid 用 β = 2、η = 1e-5）；γ 正文 0.999、表 0.995 不一致，Montezuma 要 0.99。
- 评测坑：Atari 要用 sticky actions + full action space + 108k 步上限才能和 Dopamine 分数对齐；汇总指标要在按 seed 平均之前算（Table K.1 说明）；消融至少 2 seeds、主结果 5 seeds；Table 1 和 Table 2 不是同一版代码。

## 6. 关键引用链
- 建立在：World-Models（VAE + RNN、在梦里训 controller 的原型）；PlaNet（RSSM、在 latent 里预测、高斯 latent + free nats）；Dreamer-v1（想象里 actor-critic、dynamics backprop、λ-return——v2 换掉了它的高斯 latent、重参数化梯度、free nats 和外部动作噪声）；straight-through 估计来自 Bengio 2013，Reinforce 来自 Williams 1992；对照对象 SimPLe（像素空间视频预测 + PPO）和 MuZero（arXiv:1911.08265）（只学 value-equivalent 模型 + MCTS，Table 3）。
- 后续：Dreamer-v3 直接继承 categorical latent + KL balancing，加回 free bits、加 unimix / symlog / twohot / 回报归一化，把 v2 按领域切换的 β、ρ、η 收成一套固定超参；DayDreamer 把同款 RSSM 搬到真机；TD-MPC2（arXiv:2310.16828）、dino-wm 把 Dreamer 系当 baseline（DINO-WM Table 1 里离线无 reward 的 DreamerV3 在 PushT 只有 0.30）；离散 latent 的思路在后来的 token 化 WM（IRIS 的 VQ token、Genie / arXiv:2410.11758 的 VQ latent action）里以 VQ-VAE 形式再现，属平行发展而非直接继承（我的判断）。
