# Mastering Diverse Domains through World Models（DreamerV3）

> 对应「可lip」第 27 期视频（世界模型系列第一期）。论文：[arXiv:2301.04104](https://arxiv.org/abs/2301.04104)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2301.04104（v1 2023-01-10；后续修订版即 Nature 2025 正式版《Mastering diverse control tasks through world models》，本笔记依据的是带 Methods/Table 2 的新版文本，具体版本号未核实）· 机构: Google DeepMind、University of Toronto（按作者标注的两个机构推断，文本中未显示机构名，未核实）· 代码/权重: https://github.com/danijar/dreamerv3（JAX，MIT；无预训练权重，每个任务从零 RL）· 项目页: https://danijar.com/dreamerv3（未核实）
- 一句话: 用一套固定超参的 model-based RL（RSSM latent world model + 在想象轨迹里训练 actor-critic + 一组 robustness 技巧），在 8 个领域 150+ 任务上超过各领域专用算法，并首次不用人类数据从零在 Minecraft 挖到钻石；它是 state-space latent world model 这条线的经典参考实现，DINO-WM、V-JEPA 2-AC 都把它当 baseline 或出发点。

## 1. 要解决的问题
- RL 换领域就要重新调参：PPO 通用但弱，MuZero / Rainbow / DrQ-v2 / SAC 等专用算法只在各自领域强（Introduction）。
- world model 路线（DreamerV1 只做连续控制，V2 只做 Atari）虽然直观，但"稳健地学习并利用 world model"一直是开放问题：不同领域的 reward 尺度、观测尺度、视觉复杂度差异巨大，KL 权重等超参必须逐领域调。
- 目标：单一配置跨领域工作，且性能随模型大小 / 算力可预测地提升；用 Minecraft Diamond（稀疏奖励、开放世界、长时程）做压力测试。

## 2. 方法
**预测空间：state-space（latent）。** world model 是 RSSM（Recurrent State-Space Model：确定性循环状态 h_t + 随机离散 latent z_t 的序列模型），式 (1)：
- sequence model h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})（block-diagonal GRU，8 块）；encoder z_t ~ q_φ(z_t | h_t, x_t)；dynamics predictor ẑ_t ~ p_φ(ẑ_t | h_t)；reward / continue predictor；decoder x̂_t ~ p_φ(x̂_t | h_t, z_t)。
- z_t 是一组 categorical（softmax 向量，straight-through 梯度），混入 1% uniform 防止 KL 爆炸。decoder 重建像素只是让 latent 保留信息的训练信号，想象和决策时不用。

**Backbone：** 图像用 stride-2 CNN 编解码（Minecraft 观测 64×64×3），向量输入先 symlog 再过 3 层 MLP；模型尺寸 12M–400M（Table 3，隐层 d = 256–1536，默认 200M）。全部从零训练，没有任何预训练视觉先验。

**损失（式 2–3）：** L = β_pred·L_pred + β_dyn·L_dyn + β_rep·L_rep，β = 1 / 1 / 0.1。L_dyn = max(1, KL[sg(q)‖p])，L_rep = max(1, KL[q‖sg(p)])——即 KL balancing（两个方向不同权重）+ free bits（KL 低于 1 nat 时不再优化）；这一组合让 KL 权重不必按领域调。

**动作怎么进出：** low-level 环境动作 a_{t-1} 直接作为 GRU 输入；actor π_θ(a_t | s_t)、critic v_ψ(R_t | s_t) 作用在 model state s_t = {h_t, z_t} 上，只用 world model 想象出的轨迹训练（imagination horizon 15，Table 4；λ-return，γ = 0.997，λ = 0.95），不依赖真实环境梯度；部署时 actor 直接出动作，不做 lookahead planning。

**robustness 技巧：** symlog(x) = sign(x)·ln(|x|+1) 及其逆 symexp（式 9）压缩任意量级的回归目标；reward / critic 用 symexp twohot（在指数间隔 bins 上做 softmax 分类，式 10–11），梯度大小与目标数值解耦；actor 用 Reinforce，回报按 5–95 分位数范围 S 归一化且分母下限 L = 1（式 6–7），固定熵系数 3e-4；critic 加 EMA 自正则和 replay 轨迹上的辅助损失（β_repval = 0.3）。

**训练数据：** 在线 RL，全部来自 agent 自己的交互（replay 5M，batch 16×64，replay ratio 32–1024，Table 2 / 4），不用任何离线数据集或视频。

**推理频率与延迟：** actor 是 3 层 MLP，成本极低；wall-clock 延迟论文未说明（Minecraft 环境为 20 Hz 控制）。

**怎么用于控制：** Dyna 式——world model 只作为训练 policy 的"想象环境"，同时预测 reward 和 continue（因此它是 task-specific 的）。


## 3. 实验
Setting：8 个领域 150+ 任务、固定超参、每个 agent 单张 A100（Results 节、Table 2）；baseline 是各领域的 tuned expert 加一个高质量 PPO 实现。
- Atari 200M 帧：gamer-median 830% vs MuZero 693%、PPO 180%（Table 6）。
- ProcGen 50M：normalized mean 66.01 vs PPG 64.89、PPO 42.80（Table 7）。
- DMLab 100M：human-mean-capped 71.4% vs IMPALA@1B 66.3%、IMPALA@100M 31.0%（Table 8）。
- Atari100k：gamer mean 125% vs IRIS 105%、TWM 96%、SPR 62%，仅 0.1 GPU-day（Table 9 / 10）。
- Proprio Control 500K：任务均分 871 vs D4PG 792（Table 11）；Visual Control 1M：861 vs DrQ-v2 770、CURL 479（Table 12）。
- BSuite：66% vs Boot DQN 60%，scale 类最强（Table 13）。
- Minecraft Diamond：10 个 seed 全部在 100M 步内挖到钻石，所有 baseline 为 0%（Fig 5）；但按 episode 算，钻石只出现在 0.4% 的 episode（Fig 9 说明）；回报 9.1 vs IMPALA 7.1、Rainbow 6.3、PPO 5.1（Table 5）；1 张 GPU 跑 9 天，对比 VPT 720 GPU × 9 天且用人类数据（Previous work 节）。
- 消融（Fig 6）：所有 robustness 技巧都有贡献，最重要的是 KL 目标（balance + free bits），其次是 return normalization 和 symexp twohot；**性能主要依赖无监督重建损失而非 reward / value 梯度**（Fig 6b）——这是后续"先用无监督数据学 WM"路线的依据；12M→400M 单调提升且更省交互（Fig 6c）。

## 4. 局限
- 作者承认：Deep Sea 等硬探索任务得 0（Table 13 exploration 类 0.01）；Minecraft 钻石的 episode 成功率仅 0.4%；结论里点名未来要"从互联网视频学世界知识"和"跨领域共享一个 WM"。
- 我读出来的：(1) 纯在线 RL，每个任务从零交互，真机上意味着大量试错；(2) 64×64 低分辨率 + 从零训的 CNN，没有预训练视觉表征，latent 又被 reward / continue 头绑定为 task-specific——DINO-WM 把它当离线 WM（去掉 reward）在 PushT 上只有 0.30 成功率（DINO-WM Table 1）；(3) 像素重建迫使模型建模无关细节；(4) 想象 horizon 15 步，长程规划靠 critic bootstrapping；(5) 无语言 / 目标条件，单任务单 agent；(6) 确定性 GRU + 离散 latent 对高度随机的真实场景表达力有限。

## 5. 复现要点
- 开源：danijar/dreamerv3（JAX，MIT，README 称能复现 Nature 版结果）；PyTorch 移植 NM512/dreamerv3-torch（DINO-WM 用它做 baseline）。没有预训练权重。
- 模型：12M–400M，默认 200M；两个 Control 套件用 12M 即可（Table 2）。
- 算力：每个 run 单张 A100，GPU-days（Table 2）：Minecraft 8.9、ProcGen 16.1（单 seed）、Atari 7.7、DMLab 2.9、Atari100k 0.1、Proprio 0.3、Visual Control 0.1。
- 8×H100：绰绰有余——每张卡跑一个 seed / 任务；瓶颈反而是环境 CPU（默认 16 个 env 实例，Minecraft 用 64 个远程 CPU worker）和 replay ratio。
- 坑：JAX 版本兼容；Minecraft 依赖 MineRL v0.4.4 + Java；replay ratio 决定算力 / 数据效率折中；探索弱（Freeway、Montezuma、Deep Sea）；结果需 5–10 seeds。

## 6. 关键引用链
- 建立在：World Models（Ha & Schmidhuber 2018）、PlaNet / RSSM（Hafner 2019）、DreamerV1（2020）、DreamerV2（2021，离散 latent + KL balancing）；twohot 回归来自 MuZero，free bits 来自 VAE 文献。
- 后续：DayDreamer（Wu et al. 2022）把 Dreamer 系列用于真机；DINO-WM（2024）、TD-MPC2 把它当离线 / 在线 baseline；V-JEPA 2 在 related work 中把它归为"仿真里的任务特定 WM"；Dreamer 4（Hafner et al. 2025）改用 tokenizer + transformer 动态模型并从离线视频学（细节未核实）。
