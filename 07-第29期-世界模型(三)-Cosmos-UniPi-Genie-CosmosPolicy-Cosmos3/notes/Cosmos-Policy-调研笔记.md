# Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

> 对应「可lip」第 29 期视频（世界模型系列第三期）。论文：[arXiv:2601.16163](https://arxiv.org/abs/2601.16163)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2601.16163（v1 2026-01-22；项目页标注 ICLR 2026） · 机构: NVIDIA、Stanford University · 代码/权重: https://github.com/NVlabs/cosmos-policy（Apache-2.0；权重与训练数据在 Hugging Face collection nvidia/cosmos-policy） · 项目页: https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/
- 一句话: 不改架构、单阶段 post-training，把 Cosmos-Predict2-2B 视频扩散模型直接变成机器人策略：动作块、未来观测、value 全部编码成"latent frame"塞进同一个 latent diffusion 序列联合去噪。一个模型同时是 policy、world model、value function，可做 best-of-N 的 model-based planning，并能从 rollout 经验里继续学；LIBERO 98.5%、RoboCasa 67.1%（只用 50 演示/任务）、真实 ALOHA 双臂平均分 93.6 超过 π0.5。

## 1. 要解决的问题
VLA 的 backbone 是在静态图文对上预训练的 VLM，缺少时间因果和隐式物理；视频生成模型有这些先验，但已有"视频模型做策略"的工作要么多阶段（先微调视频再训动作模块）并加新部件（独立 action diffuser / inverse dynamics model），要么（UVA、UWM）是自定义联合模型无法利用预训练视频模型。此外，policy、world model、value function 在前作（Dreamer、TD-MPC、SAILOR、Latent Policy Steering、FLARE）中是分开的模块且多从头训练。作者想要：一个统一模型，零架构改动，靠视频模型原本的去噪学习机制去建模多模态动作分布，并顺带得到可用于规划的 world model 与 value。

## 2. 方法
**backbone 与预备知识（Sec 3）**：Cosmos-Predict2-2B-Video2World，latent video diffusion：DiT 去噪器；Wan2.1 时空 VAE 把 $(1+T)\times H\times W\times 3$ 压成 $(1+T/4)\times H/8\times W/8\times 16$（首帧不做时间压缩）；T5-XXL 文本经 cross-attention 注入；EDM 目标 $\mathcal{L}=\mathbb{E}\|D_\theta(x_0+n;\sigma,c)-x_0\|_2^2$，训练时首帧（条件图）保持干净、其余帧加噪。任务建模为稀疏奖励有限时域 MDP，value 用 Monte Carlo 回报 $V(s')=\gamma^{H-t}R(s_H,a_H)$（不做 TD）。

**预测空间**：latent（Wan2.1 VAE latent）。图像模态经 VAE 编解码；动作/本体状态/value 直接写进 latent 帧，读出时不需要 VAE 解码。

**核心机制：latent frame injection（Sec 4.1，App A.1，Fig 2/8）**。以 ALOHA（两个第三人称相机 + 一个腕部相机）为例，序列共 11 个 latent 帧：(1) 空白占位（Wan VAE 首帧单独编码的实现细节）、(2) 当前本体状态 $s$、(3–5) 当前三路图像、(6) 动作块 $a$、(7) 未来本体状态、(8–10) 未来三路图像 $s'$、(11) 未来状态 value $V(s')$。非图像模态的编码：归一化到 $[-1,+1]$，展平（动作块 $K\times d_{act}$），复制 $(H'\times W'\times C')/(K\times d_{act})$ 次填满一个 latent 体；解码时对所有副本取平均再反归一化。多视角图像直接作为额外"帧"插进图像序列。只用当前时刻观测和 $t+K$ 时刻观测（$K$ 为动作块长度），无历史，也不预测中间帧。顺序 $(s,a,s',V(s'))$ 使得可以从左到右自回归解码。

**联合训练（Sec 4.2，Fig 12）**：每个 batch 按 50/25/25 切分：50% 来自演示数据训 policy $p(a,s',V(s')\mid s)$；25% 来自 rollout 数据训 world model $p(s',V(s')\mid s,a)$；25% 训 value function $p(V(s')\mid s,a,s')$。三者共用同一序列，仅靠"哪些 latent 帧作为干净条件、哪些加噪作为目标"的 mask 区分。初始 rollout 数据集 = 演示集 + 回放失败的演示（LIBERO/RoboCasa 约 10–20% 演示回放失败）。注意 policy 目标带辅助项（同时预测 $s'$ 和 $V$），消融证明这是最关键的设计。

**噪声调度改动（App A.2.1）**：基座的 log-normal $\sigma$ 分布（$P_{mean}=1.39,P_{std}=1.2$）对高噪声区权重不足，导致动作不准；改为 0.7 log-normal + 0.3 U[1,85] 混合。推理时 $\sigma_{min}$ 从 0.002 提高到 4（$\sigma_{max}=80$），因为极低噪声步反而更不准。

**推理（Sec 4.2，App A.3.1）**：直接策略模式并行去噪一次出 $a,s',V$（后两者丢弃）；规划模式自回归：先 $a$，再 $s'$，再 $V$。去噪步数 LIBERO/RoboCasa 5 步、ALOHA 10 步。动作块：LIBERO 16 步全执行；RoboCasa 预测 32 执行 16；ALOHA 25 Hz、50 步（2 s）全执行后再查询。

**从经验中学习 + planning（Sec 4.3）**：收集策略 rollout（含成败/分数），在基座 checkpoint 上继续微调，batch 90% 给 world model 与 value、10% 给 policy，得到"planning model"；**dual deployment**：原 checkpoint 出动作，planning model 做 world model + value（保证 WM/value 在原策略的 on-policy 数据上训练）。best-of-N：采样 $N$ 个动作块 → 每个动作块查 3 次 world model 得 $s'$ → 每个 $s'$ 查 5 次 value → 15 个 value 用"majority mean"聚合（先按阈值投票成败，再在多数派内取均值，抗双峰）→ 执行最高 value 的动作块。通过输入 mask，value 可训成 $V(s')$（mask 掉 $s,a$）或 $Q(s,a)$（mask 掉 $s'$），后者即 model-free 变体。

**训练数据与算力（App A.2）**：全参微调。LIBERO 4 套件各 500 演示（policy 用过滤后的成功演示，WM/value 用全集）：64×H100、batch 1920、40K 步、48 h；RoboCasa 24 任务只用 50 段人类演示/任务：32×H100、batch 800、45K 步、48 h；ALOHA 四任务合计 185 演示：8×H100、batch 200、50K 步、48 h。规划用 rollout：505 段来自各方法评测 + 143 段 ziploc 任务补采 = 648 段。

**推理延迟（App A.4.2，1×H100）**：5 步并行 0.61 s/动作块；10 步 0.95 s（ALOHA 上机器人停 0.95 s 生成 2 s 动作）；1 步 0.16 s（RoboCasa 66.4%，仅 −0.5%）。规划 $N=8$ 在 8×H100 并行需 4.9 s/动作块。

**cascaded 还是 joint**：joint——单一 DiT、单一去噪过程同时产出动作与未来观测，无 IDM、无独立动作头。

**分类体系定位**：WAM 第 3 类 joint video-action modeling（tutorial 亦如此归类）。两点出入：(a) 它预测的"视频"只是 $t+K$ 一帧多视角观测，不是完整视频；(b) 直接策略模式下未来帧被丢弃，形似第 4 类"auxiliary video prediction"，但它仍在同一次去噪中被生成，且消融（Tab 5 第 4 行 44.4%）说明未来状态预测是策略成立的关键，耦合比第 4 类更强；同时内置 value 头，带有 model-based planning 成分。

## 3. 实验
- **LIBERO**（Tab 1，4 套件 × 500 trial × 3 seed）：平均 98.5%（Spatial 98.1 / Object 100.0 / Goal 98.2 / Long 97.6），高于 CogVLA 97.4、OpenVLA-OFT 97.1、π0.5 96.9、UniVLA 95.2、π0 94.2、Video Policy（Long 94.0）、UVA（Long 90.0）。
- **RoboCasa**（Tab 2，24 任务 × 50 trial × 3 seed，评测用未见物体实例，部分场景风格未见）：67.1%，且**只用 50 演示/任务**；对比 GR00T-N1.5 + HAMLET 66.4、FLARE 66.4、Video Policy 66.0、GR00T-N1.5 64.1（均 300 演示）、π0 62.5、UWM 60.8（1000 演示）、GR00T-N1 + DreamGen 57.6（300 + 10,000 合成）。
- **真实 ALOHA**（Fig 4、Tab 3，101 trial/方法，评分 = 任务完成百分比）：平均 93.6，对比 π0.5 88.6、π0 77.9、OpenVLA-OFT+ 62.0、Diffusion Policy 33.6；分项 put X on plate 100.0、fold shirt 99.5、candies in bowl 89.6、candy in ziploc 85.4（π0.5 分别 98.3/99.5/95.2/61.5）。OOD 子集 π0.5 92.5 略高于 Cosmos Policy 89.3。所有方法同算力 8×H100 × 48 h。
- **消融**（Tab 4 LIBERO；Tab 5 RoboCasa）：去辅助目标 98.5→97.0；从头训练 98.5→94.6（ALOHA fold shirt 从头训 80.8，低 18.7 分且动作抖动）。RoboCasa 逐项：去 value 训练样本 66.6；再去 WM 样本 64.0；再去 policy 的 value 辅助目标 62.5；再去未来状态辅助目标 **44.4**——"让策略同时预测未来观测"是最要紧的一项。1 步去噪 66.4。
- **规划**（Fig 7，两个难任务 + 难初始条件）：model-based $V(s')$ 平均比无规划高 12.5 分，优于 model-free $Q(s,a)$；微调后的 world model 能预测"夹爪滑脱"这类失败（Fig 6）。

## 4. 局限
作者承认（Sec 6、App A.4.2）：规划模式约 5 s/动作块，不适合动态任务与 locomotion；有效规划需要大量 rollout（648 段）；只做一层 best-of-N，未做多步深度搜索或更长预测时域。
我读出来的：(1) 不规划时也要 0.61–0.95 s 出一个动作块，块内开环执行 2 s，对扰动无法即时反应；(2) world model 只预测 $t+K$ 单帧，不能多步 rollout，本质是一步 forward dynamics；(3) 无历史输入；(4) 语义泛化未测（LIBERO/RoboCasa 都是训练过的任务，ALOHA OOD 只是位置/物体变化），视频先验能否替代 VLM 的语言/概念泛化存疑；(5) 复制填充写入 latent 是权宜之计，动作维度与 latent 通道不匹配时信息利用率低；(6) value 来自稀疏终端奖励的 Monte Carlo 回报，长时程信用分配弱；(7) 训练算力大（LIBERO 64×H100×48 h）。

## 5. 复现要点
- 开源完整：代码（Apache-2.0）、LIBERO/RoboCasa/ALOHA 权重与训练数据在 HF；README 给出推理显存：LIBERO 6.8 GB、RoboCasa 8.9 GB、ALOHA 6.0 GB，规划模式串行最低 10 GB。
- 模型：2B（Cosmos-Predict2-2B-Video2World 的 DiT；T5-XXL 文本编码器另算），全参微调。对照：Diffusion Policy 约 150M，其他基线 2–7B（App A.2.4）。
- 8×H100 判断：**推理完全可行**（单卡即可；best-of-8 规划用 8 卡并行 4.9 s）；**训练可行**——ALOHA 配置就是 8×H100 × 48 h（batch 200）；LIBERO/RoboCasa 论文用 64/32 卡，8 卡需等比缩 batch 或拉长到 4–8 天（估计，按 GPU 数线性折算）。
- 坑：必须改噪声分布并设 $\sigma_{min}=4$，否则动作不准（App A.2.1）；Wan VAE 首帧单独编码带来的占位帧与"每张图复制四份"细节（Fig 8）；LIBERO/RoboCasa 要用未过滤（含失败）演示训 WM/value；规划前需数百段 on-policy rollout 并做 dual deployment；控制器要能接受 2 s 开环动作块。

## 6. 关键引用链
- 建立在：Cosmos-Predict2（NVIDIA 2025）、Wan2.1 VAE、EDM（Karras et al. 2022）、DiT、action chunking（ACT，Zhao et al. 2023）、Diffusion Policy；直接前作 UVA（Li, Gao, Sadigh, Song 2025，同为 Song 组）、UWM、Video Policy（Liang et al. 2025）；world model + value 思路对照 Dreamer、TD-MPC、FLARE、SAILOR、Latent Policy Steering；比较对象 OpenVLA-OFT（第一作者前作）、π0/π0.5、GR00T-N1.5、DreamGen。
- 后续在其上：tutorial（arXiv 2607.00836）把它列为 joint video-action modeling 的代表；其他后续工作未核实。
