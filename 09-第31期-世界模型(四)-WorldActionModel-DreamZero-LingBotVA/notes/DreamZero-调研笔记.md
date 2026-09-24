# World Action Models are Zero-shot Policies (DreamZero)

> 对应「可lip」第 31 期视频（世界模型系列第四期）。论文：[arXiv:2602.15922](https://arxiv.org/abs/2602.15922)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2602.15922（v1，2026-02-17） · 机构: NVIDIA（GEAR 团队；项目负责人 Seonghyeon Ye、Yuke Zhu、Linxi "Jim" Fan、Joel Jang） · 代码/权重: https://github.com/dreamzero0/dreamzero（模型权重、推理代码、RoboArena / PolaRiS / Genie Sim 3.0 评测代码；AgiBot 数据集论文称"将在后续发布中开源"，训练代码是否开源未核实） · 项目页: https://dreamzero0.github.io
- 一句话: 把 14B 的 Wan2.1 image-to-video 扩散模型改成自回归、联合去噪"未来视频 + 动作块"的 World Action Model（WAM），在真实机器人上零样本泛化到未见任务 / 环境，任务进度比 SOTA VLA 高 2 倍以上，并用一整套系统优化把 5.7s/chunk 的推理压到 150ms、实现 7Hz 闭环控制；是 2026 年"WAM 可以直接当 policy"这一主张的代表作。

## 1. 要解决的问题

- VLA（从 VLM 初始化）继承的是语义先验：知道"把可乐罐放到 Taylor Swift 那里"里的 Taylor Swift 是什么，但对"动作该怎么执行"（几何、动力学、接触）没有先验；所以对新物体泛化好，对**新动作 / 新技能**和**新环境**泛化差，必须靠大量任务特定、环境特定的重复示教（Section 1、2.1）。
- 已有的"视频模型 → 策略"做法（UniPi / AVDC 用 IDM 或光流；VPP / Mimic-Video 取特征；Cosmos Policy / GR-2 等 joint 模型）大多仍在重复示教数据上验证，要么两阶段误差累积，要么没解决实时性。
- DreamZero 想验证三件事：(1) 联合视频-动作建模能不能从"异构、非重复"的真实机器人数据里有效学习；(2) 能不能零样本泛化到训练里没有的动作；(3) 能不能把 14B 视频扩散模型做成实时闭环策略。

## 2. 方法

**预测空间与 backbone**：像素空间（经 Wan VAE 编码的 video latent）。backbone 是 Wan2.1-I2V-14B-480P（14B DiT，image-to-video）。多视角图像直接拼成一帧输入，不改 backbone（Section 3.1）。

**输入**：当前和历史观测 $o_{0:l}$（VAE）、语言指令 $c$（文本编码器）、本体状态 $q_l$（新增 state encoder）。新增参数只有 state encoder、action encoder、action decoder；text encoder、image encoder、VAE 冻结，所有 DiT block 全参微调（LoRA 效果差，脚注 7）。

**动作怎么进出——joint 联合去噪**。目标分布为
$$\pi_0(o_{l:l+H},a_{l:l+H}\mid o_{0:l},c,q_l)=\underbrace{\pi_0(o_{l:l+H}\mid o_{0:l},c,q_l)}_{\text{video prediction}}\;\underbrace{\pi_0(a_{l:l+H}\mid o_{0:l+H},q_l)}_{\text{IDM}}\quad(\text{Eq. 1})$$
即"联合预测 = 自回归视频预测 × inverse dynamics"，但不用两个模型，而是一个模型端到端联合学。训练用 flow matching：视频 latent 和归一化动作各自与高斯噪声线性插值 $z_t=t z_1+(1-t)z_0$、$a_t=t a_1+(1-t)a_0$（Eq. 2），网络 $u_\theta$ 预测两个模态拼在一起的速度 $v=[z_1,a_1]-[z_0,a_0]$（Eq. 3）。**视频和动作共享同一个去噪 timestep**（与 UWM、Cosmos Policy 不同），作者说这样训练初期收敛更快。

**自回归 + teacher forcing（chunk-wise）**：视频按 chunk 生成，每 chunk $K=2$ 个 latent 帧，默认 $M=4$ 个 chunk；训练时当前 chunk 加噪、之前 chunk 用干净真值做上下文（attention mask 见 Fig. 14）。只有视频是自回归的，动作不自回归以免闭环误差传播。AgiBot 数据视频 5 FPS、动作 30Hz、动作 horizon $H=48$，即每 chunk 覆盖 1.6s；DROID 为 5 FPS / 15Hz / $H=24$，同样 1.6s。最大上下文 8 个 latent 帧 = 33 原始帧 = 6.6s（Appendix C）。选自回归而非双向的理由（Appendix B）：双向模型要把整段视频下采样到固定长度以对齐语言描述，破坏原生帧率、损害视频-动作对齐；自回归靠 KV cache 保留原生帧率并可利用视觉历史。

**推理与闭环**（Algorithm 2）：联合去噪出一个 chunk 的视频 + 动作块，动作经 2× 上采样 + Savitzky-Golay 滤波后异步执行；执行完后**用真实观测替换 KV cache 里的预测帧**（丢弃预测视频 latent），消除自回归视频生成的误差累积——作者称这是 WAM 独有的优势。

**实时化（Section 3.2，Table 1）**：单 GPU 朴素实现 5.7s/chunk（16 步去噪、14B、串行执行）。异步执行把约束从"推理完才动"变成"推理要在当前 chunk（1.6s）用完前完成"，目标延迟 <200ms。优化：CFG 并行（条件 / 无条件分两块 GPU，单步延迟 −47%）、DiT caching（相邻速度余弦相似度高就复用，16→4 步）、torch.compile + CUDA Graphs、NVFP4 量化（Blackwell）、cuDNN attention、调度器搬到 GPU；H100 上累计 9.6×，GB200 上 16.6×。再加 **DreamZero-Flash**：训练时把视频 timestep 偏向高噪声（$t^{video}=1-\eta,\ \eta\sim\mathrm{Beta}(7,1)$，$\mathbb{E}[t^{video}]=0.125$），动作 timestep 仍均匀（Eq. 5），让模型学会"从还很噪的视频上下文直接预测干净动作"，从而 1 步去噪即可：350ms→150ms，总加速 38×，约 7Hz（Section 6 说明用的是 2 块 GB200）。

**训练数据与算力**：AgiBot G1 约 500 小时、22 个真实环境、7.2K 段，平均每段 4.4 分钟、约 42 个子任务，采集刻意"求多样不求重复"（Fig. 6、Appendix E）；Franka 用 DROID。各自 100K 步、全局 batch 128（Section 4.1）；两种机器人分开预训练。GPU 数量与训练时长论文未说明。

**怎么用于控制**：直接作为闭环 policy（自带的视频预测相当于隐式视觉规划）。


## 3. 实验

setting：默认在**未见环境 + 未见物体**下评测（采集与评测地点不同）；baseline 为 GR00T N1.6 与 π0.5，各有 from-scratch（只用 VLM 权重）和 from-pretrained（官方跨本体预训练权重）两种初始化，再在与 DreamZero 相同的数据、相同 batch / 步数下训练（Section 4）。
- **seen task（10 任务，80 次 rollout）**：AgiBot 上 DreamZero 平均任务进度 62.2%，最好的预训练 VLA 27.4%，from-scratch VLA 接近 0（Fig. 8，Section 5.1 Q1）。
- **unseen task（10 个训练中没有的任务，如解鞋带、熨衣服、握手）**：DreamZero 39.5% vs 预训练 VLA 16.3%，from-scratch <1%；单项如 Remove Hat 85.7%、Shake Hands 59.2%（Fig. 9，Q2）。DROID-Franka 未见动词 20 任务：DreamZero 49% 进度 / 22.5% 成功率，GR00T N1.6 31% / 12.5%，π0.5 33% / 7.5%（Q2）。
- **post-training（叠衣 33h、装水果 12h、收桌 40h）**：匹配或超过 VLA，装水果显著更好；引言称平均任务进度高出 SOTA VLA 10%（Fig. 10，Q3）。
- **跨本体视频迁移（只用视频预测 loss，无动作）**：9 个未见任务上 38.3% → 54.3%（12 分钟人类第一视角视频）/ 55.4%（20 分钟 YAM 机器人视频），从 AgiBot checkpoint 以 1:1 混合继续训 10K 步（Table 2，Q4）。
- **few-shot 换本体**：AgiBot 预训练模型用 55 条轨迹 / 11 任务 / 约 30 分钟 YAM play data 后训练，保留语言跟随和新物体泛化（Fig. 12，Q5；无数字）。
- **Flash**：收桌任务上 4 步 83% → 1 步 52%（原模型）→ 1 步 74%（Flash），推理 350ms→150ms（Table 3）。
- **消融（Table 4，50K 步、batch 32、PnP-Easy）**：500h 多样数据 50% vs 500h 重复数据（70 任务）33%；14B 50% vs 5B 21%（小模型视觉幻觉传给动作）；正文称把 VLA 放大到 8B / 32B VLM 初始化仍是 0% 进度（我读到的 Table 4 文本版显示为 50%±0.0%，疑为转换错误，以正文 Q2 为准，未核实）；自回归 vs 双向进度都是 50%，但 AR 动作更平滑、推理快 3–4×。
- 失败分析（Appendix H，Fig. 16）：多数失败来自视频预测错误，机器人忠实执行了错误的视觉计划——"提升视频 backbone 就能提升策略"。

## 4. 局限

- 作者承认（Section 6）：没有 WAM 的 scaling law；人类视频只试了 12 分钟；7Hz 需要 2 块 GB200，远贵于消费级 GPU 上 20Hz+ 的 VLA；只是 System 1，视觉记忆仅 6.6s，长程任务需要 System 2 规划器或更长上下文；亚厘米精度任务（插孔、装配）受限于 behavior cloning；高自由度本体需要更多 play data 学隐式 IDM。
- 我读出来的：(1) 两种本体分开预训练，"跨本体"结论建立在两个都是双臂平行夹爪的相似本体上；(2) 所有真实评测是任务进度分而非成功率（DROID 上 22.5% 成功率并不高），80 次 rollout 的方差大；(3) 失败主要是幻觉 / 语言跟随错误，但模型没有任何机制检测自己的视频是错的——执行完全信任想象；(4) DiT caching 和 NVFP4 量化不是数学等价的，作者说 minimal degradation 但没给数字；(5) 训练算力未披露，14B 全参微调对学术组不友好。

## 5. 复现要点

- 开源：GitHub dreamzero0/dreamzero 提供权重、推理代码、RoboArena（真实 DROID 评测）/ PolaRiS（DROID-sim）/ Genie Sim 3.0 评测代码；DROID checkpoint 可复现 Franka 结果；AgiBot 500h 数据集"计划开源"（是否已发布未核实）；训练代码未核实。
- 模型：14B DiT + Wan VAE + 文本编码器（Wan2.1 配置）。推理朴素实现单 GPU 5.7s/chunk；官方 7Hz 需要 2 块 GB200（CFG 并行 + NVFP4）。H100 上量化和 Flash 两行标"—"（Table 1），即 H100 上只验证到 9.6×，约 0.6s/chunk 量级（按 5.7s ÷ 9.6 推算，未核实）——对 1.6s 的 chunk 仍可异步跑，但反应性差。
- 8×H100 可行性：推理没问题（2 卡即可）；训练 14B 全参微调、batch 128、100K 步，显存需要 FSDP / 序列并行，8×H100 能跑但时长很可能以周计（论文未给训练算力，纯估计）。LoRA 作者试过效果差。
- 坑：动作用相对关节位置，过滤 idle 动作；多视角拼成一帧；torch.compile 需要静态 shape、第一条轨迹会多次重编译；需要 PyTorch ≥2.9 才有 cuDNN SDPA；Savitzky-Golay 滤波（窗 21、阶 3）是执行平滑的必要步骤。

## 6. 关键引用链

- 建立在：Wan2.1（backbone）、flow matching / rectified flow、自回归视频扩散（MAGI-1、Self-Forcing、CausVid、Pyramidal Flow 的 teacher forcing）、UniPi / AVDC（视频 → IDM 的两阶段思路，被它端到端化）、GR-1 / GR-2、UWM、UVA、Cosmos Policy、Genie Envisioner、VPP、Mimic-Video（前代 WAM）、DreamGen（同团队，视频模型做数据生成）。
- 后续 / 同期：Cosmos Policy（2601.16163，NVIDIA 另一条 joint 路线，共享作者）、LingBot-VA（2601.21998，自回归 diffusion + MoT，思路相近）；tutorial 2607.00836 与 roadmap 2607.11689 均把它列为 joint 范式代表；deepxiv 显示已有 46 次引用（截至 2026-09，具体后续未核实）。
