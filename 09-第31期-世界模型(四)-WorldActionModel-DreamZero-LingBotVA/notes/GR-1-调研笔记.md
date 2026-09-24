# Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation（GR-1）

> 对应「可lip」第 31 期视频（世界模型系列第四期）。论文：[arXiv:2312.13139](https://arxiv.org/abs/2312.13139)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2312.13139（v1 2023-12-20 前后，精确日期未核实；v2 2023-12-21；ICLR 2024）· 机构: ByteDance Research（Hongtao Wu、Ya Jing 共同一作，通讯 Tao Kong）· 代码/权重: https://github.com/bytedance/GR-1（论文正文只给项目页；仓库是否含 Ego4D 预训练权重与 CALVIN 微调权重、许可证：未核实）· 项目页: https://GR1-Manipulation.github.io
- 一句话: 一个 GPT 式 causal transformer，先在 Ego4D 的 80 万段人类第一视角视频上做"语言条件的下一帧预测"预训练，再在机器人数据上同时预测动作和未来帧；CALVIN ABCD→D 成功率 88.9→94.9%、零样本新场景 ABC→D 53.3→85.4%。它把"视频生成预训练"从 UniPi 那种独立视频规划器搬进策略网络本身——视频预测变成策略的辅助 / 联合目标而非外挂——是 GR-2、UWM、Fast-WAM 这一支"训练时学视频、推理时只出动作"路线的起点。

## 1. 要解决的问题
- 机器人数据稀疏且多模态（图像 / 状态 / 动作 / 语言），而 NLP / CV 的"生成式预训练 + 微调"范式已被证明有效。
- 作者论点：机器人轨迹本身就是视频，"根据历史帧和语言预测未来帧"与"预测动作"高度相关——能预见接下来会发生什么的模型更容易给出合适的动作。
- 已有预训练要么学视觉表征（R3M、MVP、VC-1，只做 masked / contrastive 目标），要么学 world model 再 RL；VPT / VIPER 用的是任务域内视频；UniPi 等 model-based 方法把视频模型与 IDM 分开。
- GR-1 想用**一个**模型同时做视频预测与动作预测，并让大规模**域外**（非机器人）视频预训练直接迁移到策略。

## 2. 方法
**预测空间：像素**（MAE 风格的 patch 重建，目标 patch 归一化，MSE）；不是 latent、不是 token。
- **问题形式（4.1）**：预训练 π(l, o_{t−h:t}) → o_{t+Δt}；微调 π(l, o_{t−h:t}, s_{t−h:t}) → (o_{t+Δt}, a_t)。s 是末端 6D 位姿 + 夹爪二值；a 是 delta XYZ + delta Euler + 夹爪。
- **输入编码（4.2.1，Fig 2）**：语言用冻结 CLIP text encoder；图像用冻结 MAE 预训练 ViT，CLS token 做全局表征，patch token 经 perceiver resampler 压缩；状态用线性层。
- **token 序列**：预训练 (l, o_t, [OBS])，微调 (l, s_t, o_t, [OBS], [ACT])；语言 token 每步重复以免被其他模态淹没；加可学习的相对时间步嵌入。
- **注意力（4.2.2）**：causal attention，但所有 [OBS] / [ACT] 查询 token 被 mask、不被任何 token 看到——**动作预测不条件于预测出的未来帧**，两者并行从同一上下文读出。
- **输出（4.2.3）**：[OBS] 的输出 + mask token 经一个 transformer decoder 重建未来帧 patch（L_video）；[ACT] 输出经 3 层 MLP 出 arm（Smooth-L1，L_arm）和 gripper（BCE，L_gripper）。
- **损失（4.3）**：预训练只有 L_video；微调 L = L_arm + L_gripper + L_video（式 3），权重全 1。
- **规模（App A.1）**：causal transformer 12 层 12 头 hidden 384；总参数 195M，可训练 46M（CLIP、MAE 冻结）。
- **预训练数据**：Ego4D 裁成 3 s 短片，共 80 万段、800 万帧，帧间隔 1/3 s，Δt = 1；batch 1024、50 epoch（Table 3）。
- **微调数据**：CALVIN 只用带语言标注的 1%（约 2.3 万条），Δt = 3，同时预测静态相机与腕部相机两路，输入序列长 10；batch 512、20 epoch。真机 batch 64、30 epoch。
- **推理**：每步一次前向同时出一帧预测和**一个**动作（无 action chunk）；视频分支推理时可不解码，不影响动作；控制频率 / 延迟论文未给。
- **动作层级**：low-level 末端 delta 位姿；语言条件。


## 3. 实验
- Setting：CALVIN（34 任务、Franka、语言指令；评 1000 条 5 任务链，每任务 360 步内完成算成功，指标是连续完成 1–5 个任务的成功率与 Avg. Len，App A.2）；ABCD→D 与 ABC→D（D 环境未见）两个 split。
- Baseline：MCIL、HULC（用全量含无标注的 play 数据）、RT-1、MT-R3M（同为 Ego4D 预训练的 R3M 编码器 + 同规模 GPT 策略，专门对照"视频生成预训练 vs 表征预训练"）。
- Table 1 ABCD→D：GR-1 0.949 / 0.896 / 0.844 / 0.789 / 0.731，Avg. Len 4.21；HULC 0.889 … 3.06；RT-1 2.45；MT-R3M 2.08。
- Table 1 ABC→D（零样本新场景）：GR-1 0.854 / 0.712 / 0.596 / 0.497 / 0.401，Avg. Len 3.06；RT-1 0.533 / 0.90，MT-R3M 0.529 / 0.93，HULC 0.418 / 0.67。
- Table 1 10% 数据（每任务 66 条、共 2244 条）：GR-1 0.778 / 2.00 vs HULC 0.668 / 1.11。未见语言（GPT-4 生成 50 条同义指令 / 任务，Table 6）：GR-1 0.764 / 2.17 vs HULC 0.715 / 1.82。
- Table 2 真机（Kinova Gen2 7-DoF，腕部 RealSense + 静态 Kinect；搬运 1775 条演示、抽屉 2856 条，App A.3）：搬运 Seen 0.79 / Unseen instance 0.73 / Unseen category 0.30，抽屉开关 0.75；RT-1 0.27 / 0.13 / 0.00 / 0.35；MT-R3M 0.15 / 0.13 / 0.10 / 0.30。
- **最有信息量的消融 Table 4**：去掉视频预测 + 预训练 → ABCD→D 3.33、ABC→D 2.40、10% 数据 1.04；只加视频预测（不预训练）→ 3.82 / 2.65 / 1.52；全套 → 4.21 / 3.06 / 2.00。即"视频预测当辅助目标"本身贡献约一半，Ego4D 预训练贡献另一半，数据越少收益越大。
- Fig 7(b) 真机 pick-place（估读）：无视频 / 只加视频预测 / 全套的 picking 约 0.38 / 0.29 / 0.87，transporting 约 0.33 / 0.29 / 0.83——不预训练只加视频损失在真机上反而更差。
- Table 5 预测多远（无预训练）：Δt = 1 / 3 / 5 → Avg. Len 3.61 / 3.82 / 3.67——相邻帧太像信息量低，太远又指导不了局部动作。
- Table 7 逐任务：提升最大的是积木操作（stack block 45.7→80.1，rotate / lift 类 +20 左右），开关 / 滑门类本来就接近 100%。Fig 6：预测帧大体正确但丢遮挡细节。

## 4. 局限
- 作者承认（Sec 6）：只用了有语言标注的数据，想混入无语言视频；没比较"任意视频 vs 操作相关视频"预训练；机器人数据规模与技能数有限。
- 我读出来的（方法）：预测空间是低分辨率像素 MSE，预测帧模糊、确定性、表达不了多模态未来；单步动作、无 action chunk；MAE、CLIP 冻结意味着视觉表征本身没被视频预训练改变，46M 可训练参数限制上限；无延迟 / 频率数字。
- 我读出来的（机制解释）：视频预测不参与动作生成，所以它不是"用想象来规划"，只是表示学习的正则——论文却把机制解释成"预见未来引导动作"，直到 Fast-WAM 才有受控实验澄清。
- 我读出来的（证据）：CALVIN baseline 数据条件不一致（HULC / MCIL 用全量 play 数据）；真机 unseen category 0.30、失败模式是颜色混淆，语义泛化主要靠冻结的 CLIP。

## 5. 复现要点
- 代码：GitHub bytedance/GR-1（权重未核实）；依赖 MAE ViT、CLIP、perceiver resampler。
- 算力：模型小（195M，46M 可训），CALVIN 微调单机 8 卡即可；Ego4D 预训练要 80 万段视频、batch 1024、50 epoch，存储与预处理是主要成本。
- 数据：CALVIN 只用 1% 语言标注子集；Ego4D 需自行申请下载并裁 3 s 片段。
- 坑：[OBS] / [ACT] 必须从其他 token 的 attention 中 mask 掉；语言 token 每步重复；Δt 微调时用 3（真机 / CALVIN 帧率不同要重调）；预训练与微调 lr 不同（3.6e-4 vs 1e-3）；两路相机都要预测；真机抽屉任务的失败模式（没完全关上、没勾到把手）提示要留 recovery 逻辑。

## 6. 关键引用链
- 建立在：GPT 式 causal transformer（Decision Transformer、Gato、RoboCat——RoboCat 也预测未来帧但是目标图像条件且无视频预训练）；MAE（图像编码 + patch 重建损失）；CLIP；Ego4D；VideoGPT 类视频预测任务。
- 对照：R3M（"同数据源、不同预训练目标"）；UniPi（UniPi）作为"视频模型 + IDM 分开"的对照。
- 后续（同团队）：GR-2（arXiv 2410.06158，扩到 3800 万段视频预训练 + 更大模型，tutorial 归为 joint 范式）；GR-MG、GR-3 延续。
- 后续（他人）：VPP（VPP）在 CALVIN ABC→D 把 3.06 提到 4.33，并批评 GR-1 每次只预测一帧、不用视频基础模型；UWM（UWM）指出 GR-1 用 L2 回归而非生成建模、在真机上以它为 baseline；Fast-WAM（Fast-WAM）的"视频 co-training 才是收益来源"与 Table 4 一致；WorldVLA（arXiv:2506.21539）、Motus（arXiv:2512.13030）把同一思想推到离散 token / 统一模型。
