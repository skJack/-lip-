# World Simulation with Video Foundation Models for Physical AI（Cosmos-Predict2.5 / Cosmos-Transfer2.5）

> 对应「可lip」第 29 期视频（世界模型系列第三期）。论文：[arXiv:2511.00062](https://arxiv.org/abs/2511.00062)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2511.00062（v1 2025-10-31 前后，精确日期未核实；NVIDIA 技术报告，未投会议） · 机构: NVIDIA · 代码/权重: https://github.com/nvidia-cosmos/cosmos-predict2.5 与 https://github.com/nvidia-cosmos/cosmos-transfer2.5，NVIDIA Open Model License，2B / 14B 的 pre-trained 与 post-trained 权重、驾驶 7 视角、机器人动作条件、AgiBot 3 视角、GR1 等专用权重（Tab 1 清单）；2026-09 时仓库已标注"不再活跃开发，后续转 Cosmos 3" · 项目页: 未核实
- 一句话: Cosmos 1 扩散线的第二代：去掉自回归分支，把 EDM 换成 flow matching、T5 换成 Cosmos-Reason1 VLM 当文本编码器、自家 tokenizer 换成 Wan2.1 VAE，Text2World / Image2World / Video2World 合成一个模型；数据从 2000 万小时扩到 3500 万小时、过滤后 2 亿段；后训练加了领域 SFT + 模型合并 + GRPO 式 RL + 时间步蒸馏。2B 在 PAI-Bench 上和 Wan2.2 27B 打平，Bridge 动作条件预测 PSNR 21.14 → 24.95。仍然不出动作，是 Cosmos Policy（`Cosmos-Policy`）和 DreamDojo 的基座。

## 1. 要解决的问题
- 和 Cosmos 1 同一个 motivation：真机训练慢、贵、危险，要一个能按 agent 动作生成视觉环境的"世界模拟器"。本文不改问题，只改答案的质量。
- 引言点名 Cosmos-Predict1 三个短板：数据过滤不够严、架构上 Text2World 和 Video2World 是两个模型、训练配方只有预训练加少量微调。对应三处升级：更严的过滤加人工挑的 Physical AI 后训练数据；一个模型三种模式；模型合并加 RL 后训练加 VLM 文本编码器。

## 2. 方法
**预测空间**：Wan2.1 VAE 的连续 latent，因果，压缩率 4×8×8（时间 × 高 × 宽），latent 上再做 1×2×2 patch。每次生成 93 帧、16 FPS、约 5.8 秒，对应 24 个 latent 帧。

**训练目标（Sec 3.1）**：flow matching。x_t = (1−t)x + tε，目标速度 v_t = ε − x，损失 ‖u(x_t, t, c) − v_t‖²。t 从 logit-normal 抽，再做 shift t_s = βt / (1 + (β−1)t) 偏向高噪声，256p 时 β = 1、720p 时 β = 5；另外强制 5% 的样本落在噪声最高的 2% 区间，否则帧间会出现突兀跳变。作者说 FM 和 EDM 的前后向过程数学等价，差别只在网络预测速度而不是干净样本。

**架构（Sec 3.2，Tab 3）**：沿用 Cosmos 1 的 DiT，唯一改动是去掉绝对位置嵌入、只留 3D RoPE，为的是后训练时能外推到更高分辨率和更长序列。2B：32 层、宽 2048、16 头；14B：36 层、宽 5120、40 头，与 Cosmos 1 的 14B 同尺寸。AdaLN-LoRA 秩 256 保留。文本编码器换成 Cosmos-Reason1：取多个 transformer block 的激活拼接后投到 1024 维，经交叉注意力注入。

**三种模式合一**：Image2World 和 Video2World 用"帧替换"：生成序列的前几帧始终被条件帧替换；token 上拼一个 mask 标志位说明是否条件帧，损失只算生成帧。预训练最后一阶段按 0.5 / 0.25 / 0.25 的概率抽 0 / 1 / 2 个条件帧，0 帧就是 Text2World。

**预训练课程（Sec 4.1，Tab 4）**：Text2Image 256p → 加 Image2World / Video2World 联合训 256p → 480p → 720p → 加 Text2World。条件帧数训练时随机 1 或 5，生成 92 或 88 帧。AdamW，2B 学习率 3e-5、14B 1.3e-5，权重衰减 0.001，warmup 2000 步后线性衰减。

**后训练（Sec 4.2）**：(1) 领域 SFT：用 InternVideo2 嵌入训多头分类器把数据分成物体恒存、高运动、复杂场景、驾驶、机器人操作五个领域，每个领域单独微调一个模型，各 3 万步、batch 256；(2) 4K 高质量视频上冷却；(3) 模型合并：试了 model soup、TIES、DARE-Linear、DARE-TIES，最终用 model soup；(4) RL：VideoAlign 当奖励模型（文本对齐、运动质量、画质），每个条件采 8 个样本、20 步去噪，GRPO 式组内归一化优势，256 步、batch 32，加扩散损失正则防 reward hacking；(5) 时间步蒸馏 rCM，4 步出图，指标接近教师。

**动作怎么进出**：预训练不含动作。动作条件版（Sec 6.6）加一个动作 MLP，加到 DiT 的时间步嵌入上；消融显示时间步嵌入 > 交叉注意力 > 通道拼接（Tab 15 PSNR 24.95 / 24.41 / 23.11）。给一张图和一串动作出一段未来帧，自回归接续。不输出动作。

**Cosmos-Transfer2.5（Sec 6.1–6.3）**：ControlNet 式，条件是边缘、模糊、分割、深度或驾驶的语义地图；2B，比 Transfer1 小 3.5 倍；用于 Sim2Real、Real2Real、多视角驾驶、机器人视觉增广。

**分类体系定位**：纯 WM，observation-space（Wan VAE latent）；动作层级覆盖 language instruction 和 post-training 后的 low-level robot action；不属于 WAM 四范式。

## 3. 实验
- **PAI-Bench（Tab 5、6）**：Text2World 总分 2B 后训练 0.768、14B 0.768，Wan2.2-27B-A14B 0.769、Wan2.1-14B 0.761；Image2World 2B / 14B 后训练 0.810，Wan2.2-27B 0.806、Wan2.1-14B 0.797。人评：2B 对 Wan2.2-5B 30.0% vs 26.2% 胜率，对 Wan2.1-14B 33.0% vs 34.8%；14B 对 Wan2.1-14B 48.6% vs 31.8%，对 Wan2.2-27B 38.1% vs 35.9%。
- **RL 前后（Tab 7，图）**：Text2World 与 Image2World 的奖励分均大幅上升，人评 RL 后一致更受偏好，具体数字见表（未抄录）。
- **Bridge 动作条件（Tab 14）**：Predict2.5-2B PSNR 24.95 / SSIM 0.85 / Latent L2 0.28 / FVD 146，Cosmos 1 的 7B 版 21.14 / 0.82 / 0.32 / 190，同一测试集 100 段。
- **DreamGen Bench GR1 指令跟随（Tab 16）**：14B 后训练版 Object 91.8 / 69.4（GPT / Qwen 评）、Behavior 70.2 / 59.6、Env 69.0 / 69.0；WAN2.1 72.0 / 58.0、72.3 / 55.3、48.3 / 65.5；Predict2-14B 90.0 / 62.0、59.6 / 61.7、69.0 / 65.5。
- **Transfer2.5 真机（Sec 6.2）**：双臂 Kinova Gen3 半人形平台，苹果放碗任务 100 段遥操作演示训 Diffusion Policy，用 Transfer2.5 做视觉增广后在对抗性视觉扰动下评测；具体成功率见原文 Sec 6.2 表（未抄录）。这是"生成数据训策略再上真机"，不是 WFM 自己出动作。
- 驾驶 7 视角、相机控制多视角等应用见 Sec 6.3、6.4，未抄录。

## 4. 局限
作者承认（Sec 8，未细读）：未核实。
我读出来的：(1) 仍然只是预测器，不出动作、不评策略，五种用途里只落地了"合成数据"和"动作条件模拟"两种；(2) 动作条件仍只在 Bridge 5 FPS、320×256 上验证，和 Cosmos 1 同一设定；(3) 物理对齐没有再报 Cosmos 1 那套 Isaac Sim 评测，无法判断"模型大了物理不变好"的问题是否解决；(4) 通用画质对比只到 Wan 2.1 / 2.2，没有和闭源模型比；(5) 文本编码器换成 Reason1 后没有单独消融它带来多少；(6) 预训练数据仍不公开。

## 5. 复现要点
- 开源完整：代码、2B / 14B 权重、专用权重、后训练样例、PAI-Bench 评测脚本；官方 cookbook（2026-02）有动作条件蒸馏和 RoboCasa / LIBERO 策略模型（网页信息，未在论文内）。
- 预训练 3500 万小时原始视频不可复现；算力论文 Sec 4.3 未抄录。
- 8×H100：2B 全参 post-training 可行（参考 Cosmos Policy 在 8×H100 × 48 h 训 2B），14B 建议 LoRA / DoRA（NVIDIA 官方博客有 LoRA 微调教程）。
- 坑：动作要加到时间步嵌入而不是通道拼接（Tab 15 差 1.8 dB）；高噪声区要额外采样否则帧间跳变；用户提示与训练 caption 风格差异仍需处理（Reason1 编码，具体扩写方式未核实）。

## 6. 关键引用链
- 建立在：Cosmos 1（`Cosmos-1`）的 DiT、数据管线和 post-training 思路；flow matching（Lipman 2022）与 SD3 的 shifted logit-normal（Esser 2024）；Wan2.1 VAE；Cosmos-Reason1；VideoAlign 奖励模型与 GRPO；rCM 蒸馏；DreamGen（`arXiv:2505.12705`）的合成数据范式与 bench；IRASim 的 Bridge 评测设定。
- 后续在其上：Cosmos Policy（`Cosmos-Policy`，基座为 Predict2）、DreamDojo（`arXiv:2602.06949`，基座为 Predict2.5）、GR00T-Dreams；被 Cosmos 3（`Cosmos-3`）取代。
