# Cosmos World Foundation Model Platform for Physical AI

> 对应「可lip」第 29 期视频（世界模型系列第三期）。论文：[arXiv:2501.03575](https://arxiv.org/abs/2501.03575)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2501.03575（v1 2025-01-07；v3 2025-07-09） · 机构: NVIDIA · 代码/权重: https://github.com/nvidia-cosmos/cosmos-predict1（NVIDIA Open Model License，开放权重；Cosmos Tokenizer、TokenBench、ShotBench 亦开源） · 项目页: https://research.nvidia.com/labs/dir/cosmos1/
- 一句话: 提出 world foundation model（WFM）范式——先在约 20M 小时视频上预训练通用视频生成模型，再用少量目标机器人/车辆数据 post-train 成专用世界模型；开源数据管线、视频 tokenizer、diffusion 与 autoregressive 两族 WFM、post-training 示例和 guardrail，是 DreamGen、Cosmos Policy 等 NVIDIA 机器人 WM 工作的底座。

## 1. 要解决的问题
Physical AI（带传感器和执行器的 AI）需要"观测-动作交错序列"做训练数据，真实采集昂贵且危险。作者把 WFM 定义为物理世界的数字孪生：给定过去观测 $x_{0:t}$ 和当前扰动 $c_t$（动作、文本、相机轨迹等）预测 $\hat{x}_{t+1}$（Sec 2，Fig 3）。此前视频模型（VideoLDM、IRASim）规模小或只针对单一领域。论文列出 WFM 的五种用途（policy evaluation / initialization / training、planning-MPC、synthetic data generation），但明确说明**本文未对这些用途做实验**（Sec 2.1）。

## 2. 方法
**数据管线（Sec 3）**：约 20M 小时原始视频（驾驶 11%、手部/物体操作 16%、导航 16%、自然动态 20% 等），经 TransNetV2 切镜头（2–60 s）、GPU 转码、运动/画质/文字/类型过滤、VILA-13B 打 caption、InternVideo2 语义去重（去掉约 30%），得到约 $10^8$ 个预训练片段和 $10^7$ 个微调片段。

**Tokenizer（Sec 4）**：把视频压成紧凑 token 再还原的 encoder-decoder。Cosmos-Tokenize1 时间因果（当前帧 token 不看未来帧，故图像即单帧视频，可联合训练），3D Haar 小波 + 因果时空卷积/注意力。连续版 CV 为普通 AE（latent 16 维），离散版 DV 用 FSQ（词表 64,000）；压缩率 4×8×8 / 8×8×8 / 8×16×16（T×H×W）。DAVIS 上 CV4×8×8 PSNR 35.85 vs CogVideoX tokenizer 29.29（Tab 5），快 2–12×（Tab 9）。

**预测空间**：RGB 视频，在 tokenizer 的 latent 空间建模（diffusion 用连续 latent，AR 用离散 token）。

**Diffusion WFM（Sec 5.1）**：latent diffusion，EDM 去噪损失 $\mathcal{L}=\mathbb{E}\|D_\theta(x_0+n;\sigma)-x_0\|^2$ 加按 $\sigma$ 学习的不确定性权重（Eq 5–8）。backbone 是改造 DiT（3D patchify、FPS-aware 3D RoPE、T5-XXL cross-attention、QK-norm、AdaLN-LoRA），7B / 14B 两档（Tab 11）。先训 Text2World，再把条件帧沿时间维拼接 + mask 微调成 Video2World。渐进训练 512p×57 帧 → 720p×121 帧（上下文 56,320 token），FSDP 64 + context parallel 8（Tab 12）。

**Autoregressive WFM（Sec 5.2）**：Llama3 式 decoder-only transformer 从头训练做 next-token 预测（Eq 9），输入 DV8×16×16 离散 token；4B/12B 基座 → 加 cross-attention 成 5B/13B Video2World。离散 token 输出偏糊，用 7B diffusion 模型微调出 diffusion decoder 提升画质；推理用 KV cache + Medusa 投机解码。

**动作怎么进出**：预训练 WFM 只接受文本 + 过去帧，**不输出动作**；动作仅在 post-training 时作为条件注入（Sec 6，Tab 21）：相机控制把位姿转成 Plücker 嵌入沿通道拼到 latent；机器人 (a) 指令条件视频预测，Cosmos-1X 数据集（1x EVE 人形，约 200 h、约 12,000 段）；(b) 动作条件下一帧预测，Bridge（约 20,000 段，320×256@5FPS），动作为 7 维末端增量 $(\Delta x,\Delta y,\Delta z,\Delta\theta_{r,p,y},\Delta\text{Gripper})$，5B AR 版把动作过 MLP 后经 cross-attention 注入，7B diffusion 版加到 DiT 的 timestep embedding 上，给定动作序列可自回归滚出视频；驾驶用 RDS 数据集（约 3.6M 段六路环视，约 20,000 h）做六视角联合去噪，可选轨迹条件。纯预测器，无 cascaded/joint 之分。

**Guardrail（Sec 7）**：pre-Guard = 关键词黑名单 + Aegis（LlamaGuard 微调版）过滤 prompt；post-Guard = SigLIP+MLP 帧级安全分类、RetinaFace 人脸打码。

**训练算力**：10,000 张 H100 训练三个月（Sec 5）。

**推理频率与延迟**：AR 4B 单张 H100 生成 32 帧（640×1024）需 31.04 s，13B 需 109.18 s（Tab 16）；320×512 适配 + Medusa 后 8×H100 达 10 FPS（Tab 17）。diffusion WFM 推理时延论文未说明。

**怎么用于控制**：只作可交互模拟器；planning、policy 训练、评测停留在愿景。

**分类体系定位**：纯 WM，observation-space（单视角与六视角 RGB，latent 内建模）；动作抽象层级覆盖 language instruction（Video2World、Instruction 版）、low-level robot action（ActionCond 版 7-DoF 末端增量）及相机位姿/车辆轨迹这类 interface action。不产生动作，不属于任何 WAM 范式。

## 3. 实验
- **3D 一致性**（Tab 19，RealEstate10K 500 段）：7B Text2World Sampson error 0.355、位姿估计成功率 62.6%；VideoLDM 0.841 / 4.4%；真实视频 0.431 / 56.4%。
- **物理对齐**（Tab 20，Isaac Sim 生成 8 类刚体场景 800 段）：7B Video2World 9 帧条件 PSNR 21.06、物体 IoU 0.592，1 帧条件 0.332；14B 不更好（0.598），作者结论"所有 WFM 在物理遵循上同样吃力"。
- **机器人 post-training**：指令条件视频人类偏好 7B 版 78.3% vs VideoLDM-Instruction 13.0%（Fig 24）；Bridge 动作条件预测 7B 版 PSNR 21.14 / FVD 190 vs IRASim-Action 19.13 / 593（Tab 23）。
- **相机控制**（Tab 22）：位姿估计成功率 82.0% vs CamCo 43.0%。**驾驶**（Tab 24/25）：FVD 210 vs 884；轨迹跟随误差 20.20 cm，真实视频 13.49 cm。
- **AR 失败率**（Tab 18）：单帧条件 4B 有 15% 出现"物体凭空出现"，9 帧条件 ≤2%。diffusion 在 3D 一致性和机器人视频质量上优于 AR，AR 的价值在能做实时（Sec 8）。

## 4. 局限
作者承认（Sec 8）：缺 object permanence、接触动力学不准、指令遵循不稳定；逼真不等于物理正确；评测困难且人类打分未必与下游任务相关。
我读出来的：(1) 五种用途都无闭环实验，"WM → 策略"留给后人；(2) 动作条件只在 Bridge 5 FPS、320×256 上验证，远低于控制需求；(3) 推理秒到分钟级，只有 4B AR 低分辨率用 8 卡才到 10 FPS；(4) 含大量专有数据，预训练不可复现；(5) 生成无不确定性估计，规划时无法判断该不该信。

## 5. 复现要点
- 开源：cosmos-predict1 仓库 + Hugging Face 权重（NVIDIA Open Model License）；后续被 Cosmos-Predict2（2B/14B）取代。规模：diffusion 7B/14B，AR 4B/5B/12B/13B。预训练 10,000 H100 × 3 个月，不可复现。
- 8×H100 判断：**推理可行**——AR 13B 单卡 77 GB（Tab 16），diffusion 7B 估计可单卡（论文未给显存）。**post-training 可行但受限**：论文未说明机器人 post-training 的 GPU 数；按 Sec 5.1.4 的 20 B/参数估算，7B 全参微调参数+梯度+优化器约 140 GB，FSDP 分到 8 卡约 17.5 GB/卡，512p×57 帧可跑，720p×121 帧需 context parallel=8 且整节点只放 1 个样本；14B 建议 LoRA。
- 坑：AR 单帧条件失败率高，应给 ≥9 帧；离散 token 必须接 diffusion decoder，延迟翻倍（Tab 16）；训练 caption 由 VILA 生成，需 prompt upsampler。

## 6. 关键引用链
- 建立在：DiT、EDM（Karras et al. 2022）、VideoLDM、FSQ（Mentzer et al. 2023）、Llama3 式 GPT、Medusa；机器人基线 IRASim（Zhu et al. 2024）；物理评测思路来自 Kang et al. 2024。
- 后续在其上：Cosmos-Transfer1（Alhaija et al. 2025）、Cosmos-Predict2（Cosmos Policy 的基座）、DreamGen（把 Cosmos-sft 作为候选 video world model 之一评测）、Cosmos Policy（Kim et al. 2026）。
