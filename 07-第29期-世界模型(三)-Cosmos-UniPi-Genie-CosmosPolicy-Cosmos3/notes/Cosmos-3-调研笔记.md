# Cosmos 3: Omnimodal World Models for Physical AI（Cosmos 3）

> 对应「可lip」第 29 期视频（世界模型系列第三期）。论文：[arXiv:2606.02800](https://arxiv.org/abs/2606.02800)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2606.02800（v1 2026-06，具体日期未核实；本笔记依据的版本是 v4 2026-06-23；NVIDIA 技术报告，未投会议） · 机构: NVIDIA · 代码/权重: https://github.com/nvidia/cosmos 与 https://github.com/nvidia/cosmos-framework；Hugging Face nvidia/cosmos3 开放 Cosmos3-Nano（16B）、Cosmos3-Super（64B）两个 mid-train 基座和三个 post-train 变体 Cosmos3-Super-Text2Image、Cosmos3-Super-Image2Video、Cosmos3-Nano-Policy-DROID，另有 5 个 SDG 合成数据集与 Cosmos-HUE 评测集；许可证 OpenMDW-1.1（Linux 基金会）；Cosmos3-Edge（4B）写明"later release"未开放；Tab 18 各领域 FD/ID post-train 权重是否发布未核实 · 项目页: https://research.nvidia.com/labs/cosmos-lab/cosmos3
- 一句话: 把语言、图像、视频、音频、动作放进同一个 Mixture-of-Transformers（MoT：一条 token 序列，不同模态各用一套独立的 transformer 参数，只在共享 self-attention 里交互）里，语言走自回归 next-token、其余模态走 rectified flow 联合去噪，一个权重同时是 VLM、视频生成器、forward / inverse dynamics 世界模型和联合出视频+动作的 WAM。相对 Cosmos 1（见 Cosmos-1 笔记）→ Cosmos-Predict2 / Cosmos Policy（见 Cosmos-Policy 笔记）这条线，它第一次做到三件事：(1) 不再走 GAIA-1 / Genie / Cosmos 1 AR 分支那种"视频离散 token + 自回归"路线，只让语言是离散 AR，视频/音频/动作全是连续 latent 联合去噪；(2) 动作从 Cosmos 1 的 post-training 附件变成 mid-training 的一等模态（61.3K 小时、4 类本体、FD/ID/policy 三种模式，Sec 3.2.3、4.2.2）；(3) 两塔都从 Qwen3-VL 初始化，理解与生成共用一个模型——它是 Cosmos 系列第一个基座本身就能出动作的模型，post-train 后在 RoboArena 真机榜登顶（Fig 26）。

## 1. 要解决的问题
- 作者的判断（Sec 1）：Physical AI 需要"理解"（从局部观测推断状态、语义、动力学）和"生成"（预测未来、想象动作后果）两种耦合的能力，但现状是三套模型各管一段——VLM 管感知与规划、VLA / WAM 管出动作、视频模型或 forward dynamics model 管模拟评估；家用机器人收拾餐桌要串三个模型，既浪费算力又无法共享表征。目标：一个不改架构、靠输入-输出配置切换角色的统一模型。
- Cosmos 1 提出的五种 WFM 用途当时都没做实验（见 Cosmos-1 笔记）；Cosmos Policy 证明视频模型可以 post-train 成 policy，但只是 2B 单模态视频模型的改装（见 Cosmos-Policy 笔记）。Cosmos 3 要在预训练阶段就把这些角色装进去。
- 第二个动机是数据（Sec 1、Fig 2）：同一个模型作合成数据生成器、作专用模型的 mid-training 起点、长远作训练环境。

## 2. 方法
**预测空间：observation-space，连续 latent。** 视觉生成用冻结的 Wan2.2-TI2V-5B 视频 VAE（时间 4×、空间 16×16 压缩，再 2×2 patch merge，等效每 token 32×32 像素，线性层投到 transformer 维度）；视觉理解另用一个与语言对齐、随 backbone 一起训练的 ViT（16×16 patch + 2×2 merge，DeepStack 特征聚合）；音频用 ETTA 的音频 VAE（48 kHz 立体声，hop 1920，即 25 token/s，冻结）；动作是显式向量，每个采样步一个 token（Sec 2.1）。
- **MoT 两塔（Sec 2.3，Fig 5）：** 序列 = 自回归子序列（文本 + ViT 视觉 token，走 reasoner 塔，因果注意力、只看自己，式 7）+ 扩散子序列（VAE 视频/图像 latent、音频、动作，走 generator 塔，双向注意力，K/V 拼接 [AR; DM]，式 8）；AR token 永远不看 DM token。两塔各有独立的 LayerNorm / QKV / FFN，都从 Qwen3-VL 初始化；训 generator 时 reasoner 冻结（Sec 4.2.1）。
- **位置编码（Sec 2.4）：** 带绝对时间轴的 3D MRoPE：时间步长按 δt = TPS_base / TPS 缩放（式 9，TPS_base = 24 FPS / 4 = 6），音频 25 TPS、动作 TPS = 采样率，使不同帧率的视频、音频、动作对齐到同一物理时间轴；AR 与 DM 之间插 15000 的位置间隔，否则首帧过饱和。
- **token 排布与模式（Sec 2.2）：** 统一格式 [文本…, <EOS>, <BOG>, 干净条件 token, 噪声目标 token]，条件在前、目标在后，模态顺序视觉→音频→动作（式 3–6）。由此得到 Language（纯 VLM）、T2I、T2V(+Audio)、I2V / V2V(+Audio)（P = 1 或 P > 1 个干净 latent 帧作条件）、Video transfer（edge / depth / seg / blur 或驾驶场景图作干净条件）、Action 三种：forward dynamics（FD，干净动作 + 观测 → 去噪未来视频）、inverse dynamics（ID，干净视频 → 去噪动作）、policy（视频与动作同时去噪，Fig 4）。
- **动作表示（Sec 2.1.3，Fig 3）：** a_t 定义为 v_{t−1} → v_t 的因果变量，统一成"ego 位姿增量 + 末端位姿增量 + 抓取状态"三段：位姿用相邻 SE(3) 的相对变换 ΔT_t = T_{t−1}^{-1} T_t（3D 平移 + 6D 旋转），抓取状态直接编码当前值；机器人 = 头相机位姿增量 + 末端法兰位姿增量 + 夹爪连续开合，人手 = 头相机增量 + 腕位姿增量 + 指尖坐标，相机 / 车辆只有 ego 位姿。它是**伪动作**（状态差分），刻意不含 PID 等控制器细节；没有 latent action。
- **动作怎么进出：** 每个本体域 k 有自己的线性输入 / 输出投影 W_in^(k)、W_out^(k)（式 1–2，X-VLA 思路），backbone 共享；预测的 6D 旋转经 SVD 投回 SO(3)。动作既能作条件（FD）也能作输出（ID / policy）。post-train 时动作空间可重定义：Cosmos3-Nano-Policy-DROID 新建动作编码器 / 解码 MLP，直接输出 32 步绝对关节位置（Sec 4.2.5）。
- **损失（Sec 4.2）：** 语言 next-token；其余模态 rectified flow matching（x_σ = σ·ε + (1−σ)·x_0，预测速度 v* = ε − x_0 的 masked MSE，条件 token 不计损失），各模态独立采样 σ，图像/音频/动作 logit-normal、视频 mode sampling，shift 重参数化 s 在预训练为 1 / 3 / 5、mid-training 提到 3 / 5 / 10（256p / 480p / 720p）；mid-training 总损失 = 各模态速度 MSE 加权和，动作损失 ×10。
- **训练阶段（Sec 4、Fig 8）——Reasoner：** Qwen3-VL-8B / 32B 初始化 → 22M 样本预训练 2 epoch（≤16k token）→ 2.2M 样本 Physical AI SFT（8200 步、batch 512）；无 RL / 偏好对齐，全文未提。
- **Generator 预训练：** reasoner 权重初始化 generator 塔；767M 图 + 347.7M 视频片段（其中 138.9M 段带音频）；256p / 480p / 720p 多分辨率、5–400 帧、10–30 FPS、74K token 打包上下文；模式配比 T2I / T2V / I2V / V2V = 20 / 56 / 16 / 8%；文本 dropout 10% 供 CFG。
- **Generator mid-training（Tab 6）：** 图 10%、视频 32%、视频+音频 8%、**动作 25%**、通用 transfer 20%、驾驶 transfer 5%；视频池 74.7M 精选片段；动作数据 8.4M episode / 61.3K h：第一人称人手 41.3K h（专有）、自动驾驶 10.0K h（Hyperion 内部日志）、机器人 5.4K h（AgiBot 4.37K h、Franka / DROID 442 h、Google Robot 351 h、WidowX 100 h 等，Tab 4，含失败 episode）、相机运动 4.6K h（用 ViPE / DepthAnything3 从预训练视频估位姿）。
- **post-training：** 三个专才——T2I 20k + 2k 步；I2V 10k 步约 50B token，480p × 189 帧；policy 见下。
- **参数量（Sec 2.5，Tab 2）：** Edge 4B（两塔各 2B，从头训）、Nano 16B（两塔各 8B）、Super 64B（两塔各 32B），不含 ViT 与 VAE。
- **算力：** 预训练 Nano 31.05T token / 1024 GB200，Super 17.86T / 2048 GB200；mid-training Nano 2.4T / 1024，Super 1.9T / 2048（Sec 4.2.1–4.2.2）；吞吐 Nano 520 TFLOPS/GPU、MFU 0.23，Super 673、0.30（Tab 8）。按 Tab 8 吞吐粗估 Nano 预训练约两个月（估算）。
- **推理开销（Sec 5.3，Fig 16 图内标注）：** 默认 50 步去噪 + CFG（Tab 21）；Nano 720p T2V 单卡 H100 NVL 297 s（vLLM-Omni 286 s），B200 115 s；720p T2I 4.2 s / 1.8 s；720p 189 帧占满 74K 上下文只能 batch = 1（Tab 9）。policy 模式只用 4 步、CFG 3，跳过视频 latent 解码，用 2 张 RTX Pro 6000 做 CFG 并行部署（Sec 4.2.5）；每个动作块的时延论文未给。无蒸馏。
- **怎么用于控制（Sec 4.2.5）：** DROID（76k 轨迹、350 h、86 任务、564 场景，去空闲帧和失败演示后 58K 样本，Fig 8 图内标注）上从 mid-train Nano 继续训：输入本体状态 + 三视角拼成 540×640 画布（腕视角 360×640 在上，两外视角 180×320 在下）+ DROID 短指令，输出 32 步绝对关节位置（15 Hz，即约 2.1 s 动作块）和辅助 RGB 未来帧；动作参数 lr ×5；Franky 控制器开环执行整块。


## 3. 实验
- **Reasoner（Tab 1 / Tab 10，48 个基准）：** Super 通用 73.7、机器人 57.8、智能基础设施 62.6、驾驶 79.3；Gemini 3.1 Pro 77.5 / 58.2 / 58.6 / 47.2，Qwen3-VL-32B 72.8 / 52.6 / 56.1 / 40.7——Physical AI 域领先、通用域略逊闭源。
- **视频生成自动指标：** PAIBench-G（Tab 12，1044 提示，Qwen2.5-VL-72B 判官）T2V 总分 Super 80.0 vs Veo-3.1 79.1、Wan2.2-A14B 78.0、**Cosmos-Predict2.5-2B / 14B 76.5 / 76.4**；I2V 82.8 vs 82.6 / 81.3 / 81.2 / 81.1；RBench（650 例机器人操作）Nano 58.4% vs Wan 2.6 60.7%、Predict2.5-2B 46.4%。Physics-IQ（Tab 13）I2V Super 43.8（Sora2 42.3、Wan2.2-A14B 38.3），配 WMReward best-of-N 48.9；V2V 59.7（Magi-1 56.0），BoN 63.4。
- **视频生成人评：** Cosmos-HUE（Tab 14，原子二值问题）T2V Super 89.3 vs Veo-3.1 91.3、真视频 93.6、Predict2.5-14B 82.1；I2V 89.6 vs Veo 89.7；Human World Bench（180 段第一人称操作 I2V）Super 71.9 vs Veo-3.1 67.8、Wan2.2-A14B 60.7、Predict2.5-14B 38.7。作者指出自动指标的区分度只有人评的一半不到（约 4 分 vs 约 10 分，Sec 6.2.2）。
- **机器人 FD（Tab 18，各领域用同一配方 post-train；PT-init = 未见动作数据的预训练 ckpt，MT-init = mid-train ckpt）：** DROID 上首帧 + 16 步末端动作块 → 16 帧，PSNR Super-MT 26.04、Nano-MT 25.52，Nano-PT 23.24、Super-PT 22.69，**Ctrl-World 22.99**（Fig 24：布料交互更真实；注意 Super-PT 还不如 Ctrl-World，收益来自动作 mid-training）。
- **其他动作模式（Tab 18）：** 相机 FD（100 段 5 s 真实片段，DepthAnything3 回估轨迹）Super-MT RRE 0.142°、RTE 0.026 m、ATE 0.99 m vs Lingbot-World 0.299 / 0.057 / 2.88、HY-World 1.5 0.377 / 0.042 / 1.39；驾驶 ID（内部 6 s @10 FPS 片段，直接从视频回归 ego 轨迹）Super-MT ATE 0.90 m vs DepthAnything3 9.29、VGGT 23.46（尺度漂移）；第一人称手部 FD（HWB）PSNR 16.19 vs LOME 9.36。MT-init 在所有列都优于 PT-init——"统一动作 mid-training 是可复用的动作先验"是本节主论点。
- **机器人 policy（Tab 19，RoboLab-120 仿真，120 任务 × 10 rollout，所有对手都是 DROID 微调 ckpt）：** Cosmos3-Nano-Policy-DROID 总成功率 vague / default / specific = 20.6 / 36.8 / 39.7%，PT-init 直接 post-train 16.7 / 28.1 / 30.2，π0.5 15.2 / 28.0 / 28.1，**DreamZero 14.9 / 25.7 / 23.9**，π0-FAST 9.2 / 15.5 / 14.9，GR00T N1.6 5.4 / 7.2 / 5.3，π0 2.8 / 5.0 / 3.5；唯一输的格子是 Complex-Vague（4.1 vs PT-init 7.1）。
- **真机与第三方榜：** RoboArena（DROID 平台众包双盲 A/B）2026-05-30 排第 1（Fig 26，具体 rating 只在截图里，未核实）；MolmoSpaces 仿真 2026-06-20 All Combined oracle 成功率 39.0% 第 1（Fig 27）；三个榜用同一权重、同一超参。
- **新本体快速适配（Tab 20，LIBERO-10，第三人称 + 腕视角，500 rollout / ckpt）：** MT-init 在 500 / 1000 / 1500 / 2000 步为 24.6 / 91.4 / 95.8 / 97.4%，PT-init 0.0 / 73.8 / 93.4 / 95.2%——差距主要在早期。
- **最有信息量的消融 (1)** 动作三模式协同（Tab 31，PushT，Edge 模型）：单模式各 2K 步 vs 三模式联合 6K 步，ID MSE 1.11e-3 → 3.09e-4（−72%）、policy 覆盖率 74.1 → 77.3%，但 FD PSNR 27.13 → 26.22。
- **(2)** 视频-动作一致性（App E.5，Fig 37）：把 policy 输出的动作块在 RoboLab 仿真里执行，与同时生成的视频比 PSNR，第三人称 23.19 dB、腕视角 17.33 dB。
- **(3)** 跨域协同矩阵（Fig 28 / 29，取正文数字）：相机 FD 加驾驶数据 PSNR 11.96 → 12.82，WidowX 加 Google Robot +1.39，AgiBot 先用人手数据 warm-up 再适配 +0.94（5K 步）到 +1.3–1.6。
- **(4)** 理解塔换成原版 Qwen3-VL-8B（Tab 28）T2V 域分 75.7 → 73.7、Robot 71.3 → 66.5；**(5)** SDG 合成数据（Tab 26）全加只涨 PAIBench-G 总分 +0.10。
- 音频 / transfer 一句话：Cosmos-SoundBench AVQ Super 7.31 vs Veo-3.1 7.45（Tab 15）；PAIBench-C 四种控制无需逐模态 ControlNet 即持平或超过 Cosmos-Transfer2.5（Tab 16），AVBench-C 人评画质 2.86 vs 2.59（Tab 17）。

## 4. 局限
- 作者承认：机器人 policy 只是 DROID 上的"pilot study"（Sec 4.2.5）；RoboArena 排名标注了日期、等待更多社区评测（Sec 6.2.5）；PSNR 只是短时程代理指标，合理的生成未必逐像素等于真值；LOME 基线因分布偏移表现差、RoboMIND Franka 子集只有 23 / 4 小时所以协同结论不稳；联合训练让 FD 略降（Tab 31）；"随训练推进仍需 domain-aware 采样 / 专门化"；自动指标在前沿饱和、PAIBench-G 公榜不可复现（脚注）、HUE 上真视频也拿不到 100%。全文没有 limitation 小节。
- 我读出来的（模型侧）：(1) 没有闭环 policy evaluation、MPC 或 best-of-N 规划实验——"world model 评估 policy"仍是口号，Cosmos Policy 的 value 头在这里没有对应物；(2) policy 部署是 2.1 s 开环动作块、时延未报，且要 4 步去噪 + CFG 双卡，无蒸馏；(3) 视频侧 50 步、720p 单卡 5 分钟量级，FD 作 what-if 工具远非实时；(4) 动作是位姿差分伪动作，落到真机要靠外部 IK / 关节控制器（DROID 版直接改成关节位置，说明伪动作接口并不"通用到部署"）；(5) 新本体要新建投影层再 post-train，没有零样本跨本体 policy 结果。
- 我读出来的（数据与评测侧）：(6) 无 latent action，不能像 Genie / LAPA / Motus 那样从无标注视频学动作，人手 41.3K h 是专有数据，相机动作靠外部位姿估计器打标；(7) 主要数据（人手、驾驶、多数视频）专有，预训练不可复现；(8) 评测里 Cosmos-Predict2.5 是唯一 Cosmos 前代对照，没有和 Cosmos 1、Cosmos Policy 比；(9) generator 训练时 reasoner 冻结，"理解与生成互相促进"只在 Tab 28 一个方向上被验证。

## 5. 复现要点
- 开源：代码（cosmos + cosmos-framework，OpenMDW-1.1）、Nano / Super mid-train 基座、三个 post-train 权重、SDG 数据、Cosmos-HUE；推理栈支持 vLLM-Omni（Cache-DiT、Ulysses CP、CFG 并行、HSDP、CPU offload、FP8）和 TensorRT-LLM / vLLM（reasoner）。训练代码是否含 mid-training / policy post-training 脚本未核实。
- 算力（论文）：预训练 / mid-training 用 1024–2048 张 GB200；policy post-training、LIBERO 适配和 Tab 18 各 post-train 的 GPU 数与步数（LIBERO 只到 2000 步）均未说明；I2V 后训 10k 步；policy 推理 2× RTX Pro 6000。
- 8×H100 判断（估算）——**推理可行**：Nano 16B bf16 约 32 GB，Fig 16 就是单卡 H100 NVL 跑 720p T2V；Super 64B 约 128 GB 权重需 ≥2 卡 HSDP 或 FP8 / offload。
- 8×H100 判断（估算）——**Nano post-training 可行但紧**：若只训 generator 塔 8B + 动作投影（论文未明说 policy 后训是否也冻结 reasoner），参数 + 梯度 + Adam 约 128 GB，FSDP 分 8 卡约 16 GB/卡，policy 序列短（540×640 画布 + 32 动作 token）激活可控；全 16B 可训则约 32 GB/卡，仍可行但要激活检查点；74K 上下文的视频 FD 后训需要 context parallel。**Super post-training 不可行**（约 1 TB 优化器状态，论文未用 LoRA）。
- 坑：提示必须是训练分布的结构化 JSON（含 duration / fps / 分辨率 / 宽高比 / framing / idle_frame，App B.5），要先过 prompt upsampler；多视角要按 Fig 30 拼画布并在 JSON 里写清布局；新本体 = 新域 id + 从头初始化的投影层，动作按训练集统计归一到 [−1, 1]；FD / ID 用 50 步、CFG 1，policy 用 4 步、CFG 3、shift 5（Tab 21）；720p 只能 batch 1；AR / DM 之间 15000 位置间隔和 ×10 动作损失是隐含超参；DROID 需社区空闲帧过滤和失败演示剔除；控制器要能接 32 步 @15 Hz 关节位置开环块。

## 6. 关键引用链
- 建立在：Cosmos 1（Cosmos-1，WFM 范式与数据管线）→ Cosmos-Predict2 / 2.5（视频侧唯一前代对照）；架构上是 Transfusion / Mixture-of-Transformers（Liang et al. 2024）/ BAGEL 一路，backbone 取 Qwen3-VL，VAE 取 Wan2.2；动作表示引 X-VLA（域投影）、LDA-1B、EgoVLA。
- 基线与相关工作：Ctrl-World（arXiv:2510.10125，FD 基线）、DreamZero（arXiv:2602.15922，RoboLab 基线，且其仿真操作片段进了 SDG-RobotSim）、GR00T N1 / N1.6（arXiv:2503.14734）、π0 / π0.5；相关工作引了 GAIA-1（arXiv:2309.17080）、Genie 1 / 2 / 3（Genie）、UniSim（arXiv:2310.06114）、DreamGen（arXiv:2505.12705）、DreamDojo（arXiv:2602.06949）、V-JEPA 2、LeWorldModel、Motus（arXiv:2512.13030，作为"具身 MoT 式扩展"引用）。
- **未引用**：Cosmos Policy（Cosmos-Policy，同公司同思路却未提，本笔记的对照是我加的）、LingBot-VA（arXiv:2601.21998，同期"统一视频-动作"思路相近，论文未引用 / 未核实）、UniPi、DINO-WM、HarnessWAM（HarnessWAM）。
- 后续：2026-06 之后的工作本地暂无。
