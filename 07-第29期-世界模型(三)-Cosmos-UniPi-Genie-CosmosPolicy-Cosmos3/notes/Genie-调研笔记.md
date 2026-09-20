# Genie: Generative Interactive Environments (Genie)

> 对应「可lip」第 29 期视频（世界模型系列第三期）。论文：[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2402.15391（v1 2024-02-23；ICML 2024 best paper（论文正文未标注会议））· 机构: Google DeepMind（Jeff Clune 兼 UBC）· 代码/权重: 未开源——Broader Impact 一节明确写"不放出 checkpoint、训练数据和数据样例"，论文也没给代码链接；仅 App F 给了一个单 TPU 可跑的 CoinRun 小配方 · 项目页: https://sites.google.com/view/genie-2024/home（Fig 1 题注给出）
- 一句话: 用 30k 小时无动作、无文本标注的互联网 2D 平台游戏视频，训一个 11B 的"生成式交互环境"：video tokenizer 把帧压成离散 token，latent action model (LAM) 用 VQ 瓶颈从相邻帧之间无监督学出只有 8 个离散码的 latent action，MaskGIT 式 dynamics model 按 latent action 逐帧生成下一帧，用户可以拿任意一张图（Imagen 生成图、手绘草图、照片）当起始帧"玩"进去。它第一次证明**不需要任何动作标签也能从视频里学出逐帧可控的 world model**，并做到 foundation model 规模；"VQ 瓶颈 latent action + 动作条件 dynamics"这套配方直接催生了 LAPA（`arXiv:2410.11758`）、DreamDojo 的连续 latent action、Motus 的光流 latent action，以及 Genie 2 / Genie 3 这条"可玩世界模型"产品线。

## 1. 要解决的问题
- 已有 world model（Dreamer、IRIS、TWM 一类）都要环境里采来的 (观测, 动作) 对，动作标签把训练数据锁死在单个环境里；视频生成模型（Phenaki、Imagen Video）虽能吃互联网视频，但只有文本这种"视频级"条件，不能逐帧控制（Table 1：World Models = Video + Actions / frame-level；Video Models = Video + Text / video-level；Genie = Video only / frame-level）。
- 同期 GAIA-1、UniSim 把 WM 放大到了驾驶 / 机器人视频，但仍要动作 + 文本标签（Sec 4）。Playable Video Generation（PVG，Menapace 2021）已经用 latent action 控制视频，但只在单一领域的固定场景上，不能靠提示生成新环境。
- 核心问题：能不能只用视频，无监督地把"帧间变化"压成一个人能操作的小离散动作空间，并让模型在这个动作空间上可控、可扩展（scaling）、对未见图片可泛化？

## 2. 方法
**预测空间：observation-space，RGB 160×90，但在 VQ-VAE 离散 token 空间建模**（VQ-VAE = 把连续特征量化到一个有限码本、用最近的码索引表示的自编码器；tokenizer = 把图像 / 视频转成这种离散码序列的模块）。三个组件（Sec 2.1，Fig 3），全部用 ST-transformer：
- **ST-transformer（Fig 4）：**每个 block = 空间注意力层（只在同一帧的 H×W 个 token 间做 attention）+ 时间注意力层（只在 T 帧里同一位置的 token 间做 attention，带因果 mask）+ 一个 FFW；即把完整的时空注意力分解成"帧内"和"跨帧同位置"两步，主导计算量的空间层随帧数线性而非平方增长。作者只保留时间层后的 FFW、去掉空间层后的 FFW，省下的参数放到别处，"显著改善结果"（Sec 2）。
- **(1) Video tokenizer = ST-ViViT（Fig 6）：**VQ-VAE，encoder 和 decoder 都用 ST-transformer，所以 z_t 含有 x_{1:t} 的信息（因果时序压缩），不同于 CogVideo / MaskViT 的逐帧空间压缩，也比 Phenaki 的 C-ViViT（全时空注意力，随帧数平方增长）省算力。
- tokenizer 超参：200M 参数，patch size 4，码本 1024 个码、每码 32 维；encoder 12 层 d_model 512、decoder 20 层 d_model 1024（Table 7，扩 decoder 比扩 encoder 划算，App C.2）；AdamW lr 3e-4、300k 步（Table 8）。每帧 token 数论文没直接给，按 160×90 / patch 4 ≈ 40×23 ≈ 920 个，和 Sec 3.1 "batch 256 = 3.8M token"反推的 ≈ 930 token/帧一致（估算）。
- **(2) Latent action model, LAM（Fig 5）：**encoder 看 x_{1:t} 和 x_{t+1}，输出连续 latent action ã_{1:t}；经 VQ 码本量化，**码本只有 |A| = 8 个码**、每码 32 维；decoder 只拿 x_{1:t} 和 ã_{1:t} 重建 x̂_{t+1}，用 VQ-VAE 目标训练。信息瓶颈就在这里：decoder 看不到 x_{t+1}，ã_t 必须编码"过去到未来最有意义的变化"（latent action = 这样学出来的、解释帧间变化的紧凑离散变量）。
- LAM 超参与接口：直接吃像素（patch 16），300M 参数，encoder / decoder 各 20 层 d_model 1024、16 头（Table 5）；输入归一到 [0,1]、输出过 sigmoid（App C.1）。**推理时除码本外整个 LAM 都扔掉**，用户直接给 [0, 8) 的整数索引码本；decoder 只为给 LAM 提供训练信号。码本更大指标更好但人更难玩（App C.1）。
- **(3) Dynamics model（Fig 7）：**decoder-only 的 MaskGIT ST-transformer（MaskGIT = 训练时随机遮盖部分 token、推理时多轮并行填充被遮 token 的非自回归解码）。输入 z_{1:t-1} 和 stopgrad 的 ã_{1:t-1}，一次性预测所有下一帧 token ẑ_{2:T}（因果 mask），损失 = 与真实 token 的 cross-entropy；训练时对 z_{2:T-1} 按 U[0.5, 1] 采样的 Bernoulli 比例随机 mask。
- **动作怎么进：latent action 作为 additive embedding 加到 token 上**（而不是 IRIS / TWM 那样把动作拼接成额外 token），作者说这提高了可控性（Sec 2.1）。最终 dynamics 10.1B：48 层、36 头、d_model 5120（Table 12）；lr 3e-5→3e-6、warmup 5k（Table 9）；bfloat16 + QK-norm 稳训练。
- **训练流程：**先训 tokenizer，再把 LAM（像素上）和 dynamics（token 上）一起训（co-train）；三个组件序列长度 16 帧、10 FPS。
- **推理（Sec 2.2，Fig 8）：**起始图 x_1 → tokenizer 得 z_1 → 用户选 a_1 ∈ [0, 8) → 查码本得 ã_1 → dynamics 用 25 步 MaskGIT、温度 2、随机采样出 z_2 → decoder 出图 → 循环；也可以给多帧提示。每个 latent action 的含义要靠试（作者比喻"学新手柄上的按钮"），但同一个码在不同起始图上语义一致（Fig 17：left / right / jump / no-op）。**约 1 FPS，不能实时**（Sec 5）；记忆窗口只有 16 帧。
- **动作怎么出：不出。**模型没有任何真实动作接口；要动作得走 App E.1 的两步：冻结 LAM 给目标环境视频打 latent action 标签训 policy π(a_t|x_t)，再用一小份带真实动作的专家片段建"latent → 真实动作列表"的字典 D，执行时 u_t ~ D[a_t]（字典与状态无关）。
- **训练数据 Platformers（Sec 3，App B.1）：**按标题关键词（2D platformer + speedrun / playthrough，排除 movie / unboxing）过滤公开互联网视频，切成 16 s、10 FPS、160×90 的片段，得 55M 段 ≈ 244k 小时；再用质量过滤器（团队手标 10k 段、约 10 人时，1–5 分；11M 的 ResNet18 二分类，5 = 好、1 = 差，删 2–4；按预测 + 置信度决定留否）留下 6.8M 段 ≈ 30k 小时（约原始 12%，论文说"略多于 10%"），580M 模型 FVD 61.4 → 54.8（Table 4）。
- **训练数据 Robotics（Sec 3）：**RT-1 的 ~130k 条演示 + 一份仿真数据 + QT-Opt（Kalashnikov 2018）的 209k 条真机 episode，**动作全部丢掉只当视频**；总时长论文未给。
- **参数量：**标题 11B = dynamics 10.1B + tokenizer 200M + LAM 300M（正文写总计 10.7B）；网站 demo 另训了一个更大的 decoder 输出 360p。
- **算力：**最终 dynamics batch 512、125k 步、256 TPUv5p、共 942B token、6.6×10^22 FLOPs（Sec 3.1，Table 12）。
- **控制用途：**只有 Sec 3.3 的 latent-action 行为克隆（见第 3 节）；没有规划、没有 MPC。


## 3. 实验
Setting：没有和任何外部 WM / 视频模型在同一数据上做定量对比，定量结果全是自身消融与 scaling；主模型 11B 只在 Platformers 上训，另在 Robotics 上训了 2.5B。
- **指标（Sec 3）：**FVD（Fréchet Video Distance，生成视频与真实视频在视频特征空间的分布距离，越低越好）衡量保真度；**Δ_t PSNR = PSNR(x_t, x̂_t) − PSNR(x_t, x̂'_t)** 衡量可控性：x̂_t 是用 LAM 从真实视频反推的 latent action 生成的帧，x̂'_t 是用随机采样的 latent action 生成的帧，差值越大说明动作对生成的影响越大；全文报 t = 4。
- **Scaling（Sec 3.1，Fig 9，Table 10 / 11）：**固定 tokenizer 和 LAM，dynamics 从 41M 到 2.7B（batch 256、200k 步、750B token），最终训练 loss 随参数单调下降，约 0.00248 → 0.00182（估读）；2.3B 模型 batch 128 / 256 / 448（1.9M / 3.8M / 6.6M token）loss 约 0.00186 → 0.00176（估读）。只报训练 loss，没有 FVD / ΔPSNR 随规模的曲线。
- **定性（Sec 3.2）：**全用分布外提示——Imagen2 生成图、手绘草图、真实照片（Fig 10、Fig 16）都能"动起来"；同一码在不同起始图上语义一致（Fig 17）；涌现出视差（前景比背景动得多，Fig 12）。
- **机器人（Sec 3.2，Fig 13、Fig 11）：**2.5B 模型、Platformers 上最优超参不变，Robotics 测试集 FVD 82.7；三个不同起始帧上重复同一 latent action 5 次，学到的动作一致且有语义：down / up / left（Fig 13）；还学到了物体形变（连续 10 步同一动作压扁薯片袋，Fig 11）。注意学到的只是机械臂的粗粒度整体运动，没有展示夹爪开合或精细操作。
- **用 latent action 训 agent——setting（Sec 3.3，Fig 14，App E）：**Procgen CoinRun（未在训练数据里出现的 2D 平台游戏）easy / hard 两档；专家数据来自 R2D2 agent；冻结 LAM 给专家视频打 latent action，训 π(a_t|x_t)（ST-ViViT encoder 12 层 d 512、序列长 4、batch 16、cross-entropy），再用 N 条带真实动作的专家样本建字典 D；在留出关卡上报 100 局的通关率、5 seed、95% CI；上界是用真实动作训的 oracle BC，下界是随机策略。
- **结果（Fig 15）：只要 200 条带动作样本就追平 oracle**：Easy oracle ≈ 55%、random ≈ 34%、LAM 策略 200 条时 ≈ 49%、400 条起 ≈ 52–55%；Hard oracle ≈ 51%、random ≈ 25%、LAM 策略 200 条 ≈ 49%、400 条 ≈ 53%（估读）。绝对水平都只有一半左右，oracle 本身很弱；作者的论证是 latent → 真实的映射不含观测信息，所以策略的表现说明 latent action 本身可迁移。
- **消融 Table 2（LAM 输入）：**Platformers 上 token 输入 2.3B FVD 38.8 / ΔPSNR 1.33 vs 像素输入 2.5B 40.1 / **1.91**；Robotics 1B 上 token 257.8 / 1.65 vs 像素 **136.4 / 2.07**——tokenization 丢了运动信息，LAM 必须看像素（注意 Platformers 两行参数量不同）。
- **消融 Table 3（tokenizer，~200M、patch 10、batch 128）：**纯空间 ViT 230M / 0.3 GB / FVD 114.5 / ΔPSNR 1.39；C-ViViT 225M / 1.6 GB / 272.7 / 1.37（过拟合，需强正则）；ST-ViViT 205M / 0.9 GB / **81.4 / 1.66**——时序压缩有用，但全时空注意力反而更差。
- **其他：**Table 4 数据质量过滤后 FVD 61.4 → 54.8；两处只给结论没给数——additive embedding 比拼接更可控（Sec 2.1）、去掉空间层后的 FFW 更好（Sec 2）。

## 4. 局限
- 作者承认（Sec 5、Broader Impact）：继承自回归 transformer 的毛病，会幻觉出不合理的未来；只有 16 帧记忆，长时程一致性差；约 1 FPS，离可交互的帧率还远；不放权重、数据。
- 我读出来的（模型与评测）：160×90、10 FPS 的分辨率对机器人操作太粗；动作空间只有 8 个无幅度、无语义标签的离散码，含义要靠试，且 8 这个数是为"人能玩"选的而不是为覆盖动作空间；latent action 只抓最显著的帧间变化，平台游戏里就是角色 / 镜头位移，机器人上就是手臂粗动作；11B 主模型没有任何定量指标（FVD / ΔPSNR 全来自 ≤ 2.5B 的消融模型），也没有外部 baseline；ΔPSNR 只看 t = 4，且依赖 LAM 自己反推的动作，有循环味道。
- 我读出来的（数据与迁移）：BC 迁移只在和训练数据同类型的 CoinRun 上验证，latent → 真实动作是无视状态的查表；Robotics 混了 RT-1、仿真、QT-Opt 三源数据没有消融；"文本提示"其实是外接 text-to-image 模型，Genie 本身完全不吃语言。

## 5. 复现要点
- 代码 / 权重 / 数据：全部未开源；Platformers 数据集不放出，无法原样复现；非官方复现未核实，本笔记不列。
- 算力（Table 6 / 10 / 12，App D）：最终 dynamics 10.1B 用 256 TPUv5p、batch 512、125k 步、6.6×10^22 FLOPs；scaling 系列从 41M（64 TPUv2、3 天、2.05×10^20 FLOPs）到 2.7B（256 TPUv3、16 天、6.91×10^21）；tokenizer 64 TPUv2（batch 64，PSNR 35.7）或 64 TPUv3（batch 384，PSNR 36.5）；大模型用 ZeRO stage-3 + tensor parallelism。推理显存 / 延迟只说了约 1 FPS。
- 可复现小配方（App F，"单个中档 TPU / GPU 一周内"）：CoinRun hard 模式随机策略，seed 0–10k、每关 1000 步 = 10M 转移；tokenizer enc / dec 各 8 层 d 512、1024 码、patch 4，batch 48 × 16 帧 = 768 图，16 GB 单 TPU 3 天跑 300k 步；LAM 8 层 d 512、**6 个码**；dynamics 12 层 d 512，batch 36 × 16 帧，200k 步，温度 1.0、25 步 MaskGIT（Table 15–17）。
- 8×H100 能不能跑：App F 这档单卡 16 GB 就够，8×H100 绰绰有余，还能把 dynamics 推到几百 M 做自己的 scaling；11B 主模型不可能——6.6×10^22 FLOPs 按 H100 bf16 约 40% 利用率折算约 2000 H100-天量级（估算），且没有数据。
- 坑：先训 tokenizer 再 co-train LAM + dynamics，dynamics 侧对 latent action 要 stopgrad；LAM 必须吃像素而不是 token（Table 2）；动作用 additive embedding 不要拼接；去掉空间层后的 FFW；bfloat16 + QK-norm；mask 比例 U[0.5, 1]；码本数是"指标 vs 可玩性"的权衡（App C.1）；数据质量过滤比数量重要（Table 4，只用了原始数据 ~12% 反而更好）；C-ViViT 会过拟合需要强正则（Table 3）。

## 6. 关键引用链
- 建立在（视频模型谱系）：VQ-VAE（van den Oord 2017）的离散 token 化；MaskGIT（Chang 2022）的并行 masked 解码；Phenaki（Villegas 2023，C-ViViT tokenizer）、TECO（Yan 2023）、MaskViT（Gupta 2023）这一支 token 化 transformer 视频模型；ST-transformer 取自 Xu 2020（交通流预测），ST-ViViT 的名字对着 Phenaki 的 C-ViViT 起（ViViT 原论文 Genie 没有直接引用）。
- 建立在（latent action 谱系）：ILPO（Edwards 2019，Imitating Latent Policies from Observation）、Rybkin 2019（Learning What You Can Do Before Doing Anything）、Ye 2022（Become a Proficient Player…）、LAPO（Schmidt & Jiang 2024，Learning to Act without Actions），Genie 是这条线第一次放到互联网规模；VPT（Baker 2022）用带标签数据训 IDM 再给互联网视频打真实动作，是 Genie 的对照面；PVG / Playable Environments（Menapace 2021 / 2022）的 latent-action 可玩视频。
- 建立在（world model 谱系）：Ha & Schmidhuber 2018、Dreamer（`Dreamer-v1`、`Dreamer-v2`、`Dreamer-v3`）、IRIS / TWM（transformer WM，动作拼接）；同期需动作标签的放大版 WM：GAIA-1（`arXiv:2309.17080`）、UniSim（`arXiv:2310.06114`）；像素空间动作条件预测的祖先 Visual Foresight（`arXiv:1812.00568`）。
- 后续：LAPA（`arXiv:2410.11758`）把 Genie 式 VQ latent action 预训练搬到机器人 / 人类视频上训 VLA；DreamGen（`arXiv:2505.12705`）用 LAPA 给生成视频打伪动作；DreamDojo（`arXiv:2602.06949`）从人类视频学连续 latent action；Motus（`arXiv:2512.13030`）用光流当 latent action 预训练；Cosmos（`Cosmos-1`、`Cosmos-3`）把 token 化视频 WFM 放大到 20M 小时并加真实动作条件；分类依据见 tutorial（arXiv 2607.00836）。
- 产品线：Genie 2（2024-12，3D 环境）、Genie 3（2025-08，24 fps 实时交互、可提示的世界事件）、Project Genie（2026-01）——均为博客 / 厂商报告，非论文，细节未核实。
