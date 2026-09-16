# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

> 对应「可lip」第 28 期视频（世界模型系列第二期）。论文：[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2506.09985（2025-06-11）· 机构: FAIR at Meta；Mila – Quebec AI Institute / Polytechnique Montréal · 代码/权重: https://github.com/facebookresearch/vjepa2（MIT 为主；发布 ViT-L/H/g@256 与 ViT-g@384 编码器、ViT-g 的 V-JEPA 2-AC 检查点、能量地形示例 notebook；HF 如 facebook/vjepa2-vitg-fpc64-256；repo 内另有后续 V-JEPA 2.1 权重）· 项目页: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks
- 2026-09-14：；开源代码 HEAD 204698b。
- 一句话: 先在 100 万小时互联网视频上把 V-JEPA 的"表征空间 mask-denoising"预训练扩到 1B 参数（ViT-g），再冻结编码器、只用 Droid 里不到 62 小时的无标注机器人视频后训练一个 300M 的动作条件预测器 V-JEPA 2-AC，用"目标图像 + CEM"零样本在两个实验室的 Franka 上完成 reach / grasp / pick-and-place；同时编码器在动作识别、动作预期和 VidQA 上做到 SOTA。它是"web 视频预训练 latent WM + 少量交互数据 → 真机规划"的代表。

## 1. 要解决的问题
- 交互数据稀缺：从 state-action(-reward) 序列从零学的 WM（Dreamer、TD-MPC 一类）是任务特定的、难以扩展。
- 用互联网视频 + 交互数据训练的动作条件视频生成 WM（GAIA-1、UniSim、Genie、Cosmos）主要评视觉保真度，"用生成视频来规划"算力太贵，几乎没有真机控制结果。
- 像素级生成目标把容量浪费在不可预测的细节上（草叶、树叶）；JEPA 只预测可预测部分。
- VLA / BC 需要成功的专家遥操作数据，没有显式的预测模型，也不用推理时算力做规划。
- 目标：动作无关的 web 视频预训练 + 少量交互数据（成功与失败都可用）→ 在新环境零样本规划。

## 2. 方法
**阶段一：V-JEPA 2 预训练（无动作）。**
- 目标函数（式 1）：min ‖P_φ(Δ_y, E_θ(x)) − sg(E_θ̄(y))‖₁，x 是随机 mask 掉 patch 的视频，Δ_y 是可学习的 mask token，目标由 EMA 教师编码器给出，损失只算被 mask 的位置。这就是"在表征空间做 mask-denoising"：不重建像素，只预测特征。
- 架构：encoder 与 predictor 都是 ViT；tubelet 2×16×16；3D-RoPE（把特征维分成 t / h / w 三段分别旋转，对最大模型的稳定性关键）。encoder ViT-L 300M / ViT-H 600M / ViT-g 1B（宽 1408、40 层、22 头，Table 12），predictor 固定为 ViT-s 22M。
- 数据 VideoMix22M：2200 万样本、超 100 万小时（SSv2 168K、Kinetics 733K、HowTo100M 1.1M、经 DINOv2 聚类检索式清洗的 YT-Temporal-1B 19M、ImageNet 1M 图像复制成 16 帧视频；Table 1）。
- 训练：warmup 12K + 恒定 lr 228K + cooldown 12K 步，batch 3072，主阶段 16 帧@256@4fps，cooldown 阶段升到 64 帧、256 / 384 / 512 分辨率（Table 9）。渐进分辨率使 GPU 时间减少 8.4×（直接以 64×384×384 训练约需 60 GPU-years，Sec 2.4）。四个 scaling 因素累计 +4.0 点（数据 +1.0、模型 +1.5、训练时长 +0.8、分辨率 / 时长补齐到 88.2，Sec 2.2）。

**阶段二：V-JEPA 2-AC 后训练（动作条件 latent WM，Sec 3.1、附录 11）。**
- 预测空间：state-space latent。冻结的 ViT-g 当"图像编码器"逐帧编码，z_k = E(x_k) ∈ R^{16×16×1408}（256×256 输入）。
- 数据：Droid 原始视频，只取左侧外部固定相机，4 秒片段@4fps = 16 帧，不足 4 秒的丢弃，剩下 <62 小时、约 2.3 万条轨迹（脚注 1），成功与失败都用，不用任务标签 / reward。状态 s_k 为 7 维末端位姿（xyz + 欧拉角 + 夹爪），动作 a_k = s_{k+1} − s_k 是 7 维末端增量——动作抽象层级 = low-level robot action。同时用左右两相机训练但不给相机位置条件会变差（11.1）。
- 预测器 P_φ：约 300M transformer（24 层、16 头、隐层 1024）；动作、状态、patch 特征各自过仿射层映射到隐层，序列按 (a_k, s_k, z_k) 交错；block-causal attention——第 k 步的 patch 可以看到同一步的动作 / 状态 token 和之前所有步；patch 用 3D-RoPE，动作 / 状态 token 只加时间 RoPE。
- 损失（式 2–4）：teacher-forcing L1（T = 15 个下一帧预测）+ rollout 损失（把预测喂回去，T = 2，只穿过一步递归求导），后者用于抑制自回归误差累积。优化：AdamW、wd 0.04、lr 7.5e-5→4.25e-4（4500 步 warmup）恒定 85500 步再 4500 步衰减，batch 256（11.1）。
- 动作怎么"出来"——规划而非 policy（3.2，式 5）：能量 E(â_{1:T}) = ‖P(â_{1:T}; s_k, z_k) − z_g‖₁，即想象 T 步后的特征与目标图特征的 L1 距离；用 CEM（Cross-Entropy Method：从高斯采样动作序列、取 top-k 精英更新均值方差、迭代）最小化，只执行第一个动作后重规划（receding horizon）。实际配置：800 样本、10 轮、top-10、horizon = 1（11.2）；每个动作限制在 L1 球半径 0.075 内（末端最多约 13 cm）；blocking control。pick-and-place 给 3 张目标图（抓住、接近放置点、放好），按 4 / 10 / 4 步的固定日程切换子目标。
- cascaded / joint：都不是——纯 WM + 优化器；论文列了未来可"在想象中训一个前馈 policy 来初始化规划"。
- 推理频率与延迟：单张 RTX 4090 上每个动作 16 s（Table 3），相当于 <0.1 Hz 的准静态控制。
- 可视化用 decoder（11.3）：在 Droid 上以 MSE 训一个 ViT-L 前馈帧解码器（150K 步、batch 1024），仅用于解释预测，不参与训练 / 规划。


## 3. 实验
**规划（Sec 4）。** Setting：两个实验室的 Franka Panda + Robotiq 夹爪，未标定的低分辨率单目 RGB，模型权重和推理代码完全相同，零样本、不采任何本地数据。baseline：Octo-base-1.5（OXE 1M+ 轨迹预训练，再在全量 Droid 上用 hindsight 目标图像重标注做 BC）；Cosmos-Predict1 7B latent diffusion（动作条件微调于 Droid）。
- 单目标 reach（Fig 8）：三个方向上末端都能到 4 cm 以内且误差单调下降——实质是靠无标注视频学出来的 visual servoing；能量地形（Fig 9）平滑、局部凸，最小值靠近真实动作。
- Table 2（两实验室均值，每任务 10 次）：Reach 100% vs Octo 100%；Grasp 杯 65% / 盒 25% vs Octo 15% / 0%；Reach-with-object 杯 75% / 盒 75% vs 15% / 70%；Pick-and-place 杯 80% / 盒 65% vs 15% / 10%。
- Table 3（Lab 2）：Cosmos 80 样本×10 轮、horizon 1 需 4 分钟 / 动作（一次 pick-and-place 超 1 小时），Reach 80%、Grasp 0% / 20%、P&P 0% / 0%；V-JEPA 2-AC 用 10 倍样本仅 16 s / 动作，Reach 100%、Grasp 60% / 20%、P&P 80% / 50%。
- 相机敏感性（11.4，Fig 16）：推断出的动作坐标轴旋转误差几乎随相机角度线性变化，平均绝对误差约 1.6 cm（真实位移约 5 cm），W* 近似旋转矩阵——理论上可用随机动作做无监督标定，但实验未做。
- rollout 可视化（Fig 15）：能"动"机械臂并保持背景不变，闭合夹爪时杯子随臂移动、张开时杯子不动（物体恒常性）；末帧杯子位置略低于真实（误差累积）。

**理解 / 预测（简写）。** Table 4：ViT-g384 六任务平均 88.2，SSv2 77.3 vs InternVideo2s2-1B 69.7；Table 5：EK100 动作预期 recall@5 39.7 vs PlausiVL（8B）27.6，相对提升 44%；Table 8：接 Llama 3.1 8B 后 PerceptionTest 84.0、MVP 44.5、TempCompass 76.9、TemporalBench 36.7、TOMATO 40.3 均超 PLM 8B，但 TVBench / MVBench 不及；Table 6 冻结编码器对照下平均 52.3 vs PE 49.1、SigLIP2 48.1、DINOv2 45.7——不用语言监督的视频编码器也能对齐 LLM。

## 4. 局限
- 作者承认（4.3、9）：无标定所以对相机位置敏感，实验相机位置是手动试出来的；长程规划受误差累积和搜索空间指数增长限制，pick-and-place 必须人工给子目标图；只支持图像目标，不支持语言；预测最远约 16 s；模型只到 1B。
- 我读出来的：(1) 16 s / 动作 + blocking control，只能做准静态任务，离实时差两个数量级；horizon = 1 本质是特征空间里的贪心视觉伺服，任务是按"贪心可解"挑的；(2) 每任务只 10 次试验，盒子抓取 20–30% 说明精细夹爪控制仍弱；(3) 预测器是确定性 L1 回归，不能表达多模态未来；(4) 能量是整幅特征图的 L1 距离，背景 / 机械臂构型都计入，目标图必须来自同一相机；(5) 没有给 V-JEPA 2-AC 的定量预测误差，只有可视化；(6) 只有 Droid 风格的 Franka 单臂、固定外部相机；后训练用了多少 GPU 论文未说明；(7) Cosmos 基线因规划太贵只能用 80 个样本，比较并不完全公平。

## 5. 复现要点
- 开源：代码 + 全部编码器权重 + ViT-g 的 AC 检查点（GitHub / HF），含能量地形示例；完整的真机 CEM 部署代码是否包含未核实。
- 规模：编码器 1B（ViT-g）、AC 预测器 300M、可选帧解码器 ViT-L、预训练 predictor 22M。
- 预训练算力：batch 3072 × 252K 步；论文只给"全分辨率直训约 60 GPU-years、渐进训练省 8.4×"（Sec 2.4，Fig 5 中的具体 GPU-days 我无法从文本核实），推算约数 GPU-years 量级——8×H100 不现实（需数月以上），直接用发布权重。VidQA 对齐用了 128 / 512 张 H100（14.2 / 14.3），与机器人部分无关。
- AC 后训练：GPU 数量论文未说明。估计 8×H100 数天可完成——依据：94.5K 步 × batch 256 × 16 帧，每步需冻结 ViT-g 前向约 4096 帧（每帧 256 token，约 0.5 TFLOP）加 300M 预测器在约 4.1K token 序列上的前向反向，合计每步约几 PFLOP，总量 1e20–1e21 FLOP 量级；80 GB 显存下需梯度累积 / 分片。
- 推理：单张 4090 即可；每步要编码当前帧与目标图，再跑 800×10 次预测器。
- 坑：相机位置要扫（11.4 扫了 35°–85°）；只用 Droid 左相机；动作裁剪到 L1 半径 0.075；blocking 的 operational space 控制器；4 fps 意味着每个规划动作对应约 0.25 s 的运动；rollout 损失不可省；pick-and-place 需要子目标图；短于 4 s 的 Droid 片段被丢弃。

## 6. 关键引用链
- 建立在：JEPA（LeCun 2022）、I-JEPA（Assran 2023）、V-JEPA（Bardes 2024）的表征空间预测；DINOv2（Oquab 2023）的数据清洗方法；DINO-WM（Zhou 2024）与 PLDM（Sobal 2025）的 latent 规划（论文自称最接近的工作）；Droid（Khazatsky 2024）数据；CEM（Rubinstein 1997）；Cosmos（Agarwal 2025）、Octo（2024）作为 baseline；PerceptionLM（Cho 2025）的 VidQA 配方。
- 后续：官方 repo 已放出 V-JEPA 2.1 权重（80M–2B，细节未核实）；tutorial（arXiv 2607.00836）Section 2.2 将其列为 state-space / latent state（预测式）的代表。
