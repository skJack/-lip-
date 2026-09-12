# OpenVLA: An Open-Source Vision-Language-Action Model

> 对应「可lip」VLA 入门第 21 期（一）视频。论文：[arXiv:2406.09246](https://arxiv.org/abs/2406.09246)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2406.09246（v1 2024-06-13，CoRL 2024）· 机构: Stanford / UC Berkeley / TRI / Google DeepMind / Physical Intelligence / MIT · 代码/权重: 开源（MIT license），GitHub `openvla/openvla`，HF `openvla/openvla-7b` · 项目页: https://openvla.github.io
- 一句话: 第一个完全开源、在 970k 条 Open X-Embodiment 真机轨迹上训练的 7B VLA，超过闭源 55B 的 RT-2-X，并给出 LoRA 微调 + 量化推理的"消费级 GPU 也能用"配方；此后两年几乎所有开源 VLA 论文都拿它当 baseline。

## 1. 要解决的问题
- RT-2 / RT-2-X 等 VLA 闭源，架构、训练流程、数据混合都看不到。
- 没人研究过 VLA 怎样高效微调到新机器人/新任务（RT-2-X 的 API 甚至不支持微调）。
- 目标：给机器人领域一个像 Llama 之于 LLM 的开源基座，并给出微调 best practice。

## 2. 方法
- backbone（Sec 2.1）：Prismatic-7B VLM = 融合视觉编码器（SigLIP + DINOv2，共约 600M，patch 特征按通道拼接）→ 2 层 MLP projector → Llama 2 7B。Prismatic 在 LLaVA-1.5 混合（约 1M 图文）上训练。选 DINOv2 是因为其低层空间特征对控制有帮助。
- 输入/输出（Sec 2.2）：单张 224×224 第三视角图 + 语言指令 → 7 维动作（6-DoF 末端位姿增量 + gripper），每维离散为 256 bins；bin 边界用训练数据 1%–99% 分位数（不用 RT-2 的 min-max，避免离群值压缩精度）。动作 token 覆盖 Llama tokenizer 最后 256 个最少用的 token。loss 只在动作 token 上算 cross-entropy，标准 next-token prediction。没有 action chunk、本体状态和历史。
- 训练数据（Sec 2.3，Appendix A Table 3）：OXE 中"单臂末端控制 + 至少一个第三视角相机"的子集，混合权重沿用 Octo；共 970k episodes、27 个数据集（Bridge 13.3%、Fractal 12.7%、Kuka 12.7%、DROID 10%、BC-Z 7.5%、FMB 7.1% 等）。DROID 的动作 token 准确率一直上不去，最后 1/3 训练把它去掉了。
- 关键设计发现（Sec 2.4）：(1) 视觉编码器必须一起微调（冻结明显变差，Appendix Table 10：微调 80.0% vs 冻结 46.7%）；(2) 224 与 384 分辨率无差别但后者慢 3×；(3) 要训很多 epoch——27 epochs，直到动作 token 准确率 >95%；(4) 固定 lr 2e-5，不需要 warmup。
- 算力（Sec 2.5）：64×A100 训 14 天（21,500 A100-hours），batch 2048，150k steps。
- 推理：bf16 需 15 GB 显存，RTX 4090 上约 6 Hz；附远程推理 server。

## 3. 实验
- 直接评测（Sec 5.1，全部 A/B 同初始状态）：WidowX/BridgeData V2 17 任务 ×10 trials（视觉 / 运动 / 物理 / 语义泛化 + 语言 grounding，允许 0.5 部分分），Google robot 12 任务 ×5。
  - Bridge（Appendix Table 4）：OpenVLA 70.6±3.2% vs RT-2-X 50.6% vs Octo 20.0% vs RT-1-X 18.5%。RT-2-X 只在语义泛化子类更好（它 co-fine-tune 了 web 数据）。
  - Google robot（Appendix Table 6）：OpenVLA 85.0±4.6% vs RT-2-X 78.3% vs RT-1-X 33.3% vs Octo 26.7%。
  - 摘要里"比 RT-2-X 高 16.5%"是两平台 29 任务合并后的绝对提升。
- 微调到新机器人（Sec 5.2，Appendix Table 7）：Franka-Tabletop（5 Hz）6 任务 + Franka-DROID（15 Hz）擦桌，每任务 10–150 条演示，全参数微调。Tabletop 平均 OpenVLA 67.2% vs Diffusion Policy 48.5% vs Octo 43.4% vs OpenVLA(scratch，直接从 Prismatic 微调、不经 OXE 预训练) 43.4%；DROID 擦桌 58.3% vs DP 35.0%。DP 在窄的单指令任务更强（倒玉米 100%），OpenVLA 在多物体、需语言 grounding 的任务更强。
- 参数高效微调（Table 1）：全参数 69.7%（163 GB 显存，需 2 卡 FSDP）；LoRA r=32 68.2%（59.7 GB@batch 16，只训 1.4% 参数）；只训最后一层 30.3%；冻结视觉 47.0%。LoRA 单张 A100 10–15 小时。
- 量化（Table 2）：bf16 71.3% / 16.8 GB；int8 58.1% / 10.2 GB（掉分是因为 A5000 上只有 1.2 Hz，改成 blocking control 后 int8 74.4%，Appendix Table 11）；int4 71.9% / 7.0 GB。
- LIBERO（Appendix Table 12，LoRA 微调，每 suite 500 trials ×3 seeds）：平均 76.5%（Spatial 84.7 / Object 88.4 / Goal 79.2 / Long 53.7）vs Octo 75.1 vs DP 72.4。这组数字后来被 OpenVLA-OFT、π0、MolmoAct 反复引用。
- 消融（Appendix Table 9，8 个 Bridge 任务）：去掉 OXE 预训练只用 Bridge 训练 76.3% → 45.6%；再去掉 DINOv2 → 40.6%。数据多样性比双编码器重要得多。

## 4. 局限
- 作者承认（Sec 6）：只支持单图输入，没有腕部相机 / 本体状态 / 历史；自回归 7 token 太慢，跑不了 ALOHA 这种 50 Hz 系统；成功率普遍 <90%；VLM 大小、web 数据 co-training、视觉特征选择等问题没有算力做。
- 我读出来的：(1) 离散 bins + 单步动作导致轨迹抖、精细任务弱（论文自己承认 DP 更平滑）；(2) Bridge 数据要手工过滤全零动作，否则模型学会"冻住"（Appendix C）；LIBERO 也要过滤 no-op；(3) SimplerEnv 上零样本只有 27.7%（MolmoAct Table 1 转引），对相机 / 视觉分布很敏感；(4) OXE 里 DROID 学不进去，说明 7B 容量或混合权重不够。

## 5. 复现要点
- 完全开源：权重、训练代码（PyTorch，FSDP / FlashAttention / AMP）、微调 notebook、OXE 数据 loader、远程推理 server。
- 规模：7.5B 参数；推理 bf16 15–16 GB，int4 7 GB；RTX 4090 约 6 Hz。
- 8×H100：LoRA 微调轻松（单卡 60 GB@batch 16），全参数微调 2 卡起；从头复现预训练需约 21.5k A100-hours（8 卡约 110 天），不现实。
- 官方 benchmark 数字：LIBERO 76.5%（四 suite 平均）；Bridge 70.6%；Google robot 85.0%；SimplerEnv Google Robot VM 27.7%（转引）。
- 已知的坑：必须过滤 no-op / 全零动作；LIBERO 图像要旋转 180° 并重新以 256px 渲染（Appendix E）；int8 反而比 int4 慢；控制频率与训练数据（Bridge 5 Hz non-blocking）不匹配会掉分；动作反归一化统计量按数据集存储，换机器人要重算。

## 6. 关键引用链
- 建立在：RT-2（动作 token 化范式）、Prismatic VLMs（backbone 与代码库）、Open X-Embodiment / Octo（数据与混合权重）、LoRA / QLoRA。
- 后续：OpenVLA-OFT（并行解码 + chunk + L1）、ECoT、TraceVLA、SpatialVLA、CogACT、RoboDual、VLA-Cache、MolmoAct（同样用 256-bin 离散动作但改进 token 初始化），以及几乎所有 LIBERO 表格。
