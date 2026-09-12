# Real-Time Execution of Action Chunking Flow Policies (RTC)

> 对应「可lip」VLA 入门第 24 期（四）视频。论文：[arXiv:2506.07339](https://arxiv.org/abs/2506.07339)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2506.07339（v1 2025-06-09，本地 PDF 为 v2 2025-12-05；NeurIPS 2025）· 机构: Physical Intelligence / UC Berkeley（Kevin Black、Manuel Galliker、Sergey Levine）· 代码/权重: 仿真 benchmark 与算法代码开源 github.com/Physical-Intelligence/real-time-chunking-kinetix；真机 runtime 与 π0.5 数据不公开 · 项目页: https://pi.website/research/real_time_chunking
- 一句话: 把"边执行当前 chunk 边生成下一个 chunk"变成一个 inpainting 问题——冻结推理延迟内必然执行的前几步动作，用 guidance 让新 chunk 与旧 chunk 平滑衔接——无需重训即可用于任何 diffusion/flow VLA，在 +200 ms 注入延迟下仍不掉点。

## 1. 要解决的问题

VLA 有几十亿参数，推理时间 δ 远大于控制周期 Δt（π0：4090 上仅 KV prefill 就 46 ms，而 50 Hz 的 Δt = 20 ms；远程推理还有网络延迟；OpenVLA 优化后仍 321 ms）。同步推理（执行完 s 步停下来等下一个 chunk）会产生停顿，既慢又改变了动力学、造成训练-部署分布偏移。朴素异步（提前 d 步开始推理、新 chunk 一到就切换）会在切换点出现不连续：相邻 chunk 可能选了动作分布里不同的模式（Fig. 2 的分岔示意），产生 OOD 的巨大加速度。ACT 的 temporal ensembling（平均多个 chunk）不保证平均后的动作有效，反而更糟。现有加速方法（蒸馏、并行解码）不可能把成本压到一次前向以下，所以异步是必需的。

## 2. 方法

**记号。** 预测 horizon H，执行 horizon s ≤ H，推理延迟 d = ⌊δ/Δt⌋（控制步数）。异步要求在第 s−d 步开始推理，只要 d ≤ H − s 就保证动作不断供。

**Inpainting（Sec. 3.1）。** 生成新 chunk 时，前 d 步在新 chunk 可用前已经执行，应冻结为旧 chunk 的值；剩余部分要与这个冻结前缀一致地"补全"。采用无需训练的 ΠGDM 图像 inpainting（Pokle et al. 2023；Song et al. 2023）：每个去噪步在速度场上加梯度修正
v_ΠGDM = v + min(β, (1−τ)/(τ·r²_τ)) · (Y − Â¹)ᵀ diag(W) ∂Â¹/∂A^τ，
其中 Â¹ = A^τ + (1−τ)v 是当前对最终 chunk 的一步估计，Y 是旧 chunk（右侧 pad 到 H），W 是权重掩码，r²_τ = (1−τ)²/(τ²+(1−τ)²)。梯度项是 vector-Jacobian product，用反向自动微分算。β 是作者新增的 guidance 权重裁剪：去噪步数很少（n=5）时不裁剪会发散，取 β=5（Appendix A.2）。

**Soft masking（Sec. 3.2）。** 只冻结前 d 步时 guidance 太弱（尤其 d 小），新 chunk 仍会换策略。于是把所有 H−s 个重叠步都用上：W_i = 1（i<d）；c_i·(e^{c_i}−1)/(e−1)，c_i = (H−s−i)/(H−s−d+1)（d ≤ i < H−s，指数衰减）；0（i ≥ H−s，超出旧 chunk 的部分自由生成）。

**完整系统（Alg. 1）。** 控制器每 Δt 调 GetAction 消费一个动作并交新观测；后台线程跑 InferenceLoop：等执行满 s_min 步就以当前观测和旧 chunk 剩余部分启动 GuidedInference；用最近 b=10 次延迟的最大值保守估计 d；实际执行 horizon s = max(d, s_min)，每个 chunk 可不同。开销：每个去噪步要反传一次，π0.5 上单步 14 ms → 35 ms，总模型延迟 76 → 97 ms（Table 3）。

## 3. 实验

**仿真（Sec. 4.1，Fig. 5）。** Kinetix 里 12 个动态任务（投掷、接球、平衡、足式等），力控、无法"停住"，加动作噪声迫使闭环纠正。专家用 RPO 训 6 个种子，每环境 1M 转移；flow 策略 H=8、4 层 MLP-Mixer；2048 次 rollout/数据点；延迟 0–4。Baseline：naive async、BID（拒绝采样，N=32, K=3）、TE。结论（曲线估读）：平均 solve rate 在 d=0 时各方法 ≈0.9（TE 略低），d=4 时 RTC ≈0.75、BID 略低、naive 与 TE ≈0.45；hard masking 在小 d 时略差于 soft；只有 RTC 和 BID 随执行 horizon 缩短而单调变好。BID 计算量大得多（每次采 64 个 chunk）。

**真机（Sec. 4.2，Fig. 6）。** π0.5，双臂 6-DoF，H=50，Δt=20 ms，n=5，模型延迟 76 ms（baseline）/97 ms（RTC），LAN 再加 10–20 ms → d≈6；另注入 +100/+200 ms 模拟更大模型或云端推理（d≈11/16）。6 个任务：light candle（划火柴点蜡烛，5 步 40 s）、plug ethernet（6 步 120 s）、make bed 移动（3 步 200 s）、shirt folding（1 步 300 s）、batch folding（4 步 300 s）、dishes in sink 移动（8 步 300 s）。共 480 个 episode、28 小时纯执行。Baseline：synchronous（s=25 后暂停）、TE sparse、TE dense。指标：平均 throughput = 完成比例 / episode 时长。估读：RTC 在 +0/+100/+200 ms 约 0.83 / 0.78 / 0.87 任务/分钟，无退化；synchronous ≈0.72 / 0.58 / 0.5 线性下降；两种 TE 在 +0 ms 约 0.68，在 +100/+200 ms 因震荡触发机器人保护停机而无法运行。+100 和 +200 ms 处 RTC 的优势统计显著。去掉推理停顿后按控制步数看，RTC 仍更快完成（更少重试）；light candle（唯一不能重试的精度任务）和 bed making 上最终得分优势最大。

**延迟表（Appendix A.3）。** RTC 97 ms；BID N=16 依实现 115/169/223 ms；完整链路 RTC 移动平台 139 ms（网络 21 ms、CPU 缩图 11 ms）、固定平台 109 ms。

## 4. 局限

作者承认：计算开销显著（每去噪步 2.5×）；只适用于 diffusion/flow 策略；真机没做足式等更动态的场景。我读出来的：(1) 冻结前缀依赖延迟估计，估计偏小会断供、偏大会牺牲反应性；(2) guidance 本质是让新 chunk 迁就旧 chunk，当旧 chunk 真的错了（例如高层指令刚切换）会延缓纠正；(3) 对自回归 VLA（π0-FAST）无效；(4) 后续的训练时 RTC（Black et al. 2025，arXiv 2512.05964）已把这一开销转移到训练阶段，MEM 和 π0.7 都改用了它。

## 5. 复现要点

- 仿真部分完全开源（Kinetix benchmark、专家训练、flow 策略、所有 baseline），资源需求小：专家训练 4 h/4×H100，策略训练 1.5 h/2×H100（Appendix A.7）。
- 真机部分：算法只需一个 flow/diffusion 策略 + 能对去噪函数做 VJP 的框架（JAX/PyTorch 自动微分即可）；openpi 后续是否内置 RTC 未核实。作者的 π0.5 微调约 24 h/8×H100；推理单张 RTX 4090（bf16，5 步）。
- 超参（Table 5）：n=5，H=50，s_min=25，β=5，b=10。
- 坑：β 必须裁剪（τ=0 处权重无穷）；n 很小时高 β 会发散；延迟估计用 max 而非均值；需要真正的多线程异步推理框架，控制器与推理线程共享状态要加锁。

## 6. 关键引用链

**建立在：** ACT（action chunking 与 temporal ensembling）、Diffusion Policy（chunk 执行策略）、π0 / π0.5（基座与延迟数据）；ΠGDM（Song et al. 2023）与 Pokle et al. 2023（无训练 inpainting）；Diffuser（Janner et al. 2022，早期扩散 inpainting 规划）；BID（Liu et al. 2024，最接近的对照）；Kinetix（Matthews et al. 2024）；MPC 的 warm-start 思想。
**后续：** Training-time action conditioning for efficient real-time chunking（Black, Ren, Equi, Levine 2025，arXiv 2512.05964）；MEM 与 π0.7 都用（推理时或训练时的）RTC 做异步推理，π0.7 训练时模拟 0–12 步延迟。
