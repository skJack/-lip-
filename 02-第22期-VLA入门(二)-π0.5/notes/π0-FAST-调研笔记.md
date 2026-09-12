# FAST: Efficient Action Tokenization for Vision-Language-Action Models

> 对应「可lip」VLA 入门第 22 期（二）视频。论文：[arXiv:2501.09747](https://arxiv.org/abs/2501.09747)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2501.09747（v1 2025-01-16，本地 PDF 为 v1）· 机构: Physical Intelligence / UC Berkeley / Stanford（Pertsch、Stachowicz 为共同一作）· 代码/权重: 已开源——FAST+ 通用 tokenizer 在 HuggingFace `physical-intelligence/fast`（AutoProcessor 三行调用）；π0-FAST 权重在 openpi（`pi0_fast_base`、`pi0_fast_droid`）· 项目页: https://pi.website/research/fast
- 一句话: 用 DCT + BPE 把高频 action chunk 压缩成少量高信息量的离散 token，让自回归 VLA 也能学 50 Hz 灵巧任务，训练比 flow matching 版 π0 快 5 倍，并首次训出能零样本部署到新环境的 DROID 策略。

## 1. 要解决的问题

RT-2 / OpenVLA 的 action tokenization 是逐维、逐时间步的 256-bin 均匀分箱。作者用一个玩具实验（Sec. III，Fig. 3：预测插值 4 个随机点的三次样条）说明：采样率从 25 提到 800 步时数据本身没变，但 naive 分箱训出的自回归模型误差陡增，最后退化成"复制上一个 token"。原因是 next-token prediction 的学习信号正比于 token 的边际信息量，高频平滑信号的相邻 token 几乎冗余，信息量趋近于零。此外 1 秒的 50 Hz 双臂 chunk 要 700 个 token，训练慢、推理慢。OpenVLA 在低频的 Bridge/RT-1 上能训、在 15 Hz 的 DROID 上难以拟合，正是这个问题。π0 用 flow matching 绕开了它，但代价是训练收敛慢、语言跟随弱（Sec. VI-E 的观察）。

## 2. 方法

**FAST 流程（Sec. IV-B，Alg. 1）：**
1. 归一化：每个动作维度按训练集的 1%/99% 分位数映射到 [−1, 1]（对离群动作鲁棒，也方便跨 embodiment）。
2. 对每个动作维度单独做 DCT（离散余弦变换）：低频系数描述整体形状，高频描述突变。
3. 量化：系数乘 scale γ 再取整；γ 越小压缩越狠、越有损；矩阵变得稀疏。
4. 展平：按"列优先"——先把所有维度的最低频系数排在前面——因为自回归先预测决定整体形状的低频分量，rollout 更稳定。
5. BPE：在展平后的整数序列上训一个 BPE 词表，把大量的 0 和常见系数组合"压扁"成稠密 token。选 BPE 是因为词表大小固定、能直接覆盖到 VLM 词表里。

只有两个超参：γ = 10、BPE 词表 1024，作者称不敏感。所有操作可逆，解码快。BPE 是唯一需要"训练"的部件（几分钟）。

**FAST+ 通用 tokenizer（Sec. IV-C）：** 在约 100 万条 1 秒 action chunk 上训 BPE，数据覆盖单臂/双臂/移动机器人、joint / 末端世界系 / 末端相机系三种动作参数化、5–50 Hz（Appendix A 的表列了 30 个子集及权重），动作维度 pad 到 32。可当黑盒直接用于任何机器人的 1 秒 chunk。

**接入 VLA（Sec. V-A，Appendix C）：** 与 RT-2/OpenVLA 一样，用最少用的 VLM 文本 token 覆盖为动作 token；本体状态用 256 bin 分箱后当文本输入（输入侧用 naive 分箱没问题）。backbone 用 π0 的 PaliGemma-3B（去掉 action expert，纯自回归）或 OpenVLA（Prismatic 7B）。训练：1k 步 warmup 后恒定 LR 5e-5，AdamW(0.9, 0.95)，无 weight decay，梯度裁剪 1，EMA 0.999。推理贪心解码，双臂任务用温度 0.7（避免停在初始位）。

**推理成本：** π0-FAST 每个 chunk 要解码 30–60 个 token，且全部经过 2B 语言模型，4090 上约 750 ms；扩散版 π0 只需 300M expert 跑 10 步，约 100 ms（Sec. VI-E）。

## 3. 实验

**压缩率（Table I）：** 1 秒 chunk 的 token 数，naive → FAST：Bridge v2（7 维 5 Hz）35 → 20（1.75×）；DROID（7 维 15 Hz）105 → 29（3.6×）；Bussing（7 维 20 Hz）140 → 28（5.0×）；Shirt fold（14 维 50 Hz）700 → 53（13.2×）。规律：FAST 每条臂稳定在约 30 token/秒，与控制频率基本无关。

**Tokenizer 对比（Fig. 6，柱状图估读）：** 任务为 LIBERO 仿真、DROID（15 Hz）、table bussing（20 Hz UR5）、T-shirt folding（50 Hz ARX）。naive 分箱在 bussing 和 T-shirt 上接近 0；FSQ（学习式向量量化）约 22%；FAST 约 85% / 65%；FAST+ 约 80% / 67%。四任务平均：naive ≈18、FSQ ≈45、FAST ≈68、FAST+ ≈65（%）。结论：压缩类 tokenizer 都远好于 naive；FAST 不需要训练网络却不输甚至好于 FSQ；通用 FAST+ 与逐数据集训的 FAST 持平。

**FAST+ 泛化（Fig. 8）：** 在 13 个训练时没见过的数据集（含灵巧手、UMI、人形、Waymo 驾驶，Table III）上压缩率均 ≥2×，人形/灵巧臂上可达约 10–14×（估读）。

**消融（Sec. VI-D）：** OpenVLA + FAST+ 在 T-shirt folding 上从接近 0 提到约 58%（估读），说明方法与 backbone 无关；去掉 BPE 仍好于 naive 但明显变差，因为大量 0-token 稀释了学习信号并拖慢解码。

**vs 扩散 π0（Fig. 9）：** 小数据（LIBERO、T-shirt，<50 h）两者相当；大数据 table bussing 上 FAST 用 1/3 的训练步数达到高性能；DROID 零样本上 FAST ≈58% vs π0 ≈20%（估读），作者归因为扩散版 π0 经常忽略语言指令。

**Generalist（Fig. 11、Fig. 15）：** 在 π0 的全部混合数据（903M 步 + 9.1% 开源）上训 π0-FAST，5 个任务平均与扩散 π0 持平（估读均 ≈68%），含最难的 laundry folding；但 GPU 时数少 5 倍；compute-matched 的 π0 明显更差。

**DROID 零样本：** 16 类任务 44 次试验的定量评测（Table II），并在三所大学的真实场景做定性测试（Fig. 7），能做 pick-place、开关抽屉、开水龙头；失败案例也"行为合理"。这是首个不做 co-training / fine-tuning 就零样本部署的 DROID 策略。

## 4. 局限

作者承认：推理慢（750 ms/chunk）；只在静态机械臂上做了策略实验，移动/灵巧手/人形只验证了压缩率；语言跟随为何自回归优于扩散留待研究；低保真区间 FSQ 的压缩效率更高（Fig. 12）。我读出来的：(1) DCT 假设动作在 chunk 内平滑，抓握开合这类阶跃信号需要更多高频系数；(2) 离散化损失由 γ 控制，不是无损；(3) 750 ms 的延迟意味着 π0-FAST 不适合动态任务——这直接推动了 π0.5 "离散 token 预训练 + flow expert 后训练"的混合方案。

## 5. 复现要点

- FAST+ tokenizer 开源（HF `physical-intelligence/fast`，`tokenizer.fit()` 可在自有数据上重训）；π0-FAST base 与 DROID checkpoint 在 openpi。
- 3B 模型；论文给出的参考成本：DROID 上 75k 成功 episode（21M 样本）、240k 步、batch 256，约 4 天 / 8×H100（Appendix D）。8×H100 上 fine-tune 和推理均可行。
- 坑：(1) 输入必须先按 1%/99% 分位归一化再喂 FAST+；(2) 一次只 tokenize 1 秒 chunk；(3) DROID 训练要过滤全零动作的 idle 步、只用 success episode；(4) 自回归解码要处理非法 token 序列的兜底；(5) 双臂任务需温度采样才会离开初始位。

## 6. 关键引用链

**建立在：** π0（backbone 与数据）；RT-2 / OpenVLA（naive 分箱与词表覆盖法）；BPE（Gage 1994；Sennrich et al. 2015）；DCT / JPEG；VQ-VAE、FSQ（学习式压缩对照）；DROID、Bridge v2、OXE、LIBERO。
**后续：** π0.5 预训练全用 FAST token；Knowledge Insulation（Driess et al. 2025）用 FAST token 监督 backbone、flow expert 停梯度，成为 π0.6 / MEM / π0.7 的标准配方。
