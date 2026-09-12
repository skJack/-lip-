# MEM: Multi-Scale Embodied Memory for Vision Language Action Models

> 对应「可lip」VLA 入门第 24 期（四）视频。论文：[arXiv:2603.03596](https://arxiv.org/abs/2603.03596)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2603.03596（v1 2026-03-04，本地 PDF 为 v2 2026-03-08）· 机构: Physical Intelligence / Stanford / UC Berkeley / MIT（Torne、Pertsch 共同一作）· 代码/权重: 未开源 · 项目页: https://pi.website/research/memory
- 一句话: 给 π0.6 加两级记忆——短期用改造 ViT 注意力得到的 video encoder 压缩几秒到一分钟的稠密观测，长期用高层策略自己维护的、经压缩的自然语言摘要——使 VLA 能完成需要记住 15 分钟内事件的厨房任务，并能利用刚失败的尝试在上下文内改变操作策略。

## 1. 要解决的问题

主流 VLA（π0/π0.5/π0.6）只看当前帧。长程任务要两种完全不同的记忆：短期稠密记忆解决遮挡、判断动力学、换抓取方式（需要图像，但只需几秒）；长期记忆只需几 bit 的语义（"已经加过盐""橱柜开着"）但要维持十几分钟。把所有历史帧塞进上下文不可行——图像编码占 VLA 计算的大头，Fig. 4 显示帧数一多延迟就超过 RTC 论文给出的实时阈值。已有方案各有取舍：只用本体历史丢掉环境信息，keyframe 稀疏化丢掉动力学，pool 成一个 token 压缩过猛，纯语言记忆丢失空间细节。此外给策略加历史常引入 causal confusion（抄上一步动作）导致性能反而下降。

## 2. 方法

**分解（Sec. III-A）。** π(a_{t:t+H}, ℓ_{t+1}, m_{t+1} | o_{t−T:t}, m_t, g) ≈ π_LL(a | o_{t−K:t}, ℓ_{t+1}, g)·π_HL(ℓ_{t+1}, m_{t+1} | o_t, m_t, g)：低层看短窗 K ≪ T 的观测和 subtask ℓ_{t+1}；高层看当前观测、任务 g 和语言记忆 m_t，输出下一个 subtask 和**更新后的记忆** m_{t+1}。新颖点在于高层自己预测记忆更新。

**语言记忆（Sec. III-B）。** m_t 是过去语义事件的摘要（"我把盘子放进了柜子并移到台面" → "…并拿起了一个碗"）。训练标签的生成：把 episode 的 subtask 标注 + 每段成功/失败标志喂给现成 LLM，让它输出"对未来仍相关"的最小摘要，并显式要求压缩和删除（三个不同颜色的碗 → "三个碗放进右上柜"）。压缩的好处：推理快、跨步传递的信息少所以训练-推理分布偏移小；尤其失败的重复尝试不会写进记忆。

**Video encoder（Sec. III-C，Appendix C）。** 在 ViT 上每 4 层插入一次因果的时间注意力：同一 patch 位置跨时间步做 attention（时空可分离，复杂度从 O(n²K²) 降到 O(Kn² + nK²)），其余层仍是帧内空间注意力。不引入新参数，只加一个正弦时间位置编码且 e(0)=0，所以 K=1 时与原 VLM 完全一致，可直接用预训练 ViT 权重初始化。最后只把当前帧的 patch 表示送入 VLA backbone，token 数与无记忆模型相同——时间信息被"挤"进当前帧表示。本体状态历史用线性投影成 K 个连续 token（π0.6 原本用文本 token，历史会爆炸）。

**接入 π0.6（Sec. III-D）。** Gemma3-4B 初始化，860M flow expert，KI 配方（FAST token + flow，expert 停梯度），448×448，最多 4 路相机。预训练用 6 帧（5 历史 + 当前，步长 1 s）；post-training 可扩到 18 帧 / 54 s（类似 LLM 的长上下文扩展）。预训练数据：遥操作示范、策略 rollout、人类纠正（同 π*0.6）、图文任务、视频-语言任务（视频描述）。真机推理用 RTC（推理时或训练时版本）。

## 3. 实验

每策略每任务 10 次，报均值 ± 标准误。

**长程任务（Sec. IV-A，Fig. 6）。** Recipe setup：按 prompt 从冰箱/柜子/抽屉取齐 6–7 件食材器具放到指定位置并关门，训练 42 个食谱、在未见厨房和物体上评 5 个；Clean up kitchen：擦台面、收食物进冰箱、洗碗放架，约 8 个子任务。估读平均 task progress：π0.6 无记忆 ≈32%，只有 video 记忆 ≈34%，naive（拼接全部历史 subtask）+ video ≈37%，只有语言记忆 ≈30%，MEM ≈72%；clean kitchen 上 ≈22% vs ≈88%。分析：无 video 记忆时不知道洗了多久、会"卡住"；无语言记忆记不住食谱进度和该关的门；naive 语言记忆差的原因是分布偏移——示范里每条 subtask 只出现一次，部署时反复失败会产生"pick up bowl ×3"的序列，而 MEM 的记忆在成功前根本不更新。

**In-context 适应（Sec. IV-B，Fig. 7）。** 用 π*0.6 的方式收集失败后的人类纠正，微调时把失败尝试留在短期记忆里。夹筷子（OOD 桌高）：无记忆 ≈78% → 有记忆 ≈90%（图中标注 +11%，估读）；开冰箱（门轴方向不明，≤4 次抓取算成功）：≈30% → ≈90%（+62%）。无记忆的策略会重复同一错误。

**记忆方案对比（Sec. IV-C，Fig. 8）。** 六个记忆任务：swap 3 mugs、find object（人把物体放进 4 个抽屉之一）、unpack groceries（袋内不可见）、scoop coffee（恰好两勺）、grilled cheese（计时翻面）、window cleaning（记住擦过哪）。对照均在 π0.6 上重实现且去掉语言记忆：Pool Memory（历史帧编码后平均池化成一个 token，类 ContextVLA）、Proprio Memory（只用本体历史，类 TA-VLA）。估读平均：无记忆 ≈28%、Pool ≈34%、Proprio ≈38%、MEM ≈73%；find object 无记忆 ≈30%（接近 25% 的随机水平）vs MEM ≈88%；scoop coffee ≈52%（随机 50%）vs ≈80%。Pool 在需要长时记忆（杯子位置、剩余物品数）的任务上差，Proprio 只在需要记自身状态时有效。

**预训练的作用（Fig. 9）。** 只在 post-training 加 video encoder（类 CronusVLA）平均 ≈50% vs 完整 MEM ≈72%；即便如此也好于 Pool，说明架构本身有效。

**不需要记忆的灵巧任务（Fig. 10）。** 7 个任务（bussing、shirt folding、batch folding、box building 等）MEM 与 π0.6 持平（平均估读均 ≈85%），没有出现加记忆掉点的现象，作者归功于混合了不同最优性/速度/频率的机器人数据与互联网视频，抑制了 video encoder 学到伪相关。

## 4. 局限

作者承认：记忆只在单个 episode 内，跨天/跨周的部署记忆是未来工作。我读出来的：(1) 语言记忆的训练标签由 LLM 生成，质量与 prompt 绑定，且要求 episode 有 subtask 级标注和成功/失败标志；(2) 记忆内容由模型隐式决定，无法被用户或高层直接读写（虽然它是文本）；(3) 长程任务只评了 10 次/任务、5 个食谱；(4) 未报告加 video encoder 后的具体延迟数字（Fig. 4 只给了趋势）；(5) 不开源。

## 5. 复现要点

- 未开源。语言记忆部分与架构无关，可以在开源 π0.5 的高层上复现：需要 subtask 标注 + LLM 生成摘要标签，再把 m_t 加进 prompt、m_{t+1} 加进输出文本。
- Video encoder 需要改 SigLIP/ViT 的注意力（每 4 层加因果时间注意力、时间位置编码），并在预训练阶段就带历史帧训练——只在 post-training 加效果打折（Fig. 9）。长上下文训练基础设施是主要工程量。
- 8×H100 上微调 3–5B 模型可行，但 K=6–18 帧 × 4 相机的序列长度会显著增加显存与吞吐压力。
- 坑：本体历史不要用文本 token；语言记忆标签必须做压缩；历史帧要 dropout（π0.7 给出 0.3）以保持无记忆时的能力。

## 6. 关键引用链

**建立在：** π0.6 模型卡、π*0.6（数据与纠正采集）、KI、FAST、Gemma 3；RTC 与训练时 RTC；π0.5 / Hi Robot 的高层-低层 subtask 接口；时空可分离注意力（Bertasius et al. 2021 TimeSformer、ViViT）；对照方法 ContextVLA（Jang et al. 2025）、TA-VLA（Zhang et al. 2025）、CronusVLA（Li et al. 2025）；相关的 TraceVLA、MemoryVLA、OneTwoVLA（纯语言记忆）；causal confusion 文献（Chi et al. 2025 等）。
**后续：** π0.7 直接采用 MEM 的 video history encoder 与连续本体嵌入，并声称零样本达到 MEM 微调 specialist 的水平（π0.7 Fig. 8）。
