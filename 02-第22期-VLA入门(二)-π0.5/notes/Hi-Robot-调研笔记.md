# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models

> 对应「可lip」VLA 入门第 22 期（二）视频。论文：[arXiv:2502.19417](https://arxiv.org/abs/2502.19417)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2502.19417（v1 2025-02-26，本地 PDF 为 v2 2025-07-15；ICML 2025）· 机构: Physical Intelligence / Stanford / UC Berkeley（一作 Lucy Xiaoyang Shi）· 代码/权重: 未开源 · 项目页: https://www.pi.website/research/hirobot
- 一句话: 用两个 PaliGemma-3B——一个微调成"高层 VLM"把开放式 prompt 和用户插话翻译成原子指令，另一个是 π0——组成 System 2 / System 1 层级系统，并用大 VLM 合成"假想的用户对话"来训练高层；真机上高层 VLM 的指令准确率比零样本 GPT-4o 高 40 个百分点以上。

## 1. 要解决的问题

π0 这类 VLA 只能跟随"拿起可乐罐"式的原子指令。真实 prompt 是"给我做个素三明治，不要番茄，如果有火腿再给朋友做一个"，中途还会插话"那个不是垃圾"。这需要：(1) 把复杂语言情境化到当前观测；(2) 组合已有技能解决新任务；(3) 实时接受纠正。前作：SayCan 等用 LLM 规划但看不到图像；VoxPoser/MOKA 等用 VLM 输出技能参数但灵巧性差、不支持实时语言交互；YAY Robot 能实时纠正但只支持一个 prompt、纠正语句限于人工数据；RACER 依赖仿真生成恢复行为。

## 2. 方法

**分解。** p(A_t | o_t) 拆成高层 p^hi(ℓ̂_t | I¹…ⁿ_t, ℓ_t) 与低层 p^lo(A_t | I¹…ⁿ_t, ℓ̂_t, q_t)。高层输入多路相机图像 + 开放式 prompt ℓ_t（含用户插话），输出原子指令 ℓ̂_t，可附带一段说给用户听的话 u_t（TTS 播出后从 ℓ̂_t 里去掉）。低层就是 π0（flow matching，50 步 chunk）。

**调度。** 高层每 1 秒重跑一次，或者一有新的用户输入就立刻重跑（Sec. 4.1）；用户说"leave it alone"被满足后可以示意机器人切回原任务。语音链路：领夹麦克风 → Whisper large-v2 本地转写；回复用 Cartesia TTS（Appendix B）。

**数据（Sec. 4.3，Appendix A）。** (1) 遥操作示范 D_demo 只有粗任务标签；(2) 切成 1–3 秒的技能片段 ℓ̂_t（"pick up one piece of lettuce"），并从原始动作里启发式抽取"move the right arm to the left"这类运动原语，得到 D_labeled；(3) 用一个大 VLM p^gen 看着图像 + 当前技能标签 + 本 episode 之前的技能序列，"想象"一个会导致这条指令的用户 prompt/插话 ℓ_t 和机器人回复 u_t，得到合成集 D_syn。prompt 里显式枚举场景类型（negative task、situated correction、specific constraint）和回复类型（confirmation、clarification、error handling），并要求 p^gen 用世界知识（乳糖不耐 → 不放奶酪）。

**训练。** 高层在 D_syn ∪ D_labeled 上做 next-token 交叉熵；低层在 D_labeled ∪ D_demo 上做 flow matching。两者都从 PaliGemma-3B 全参数微调；AdamW(0.9, 0.95)、LR 1e-5（1k 步 warmup 后恒定）、batch 512、EMA 0.999；高层训练约 2 小时 / 8×H100（Appendix C）。每个任务域训一个独立的高层。

**延迟（RTX 4090，Appendix B）。** 低层 73 ms（on-board）/86 ms（Wi-Fi）；高层 prefill 47 ms + 每 token 13.2 ms（H100 上 17.3 + 5.7 ms）。

## 3. 实验

**平台/任务。** UR5e 单臂：table bussing（"只收垃圾不收餐具""收所有黄色的东西"，插话"这不是垃圾"）；双臂 ARX：sandwich making（"素三明治，我对腌黄瓜过敏"，"够了别加了"）；移动双臂 ARX：grocery shopping（"给我拿点吃的准备电影之夜""来点甜的"，插话"我还想要 KitKat"）。每任务每方法 20 次试验，盲评。指标：Instruction Accuracy（IA，高层输出与"用户意图 + 当前观测"一致的比例）和 Task Progress（TP，放对位置的物体比例）。

**Baseline。** 人类专家做高层（oracle）；GPT-4o 做高层（同一低层，prompt 里给出按频率排序的技能标签列表，Appendix C 附完整 prompt）；flat VLA（π0 直接吃复杂 prompt）；flat VLA + 合成数据；Hi Robot 无合成数据（≈VLM 版 YAY Robot）。

**主结果（Fig. 5，柱状图估读）。** 三任务平均 IA：flat ≈33%、GPT-4o ≈27%、Hi Robot ≈73%（caption：比 GPT-4o 高 40% 以上）；平均 TP：≈42% / ≈60% / ≈77%，oracle 约 85%。GPT-4o 的典型失败：物体识别错（"pick up bermuda triangle"）、跳步、忽略用户约束、夹爪还拿着东西就发下一条抓取指令、不能维持一致的内部状态（Fig. 6 定性对比）。

**消融。** (A) 去掉合成数据：平均 IA 差 46 个百分点、TP 差 39（Fig. 7 标注），无合成数据的模型会无视"这不是垃圾"或放入过敏成分。(B) 同样用合成数据但改成 flat 策略：IA 差 19、TP 差 34（Fig. 8 标注），flat 模型倾向回退到"清空所有物品"的默认行为。

**人类 oracle 的信息。** 有人给指令时低层几乎不出错，说明失败主要来自推理而非执行。

## 4. 局限

作者承认（Sec. 6，Appendix C.4）：高层没有记忆，长上下文推理弱；合成数据依赖 prompt 工程；高层和低层各自训练，高层不知道低层的能力边界，也不知道指令是否执行成功；低层会因"就近抓取"的数据偏差暂时无视指令（乳糖不耐者的三明治仍去抓奶酪）；掉落物体后的 OOD 恢复差。我读出来的：每个任务域训一个高层，不是通用高层；20 次试验、估读差异有噪声；"每秒重规划"没有完成检测，容易在子任务完成前后切换指令。

## 5. 复现要点

- 未开源。但组件都可得：低层用 openpi 的 π0；高层是 PaliGemma-3B 的 VQA 式微调（输入图像 + prompt，输出指令字符串），2 小时 / 8×H100，batch 512 需多卡；合成数据用任意大 VLM API。
- 需要自己准备：1–3 秒粒度的技能标注（成本最高）、运动原语的启发式抽取、语音链路。
- 坑：合成 prompt 要条件在历史技能序列上，否则多步任务不连贯；GPT-4o baseline 必须给技能列表，否则输出低层听不懂的指令；高层输出里要把"说话"和"指令"分开。

## 6. 关键引用链

**建立在：** π0（低层）；PaliGemma；SayCan（LLM 高层规划的原型）；YAY Robot（Shi et al. 2024，语言纠正的层级系统）；RT-H（语言动作层级）；OLAF、RACER（语言纠正的对照）；RLVF（Stephan et al. 2024，合成数据的场景分类思路）；Kahneman 的 System 1/2 隐喻。
**后续：** π0.5 把高层和低层合成同一个模型（subtask prediction 作为 chain-of-thought）并加入 verbal instruction 数据；π0.7 的"语言 coaching → 训练高层策略"直接延续了这里的高层/低层接口；MEM 给高层加上语言记忆，正是针对本文承认的"高层无记忆"。
