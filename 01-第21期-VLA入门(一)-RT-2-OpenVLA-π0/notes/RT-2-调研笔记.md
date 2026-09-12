# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

> 对应「可lip」VLA 入门第 21 期（一）视频。论文：[arXiv:2307.15818](https://arxiv.org/abs/2307.15818)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2307.15818（v1 2023-07-28，CoRL 2023）· 机构: Google DeepMind · 代码/权重: 未开源（PaLI-X / PaLM-E 均为内部模型；RT-2-X 只在 Open X-Embodiment 项目中以 API 形式给合作者用）· 项目页: https://robotics-transformer2.github.io
- 一句话: 第一次把"把机器人动作当成文本 token、直接微调大 VLM"这条路线跑通并命名为 VLA；证明网络规模的视觉-语言预训练能迁移成机器人策略的语义泛化与"涌现"能力。

## 1. 要解决的问题
- 此前用 LLM/VLM 做机器人的工作（SayCan、PaLM-E、Code as Policies）只在高层规划用大模型，相当于一个 state machine，低层控制器享受不到 web 预训练知识；RT-1 这类从零训练的策略只见过十几万条真机数据，语义泛化弱。
- 问题：能否让预训练 VLM 直接输出低层动作，把语义理解和推理带进闭环控制？

## 2. 方法
- backbone：PaLI-X（5B / 55B，ViT-22B + UL2 式 encoder-decoder）和 PaLM-E（12B，decoder-only，ViT-4B）。不加任何新参数。
- 动作表示：沿用 RT-1 的离散化，8 个整数 `terminate Δpos_x Δpos_y Δpos_z Δrot_x Δrot_y Δrot_z gripper`，每维 256 bins 均匀量化，目标串形如 "1 128 91 241 5 101 127"。PaLI-X 直接用 1000 以内的整数 token；PaLM-E 覆盖 256 个最不常用 token（symbol tuning）。输入用 VQA 格式 "Q: what action should the robot take to [instruction]? A:"（Sec 3.2）。
- Co-fine-tuning：微调时把原 VLM 的 web 数据（WebLI 等 VQA/captioning）和机器人数据混在同一 batch，而不是只用机器人数据；机器人数据约占 50%（PaLI-X）/ 66%（PaLM-E）（Appendix B）。
- 输出约束：机器人任务解码时只允许采样合法 action token。
- 训练（Appendix E）：PaLI-X-55B lr 1e-3、batch 2048、80K steps；5B 版 270K steps；PaLM-E-12B lr 4e-4、batch 512、1M steps。
- 推理（Sec 3.3）：模型部署在多 TPU 云服务上，机器人通过网络查询；55B 1–3 Hz，5B 约 5 Hz。每步只出一个动作，没有 action chunk。
- CoT 变体（Sec 4.4）：PaLM-E 版再微调几百步，数据里加 "Plan" 字段："Instruction: I'm hungry. Plan: pick rxbar chocolate. Action: 1 128 124 136 121 158 111 255."

## 3. 实验
- Setting：Google 7-DoF 移动机械臂，RT-1 数据（13 台机器人、17 个月、办公室厨房），约 6000 次真机评测。baseline：RT-1（35M）、VC-1、R3M、MOO。
- 主结果（Appendix Table 4 / Fig 5）：seen tasks 上 RT-2-PaLI-X-55B 91%、RT-2-PaLM-E-12B 93%、RT-1 92%——in-distribution 没有差别；unseen（objects / backgrounds / environments）平均 RT-2 两版都是 62%，RT-1 32%、MOO 35%、VC-1 10%、R3M 12%，约 2×。
- 涌现能力（Appendix Table 5）：symbol understanding / reasoning / person recognition 三类平均，RT-2-PaLI-X-55B 60%、PaLM-E-12B 40%、RT-1 17%、VC-1 11%。PaLM-E 在 math 子项更好（35 vs 25），归因于预训练混合。
- 消融（Appendix Table 6，unseen 平均）：PaLI-X-5B 从零训练 9% / 只用机器人数据微调 42% / co-fine-tune 44%；55B 微调 52% / co-fine-tune 63%。结论：模型越大越泛化；co-fine-tuning 优于纯微调；从零训练基本不 work。
- Language-Table 仿真（Table 1）：RT-2-PaLI-3B 90±10 vs LAVA 77、RT-1 74、BC-Zero 72。

## 4. 局限
- 作者承认（Sec 5，Appendix G）：没有学到新的 motion，物理技能仍限于机器人数据分布（按部位抓取、擦桌/用工具、折毛巾、多层间接推理都做不到）；推理成本高，55B 只有 1–3 Hz；可用的 VLM 太少且闭源。
- 我读出来的：每步单动作、无 chunk，控制频率低；256 bins 离散精度有限、自回归 8 个 token 串行；评测全在 Google 内部平台，外界无法复现；co-fine-tuning 需要原 VLM 的预训练数据，开源模型做不到同样的事。

## 5. 复现要点
- 不可复现（模型、数据、评测平台全闭源）。要"类 RT-2"，用 OpenVLA（同样的 256-bin 离散 token 范式，7B，开源）。
- 可引用的官方数字：unseen 平均 62% vs RT-1 32%；涌现评测 60%。SimplerEnv 上 RT-2-X 的 Google Robot Visual Matching 平均 60.7%、Variant Aggregation 64.3%（转引自 MolmoAct Table 1，未在本文核实）。

## 6. 关键引用链
- 建立在：RT-1（动作离散化与数据）、PaLI-X / PaLM-E（VLM backbone）、SayCan / Code as Policies（作为对照的"高层规划"路线）、symbol tuning。
- 后续：RT-2-X / Open X-Embodiment（跨本体数据）、OpenVLA（开源复刻）、ECoT（在 OpenVLA 上做 embodied CoT）、π0-FAST（更好的动作 tokenizer）、Gemini Robotics（同团队下一代）。
