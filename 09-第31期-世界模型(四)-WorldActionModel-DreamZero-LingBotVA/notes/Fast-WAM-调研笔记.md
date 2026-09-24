# Fast-WAM: Do World Action Models Need Test-time Future Imagination?（Fast-WAM）

> 对应「可lip」第 31 期视频（世界模型系列第四期）。论文：[arXiv:2603.16666](https://arxiv.org/abs/2603.16666)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2603.16666（v1 2026-03 中旬，精确日期未核实；v2 2026-03-23；用 NeurIPS 2025 模板，会议 / 录用未核实）· 机构: 清华大学 IIIS、Galaxea AI（Tianyuan Yuan、Zibin Dong、Yicheng Liu、Hang Zhao）· 代码/权重: 未核实（论文只给项目页）· 项目页: https://yuantianyuan01.github.io/FastWAM/
- 一句话: 一篇"消融论文"：在同一 Wan2.2-5B 骨干、同一训练配方下造出三个 WAM 变体——推理时不想象未来的 Fast-WAM、联合去噪视频 + 动作的 Fast-WAM-Joint、先生成视频再解动作的 Fast-WAM-IDM——外加一个不做视频 co-training（把视频预测当联合训练目标）的对照。三个带 co-training 的变体 RoboTwin 91.8 / 90.6 / 91.3、LIBERO 97.6 / 98.5 / 98.0 几乎一样，去掉 co-training 掉到 83.8 / 93.5，真机叠毛巾从约 0.75 掉到 0.10；而 Fast-WAM 190 ms、比 IDM 变体（810 ms）快 4 倍以上。结论：WAM 的收益主要来自训练时的视频预测目标塑造表示，而非推理时显式生成未来——对 2026 年"WAM 一定要想象"这一主流叙事的直接质疑。

## 1. 要解决的问题
- WAM（统一建模未来视觉预测与动作）被认为比 VLA 更懂物理动力学；但主流实现是 imagine-then-execute（先迭代去噪出未来视频再出动作：LingBot-VA、DreamZero、Vidar、Mimic-Video），测试时延迟大。
- WAM 的收益可能来自两个纠缠的因素：(1) 训练时的视频预测目标（更好的物理先验与动作条件表示）；(2) 推理时显式生成未来（额外"预见"）。已有系统把二者绑在一起，无法归因。
- 若价值主要在 (1)，就可以保留视频 co-training 而去掉测试时的未来合成，换来实时性。

## 2. 方法
**预测空间**：Wan2.2 VAE 的视频 latent（训练时）；推理时不预测任何未来。
- **形式化（3.1）**：imagine-then-execute 是 p(a_{1:H}|o, l) = ∫ p(v_{1:T}|o, l) p(a_{1:H}|o, l, v_{1:T}) dv（式 2）；Fast-WAM 直接学 p_θ(a_{1:H}|z(o, l))（式 4），z 是视频骨干对当前上下文的一次前向编码、被视频 co-training 塑造。为控制变量只做单 action chunk 生成，省去外层自回归循环。
- **架构（3.2，Fig 2a）**：Wan2.2-5B 的视频 DiT 当世界建模骨干，复用其 T5 文本编码器（cross-attention 注入所有 token）与视频 VAE；新增一个 action expert DiT（同结构、hidden 1024、约 1B），两者以 Mixture-of-Transformer（MoT，共享 attention、分模态参数）组成，总 6B。
  - token 分三组：首帧干净 latent（共享视觉锚点）、未来帧噪声 latent（只在训练用）、动作 token。
  - 结构化 attention mask（Fig 2b）：未来视频 token 在视频分支内双向 + 可看首帧；动作 token 在动作分支内双向 + 可看首帧；**动作 token 不能看未来视频 token**，首帧 token 不看任何人。
  - 推理时干脆不实例化未来视频 token，首帧 latent 过一次视频骨干产出 z 供动作 expert 去噪。
- **损失（3.2）**：视频与动作共用 flow matching：y_t = (1−t)y + tε，L_FM = E‖f_θ(y_t, t, o, l) − (ε − y)‖²（式 5–6）；L = L_act + λ L_vid（式 9），λ 未给。噪声时间步 logit-normal（沿用 Wan）。
- **实现（4.1）**：多相机图像拼成一张进 VAE；action horizon h = 32；视频时间下采样 4×，每 chunk 9 帧；推理 10 步去噪，CFG = 1.0；AdamW 1e-4、wd 0.01、cosine，混合精度、梯度裁剪 1.0；延迟在单张 RTX 5090D V2 32GB 上测。
- **受控变体（3.3）**：
  - Fast-WAM-Joint——允许视频 token 与动作 token 互相 attention、联合去噪（对应 DreamZero / UWM / Motus）。
  - Fast-WAM-IDM——先生成未来视频 token，再条件于生成结果出动作（对应 LingBot-VA / Vidar / UniPi），训练时按 LingBot-VA 以 p = 0.5 对真值视频 token 做噪声增广。
  - Fast-WAM w/o video co-train——架构与推理不变、只去掉 L_vid。
- **训练数据**：**无 embodied pretraining**——直接在各 benchmark 数据上训练（LIBERO 4 套件各 500 条演示，20k 步；RoboTwin 2.0 2,500 干净 + 25,000 随机化演示、50+ 任务，30k 步；真机 60 小时遥操作，30k 步）。GPU 数与训练时长未给。
- **推理开销（Fig 4 右）**：Fast-WAM 190 ms / chunk，Joint 580 ms，IDM 810 ms，π0.5 180 ms。
- **动作层级**：low-level action chunk（h = 32），语言条件。


## 3. 实验
- Setting（4.2）：LIBERO（Spatial / Object / Goal / Long，40 任务 2000 次试验）；RoboTwin 2.0（50+ 双臂任务，clean / randomized 各 100 次 / 任务）；真机 Galaxea R1 Lite 叠毛巾（长程、形变物体；报成功率 + 平均完成时间，Fig 3）。
- Baseline：π0、π0.5、OpenVLA（有 embodied 预训练）；Motus、LingBot-VA（有预训练；另有二者"from Wan2.2、无预训练"版本）。
- Table 1 RoboTwin（clean / rand / avg）：Fast-WAM 91.88 / 91.78 / **91.8**；LingBot-VA（预训练）92.90 / 91.50 / 92.2；LingBot-VA from Wan2.2 80.6；Motus（预训练）87.8、from Wan2.2 77.3；π0.5 79.8；π0 62.2。变体：Joint 90.6、IDM 91.3、w/o co-train 83.8（82.76 / 84.80）。
- Table 2 LIBERO（Spatial / Object / Goal / Long / avg）：Fast-WAM 98.2 / 100.0 / 97.0 / 95.2 / **97.6**；LingBot-VA 98.5；Motus 97.7；π0.5 96.9；π0 94.1；OpenVLA 76.5。变体：Joint 98.5（99.6 / 99.4 / 98.2 / 96.8）、IDM 98.0、w/o co-train 93.5（Spatial 89.2、Long 90.0 掉得最多）。
- **关键对比（4.3.2）**：三种带 co-training 变体之间差距 ≤1.2（RoboTwin）/ ≤0.9（LIBERO），而去掉 co-training 的差距是 8.0 / 4.1——"想象怎么做、做不做"不重要，"训练时学不学预测视频"重要。
- Fig 4 真机叠毛巾（散点估读；10% 为正文数字）：π0.5（预训练）成功率约 1.0、完成时间约 118 s 最好；Fast-WAM-IDM 约 0.90 / 177 s；Fast-WAM 约 0.75 / 152 s；Joint 约 0.70 / 227 s；π0.5 无预训练约 0.40 / 206 s；Fast-WAM w/o co-train 0.10 / 约 241 s。
- Fig 4 延迟（图中标注）：π0.5 180、Fast-WAM 190、w/o co-train 190、Joint 580、IDM 810 ms。
- Table 3 逐任务：Fast-WAM 在 Blocks Ranking Size（94 / 98 vs Motus 75 / 63）、Press Stapler（90 / 97 vs Joint 52 / 50）强；Open Microwave（62 / 45 vs Motus 95 / 91）、Turn Switch、Place Can Basket 弱；Joint 在 Open Microwave 崩到 3 / 14。
- 最有信息量的消融就是 w/o video co-train 本身；没有 λ、horizon、去噪步数、骨干规模的消融。

## 4. 局限
- 作者承认（Sec 5）：未研究更大规模预训练数据和模型规模下结论是否成立；为控制变量省掉外层自回归、只做单 chunk（3.1）。
- 我读出来的（结论范围）：所有变体都没有 embodied pretraining、都在分布内 benchmark 上评测，而 DreamZero 等主张显式想象的价值在于 zero-shot 泛化到未见任务 / 环境，本文没测这种情况；视频 co-training 的数据就是同一批机器人演示，没验证"用无动作的外部视频 co-training"这一 WAM 卖点。
- 我读出来的（真机）：只有一个任务、rollout 次数未报，且预训练 π0.5 反而最好，说明这套 WAM 配方尚未超过成熟 VLA；IDM 变体在真机成功率最高（约 0.90 vs 0.75），与标题结论有张力，作者以完成时间和延迟辩护。
- 我读出来的（实现）：Joint 变体在若干任务上崩溃暗示各变体实现质量不完全对等；λ、噪声增广等超参未报；延迟是单卡 5090、无系统优化（DreamZero 用 CFG 并行 / 缓存 / 量化把 14B 压到 150 ms），"4× 更快"的对比对象是自己的变体。

## 5. 复现要点
- 代码 / 权重：未核实。骨干 Wan2.2-5B（含 VAE、T5）开源可得；action expert 与 MoT 需自行实现或参考 Motus / LingBot-VA 的开源代码（两者同样基于 Wan2.2-5B，结构相近）。
- 算力：6B 全参训练，LIBERO 20k 步 / RoboTwin 30k 步；GPU 数未报，按 Motus / LingBot-VA 同类配置估计需要 8 卡 H100 / H800 级；推理单张 5090（32GB）。
- 数据：LIBERO 四套件 2000 条；RoboTwin 2.0 27.5k 条（需自己在仿真里生成）；真机 60 小时不可得。
- 坑：attention mask 必须保证动作 token 看不到未来视频 token（否则就成了 Joint）；首帧 token 不看其他 token；多相机拼图进 VAE；视频 4× 时间下采样、9 帧 / chunk 与 h = 32 对齐；IDM 变体要做 p = 0.5 噪声增广；λ 未给要自己扫。
- 别外推：结论只在分布内验证过，不要外推到 zero-shot 场景。

## 6. 关键引用链
- 建立在：Wan2.2-5B 视频 DiT；π0 / π0.5 的 action expert + flow matching（π0、π0.5）；Mixture-of-Transformers（Motus 同样用 MoT）。
- 变体原型：LingBot-VA（LingBot-VA，IDM 变体的原型与噪声增广配方）；DreamZero（DreamZero，"WAM"命名来源与 Joint 变体原型）；UWM（UWM）、Motus（arXiv:2512.13030）作为 joint 建模代表；UniPi（UniPi）作为 imagine-then-execute 源头。
- 先行者：VPP（VPP）与 UVA 作为"不解码视频"的先行者；GR-1（GR-1）Table 4 是同一命题的早期证据。
- 后续 / 同期讨论：tutorial（WM-to-WAM-教程）把它当第 4 范式代表并指出该范式的风险；roadmap（roadmap-wam-to-embodied-brains）、HarnessWAM（HarnessWAM）把"想象 / 慎思"移到模型外的 harness 层；survey（arXiv:2605.00080）对"WM for policy"耦合方式的分类可与本文三变体对照。
