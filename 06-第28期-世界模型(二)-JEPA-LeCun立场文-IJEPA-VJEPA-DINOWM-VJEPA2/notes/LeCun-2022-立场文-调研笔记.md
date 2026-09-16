# A Path Towards Autonomous Machine Intelligence（LeCun 2022 立场文，Version 0.9.2）

> 对应「可lip」第 28 期视频（世界模型系列第二期）。论文：[OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- 来源: OpenReview https://openreview.net/forum?id=BZ5a1r-kVsf（PDF: https://openreview.net/pdf?id=BZ5a1r-kVsf；Version 0.9.2，2022-06-27；**不在 arXiv**，无同行评审；约 62 页据他人 bib 条目 "pages 1–62"，未核实）· 机构: Meta AI（FAIR）+ NYU Courant Institute（据 2023 讲义署名及 Semantic Scholar 条目，未核实原文署名）· 代码/权重: 无（立场文，无模型、无实验）· 项目页: 无；官方介绍为 Meta AI 博客 2022-02-23《Yann LeCun on a vision to make AI systems learn and reason like animals and humans》（先于全文发布，称其为 "upcoming position paper"）。引用数未核实（Semantic Scholar API 限流）。
- 来源说明: 2026-09-11 OpenReview 的 PDF / forum / API 均被 Cloudflare 验证挡住，当时本目录无 论文 PDF；**2026-09-14 已从 Wayback 存档拿到原版 PDF并逐项核对，结果见下面第 0 节。**本笔记全部依据二手材料，正文用标签标明：**[讲义]** = Dawid & LeCun《Introduction to Latent Variable Energy-Based Models: A Path Towards Autonomous Machine Intelligence》arXiv 2306.02572（Les Houches 2022 讲义，LeCun 本人合著，自述"总结 Ref. [6] 即本文的主要思想"，其图 6 / 10 / 11 标注 adapted from [6]）；**[Meta 博客]** = 上述 2022-02-23 博客与 I-JEPA 博客（2023-06-13）；**[转述稿]** = Temple 大学 TAGIT 讨论班的论文汇报幻灯片（Hongzheng Wang，24 页，按原文小节标题逐节转述并保留公式）；**[引用方]** = 本仓库 V-JEPA / V-JEPA 2 / LeWorldModel / WAM 综述的 tex 源码里引用本文的段落；**[第三方讲义]** = Hadachi（Univ. of Tartu）课程讲义。摘要原文取自 LessWrong 2022-06-27 转帖。原文章节号、图号一律未核实，下文只用小节标题定位。
- 一句话: 一篇没有实验的路线图：提出六模块、全可微的认知架构（configurator / perception / world model / cost / actor / short-term memory），把"自监督学一个预测世界模型 → 在模型里做规划（Mode-2）→ 把规划结果蒸馏成反应式策略（Mode-1）"定为通往自主智能的主线，并给出 JEPA / H-JEPA（在表征空间而非像素空间做预测、用非对比正则化防 collapse 的非生成式架构）作为世界模型的具体形态。在世界模型发展史里，它把 world model 从 RL 里的一个子模块（Dyna / Ha & Schmidhuber / Dreamer 一线）提升为整个智能体的中心，并公开反对像素级生成式预测；此后 I-JEPA（2023）→ V-JEPA（2024）→ DINO-WM（2024）→ V-JEPA 2-AC（2025）→ LeJEPA（2025）→ LeWorldModel（2026）这条 latent WM 路线都以它为出发点，与 Cosmos / Genie / Veo 一类生成式视频 WM 形成今天的两大阵营。

## 0. 原文核对（2026-09-14，`论文 PDF` 已到手）

本笔记 2026-09-11 依据二手材料写成；2026-09-14 拿到原文（Wayback 存档的 OpenReview PDF）后逐项核对，结果如下。**下文各节里的 [讲义] / [转述稿] 标签和"未核实"字样保留不动，以本节为准。**

- ✅ 基本信息：62 页（正文 48 页 + 参考文献 pp.49–56 + 三个附录 pp.57–62）；署名 Courant Institute (NYU) + Meta FAIR；Version 0.9.2，2022-06-27；无实验、无代码。
- ✅ 原文章节 ↔ 笔记里引用的 [讲义] 小节：原文 §2 引言（三个挑战、四项贡献、20 小时学开车）↔ 讲义 Sec 2.2；原文 §3 架构（图 2 六模块、§3.1.1–3.1.4 Mode-1/Mode-2/蒸馏/推理即能量最小化、§3.2 代价模块式 (1)–(3)、§3.3 训练 critic 图 7）↔ 讲义 Sec 2.3；原文 §4.1–4.3 SSL/潜变量/EBM 训练（图 8–11、式 (4)–(9)）↔ 讲义 Sec 3–4；原文 §4.4–4.5 JEPA/四准则/VICReg（图 12–14、式 (10)–(12)）↔ 讲义 Sec 6.1；原文 §4.6–4.7 H-JEPA/分层规划（图 15–16）↔ 讲义 Sec 6.2；原文 §4.8 不确定性（图 17，七种来源：偶然 3 类 + 认知 4 类）、§4.9 世界状态记忆（式 (15)–(20)）、§4.10 五种数据流、§5 actor、§6 configurator、§7 相关工作、§8 讨论（8.1 缺什么、8.2 脑区对应/常识、8.3 规模不够/奖励不够/要不要符号）。
- ✅ 笔记第 2 节写的六模块职责、Mode-1/Mode-2 流程、代价 = IC + TC 线性组合且权重由 configurator 定、critic 用记忆里的 (τ, s_τ, IC(s_τ)) 三元组训练、JEPA 能量 = D(s_y, Pred(s_x, z))、四条训练准则、H-JEPA 堆叠、高层"动作"= 低层状态的目标（用代价 C(s[2]) 判定）、k^t 轨迹爆炸要剪枝、五种数据流、configurator 分解子目标"留待未来研究"——全部与原文一致。
- ✅ 原文 §7.1 **直接引用了** Ha & Schmidhuber 2018、Hafner et al. 2018/2020（PlaNet/Dreamer）、Sutton 1991 Dyna、Director（Hafner 2022）——笔记第 7 节"是否直接引用未核实"可改为已核实。
- ✅ "configurator 最神秘"是**原文自己的话**（§8.1 "the configurator module is the most mysterious"），不只是第三方讲义的说法。
- ⚠️ 笔记第 4 节"作者自己承认的局限"里的"能量不可校准、两个 EBM 不能相加"**原文没有**（全文无 calibrat / label bias 字样），那是 Dawid & LeCun 2023 讲义 Sec 3.1 的内容，引用时要标讲义而不是本文。
- ⚠️ 笔记第 1 节引的"LLM 给人推理的错觉但没有对现实的理解"不是原文措辞。原文的说法是：§8.2.2 "LLM 没有对底层现实的直接经验，其常识非常浅、可能与现实脱节"；§8.3.1 "当前模型只能做非常有限形式的推理……动态指定目标基本不可能"。
- ⚠️ 原文正文里两处 "Appendix 8.3.3" 是 LaTeX 交叉引用坏了，分别指附录"符号与记法"（p.57，图 18）和附录"EBM 对比训练的损失函数"（p.61，表 1）；§4.5.1 的那处指附录"摊销推断"（p.59）。表 1 共 10 行对比损失（原表 Min-hinge 和 Square-hinge 都编成第 6 行）。
- ⚠️ 原文 §4.10 末句的数据流编号与列表对不上（把 egomotion 写成 mode 3），中译里加了译注。
- 补充原文有、笔记没写的点：§4.8.1 建议**把世界模型和自我模型（ego model）分开**，自我模型可无潜变量，并可作为多智能体场景里他人模型的模板；§4.9 世界状态放在键值记忆里、事件只改被影响的实体条目（瓶子/厨房/餐厅的例子）；§6 configurator 的两个理由是硬件复用和知识共享，调制方式可以是给 Transformer 加额外 token；§8.2.1 的脑区对应（感知 ↔ 感觉皮层、WM/critic ↔ 前额叶、IC ↔ 基底核/杏仁核、短期记忆 ↔ 海马、configurator ↔ 前额叶执行控制、actor ↔ 前运动皮层）和"意识错觉是 configurator 的副作用"的思辨；§8.3.2 明确说本提案更像最优控制而不是 RL。

## 1. 要解决的问题
- 三个挑战 [讲义 Sec 2.2；转述稿 "Main Challenges"]：(1) 机器如何主要靠观察学会表示世界、预测、行动——真实交互昂贵且危险；(2) 如何以与梯度学习兼容的方式推理和规划——逻辑符号推理不可微；(3) 如何在多个抽象层级、多个时间尺度上表示感知和行动计划——把复杂动作分解为低层动作序列。
- 出发点：动物和人主要靠观察、以极少交互学到大量背景知识；"common sense" 被定义为"用世界模型填补感知 / 记忆得不到的信息（例如预测未来）"[讲义 Sec 2.2]，或"一组世界模型，告诉 agent 什么可能、什么合理、什么不可能"[转述稿]。人约 20 小时学会开车，当前系统需要海量标注 / 试错仍不可靠 [讲义 Sec 2.2；Meta 博客]。
- 对现状的诊断 [讲义 Sec 2.2]：SL 要大量标签、RL 要大量试错（现实里每个动作有代价甚至致命）；现有系统专用、脆弱、输入到输出的计算步数固定、不能推理和规划；（2022 年的）LLM "给人推理的错觉但没有对现实的理解"。
- 自监督学习（SSL：用输入的一部分预测另一部分）的核心难题 [讲义 Sec 2.4]：对图像 / 视频做单点预测会得到所有可能未来的平均（模糊）；现实随机、感知有限，不可能预测每个细节，而决策只需要与任务相关的那部分。因此需要 (1) 表示预测的不确定性、(2) 允许多模态预测；概率模型在高维连续域要算 partition function，不可行 → 改用能量模型 + 潜变量。
- 论文自述的四项贡献 [转述稿 "Main Contributions"]：全模块可微、多数可训练的整体认知架构；JEPA 与 H-JEPA——学习表征层级的非生成式预测世界模型架构；产生"既有信息量又可预测"的表征的非对比 SSL 范式；以 H-JEPA 为基础、在不确定性下做分层规划的预测世界模型。

## 2. 方法
**六模块架构 [Meta 博客；讲义 Sec 2.3 图 1；转述稿 "The Model Architecture"]。** 全部模块可微，整体可用梯度训练。
- configurator：执行控制（executive control），接收所有其它模块的输入，调节它们的参数和连接图以适配当前任务；最重要的功能是给 agent 设子目标并为该子目标配置 cost 模块；带来硬件复用与知识共享 [转述稿 "Actor and Configurator"]。
- perception：估计当前世界状态 s[0] = Enc(x)，可能是多层次的表示。
- world model：最复杂的模块，两个作用——估计感知没给出的缺失信息、预测合理的未来状态 s[t+1] = Pred(s[t], a[t])；可条件在潜变量 z 上以表示多个可能的未来。
- cost：标量"不适度"= intrinsic cost（IC，硬编码不可训练，"痛、饿"一类，定义 agent 的基本行为本性）+ critic（TC，可训练，预测未来 IC 的值，吸收文化 / 规范等外部影响）[讲义 Sec 2.3；转述稿 "Cost Module as the Driver of Behavior"]。
- actor：找到使未来 cost 最小的动作序列（Mode-2）；训练 policy 网络产生 Mode-1 动作；产生潜变量的多种配置以表示 agent 不知道的那部分世界状态 [转述稿]。
- short-term memory：存过去 / 当前 / 预测的世界状态及对应 cost（state-cost episodes），供世界模型预测和 critic 训练；论文建议世界状态放在可写记忆里、事件只更新受影响的部分，而不是每步重传整个向量 [转述稿 "Keeping track of the state of the world"]。
- 整个 perception–planning–action 回路相当于世界模型是学出来的 MPC（model-predictive control：每步在模型里优化一段动作序列、只执行第一个）；与 RL 的区别是 cost 已知且模块可微，预测未来 cost 不需要真的执行动作 [讲义 Sec 2.3]。

**两种感知–行动回路（对应 Kahneman 的 System 1 / System 2）[转述稿 "Typical Perception-Action Loops"；讲义 Sec 2.2 脚注]。**
- Mode-1 反应式：s[0] = Enc(x) → 策略直接出动作 a[0] = A(s[0]) → f[0] = C(s[0]) 与 (s[0], f[0]) 存入短期记忆；可选地用世界模型预测 s[1] = Pred(s[0], a[0])。一次前向。
- Mode-2 推理与规划：感知 → actor 提议动作序列 (a[0..T]) → 世界模型模拟出一条或多条状态序列 (s[1..T]) → 由 IC + critic 估计总 cost → actor 提出更低 cost 的序列，迭代 → 收敛后只把第一个动作送到执行器 → 每步把状态与 cost 存入记忆。"推理 = 用世界模型模拟 + 对动作序列做能量最小化"（Reasoning as Energy Minimization）。
- 从 Mode-2 到 Mode-1（学新技能）：Mode-2 昂贵、一次只能专注一个任务；先在 Mode-2 下运行，用其结果训练 policy 模块，之后同样情形可直接 Mode-1 出动作——即 amortized inference（把昂贵的优化结果蒸馏成一次前向）。

**世界模型的设计与训练 [转述稿 "Designing and Training the World Model"；讲义 Sec 3–6]。**
- 三个难点：模型质量取决于状态序列的多样性；世界不完全可预测，可能有多个合理的未来；要在不同时间尺度、不同抽象层级预测（高层目标与子目标）。
- energy-based model（EBM）：标量能量 F(x, y)，兼容的 (x, y) 低能量、不兼容高能量；推理 = argmin_y F；不要求预测唯一的 y，天然可表示多模态；不做归一化（概率模型在高维连续域不可行，归一化还带来 label bias）[讲义 Sec 3–3.1]。代价：能量不可校准（任意单位），两个独立训练的 EBM 不能直接相加，所以架构要避免在模块之间传递能量 [讲义 Sec 3.1]。
- 潜变量 EBM：E(x, y, z)，z 表示 y 里有而 x 里没有的信息（例如其他司机的意图），F(x, y) = min_z E 或对 z 取自由能；z 使确定性模块变成非确定性（可采样出多个预测）。关键陷阱：必须限制 z 的信息容量，否则模型把预测所需的全部信息塞进 z（"作弊"）[讲义 Sec 3.2]。
- 训练 EBM 与 representation collapse：collapse = 能量面变平、对所有 y 给同样能量，典型例子是 joint-embedding 的两个 encoder 都输出常数。两类办法：contrastive（压低数据能量、抬高对比样本能量，对任何架构有效但对比样本随维度指数增长，NLL 也是它的特例）vs regularized / architectural（限制低能量区域的体积：瓶颈、稀疏、离散、变分、VICReg）。论文主张后者 [讲义 Sec 4.1–4.2；对比损失清单在原文附录 8.3.3，据讲义引用]。
- JEPA（joint-embedding predictive architecture）：两个可以不同、不共享参数的 encoder 得到 s_x、s_y，predictor 在 z 的帮助下从 s_x 预测 s_y，能量 = 两个表征的距离 D(s̃_y, s_y)。与生成式的区别：在表征空间预测，y 的 encoder 可以丢掉不可预测的细节（草叶、水波、地毯纹理），"JEPA 的妙处在于它自然产生去掉无关细节的抽象表征"[Meta 博客原话]；多模态由 encoder 的不变性和潜变量 z 两条途径表示 [讲义 Sec 6.1 图 9；转述稿]。
- JEPA 的训练准则（四项）[Meta 博客；讲义图 10]：最大化 s_x 对 x 的信息量、最大化 s_y 对 y 的信息量、最小化预测误差、最小化 z 的信息容量。信息量项用 VICReg（variance–invariance–covariance regularization：每维方差高于阈值的 hinge、批内协方差去相关、预测不变性；加 expander 去非线性依赖）[讲义 Sec 6.1]。
- H-JEPA：堆叠 JEPA——JEPA-1 用低层表征做短期预测，JEPA-2 以 JEPA-1 的输出为输入、抽出更抽象的表征做长期预测；层间可加 CNN / pooling 粗粒化 [讲义 Sec 6.2 图 11]。Meta 博客的例子：从毫秒级手部轨迹到"做可丽饼"级别的任务序列。
- 分层规划：各层都预测，高层长程、低层短程；总任务由高层目标 C(s2[4]) 定义；高层"动作"不是真动作，而是低层预测状态的目标（即子目标）[转述稿 "Hierarchical Planning"]。
- 不确定性：把随机性推入潜变量，z 可优化、可预测或可采样；每步 k 个离散取值 → 轨迹数按 k^t 增长，必须定向搜索与剪枝 [转述稿 "Handling uncertainty"]。
- 数据流（五种信息获取方式）[转述稿 "Data Streams"]：passive observation、active foveation（只调注意力）、passive agency（观察别的 agent 行动以推断动作的因果效应）、active egomotion（移动传感器但不扰动环境）、active agency（自己的动作影响感知流）。物理规律原则上可从观察推得，但高效训练世界模型可能需要"agentive"采集。
- 归纳：预测空间 = latent（表征空间），明确反对像素 / 生成式；动作进入 = Pred(s, a)；动作出来 = Mode-2 优化（规划）或 Mode-1 policy（蒸馏）；损失 = 预测误差 + 信息量正则；数据规模、推理开销 = 立场文未给数字。


## 3. 实验
无实验、无数字。它提出的可检验主张及截至 2026-09 的兑现情况：
1. 非对比正则化的 JEPA 不 collapse，且表征"既有信息量又可预测"→ 部分验证：VICReg（ICLR 2022）；I-JEPA、V-JEPA 实际改用 EMA teacher（目标编码器取在线编码器的指数移动平均并 stop-gradient）防 collapse，而非 VICReg（见 `V-JEPA` 笔记）；LeJEPA（Balestriero & LeCun 2025，arXiv 2511.08544）补上理论（各向同性高斯是最优嵌入分布）与 SIGReg 正则；LeWorldModel 用 SIGReg 从像素端到端训练不 collapse（见其笔记）。
2. 表征空间预测优于像素重建 → V-JEPA 论文做了 feature prediction vs pixel prediction 的直接对照（数字见 `V-JEPA` 笔记第 3 节）；I-JEPA 博客称抽象预测目标"消除了不必要的像素级细节"。
3. 预测式 latent WM + 少量交互数据即可做 Mode-2 规划 → DINO-WM（冻结 DINOv2 + CEM）、V-JEPA 2-AC（<62 h Droid 后训练、Franka 零样本，16 s / 动作）验证了可行性，也暴露了 Mode-2 的代价（见 `dino-wm`、`vjepa2` 笔记）。
4. Mode-2 → Mode-1 蒸馏 → V-JEPA 2 只把"在想象中训前馈 policy"列为未来工作；Dreamer 系在想象中训 actor-critic 但用的是重建式 WM；JEPA 路线上我没有找到系统验证（未核实）。
5. H-JEPA 多层预测 + 分层规划 → 我没有找到在真实视频上训练 H-JEPA 并做分层规划的公开实现（未核实）；tutorial 里 neural-symbolic / 子目标一类工作从别的路径做分层。
6. 潜变量 z 表示不确定性并可限制其信息量 → I-JEPA / V-JEPA / V-JEPA 2-AC 实际都用确定性 predictor（无 z），多模态未来仍是这条路线的短板（`vjepa2` 笔记第 4 节）。
7. 硬编码 IC + 可训练 critic 驱动行为、configurator 设子目标 → 无公开实现。

## 4. 局限
- 作者自己承认的 [讲义 Sec 3.1；转述稿；第三方讲义，措辞未核实]：能量不可校准，跨模块组合困难；潜变量的搜索空间随步数指数增长；世界不完全可预测；configurator 没有给出训练算法（第三方讲义称之为"最神秘的部分"）；H-JEPA 能否在真实视频上大规模训练未知；intrinsic cost 的设计决定 agent 的"本性"与安全，论文只提出问题。
- 我读出来的：(1) 无实验、无数字，所有主张靠后续论文兑现，兑现的只有 JEPA 表征学习和单层 latent 规划两项；(2) 没有语言模态、没有工具使用——写于 ChatGPT 之前，对 LLM 的判断与 2023 年后 LLM agent 的实践相冲突，也没有讨论 LLM 能否充当 configurator；(3) 假设单一可微系统端到端训练，实践却走向"预训练大编码器 + 小 predictor 后训练"（V-JEPA 2-AC）或"冻结特征 + 小 WM"（DINO-WM）；(4) 表征空间能量与任务成功是否对齐没有讨论——V-JEPA 2-AC / DINO-WM 的目标图像距离只在分布内可信；(5) 非生成式意味着不能产出可视化和合成数据，与 2025–26 年生成式 WM 当数据引擎、评测器的用途（Cosmos、Ctrl-World、Veo）正交；(6) 五种数据流里最关键的 active agency 机器人数据仍稀缺，论文没说怎么解决；(7) "Mode-2 昂贵"被低估了——后代 DINO-WM 一次规划 53 s、V-JEPA 2-AC 16 s / 动作。

## 5. 复现要点
- 无代码、无模型。最接近的官方实现：VICReg（facebookresearch/vicreg）、I-JEPA（facebookresearch/ijepa）、V-JEPA（facebookresearch/jepa）、V-JEPA 2（facebookresearch/vjepa2，见 `vjepa2` 笔记）、DINO-WM（见 `dino-wm` 笔记）、LeJEPA（代码状态未核实）、LeWorldModel（见其笔记）。
- 想按本文思路搭最小系统：JEPA 世界模型（DINO-WM / LeWorldModel 量级，单卡可训）+ Mode-2 的 CEM 或梯度规划 + 用规划结果蒸馏 policy。常见坑：collapse（要 VICReg / SIGReg 一类正则或 EMA teacher）；潜变量作弊（限制 z 信息量，或干脆不用 z）；能量尺度不可跨模型比较；规划轨迹 k^t 爆炸要靠 CEM / 剪枝；目标以图像给出时须与当前视角一致。
- 算力：立场文不涉及；参照 `vjepa2`、`dino-wm`、`LeWorldModel` 笔记。

## 6. 关键引用链
- 建立在：Craik 1943 的 mental models；Kahneman《Thinking, Fast and Slow》（System 1 / 2 → Mode-1 / 2）；MPC / 最优控制；LeCun 等 2006 的 EBM tutorial；Hopfield / Boltzmann / denoising AE 一线 EBM；Becker & Hinton 1992 的联合嵌入思想；BYOL / SimSiam / Barlow Twins / VICReg 的非对比 SSL；Ha & Schmidhuber 2018 与 Dreamer 一线的 latent WM（是否直接引用未核实）。
- 后续与推进：Dawid & LeCun 2023 讲义（把 EBM / JEPA 部分数学化）；I-JEPA（Assran 2023，arXiv 2301.08243，图像）→ V-JEPA（`V-JEPA`，视频）→ V-JEPA 2 / 2-AC（`vjepa2`，加动作条件、真机规划）；DINO-WM（`dino-wm`，LeCun 合著，冻结特征上的 Mode-2 规划）；LeJEPA 2025 → LeWorldModel（`LeWorldModel`，像素端到端 JEPA WM）；WAM 综述（`survey-wam-next-frontier`）把它列为 world model 定义来源之一 [引用方]；roadmap（`roadmap-wam-to-embodied-brains`）把它列为分层 embodied brain 思想来源；tutorial（tutorial（arXiv 2607.00836））的 state-space / latent 预测式一类都是它的后代；`Dreamer-v3` 是它"在想象中训 policy"主张的重建式对照。
