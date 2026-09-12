# π*0.6: a VLA That Learns From Experience (RECAP)

> 对应「可lip」VLA 入门第 23 期（三）视频。论文：[arXiv:2511.14759](https://arxiv.org/abs/2511.14759)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2511.14759（v1 2025-11-18，本地 PDF 为 v2 2025-11-19）· 机构: Physical Intelligence · 代码/权重: 未开源；π0.6 基座只有模型卡（website.pi-asset.com/pi06star/PI06_model_card.pdf）· 项目页: https://pi.website/blog/pistar06
- 一句话: 提出 RECAP——用一个多任务分布式 value function 估计 advantage，把二值化的 advantage 指示符当文本 token 喂给 VLA（advantage conditioning），从而用示范 + 自主 rollout + 人类干预的混合数据做迭代式 offline RL；在 π0.6 上把做浓缩咖啡、折多样衣物、装箱的 throughput 翻倍以上、失败率减半，支撑了 13 小时连续制作咖啡的部署。

## 1. 要解决的问题

模仿学习有复合误差，上限是示范数据的水平。要让 VLA "熟能生巧"，就要从自主执行的经验（含失败）里学，即 RL。难点：(1) 对几十亿参数、flow matching 输出的 VLA 做稳定可扩展的 RL——PPO 类方法需要 log-likelihood，flow 模型没有闭式似然；(2) 数据异构：示范、旧策略 rollout、新策略 rollout、专家中途干预（HG-DAgger 式）都要能用；(3) 真实世界奖励稀疏、模糊。此前 VLA+RL 的工作多用离散动作或高斯策略、残差策略、在 VLA 之上再选动作，或只做简单任务；本文要端到端训整个 flow VLA，任务是 5–15 分钟的长程灵巧任务。

## 2. 方法

**三步循环（Sec. IV，Alg. 1）：** (1) 采集：部署当前策略，每个 episode 标注成功/失败，可选专家实时干预纠正；(2) 训练 value function；(3) advantage-conditioned 策略训练。预训练阶段对全部示范数据做 (2)(3)；之后每个任务先用示范 SFT，再做 1 次或多次 (1)(2)(3)。每轮 value function 和策略都从预训练 checkpoint 重新微调（而不是接着上一轮），以避免漂移。

**奖励与 value（Sec. V-C）。** 通用稀疏奖励：r_t = 0（终止且成功）、−C_fail（终止且失败）、−1（其他）。于是 value = −（到成功的剩余步数），按任务最大长度归一化到 (−1, 0)。Value function p_φ(V | o_t, ℓ) 是分布式的：把经验回报离散成 B=201 个 bin，用交叉熵训练（Eq. 2，Monte Carlo 估计，即行为策略的 on-policy value）；V = Σ_b p(b)·v(b)。架构与 VLA 相同但 backbone 用 670M 的 Gemma 3，另混入少量网页多模态数据防过拟合。Fig. 4 的可视化显示它能定位 episode 中的失误（value 骤降处）和进度快慢。

**Advantage conditioning（Sec. IV-B）。** 从正则化 RL 的结论出发：若 π̂ ∝ π_ref · p(I | A^{π_ref})^β，I 表示"该动作优于 π_ref"，则 π̂ 保证改进（CFGRL，Frans et al. 2025）。用 Bayes 改写 p(I|A) = π_ref(a|I,o)/π_ref(a|o)，得
π̂(a | o, ℓ) ∝ π_ref(a | o, ℓ)·(π_ref(a | I, o, ℓ)/π_ref(a | o, ℓ))^β；
β=1 时 π̂ 就是条件策略 π_ref(a | I, o, ℓ)。所以只要训练一个既能表示 π(a|o,ℓ) 又能表示 π(a|I,o,ℓ) 的模型即可——类似 classifier-free guidance。指示符 I_t = 𝟙(A^{π_ref}(o_t, a_t, ℓ) > ε_ℓ)，ε_ℓ 是任务相关阈值：预训练取该任务 value 分布的 30% 分位（约 30% 数据为正），微调时取约 40% 的 rollout 为正，T 恤任务只取 10%（Appendix E）。用阈值而非测试时的 β 来调"正则化 vs 最优"，因为大 β 的 CFG 会把动作推到支撑集边缘、动作过激，且对自回归部分无效。训练目标 Eq. 3：所有数据都用监督学习，附加 α 加权的条件项；实践中改为 30% 概率 drop 掉指示符（等价于调 α），这样既能直接用 I=True 采样，也能做 CFG（β∈[1.5, 2.5] 时有用）。人类纠正段强制 I_t=True。Advantage 估计：post-training 用 N=50 步 lookahead 的 n-step 估计；预训练用 N=T 的 Monte Carlo 估计，可在训练时对每个样本只调一次 value function 在线算。

**π0.6 → π*0.6（Sec. V-A/B）。** π0.6 相对 π0.5：更多机器人平台的预训练数据；VLM 换成 Gemma 3 4B；action expert 增到 860M；训练用 Knowledge Insulation（KI）：backbone 同时预测文本 subtask ℓ̂ 和 FAST 离散动作，flow expert 单独用 flow 损失且停梯度、不影响 backbone。π*0.6 只是在 ℓ̂ 之后、动作之前插入文本 "Advantage: positive/negative"，只影响动作的似然。连续动作似然用 flow 损失作为下界（Appendix B）。输出 50 Hz 关节角 + 夹爪的 chunk。

**平台。** 静态双臂 6-DoF + 平行夹爪，三相机（中央 + 两腕），Fig. 5。

## 3. 实验

**任务（Sec. VI-A）：** laundry（T 恤/短裤，从筐里取出、展平、折叠、码放，200 s）；laundry diverse（11 类衣物，定量只报最难的衬衫，500 s）；laundry targeted failure removal（单件橙色 T 恤、领口朝上的严格标准，200 s）；cafe 双份浓缩（磨粉、压粉、锁把手、萃取、上杯，200 s，无重大失误）；box assembly（工厂场景，平板纸箱折成箱 + 贴标签 + 放入货箱，600 s）。指标：throughput（每小时成功次数）与人工标注的成功率。

**数据量（Appendix E）：** T 恤任务每轮 300 条自主 episode（4 台机器人，不用纠正）×2 轮；diverse laundry 450 评测 + 287 纠正；failure removal 约 1000 自主 + 280+378 纠正（3 台）；box 每轮 600 示范 + 360 纠正（3 台）；cafe 单轮 429 纠正 + 414 自主。

**Baseline：** π0.5 预训练、π0.6 预训练（纯监督）、π*0.6 RL 预训练、π*0.6 offline RL + SFT（示范 I 全为 True）、完整 RECAP；策略提取对照 AWR、PPO（DPPO/FPO 变体 + SPO 信赖域，η=0.01）。

**主结果（Fig. 7/8，柱状图估读）：** throughput（次/小时）laundry：π0.5 ≈3、π0.6 ≈26、π*0.6 预训练 ≈25、+SFT ≈37、RECAP ≈60；diverse laundry ≈0/≈4/≈5/≈4/≈8.5；espresso ≈0/≈3.5/≈9/≈15/≈28；box ≈0/≈2.5/≈1.5/≈10/≈13。成功率 laundry ≈15/63/75/90/95%；diverse ≈0/35/47/43/75%；espresso ≈0/10/25/38/92%。正文结论：从 SFT 到最终模型，diverse laundry 与 espresso 的 throughput 翻倍以上、失败率约减半；除 diverse laundry 外成功率都在 90%+；box 各子阶段（取板、折箱、贴标、入箱）RECAP 最高且最均衡，失败多因超时。

**多轮迭代（Fig. 9/10）：** laundry 纯自主数据两轮后 throughput 提升 50%（成功率第一轮就到 90%+，第二轮主要提速）；box 第二轮后 throughput 2×，折箱与贴标成功率约 90%。

**策略提取对比（Fig. 11）：** 同样的数据，AWR 成功率尚可但速度慢、throughput 低；PPO 在这种离线设定下需极小信赖域才能稳定，效果差；两者都难以超过 offline RL + SFT 起点，RECAP 的 throughput 远高于二者。

**去除特定失败模式（Fig. 12）：** 对抗性初始摆放 + 严格领口标准，两轮各 600 条纯自主数据后成功率 97%（起点估读约 20–30%）。

**部署：** 连续 13 小时做咖啡；新家庭折陌生衣物 2 小时以上无中断（Sec. I）。

## 4. 局限

作者承认（Sec. VII）：不是全自主——奖励标注、干预、场景复位都靠人；探索是贪心的，只靠策略随机性和人类干预；是"采一批、训一次"的迭代 offline RL，不是并发的 online RL；value 用 on-policy Monte Carlo 估计而非 off-policy Q。我读出来的：(1) 二值 advantage 丢掉了幅度信息，且阈值按分位数拍定；(2) 每个任务都要单独跑循环、单独收几百到上千条 episode，得到的是 specialist 而非 generalist 的提升（π0.7 随后试图把这些 rollout 蒸馏回通用模型）；(3) value function 只在训练时用，推理时没有当失败检测器使用；(4) 模型不开源，无法复现。

## 5. 复现要点

- 未开源（模型、数据、代码均无）。完整复现需要：Gemma 3 4B + 860M expert 的 KI 训练框架、670M value function、多台真机、人类标注与干预流程。
- 若在开源 π0.5 上模拟：可以用 openpi 训一个小 value head，把 advantage 二值化后作为 prompt 文本插入指令——方法本身与架构无关（CFGRL 思路）。8×H100 足够微调 3B 模型，真机数据采集是主要成本。
- 关键超参：B=201 bins；ε_ℓ 按分位数（30% / 40% / 10%）；advantage dropout 30%；N=50 lookahead；每轮从预训练 checkpoint 重新微调；纠正段强制正 advantage。
- 坑：奖励里 C_fail 要足够大；不同任务长度要归一化；CFG β 过大动作过激；PPO 在离线设定下不稳。

## 6. 关键引用链

**建立在：** π0.5 与 π0.6 模型卡（基座）；Knowledge Insulation（Driess et al. 2025，arXiv 2505.23705）；FAST；CFGRL（Frans, Park, Abbeel, Levine 2025，"Diffusion Guidance Is a Controllable Policy Improvement Operator"，arXiv 2505.23458）；reward-conditioned policies（Kumar et al. 2019）、Upside-Down RL、Decision Transformer（条件化思路）；AWR（Peng et al. 2019）；分布式 RL（Bellemare et al. 2017）；HG-DAgger（Kelly et al. 2019）；DPPO / FPO / SPO（PPO baseline）；Gemma 3。
**后续：** π0.7 把 π*0.6 的 RL rollout 数据（含失败）连同元数据一起放进预训练，宣称通用模型在同样任务上追平甚至超过这些 RL specialist；MEM 的 in-context 适应实验沿用了本文的纠正数据采集方式。
