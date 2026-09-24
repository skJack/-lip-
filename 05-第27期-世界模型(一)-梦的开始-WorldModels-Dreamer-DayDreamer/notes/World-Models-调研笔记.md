# World Models（Ha & Schmidhuber 2018；NeurIPS 版题为 Recurrent World Models Facilitate Policy Evolution）

> 对应「可lip」第 27 期视频（世界模型系列第一期）。论文：[arXiv:1803.10122](https://arxiv.org/abs/1803.10122)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 1803.10122（本地 PDF 是 v4 2018-05-09；v1 据我所知 2018-03-27，未核实；NeurIPS 2018 的版本改题为 "Recurrent World Models Facilitate Policy Evolution"，未核实；tex 用的是 icml2018 模板）· 机构: Google Brain（Ha）、NNAISENSE 与 Swiss AI Lab IDSIA, USI & SUPSI（Schmidhuber）· 代码/权重: 论文只给交互版文章 https://worldmodels.github.io（Distill 风格，训练好的 V/M/C 用 deeplearn.js 直接在浏览器里跑，等于公开了权重）；实验代码据我所知在 https://github.com/hardmaru/WorldModelsExperiments（TensorFlow 1 + estool，含 CarRacing / DoomRNN 训练脚本与模型文件，许可证未核实）· 项目页: https://worldmodels.github.io（tex 源注释里给的 DOI 10.5281/zenodo.1207631）
- 一句话: 把 agent 拆成三块——V（VAE 把 64×64 帧压成 latent z）、M（MDN-RNN 预测下一个 z 的混合高斯）、C（867 参数的线性层，用 CMA-ES 进化而非梯度训练）；V、M 用随机策略数据无监督训练、完全不看 reward，C 只吃 [z_t, h_t] 就在 CarRacing-v0 上第一次达到"解决"标准（906 ± 21），并在 VizDoom Take Cover 里把 C 完全放进 M 生成的"梦"里训练再迁回真实环境（1092 ± 556，Table 2）。它是深度学习时代 "world model" 一词的流行起点：第一次在像素级 RL 上把"在 latent 空间做动态预测"、"在想象中训 policy"、"用采样温度 τ 抑制 policy 钻模型漏洞"三件事串起来；PlaNet / Dreamer 直接继承前两件（见第 7 节），并抛弃了它的 V/M 分开训练、进化 controller 和无 reward 头。

## 1. 要解决的问题
- model-free RL 在实践中只敢用小网络：credit assignment 是瓶颈，几百万参数的大模型学不动（Intro）。作者的分工方案是"大 world model + 小 controller"：WM 有可微损失、能用 backprop 在 GPU 上高效训练，C 很小所以可以交给进化算法在低维空间里解 credit assignment。
- 从原始 RGB 像素直接学出时空表征，不靠手工特征（LIDAR、角度、边缘检测、frame stacking——之前 CarRacing 上的 DQN / A3C 都要这些预处理，3.3 节）。
- 学到的 dynamics model 能不能完全替代真实环境来训练 policy 再迁移回去（4.1 节）？此前 Oh 2015、Chiappa 2017 学了动作条件的模拟器但没敢用它替代环境，因为确定性模型很容易被 policy 利用（4.5 节）。
- 作者自我定位：不是综述也不是新方法，而是把 Schmidhuber 1990–2015 一系列 "RNN world model + controller"（C–M）论文的核心概念用现代工具做一个简化的实验演示，术语沿用 "On Learning to Think"（2015）（Intro）。

## 2. 方法
**预测空间：state-space（latent），重建式。** 三个模块（Section 2，Fig 4 / Fig 8）：
- **V（ConvVAE，A.1）：** 输入 resize 到 64×64×3，4 层 stride-2 conv 出 μ、σ，z ~ N(μ, σI)，4 层 deconv 重建；损失 = L2 重建 + KL；只训 1 epoch。N_z = 32（CarRacing）/ 64（Doom），参数 4.35M / 4.45M（3.2 / 4.2 节的参数表）。作者强调高斯先验虽然限制了信息量，但让 WM 对 M 生成的"不真实的 z"更鲁棒。
- **M（MDN-RNN，2.2 节、A.2）：** LSTM（256 / 512 hidden）+ Mixture Density Network 输出层，建模 P(z_{t+1} | a_t, z_t, h_t)：5 个高斯分量、对角协方差、不建模相关系数 ρ。Doom 版额外输出下一帧的 done 概率 d_{t+1}，>50% 就判死（作者说死亡是低概率事件，截断比从 Bernoulli 采样稳定）。teacher forcing 训练 20 epoch，每次组 batch 都从存好的 μ、σ 重新采样 z 以免过拟合某个具体 z。采样时有温度 τ（借自 SketchRNN），控制下一 z 的随机程度。参数 0.42M / 1.68M。
- **C（2.3 节，式 1，A.3）：** a_t = W_c[z_t, h_t] + b_c，单层线性，tanh 限幅。CarRacing 输入 32 + 256 = 288 维 → 3 维连续动作（转向、油门、刹车），共 867 参数；Doom 输入 z(64) + LSTM 的 h(512) 和 c(512) = 1088 维 → 1 维连续值分三段映射成左 / 不动 / 右，1088 参数（无偏置）。
- **训练流程（3.2 / 4.2 节，两个任务一样）：** (1) 随机策略采 10,000 条 rollout；(2) 训 V；(3) 用 V 把所有帧编码成 z，训 M；(4) 定义 C；(5) 用 CMA-ES（Covariance-Matrix Adaptation Evolution Strategy，一种只需要每个候选解的总回报、不需要梯度的黑盒优化；A.4：种群 64，每个个体跑 16 个随机 seed 取平均回报当 fitness）优化 C。V 和 M 分开训（3.1 节脚注：端到端可行但分开更实用，各自单 GPU 不到 1 小时）；V、M 完全不看 reward，只有 C 看 reward（3.1 节）。
- **动作怎么进、怎么出：** a_t 和 z_t 一起作为 LSTM 输入更新 h_{t+1}（2.4 节伪代码）；动作由 C 直接反应式输出，没有 lookahead 规划——作者明说 agent 不需要 plan ahead，因为 h_t 已含"对未来的概率分布"，像棒球击球手一样凭本能出手（3.3 节）。
- **WM 的两种用法：** CarRacing 里 WM 只当特征提取器（C 在真实环境里进化）；Doom 里把 M 包成 gym.Env 接口的虚拟环境 DoomRNN，C 完全在 latent 梦里训练（不需要 V 解码任何像素），学完直接部署到真实 VizDoom（4.2 节）。
- **"Cheating the world model"（4.5 节）：** 早期实验里 C 找到了对抗策略——在梦里怪物永远不发火球、火球刚成形就被"魔法般熄灭"（Fig 18）。原因：M 只是近似的概率模型，而 C 拿到了 M 的全部隐状态，等于拿到了游戏引擎的内部状态，可以直接操纵它；policy 会跑到训练分布之外、模型出错的地方。作者据此解释为什么此前的模拟器工作不敢用模型替代环境，也指出 PILCO 一类贝叶斯不确定性只能部分缓解。
- **对策：MDN + 温度 τ（4.3 / 4.5 节）：** 用 MDN 而非确定性 RNN，即使真实环境是确定性的也把它当随机环境训；τ 越高梦越随机越难，在"真实性 vs 可利用性"之间折中，作者发现高 τ 下训好的 agent 在真实环境里反而更好。混合分布的离散分量对"怪物开不开火"这种离散随机事件很关键：τ = 0.1 时 mode collapse，怪物在梦里从不开火，梦里满分、真实里崩溃。
- **迭代训练流程（Section 5，只是提出、没做）：** 初始化 M、C → 真实环境采 N 条 → 训 M 建模 P(x_{t+1}, r_{t+1}, a_{t+1}, d_{t+1} | x_t, a_t, h_t) 并在 M 里训 C → 循环；把 M 的预测损失取负作为 curiosity 奖励驱动探索；M 吸收了运动技能后 C 可以专注更高层技能。这正是后来 Dreamer 系列真正实现的循环（Dyna 式：先采集、再在模型里训 policy、再采集）。
- **推理开销：** V 编码 4 层 conv + 一步 LSTM + 一次矩阵乘，极轻；DoomRNN 不用渲染、不跑游戏引擎，比 VizDoom 便宜（A.5）；wall-clock 数字论文没给。


## 3. 实验
Setting：CarRacing-v0（Box2D 俯视赛车，赛道每局随机生成，3 维连续动作，官方"解决"标准 = 100 次连续 trial 平均 ≥ 900）；DoomTakeCover-v0（VizDoom，躲对面怪物的火球，回报 = 存活步数，每局最多 2100 步约 60 s，"解决" = 100 次连续 rollout 平均 > 750 步）。baseline 是 Gym 排行榜、DQN、A3C、随机策略；没有 model-based baseline。
- Table 1（CarRacing，100 次随机 trial）：DQN 343 ± 18、A3C 连续 591 ± 45、A3C 离散 652 ± 10、Gym 榜首 838 ± 11；只给 z 的线性 C 632 ± 251、z + 40 单元 tanh 隐层（1443 参数）788 ± 141、Full World Model（z + h）906 ± 21——首个报告的"解决"该任务的方法。A.4 / Fig 24：CMA-ES 跑 1800 代，最佳个体 1024 次评测均值 900.46；Fig 25 是 906 ± 21 的直方图。
- 最有信息量的消融就是 Table 1 的 V-only vs V + M：z 只含当前帧、没有预测能力，agent 摇摆并在急弯冲出（Fig 11）；加上 h_t 后驾驶稳定、能攻急弯（Fig 12）。结论：M 的隐状态本身就是好的控制特征，参数量不变（867）只是换了输入。
- Table 2（Take Cover：在不同 τ 的梦里训 C，然后在真实环境测 100 次）：τ = 0.10 梦 2086 ± 140 / 真实 193 ± 58；0.50 → 2060 ± 277 / 196 ± 50；1.00 → 1145 ± 690 / 868 ± 511；1.15 → 918 ± 546 / 1092 ± 556；1.30 → 732 ± 269 / 753 ± 139；随机策略 210 ± 108，Gym 榜首 820 ± 58。低 τ 时梦里接近满分、真实里比随机策略还差；τ 太高梦太难学不到东西；1.30 分数低于 1.15 但方差小（更保守的策略）。作者把 τ 当需要调的超参。
- 梦内 vs 真实：τ = 1.15 的最佳个体在 DoomRNN 里 1024 次均值 959（A.5，Fig 28），迁到真实 VizDoom 反而 1092 ± 556（Fig 29）——高温的梦比真实环境更"脏"，所以迁移后分数更高（4.4 节）。
- 定性（4.3 / 4.4 节）：M 仅从随机 episode 的像素学会了游戏逻辑——左移、墙壁阻挡、同时追踪多个火球的轨迹、判定死亡；V 连怪物数量都数不准，但不影响策略迁移。
- 没有的：每个环境只有一条训练曲线，没有多 seed 统计；A.5 明说从未在真实 VizDoom 里训过 C，所以没有"梦里训 vs 真实训"的对照；没有样本效率 / wall-clock 对比。

## 4. 局限
- 作者承认（4.5 节、Section 7）：(1) 单独训的 VAE 会编码任务无关细节（Doom 墙上的砖纹）却漏掉任务相关细节（CarRacing 路面 tile）；和预测 reward 的 M 联合训练能聚焦任务，但代价是 VAE 不能跨任务复用；(2) C 能利用 M 的缺陷，τ 只是折中不是解决；(3) LSTM 容量有限、灾难性遗忘，未来要换更大模型（attention 等）或外部记忆；(4) 逐步模拟未来、没有层级规划或抽象推理；(5) 随机策略数据只够简单任务，复杂环境需要 Section 5 的迭代采集（未做）；(6) 作者指出可以用 WM 的可微性直接 backprop 训 policy，但本文没做。
- 我读出来的：(1) 两个环境都极简（俯视 2D / 单走廊、单任务、64×64、无接触物理），离机器人操作很远；(2) CMA-ES 只能处理几千参数，controller 被迫是线性层，policy 能力上限低；(3) CarRacing 里没验证"在梦里训"，Doom 里没验证"当特征"，两种用法各只测了一个环境；(4) τ 是全局标量，Table 2 的调参本身依赖真实环境评测，每换环境都要扫；done 的 0.5 阈值、5 个混合分量都是手工选择；(5) 随机策略数据在 CarRacing 上覆盖不足（作者自己说路面 tile 重建差），真机上更不可能靠 10,000 条随机 rollout 起步；(6) 单次运行、无 seed 方差，VizDoom 迁移的 ±556 方差极大；(7) M 只预测 z（和 done），没有 reward 头，所以 latent 与任务无关——这既是它"可复用"的优点，也是后来 PlaNet 立刻改掉的地方。

## 5. 复现要点
- 代码：官方交互文章 worldmodels.github.io（浏览器 demo 即可验证 V / M / C 行为）；实验仓库据我所知是 hardmaru/WorldModelsExperiments（TF1，CMA-ES 来自作者自己的 estool），许可证未核实；第三方 PyTorch 复现很多（如 ctallec/world-models，未核实）。
- 数据：全部自采，每个环境 10,000 条随机 rollout。环境依赖是主要坑：DoomTakeCover-v0 已从 gym 移除、需要旧版 gym 的 doom 包或 vizdoom 自己封装（未核实）；CarRacing-v0 在 gymnasium 里已升到 v2 / v3，分数不能直接比（未核实）。
- 算力（3.1 节脚注、A.4、致谢）：V、M 各单 GPU 不到 1 小时；CMA-ES 在 64 核 CPU 机器上跑，种群 64 × 16 rollouts = 每代 1024 次仿真，CarRacing 1800 代，Google Cloud 虚拟机。8×H100 对 V / M 完全过剩，瓶颈是 CPU 核数（Box2D / VizDoom 仿真）；在 实验室集群上要先看每台机器的 CPU 核数而不是显卡。
- 坑：MDN-RNN 训练时要从 μ、σ 重采样 z；done 用 0.5 阈值而不是采样；τ 必须扫（Table 2 从 0.1 到 1.3 结果天壤之别）；CarRacing 的"解决"标准要 100 次连续 trial 平均 ≥ 900，单次评测方差大；VAE 只训 1 epoch；随机策略在 CarRacing 上覆盖不足；CMA-ES 结果对 seed 敏感而论文只报一次运行。

## 6. 关键引用链
- 建立在（Schmidhuber 一线）：1990 "Making the World Differentiable"（FKI-126-90，RNN 控制器 + RNN 世界模型，Fig 20 / 21 直接翻印 1990 年的图）、IJCNN 1990 在线算法、NeurIPS 1991 "RL in Markovian and non-Markovian environments"、1990 curiosity & boredom（Section 5 的 curiosity 来源）、"On Learning to Think"（2015，arXiv 1511.09249；本文术语和迭代训练流程的出处）、One Big Net（2018，C 与 M 合一）。
- 建立在（技术组件与同行）：VAE（Kingma & Welling 2013）；MDN（Bishop 1994）+ Graves 2013 手写生成 + SketchRNN（Ha & Eck 2017，τ 的来源）；CMA-ES（Hansen & Ostermeier 2001；Hansen 2016 教程）与作者自己的 Evolving Stable Strategies（种群 64 × 16 seed 的设置）、Salimans 2017 ES；PILCO（Deisenroth & Rasmussen 2011，用贝叶斯不确定性抑制模型被利用）；E2C（Watter 2015）、Wahlström 2015、Finn 2016 deep spatial autoencoders（先压缩像素再学动态）；Oh 2015 action-conditional video prediction、Chiappa 2017 recurrent environment simulators（学了模拟器但没用它替代环境）；I2A（Weber 2017）、Predictron（Silver 2016）；Alvernaz & Togelius 2017（VAE + 神经进化打 Doom，作者称与本文最相似）。
- 后续继承（本仓库 slug）：PlaNet 保留"在 latent 空间做动态预测"和随机 latent（RSSM 的 stochastic 部分接替了 MDN 的角色），把 V、M 合并成一个联合训练的模型；Dreamer-v1 保留"在想象里训 policy"，并把本文 Section 7 只提了一句的"用 WM 的可微性 backprop 训 policy"真正做出来（actor-critic 在 latent 想象里训练）；Dreamer 系列的 collect → train WM → train in imagination 循环就是本文 Section 5 纸上的迭代流程。
- 后续继承（更远）：Dreamer-v2、Dreamer-v3 继续这条线，DayDreamer 把它搬到真机；Genie 把"在生成的环境里训 agent"放大到互联网视频规模；综述 arXiv:2605.00080 把本文列为机器人 WM 的概念源头（其笔记第 7 节）；WM-to-WAM-教程 未引用。
- 后续抛弃：
  - V / M 分开训练——PlaNet 起 encoder、dynamics、reward 头联合训练，让 latent 带任务信息（本文 Section 7 自己预见了这个取舍）。
  - CMA-ES 进化线性 controller——Dreamer 改用梯度训练的 MLP actor-critic，controller 不再受"几千参数"限制。
  - 无 reward 预测头——PlaNet / Dreamer 都加 reward（DreamerV3 再加 continue）头；本文只有 Doom 版的 done 头。
  - 温度 τ 这个旋钮——后续靠短 horizon 想象（DreamerV3 是 15 步）、随机 latent 和 KL 正则来限制 policy 钻漏洞，不再显式调采样温度。
  - 像素重建本身——MuZero（arXiv:1911.08265） 改为只要求 value / reward 等价；TD-MPC2（arXiv:2310.16828）、dino-wm、vjepa2、LeCun-2022-立场文（JEPA）改为不重建像素，正是回应本文 Section 7 "VAE 编码任务无关细节"的局限。
