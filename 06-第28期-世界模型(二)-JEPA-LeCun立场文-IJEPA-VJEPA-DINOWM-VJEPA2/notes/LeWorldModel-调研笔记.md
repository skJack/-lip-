# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels（LeWM）

> 对应「可lip」第 28 期视频（世界模型系列第二期）。论文：[arXiv:2603.19312](https://arxiv.org/abs/2603.19312)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2603.19312（v1 2026-03-13 据 arXiv 元数据；本地 PDF 为 v3 2026-06-03；用 NeurIPS 2026 模板、页眉标 Preprint，录用未核实）· 机构: Mila & Université de Montréal、New York University（Yann LeCun）、Samsung SAIL、Brown University（Randall Balestriero）；Maes 与 Le Lidec 共同一作 · 代码/权重: https://github.com/lucas-maes/le-wm（据 GitHub README 2026-09-11 查看：MIT；Hugging Face 上有 pusht / cube / tworooms / reacher 四个环境的 checkpoint 与 HDF5 数据集，Google Drive 另有 PLDM / DINO-WM / GCBC / GCIQL / GCIVL 等 baseline checkpoint；依赖作者自己的 stable-worldmodel 库）· 项目页: https://le-wm.github.io
- 一句话: 第一个从像素端到端稳定训练的 JEPA 世界模型：encoder（ViT-Tiny）和 latent predictor 联合训练，损失只有两项——下一步 latent 的 MSE 预测损失 + SIGReg（把 latent 分布拉向各向同性高斯的正则项，来自 LeJEPA）——不用 stop-gradient、EMA teacher、预训练 encoder、像素重建或 reward；约 15M 参数、单 GPU 几小时、每帧只压成 1 个 192 维向量，所以 CEM 规划一次 0.98 s、比 DINO-WM 快 48×，在 PushT / Reacher 上成功率还高于 DINO-WM 和 PLDM。它在世界模型发展史里改变的是一个前提："JEPA 型 latent WM 要么靠冻结大编码器（DINO-WM、V-JEPA 2-AC）、要么靠一堆启发式正则（PLDM 的 VICReg 七项损失）才不塌"——把 latent WM 拉回"小模型、单卡、从零训、一个超参"的范围，代价是每个环境单独训、只在仿真验证。

## 1. 要解决的问题
- JEPA（Joint Embedding Predictive Architecture，LeCun 2022 提出：encoder 把观测编成 latent，predictor 在 latent 空间预测未来，不重建像素）做世界模型的核心难题是 representation collapse（表征塌缩）：只优化"预测下一帧 latent"时，encoder 把所有输入映射成同一个常数就能让损失为零，表征毫无信息（3.1）。
- 现有防塌手段各有代价（Fig 2、Sec 2）：I-JEPA / V-JEPA 用 EMA teacher（目标 encoder 是在线 encoder 参数的指数滑动平均）+ stop-gradient，但作者引 Ponce et al. 2026 指出这两者并不对应任何明确定义的目标函数；DINO-WM、V-JEPA 2-AC、OSVI-WM 冻结预训练 encoder，避免了塌缩但表征上限被预训练知识锁死、也不再端到端；PLDM 是唯一端到端从像素的 JEPA WM，但用 VICReg 衍生的七项损失、六个权重，训练不稳、网格搜索 O(n^6)（附录 C.2）；Dreamer / TD-MPC 需要 reward 或特权状态，是 task-specific 的。
- 目标：端到端、从像素、reward-free、reconstruction-free、无启发式、单卡可训，且规划快到接近实时。

## 2. 方法
**预测空间：state-space（latent），每帧一个 192 维向量。** latent dynamics（在 latent 空间里学动力学）：z_t = enc(o_t)，ẑ_{t+1} = pred(z_t, a_t)（式 LeWM，3.1）；训练全离线、无 reward、无任务标签，轨迹只要覆盖动力学即可，不要求最优（3.1）。
- **Encoder：** ViT-Tiny（约 5M；patch 14、12 层、3 头、隐层 192，Hugging Face 实现），输入 224×224；取最后一层 [CLS] token，再过一个带 BatchNorm 的 1 层 MLP projector——这步不可省，因为 ViT 末尾的 LayerNorm 会让 SIGReg 优化不动（3.1）。与 DINO-WM 每帧 14×14 = 196 个 patch token 相比，token 数少约 200×（Fig 3 说明）。
- **Predictor：** transformer，6 层、16 头、dropout 0.1，约 10M（附录 D 称 ViT-S backbone）；动作经 AdaLN（Adaptive LayerNorm：用动作生成每层 LayerNorm 的 scale / shift）逐层注入，AdaLN 参数零初始化，让动作条件逐渐生效；输入历史 N 帧 latent（PushT / Cube N = 3，TwoRoom N = 1）配 temporal causal mask，自回归预测下一帧；predictor 后再接一个同样的 projector（3.1、附录 D）。总参数约 15M（摘要）。
- **动作怎么进 / 层级：** 各环境的 low-level 连续控制量（PushT 2D 推动、Reacher 关节、Cube 末端）；frameskip 5，连续 5 个动作打包成一个 action block 作为 predictor 一步的条件（附录 D）。
- **损失（式 2–4）：** L = ‖ẑ_{t+1} − z_{t+1}‖²₂ + λ·SIGReg(Z)。SIGReg（Sketched Isotropic Gaussian Regularizer，出自 LeJEPA）：把 N×B 个 latent 投影到 M 个随机单位方向 u^(m) 上，对每个一维投影算 Epps–Pulley 正态性检验统计量 T（经验特征函数与 N(0,1) 特征函数之差的加权积分，式 EP），对 M 取平均；Cramér–Wold 定理保证"所有一维边缘都是标准高斯 ⇔ 联合分布是各向同性高斯"（附录 A）。默认 M = 1024、λ = 0.1；M 和积分节点数对结果几乎无影响（Fig 15），所以 λ 是唯一有效超参，可二分搜索（4.3）。
- **没有 stop-gradient、没有 EMA、没有预训练 encoder**：梯度穿过损失的所有项（含目标 z_{t+1}），encoder / predictor / projector 联合优化（附录 D 的伪代码 Alg 3）。SIGReg 按时间步各自施加、不约束时间维；作者推测这正是 latent 轨迹自发"变直"的原因（附录 H）。
- **训练数据（附录 E）：** 每环境 1 万条离线轨迹——TwoRoom 10k 条 × 平均 92 步（噪声启发式策略）、OGBench-Cube 单方块 10k × 200（benchmark 自带采集器）、Reacher 10k × 200（SAC 策略）；PushT 沿用 DINO-WM 的 2 万条专家轨迹（平均 196 步）。全部只训 10 epoch（作者称已达 DINO-WM 论文的水平）；batch 128、子轨迹 4 帧 + 4 个动作块（附录 D）。
- **动作怎么"出来"——规划，不是 policy（3.2、附录 B / D）：** cost C = ‖ẑ_H − z_g‖²₂，z_g = enc(o_g)，即目标图像条件的终端 latent 距离（V-JEPA 2 论文把同样的东西叫 energy——energy-based model 的说法：一个标量函数给"预测状态-目标"配对打分，越低越相容，规划就是找让它最小的动作）；CEM（Cross-Entropy Method：从高斯采样动作序列、取 top-k 精英更新均值方差、迭代）：300 条候选、PushT 30 轮 / 其余 10 轮、top-30 精英、初始方差 1；planning horizon（一次规划向前看几步）H = 5 个动作块 = 25 环境步；MPC（Model Predictive Control：执行一段后用新观测重规划）形式是"整段 5 块执行完再重规划"，沿用 DINO-WM 设置。梯度型求解器差很多：SGD 26、RMSProp 67.3、Adam 84 vs CEM 96（Tab 10）。没有 policy、没有 IDM，动作纯粹是优化变量。
- **推理开销：** 完整一次规划 0.98 s vs DINO-WM 47 s（Fig 3 左，50 次平均；计时硬件未说明），即 48×；固定 FLOPs 预算下 DINO-WM 掉到 PushT 13 / Cube 48，LeWM 90 / 74（Fig 3 中、右）。训练：论文只说"单 GPU 几小时"，型号与时长未说明。
- **诊断工具：** 事后单独训的 cross-attention decoder（196 个 query token → 224×224，附录 D）只用于可视化（Fig 7、9、10）；surprise = 预测 latent 与实际 latent 的差，用于 5.2 的 VoE 实验。


## 3. 实验
Setting（4.1、附录 F.1）：4 个仿真环境——TwoRoom（2D 导航）、Reacher（DM Control 双关节）、PushT（2D 推 T 块）、OGBench-Cube（3D 单臂拾放单方块，Fig 5）；起点从离线轨迹随机采，目标图取同一条轨迹 25 步之后的帧，执行预算 50 步（即最多两轮规划），50 条评测轨迹报成功率；baseline：DINO-WM（默认去掉 proprioception 以公平，另报 +prop）、PLDM（作者对其六个损失权重在 PushT 上做了 256 组网格搜索后跨环境固定，Tab 2）、GCBC、GCIQL、GCIVL（后三者用 DINOv2 patch 特征编码观测和目标，附录 C.3–C.4）、Random；所有方法超参跨环境固定。
- Fig 6（主结果，柱上标注）：PushT LeWM 96 / DINO-WM+prop 92 / PLDM 78 / DINO-WM 74 / GCBC 75 / GCIQL 20 / GCIVL 33 / Random 2；Reacher LeWM 86 / PLDM 78 / DINO-WM 79 / Random 10；OGBench-Cube LeWM 74 / PLDM 65 / DINO-WM 86 / GCBC 84 / GCIQL 64 / GCIVL 56 / Random 48；TwoRoom LeWM 87 / DINO-WM+prop 100 / PLDM 97 / DINO-WM 100 / GCBC、GCIQL、GCIVL 均 100 / Random 0。
- 读法：接触丰富的 PushT 上纯像素 LeWM 比 PLDM 高 18 点、还超过带 proprio 的 DINO-WM（4.2）；视觉最复杂的 3D Cube 上输给冻结 DINOv2 的 DINO-WM 12 点、也输 GCBC；最简单的 TwoRoom 反而最差——作者归因于数据多样性、内在维度太低，encoder 难以把它们匹配到高维各向同性高斯（4.2、Sec 6）。
- Tab 5（3 seed × 50 条，PushT）：LeWM 96.0 ± 2.83、DINO-WM 92.0 ± 1.63、PLDM 78.0 ± 5.0——端到端前作 PLDM 方差最大；这里的 DINO-WM 数值与 Fig 6 的 +prop 一栏一致，应是带 proprio 的版本，正文未点明。
- 最有信息量的消融 I（全在 PushT，附录 G）：predictor dropout 0 / 0.1 / 0.2 / 0.5 → 78 / 96 / 85.3 / 66.7（Tab 9，不加 dropout 直接掉 18 点）；predictor 尺寸 tiny / small / base → 80.7 / 96.0 / 86.7（Tab 6，更大不更好）；加像素重建损失 96.0 → 86.0（Tab 7，与 DINO-WM 的结论一致）；encoder 换 ResNet-18 94.0（Tab 8，对 backbone 不敏感）。
- 消融 II：λ ∈ [0.01, 0.2] 均 > 80，峰值 0.09 处约 98，默认 0.1 处约 80，0.5 时掉到约 54（Fig 16，估读；正则压过预测损失）；embedding 维度 8 / 24 / 96 / 192 / 384 时 LeWM 约 42 / 74 / 92 / 94 / 92、PLDM 约 46 / 52 / 78 / 78 / 84（Fig 15 左，估读），作者给的阈值约 184 维；随机投影数、积分节点数几乎无影响（Fig 15 中、右）。注意：论文没有 λ = 0 的数字，"只用预测损失会 collapse"只在 3.1 陈述。
- 训练曲线（Fig 18 / 19）：LeWM 两项损失单调下降、SIGReg 早期迅速降到平台；PLDM 七项中多项震荡、非单调（4.3）。
- 物理探针（Tab 1，PushT，线性探针 MSE）：agent 位置 LeWM 0.052 / PLDM 0.090 / DINO-WM 1.888；block 位置 0.029 / 0.122 / 0.006；block 角度 0.187 / 0.446 / 0.050——位置线性可读，角度不如 DINOv2（作者归因于 DINOv2 用了两个数量级更多的数据）。Cube（Tab 4）：三者对 block yaw / quaternion 都读不出（r < 0.3），DINO-WM 在关节速度、末端 yaw 上明显更好，LeWM 在末端位置、方块位置上最好。TwoRoom（Tab 3）：LeWM 探针与 PLDM 持平，说明那里的规划差距不来自表征。
- VoE（violation-of-expectation，5.2，Fig 8）：物体瞬移（物理违背）在三个环境都引起显著 surprise 尖峰（paired t-test p < 0.01），颜色突变（视觉扰动）不显著——对物理比对外观敏感；对照 DINO-WM 在 Cube 上对两种扰动都不显著（Fig 14）。
- 附录 H（Fig 17）：latent 轨迹相邻速度的余弦相似度随训练上升，比带显式时间平滑项的 PLDM 还直，是无正则下的涌现现象。

## 4. 局限
- 作者承认（Sec 6）：规划只能短 horizon，长程要分层世界模型；依赖覆盖充分的离线数据，且数据多样性低时 SIGReg 在低维简单环境反而有害（TwoRoom）；需要动作标签，可用 IDM 缓解；大规模视频预训练是未来方向。
- 我读出来的（一，实验设定）：全是仿真、4 个小环境、单视角 224×224，没有真机；intro 说评了 locomotion，实际没有 locomotion 环境；评测目标取自同一条轨迹 25 步之后，全在分布内；Cube 上 Random 也有 48，说明成功判据宽松；"整段 5 块执行完再重规划"接近开环，闭环纠错能力未测；只有 3 seed。
- 我读出来的（二，方法本身）：每帧压成 1 个 192 维向量既是速度来源也是天花板——Cube 上末端朝向、方块旋转都读不出（Tab 4、Fig 7），视觉最复杂的任务输给冻结 DINOv2 12 点，"端到端小 encoder"在 3D 场景的上限还没证明；predictor 是确定性 L2 回归，不能表达多模态未来，也没有 rollout 损失；λ 峰值 0.09 与默认 0.1 相差约 18 点（Fig 16 估读），"一个超参"不等于不敏感。
- 我读出来的（三，比较与报告）：48× 是相对 DINO-WM 的 196 token 算的，来自 token 数而非算法本身，且规划硬件未说明；PLDM 基线的权重是作者自己搜的、DINO-WM 默认被去掉 proprio，公平性各有争议；训练算力只有"单 GPU 几小时"一句；每个环境单独训一个模型，没有跨环境 / 跨任务的通用性实验，也没有语言接口。

## 5. 复现要点
- 开源（据 GitHub README，2026-09-11 查看，未实际跑过）：MIT；代码 + 四个环境的 checkpoint（Hugging Face）+ HDF5 数据集（默认解压到 ~/.stable-wm/）+ baseline checkpoint（Google Drive）；Python 3.10 + uv，核心依赖 stable-worldmodel[train,env]、stable-pretraining，配置用 Hydra，训练要配 WandB。
- 规模 / 算力：encoder ViT-Tiny 约 5M + predictor 约 10M；论文只说单 GPU 几小时、10 epoch（最大数据集 PushT 2 万条 × 196 步）。8×H100 上可并行跑四个环境 × 多 seed × λ 扫描，毫无压力；瓶颈是环境依赖（OGBench 的 MuJoCo、DM Control）而不是显卡。
- 坑（以论文消融为准）：[CLS] 后必须接带 BatchNorm 的 projector，直接用 LayerNorm 输出 SIGReg 训不动（3.1）；AdaLN 零初始化；predictor dropout 0.1 是 78 → 96 的差别（Tab 9）；predictor 用 ViT-S，更大反而掉（Tab 6）；embedding 维度别低于约 184（Fig 15）；λ 在 [0.01, 0.2] 内二分搜，默认 0.1 不是峰值（Fig 16）；别加重建损失（Tab 7）；规划用 CEM 而不是梯度（Tab 10）；frameskip 5、历史 N = 3、horizon 5；TwoRoom 这类低维环境 SIGReg 可能有害；PLDM 基线的六个权重要自己搜（作者搜了 256 组）。
- 论文未说明：GPU 型号、训练墙钟时间、规划计时硬件、Reacher 的历史长度 N、planning 时 predictor 的 dropout 是否关闭。

## 6. 关键引用链
- 建立在（JEPA 一脉）：JEPA / H-JEPA 立场文（`LeCun-2022-立场文`）——本文是它"latent 预测 + 规划"路线的最小实现；LeJEPA / SIGReg（Balestriero & LeCun 2025，arXiv 2511.08544）——防塌正则直接搬来；PLDM（Sobal et al. 2025，Stress-Testing Offline Reward-Free RL）——唯一的端到端从像素 JEPA WM 前作、主要对照，TwoRoom 环境也出自它；JEPA "关注慢特征"（Sobal et al. 2022）解释了 Fig 10 早期 decoder 只重建慢变量的现象；I-JEPA / V-JEPA（`V-JEPA`）与 V-JEPA 2（`vjepa2`）——被列为"EMA + stop-gradient"和"冻结预训练 encoder"两条路线的代表，本文要去掉的正是它们的启发式。
- 建立在（环境与规划）：`dino-wm`——PushT 数据、Reacher 设置、CEM / MPC 配置全部沿用，差别只是冻结 DINOv2 特征 vs 端到端 encoder；Dreamer（`Dreamer-v3`）、TD-MPC2——task-specific 的对照；OGBench（Park et al. 2025）提供 Cube 环境与 GCIVL / GCIQL 基线；VICReg（Bardes et al. 2022）是 PLDM 损失的来源；Epps–Pulley 检验（1983）与 Cramér–Wold 定理；VoE 评测借自 Garrido et al. 2025（IntPhys 一脉）。
- 与 V-JEPA 2-AC 的对照（`vjepa2`）：两者都是"latent JEPA + 图像目标 CEM"，但一个是 1B 冻结编码器 + 300M 后训练预测器、16 s / 动作、真机 Franka；另一个是 15M 全可训、< 1 s / 规划、仅仿真。V-JEPA 2-AC 用冻结 encoder 回避 collapse，LeWM 用 SIGReg 正面解决；作者把"在大规模视频上预训练"列为未来方向，即把 SIGReg 配方推到 V-JEPA 2 的数据规模，两条线在那里会合。
- 后续 / 同组：Causal-JEPA（Nam, Le Lidec, Maes, LeCun, Balestriero 2026，arXiv 2602.11389，对象级 latent 干预）；Temporal Straightening for Latent Planning（Wang et al. 2026，arXiv 2603.12231，本文附录 H 呼应）；tutorial（arXiv 2607.00836，Section 2.2）把它归为 state-space / latent state 预测式一类；survey（综述 arXiv:2605.00080）Sec 4.2 列为 MPC 评测器、8.2 列为 latent WM 省算力的例子。
