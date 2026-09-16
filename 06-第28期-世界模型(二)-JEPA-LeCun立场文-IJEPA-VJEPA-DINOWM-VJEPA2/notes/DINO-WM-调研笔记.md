# DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

> 对应「可lip」第 28 期视频（世界模型系列第二期）。论文：[arXiv:2411.04983](https://arxiv.org/abs/2411.04983)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2411.04983（v1 2024-11-07；LaTeX 源码 `main.tex` 用的是 `\usepackage[accepted]{icml2025}`，即 ICML 2025 录用后的定稿版）·· 机构: New York University（Yann LeCun 兼 Meta-FAIR，据项目页）· 代码/权重: https://github.com/gaoyuezhou/dino_wm（项目页给出；论文称 code and models 已开源）· 项目页: https://dino-wm.github.io
- 一句话: 把冻结的 DINOv2 patch 特征当 world model 的状态，用一个约 19M 的 ViT 在离线随机轨迹上学 latent dynamics（不重建像素、不预测 reward），测试时用 CEM / MPC 直接在特征空间对"目标图像"做零样本规划；在 6 个仿真环境里把离线 WM 的规划成功率从 DreamerV3 的水平大幅拉高（PushT 0.30→0.90），是 V-JEPA 2-AC 的直接前身。

## 1. 要解决的问题
- feed-forward policy 部署后不再优化，泛化要求训练时见过所有情形；world model 允许测试时做行为优化。
- 但已有 WM 两头都不通用：在线 WM（Dreamer、TD-MPC2）靠 reward 学 latent，是 task-specific 的，且需要环境访问；离线 WM 又要额外的辅助信息——专家演示、关键点、预训练 inverse dynamics model 或稠密 reward。
- 像素空间预测（diffusion）算力太贵不适合 MPC；latent 预测又通常绑着重建目标或 reward。核心问题：有没有一种不牺牲通用性的"辅助信息"？答案是互联网预训练的视觉表征。

## 2. 方法
**预测空间：state-space（latent）。** 三个模块（3.1）：observation model z_t = enc(o_t)、transition model p_θ(z_{t+1} | z_{t-H:t}, a_{t-H:t})、可选 decoder q_θ(o_t | z_t)（只用于可视化）。
- **Backbone / encoder：** 冻结的 DINOv2（输入 196×196，输出 14×14×384 的 patch 特征，对应 ViT-S/14，A.9）。用 patch 特征而不是 CLS 全局向量是关键（Table 2）。
- **transition model：** 去掉 tokenizer 的 ViT（decoder-only transformer），深度 6、16 头、MLP 2048，约 19M 参数（A.9）；帧级 causal attention——z_t 的每个 patch 只看 z_{t-H:t-1} 的所有 patch，一次预测整帧，而不是 IRIS 那样逐 token 自回归。
- **动作怎么进：** 动作经 MLP 映射为 10 维向量后拼接到每个 patch 向量上；proprioception 同样拼接。动作层级是各环境的 low-level 控制量（2-DoF 力、pusher 位移、关节、XArm 末端）。
- **损失（式 1）：** teacher forcing 的 latent 一致性 L_pred = ‖p_θ(enc(o_{t-H:t}), φ(a_{t-H:t})) − enc(o_{t+1})‖²，全程不重建像素；decoder 用式 (2) 单独训练，且不能把重建梯度回传给 predictor（否则 PushT 0.92→0.80，Table 7）。
- **规划（3.2，A.5）：** 给定当前图 o_0 和目标图 o_g，cost C = ‖ẑ_T − z_g‖²，ẑ_t = p(ẑ_{t-1}, a_{t-1})；CEM 采样 N 条动作序列、取 top-K 更新高斯，执行前 k 个动作后重规划（MPC）。梯度下降规划可行但明显更差（PushT：GD 0.28、开环 CEM 0.86、MPC 0.90，Table 8）。
- **cascaded 还是 joint：** 都不是——纯 WM + 优化器，动作是优化变量，没有 policy / IDM。
- **训练数据：** 每个环境的离线随机或加噪轨迹（Table 11）：PointMaze 2000 条×100 步、Wall 1920×50、Reacher 3000×100、PushT 18500 条（专家回放加噪）、PushObj 20000、Rope / Granular 各 1000 条（A.1 写 20 步、Table 11 写 5 步，原文不一致）；H = 1 或 3，frameskip 5（形变环境 1）；共享超参 224 分辨率、AdamW、100 epoch、batch 32（Table 12）。
- **推理频率与延迟（Table 10，A6000）：** 单步前向 0.014 s（batch 32）；一次 CEM 规划（100 样本×10 轮）53 s——非实时，但比 Flex 形变仿真（3.0 s / 步）快得多。
- **控制用途：** 目标图像条件的零样本规划（无 reward、无演示、无 IDM）。


## 3. 实验
Setting：6 个环境（Maze、Wall、Reach、PushT、Rope、Granular），目标由随机采样的目标图像给出，前四个报 50 例成功率 SR，后两个报 10 例 Chamfer Distance；baseline IRIS、DreamerV3、TD-MPC2 都在同样离线数据上无 reward 训练后做 MPC。
- Table 1：DINO-WM SR 0.98 / 0.96 / 0.92 / 0.90，CD 0.41 / 0.26；DreamerV3 1.00 / 1.00 / 0.64 / 0.30，CD 2.49 / 1.05；IRIS 0.74 / 0.04 / 0.18 / 0.32；TD-MPC2 全 0（无 reward 学不出 latent）。简单导航持平，接触丰富的操作任务差距最大。
- Table 2（换 encoder）：Wall / Reach / PushT 上 DINO CLS 0.58 / 0.60 / 0.44、R3M 0.34 / 0.40 / 0.42、ResNet 0.12 / 0.06 / 0.20 vs DINO patch 0.96 / 0.92 / 0.90——空间 patch 特征是关键。
- Table 3（未见配置）：WallRandom 0.82 vs DreamerV3 0.76；PushObj（新形状）0.34 vs 0.18，对所有方法都难；GranularRandom CD 0.63 vs 最好 baseline 0.98。
- Table 4 / 9：预测帧 LPIPS PushT 0.007 vs AVDC 0.046、DINO CLS 0.039（Intro 称最难任务上比前作好 56%）。
- 最有信息量的消融：Table 5 数据量 200→1000→18500 条，PushT SR 0.08→0.48→0.92；Table 6 causal mask：H = 3 时有 mask 0.92、无 mask 0.08（无 mask 训练时会偷看未来帧）。
- Fig 6 / 7：AVDC 生成的视频好看但物理不合理，动作条件版长程 rollout 发散。

## 4. 局限
- 作者承认：需要覆盖充分的离线 state-action 数据；必须有真实动作标签，无法直接用互联网视频；只在原始动作空间规划，细粒度任务需要分层。
- 我读出来的：全部是仿真、单视角 224×224、多为 2D / 小场景；"随机轨迹就能覆盖"在真机上不现实；目标 cost 是特征空间 MSE，要求目标图与当前视角一致；predictor 是确定性 L2 回归，不能表达多模态未来；只有 teacher forcing 没有 rollout 损失，长程会累积误差；DINOv2 是图像模型，速度等动态信息只能靠历史 H 帧推断；一次规划 53 s 远非实时；PushObj 0.34 说明对未见物理参数的泛化仍弱。

## 5. 复现要点
- 开源：代码 + 模型（gaoyuezhou/dino_wm），依赖 DINOv2、lucidrains/vit-pytorch；predictor 约 19M，encoder 冻结（ViT-S 约 22M）。
- 算力：训练用 GPU 论文未说明，推理用单张 A6000（A.8）。估计单卡即可：模型很小，DINOv2 特征可离线缓存，最大数据集 PushT 18500 条×100–300 步。8×H100 上训练 / 推理都毫无压力，可并行跑六个环境和多 seed。
- 坑：Rope / Granular 依赖 Nvidia Flex（安装困难）；PointMaze 来自 D4RL、PushT 来自 Diffusion Policy；frameskip 与历史长度 H 直接影响成功率（Table 6）；必须用 causal mask；decoder 损失不能回传；CEM 采样数 / 轮数与重规划频率是主要调参项；PushT 目标需在 25 步内可达。

## 6. 关键引用链
- 建立在：DINOv2（Oquab et al. 2024）的 patch 特征；IRIS（Micheli 2023）的 transformer WM（本文改为帧级预测）；PlaNet / DreamerV3、TD-MPC2 的 latent WM；Visual Foresight（Finn & Levine 2017；Ebert 2018）的图像目标 CEM-MPC；AVDC（Ko 2023）作为视频生成对照；I-JEPA / V-JEPA 的"在表征空间预测"思想。
- 后续：V-JEPA 2-AC（Assran et al. 2025）把同一配方换成视频编码器并扩展到真实 Franka，明确把本文列为最接近的工作；PLDM（Sobal et al. 2025）同期研究 reward-free 离线数据上的 latent dynamics 规划。
