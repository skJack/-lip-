# Learning Universal Policies via Text-Guided Video Generation（UniPi）

> 对应「可lip」第 31 期视频（世界模型系列第四期）。论文：[arXiv:2302.00111](https://arxiv.org/abs/2302.00111)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2302.00111（v1 2023-02-01 前后，精确日期未核实；v3 2023-11-20；NeurIPS 2023）· 机构: MIT、Google DeepMind、UC Berkeley、Georgia Tech、University of Alberta（Yilun Du、Mengjiao "Sherry" Yang 共同一作）· 代码/权重: 官方未开源（项目页只有视频；LAPA 论文复现 UniPi 时改用 AVDC 的视频模型 + 自训 IDM，可作旁证；后来是否放出未核实）· 项目页: https://universal-policy.github.io
- 一句话: 把"决策"改写成"文本条件的视频生成"：给当前帧 + 语言指令，先用视频扩散模型生成一段完成任务的未来帧（video plan），再用一个小的 inverse dynamics model（IDM，从相邻两帧反推动作的模型）把帧序列翻译成机器人关节动作。它是 imagine-then-execute（先想象再执行 / cascaded 两阶段）范式的原型，第一次把互联网规模的文本-视频预训练拉进机器人策略，并主张"图像 = 统一状态接口、文本 = 统一任务接口"；之后 AVDC、SuSIE、VPP、DreamGen、DreamZero 都以它为起点或对照。

## 1. 要解决的问题
- 通用 agent 的障碍：不同环境 state / action 空间不同（MuJoCo 关节 vs Atari 像素 + 离散动作），知识难共享；Gato 式统一 tokenization 又用不上预训练视觉 / 语言模型的知识；跨环境很难写 reward。
- 作者归咎于 MDP 抽象本身（Sec 2.1）：没有统一 state 接口；必须有实值 reward；dynamics T(s'|s,a) 绑定具体本体和动作空间。
- 提出 Unified Predictive Decision Process（UPDP，Sec 2.2）：用图像做统一 state、文本做任务说明（替代 reward），并把"与环境无关的规划（视频生成）"和"与本体相关的控制（IDM）"拆开，让规划器可跨环境复用、迁移、调试。
- 代价：需要视频 + 文本数据而非 reward，但可以借 web-scale text-to-video 预训练。

## 2. 方法
**预测空间：像素级 RGB 视频**（Video U-Net 直接在像素上扩散，不经 VAE latent）。
- **UPDP 形式化（2.2）**：G = ⟨X, C, H, ρ⟩，ρ(·|x_0, c) 是条件视频生成器，输出 H 步图像序列；π(·|{x_h}_{h=0}^H, c) 是"轨迹条件策略"（实现上就是 IDM），从视频推出 H 个动作。两者从离线数据 D = {(x_i, a_i)_{i<H}, x_H, c} 分别估计；有限 H、episodic 任务。
- **视频规划器（3.1）**：基于 Imagen Video 的 video diffusion，T5-XXL（4.6B）编码文本，classifier-free guidance ŝ = (1+ω)s(·|c, x_0) − ω s(·)。三个关键设计：
  - 首帧条件：训练时显式把观测帧当条件，而不是像 Diffuser 那样采样时固定首帧（后者后续帧会漂离观测）。
  - tiling 保持轨迹一致性：把观测帧复制到每个时间位置、在通道维与噪声帧拼接，复用 temporal super-resolution 架构。
  - 时间层级：先生成稀疏关键帧（10×48×64，每 8 帧取 1），再用 temporal super-resolution 补成密集视频（20×48×64，每 4 帧取 1）；用 temporal conv 而非 temporal attention 混合时间（App A.1）。
- **测试时可控（3.1 Flexible Behavioral Modulation）**：采样时可乘一个先验 h(τ)（学到的分类器或某张中间图像的 Dirac delta）引导计划；Fig 7 用中间图像指定"先动哪块积木"。
- **动作怎么出来（3.2，App A.2）**：任务特定 IDM——4 层 3×3 conv（带残差）+ 全局 mean pool + MLP(128, 7)，从图像回归 7 维控制（6 关节 + 1 接触 / 夹爪，MSE），与规划器独立训练，可用更小甚至次优的数据；组合任务用 20k 条带动作标注视频训 IDM（App A.4），CLIPort 任务用 200k（App A.5）。
- **动作怎么进模型**：不进——视频模型只吃语言 + 首帧，低层动作只在 IDM 一端出现。
- **执行**：论文所有实验开环——生成 H 帧、IDM 解出 H 个动作后顺序执行；闭环 MPC（每步重新生成）只是提了可行、出于算力没做。
- **训练数据与算力（App A.1）**：仿真视频模型 1.7B（首帧条件）+ 1.7B（temporal SR），各 2M 步、batch 2048、256 TPU-v4。
- 真机：先在 Imagen Video 同款数据（14M 视频-文本 + 60M 图文 + LAION-400M）预训练，再在 Bridge 7.2k 段视频-文本（task ID 当文本，80/20 划分）微调，分辨率级联 16×40×24 (1.7B) → 32×40×24 (1.7B) → 32×80×48 (1.4B) → 32×320×192 (1.2B)。
- **推理开销**：作者承认生成高保真视频要约 1 分钟；progressive distillation 初步实验 16× 加速（Sec 6）。


## 3. 实验
Setting 三组：
- (1) 组合泛化：PDSketch 的积木任务（先把白块放进碗染色，再按语言关系摆到盘子里），语言指令 70% / 30% 划分 seen / novel，物体位姿随机，200k 脚本视频，动作是连续关节空间（不同于 PDSketch 的 pick-place 原语）。
- (2) 多环境迁移：CLIPort 10 个训练任务 → 3 个测试任务，200k 视频。(3) Bridge 真机数据上的视频生成质量。
- Baseline：Transformer BC（state / image 输入，代表 Gato / Multi-Game DT / RT-1）、Image + Trajectory Transformer、Diffuser（直接扩散关节动作而非视频）；都用 T5 语言嵌入；offline RL 不适用（无 reward）。
- Table 1（组合任务，完成率 %）：UniPi Seen Place 59.1 / Relation 53.2，Novel Place 60.1 / Relation 46.1；最好 baseline Seen 19.4（State BC）/ 12.8（Image+TT），Novel 13.2 / 9.6——差 3–5 倍，且 novel 与 seen 几乎持平。
- Table 3（CLIPort 多任务，新环境）：Place Bowl 51.6、Pack Object 75.5、Pack Pair 45.7 vs baseline 最高 14.8 / 21.7 / 10.5。
- Table 4（Bridge，24×40 分辨率，32 样本）：预训练 vs 无预训练 CLIP 24.54 vs 24.43、FID 14.54 vs 17.75、FVD 264.66 vs 288.02。
- 作者另训一个"末帧成功分类器"当代理指标：预训练 77.1% vs 从头 72.6%；并指出 CLIP score 反映不出"计划是否完成任务"，需要面向控制的生成指标。
- 最有信息量的消融 Table 2（seen place / relation）：三项全无 13.2 / 12.4 → 加首帧条件 52.4 / 34.7 → 加 tiling 一致性 53.2 / 39.4 → 加时间层级 59.1 / 53.2。首帧条件是最大单项；时间层级对多步的 relation 任务贡献最大（+13.8）。
- 定性：Fig 8 互联网预训练使模型能生成 Bridge 里没有的新指令视频，从头训练的模型会生成别的任务；Fig 9 对黑边裁剪、PS 贴入物体的背景改动鲁棒；Fig 4 生成的视频计划与实际执行大致对齐。
- 注意：**没有真机闭环执行结果**——真机部分只评视频质量。

## 4. 局限
- 作者承认（Sec 6）：视频扩散慢（分钟级）；环境基本全观测，部分可观测时视频模型可能"幻觉"出不存在的物体 / 运动；建议与 LLM 结合。
- 我读出来的（方法）：全部开环、无重规划；IDM 任务特定且要动作标注，"通用"只在视频那一半；仿真分辨率 48×64 极低，完成率 50–60% 说明视频→动作链路损失大；Bridge 的文本只是 task ID，语言泛化证据有限。
- 我读出来的（证据）：baseline 都很弱（10–20%），对比不够强；后来的对照实验显示 UniPi 在 Language Table 只有 13–22%、SIMPLER 1.3%（LAPA Table 1 / Table 11，用 AVDC 替代实现）、CALVIN ABC→D Avg. Len 0.92（VPP Table 1，取自 SuSIE 论文）——瓶颈是长程计划出错和 IDM 从少量标注学不准 7-DoF 动作；真机无闭环成功率；256 TPU-v4 × 2M 步不是学术界可复现的规模。

## 5. 复现要点
- 官方代码无。可用替代：AVDC（Ko et al. 2023，flow-diffusion 仓库）实现了同类"文本 → 视频 → 动作"管线并开源；LAPA 用它做 UniPi baseline（4 张 A100 可训）。
- 算力：原文 1.7B × 2、256 TPU-v4、2M 步无法照搬；现实替代是低分辨率 + 开源视频模型（SVD / Wan / Cosmos）微调。
- 数据：组合任务与 CLIPort 各 200k 脚本视频（PyBullet，需自己写脚本 agent）；Bridge 7.2k 段。
- 坑：必须显式首帧条件 + tiling（Table 2，否则掉到 13%）；IDM 要同域带动作标注视频；开环执行长程会累积误差，实用时要重规划（LAPA 复现时每执行 2 步重新生成）；别只看 FVD，要看任务完成。

## 6. 关键引用链
- 建立在：Imagen Video / Video Diffusion Models（Ho et al. 2022）的级联视频扩散；Diffuser（Janner 2022）的"扩散轨迹 + 测试时引导"；VPT（Baker 2022）用 IDM 给无标注视频打标；Visual Foresight（arXiv:1812.00568）的"像素预测 + 规划"传统；Gato / Multi-Game DT 作为被批评的统一 tokenization 路线。
- 后续（cascaded 路线）：AVDC（Ko 2023，用光流替代 IDM）、SuSIE（Black 2023，生成子目标图像 + goal-conditioned policy）；UniSim（arXiv:2310.06114，同一作者 Sherry Yang）把"文本 / 动作条件视频生成"扩成通用交互模拟器；DreamGen（arXiv:2505.12705）把"生成视频 + IDM 标动作"做成数据引擎。
- 后续（改接口）：VPP（VPP）把"解码视频"改为"取视频模型特征"；GR-1（GR-1）、UWM（UWM）、DreamZero（DreamZero）、Cosmos Policy（Cosmos-Policy）转向 joint 建模；LAPA（arXiv:2410.11758）拿它当"无动作标注预训练"的 baseline 并指出其瓶颈；Fast-WAM（Fast-WAM）质疑测试时想象是否必要。
- 分类：tutorial（WM-to-WAM-教程）与 survey（arXiv:2605.00080）都把 UniPi 当 imagine-then-execute 的起点。
