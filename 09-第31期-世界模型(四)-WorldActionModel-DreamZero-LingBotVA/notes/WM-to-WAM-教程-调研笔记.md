# From World Models to World Action Models: A Concise Tutorial for Robotics

> 对应「可lip」第 31 期视频（世界模型系列第四期）。论文：[arXiv:2607.00836](https://arxiv.org/abs/2607.00836)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2607.00836（v1，2026-07-01） · 机构: SUSTech CLEAR Lab（据项目页域名 clearlab-sustech.github.io 推断，论文 HTML 版未列单位，未核实） · 代码/权重: 无（tutorial，不含模型） · 项目页: https://clearlab-sustech.github.io/WorldModelSurvey/
- 一句话: 把被各社区混用的"world model"一词钉死为 action-conditioned predictive model，按预测空间分成 observation-space / state-space 两大类；再把 world action model（WAM）定义为"把预测的未来和可执行动作耦合起来的 policy"并归纳出四种范式。本项目直接采用它作为 WM/WAM 的分类体系。

## 1. 要解决的问题

- "world model" 在不同社区指的东西差异很大：RL 里的 latent dynamics model（Dreamer、TD-MPC2）、CV 里的 video prediction / video generation（Cosmos、Genie 3）、机器人里的 physics-informed simulator（PhysTwin）、以及最近的 action-conditioned generative model，全都自称 world model。它们"建模什么、预测什么、预测拿来干什么"完全不同，导致文献无法对比、新人无法入门（Section 1）。
- 2025–2026 年又出现一批"既预测视频又输出动作"的模型（DreamZero、Cosmos Policy、LingBot-VA、UWM、Motus……），有的自称 WAM、有的自称 VLA、有的自称 unified model。tutorial 的目标是给一个最小但够用的定义和分类，让 WM 和 WAM 的边界清晰。
- 它不做实验、不做 benchmark，只做概念梳理——价值在"分类是否好用"，而不在数字。

## 2. 方法（它的框架）

**2.1 基本定义（Section 1）**
- *world* = robot + environment（environment 又分 objects of interest 和 ambient environment，Fig. 1）。
- *embodied AI task* = 设计一个 policy，把初始 world configuration 变成目标 configuration。policy 接收语言指令 $l$ 和当前观测 $o_t$，输出动作 $a_t$（Fig. 3）；policy 可以是 PID、MPC、VLA，也可以是 WAM——**注意：tutorial 把 WAM 定义为一种 policy，而 WM 本身不是 policy。**
- *world model*：给定动作，预测未来观测 $o_{t+1}$ 或状态 $x_{t+1}$，通常还条件在历史 $o_{0:t}$ 上（Fig. 4）。一步形式为
  $$y_{t+1}\sim p_\theta(\cdot\mid o_{0:t},a_t),$$
  $y=o$ 叫 observation-space WM，$y=x$ 叫 state-space WM；也可以预测一段轨迹 $y_{t+1:t+H}$。这里的"动作"很宽：可以是关节命令，也可以是 latent action、语言指令、甚至观察用的相机位姿。

**2.2 Observation-space WM 的两轴设计空间（Section 2.1，Fig. 6）**
- 纵轴：观测的空间显式度。RGB（数据最多、传感器最便宜：Cosmos、DreamDojo、Genie 3 等）→ multi-view RGB（Ctrl-World、EnerVerse、Vidar）→ RGB-D（FlowDreamer、RoboScape、TesserAct）→ point cloud（ParticleFormer、PointWorld）。越显式几何越好，但数据越少；常见折中是用单目深度估计把 RGB 视频标成伪 RGB-D。
- 横轴：动作的抽象层级，**这一轴决定了 WM 能拿来干什么**：
  1. low-level robot action（关节/末端命令）→ 能预测具体控制命令的物理后果 → 用作 learned simulator 做 MPC、policy evaluation、synthetic data generation（PointWorld、FlowDreamer、Ctrl-World、RoboScape、DreamDojo）。
  2. interface action（用户级控制、相机/视角控制）→ 把静态场景变成可交互的视觉环境，不预测机器人动力学（Genie 3、RTFM）。
  3. latent action（从无标注视频经重建目标 + 信息瓶颈学出来的、解释帧间变化的紧凑变量）→ 主要用于预训练，让 WM 能吃 action-free 视频（AdaWorld、DreamDojo）。
  4. language instruction → 最抽象、几乎不接地到底层控制，用于高层视觉规划——生成"任务应该怎么展开"的未来帧（UniPi、SuSIE、AVDC、TesserAct、EnerVerse、Vidar）。

**2.3 State-space WM 的四种状态表示（Section 2.2，Fig. 7）**
- latent state：先把观测编码成向量再预测。两种来路——在目标域自己学（重建式：DayDreamer、DreamerV3；预测式、不重建像素：V-JEPA 2、TD-MPC2、LeWorldModel、OSVI-WM），或直接用预训练视觉/VLM 特征（DINO-WM、LaDi-WM、WoMAP、object-centric WM）。通常配 robot action，作 RL 或 MPC 的紧凑模拟器。
- point track：预测关键点/任意点的 2D 或 3D 轨迹（ATM、"Flow as the cross-domain manipulation interface"、SKIL、3DFlowAction），去掉背景冗余、保留显式运动，常作为下游控制器的参考轨迹；代价是"选哪些点"这个结构先验可能限制可扩展性。
- neural-symbolic：状态是感知接地的谓词集合，动作是高层技能，transition 描述技能的前置条件和效果（类似 PDDL，但谓词由神经网络接地：VisualPredicator、ExoPredicator），面向长程规划。
- physical state：状态是物体位姿、速度、接触、质量、摩擦等物理量，transition 是物理引擎；流程通常是 3D 重建 → 用真实轨迹对齐仿真参数 → 用仿真做预测（PIN-WM、PhysTwin、EmbodieDreamer、ContactGaussian-WM）。

**2.4 World Action Model 与四范式（Section 3，Fig. 8）**
tutorial 只关注 language-conditioned、observation-space 这一支的 WAM，定义为联合分布
$$(o_{t+1:t+H},\,a_{t:t+H-1})\sim p_\psi(\cdot\mid o_t,l).$$
按"未来表示得多显式、和动作耦合得多紧"分四种：
1. **imagine-then-execute**（cascaded 两阶段）：先 $o_{t+1:t+H}\sim p_\theta(\cdot\mid o_t,l)$ 生成视觉子目标，再用单独的 inverse dynamics model 或 goal-conditioned policy $a_t\sim q_\phi(\cdot\mid o_t,o_{t+1})$ 落成动作。代表：UniPi、SuSIE、ATM、Dreamitate、AVDC、Vidar。优点：视频模型和 IDM 可以用不同数据训练，可以插入末端位姿/物体位姿/光流等中间量，可解释；缺点：视频生成错误会原样传给 IDM。
2. **video-feature-conditioned action prediction**：不解码完整视频，取视频模型 backbone 的中间时空特征 $f_t=\mathcal{H}(u_\theta(o_t,l))$，再 $a_t\sim q_\phi(\cdot\mid o_t,f_t)$。代表：VPP、"Video generators are robot policies"、DiT4DiT、Mimic-Video。省掉多步采样，但接口变成不可检视的 latent，无法判断策略是否真的用了未来信息。
3. **joint video-action modeling**：一个生成模型联合建模 $(o_{t+1:t+H},a_{t:t+H-1})$，通常从视频生成 backbone（Cosmos、Wan）出发改输出空间，再用带动作标签的机器人数据适配。代表：DreamZero、Cosmos Policy、GR-2、LingBot-VA。优点：视频和动作在共享表示里一起学，一致性好；缺点：需要带动作标签的机器人数据，且要同时优化高维视频生成和精确动作预测。
4. **auxiliary video prediction for policy learning**：视频预测只是训练时的辅助 loss，推理时可去掉视频分支只用动作头。tutorial 归入此类：Fast-WAM、UWM、UVA、UniVLA、Motus。省推理成本，但未来不作为显式 plan 参与执行，收益取决于辅助任务是否真的塑造了有用表示，两个 loss 的平衡也不容易。

tutorial 的总结：核心 trade-off 是"未来表示得多显式"×"和动作耦合得多紧"。


## 3. 实验

无实验（tutorial）。它的"数据"就是 Fig. 6/7/8 三张分类图和对应引用。按它的归类整理本项目关心的论文：

| 本项目论文 | tutorial 的归类 | 出处 |
|---|---|---|
| DreamerV3、DayDreamer | state-space / latent state（重建式，目标域自学） | Section 2.2 |
| V-JEPA 2、TD-MPC2、LeWorldModel | state-space / latent state（预测式，不重建像素） | Section 2.2 |
| DINO-WM、LaDi-WM | state-space / latent state（预训练视觉特征） | Section 2.2 |
| Ctrl-World | observation-space / multi-view RGB + low-level action；用途 policy evaluation、synthetic data | Section 2.1 |
| Cosmos（"World simulation with video foundation models for physical AI"） | observation-space / RGB + language | Section 2.1 |
| DreamDojo | observation-space / RGB + low-level 或 latent action | Section 2.1 |
| DreamZero、Cosmos Policy、LingBot-VA、GR-2 | WAM / joint video-action modeling | Section 3 |
| UWM、Motus、UVA、UniVLA、Fast-WAM | WAM / auxiliary video prediction | Section 3 |
| VPP、Mimic-Video、DiT4DiT | WAM / video-feature-conditioned | Section 3 |
| UniPi、SuSIE、AVDC、ATM、Dreamitate、Vidar | WAM / imagine-then-execute | Section 3 |

（WorldVLA、DreamGen 未被该 tutorial 引用。按其定义，WorldVLA 属 joint 建模（自回归离散 token，动作与图像 token 交错），DreamGen 属"WM 作为数据生成器"，落在 observation-space RGB + language 一格。）

## 4. 局限

- 作者自己的范围限定：WAM 部分只讨论 language-conditioned observation-space 一支；state-space 的 WAM（在 latent / 3D 空间里联合出动作）没有展开。
- 没有任何量化内容：不给 benchmark、不比性能、不讨论推理延迟和算力——而这恰恰是四范式在实际系统里取舍的关键（DreamZero 需要 2 块 GB200 才到 7Hz，VPP 一类方法就是为了省这个）。
- 归类有主观性：UWM / Motus 归 auxiliary 有争议（见第 2 节）；latent action 被放在 observation-space 的动作轴上，但在 Motus / UniVLA 里 latent action 同时也是策略输出，边界模糊。
- 没讨论 WM 作为 evaluator（Ctrl-World 一类）和作为 data generator（DreamGen）这两种越来越重要的用法，只在动作抽象层级里顺带提了一句。
- 没讨论幻觉 / 物理不一致如何度量，也没讨论 WM 的记忆长度、多视角一致性等系统问题。

## 5. 复现要点

- 无代码、无模型。项目页 https://clearlab-sustech.github.io/WorldModelSurvey/ 维护论文列表（内容量未核实）。
- 作为阅读地图使用：本笔记第 3 节的表可直接扩展成本项目的论文索引。
- 分类时的操作建议：先看论文的输入输出（预测什么、条件是什么动作），再看推理时视频分支是否参与（区分 joint 与 auxiliary），最后看有没有单独的 IDM（区分 cascaded 与 joint）。

## 6. 关键引用链

- 建立在：Dreamer 系列 / TD-MPC2（latent WM + RL / MPC）、UniPi（language-conditioned 视频规划 + IDM，即 imagine-then-execute 的原型）、V-JEPA 2 / DINO-WM（latent WM 做 zero-shot planning）、Genie 3 / Cosmos（interface / language 条件的视频世界模型）、DreamZero / Cosmos Policy / LingBot-VA / GR-2（joint 范式）、VPP / Mimic-Video（feature-conditioned）、UWM / UVA / UniVLA / Motus / Fast-WAM（auxiliary）。
- 同期综述：World Model for Robot Learning: A Comprehensive Survey（2605.00080，按功能分类）和 From World Action Models to Embodied Brains（2607.11689，按接口和系统分层）——三篇合起来是本项目的 WM/WAM 理论底座。
- 后续引用：2026-07 发布，暂无已知后续工作。
