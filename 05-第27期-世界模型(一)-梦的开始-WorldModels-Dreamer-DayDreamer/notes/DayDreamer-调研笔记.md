# DayDreamer: World Models for Physical Robot Learning（DayDreamer）

> 对应「可lip」第 27 期视频（世界模型系列第一期）。论文：[arXiv:2206.14176](https://arxiv.org/abs/2206.14176)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2206.14176（v1 2022-06-28；用 corl_2022 模板，PDF 元数据 Subject 为 "Proceedings of the 6th Conference on Robot Learning (CoRL 2022)"，据我所知为 CoRL 2022 正式论文，未核实）· 机构: UC Berkeley（Wu、Escontrela、Hafner 三人共同一作，Goldberg、Abbeel）· 代码/权重: 论文称公开全部 infrastructure 但正文没给 URL；据我所知为 https://github.com/danijar/daydreamer（基于 DreamerV2 的 TensorFlow 代码，许可证未核实）；没有预训练权重，每台机器人从零训练 · 项目页: https://danijar.com/daydreamer（视频）
- 一句话: 把 DreamerV2 的官方实现原封不动（不改算法、四个任务同一套超参）、只加一个异步 learner / actor 架构，放到 4 台真实机器人上做纯在线 RL——不用仿真、不用演示、不用 reset 策略：Unitree A1 四足从仰躺出发 1 小时学会翻身、站立、行走并在 10 分钟内适应推搡，UR5 / XArm 从像素 + 稀疏奖励 8–10 小时学会 pick-and-place 并接近人类操作速度，Sphero 只靠俯视图像 2 小时学会导航。它对世界模型发展史的贡献不是新算法，而是第一次把"latent WM + 想象中训 actor-critic"这条线以小时量级搬上真机并开源了整套机器人接口，成为 tutorial 里与 DreamerV3 并列的"目标域自学、重建式 latent WM"代表；Dreamer-v3 笔记已把它列为 Dreamer 系列的真机后续，本笔记不重复算法，只写真机部署本身。

## 1. 要解决的问题
- 真机 RL 需要的交互量太大，所以主流路线绕开真机：在仿真里 domain randomization 后 sim-to-real、用机器人 fleet 攒数据集、或依赖人类演示 / 任务先验（Related Work、Appendix C）。代价是建仿真任务和采演示都费时，仿真有精度误差，且部署后的行为不再随环境变化而适应。
- Dreamer（V1 / V2）已在仿真和游戏里证明数据高效，但"能否让物理机器人学得更快"是未知数（Abstract）；真机额外带来：动作延迟约束（A1 20 Hz 控制）、多模态传感器融合（本体 + RGB + depth）、没有 reset、硬件磨损、光照等环境漂移。
- 三个研究问题（Sec 3 开头）：不用仿真能否直接在真机上学？同一算法能否跨机器人平台 / 传感模态 / 动作空间？数据效率相对已有 RL 算法如何？

## 2. 方法
**预测空间：state-space（latent）；算法本体 = DreamerV2，本文没有提出新算法（Intro 贡献一原话 "without introducing new algorithms"）。**
- **版本核实：** Implementation 段原话 "build on the official implementation of DreamerV2"；Appendix D 超参表（论文唯一一张表，无编号）也是 V2 配方：RSSM 512、32 个 latent × 32 类的离散 latent、KL balancing 0.8、LayerNorm + ELU、MLP 4×512、replay 10^6（FIFO）、预热 10^4 步、batch 32 × 长度 32、想象 horizon H = 15（正文写 H = 16，表格写 15，原文不一致）、γ = 0.95、λ = 0.95、target critic 每 100 步更新、lr 1e-4、grad clip 100。没有 symlog / twohot / free bits / 回报归一化——这些是半年后 DreamerV3 才加的（见 Dreamer-v3 笔记）。tex 源码的超参表里有几行被注释掉的条目（KL scale AutoAdapt(3.5, 1e-2, 1)、entropy scale AutoAdapt(0.5, 1e-5, 1)、advantage EMA 0.99），暗示实际代码带有 V2 论文之外的自适应 KL / 熵系数，即介于 V2 和 V3 之间，但正式文本没写，未核实。
- **沿用 V2 的部分（不重复，见 Dreamer-v2 / Dreamer-v3 笔记）：** RSSM、想象中 λ-return actor-critic（式 3）、离散动作用 Reinforce、连续动作用 reparameterization 穿过可微 dynamics（式 4，含熵项 η）、actor / critic 梯度不回传到 WM（否则模型会过度乐观）。
- **式 (1) 的四个网络：** enc(s_t | s_{t-1}, a_{t-1}, x_t)、dec(s_t) ≈ x_t、dyn(s_t | s_{t-1}, a_{t-1})、rew(s_{t+1}) ≈ r_t。encoder 把所有模态（关节角、gripper、末端位姿、RGB、depth）融合成离散 code（Fig 3 把它叫 deep Kalman filter 结构）；decoder 重建每个模态，只作训练信号和人工检视（Fig B.1），学 behavior 时不解码。
- **本文真正新增的是工程：异步 learner / actor（Fig 2、Sec 2 开头、Implementation 段）。** learner 线程不停地在 replay 上训 WM + actor-critic，actor 线程并行地用最新 policy 算动作、和机器人交互，两者之间不做 rate limiting，所以 V2 的 train-frequency 超参消失（Sec 2 末）。两个理由：A1 20 Hz 下同步训练会拖慢动作输出（延迟要求）；机械臂 2 Hz / 0.5 Hz 太慢，GPU 不该等机器人。代价（我的读法）：replay ratio 由机器人速度决定、不受控，不同硬件上的"数据效率"不可直接比较——DreamerV3 后来把 replay ratio 做成显式超参。
- **动作怎么进出：** low-level 机器人动作直接作为 RSSM 输入；actor π(a_t | s_t) 在 latent 状态上直接输出动作，部署时没有 CEM / MPC 之类的 lookahead planning；想象 rollout 在 latent 里并行做，典型 batch 16K（单 GPU，Sec 2 Actor Critic 段）。
- **真机部署的四件事（本篇最值得记的）：**
  - reward 从哪来——每个任务人工写、由真实传感器 / 环境代码算出，WM 的 reward head 只是学着从 latent 预测它（Sec 2 "the robot has to discover task rewards by interacting"）；作者提到也可以把 reward 写成 decoder 输出的函数，但没做。
  - reset——只有 A1 是真正无 reset（从仰躺学起，摔倒是任务的一部分，Fig 4 圆点）；UR5 / XArm 靠 bin 的物理结构和硬编码规则（Z 轴只在持物时可动、到正确 bin 上方自动松爪、XArm 用绳子把物体系在 gripper 上防卡角）；Sphero 每 100 步用大功率随机动作打乱位置当自动 reset。
  - 安全——A1 用 Butterworth 低通滤波去掉高频电机指令、动作是目标电机角由硬件 PD 实现、3D 打印保护壳（Acknowledgements）、到场地边缘人工搬回但不改关节配置和朝向；机械臂的动作空间裁剪即安全约束；没有任何学习型安全机制。
  - 延迟——靠异步架构保证 actor 不被训练阻塞；具体毫秒数、图像分辨率、GPU 型号原文都没报。
- **训练数据与推理开销：** 全部在线自采，replay 从空开始；无仿真、无演示、无预训练视觉特征。actor 是 MLP，推理便宜，但没有报数。


## 3. 实验
Setting：4 台机器人、4 个任务、同一套超参；每个任务只有 1 个训练 run（Fig 4 注明 single run，阴影是时间 bin 内 1 std，其余图未说明），没有结果表，数字只在正文或曲线里——下列标"估读"的都是我从图上读的。Baseline 按输入 / 动作类型各选一个（Baselines 段）：SAC（低维输入 + 连续动作）、Rainbow DQN（像素 + 离散动作；本体感知按 MuZero 做法广播成额外通道拼到 RGB 上）、PPO（只在 UR5）、DrQ-v2（像素 + 连续动作）、人类（joystick 操作 UR5，3 人 × 20 min，作为上界）。

| 任务（章节 / 图） | 机器人与控制频率 | 观测 | 动作 | reward | 时长 → 结果 | baseline |
|---|---|---|---|---|---|---|
| A1 四足行走（Sec 3.1，Fig 4、8） | Unitree A1，12 直驱电机，20 Hz | 电机角、姿态、角速度（无视觉） | 连续：目标电机角 → 硬件 PD | 稠密，式 (5) 五项级联 | 1 h 翻身 + 站立 + 行走；再 10 min 抗推 | SAC |
| UR5 多物体 pick-and-place（Sec 3.2，Fig 5） | UR5，2 Hz | 关节角、gripper 位置、末端笛卡尔位置 + 第三人称 RGB | 离散：末端 X / Y / Z 增量 + 开合爪 | 稀疏：抓住 +1、同 bin 松开 −1、对面 bin +10 | 8 h → 2.5 obj/min | Rainbow、PPO、人类 |
| XArm pick-and-place（Sec 3.3，Fig 6、A.1） | XArm 7-DoF，约 0.5 Hz | 同 UR5 + RealSense depth | 同 UR5 | 同 UR5 | 10 h → 3.1 obj/min | Rainbow、人类 |
| Sphero 导航（Sec 3.4，Fig 7） | Sphero Ollie 双电机，2 Hz | 俯视 RGB（无本体感知） | 连续：电机扭矩 | 稠密：−L2 到固定目标 | 2 h → 平均距离 0.15 | DrQ-v2 |

- **A1（Fig 4、Fig 8）：** reward 式 (5)：upright（基座 up 向量与竖直方向的点积）、hip / shoulder / knee（关节角到站姿目标的 L1）、velocity（前向速度占比 × clip(v_x / 0.3)，即目标速度 0.3，单位未标），前一项 ≥ 0.7 后一项才激活；Fig 4 说最大 14（按式 (5) 逐项上限相加只有 13，upright 项写成 (ẑ·e_z − 1)/2 上限为 0，疑为符号排版误差）。结果：前 5 min 学会翻身落地，再 20 min 学会站立，约 1 h 学出 pronking 步态；Fig 4 估读：Dreamer 60 min 时均值约 11，SAC 约 20 min 后停在 7–8（只会翻身，且需人工解开腿的 dead-lock）。之后用长杆反复推倒，10 min 额外在线学习后能抗轻推、被重推后快速翻身站起（Fig 8）。
- **UR5（Fig 5）：** 3 个球；抓住由 gripper 部分闭合检测。8 h 达 2.5 obj/min，前 2 h 几乎没进展，先学会定位物体并在附近开合爪，后期能把球从角落推出来；Fig 5 估读：人类约 2.9 obj/min，Dreamer 约 4 h 后起飞、8 h 约 2.5；Rainbow ≈ 0（曲线到约 5 h 截止）、PPO 0.1–0.3 到 8 h，两者只学会抓起后立刻在同一 bin 放下的短视行为；作者归因为 model-free 需要的数据量真机采不起。
- **XArm（Fig 6，Appendix A / Fig A.1）：** 软物体，用绳子系在 gripper 上（bin 不倾斜、防卡角，代价是动力学更复杂）。10 h 达 3.1 obj/min，与人类相当；Fig 6 估读：人类约 3.05，Dreamer 6 h 前 ≈ 0、8.5 h 约 2.2、9 h 峰值约 3.1、之后在 1.5–3 之间大幅波动；Rainbow 全程 0。学会用绳子把物体先拉出角落再抓（多模态行为）。Fig A.1：夜间训练，日出强光使性能跌到 0，不改算法继续在线学约 5 h 后恢复并超过原水平（估读：日出约 9.7 h、恢复线约 15 h、之后峰值约 9 obj/min）——该图纵轴量级（日出前 4–5 obj/min）与 Fig 6 的 3.1 对不上，原文没有解释，估计是不同 run 或不同统计口径。
- **Sphero（Fig 7）：** 机器人对称、无本体感知，朝向只能从历史帧推断（RSSM 的 h_t 负责）。2 h 内平均距目标 0.15（以场地尺寸为单位、按时间步平均）；Fig 7 估读：Dreamer 与 DrQ-v2 都从约 0.6–0.8 降到 40–60 min 后的 0.2–0.35，130 min 时都约 0.2，两者相当（作者说与 DrQ-v2 论文的仿真结论一致）——即视觉连续控制上 WM 相对好的 model-free 没有优势。
- 一个 wall-clock 账（我的推测，未核实）：预热 10^4 步按控制频率折算，A1 约 8 min、UR5 约 1.4 h、XArm 约 5.6 h，恰与 Fig 5 / Fig 6 曲线起飞时间（约 2 h / 6 h）吻合——机械臂"前几小时没进展"很可能主要是随机预热期而非学习难度；Sphero（2 Hz）40 min 就下降则对不上。
- 消融：没有。没有多 seed、没有异步 vs 同步、replay ratio、模型尺寸、有无 depth 之类的对比。Fig B.1 是定性想象 rollout（每隔一帧解码），UR5 那组里一条轨迹中静止的橙球变成了绿球——WM 的物体恒常性错误，作者自己点出来了。

## 4. 局限
- 作者承认（Discussion）：长时间真机训练造成硬件磨损，需要人工干预或维修；没有把 Dreamer 和 baseline 训得更久去探上限；更难的任务可能要把真机学习与仿真结合。
- 我读出来的：(1) 每任务单 run、无 seed 方差、baseline 调参投入不明，"8 h 打败 Rainbow / PPO"是单次观察；(2) "无 reset、从零"只对 A1 成立，机械臂和 Sphero 的任务先验（bin 结构、Z 轴锁定、自动松爪、绳子、随机动作 reset）是成功的一部分；(3) reward 全部人工设计并依赖特权信息（A1 的姿态 / 速度、UR5 的 gripper 闭合 + bin 位置、Sphero 的外部定位），tex 源码注释里作者还写到膝盖会磕地、要调 knee 目标角——reward 设计是脆弱的隐性工作量；(4) 安全全靠滤波、PD、保护壳和人在旁边；(5) 每个任务从零学，WM 不跨任务、不跨机器人复用，无语言 / 目标条件；(6) 重建式 WM 对光照变化脆弱（Fig A.1 日出即崩），作者自己把"不重建的 latent WM"列为未来方向（Appendix C）；(7) 异步架构使 replay ratio 与硬件速度绑定，结果难以严格复现；(8) 没有任何泛化评测（单场景、Sphero 单一固定目标）。

## 5. 复现要点
- 开源：论文只说 release infrastructure；据我所知代码在 github.com/danijar/daydreamer（TensorFlow 2，含四台机器人的环境封装，许可证未核实），上游 DreamerV2 为 MIT；无预训练权重。
- 硬件：Unitree A1（需保护壳）、UR5 + 夹爪 + 第三人称相机、XArm 7-DoF + RealSense（RGB-D）、Sphero Ollie + 俯视相机；每个 run 一张 GPU（"single GPU"，型号未写）。
- 算力：模型很小（RSSM 512、MLP 4×512；图像分辨率原文未写），单卡即可；8 张 H100 一类的机器对算力毫无压力，瓶颈全在真机——每个 run 1–10 小时机器人时间、必须有人在旁边、硬件磨损。
- 坑：学习与采集必须异步，否则 20 Hz 跟不上；replay ratio 随机器速度变，跨硬件不可比；预热 10^4 步在慢机器人上就是几小时的随机动作；A1 的级联阈值 0.7、knee 目标角、速度目标 0.3 要按机器人调，式 (5) 的 upright 项符号以代码为准；Butterworth 滤波 + PD 是电机保护的前提；机械臂任务的硬编码规则（Z 锁定、自动松爪、绳子）去掉会难很多；光照必须恒定（XArm 只在夜间训）；H 以表（15）为准；单 run 曲线后半段方差大（Fig 6），至少跑 3 个 seed 才有结论。

## 6. 关键引用链
- 建立在（Dreamer 谱系）：DreamerV2（Dreamer-v2；离散 latent + KL balancing，本文直接用其官方实现）← DreamerV1（Dreamer-v1；想象中训 actor-critic）← PlaNet / RSSM（PlaNet）← World Models（World-Models）；latent 里的大批量并行想象类比 Isaac Gym。
- 建立在（真机 model-based 前人）：Visual Foresight（arXiv:1812.00568；像素级视频预测 + 规划，本文批评它只能短程、规划时要生成图像太贵）、SOLAR（Zhang 2019）、PDDM（Nagabandi 2019）、Yang 2019 / 2022 的四足 foot-placement 模型（需领域控制器）。作者点名"不重建的 latent WM"（DreamerPro、Dreaming、BLAST）为未来方向——后来 TD-MPC2、V-JEPA 2 走的正是这条路（arXiv:2310.16828、vjepa2）。
- 后续：DreamerV3（Dreamer-v3，2023-01）解决 V2 逐领域调参的问题并把 replay ratio 显式化，Dreamer-v3 笔记已把本文列为 Dreamer 系列的真机后续；tutorial（arXiv:2607.00836，Section 2.2）把本文与 DreamerV3 并列为"重建式、目标域自学的 latent WM"；DINO-WM / V-JEPA 2-AC（dino-wm、vjepa2）用预训练视觉特征 + 离线数据回答它"每任务从零学、光照一变就崩"的问题；综述 arXiv:2605.00080 对真机 WM 的综述可作延伸阅读（是否点名本文未核实）。
