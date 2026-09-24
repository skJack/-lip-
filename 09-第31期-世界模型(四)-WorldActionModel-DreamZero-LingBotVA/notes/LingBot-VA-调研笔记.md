# Causal World Modeling for Robot Control (LingBot-VA)

> 对应「可lip」第 31 期视频（世界模型系列第四期）。论文：[arXiv:2601.21998](https://arxiv.org/abs/2601.21998)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2601.21998（v1 2026-01-29，v2 2026-03-22）· 机构: Robbyant / 蚂蚁集团（正文未印机构名，据论文模板 robbyant.cls 中的 Ant Group + Robbyant 标识及 GitHub org 判断）· 代码/权重: https://github.com/robbyant/lingbot-va（Apache-2.0，推理 + post-training 代码）+ https://huggingface.co/robbyant/lingbot-va（LingBot-VA-base / -posttrain-robotwin / -posttrain-libero-long 三个 checkpoint，另有 LeRobot 格式的 post-training 数据）· 项目页: https://technology.robbyant.com/lingbot-va
- 一句话: 把 Wan2.2-5B 视频生成模型改造成"逐 chunk 自回归的视频-动作世界模型"：每个自回归步先用 flow matching 预测未来 K 个视频 latent，再由一条同深度、更窄的动作流以 inverse dynamics 方式解出对应动作；用 KV cache 保存整条历史、每步把真机观测喂回，实现因果一致的闭环控制。RoboTwin 2.0（92.9/91.6）、LIBERO（98.5）和 6 个真机任务上超过 π0.5，且 50 条 demo 即可 post-train 到新平台。

## 1. 要解决的问题

- **VLA 的 representation entanglement**（Sec 1、3.1）：前馈式 a_t ~ π(·|o_t) 让一张网从"观测-动作"这一种监督里同时学场景理解、物理动态和运动控制，作者认为这导致样本效率低、泛化差，更像 pattern matching 而非理解 dynamics。
- **已有"WM 进策略"方案的三个短板**（Sec 1、3.2）：针对 UniSim 类交互式模拟器、UVA/UWM 类 chunk 式 video-action diffusion、Gen2Act/Act2Goal 类离线子目标生成——(i) *reactivity gap*：开环生成一长段、不吸收实时反馈；(ii) *长期记忆缺失*：逐 chunk 生成时历史没被持久缓存，长程漂移；(iii) *违反因果*：段内双向注意力让"未来"影响"过去"，与物理世界"现在只依赖过去"不符。
- **推理延迟**（Sec 1）：视频 token 远多于动作 token，且每个都要多步去噪，大视频模型直接部署达不到实时。

## 2. 方法

**总体表述（Sec 3.1–3.2）。** 控制被拆成两个概念阶段：视觉动态预测 o_{t+1} ~ p_θ(·|o_{≤t})（Eq 6）与 inverse dynamics a_t ~ g_ψ(·|o_t, o_{t+1})——inverse dynamics model（IDM）即"给定当前和下一时刻的观测，反推让状态这样变化的动作"。为做闭环写成 chunk 级自回归：
- 视觉动态 z_{t+1:t+K} ~ p_θ(·| z_{≤t}, a_{<t})（Eq 8），chunk 内各 token 用双向注意力并行生成，chunk 之间保持因果；
- 动作解码 a_{t:t+K-1} ~ g_ψ(·| ẑ_{t+1:t+K}, z_{≤t}, a_{<t})（Eq 9）：既看预测出的未来 latent，也看观测历史和动作历史（动作含绝对 EEF pose，动作历史即本体状态轨迹）。

**预测空间。** 视频 VAE latent 而非像素：Wan2.2 的 causal video VAE（4×16×16 时间×高×宽压缩）再 patchify 2×，多路相机图像沿宽度拼接后每帧共 N=192 个空间 token（Sec 4.2）。latent 可解码回 RGB，按本项目分类属 observation-space。

**Backbone 与 Mixture-of-Transformers（Sec 3.3、4.2）。** 双流 DiT：视频流由 Wan2.2-5B 初始化（d_v=3072，30 层）；动作流同样 30 层但 d_a=768（宽度 1/4），约 350M 参数，总计 5.3B。MoT 的意思是：两种模态各自有独立的 QKV 投影和 FFN 参数，但在同一层里做**联合自注意力**——动作 token 先线性投影到 d_v 参与联合注意力，再投影回 d_a 做残差。视频和动作互相看得见，又不共享参数、不互相干扰。动作向量经单层 MLP（隐层 256）变成 token，输出端由线性头解回 30 维动作（双臂各 7 EEF pose + 7 关节 + 1 夹爪，缺的维度补零，Sec 4.1）；指令用冻结的 T5 编码、cross-attention 注入。

**视频-动作交错（Sec 3.3）。** 视频按 τ=4 时间稀疏化，每个视频时刻对应 4 个动作，序列形如 [z_t, a_{t,1..4}, z_{t+1}, …]，所以预测 K 个视频 latent 就同时得到 4K 个动作。训练时 K 在 [1,4] 内随机采样（Sec 4.2；Sec 3.3 举例写的是 [1,8]），部署用 K=4。

**动作流初始化。** 从头随机初始化会让动作 token 的输出分布与视频 token 差太远、破坏联合注意力（Fig 7 梯度范数震荡）；作者把预训练视频权重按动作维度插值，再乘 α=√(d_v/d_a) 保方差。

**训练目标（Sec 3.3）。** 交错序列当作一条序列做 teacher forcing + 因果注意力 mask（Fig 3），一次前向同时优化所有时刻。损失 L = L_dyn + λ·L_inv（λ=1）：L_dyn（Eq 11）是视频 token 的 flow matching 速度场回归，条件是（可能加噪的）历史 z̃_{≤t}、a_{<t} 和指令 c；L_inv（Eq 12）是动作 token 的 flow matching，条件里多了 z̃_{t+1}。关键技巧 **Noisy History Augmentation**（Eq 10）：以 p=0.5 把历史 latent 沿 flow-matching 插值路径加噪到 s_aug ~ U[0.5,1]，让动作解码器学会从"半去噪"的视频 latent 里取动作信息；推理时视频只需去噪到 s≈0.5–0.6 而不是 1，去噪步数减半。Post-training 阶段再加一项 forward dynamics 损失 L_fdm（Eq 13）配合下面的异步推理。

**推理与闭环（Sec 3.4，Alg 1/2，Fig 4）。**
- 同步版（Alg 1）：视频 chunk 用 Euler 3 步积分到 s=0.6，动作 chunk 用 10 步积分到 s=1.0（Sec 4.2；Alg 1 写的是 0.5；video CFG 5.0，action CFG 1.0）；执行完 K 步动作后把**真实观测**编码进 KV cache，再预测下一 chunk。这就是 closed-loop rollout：模型的历史里放的是真机观测而不是自己的想象，teacher forcing 的训练/测试分布因此一致。
- 异步版（Alg 2）：Branch A 机器人执行当前动作 chunk，Branch B 同时预测下一 chunk。朴素做法是把上一轮对 ẑ_t 的预测直接留在 cache 里继续外推，但视频模型偏好时间平滑，会"接着自己的幻觉往下画"而忽略真实反馈 z_{t-1}，导致开环退化（Table 3：naive async 74.3，Horizon-3 任务只剩 32.9）。修正是 **FDM-grounded** 步：用最新真实观测 z_{t-1} + 正在执行的动作 a_t 做一次 forward dynamics 前向，"想象"出 z_t 写入 cache，再预测 z_{t+1}、a_{t+1}——每轮都重新对齐到真实世界。
- 控制频率/延迟：论文没有给出真机的 Hz 或每步毫秒数（tex 中亦无），只说异步版任务完成时间比同步快 2×（Sec 4.4）；RoboTwin 实验里动作 50 Hz、视频降到 12.5 Hz（Sec 4.3.2）。

**训练数据与算力（Sec 4.1–4.2）。** Agibot、RoboMind、InternData-A1（仿真）、OXE 的 OpenVLA 子集、UMI 系列人类演示（不含 DexUMI）、RoboCOIN，加自采数据，共约 16K 小时，每个数据源 90/10 切分；预训练 1.4T token，AdamW，峰值 lr 1e-4，wd 0.01，cosine，bf16，grad clip 2.0，text dropout 0.1 用于 CFG，序列打包到 10K token。GPU 数量与训练时长论文未说明。Post-training：50 条 demo 即可，lr 1e-5 训 3K 步（或 1e-4 训 1K 步，稍差但快）。

**用途。** 论文里它只作为语言条件的闭环 policy；Fig 1 提到模型还能从机器人视频做视觉动态预测和 inverse dynamics 推断，但未作为 planning / 评测 / 数据生成工具做实验。

**在本项目分类体系中的位置。** WAM，范式 3 **joint video-action modeling**（tutorial 的归类）。注意它的"joint"是架构层面的（一条交错序列、一个 MoT、一个训练目标）；每个自回归步内部是"先视频、后动作"的两段式，动作条件于半去噪的视频 latent——语义上等于把 imagine-then-execute 折叠进一个网络，而"动作头只看半去噪 latent"又与 video-feature-conditioned 一路相近。综述 2605.00080 把它和 Motus、BagelVLA、Fast-WAM 一起归为 MoE/MoT-style，与 tutorial 的归类不矛盾，只是切分粒度不同。

## 3. 实验

**真机（Sec 4.3.1，Fig 5，Appendix A Table S2–S7）。** 双臂平台（Fig 9/10 图中臂身印有 Franka Robotics 标识，型号论文未说明），6 个任务，每任务 50 条 demo、lr 1e-4 微调 500 步；每任务 20 trial，与 π0.5 交替评测；指标是 Progress Score（PS，分步计分，重试得 0.5）和 Success Rate（SR）。逐任务 Ours vs π0.5（PS/SR）：
- Make Breakfast（10 步）：97.0/75 vs 73.0/70（Table S2）
- Pick Screws（5 步）：82.5/70 vs 74.0/50（Table S3）
- Fold Clothes（6 步）：48.8/35 vs 62.9/30（Table S4）——PS 反而低于 π0.5
- Unpack Delivery（5 步）：84.5/65 vs 73.0/25（Table S5）
- Insert Tubes：85.8/40 vs 79.2/30（Table S6）
- Fold Pants（3 步）：76.7/70 vs 30.0/30（Table S7）

**仿真（Sec 4.3.2）。** RoboTwin 2.0 全部 50 个双臂任务（Table 1；训练集 2,500 干净场景 demo + 25,000 随机化 demo，50K 步）：Easy 92.93 / Hard 91.55，第二名 Motus 88.7/87.0，π0.5 82.7/76.8，π0 65.9/58.4，X-VLA 72.9/72.8（X-VLA 数字转引自 Motus 论文）；按任务步数分组，Horizon=3 时 93.22/93.28，比第二名高 +8.2/+9.1，是"长程更占优"的核心证据。逐任务表（Table S1）：Blocks Ranking Size 94/96 vs π0.5 49/26、Stack Blocks Three 99/98 vs X-VLA 6/10 差距很大，但 Hanging Mug 40/28、Turn Switch 44/45（Motus 84/78）仍然弱。LIBERO（Table 2；4 个 suite 各 50 demo，微调 4K 步，3 seed × 500 trial）：Spatial 98.5±0.3 / Object 99.6±0.3 / Goal 97.2±0.2 / Long 98.5±0.5，平均 98.5，高于 X-VLA 98.1、OpenVLA-OFT 97.1、π0 94.1（baseline 数字转引自 X-VLA 论文）。

**消融（Sec 4.4，Table 3，RoboTwin Easy）。** 同步基线 92.9；FDM-grounded 异步 90.4（Horizon-3 85.6）；朴素异步 74.3（Horizon-3 32.9）；用原始 Wan2.2-5B 而非 LingBot-VA 预训练权重做同样的 post-training 只有 80.6（Horizon-3 67.6）——说明 16K 小时的视频-动作联合预训练本身贡献很大。异步比同步任务完成快 2×。

**数据效率（Sec 4.5.1，Fig 8，PS，Ours vs π0.5）。** RoboTwin Easy：5 demo 46.6 vs 36.3，10 demo 58.2 vs 50.7，25 demo 74.2 vs 70.5，50 demo 84.6 vs 81.2；Make Breakfast：10 demo 61.1 vs 45.5，25 demo 81.7 vs 60.0，50 demo 97.0 vs 73.0。真机上差距随数据减少并不缩小；仿真上到 25–50 demo 时差距只剩 3–4 个点（正文写的"RoboTwin 10 demo 高 10.3"按图对应的是 5 demo）。**时序记忆（Sec 4.5.2，Fig 9）**：Wipe Plate（必须恰好擦 6 次）100% vs 47%，Search Box（右盒发现空后应去开左盒）100% vs 50%。**泛化（Sec 4.5.3，Fig 10）**：只用 Pick tissues 训练，测 Pick apples/blocks/bowls/pears；Pick tubes 在 ID 区域训练、OOD 区域测——只有定性图，无数字。

## 4. 局限

- 作者承认的（Sec 6）：视频压缩仍然是算力瓶颈；没有触觉/力/音频，接触密集任务受限。
- 实时性没有量化：5.3B 模型、每 chunk 3+10 步 Euler，论文只给"异步比同步快 2×"，没有 Hz、没有每步延迟；README 给的推理显存约 24GB。能否做高频反应式控制，得自己测。
- 幻觉/物理不一致真实存在：Sec 3.4 明确写视频模型"倾向于继续自己的幻觉视频而忽略真实反馈"，朴素异步崩到 74.3 就是证据；闭环性质靠 FDM-grounded 这个补丁维持。Sec 3.4 文字又说异步模式下"丢弃 t-1 之前的历史"，与 Alg 2 中 KV cache 持续累积的描述不完全一致，异步部署下"持久记忆"究竟保留多少，论文未说清。
- 真机证据偏弱：每任务 20 trial，方差大；Fold Clothes 的 PS 低于 π0.5（Table S4），与正文"六个任务两个指标都 SOTA"的表述矛盾。
- 对比不完全同协议：RoboTwin 的 X-VLA 和 LIBERO 的全部 baseline 都是转引数字；预训练算力未披露（1.4T token、5.3B），复现门槛高；语言理解只靠冻结 T5、没有 VLM，指令泛化能力未测。

## 5. 复现要点

- 开源情况：代码 Apache-2.0，含推理与 post-training（README 说训练脚本用 FSDP、8 卡）；HF/ModelScope 有 base 与两个 post-train checkpoint；LeRobot 格式的 post-training 数据集也放出。README 明确的坑：训练和推理必须手动切换模型配置里的 `attn_mode`。
- 模型规模：5.3B（视频流 Wan2.2-5B + 动作流 ~350M）。推理显存约 24GB（RoboTwin 评测）、约 18GB（开 offload 做图生视频-动作）（README）。
- 8×H100 能否训练/推理：推理和 post-training 可以。post-training 按论文 3K 步、lr 1e-5、50 demo，但序列长度到 1e5–1.5e5 token（Sec 4.3.1/4.3.2），显存压力主要来自长序列，需要 FSDP + 序列打包。预训练不现实：论文未给 GPU 数，按 6·N·D 估算 6×5.3e9×1.4e12 ≈ 4.5e22 FLOPs，8×H100 以 40% MFU 算约 5 个月（估计），且 16K 小时数据里含内部数据，拿不全。
- 依赖：Wan2.2-5B 权重与 VAE、T5 编码器；RoboTwin 2.0 / LIBERO 仿真环境；真机需要把自己的动作空间映射到它的 30 维统一表示。

## 6. 关键引用链

- 建立在：Wan2.2（视频 backbone 与 causal VAE）；flow matching / rectified flow（Lipman 等、Liu 等）；Mixture-of-Transformers（Liang 等 2025）；Motus（同样的 MoT + 视频稀疏化 + RoboTwin 多任务协议，论文多处引用 [5]）；Vidar（视频-动作 token 交错与 IDM 解码）；Seer、UniPi、VPP 等 IDM 式策略；UWM/UVA（chunk 式 video-action diffusion，作为对照）；Genie/Genie 2（自回归交互式世界模型）；异步执行思路与 RTC（Black 等 2025）、VLASH 同期。
- 后续/同期：综述 2605.00080 把它列为 MoE/MoT-style 代表并收录其 LIBERO 98.5、RoboTwin 92.9/91.6 数字（Table 5/6），同组还有 LingBot-VLA（RoboTwin 88.6/86.7）；Fast-WAM（2026）直接质疑"测试时是否需要视频想象"，与本文形成对照；DreamZero、Cosmos Policy 是同期的 joint 范式竞品。
