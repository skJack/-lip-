# VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model

> 对应「可lip」第 28 期视频（世界模型系列第二期）。论文：[arXiv:2602.10098](https://arxiv.org/abs/2602.10098)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2602.10098（v1 2026-02-10；GitHub README 标 ECCV 2026，录用未核实；LaTeX 用 `galbot.cls` 模板）· 机构: 中国科学技术大学（Jingwen Sun、Hanxin Zhu、Guangzhong Sun、Zhibo Chen†）、中关村学院、上海交大（Wenyao Zhang）、清华（Zekun Qi）、宁波东方理工（Xin Jin†）、国科大、南开 · 代码/权重: https://github.com/ginwind/VLA-JEPA（代码已放；预训练 checkpoint 是否发布未核实）· 项目页: 无
- 一句话: 把 JEPA 的"在表征空间预测未来、未来只当目标不当输入"用到 VLA 预训练上：Qwen3-VL-2B 从当前帧 + 语言吐出几个可学习的"latent action" token，一个 12 层的 latent world model 拿它们去预测**冻结 V-JEPA 2 编码器**给出的未来帧特征（L1，teacher forcing），未来帧永远不进 VLM——从而在无动作标注的人类视频（SSv2 22 万条）和机器人数据（Droid 7.6 万条）上统一预训练；机器人数据再加一个 DiT-B 流匹配动作头，总损失 = 流匹配 + β × 世界模型损失。LIBERO 平均 97.2（π0.5 96.9、OpenVLA-OFT 97.1），LIBERO-Plus 平均 79.5（OpenVLA-OFT 69.6，+9.9），真机 Franka 分布内 0.70 vs π0 0.57 / π0.5 0.37。它是"JEPA 当 VLA 的辅助训练目标"这一用法（与 NVIDIA 的 FLARE 同类）里第一个把 V-JEPA 2 当目标编码器、且拿人类视频做统一预训练的工作。

## 1. 要解决的问题
- 从互联网视频学"latent action"再迁到 VLA（LAPA、UniVLA、Moto、villa-X）是热门路线，但作者指出这类目标学到的常常不是控制需要的东西。四个失败模式（§1）：
  1. **像素级目标偏向外观**：用未来像素或帧差压缩（VQ-VAE）当监督，信号被纹理、光照、背景、视角这些"变化大但不可控"的因素主导；
  2. **真实视频放大噪声运动**：人类 / 野外视频里相机运动、背景变化比交互引起的状态变化还强，帧差式 latent action 变成"噪声运动编码器"；
  3. **信息泄漏 → 捷径**：很多流水线把当前帧和未来帧一起喂给同一个模块，latent action 干脆把未来帧本身编进去，损失很低但对控制无意义；
  4. **多阶段流水线脆弱**：表征预训练 → latent action 学习/对齐 → 策略学习，三阶段以上，工程复杂、阶段间不一致。
- 归结为一个原则：**预测反映"动作相关状态转移"的未来 latent，同时不让未来信息泄进预测器**——这正是 JEPA 的形状（预测表征而非像素、未来只当目标）。

## 2. 方法
**整体（Fig 2，`figures/arch.png`）：** 两阶段，同一套网络。阶段 I 人类视频预训练只有 alignment 损失；阶段 II 机器人数据微调 = alignment + 动作预测。

**骨干与 token（§3.1、附录 A）：**
- VLM：Qwen3-VL-2B（SigLIP-2 视觉编码器），**全部参数可训**（只有 V-JEPA 2 世界状态编码器冻结）。
- 词表里加两类特殊 token：⟨latent_i⟩（第 i 步的 latent action，表示 s_i → s_{i+1} 的转移）和 ⟨action⟩（具身动作 token）。每个 ⟨latent_i⟩ 重复 K 次以增强注意（K = 24 / T，T = 未来视频步数，默认 T = 8 → K = 3）；⟨action⟩ 重复 32 次。
- VLM 输入：**只有初始时刻** t₀ 的多视角图 + 语言指令 + 这些特殊 token；输出对应位置的隐向量 z_{t_i}（式 2）。未来帧从不进 VLM。

**世界状态编码器（§3.2，式 1）：** 冻结 V-JEPA 2 编码器 F 逐帧编码（输入 256×256，每步 256 个 token、2048 维），多视角按向量拼接：s_t = ‖_v F(I_{v,t})。用两个视角；只有一个视角时复制一份拼接，多于两个时选两个。

**Latent world model（§3.2，式 3，附录 Table 7）：** 12 层、8 头、2048 维的自回归 transformer；每个时间步的序列 = [K 个 latent action token（来自 VLM）| 256 个世界状态 token（来自 V-JEPA 2）]；时间步内双向全注意、跨时间步严格因果。给 s_{t_0:i} 和 z_{t_0:i}，预测 ŝ_{t_1:i+1}。

**损失（式 4–5）：** 作者把它写成"语义空间预测似然的 ELBO"，因为 F 是确定性的、KL 项为零，落地就是 latent 回归 L_WM = Σ_k (ŝ_{t_k} − s_{t_k})（teacher forcing）。梯度回到世界模型和 VLM，把"世界怎么变"的信息压进 ⟨latent_i⟩。

**动作头（§3.3，附录 Table 8）：** 在 ⟨latent_i⟩ 之后追加 ⟨action⟩ token，VLM 的因果注意力让它看到图、语言和 latent action，输出 z_a（式 6）当条件；DiT-B（16 层、12 头、1024 维）做条件流匹配（式 7–8），动作 7 维（末端 Δ 位置 + Δ 轴角，min-max 归一化；夹爪二值化），状态 8 维，动作 horizon 7，去噪 4 步。机器人数据总损失 L = L_FM + β·L_WM（式 9，β 未给数值）。

**训练（附录 B）：** 8×A100，全局 batch 256；SSv2（22 万人类视频）+ Droid（7.6 万轨迹）联合预训练 50K 步；LIBERO / SimplerEnv 各接着 30K 步；真机 100 条演示再 20K 步。lr：VLM 与世界模型 1e-5，动作头 1e-4，余弦 + 线性 warmup。

**与 JEPA 家族的关系：** V-JEPA 2-AC 是"冻结编码器 + 预测器 + 测试时 CEM 出动作"，这里是"冻结编码器当**目标** + 预测器只在训练时存在 + 动作由流匹配头直接出"。JEPA 的三件套都在（学生 = VLM 通路、老师 = 冻结 V-JEPA 2、预测器 = latent WM），但老师不是 EMA 而是固定的外部模型，条件 z 从"位置"变成了"VLM 生成的 latent action"。推理时世界模型**不参与**，只跑 VLM + 动作头。


## 3. 实验
Setting：LIBERO（4 suite × 10 任务 × 50 episode）、LIBERO-Plus（7 种扰动）、SimplerEnv（Google Robot / WidowX，visual matching）、真机 Franka Research 3 + Robotiq 2F-85 + 3 个 D435（两个第三人称 + 一个腕部），100 条演示、3 个抓放任务，每任务 10 次。基线：LAPA、UniVLA、villa-X、Moto、CoT-VLA、WorldVLA、RoboVLMs、GR00T N1、OpenVLA-OFT、π0、π0-Fast、π0.5（真机上 π0 / π0.5 在同一批演示上微调）。
- **LIBERO（Table 1）**：VLA-JEPA Spatial 96.2 / Object **99.6** / Goal 97.2 / Long **95.8** / 平均 **97.2**；OpenVLA-OFT 97.6 / 98.4 / 97.9 / 94.5 / 97.1；π0.5 98.8 / 98.2 / 98.0 / 92.4 / 96.9；UniVLA 95.2；GR00T N1 93.9；villa-X 90.1；LAPA 65.7。去掉人类视频：96.1。
- **LIBERO-Plus（Table 3）**：平均 **79.5** vs OpenVLA-OFT 69.6、π0-Fast 61.6、π0 53.6、UniVLA 42.9、WorldVLA 25.0；7 个扰动里 5 个最好（Robot 67.1、Language 85.4、Light 95.6、Background 93.6、Layout 85.1），Camera 63.3 第二，Noise 66.3 输给 π0 79.0。去掉人类视频掉到 62.9——**人类视频的收益主要在鲁棒性**。
- **SimplerEnv（Table 2）**：Google Robot 平均 65.2（最高），WidowX 平均 57.3（与 LAPA* 并列最高）；但**去掉人类视频反而更高**（Google Robot 78.4），作者解释为分布内 / real-to-sim 场景更依赖高质量机器人演示。
- **真机（Fig 5，柱状图读数）**：分布内 VLA-JEPA 0.70 / π0 0.57 / π0.5 0.37；任务 OOD 0.17 / 0.00 / 0.20；布局 OOD 0.47 / 0.37 / 0.27。定性：π0.5 更会按指令找对物体但常撞安全边界；VLA-JEPA 会抓错物体但轨迹稳，且学会了**失败后张开夹爪重抓**（作者归因于人类视频里大量重抓行为）。
- **消融**：未来步数 T = 4 / 8 / 16 → LIBERO 平均 94.8 / 96.1 / 95.5（T 接近动作 horizon 最好）；latent action 对图像 token 的注意力图（Fig 7）：LAPA 散在整张图、UniVLA 盯背景语义、VLA-JEPA 集中在机械臂 / 手 / 被操作物体；人类视频比例越高 LIBERO-Plus 越稳（Fig 6）。

## 4. 局限
- 作者承认：语言跟随弱于 π0.5（会抓错物体）；人类视频没有教会新动作，只是让已有技能更稳；SimplerEnv 上人类视频甚至有害。
- 我读出来的：(1) LIBERO 97.2 vs 97.1 / 96.9 在 500 episode 尺度上没有统计差别，真正的差距在 LIBERO-Plus；(2) 真机每任务 10 次、100 条演示，π0 / π0.5 是作者自己微调的；(3) 没有消融"目标编码器换成 DINOv2 / SigLIP 会怎样"，所以"V-JEPA 2 作为目标"这个选择本身没有被单独验证（JEPA-VLA 那篇做了类似对照，见引用链）；(4) β、K = 24/T、32 次重复都是经验值；(5) VLM 全参数训练 + 8×A100 50K 步，不是轻量方法；(6) 世界模型推理时不用，不能拿来做规划或 what-if；(7) 与同期 JEPA-VLA（2602.11832，直接把 V-JEPA 2 特征注入 VLA）互不比较；(8) 只有单臂、抓放类任务。

## 5. 复现要点
- 开源：代码 github.com/ginwind/VLA-JEPA；依赖 Qwen3-VL-2B、V-JEPA 2 编码器权重（用哪一档未写）、SSv2（22 万视频）、Droid（7.6 万轨迹，TB 级）。预训练 checkpoint 是否放出未核实——若没放，完整复现要 8×A100 × 50K 步 × batch 256。
- 只做微调评测的话：拿预训练权重在 LIBERO 上 30K 步（batch 256），8×H100 可行；SimplerEnv 需 Fractal + BridgeV2。
- 坑：两视角要求（单视角复制拼接）；动作是末端 Δ位置 + Δ轴角，和 π0 系的关节空间动作不同，换 benchmark 要重新归一化；K = 24 / T 与 T 绑定；VLM 与世界模型 lr 1e-5、动作头 1e-4 两档；真机是 Franka Research 3 + 三个 D435。

## 6. 关键引用链
- 建立在：JEPA / I-JEPA / V-JEPA / V-JEPA 2（`LeCun-2022-立场文`、`V-JEPA`、`vjepa2`）——目标编码器与"未来只当目标"的原则；latent action 预训练 LAPA（`arXiv:2410.11758`）、UniVLA、Moto、villa-X——被批评的对象与主要基线；Qwen3-VL、SigLIP-2；DiT 与流匹配动作头（π0 一系）；Droid、SSv2 数据。
- 同期 / 对照：JEPA-VLA（2602.11832，未抓）——不训练、直接把 V-JEPA 2 最近两帧特征经 gated cross-attention 注入 OpenVLA-OFT，LIBERO 90.3 → 96.4，并做了 V-JEPA 2 vs DINOv2 / SigLIP 的对照；FLARE（2505.15659，未抓）——NVIDIA 在 DiT 策略里加"未来 token"对齐未来观测嵌入，同属"JEPA 当辅助目标"，但目标编码器是自训的 EMA 嵌入模型而非 V-JEPA；SRPO（2511.15605，未抓）——用 V-JEPA 2 latent 当进度奖励。
- 分类：tutorial（arXiv 2607.00836）（auxiliary 范式）、`survey-wam-next-frontier`（latent-only Joint WAM）、综述 arXiv:2605.00080（latent-space WM）。
