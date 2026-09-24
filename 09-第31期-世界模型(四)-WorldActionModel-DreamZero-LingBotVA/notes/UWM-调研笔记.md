# Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets (UWM)

> 对应「可lip」第 31 期视频（世界模型系列第四期）。论文：[arXiv:2504.02792](https://arxiv.org/abs/2504.02792)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2504.02792（v1 2025-04-03，v3 2025-05-23；RSS 2025，项目页另注明获 ICML 2025 "Building Physically Plausible World Models" Workshop Best Paper） · 机构: University of Washington（WEIRD Lab）+ Toyota Research Institute · 代码/权重: https://github.com/WEIRDLabUW/unified-world-model（代码 + DROID pretrained/cotrained 与 LIBERO-90 checkpoint，README 给 Google Drive 链接） · 项目页: https://weirdlabuw.github.io/uwm/
- 一句话: 把 action diffusion 和 video diffusion 放进同一个 DiT，用两个**独立的 diffusion timestep** 分别控制"动作"和"未来画面"各自的噪声程度（= 遮盖程度），于是同一套权重既是 policy，也是 forward dynamics、inverse dynamics 和 video predictor；还能靠"把动作 timestep 固定为 T"直接吃无动作标注的视频做 co-training。在 DROID 子集预训练 + 真机微调上明显强于 Diffusion Policy，是 Motus 等后续统一模型的原型。

## 1. 要解决的问题
- Imitation learning（behavior cloning，用 diffusion / flow 模型拟合 observation→action）在分布内可靠，但出分布就脆弱；扩展只能靠昂贵的遥操作示范。
- 示范轨迹和视频里本来就含有"世界如何随动作变化"的时间动力学信息，BC 只学 state→action 映射，把这部分监督扔掉了。World model 能学动力学，甚至能从 action-free 视频学，但此前没有清楚的路径把 WM 学到的动力学用于提升 BC 策略的鲁棒性与泛化。
- 前作卡点（Sec I、V）：GR-1 类 video-action transformer 用 L2 回归而非生成建模；PAD 类 joint video-action diffusion 对所有模态共用一个 timestep，只能从 (a, o') 的联合分布采样，无法灵活做条件/边缘推断，作者认为这让表示缺乏"动作→画面"的因果理解；两阶段方案（先预训练视频模型再接动作头）限制了视频与动作的特征共享。

## 2. 方法
**预测空间**：像素级 RGB，但在 latent 里做——224×224×3 的图经冻结的 SDXL VAE 压成 28×28×4 latent（Sec III-C）；多相机（DROID 3 路）、h_o = 2 帧堆叠。动作是长度 h_a = 16 的 action chunk（delta end-effector pose + 1 维夹爪，Appendix -C1）。

**核心设计：coupled video-action diffusion + 解耦 timestep（Sec III-B）**
- 数据元组 (o, a, o')：当前观测、动作块、执行动作块之后的未来观测。学一个联合噪声预测网络 s_θ(o, a_{t_a}, o'_{t_o'}, t_a, t_o')，同时输出动作噪声 ε_a 和未来图噪声 ε_o'。
- 关键洞察：diffusion 的 timestep 等价于"部分遮盖"——t = T 时输入是纯高斯噪声，该变量相当于被完全 mask（边缘化）；t = 0 时输入是干净数据，相当于把它当条件。把动作和图像的 timestep 拆开独立控制，就能从同一个网络里取出不同的条件/边缘分布。
- 训练目标（Eq. 1）：t_a, t_o' 各自独立从 U(0, T) 采样，
  ℓ(θ) = E[ w_a‖ε_a^θ − ε_a‖² + w_o'‖ε_o'^θ − ε_o'‖² ]，其中 a_{t_a} = √ᾱ_{t_a}·a + √(1−ᾱ_{t_a})·ε_a，o'_{t_o'} 同理；w_a = w_o' = 1.0（Table V）。训练时模型见到所有噪声等级组合。
- 推理四种模式（Eq. 2–5）：**policy** p(a|o)——t_o' = T，o'_T ~ N(0, I) 固定不动，只对动作做反向扩散（Eq. 2）；**video prediction** p(o'|o)——t_a = T 固定，只去噪图像（Eq. 3）；**forward dynamics** p(o'|o, a)——t_a = 0、喂真实动作，去噪图像（Eq. 4）；**inverse dynamics** p(a|o, o')——t_o' = 0、喂目标未来帧，去噪动作（Eq. 5）。
- **推理时只出动作**：图像 token 仍作为纯噪声输入 transformer（占算力，但不需要过 VAE 解码），动作用 DDIM 10 步采样（训练用 100 步，Table V）；预测 16 步、执行前 8 步再重规划（h'_a = 8，Appendix -A2）。

**Backbone（Sec III-C，Appendix -A1）**：DiT（AdaLN 条件化）。当前观测每帧每相机过 ResNet-18（ImageNet 初始化、可训练）得特征，与两个 sinusoidal timestep embedding 拼接后经 AdaLN 注入每个 block（AdaLN = 用条件向量生成每层 LayerNorm 的 scale/shift，是 DiT 的标准条件化方式）。序列 token = 动作 token（每步一个共享 MLP 编码）+ 未来图 latent 的时空 patch（patch (4,4,2)，3D conv）+ 8 个可学习 register token（"寄存器"：不对应任何输出、最后被丢弃的冗余 token；作者认为图像和动作是两种异质模态，需要一个中介来交换信息，而 DiT 的每个输出 token 都要预测噪声、没有空余位置，registers 补上这个空位）。规模：embed 768、12 层、12 头（Table V），论文未给总参数量。

**cascaded 还是 joint**：joint——同一个 transformer 联合建模，靠 timestep 做边缘化/条件化，不是"先生成视频再反推动作"的两阶段。

**action-free 视频怎么用（Sec III-D，Appendix -A2）**：视频样本把 t_a 固定为 T，用 N(0, 1) 随机噪声填充缺失的动作，损失仍是 Eq. 1（机器人样本和视频样本混在同一 batch、从混合数据集均匀采样），动作损失对 batch 内所有样本计算——本质是 diffusion 版本的"mask 掉动作"，不需要额外的 mask token。

**训练数据（Sec IV-B1，Appendix A）**：真机预训练用 DROID 子集 2000 条轨迹（按采集地点采样）；co-training 再加 2000 条去掉动作标注的 DROID 轨迹当视频；5 个真机任务各用 50/50/100/50/150 条示范微调（Table VI）。仿真用 LIBERO-90 共 4500 条轨迹预训练，LIBERO-10 中随机 5 个任务各 50 条示范微调。预训练 100K 步、微调 10K 步（Rice-Cooker 50K）。注意模型**没有语言输入**，多任务预训练只以图像为条件，任务靠专门微调区分。

**推理频率与延迟**：真机控制频率 10 Hz（Appendix -C1），每次预测 16 步执行 8 步；模型延迟论文未说明。

**怎么用于控制**：主要当 policy（微调后闭环执行）；inverse dynamics 模式用于轨迹跟踪（Table III）；forward dynamics 模式只做了可视化验证（Fig 8）。

**分类体系位置**：WM 部分是 observation-space（多视角 RGB 的 VAE latent），动作抽象层级为 low-level robot action；作为 WAM，tutorial 归入 4 auxiliary video prediction——基本同意：policy 模式下视频分支被 t_o' = T 边缘化、不解码视频。但它和"训练完就扔掉视频头"的方法不同：视频 token 推理时仍在 transformer 里跑，而且同一权重可切到 forward / inverse dynamics 模式，更准确的说法是"可灵活边缘化的 joint 模型，控制时取其边缘"。

## 3. 实验
**setting**：真机 Franka Panda + DROID 平台（2 场景相机 + 1 腕部相机），5 个任务，每任务 50 个固定初始化（Rice-Cooker 20 个），ID 与加入未见干扰物的 OOD 两种，每个初始化给三次机会（Appendix -C3）；仿真 LIBERO，3 seeds × 50 初始化，评测时把物体初始化范围扩大 0.03 并移除背景物体制造分布偏移（Appendix A）。**baseline**：Diffusion Policy（DiT 版）、PAD（作者重实现：joint diffusion + 通道拼接条件）、GR1（作者重实现：回归）。

**主要结果**
- Table I 真机 ID 成功率（pretrain / cotrain）：Stack-Bowls 0.86/0.92，Block-Cabinet 0.76/0.84，Paper-Towel 0.78/0.86，Hang-Towel 0.82/0.86，Rice-Cooker 0.60/0.65；DP 对应 0.48、0.60、0.52、0.64、0.35；GR1（pretrain）0.66、0.66、0.60、0.66、0.40；PAD 多个任务为 0。OOD：UWM 0.76/0.84、0.60/0.72、0.78/0.84、0.64/0.76，DP 0.36、0.26、0.48、0.28。作者称 ID 上最多领先最佳 baseline 20 个点（Sec IV-B2）。
- Table II LIBERO（分布偏移设置）平均成功率：UWM 0.79±0.11，DP 0.71±0.12，GR1 0.58±0.14，PAD 0.57±0.19。作者承认仿真里 OOD 增益比真机小（Sec IV-C）。
- Table III 轨迹跟踪（给参考未来帧、限时为轨迹长度）：inverse dynamics 模式 0.65 / 0.55（Book-Caddy / Soup-Cheese）高于 policy 模式 0.47 / 0.26；policy 给 1000 步也能到 1.00 / 0.97，说明 policy 会偏离参考轨迹再恢复。
- Table IV 分类 OOD（光照/背景/杂物各两档，每档 5 次）：Stack-Bowls UWM(Co) 21/30、UWM(Pre) 15/30、DP 12/30；Block-Cabinet 15/30、8/30、6/30——视频 co-training 的收益在更强分布偏移下最明显。
- Fig 10：从头训练时 UWM 与 DP 相当，预训练带来的提升 UWM 更大。

**最有信息量的消融**
- Table VIII：预测未来观测 0.86/0.76（Stack-Bowls/Block-Cabinet）> 重建当前观测 0.70/0.66 > 无重建的 DP 0.48/0.60——收益来自"学动力学"而不只是"学图像特征"。
- Table VII（LIBERO 单任务从头训）：8 个 register 0.88/0.90，无 register 0.81/0.85；AdaLN 换成 cross-attention 掉到 0.78/0.86。
- Table IX：用互联网人类视频（Kinetics-400 + Something-Something-v2，随机裁剪补齐 3 路相机）co-train 得 0.88/0.80，优于纯机器人数据 0.86/0.76，但不如同域机器人视频 0.92/0.84。

## 4. 局限
作者承认（Sec VII）：尚未从大规模人类视频学习（embodiment gap）；forward dynamics 生成常带伪影，用它做 planning 效果可能受限；期待更密集的视频预测。

我读出来的：
- 没有语言接口，策略只以图像为条件；"多任务预训练"实际靠每个任务单独微调，不能直接当 language-conditioned 执行器。
- 视频分支很稀疏：只预测 16 步动作之后的 2 帧、28×28 latent，物理一致性只在粗粒度上被约束；forward dynamics 没有定量评测（只有 Fig 8 可视化），能否做可信的 what-if 未知。
- 规模小（12 层 768 维 DiT + ResNet-18），预训练只用 DROID 的 2000 条轨迹；扩到全量 DROID / OXE 的 scaling 曲线没有给。
- 推理延迟未报告；PAD/GR1 都是作者重实现并改成 UWM 的输入输出格式，PAD 的极低分数可能部分来自实现选择（作者自己归因于通道拼接条件化）。

## 5. 复现要点
- 开源：代码 + 预训练 / co-train checkpoint（DROID、LIBERO-90）可下载；数据集需自备 DROID / Robomimic / LIBERO（README 提供 Zarr 封装）。
- 模型规模：DiT 12 层、768 维、12 头 + ResNet-18 + 冻结 SDXL VAE，总参数论文未给（按配置估计 1–2 亿量级，未核实）。
- 算力：DROID 子集 100K 步预训练在 4×A100 上 24 小时（Appendix -A3），batch 36×4；微调 batch 36×2。
- 8×H100：训练与推理都完全可行，一天内可复现预训练；真机部署需要 DROID 式平台（Franka + 3 相机）。
- 已知的坑：README 提示 DROID 的 tensorflow-dataset 可能需从源码安装；作者建议在高度多模态的数据集上增加 register 数量（Appendix -A2）；真机评测协议特殊（顶视相机叠图对齐初始化、每个初始化三次尝试），横向比较数字时注意。

## 6. 关键引用链
- 建立在：DDPM / Diffusion Policy（action diffusion）、DiT + AdaLN（Peebles & Xie）、latent diffusion + SDXL VAE、**UniDiffuser**（Bao et al. 2023，"用各模态独立 timestep 统一边缘/条件/联合分布"，作者明确说框架建立在其核心洞察上，Sec V）、Uni[MASK]（Carroll et al. 2022，决策中的 mask 统一推断）、**PAD**（Guo et al. 2024，最接近的 joint video-action diffusion 前作）、GR-1、VPP（视频特征喂动作头的两阶段路线）、register tokens（Darcet et al. 2024）。
- 后续：**Motus**（2512.13030）把 UWM 当作"统一世界模型的理论原型"，用 MoT + 预训练 VGM/VLM 替换其从头训练的小 DiT；tutorial 2607.00836 将 UWM 归入 auxiliary video prediction 一类。
