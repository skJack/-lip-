# Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations（VPP）

> 对应「可lip」第 31 期视频（世界模型系列第四期）。论文：[arXiv:2412.14803](https://arxiv.org/abs/2412.14803)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2412.14803（v1 2024-12-19 前后，精确日期未核实；v2 2025-05-04；ICML 2025）· 机构: 清华大学 IIIS、上海 AI Lab、上海期智研究院、UC Berkeley、RobotEra（Yucheng Hu、Yanjiang Guo 共同一作，通讯 Jianyu Chen）· 代码/权重: https://github.com/roboterax/video-prediction-policy（论文只说"代码见补充材料"；仓库与权重是否放出、许可证：未核实）· 项目页: https://video-prediction-policy.github.io
- 一句话: 不解码视频、只做一次前向：把微调过的视频扩散模型（SVD 1.5B → 操作域的文本条件视频预测模型 TVP）当"预测式视觉编码器"，取其上采样层的中间特征（形状 T×H×W，显式含 1 帧现在 + 15 帧未来）经 Video Former 压成 token，喂给 DiT diffusion policy 出 action chunk；CALVIN ABC→D Avg. Len 3.06（GR-1）→ 4.33，真机灵巧手比最强 baseline 高约 30 个点，7–10 Hz。它把 imagine-then-execute 的"分钟级去噪"换成"140 ms 的一步特征"，是 video-feature-conditioned 范式的命名代表，也是 Fast-WAM"想象不必解码"的直接先声。

## 1. 要解决的问题
- 策略的视觉编码器（R3M、VC-1、Voltron 等）用单帧重建 / 双帧对比预训练，只编码静态信息，不显式表示未来（Fig 1）。
- 视频扩散模型（VDM）能预测未来、懂物理，但已有用法要完整去噪出未来帧再解动作：UniPi 慢到只能开环、SuSIE 只用一帧未来、像素里冗余细节多；GR-1 每次只预测一帧、不用视频基础模型、预测质量差（Sec 2）。
- 假设：VDM 内部表征同时含当前与预测未来，策略可以在这个表征里"隐式跟踪机械臂的运动"学出 inverse dynamics，只需少量演示把动作空间对齐到视觉空间。

## 2. 方法
两阶段（Fig 2）。**预测空间：视频扩散模型的 latent 特征**（不解码到像素；可视化时才解码）。
- **阶段 1：Manipulation TVP 模型（3.1）**：以开源 Stable Video Diffusion（1.5B）为底，加 CLIP 文本特征的 cross-attention 变成文本条件（原 SVD 只条件首帧），输出改成 16×256×256；首帧 s_0 与每个预测帧在通道维拼接作条件。
  - 损失是去噪重建 ‖V_θ(x_t, l_emb, s_0) − x_0‖²（式 3），按数据源加权 λ_H L_{D_H} + λ_R L_{D_R} + λ_C L_{D_C}（式 4：人类视频 / 机器人视频 / 自采）。训练完**冻结**。
- **阶段 2：动作学习（3.2）**：只跑一次前向——把 s_0 与最终噪声 latent x_{t'}（近似白噪声）拼接输入 TVP，取第 m 个上采样层特征 L_m ∈ R^{T×C_m×W_m×H_m}，各层插值到同一空间尺寸后沿通道拼接成预测式表征 F_p（多相机各自预测：F^static、F^wrist）。
  - **Video Former**：T×L 个可学习 query 对每帧做 spatial attention（跨视角）、再 temporal attention + FFN，压成固定 token（式 5）。
  - **动作头**：DiT 版 diffusion policy（沿用 MDT 的块），Q'' 经 cross-attention 注入，损失 ‖D_ψ(a_k, l_emb, Q'') − a_0‖²（式 6），10 步采样，action chunk 10 步（CALVIN 10×7，灵巧手 10×18，Table 13）。
- **动作怎么进出**：视频模型不吃动作（只吃语言 + 首帧）；动作只从策略头出来；IDM 是"隐式"的——策略学着从预测表征里读出机械臂将去哪。
- **推理时视频分支参与**：参与，但只一步前向、不去噪、不解码；每个观测过 TVP 一次 <160 ms，端到端约 140 ms（Table 4），RTX 4090 上 7–10 Hz，配合 action chunk。
- **训练数据（Table 8）**：TVP 微调共 375,192 条轨迹：SSv2 人类 191,642（采样比 0.30）、RT-1 87,212（0.15）、Bridge 23,377（0.15）、BC-Z 43,264、Taco-Play、Jaco-Play、CALVIN-ABC 18,033（0.10）、MetaWorld 2,500、自采 Panda 2,000、灵巧手 2,476。
- **算力（4.1）**：TVP 微调 8×A100 训 2–3 天；策略阶段 4×A100 6–12 h。
- **视频质量（Table 9）**：Bridge FVD 41.4 vs Seer 246.3——得益于 SVD 预训练。
- **动作层级**：low-level（Panda 7 维，Xhand 18 维），语言条件。


## 3. 实验
- Setting（4.1、4.3）：CALVIN ABC→D（只用 ABC 语言标注数据，同 GR-1 设定）；MetaWorld 50 任务（每任务 50 条 oracle 演示，单一语言条件策略）；真机 Franka Panda（2000 条、30+ 任务 6 类）与 XArm + 12-DoF Xhand 灵巧手（4000 条、100+ 任务 13 类，另 4 个工具使用任务：勺 / 锤 / 电钻 / 移液器）。
- Baseline：RT-1、Diffusion Policy、RoboFlamingo（直接动作）；UniPi、MDT、SuSIE、GR-1、Vidman（未来预测相关）；RoboUniview（3D）。
- Table 1 CALVIN ABC→D：VPP 0.965 / 0.909 / 0.866 / 0.820 / 0.769，Avg. Len **4.33**；RoboUniview 3.65、Vidman 3.42、GR-1 3.06、SuSIE 2.69、RoboFlamingo 2.47、MDT 1.55、UniPi 0.92。
- Table 1 10% 数据：VPP 3.25 vs GR-1 1.41——比所有 baseline 全量数据还高。（摘要的 +18.6% 是对 3.65，引言的 +41.5% 是对 GR-1 的 3.06。）
- Table 2 MetaWorld 50 任务平均成功率：VPP 0.682 vs GR-1 0.574、SuSIE 0.410、RT-1 0.346、DP 0.279；Hard 子集 0.526 vs 0.451。
- **Table 3（换编码器，其余不变）**：VDM 4.33；Stable-VAE 2.58；VC-1（在同样视频上 MAE 微调后）1.23；Voltron 1.54——关键是"预测式"表征而非单纯的生成式预训练。
- Table 4 消融：去互联网数据 3.97；去 CALVIN 视频（TVP 没见过下游域）3.31；去互联网数据且不用 SVD 预训练 1.63；去 Video Former 3.86 且延迟 140→450 ms；只用最后一层特征 3.60。
- 附录消融：Table 10 单层 3 / 6 / 9 / 12：3.72 / 3.88 / 4.29 / 4.05 vs 多层聚合 4.33；Table 11 扩散时间步 10 / 20 / 30：4.21 / 4.33 / 4.25；Table 12：2 步去噪 4.19 不比 1 步好且时间翻倍，去 temporal attention 4.18，单视角 3.58。
- Table 5 真机：Panda seen 0.85 / unseen 0.73 vs GR-1 0.52 / 0.38、SuSIE 0.56 / 0.46、DP 0.42 / 0.25（200+ rollout）；灵巧手 seen 0.75 / unseen 0.60 / 工具 0.68 vs SuSIE 0.45 / 0.28 / 0.23、GR-1 0.32 / 0.15 / 0.15（500+ rollout）。
- Table 7：灵巧手 cup-upright / stack / unplug 等类 baseline 全为 0，VPP 0.52–0.64。摘要说的 +31.6% 口径未核实（按 Table 5 与 SuSIE 的差约 0.30）。
- Fig 6：未见物体（网球、可乐、勺子）上 TVP 仍生成合理未来，真实执行轨迹与预测一致；作者把泛化归因于"视频模型对新物体也能预测 + 低层只需跟踪机械臂"。

## 4. 局限
- 作者承认：正文几乎没有局限一节；只在对比 Vidman 时指出不把下游视频喂进视频模型效果差，等于承认 VPP 也依赖下游域视频微调（Table 4：去 CALVIN 视频掉到 3.31）。
- 我读出来的（方法）：TVP 冻结、策略只在 SVD 的一步特征上学，语义 / 视觉泛化上限被 SVD 决定；视频模型不吃动作，因此不能做 what-if，"预测"只是任务条件的期望未来；action chunk 10 步开环执行，长程误差与延迟的权衡没讨论；每次观测都要过 1.5B U-Net，140 ms 对灵巧手已近上限。
- 我读出来的（证据）：一步前向表征是否真编码了未来只有可视化证据（Fig 4），没有量化（如探针预测未来位姿）；CALVIN baseline 数字来自不同论文 / 设定的混合（RT-1 取自 RoboFlamingo 论文、UniPi 取自 SuSIE 论文）；真机 baseline 只有三种。
- 成本：对新领域仍需微调 TVP（375k 条视频、8×A100 数天）。

## 5. 复现要点
- 代码：GitHub roboterax/video-prediction-policy（权重未核实）；依赖 SVD、MDT（DiT policy）、CLIP。
- 算力：TVP 微调 8×A100 2–3 天（375k 轨迹，每卡 batch 4）；策略 4×A100 6–12 h；推理单张 4090。8×H100 上一周内可完整复现 CALVIN。
- 数据：SSv2 + OXE 子集（RT-1、Bridge、BC-Z、Taco / Jaco-Play）+ CALVIN-ABC 视频 + MetaWorld；按 Table 8 采样比混合；验证集按 Seer 划分（Bridge 5,558 / SSv2 2,048 条留出）。
- 坑：一定要把下游域视频喂进 TVP 微调（否则 −1.0 Avg. Len）；取上采样层多层聚合而非末层；扩散时间步取 20 左右；Video Former 不可省（精度和延迟都受影响）；每个相机视角单独预测；灵巧手要加 PD 平滑（App A.2）。

## 6. 关键引用链
- 建立在：Stable Video Diffusion；Seer（文本条件视频预测前作与 FVD 基准）；"diffusion 特征可当表征"（Xiang 2023；Gupta 2024 image diffusion 用于具身）；MDT 的 DiT diffusion policy。
- 对照：UniPi（UniPi）、SuSIE 作为"完整去噪再解动作"的对照；GR-1（GR-1）作为主要 baseline。
- 后续（同范式）：Fast-WAM（Fast-WAM）把"不解码"推进到"连未来 token 都不实例化"并做受控对照；"Video generators are robot policies"、DiT4DiT、Mimic-Video（tutorial 同一范式）。
- 后续（其他范式）：LingBot-VA（LingBot-VA）、DreamZero（DreamZero）走回显式自回归想象但解决实时性；Motus（arXiv:2512.13030）/ UWM（UWM）用统一模型按需切换模式。
- 分类：tutorial（WM-to-WAM-教程）以 VPP 命名第 2 范式；survey（arXiv:2605.00080）归入"WM for policy"的特征耦合类。
