# Revisiting Feature Prediction for Learning Visual Representations from Video（V-JEPA）

> 对应「可lip」第 28 期视频（世界模型系列第二期）。论文：[arXiv:2404.08471](https://arxiv.org/abs/2404.08471)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2404.08471（v1 2024-02-15，编号按 2024-04 批次；PDF 版面日期 2024-04-15；arXiv 页面无 journal-ref，会议/期刊录用信息未核实）· 机构: FAIR at Meta；Inria；ENS / CNRS / PSL；Univ. Gustave Eiffel；NYU Courant / CDS（Yann LeCun 为作者，Assran 与 Ballas 共同末位作者）· 代码/权重: https://github.com/facebookresearch/jepa（CC BY-NC 4.0，不可商用；发布 ViT-L/16₂₂₄、ViT-H/16₂₂₄、ViT-H/16₃₈₄ 预训练权重和 K400 / SSv2 / IN1K / Places205 / iNat21 五个任务的 attentive probe 权重——据 repo README，非论文内容）· 项目页: Meta 博客 https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
- 一句话: 用 JEPA（joint-embedding predictive architecture：不重建输入，而是在表征空间里从一部分输入的表征去预测另一部分输入的表征）在约 200 万段公开视频上做视频自监督预训练——把视频切成 2×16×16 的 tubelet（时空小块）token，mask 掉约 90% 的时空块，让 predictor 只预测被 mask 区域的特征（目标来自 EMA teacher——编码器权重的滑动平均副本，作为稳定目标——并加 stop-gradient 即不向目标回传梯度；细节见 §2），不用像素重建、文本、负样本或预训练图像编码器；冻结 backbone + attentive probe（注意力探针，见 §2）下 ViT-H/16₃₈₄ 拿到 K400 81.9、SSv2 72.2、IN1K 77.9（摘要），在需要运动理解的 SSv2 上比 DINOv2 / OpenCLIP 等图像模型高 21 点以上（Table 6）。在世界模型发展史里，它证明了"纯特征预测、不碰像素"的视频自监督在 ViT-H（630M）/ 2M 视频的规模上可行，且比像素重建更省算力、更 label-efficient；它的 encoder + predictor + EMA 目标（式 1）几乎原样成为 V-JEPA 2 的预训练配方，V-JEPA 2-AC 再把 predictor 换成动作条件版本——它是 latent 世界模型这一支的表征基础，但本身没有动作、不是 WM（world model）。

## 1. 要解决的问题
- 核心问题（Sec 1）：用现代工具（ViT、masked modeling、query-based pooling、JEPA、更大数据）之后，"只做特征预测"能不能单独撑起视频无监督表征学习？此前实现 predictive feature principle（时间相邻的感知信号，其表征应互相可预测）的工作要么在冻结的预训练编码器上训 predictor，要么靠监督的 action forecasting 损失或对比负样本防塌，且多是小卷积网络（Sec 2）。
- 像素重建（MAE / VideoMAE / Hiera / OmniMAE）必须把容量花在不可预测的低层细节上；在表征空间预测可以丢掉无关或不可预测的像素细节（Sec 2）。I-JEPA / data2vec 已在图像、音频、文本上验证，但没人在视频上做到规模。
- representation collapse（表征塌缩：编码器对任何输入都输出常数，损失为零但表征无用）：朴素的特征回归 min‖P(E(x)) − E(y)‖ 有平凡解，需要不靠负样本、也不靠手工增广不变性的防塌方案（Sec 3.1）。
- 评测层面：DINOv2 / OpenCLIP 这类图像模型 K400 很强但 SSv2 只有 50.6 / 34.8（Table 6）——静态图像学不到运动概念；视频模型能否在运动任务上反超、在外观任务上追平？

## 2. 方法
**预测空间：latent（表征空间）。无动作条件、不出动作。** JEPA 三件套（Sec 3，Fig 2）：encoder E_θ、predictor P_φ、条件变量 z——这里 z ← Δ_y 只是目标块的时空位置，不是动作。
- **损失（式 1）：** min_{θ,φ} ‖P_φ(E_θ(x), Δ_y) − sg(Ē_θ(y))‖₁。sg = stop-gradient（不向目标一侧回传梯度）；Ē_θ 是 x-encoder 权重的 EMA（exponential moving average，指数滑动平均）副本，即 y-encoder / EMA teacher：它不被训练，只慢慢跟着学生走，用来产出稳定的预测目标。只在被 mask 的 token 上取平均 ℓ1（附录 B 式 2）。与 I-JEPA 的 ℓ2 不同，作者改用 ℓ1 回归，称更稳定。
- **为什么不塌（Sec 3.1）：** 沿用 BYOL 的 EMA + stop-gradient + predictor 组合。理论动机：ℓ1 下最优 predictor 是条件中位数 median(Y | E(x))，此时 encoder 的梯度等于 MAD(Y | E(x))（中位绝对偏差）的梯度——encoder 只有尽量保留视频信息才能压低目标的偏差；假设 EMA 让 predictor 比 encoder 更新得快、始终接近最优，从而不塌。这是经验 + 启发式论证，不是证明。
- **预测任务 = mask-denoising（在表征空间做"去 mask"，Sec 3.2）：** y 是若干（可重叠）空间连续块的并集，长宽比在 (0.75, 1.5) 随机，并沿整个时间维复制（tube 状，避免利用时空冗余作弊）；x 是补集。每个 clip 采两种 mask：short-range（8 个块、各占每帧 15%）和 long-range（2 个块、各占每帧 70%），平均 mask 比例约 90%——称 multi-block；两种 mask 共用一次 y-encoder 前向（multi-mask，附录 B）。
- **输入与 tokenizer（附录 B、Fig 7）：** 16 帧、帧间隔 4（附录 B 说约 2 s @30fps，Sec 3.4 说约 3 s，原文不一致）、224×224；3D 卷积做 tubelet（时空小块）2×16×16 → 8×14×14 = 1568 token；绝对 3D sin-cos 位置编码；无 [cls] token。mask 通过直接丢 token 实现：x-encoder 在输入端丢，y-encoder 在输出端丢（contextualized target：目标特征是看过完整视频后算出来的，来自 data2vec）。
- **encoder / predictor（Sec 3.3、附录 C）：** encoder 是标准 ViT——ViT-L/16、ViT-H/16（Table 6 标 200M / 630M）、ViT-H/16₃₈₄（384 分辨率）；predictor 是窄 ViT，12 层、宽 384、头数同 backbone，输入 x 的 token 加可学习 mask token（共享向量 + 3D 位置编码），输出每个 mask token 的 d 维预测。
- **数据（Sec 3.4）：** VideoMix2M = HowTo100M + Kinetics-400/600/700（K710）+ SSv2，去掉与各验证集重叠，约 200 万视频。
- **训练（Table 8、附录 C）：** 90K 迭代；batch 3072（L / H）、2400（H₃₈₄）；AdamW，lr 2e-4 → 6.25e-4（12K warmup）→ 1e-6 余弦；wd 0.04 → 0.4 线性增；EMA 动量 0.998 → 1.0 线性增；所有调度按 112.5K 步计算但在 90K 截断（作者说最后 25% 调度太激进，截断反而更好）；随机缩放 (0.3, 1.0)、水平翻转；bf16，A100 80G，GPU 数未说明。看过样本 270M（L / H）、210M（H₃₈₄）（Table 5 / 16）。
- **评测 = 冻结 backbone + attentive probe（Sec 4.3、附录 D、Table 9）：** attentive probe（注意力探针）= 一个可学习 query token 对 encoder 输出的全部 token 做 cross-attention（12 头）+ 残差 + 两层 MLP（GeLU）+ LayerNorm + 线性分类器，只训这一小块（20 epoch、lr 1e-3、wd 0.01）；对比 linear probe（线性探针：token 平均池化后直接接线性分类器）。K400 用 8 个 clip、SSv2 用 2 个 clip 的 token 拼接后进 probe，测试取 3 个空间视图；图像任务把图复制成 16 帧静止视频（作者说 2 帧即可）；图像模型评视频则逐帧编码后拼接。AVA 用预计算的 Faster-RCNN 框 + 冻结特征（末层 + 3 个中间层）线性分类，报 60 类 mAP。全量微调沿用 VideoMAE 协议（Table 11）。
- **看 predictor 学到了什么（Sec 6、Fig 6）：** 冻结 encoder + predictor，另训一个条件扩散 decoder，只喂被 mask 区域的预测特征、看不到上下文像素；仅用于解释，V-JEPA 本身不是生成模型。


## 3. 实验
Setting：VideoGLUE 子集——K400（外观主导的动作识别）、SSv2（运动主导的动作分类）、AVA（时空动作定位，mAP）+ 图像任务 IN1K / Places205 / iNat21；主协议是冻结 backbone + attentive probe，另报全量微调和 low-shot；消融用单中心视图，主表用多视图（K400 16×8×3、SSv2 16×2×3）。
Baseline：像素重建视频模型 VideoMAE / VideoMAEv2 / Hiera / OmniMAE 与蒸馏式的 MVD；图像模型 DINOv2 / OpenCLIP / I-JEPA。所有 baseline 都按同一 attentive probe + 多 clip 协议重新评测（附录 E.1），所以数字与各自原文不同。
- **特征 vs 像素（Table 1，同一 ViT-L/16、VideoMix2M、90K、multi-block）：** 目标换成归一化像素的 MSE 后 K400 73.7 → 68.6、SSv2 66.2 → 66.0、IN1K 74.8 → 73.3、K400 微调 85.6 → 85.4——冻结评测 K400 差 5 点、其余差不到 2 点，微调几乎持平。
- **数据规模（Table 2，算力固定 90K × 3072）：** ViT-L 三任务均值 K710（700K）70.9 → +SSv2（900K）71.0 → +HT（1.9M）71.1 → VideoMix2M（2M）71.5；ViT-H 72.0 → 72.8。但单任务最优数据各不同：SSv2 最好是 K710+SSv2（67.4），K400 最好是只用 K710（75.8）——数据分布比数据量更影响单任务。
- **模型规模（Table 6）：** ViT-L → ViT-H → ViT-H₃₈₄：SSv2 69.5 → 71.4 → 72.2，IN1K 74.8 → 75.9 → 77.4，iNat 67.8 → 67.9 → 72.6；K400 80.8 → 82.0 → 81.9（分辨率对 K400 无益）。
- **probe 的影响（Table 3 / 12 / 13 / 14）：** ViT-L 单视图平均池化 → attentive probe：K400 56.7 → 73.7（+17）、SSv2 50.1 → 66.2（+16.1）；Table 12：linear 56.7 / 50.1 vs attentive（多 clip）80.8 / 69.5，VideoMAE-L 同样受益（52.5 → 77.8、41.3 → 61.2）；Table 13：DINOv2 g/14 K400 linear 78.4 → attentive 83.4、SSv2 38.3 → 50.0；Table 14：K400 1 clip → 8 clip 73.7 → 80.9。作者解释：式 1 的目标未归一化，encoder 没理由产出线性可分的子空间。
- **mask 策略（Table 4，ViT-L 在 K710+SSv2，K400 / SSv2 / IN1K）：** random-tube[0.9] 51.5 / 46.4 / 55.6；causal multi-block[6] 61.3 / 49.8 / 66.9；causal multi-block[12] 71.9 / 63.6 / 72.2；multi-block 72.9 / 67.4 / 72.8。附录 E.4、Fig 8（26 组 ViT-B/16 on K400、linear probe，只有曲线无数值）：每 clip 采 2 个 mask 好于 1 个；同样 90% 比例下多个小块好于一个大块；空间 / 时间 mask 比例低会让任务变平凡，默认空间约 90%、时间 100%。
- **同架构 vs 像素重建（Table 5，ViT-L/16 或 Hiera-L，224）：** V-JEPA（270M 样本、90K 迭代）冻结 K400 80.8 / SSv2 69.5 / AVA 25.6 / IN1K 74.8 / Places 60.3 / iNat 67.8，全部高于 VideoMAE（410M 样本、400K 迭代：77.8 / 65.5 / 21.6 / 71.1 / 59.3 / 64.6）、Hiera-L（770M、1500K：75.5 / 64.2 / 15.8 / 68.9 / 58.5 / 56.9）和 OmniMAE（2400M、1170K），唯一例外 IN1K 74.8 vs OmniMAE 75.1（后者直接在 ImageNet 上训过）。
- **微调（Table 5 / 15）：** ViT-L 微调 K400 85.6 / SSv2 75.1——ViT-L 里最好，SSv2 与 Hiera-L 持平，K400 低于 Hiera-L 的 87.3；ViT-H 86.6 / 77.0（270M 样本）vs VideoMAEv2-H 86.9 / 76.8（1600M）、MVD-H 87.2 / 77.3（2400M，且用了 IN1K 和蒸馏教师）。Fig 4 / 5：达到同等 SSv2 性能看的样本少一个量级，墙钟约 2×（Sec 5.2；Fig 5 的墙钟是单卡 batch 10 测速后线性外推的，估读性质）。Table 16：看过样本 V-JEPA-H₃₈₄ 210M vs DINOv2 1900M、VideoMAEv2 1600M、OpenCLIP 39000M。
- **vs 最强图像 / 视频模型（Table 6）：** ViT-H/16 对 VideoMAE / VideoMAEv2 / OmniMAE / MVD / Hiera 的最大公开模型至少 +5 SSv2、+2 K400、+5 AVA、+1 IN1K、+2 Places、+0.2 iNat（Sec 5.2）；对 DINOv2 / OpenCLIP / I-JEPA 在 SSv2 上 +21 以上，AVA 也高（25.8 vs 24.3）；但图像任务仍明显落后：IN1K 77.4 vs DINOv2 86.2、iNat 72.6 vs 88.8、Places 62.8 vs OpenCLIP 70.2；K400 82.0 vs DINOv2 83.4。IN1K 换两层 probe 到 77.9（Sec 5.2）。
- **label efficiency（Table 7，probe 训练集取 5% / 10% / 50%，3 个随机切分）：** K400 ViT-H₃₈₄ 68.2 / 72.8 / 80.6 vs VideoMAE-H 62.3 / 68.5 / 78.2、MVD-L 62.6 / 68.3 / 77.2、VideoMAEv2-g 37.0 / 48.8 / 67.8；SSv2 54.0 / 59.3 / 67.9 vs 41.4 / 48.1 / 60.5、42.9 / 49.5 / 61.0、28.0 / 37.3 / 54.0。标注减 10×，V-JEPA 掉约 12（K400）/ 13.9（SSv2）点，VideoMAEv2 掉 30 / 26 点（Sec 5.3）。
- **predictor 可视化（Fig 6）：** 扩散解码的预测与未 mask 区域时空一致、运动连贯；不同随机种子给出不同位置 / 外形的物体（保留位置不确定性）；部分遮挡后物体仍在（object permanence）。定性，无量化指标。

## 4. 局限
- 作者承认：视频预训练数据集比图像模型的互联网级数据受限、缺乏视觉多样性，所以图像任务落后于 DINOv2 / OpenCLIP（Sec 5.2，作者以"假设"口吻说）；最优预训练数据分布因任务而异（Table 2）；V-JEPA 不是生成模型，Fig 6 只是定性解释（Sec 6）；防塌只有启发式论证（Sec 3.1）。
- 我读出来的：(1) 没有动作、没有语言、也没有沿时间向前的预测——mask 时间双向，causal 版更差（Table 4），predictor 学的是"表征空间的时空补全"而非动力学；(2) 只看 16 帧 × 间隔 4（约 2–3 s）、224 分辨率、固定 8×14×14 网格，长时程与精细几何未验证；(3) ℓ1 点预测 = 条件中位数，天然不能表达多模态未来；(4) 表征不线性可分（Table 3 / 12），任何下游使用都要再训 probe 或 head；(5) 评测只有分类 / 检测，没有深度、几何、机器人或规划任务；(6) 图像任务靠复制帧当视频评，与真正的图像模型不完全对等；(7) GPU 数与总算力未报告，Fig 5 的墙钟是外推值；(8) EMA 调度、25% 调度截断、ℓ1 等设计都是经验性的，说明 JEPA 训练稳定性仍脆弱（这正是后来 LeWorldModel 要解决的问题）；(9) clip 时长主文与附录说法不一致。

## 5. 复现要点
- 开源：facebookresearch/jepa（PyTorch），CC BY-NC 4.0（不可商用）；含三个预训练 encoder 和五个任务的 attentive probe 权重（据 repo README）。predictor 权重是否发布未核实。
- 规模：encoder ViT-L/16 约 200M（Table 6 所列）/ ViT-H/16 630M；predictor 12 层 × 384 宽，约 22M（按 ViT-S 量级估算，论文未给参数量）。
- 算力：A100 80G、bf16（Table 8），GPU 数与总时长论文未说明。我的估算：每次迭代 y-encoder 要对 3072 个 clip 的全部 1568 token 前向（ViT-H 约 2 TFLOP / clip），x-encoder 与 predictor 只处理约 10% 可见 token 加 mask token、两种 mask 各跑一次，合计约 4 TFLOP / clip，90K 步约 1e21 FLOP 量级——8×H100 在较好 MFU（model FLOPs utilization，实际算力利用率）下约一到两周可复现 ViT-H 预训练，ViT-L 约减半；直接用发布权重更现实。batch 3072 在 8 卡上要梯度累积。
- 数据：HowTo100M（约 120 万 YouTube 视频）、K710、SSv2 都要自行下载；YouTube 源视频持续失效，实际能拿到的 VideoMix2M 与论文不同（可得规模未核实）；帧间隔 4 的视频解码是 dataloader 瓶颈（我的推断，论文未说明）。
- 坑（训练）：(1) 用 ℓ1 不用 ℓ2（Sec 3.1）；(2) EMA 动量 0.998 → 1.0、wd 0.04 → 0.4 线性增、调度按 112.5K 计算在 90K 截断（附录 C）；(3) mask 必须贯穿整个时间维且空间比例约 90%，否则任务平凡（Fig 8c）；每 clip 采 short + long 两个 mask；(4) 目标在 y-encoder 输出端 mask（contextualized target），不是输入端；(5) 无 [cls] token。
- 坑（评测）：必须用 attentive probe + 多 clip，linear probe 会低估约 20 点（Table 12）；probe 超参见 Table 9（20 epoch、lr 1e-3、wd 0.01、K400 batch 256）；图像输入至少复制成 2 帧；AVA 评测依赖外部 Faster-RCNN 框（附录 D）；微调协议照 VideoMAE（Table 11）。

## 6. 关键引用链
- 建立在：JEPA 与 EBM 的立场文 `LeCun-2022-立场文`（LeCun 2022；式 1 的架构和"只预测可预测部分"的动机都来自它）；I-JEPA（Assran et al. CVPR 2023）的图像 JEPA + multi-block mask（本文改 ℓ1、扩到时空）；data2vec（Baevski et al. 2022）的 contextualized target 与 multi-mask 摊销；BYOL（Grill et al. 2020）与 Tian et al. 2021 的 EMA + predictor 防塌分析；MAE（He et al. 2021）/ VideoMAE（Tong et al. 2022）的 masked modeling 与 tube mask；CAE（Chen et al. 2022）的 query-based pooling → attentive probe；DINOv2 / OpenCLIP / VideoMAEv2 / Hiera / MVD 作 baseline；VideoGLUE（Yuan et al. 2023）评测集。
- 后续 `vjepa2`（Assran et al. 2025）：同一目标函数扩到 VideoMix22M / ViT-g 1B、加 V-JEPA 2-AC 动作条件后训练做真机规划。v1 → v2 差异汇总：数据 2M 视频 → 22M 样本（>100 万小时）；encoder 630M → 1B；predictor 12 层 × 384 → ViT-s 22M（同量级）；位置编码 绝对 3D sin-cos → 3D-RoPE；训练 90K 步、固定 16 帧 @224（H₃₈₄ 用 384）→ 252K 步、主阶段 16 帧 @256、cooldown 阶段升到 64 帧 @256–512（渐进分辨率）；损失写法相同（ℓ1 + sg + EMA）；v1 无任何动作 / 规划 / 语言实验，v2 在冻结 encoder 上加 300M 动作条件 predictor 并用 CEM 规划。
- 其他后续：`dino-wm`（Zhou et al. 2024）是平行路线——直接拿冻结 DINOv2 patch 特征当 WM 状态、在上面学动作条件 latent dynamics，其"在表征空间预测"的思想引自 I-JEPA / V-JEPA；`LeWorldModel`（2026）从像素端到端训 JEPA 世界模型，直面本文暴露的 JEPA 训练稳定性问题；分类定位见 tutorial（arXiv 2607.00836） Section 2.2（V-JEPA 2 归入 state-space latent 的预测式一支）与 综述 arXiv:2605.00080。
