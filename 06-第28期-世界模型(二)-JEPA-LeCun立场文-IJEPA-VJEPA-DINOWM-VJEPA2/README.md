# 第 28 期 · 世界模型（二）

**视频标题**：世界模型第2期 · JEPA：不生成像素，在表征空间预测未来｜从 LeCun 立场文到 V-JEPA 2 真机零样本抓放

世界模型系列第二期，讲 LeCun 的 JEPA 这条线：2022 年立场文的思想（用一张图讲清生成式和 JEPA 的区别——预测下一帧的「摘要」，不预测下一帧本身），I-JEPA、V-JEPA、DINO-WM 各改了哪一处（用同一个形状的图逐篇对比），然后重点讲 V-JEPA 2 / V-JEPA 2-AC 的两阶段训练（第二阶段是一页键盘控制的分步动画）和真机结果（两个实验室的 Franka 零样本抓放，16 秒一步；像素世界模型 Cosmos 4 分钟一步、抓取 0%），最后说它真正留下的东西：一个被后来的 VLA 和世界模型工作当零件用的编码器。

## 这期讲了哪些论文

| 角色 | 论文 | 原文 | 我的笔记 |
|---|---|---|---|
| 本期主讲 | A Path Towards Autonomous Machine Intelligence（LeCun 2022 立场文） | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | [`notes/LeCun-2022-立场文-调研笔记.md`](notes/LeCun-2022-立场文-调研笔记.md) |
| 本期主讲 | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning（含 V-JEPA 2-AC） | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) | [`notes/V-JEPA-2-调研笔记.md`](notes/V-JEPA-2-调研笔记.md) |
| 本期略讲（改进思路） | Revisiting Feature Prediction for Learning Visual Representations from Video（V-JEPA） | [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) | [`notes/V-JEPA-调研笔记.md`](notes/V-JEPA-调研笔记.md) |
| 本期略讲（改进思路） | DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning | [arXiv:2411.04983](https://arxiv.org/abs/2411.04983) | [`notes/DINO-WM-调研笔记.md`](notes/DINO-WM-调研笔记.md) |
| 本期略讲（挖法与消融） | I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) | —（没有单独笔记） |
| 前置 / 背景 | VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model | [arXiv:2602.10098](https://arxiv.org/abs/2602.10098) | [`notes/VLA-JEPA-调研笔记.md`](notes/VLA-JEPA-调研笔记.md) |
| 前置 / 背景 | LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels | [arXiv:2603.19312](https://arxiv.org/abs/2603.19312) | [`notes/LeWorldModel-调研笔记.md`](notes/LeWorldModel-调研笔记.md) |
| 补充 | 口播里每个数字的出处（哪些是论文原数、哪些是从曲线估读、哪些是我的判断） | — | [`notes/技术事实核查.md`](notes/技术事实核查.md) |

LeCun 立场文和 V-JEPA 2 是这期展开讲的两篇；V-JEPA 和 DINO-WM 各讲它改了哪一处。I-JEPA（arXiv:2301.08243）只讲了挖法和两个消融，没有单独的笔记。VLA-JEPA 和 LeWorldModel 出现在最后「后来谁在用 JEPA」那一页，笔记放这里备查。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键见 [`slides/使用说明.md`](slides/使用说明.md)，那里还有每一页对应讲什么的表）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 第 16 页是一页分步动画（V-JEPA 2 第二阶段怎么训）：翻到那一页后按 `J` 走下一步、`K` 退一步，右侧六个步骤跟着亮。第 4、5、7、9 页是同一个形状的自绘图，方便对比生成式 / I-JEPA / V-JEPA / DINO-WM 各改了哪一处。

