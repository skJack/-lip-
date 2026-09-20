# 第 29 期 · 世界模型（三）

**视频标题**：世界模型第3期 · Cosmos：视频生成模型怎么变成机器人策略｜从 UniPi、Genie 到 NVIDIA 的 Cosmos Policy 和 Cosmos 3

世界模型系列第三期，讲「拿视频生成模型当世界模型」这条线。UniPi（文本条件的视频生成当策略，逆动力学模型把视频翻成动作）和 Genie（3 万小时游戏视频无监督学出 8 个隐动作，逐帧可控）两篇讲动机、方法和它们留下的范式；Cosmos 1 讲世界基础模型的定义、2000 万小时视频的五步清洗管线和动作怎么在后训练时接进来，Predict 2.5 一页带过。重点是 2026 年的两篇：Cosmos Policy 不改架构，把动作块、未来观测、value 伪装成 latent 帧混进视频序列，一次后训练同时得到策略、世界模型和价值函数（LIBERO 98.5%，真机 ALOHA 超过 π0.5，先想象再打分的规划高 12.5 分）；Cosmos 3 把语言、图像、视频、音频、动作放进一个两塔模型，动作成了 mid-training 阶段就有的正式模态（DROID 策略在 RoboArena 真机榜第一）。全期挂在一张总图上：动作从模型外面 → 输入 → 输出 → 正式模态。

## 这期讲了哪些论文

| 角色 | 论文 | 原文 | 我的笔记 |
|---|---|---|---|
| 本期主讲 | Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning | [arXiv:2601.16163](https://arxiv.org/abs/2601.16163) | [`notes/Cosmos-Policy-调研笔记.md`](notes/Cosmos-Policy-调研笔记.md) |
| 本期主讲 | Cosmos 3: Omnimodal World Models for Physical AI | [arXiv:2606.02800](https://arxiv.org/abs/2606.02800) | [`notes/Cosmos-3-调研笔记.md`](notes/Cosmos-3-调研笔记.md) |
| 本期略讲（动机、方法与影响） | Learning Universal Policies via Text-Guided Video Generation（UniPi） | [arXiv:2302.00111](https://arxiv.org/abs/2302.00111) | [`notes/UniPi-调研笔记.md`](notes/UniPi-调研笔记.md) |
| 本期略讲（动机、方法与影响） | Genie: Generative Interactive Environments | [arXiv:2402.15391](https://arxiv.org/abs/2402.15391) | [`notes/Genie-调研笔记.md`](notes/Genie-调研笔记.md) |
| 本期略讲（动机、方法与影响） | Cosmos World Foundation Model Platform for Physical AI（Cosmos 1） | [arXiv:2501.03575](https://arxiv.org/abs/2501.03575) | [`notes/Cosmos-1-调研笔记.md`](notes/Cosmos-1-调研笔记.md) |
| 前置 / 背景 | World Simulation with Video Foundation Models for Physical AI（Cosmos-Predict 2.5 / Transfer 2.5） | [arXiv:2511.00062](https://arxiv.org/abs/2511.00062) | [`notes/Cosmos-Predict2.5-调研笔记.md`](notes/Cosmos-Predict2.5-调研笔记.md) |
| 补充 | 口播里每个数字的出处（哪些是论文原数、哪些是从曲线估读、哪些是我的判断） | — | [`notes/技术事实核查.md`](notes/技术事实核查.md) |

Cosmos Policy 和 Cosmos 3 是这期按动机 → 架构 → 方法 → 数据 → 实验展开的两篇；UniPi、Genie 讲动机、方法和影响，实验一口气带过；Cosmos 1 讲接口、数据管线和它自己说明没做实验的五种用途。Cosmos-Predict 2.5 只在第 17 页用一张表和 Cosmos 1 对比，笔记放这里备查。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键见 [`slides/使用说明.md`](slides/使用说明.md)，那里还有每一页对应讲什么的表）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 第 22 页是一页分步动画（Cosmos Policy 的规划模式 best-of-N：采 8 个动作块 → 想象后果 → 打分 → 执行最高分）：翻到那一页后按 `J` 走下一步、`K` 退一步，右侧六个步骤跟着亮。第 3 页和第 41 页是同一张自绘总图「动作在哪进出」，五篇论文各一行，蓝框是动作；第 15、20、31、33、35 页也是自绘图（Cosmos 1 的数据管线、动作块怎么变成 latent 帧、Cosmos 3 的序列模板、训练流程、推理流程）。

