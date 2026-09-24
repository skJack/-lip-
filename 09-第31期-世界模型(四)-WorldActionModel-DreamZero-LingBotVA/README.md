# 第 31 期 · 世界模型（四）

**视频标题**：世界模型第4期 · World Action Model：视频预测到底怎么帮到动作｜从 UniPi、GR-1、VPP、UWM 到 NVIDIA DreamZero 和蚂蚁 LingBot-VA

世界模型系列第四期，讲 2026 年被叫做 World Action Model（WAM）的这一类模型——一个模型同时吐未来画面和动作。开头先用三行把世界模型、VLA、WAM 的区别说清楚（真正的差别是「预测→选动作」这笔账算在训练时还是测试时）；然后 UniPi、GR-1、VPP、UWM 四篇铺垫各一页，它们对「视频预测怎么帮到动作」给了四种完全不同的答案。重点是 NVIDIA GEAR 的 **DreamZero**（14B 的 Wan2.1 联合去噪视频和动作，执行完把真实画面写回 KV cache 治好自回归的误差累积，真机零样本做训练里没有的动作，38× 工程优化压到 150 毫秒）和蚂蚁 Robbyant 的 **LingBot-VA**（视频一帧配四个动作的交错序列、双流 MoT、KV cache 存整条历史，朴素异步会让长程任务从 85.6 崩到 32.9）；最后是今年 7 月的 VA 2.0（语义 tokenizer + latent action + 稀疏 MoE + 多 chunk 预测 + 分层 planner）和反方 Fast-WAM（受控消融说推理时想不想象几乎不加分）。全期挂在一张总图上：动作从模型外面，一步一步走进了模型里面。

## 这期讲了哪些论文

| 角色 | 论文 | 原文 | 我的笔记 |
|---|---|---|---|
| 本期主讲 | World Action Models are Zero-shot Policies（DreamZero） | [arXiv:2602.15922](https://arxiv.org/abs/2602.15922) | [`notes/DreamZero-调研笔记.md`](notes/DreamZero-调研笔记.md) |
| 本期主讲 | Causal World Modeling for Robot Control（LingBot-VA） | [arXiv:2601.21998](https://arxiv.org/abs/2601.21998) | [`notes/LingBot-VA-调研笔记.md`](notes/LingBot-VA-调研笔记.md) |
| 本期铺垫（各一页讲想法和它留下的问题） | Learning Universal Policies via Text-Guided Video Generation（UniPi） | [arXiv:2302.00111](https://arxiv.org/abs/2302.00111) | [`notes/UniPi-调研笔记.md`](notes/UniPi-调研笔记.md) |
| 本期铺垫（各一页讲想法和它留下的问题） | Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation（GR-1） | [arXiv:2312.13139](https://arxiv.org/abs/2312.13139) | [`notes/GR-1-调研笔记.md`](notes/GR-1-调研笔记.md) |
| 本期铺垫（各一页讲想法和它留下的问题） | Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations（VPP） | [arXiv:2412.14803](https://arxiv.org/abs/2412.14803) | [`notes/VPP-调研笔记.md`](notes/VPP-调研笔记.md) |
| 本期铺垫（各一页讲想法和它留下的问题） | Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets（UWM） | [arXiv:2504.02792](https://arxiv.org/abs/2504.02792) | [`notes/UWM-调研笔记.md`](notes/UWM-调研笔记.md) |
| 本期重点之一（无笔记） | Native Video-Action Pretraining for Generalizable Robot Control（LingBot-VA 2.0） | [arXiv:2607.08639](https://arxiv.org/abs/2607.08639) | —（没有单独笔记） |
| 前置 / 背景 | Fast-WAM: Is Test-Time Imagination Necessary for World Action Models? | [arXiv:2603.16666](https://arxiv.org/abs/2603.16666) | [`notes/Fast-WAM-调研笔记.md`](notes/Fast-WAM-调研笔记.md) |
| 前置 / 背景 | From World Models to World Action Models: A Concise Tutorial for Robotics | [arXiv:2607.00836](https://arxiv.org/abs/2607.00836) | [`notes/WM-to-WAM-教程-调研笔记.md`](notes/WM-to-WAM-教程-调研笔记.md) |
| 补充 | 口播里每个数字的出处（哪些是论文原数、哪些是从曲线估读、哪些是我的判断） | — | [`notes/技术事实核查.md`](notes/技术事实核查.md) |

DreamZero 和 LingBot-VA 是这期按公式 → 架构 → 推理 → 加速 → 数据 → 实验展开的两篇；UniPi、GR-1、VPP、UWM 各一页，只讲它们的想法和留下的问题（UWM 给了两页，因为它是 DreamZero 的语法）。LingBot-VA 2.0 按架构 → tokenizer → 训练 → planner → 数据 → 实验也走了一遍，但它至今没放代码和权重，研究仓库里也没有单独的笔记。Fast-WAM 是这期的反方（第 49 页），WM→WAM 教程是全期分类口径的来源，两篇笔记放这里备查。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键见 [`slides/使用说明.md`](slides/使用说明.md)，那里还有每一页对应讲什么的表）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期没有动画页。第 6 页和第 53 页是同一张自绘总图「动作从模型外面走进模型里面」，五拨工作各一行，蓝框是动作，全期都挂在这张图上；另外第 17、20、21、22、26、35、46 页也是自绘图（DreamZero 的式 1、误差累积 vs 被现实校正、「把马克笔放进杯子」走一遍推理的上下两页、500 小时数据的采集机制、LingBot-VA 的双流 MoT、VA 2.0 的数据配方）。

