# 第 27 期 · 世界模型（一）

**视频标题**：世界模型第1期 · 梦的开始：从 World Models 到 DayDreamer，让策略在梦里练，机器狗真机 1 小时学会走路

世界模型系列第一期，讲「在模型的梦里训练策略」这条线的起点：2018 年的 World Models（V、M、C 三个模块，h 为什么记得住历史，策略怎么钻世界模型的漏洞）和 2022 年把 Dreamer 搬上四台真机的 DayDreamer（四足机器狗真机 1 小时学会走路）；中间 PlaNet 的 RSSM、Dreamer v1 的想象中 actor-critic 和训练循环、v2 的离散 latent、v3 的一套超参只讲改进的思路。

## 这期讲了哪些论文

| 角色 | 论文 | 原文 | 我的笔记 |
|---|---|---|---|
| 本期主讲 | World Models（Ha & Schmidhuber） | [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) | [`notes/World-Models-调研笔记.md`](notes/World-Models-调研笔记.md) |
| 本期主讲 | DayDreamer: World Models for Physical Robot Learning | [arXiv:2206.14176](https://arxiv.org/abs/2206.14176) | [`notes/DayDreamer-调研笔记.md`](notes/DayDreamer-调研笔记.md) |
| 本期略讲（改进思路） | Learning Latent Dynamics for Planning from Pixels（PlaNet，RSSM） | [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) | [`notes/PlaNet-调研笔记.md`](notes/PlaNet-调研笔记.md) |
| 本期略讲（改进思路） | Dream to Control: Learning Behaviors by Latent Imagination（Dreamer v1） | [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) | [`notes/Dreamer-v1-调研笔记.md`](notes/Dreamer-v1-调研笔记.md) |
| 本期略讲（改进思路） | Mastering Atari with Discrete World Models（Dreamer v2） | [arXiv:2010.02193](https://arxiv.org/abs/2010.02193) | [`notes/Dreamer-v2-调研笔记.md`](notes/Dreamer-v2-调研笔记.md) |
| 本期略讲（改进思路） | Mastering Diverse Domains through World Models（Dreamer v3） | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) | [`notes/Dreamer-v3-调研笔记.md`](notes/Dreamer-v3-调研笔记.md) |
| 补充 | 口播里每个数字的出处（哪些是论文原数、哪些是从曲线估读、哪些是我的判断） | — | [`notes/技术事实核查.md`](notes/技术事实核查.md) |

World Models 和 DayDreamer 是这期展开讲的两篇；PlaNet、Dreamer v1/v2/v3 只讲各自改了什么（RSSM、想象里训 actor-critic、离散 latent、一套超参），笔记都是完整的。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键见 [`slides/使用说明.md`](slides/使用说明.md)，那里还有每一页对应讲什么的表）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

