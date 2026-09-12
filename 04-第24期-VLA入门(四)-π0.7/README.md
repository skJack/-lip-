# 第 24 期 · VLA 入门（四）

**视频标题**：π0.7 论文讲清楚：一个通用机器人模型不做微调，直接叠衣服、做咖啡、进没见过的厨房

π0.7 用一个通用模型零样本追平 RL specialist；方法是把 prompt 扩写成子任务文本 + 世界模型生成的子目标图 + 元数据；前置 MEM 讲历史帧怎么进模型。

## 这期讲了哪些论文

| 角色 | 论文 | 原文 | 我的笔记 |
|---|---|---|---|
| 本期主讲 | π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities | [arXiv:2604.15483](https://arxiv.org/abs/2604.15483) | [`notes/π0.7-调研笔记.md`](notes/π0.7-调研笔记.md) |
| 前置 / 背景 | MEM: Multi-Scale Embodied Memory for Vision Language Action Models | [arXiv:2603.03596](https://arxiv.org/abs/2603.03596) | [`notes/MEM-调研笔记.md`](notes/MEM-调研笔记.md) |
| 前置 / 背景 | Real-Time Execution of Action Chunking Flow Policies (RTC) | [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) | [`notes/RTC-调研笔记.md`](notes/RTC-调研笔记.md) |

MEM 是这期第 5–6 页的前置，π0.7 的历史编码器直接沿用它；RTC（实时动作块执行）在推理那一页只带了一句，笔记放这里备查。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键见 [`slides/使用说明.md`](slides/使用说明.md)，那里还有每一页对应讲什么的表）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 6 页官方演示视频。仓库里不放视频文件（230 MB），`deck.html` 直接从 Physical Intelligence 官网（website.pi-asset.com）加载，所以**看视频页需要联网**；其他页面离线可看。

