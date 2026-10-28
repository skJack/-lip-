# 第 39 期 · 具身智能基础（四）

**视频标题**：VLN 视觉语言导航：听一句话，在没来过的房子里自己走到地方

VLN（Vision-Language Navigation，视觉语言导航）：机器人用自己的相机看、听懂一句话、自己走到地方。前面几期讲的 VLA 是机器人的手，VLN 是腿。这期一期讲完：先讲清楚任务——每走一步问一次模型、从一句话到电机中间隔了哪几层、和 VLA 最大的不同是目标不在眼前；再讲第一个考场 **R2R**（Matterport3D 的 90 栋房子、导航图、众包写的 2 万多条指令、按楼划分、SR / SPL 怎么打分）；然后按三条路线走：**专门训导航模型**（Speaker-Follower 自己造数据，五年从两成爬到八成）、**换考场 VLN-CE**（去掉导航图，一步 25 厘米）、**大模型当大脑**（NavGPT）、**导航 VLA**（NaVid 从架构、token 压缩、数据、训练、推理到真机详讲 10 页，之后 Uni-NaVid、NaVILA、StreamVLN、DualVLN）；2026 年**大厂下场**（阿里千问 Qwen-RobotNav、高德 ABot-N0 / N1、Mistral Robostral Navigate），强化学习和世界模型两个新方向，一张 R2R-CE 两年半从 37% 到 77% 的爬坡图；最后是 **NavGPT-3**：大模型当调度员、导航 VLA 当司机，RxR-CE 90.43% 第一次追平人类（但人类分数是在离散环境测的，真机上还差得远）。每个模型的预训练底座（看图用什么、语言用什么）在核查文件里有一张对照表。

## 这期的材料从哪来

这期讲的是一个方向，不是一篇论文：从 2018 年的 R2R 讲到 2026 年 10 月的 NavGPT-3，每条路线先详讲开山的那篇，再带过后来的。下表是 deck 里用到的论文（arXiv 链接）和演示视频的出处，每页数字的出处在 `notes/技术事实核查.md`。仓库里不放视频文件，视频页换成了片段的一帧，想看动的点下面的链接。

| 类型 | 材料 | 链接 |
|---|---|---|
| 第一个考场 | R2R：Vision-and-Language Navigation（CVPR 2018） | [arXiv:1711.07280](https://arxiv.org/abs/1711.07280) |
| 第一个考场 | Matterport3D：R2R 用的 90 栋真实房子 | [arXiv:1709.06158](https://arxiv.org/abs/1709.06158) |
| 打分 | On Evaluation of Embodied Navigation Agents（SPL；点导航、物体导航的定义） | [arXiv:1807.06757](https://arxiv.org/abs/1807.06757) |
| 后来的考场 | RxR（多语言、路线更长） | [arXiv:2010.07954](https://arxiv.org/abs/2010.07954) |
| 后来的考场 | REVERIE（只说找什么，到了还要认出物体） | [arXiv:1904.10151](https://arxiv.org/abs/1904.10151) |
| 后来的考场 | CVDN（走一半可以问路） | [arXiv:1907.04957](https://arxiv.org/abs/1907.04957) |
| 后来的考场 | Touchdown（纽约街景） | [arXiv:1811.12354](https://arxiv.org/abs/1811.12354) |
| 后来的考场 | AerialVLN（无人机） | [arXiv:2308.06735](https://arxiv.org/abs/2308.06735) |
| 路线一 · 开山 | Speaker-Follower（反过来训一个模型，自己造数据） | [arXiv:1806.02724](https://arxiv.org/abs/1806.02724) |
| 路线一 | PREVALENT（第一个 VLN 预训练） | [arXiv:2002.10638](https://arxiv.org/abs/2002.10638) |
| 路线一 | HAMT（把整段历史完整编码） | [arXiv:2110.13309](https://arxiv.org/abs/2110.13309) |
| 路线一 | DUET（拓扑图，可以跳回之前的节点） | [arXiv:2202.11742](https://arxiv.org/abs/2202.11742) |
| 路线一 | ScaleVLN（490 万条数据，R2R 测试 80%） | [arXiv:2307.15644](https://arxiv.org/abs/2307.15644) |
| 换考场 | VLN-CE（去掉导航图，在连续环境里一步步走） | [arXiv:2004.02857](https://arxiv.org/abs/2004.02857) |
| 换考场 | ETPNav（路点预测 + 边走边建图） | [arXiv:2304.03047](https://arxiv.org/abs/2304.03047) |
| 路线二 · 开山 | NavGPT（GPT-4 零样本，画面先写成文字） | [arXiv:2305.16986](https://arxiv.org/abs/2305.16986) |
| 路线二 | MapGPT | [arXiv:2401.07314](https://arxiv.org/abs/2401.07314) |
| 路线二 | NavGPT-2 | [arXiv:2407.12366](https://arxiv.org/abs/2407.12366) |
| 路线三 · 详讲 | NaVid（第一个让 VLM 看视频来导航） | [arXiv:2402.15852](https://arxiv.org/abs/2402.15852) |
| 路线三 | Uni-NaVid（一个模型做四件事） | [arXiv:2412.06224](https://arxiv.org/abs/2412.06224) |
| 路线三 | NaVILA（大模型出语言动作 + 腿的运动策略） | [arXiv:2412.04453](https://arxiv.org/abs/2412.04453) |
| 路线三 | StreamVLN（近的全记，远的压缩） | [arXiv:2507.05240](https://arxiv.org/abs/2507.05240) |
| 路线三 | DualVLN / InternVLA-N1（大模型点目标，小模型画轨迹） | [arXiv:2512.08186](https://arxiv.org/abs/2512.08186) |
| 大厂 | 阿里千问 Qwen-RobotNav | [arXiv:2606.18112](https://arxiv.org/abs/2606.18112) |
| 大厂 | 高德 ABot-N0 | [arXiv:2602.11598](https://arxiv.org/abs/2602.11598) |
| 大厂 | 高德 ABot-N1 | [arXiv:2607.10383](https://arxiv.org/abs/2607.10383) |
| 大厂 | Mistral Robostral Navigate | [arXiv:2607.20785](https://arxiv.org/abs/2607.20785) |
| 新方向 | VLN-R1（强化学习） | [arXiv:2506.17221](https://arxiv.org/abs/2506.17221) |
| 新方向 | AstraNav-World（世界模型：先想象，再走） | [arXiv:2512.21714](https://arxiv.org/abs/2512.21714) |
| 合流 | NavGPT-3（大模型当调度员，导航 VLA 当司机） | [arXiv:2610.10787](https://arxiv.org/abs/2610.10787) |
| 综述 | 同济 2026 VLN 综述（第 11 页论文数量、第 55 页真机实测） | [arXiv:2607.09792](https://arxiv.org/abs/2607.09792) |
| 其他 | Embodied Agents Take Control（第 8 页：25 次失败里 22 次自称到了） | [arXiv:2607.26148](https://arxiv.org/abs/2607.26148) |
| 其他 | DD-PPO（第 10 页：点导航基本被解决） | [arXiv:1911.00357](https://arxiv.org/abs/1911.00357) |
| 其他 | DAgger（第 34 页） | [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) |
| 有什么用 | TrackVLA（跟着人走） | [arXiv:2505.23189](https://arxiv.org/abs/2505.23189) |
| 有什么用 | Mobility VLA（拍张照片，找东西） | [arXiv:2407.07775](https://arxiv.org/abs/2407.07775) |
| 有什么用 | OpenFly（无人机） | [arXiv:2502.18041](https://arxiv.org/abs/2502.18041) |
| 第 4、10 页 · 视频 | NaVILA 项目页 Real-world Results（宇树 G1 走廊、大熊雕像）；第 6 页的仿真画面也来自这里 | <https://navila-bot.github.io/> |
| 第 10 页 · 视频 | Habitat 官方宣传片（按坐标导航那段） | <https://aihabitat.org/> |
| 第 10 页 · 视频 | VLFM 项目页（Spot 找微波炉） | <https://naoki.io/portfolio/vlfm> |
| 第 12 页 · 视频 | TrackVLA 项目页（跟着穿黑衣服的人） | <https://pku-epic.github.io/TrackVLA-web/> |
| 第 31、36、39 页 · 视频 | NaVid 项目页（真机复杂指令、仿真 R2R、真机简单指令） | <https://pku-epic.github.io/NaVid/> |
| 第 45 页 · 视频 | DualVLN 项目页（G1 人形，绕开走过的人） | <https://internrobotics.github.io/internvla-n1-dualvln.github.io/> |
| 第 47 页 · 视频 | Qwen-RobotNav README（机器狗从客厅区走到病房） | <https://github.com/QwenLM/Qwen-RobotNav> |
| 第 50 页 · 视频 | ABot-N1 项目页「城市级长程自主导航」 | <https://amap-cvlab.github.io/ABot-Navigation/ABot-N1/> |
| 第 52 页 · 视频 | Mistral 官方视频（Robostral Navigate 博客里嵌的 YouTube） | <https://www.youtube.com/watch?v=7dpLB9NoY1A> |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：每页数字的出处（论文的表号、图号、章节），每条标了「原文 / 自核 / 推断 / 示意」；里面还有一张各模型预训练底座的对照表（从 ResNet-152 + LSTM 到 Qwen3-VL），和一节「路线三用的是 VLM 还是视频模型」。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 12 段演示视频（第 4、10、12、31、36、39、45、47、50、52 页，第 10 页三段），都是从各项目页和官方视频里切的，仓库里不放（88 MB），视频页换成了片段的一帧，原片链接见上面的表。第 6 页是分步动画：在这一页按 `G` 或 `J` 走一步、`K` 退一步。
