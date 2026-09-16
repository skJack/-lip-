# 可lip · Slides 与论文笔记

这是我在抖音 / B站 / 小红书 / 视频号上「可lip」账号讲论文的视频的配套资料：每期视频里用的 PPT（HTML 版，可下载本地翻页）和我读对应论文时写的整理笔记。目前两个系列：

- **VLA 入门**（第 21–24 期）：机器人大模型 VLA（Vision-Language-Action），按 Physical Intelligence 的 π 系列时间线走：RT-2 → OpenVLA → π0 → π0.5 → π*0.6 → π0.7
- **世界模型**（第 27 期起）：第一期从 2018 年的 World Models 讲到 DayDreamer（在模型的梦里训策略）；第二期讲 LeCun 的 JEPA 这条线（不生成像素、在表征空间预测），从 2022 年立场文讲到 V-JEPA 2 真机抓放

| 期 | 系列 | 视频标题 | 讲的论文 | 目录 |
|---|---|---|---|---|
| 第 21 期 | VLA 入门（一） | 机器人大模型 VLA 入门：从 Google 的 RT-2 到 Physical Intelligence 的 π0，三篇论文讲清动作是怎么输出的 | RT-2、OpenVLA、π0（前置：SayCan） | [`01-第21期-VLA入门(一)-RT-2-OpenVLA-π0/`](01-第21期-VLA入门(一)-RT-2-OpenVLA-π0/) |
| 第 22 期 | VLA 入门（二） | π0.5：机器人怎么进没见过的家干活｜VLA 入门 | π0.5（前置：π0-FAST、Hi Robot） | [`02-第22期-VLA入门(二)-π0.5/`](02-第22期-VLA入门(二)-π0.5/) |
| 第 23 期 | VLA 入门（三） | π*0.6 强化学习：让机器人越练越强 | π*0.6 (RECAP) | [`03-第23期-VLA入门(三)-πstar0.6-RECAP强化学习/`](03-第23期-VLA入门(三)-πstar0.6-RECAP强化学习/) |
| 第 24 期 | VLA 入门（四） | π0.7 论文讲清楚：一个通用机器人模型不做微调，直接叠衣服、做咖啡、进没见过的厨房 | π0.7（前置：MEM、RTC） | [`04-第24期-VLA入门(四)-π0.7/`](04-第24期-VLA入门(四)-π0.7/) |
| 第 27 期 | 世界模型（一） | 世界模型第1期 · 梦的开始：从 World Models 到 DayDreamer，让策略在梦里练，机器狗真机 1 小时学会走路 | World-Models、DayDreamer、PlaNet、Dreamer-v1、Dreamer-v2、Dreamer-v3 | [`05-第27期-世界模型(一)-梦的开始-WorldModels-Dreamer-DayDreamer/`](05-第27期-世界模型(一)-梦的开始-WorldModels-Dreamer-DayDreamer/) |
| 第 28 期 | 世界模型（二） | 世界模型第2期 · JEPA：不生成像素，在表征空间预测未来｜从 LeCun 立场文到 V-JEPA 2 真机零样本抓放 | LeCun-2022-立场文、V-JEPA-2、V-JEPA、DINO-WM（前置：VLA-JEPA、LeWorldModel） | [`06-第28期-世界模型(二)-JEPA-LeCun立场文-IJEPA-VJEPA-DINOWM-VJEPA2/`](06-第28期-世界模型(二)-JEPA-LeCun立场文-IJEPA-VJEPA-DINOWM-VJEPA2/) |

## 目录结构

每期一个文件夹，编号就是视频期数，里面固定两个子目录：

```
NN-第NN期-系列名(x)-论文名/
├── README.md            这期讲了什么、涉及哪些论文（带 arXiv 链接）
├── slides/
│   ├── deck.html        视频里用的 PPT，Chrome 打开，方向键翻页
│   ├── media/           PPT 用到的图
│   ├── 使用说明.md       快捷键 + 每一页对应讲什么
│   └── 预览-全部页面.png  全部页面缩略图
└── notes/
    └── <论文名>-调研笔记.md   固定 6 段：要解决的问题 / 方法 / 实验 / 局限 / 复现要点 / 关键引用链
```

## 怎么看 Slides

1. 点右上角 **Code → Download ZIP**（或 `git clone`），解压
2. 用 Chrome 打开某一期的 `slides/deck.html`
3. `→` / 空格下一页，`←` 上一页，`F` 全屏，左上角 `☰` 展开页面列表；`deck.html#10` 直接跳第 10 页

第 24 期的 6 页演示视频不在仓库里，`deck.html` 从 Physical Intelligence 官网直接加载，看那几页需要联网。第 27 期的视频（World Models 官网的演示片段、DayDreamer 的官方视频）体积不大，直接放在 `slides/media/video/` 里，离线可看。第 28 期没有视频，第 16 页是键盘控制的分步动画（`J` 下一步、`K` 上一步）。

## 关于笔记和图片

- 笔记是我自己读论文的整理，数字来自论文正文和图表；PI 的结果图很多是没有数值的柱状图，笔记里标「估读」的数是从图上估的，视频里我都说的「左右」。
- PPT 里的论文插图版权归各论文作者（Google DeepMind、Stanford、Physical Intelligence、UC Berkeley 等），来源在各期的 `使用说明.md` 和 `notes/` 里都有标注；本仓库只用于学习交流。
- 论文原文请去 arXiv 看，仓库里不放 PDF。

视频在各平台搜「可lip」。有问题欢迎开 issue 或在视频评论区留言。
