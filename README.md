# 可lip · Slides 与论文笔记

这是我在抖音 / B站 / 小红书 / 视频号上「可lip」账号讲论文的视频的配套资料：每期视频里用的 PPT（HTML 版，可下载本地翻页）和我读对应论文时写的整理笔记。目前三个系列，外加两个单期（模型拆解、具身智能仿真）：

- **跟着 H3 学视频生成**（第 16 期，拆成四个视频）：拿 MiniMax 2026 年 8 月开源的音画一体视频模型 H3（海螺 3.0）当例子，一期拆它一个模块，顺便把视频生成的知识点讲清楚：Flow Matching 训练方法 → 信息流（四种输入拼成一条序列）→ AdaLN → 位置编码。H3 没有论文，讲的都来自它的开源代码
- **VLA 入门**（第 21–24 期）：机器人大模型 VLA（Vision-Language-Action），按 Physical Intelligence 的 π 系列时间线走：RT-2 → OpenVLA → π0 → π0.5 → π*0.6 → π0.7
- **世界模型**（第 27 期起）：第一期从 2018 年的 World Models 讲到 DayDreamer（在模型的梦里训策略）；第二期讲 LeCun 的 JEPA 这条线（不生成像素、在表征空间预测），从 2022 年立场文讲到 V-JEPA 2 真机抓放；第三期讲拿视频生成模型当世界模型这条线，UniPi、Genie 做背景，重点是 NVIDIA 的 Cosmos Policy 和 Cosmos 3，看动作怎么一步步从模型外面走进模型里面
- **模型拆解**（单期）：第 30 期讲 TypeSafe 的 Jev——一个不生成文本、只返回校准概率的决策模型。它没有论文也没有权重，资料是官方文档 + 社区一万次 API 探针的逆向 + 九份独立评测
- **具身智能基础**（单期）：第 32 期讲仿真到底在算什么——从一个小球落地讲 MuJoCo 每 2 毫秒算的碰撞、受力、积分，拆开 LIBERO 怎么出题、怎么判，再让 π0.5 在 LIBERO 里完整跑一局（抓空、重抓、放好），最后看 LIBERO-PRO、RoboTwin 2.0、RoboCasa365、BEHAVIOR、RoboLab 这些新 benchmark 换了哪一层。例子全是自己跑的

| 期 | 系列 | 视频标题 | 讲的论文 | 目录 |
|---|---|---|---|---|
| 第 16 期 | 跟着 H3 学视频生成（一） | 跟着 H3 学视频生成①：Flow Matching，所有视频模型的训练方法 | Flow Matching、Rectified Flow（H3 本身没有论文） | [`10-第16期-跟着H3学视频生成(一)-FlowMatching/`](10-第16期-跟着H3学视频生成(一)-FlowMatching/) |
| 第 16 期 | 跟着 H3 学视频生成（二） | 跟着H3学视频生成②：不架桥了，39230行拼成一条序列 | 无论文（H3 开源代码，对照 Wan2.2） | [`11-第16期-跟着H3学视频生成(二)-信息流/`](11-第16期-跟着H3学视频生成(二)-信息流/) |
| 第 16 期 | 跟着 H3 学视频生成（三） | 跟着H3学视频生成③：33B的模型，推理时只装19.3B | DiT 的 adaLN-Zero（H3 开源代码） | [`12-第16期-跟着H3学视频生成(三)-AdaLN/`](12-第16期-跟着H3学视频生成(三)-AdaLN/) |
| 第 16 期 | 跟着 H3 学视频生成（四） | 跟着H3学视频生成④：音画同步没有专门模块，全靠位置编码 | RoPE / RoFormer（H3 开源代码） | [`13-第16期-跟着H3学视频生成(四)-位置编码/`](13-第16期-跟着H3学视频生成(四)-位置编码/) |
| 第 21 期 | VLA 入门（一） | 机器人大模型 VLA 入门：从 Google 的 RT-2 到 Physical Intelligence 的 π0，三篇论文讲清动作是怎么输出的 | RT-2、OpenVLA、π0（前置：SayCan） | [`01-第21期-VLA入门(一)-RT-2-OpenVLA-π0/`](01-第21期-VLA入门(一)-RT-2-OpenVLA-π0/) |
| 第 22 期 | VLA 入门（二） | π0.5：机器人怎么进没见过的家干活｜VLA 入门 | π0.5（前置：π0-FAST、Hi Robot） | [`02-第22期-VLA入门(二)-π0.5/`](02-第22期-VLA入门(二)-π0.5/) |
| 第 23 期 | VLA 入门（三） | π*0.6 强化学习：让机器人越练越强 | π*0.6 (RECAP) | [`03-第23期-VLA入门(三)-πstar0.6-RECAP强化学习/`](03-第23期-VLA入门(三)-πstar0.6-RECAP强化学习/) |
| 第 24 期 | VLA 入门（四） | π0.7 论文讲清楚：一个通用机器人模型不做微调，直接叠衣服、做咖啡、进没见过的厨房 | π0.7（前置：MEM、RTC） | [`04-第24期-VLA入门(四)-π0.7/`](04-第24期-VLA入门(四)-π0.7/) |
| 第 27 期 | 世界模型（一） | 世界模型第1期 · 梦的开始：从 World Models 到 DayDreamer，让策略在梦里练，机器狗真机 1 小时学会走路 | World-Models、DayDreamer、PlaNet、Dreamer-v1、Dreamer-v2、Dreamer-v3 | [`05-第27期-世界模型(一)-梦的开始-WorldModels-Dreamer-DayDreamer/`](05-第27期-世界模型(一)-梦的开始-WorldModels-Dreamer-DayDreamer/) |
| 第 28 期 | 世界模型（二） | 世界模型第2期 · JEPA：不生成像素，在表征空间预测未来｜从 LeCun 立场文到 V-JEPA 2 真机零样本抓放 | LeCun-2022-立场文、V-JEPA-2、V-JEPA、DINO-WM（前置：VLA-JEPA、LeWorldModel） | [`06-第28期-世界模型(二)-JEPA-LeCun立场文-IJEPA-VJEPA-DINOWM-VJEPA2/`](06-第28期-世界模型(二)-JEPA-LeCun立场文-IJEPA-VJEPA-DINOWM-VJEPA2/) |
| 第 29 期 | 世界模型（三） | 世界模型第3期 · Cosmos：视频生成模型怎么变成机器人策略｜从 UniPi、Genie 到 NVIDIA 的 Cosmos Policy 和 Cosmos 3 | Cosmos-Policy、Cosmos-3、UniPi、Genie、Cosmos-1（前置：Cosmos-Predict2.5） | [`07-第29期-世界模型(三)-Cosmos-UniPi-Genie-CosmosPolicy-Cosmos3/`](07-第29期-世界模型(三)-Cosmos-UniPi-Genie-CosmosPolicy-Cosmos3/) |
| 第 30 期 | 模型拆解 | Jev：一个字都不写的模型，一周拿下 HN 近两千分——结构、RLCD 训练、100% 合成数据，和我接进 PaperDance 的实测 | 无论文（官方文档 + 社区逆向 + 独立评测） | [`08-第30期-Jev-不生成文本的决策模型/`](08-第30期-Jev-不生成文本的决策模型/) |
| 第 31 期 | 世界模型（四） | 世界模型第4期 · World Action Model：视频预测到底怎么帮到动作｜从 UniPi、GR-1、VPP、UWM 到 NVIDIA DreamZero 和蚂蚁 LingBot-VA | DreamZero、LingBot-VA、UniPi、GR-1、VPP、UWM（前置：Fast-WAM、WM-to-WAM-教程） | [`09-第31期-世界模型(四)-WorldActionModel-DreamZero-LingBotVA/`](09-第31期-世界模型(四)-WorldActionModel-DreamZero-LingBotVA/) |
| 第 32 期 | 具身智能基础 | 具身智能仿真，一次讲清楚：从 MuJoCo、LIBERO 到 π0.5 实战 | MuJoCo、LIBERO、π0.5（自己跑）；新 benchmark：LIBERO-PRO、RoboTwin 2.0、RoboCasa365、BEHAVIOR、RoboLab | [`14-第32期-具身智能仿真-MuJoCo-LIBERO-π0.5/`](14-第32期-具身智能仿真-MuJoCo-LIBERO-π0.5/) |

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

第 16 期（H3 四期）的 deck 比其他系列早，按键不一样：画布 1280×720，`→` / 空格 / 单击会先把本页的动画一个个放出来、放完才翻页，`←` 回退，`Home` / `End` 首尾页；没有 `F`、`☰` 和 `#10` 这种跳页（第三、四期左上角有一排页码按钮，鼠标移过去才显出来）。这四份 deck 的图全是 SVG / Canvas 现画的，没有图片文件。

第 30 期的 4 页视频也不在仓库里（3 页 TypeSafe 官方 demo、1 页我自己产品的前后对比），那几页换成了截图。第 24 期的 6 页演示视频不在仓库里，`deck.html` 从 Physical Intelligence 官网直接加载，看那几页需要联网。第 27 期的视频（World Models 官网的演示片段、DayDreamer 的官方视频）体积不大，直接放在 `slides/media/video/` 里，离线可看。第 28 期和第 29 期没有视频，各有一页键盘控制的分步动画（第 28 期第 16 页、第 29 期第 22 页，`J` 下一步、`K` 上一步）。第 32 期的 3 段视频（小球落地、步长对比、π0.5 一局的高清回放）是我们自己用 MuJoCo 渲染的，放在 `slides/media/` 里，离线可看。

## 关于笔记和图片

- 第 30 期没有论文笔记，只有一份「技术事实核查」：视频里每个数字的出处，标明哪些是厂商自己公布的、哪些是第三方测的、哪些是我的推断。
- 第 32 期也没有论文笔记：`notes/` 里一份「技术事实核查」（每个数字是自己跑出来的、读代码看到的，还是引用论文和排行榜的），一份「benchmark 数字核查」（新 benchmark 的分数逐条回原文核对的明细，带论文版本、表号和排行榜链接）。
- 第 16 期（H3 四期）也没有论文笔记：H3 没发论文，每期的 `notes/` 是一份「技术事实核查」——H3 的数字来自官方开源代码、权重配置和我自己的推理实测，通用知识（Flow Matching、AdaLN、RoPE）标了论文出处。
- 笔记是我自己读论文的整理，数字来自论文正文和图表；PI 的结果图很多是没有数值的柱状图，笔记里标「估读」的数是从图上估的，视频里我都说的「左右」。
- PPT 里的论文插图版权归各论文作者（Google DeepMind、Stanford、Physical Intelligence、UC Berkeley、MIT、NVIDIA 等），第 30 期里的产品截图和图表版权归 TypeSafe AI 及各报道方，第 32 期第 36 页的 benchmark 缩略图取自各论文首图和项目页；来源在各期的 `使用说明.md` 和 `notes/` 里都有标注；本仓库只用于学习交流。
- 论文原文请去 arXiv 看，仓库里不放 PDF。

视频在各平台搜「可lip」。有问题欢迎开 issue 或在视频评论区留言。
