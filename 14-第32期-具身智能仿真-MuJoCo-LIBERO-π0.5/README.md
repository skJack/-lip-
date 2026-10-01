# 第 32 期 · 具身智能基础

**视频标题**：具身智能仿真，一次讲清楚：从 MuJoCo、LIBERO 到 π0.5 实战

仿真到底在算什么。先放一张全景图：模型出动作，LIBERO 出题判卷，robosuite 管机器人和控制器，MuJoCo 算物理、出画面。然后从一个小球落地讲 MuJoCo：场景文件 MJCF、mjModel / mjData、每 2 毫秒一次的碰撞检测 → 受力 → 积分，第 221 步球刚碰到地面时地面推了 45 牛；物理用的碰撞形状和相机拍的外观是两套，LIBERO 里的黑碗在物理引擎眼里是 40 个小方块。接着拆开 LIBERO：一道题就是一份 BDDL，动作走 1 步、物理算 25 步，动作放的是目标、手被控制器拽着追，怎样算成功。再让 π0.5 在 LIBERO 里完整跑一局：服务端 / 客户端两个进程，第 47 步抓空、第 79 步重抓、第 109 步成功。最后看 LIBERO-PRO、LIBERO-Plus、RoboTwin 2.0、RoboCasa365、BEHAVIOR、RoboLab 这些新 benchmark 换了哪一层、换引擎换了什么（MuJoCo / SAPIEN / Isaac Sim），以及 π0.5 和各榜第一在上面考了多少。**这期的例子全是我们自己跑的**（一张 RTX 3090），小球和 π0.5 那一局的画面都是用 MuJoCo 重新渲染的。

## 这期的材料从哪来

这期没有单篇主讲的论文，讲的是仿真本身：MuJoCo、LIBERO、π0.5 都是我们自己跑的，数字来自运行记录和读代码；最后几页新 benchmark 的分数来自各家论文和官方排行榜（2026-09-28 查）。下面是用到的官方材料、代码和论文。

| 类型 | 材料 | 链接 |
|---|---|---|
| 官方 | MuJoCo 文档（XML reference、Computation 一章的软接触） | <https://mujoco.readthedocs.io/en/stable/overview.html> |
| 论文 | MuJoCo: A physics engine for model-based control（IROS 2012） | <https://doi.org/10.1109/IROS.2012.6386109> |
| 代码 | robosuite 1.4.0（OSC 控制器、一次 step 拆 25 个子步） | <https://github.com/ARISE-Initiative/robosuite/tree/v1.4.0> |
| 论文 | LIBERO（NeurIPS 2023 数据集与 benchmark 赛道） | [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) |
| 代码 | LIBERO 代码（BDDL 题目文件、判成功的谓词） | <https://github.com/Lifelong-Robot-Learning/LIBERO> |
| 论文 | π0.5 | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) |
| 代码 | openpi（π0.5 的 LIBERO 权重和评测脚本 examples/libero） | <https://github.com/Physical-Intelligence/openpi> |
| 论文 | LIBERO-PRO | [arXiv:2510.03827](https://arxiv.org/abs/2510.03827) |
| 论文 | LIBERO-Plus | [arXiv:2510.13626](https://arxiv.org/abs/2510.13626) |
| 论文 | RoboTwin 2.0 | [arXiv:2506.18088](https://arxiv.org/abs/2506.18088) |
| 排行榜 | RoboTwin 2.0 排行榜 | <https://robotwin-platform.github.io/leaderboard> |
| 论文 | RoboCasa365 | [arXiv:2603.04356](https://arxiv.org/abs/2603.04356) |
| 排行榜 | RoboCasa365 排行榜 | <https://robocasa.ai/leaderboard.html> |
| 论文 | BEHAVIOR-1K | [arXiv:2403.09227](https://arxiv.org/abs/2403.09227) |
| 官方 | BEHAVIOR Challenge 2025（结果）和 2026 | <https://behavior.stanford.edu/challenge/index.html> |
| 论文 | BEHAVIOR Challenge 2025 冠军报告 | [arXiv:2512.06951](https://arxiv.org/abs/2512.06951) |
| 论文 | RoboLab（RSS 2026） | [arXiv:2604.09860](https://arxiv.org/abs/2604.09860) |
| 排行榜 | RoboLab 排行榜 | <https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html> |
| 论文 | ManiSkill3（SAPIEN 上的 GPU 并行仿真） | [arXiv:2410.00425](https://arxiv.org/abs/2410.00425) |
| 官方 | Isaac Sim 的显卡要求（要带 RT 核） | <https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html> |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：视频里每个数字的出处，分三类：自己跑出来的（MuJoCo、LIBERO、π0.5 的运行记录）、读代码看到的（LIBERO、robosuite、openpi）、引用论文和排行榜的；哪些是我的推断也标出来了。最后还有一段视频里没讲的：同一个开局跑两次，过程为什么不一样。
- [`notes/benchmark数字核查.md`](notes/benchmark数字核查.md)：新 benchmark 那几页的每个数字回原文核对的明细：论文版本、表号 / 图号、排行榜链接和访问日期；B 节是第 36 页五张缩略图的出处

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 3 段视频：小球落地、步长 0.002 和 0.05 的对比、π0.5 那一局的高清回放，都是我们自己用 MuJoCo 渲染的，直接放在 `slides/media/` 里，离线可看。
