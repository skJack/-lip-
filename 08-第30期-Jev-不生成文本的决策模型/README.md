# 第 30 期 · 模型拆解

**视频标题**：Jev：一个字都不写的模型，一周拿下 HN 近两千分——结构、RLCD 训练、100% 合成数据，和我接进 PaperDance 的实测

TypeSafe AI 2026-09-15 发布的 Jev：不生成文本，给它一段材料（state）和几道选择题（choice / score / noul），它返回每个选项一个校准过的概率，一次前向约 100 ms，输出 token 免费。**这期没有论文也没有开源权重**——官方只公开了接口文档、一个评测站、一篇采访和一篇博客，结构是社区用一万次 API 探针反推出来的。这期按「它是什么 → 结构 → 训练 → 数据 → 实验 → 应用」拆开讲，最后是把它接进 PaperDance（第 19 期讲的刷论文产品）的实测。

## 这期的材料从哪来

没有论文，没有开源权重。所以这期的依据分三类：官方公开的、媒体采访、社区反推和独立评测。

| 类型 | 材料 | 链接 |
|---|---|---|
| 官方 | TypeSafe 发布文：Introducing System One Models & Jev | <https://typesafe.ai/blog/introducing-system-one-models-and-jev> |
| 官方 | 接口文档（state / choice / score / noul、限制、弱点清单） | <https://docs.typesafe.ai> |
| 官方 | 工作流评测站（四个工作流，Jev vs GPT-5.6 / Opus 5） | <https://evals.typesafe.ai/> |
| 官方 | 创始人博客 The Bitterest Lesson（任务 > 数据 > 算力 > 算法） | <https://www.completeskeptic.com/p/the-bitterest-lesson> |
| 媒体 | TechCrunch 专访（transformer、训练数据 100% 合成、一半员工做合成数据） | <https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/> |
| 社区 | Hacker News 主帖（创始人在评论区回复） | <https://news.ycombinator.com/item?id=49717558> |
| 逆向 | Archer Hume《Jev's Architecture Unmasked》（一万次探针反推计算图） | <https://archerhume.com/posts/jevs-architecture-unmasked/> |
| 复现 | 开源复现清单（Laya / kev / NanoJev / open-jev 等 15 个） | <https://systemonemodels.org/examples/alternatives/> |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：视频里每个数字的出处，按页排——标「官方」的来自 TypeSafe 自己的材料，标「反推」的来自第三方探针和独立评测，标「推断」的是我自己的重构。
- [`notes/请求响应示例.json`](notes/请求响应示例.json)：一次真实调用的完整请求和返回（第 8 页那张图的原始 JSON）

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 4 页视频：3 页是 TypeSafe 官方 demo（终端对比、DOOM、Wikirace），1 页是我自己产品 PaperDance 的前后对比。仓库里**不放视频文件**，这几页换成了截图；官方 demo 的原片在 [发布文](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 里可以看。
