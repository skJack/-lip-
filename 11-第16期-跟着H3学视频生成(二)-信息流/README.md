# 第 16 期 · 跟着 H3 学视频生成（二）

**视频标题**：跟着H3学视频生成②：不架桥了，39230行拼成一条序列

系列第二期，看 H3 模型本身：它把文本、条件帧、音频、视频四种东西拼成一条 39,230 行的序列，50 层 transformer 一套权重、零 cross-attention。先对照 Wan2.2 的做法（条件帧在通道上拼进输入、文本单独一条流、每层一座单向的 cross-attention 桥），再用一张全景图走完「首帧图 + 一句话 → 5 秒有声视频」：每种原料怎么变成行、怎么对齐到 5376 维、attention 里谁看谁、输出头留下哪些行。最后算账：加条件只改拼法、音画同步是原生的，代价是序列长、attention 占掉八成算力。

## 这期的材料从哪来

H3 没有论文，这个系列讲到的 H3 细节全部来自它的开源代码和权重配置；下面是这一期用到的官方材料和论文。

| 类型 | 材料 | 链接 |
|---|---|---|
| 官方 | MiniMax H3 代码仓库 | <https://github.com/MiniMax-AI/MiniMax-H3> |
| 官方 | 权重和 model card（FL2VA / Ref2VA 两个 checkpoint） | <https://huggingface.co/MiniMaxAI/MiniMax-H3> |
| 官方 | diffusers 的 minimax-h3 分支（H3 的 transformer 和 packing 代码，我读的就是这一版） | <https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md> |
| 官方 | MiniMax 开源公告（2026-08-03） | <https://www.minimax.cn/news/minimax-h3-open-source> |
| 对照 | Wan2.2 代码（第 4 页的通道拼接和 cross-attention 对照的是这里的 `wan/modules/model.py`） | <https://github.com/Wan-Video/Wan2.2> |
| 对照 | Wan: Open and Advanced Large-Scale Video Generative Models（Wan 技术报告） | [arXiv:2503.20314](https://arxiv.org/abs/2503.20314) |
| 榜单 | Artificial Analysis 的排名推文（视频编辑第一，文生视频、图生视频前三） | <https://x.com/ArtificialAnlys/status/2083042088338538594> |
| 社区 | Hugging Face 社区博客：What Is MiniMax H3 (Hailuo 3.0)? | <https://huggingface.co/blog/ResterChed/minimax-h3-hailuo-3-0> |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：口播里每个数字和机制的出处：H3 的数字来自官方开源代码、权重配置和我自己的推理实测，通用知识标了论文出处；最后还有一段评论区可能会问的 Q&A。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这个系列的 deck 比仓库里其他系列早，按键不一样：`→` / 空格 / 单击会先把本页的动画一个个放出来，放完才翻页；没有 `F`、`☰` 和 `#10` 这种跳页。整份 deck 是一个 HTML 文件，图全是 SVG / Canvas 现画的，没有图片文件。
