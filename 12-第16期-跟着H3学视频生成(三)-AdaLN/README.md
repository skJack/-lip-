# 第 16 期 · 跟着 H3 学视频生成（三）

**视频标题**：跟着H3学视频生成③：33B的模型，推理时只装19.3B

系列第三期，讲 H3 为「拼成一条序列」配套的第一个设计：AdaLN。先讲 LayerNorm 为什么要有、scale / shift 在补什么；再讲 AdaLN 的三个调制参数（scale、shift、gate）为什么由噪声水平 t 生成；然后是 H3 独有的二维参数表——按（噪声水平, 模态）查 9 个格子，这是 50 层里唯一区分「每一行是什么」的机制，参考帧和目标视频就靠取不同的格子区分；最后是工程上的便宜：这层 13B 参数的输入只有 t，推理前可以全部算完，33B 的模型显存里只常驻 19.3B。

## 这期的材料从哪来

H3 没有论文，这个系列讲到的 H3 细节全部来自它的开源代码和权重配置；下面是这一期用到的官方材料和论文。

| 类型 | 材料 | 链接 |
|---|---|---|
| 官方 | MiniMax H3 代码仓库 | <https://github.com/MiniMax-AI/MiniMax-H3> |
| 官方 | 权重和 model card（FL2VA / Ref2VA 两个 checkpoint） | <https://huggingface.co/MiniMaxAI/MiniMax-H3> |
| 官方 | diffusers 的 minimax-h3 分支（H3 的 transformer 和 packing 代码，我读的就是这一版） | <https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md> |
| 官方 | MiniMax 开源公告（2026-08-03） | <https://www.minimax.cn/news/minimax-h3-open-source> |
| 论文 | Scalable Diffusion Models with Transformers（DiT，adaLN-Zero 的出处） | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) |
| 论文 | Layer Normalization | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) |
| 论文 | Root Mean Square Layer Normalization（RMSNorm，H3 实际用的归一化） | [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：口播里每个数字和机制的出处：H3 的数字来自官方开源代码、权重配置和我自己的推理实测，通用知识标了论文出处；最后还有一段评论区可能会问的 Q&A。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这个系列的 deck 比仓库里其他系列早，按键不一样：`→` / 空格 / 单击会先把本页的动画一个个放出来，放完才翻页；没有 `F`、`☰` 和 `#10` 这种跳页。整份 deck 是一个 HTML 文件，图全是 SVG / Canvas 现画的，没有图片文件。左上角有一排页码按钮，鼠标移过去才显出来，点哪个跳哪页。
