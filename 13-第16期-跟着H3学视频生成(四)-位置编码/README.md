# 第 16 期 · 跟着 H3 学视频生成（四）

**视频标题**：跟着H3学视频生成④：音画同步没有专门模块，全靠位置编码

系列第四期，讲第二个配套设计：位置编码。从 attention 为什么不带顺序讲起，到 RoPE 为什么转一转就成了相对位置编码（128 维分成 64 对、各有转速，附计算代码），再到视频模型通用的 3D RoPE。然后是 H3 的两个特殊设计：文本、条件、音频、视频共用一根时间轴（1 格 = 1/40 秒，同一时刻的视频帧和音频格坐标相同，音画同步就是这么来的）；每个头 128 维只转 96 维，留 32 维直通给「距离远但必须强关联」的行。最后把四期串成一句话：H3 的结构不写在权重里，写在每一行的三个标签里。

## 这期的材料从哪来

H3 没有论文，这个系列讲到的 H3 细节全部来自它的开源代码和权重配置；下面是这一期用到的官方材料和论文。

| 类型 | 材料 | 链接 |
|---|---|---|
| 官方 | MiniMax H3 代码仓库 | <https://github.com/MiniMax-AI/MiniMax-H3> |
| 官方 | 权重和 model card（FL2VA / Ref2VA 两个 checkpoint） | <https://huggingface.co/MiniMaxAI/MiniMax-H3> |
| 官方 | diffusers 的 minimax-h3 分支（H3 的 transformer 和 packing 代码，我读的就是这一版） | <https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md> |
| 官方 | MiniMax 开源公告（2026-08-03） | <https://www.minimax.cn/news/minimax-h3-open-source> |
| 论文 | RoFormer: Enhanced Transformer with Rotary Position Embedding（RoPE） | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 对照 | Wan2.2 代码（3D RoPE 的对照：128 维全转、三轴 44/42/42） | <https://github.com/Wan-Video/Wan2.2> |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：口播里每个数字和机制的出处：H3 的数字来自官方开源代码、权重配置和我自己的推理实测，通用知识标了论文出处；最后还有一段评论区可能会问的 Q&A。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这个系列的 deck 比仓库里其他系列早，按键不一样：`→` / 空格 / 单击会先把本页的动画一个个放出来，放完才翻页；没有 `F`、`☰` 和 `#10` 这种跳页。整份 deck 是一个 HTML 文件，图全是 SVG / Canvas 现画的，没有图片文件。左上角有一排页码按钮，鼠标移过去才显出来，点哪个跳哪页。
