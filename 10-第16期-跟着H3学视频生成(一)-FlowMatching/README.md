# 第 16 期 · 跟着 H3 学视频生成（一）

**视频标题**：跟着 H3 学视频生成①：Flow Matching，所有视频模型的训练方法

「跟着 H3 学视频生成」系列第一期。MiniMax 2026 年 8 月开源的 H3（海螺 3.0）能把画面和声音一起生成，这个系列一期拆它一个模块，顺便把视频生成里对应的知识点讲清楚。第一期讲 H3 和 SD3、Flux、Wan 共用的训练方法 Flow Matching：生成模型在学一个从噪声分布到视频分布的映射；训练时把画面和噪声按 t 混合，让模型猜速度 v = x₁ − ε（答案出题时就已知，不用人工标注）；用一张 3×3 的数表把 loss 从头算一遍；最后讲推理为什么不能一步走完、H3 为什么走 8 步。

## 这期的材料从哪来

H3 没有论文，这个系列讲到的 H3 细节全部来自它的开源代码和权重配置；下面是这一期用到的官方材料和论文。

| 类型 | 材料 | 链接 |
|---|---|---|
| 官方 | MiniMax H3 代码仓库 | <https://github.com/MiniMax-AI/MiniMax-H3> |
| 官方 | 权重和 model card（FL2VA / Ref2VA 两个 checkpoint） | <https://huggingface.co/MiniMaxAI/MiniMax-H3> |
| 官方 | diffusers 的 minimax-h3 分支（H3 的 transformer 和 packing 代码，我读的就是这一版） | <https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md> |
| 官方 | MiniMax 开源公告（2026-08-03） | <https://www.minimax.cn/news/minimax-h3-open-source> |
| 论文 | Flow Matching for Generative Modeling（Lipman et al. 2022） | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) |
| 论文 | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow（Rectified Flow） | [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) |
| 论文 | Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（SD3，H3 采样用的 timestep shift 出自这里） | [arXiv:2403.03206](https://arxiv.org/abs/2403.03206) |
| 背景 | Generative Adversarial Networks（GAN，2014） | [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) |
| 背景 | Denoising Diffusion Probabilistic Models（DDPM，2020） | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：口播里每个数字和机制的出处：H3 的数字来自官方开源代码、权重配置和我自己的推理实测，通用知识标了论文出处；最后还有一段评论区可能会问的 Q&A。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这个系列的 deck 比仓库里其他系列早，按键不一样：`→` / 空格 / 单击会先把本页的动画一个个放出来，放完才翻页；没有 `F`、`☰` 和 `#10` 这种跳页。整份 deck 是一个 HTML 文件，图全是 SVG / Canvas 现画的，没有图片文件。第 4 页可以拖滑块，看 t 从 0 到 1 时画面怎么从一屏噪声变清晰。
