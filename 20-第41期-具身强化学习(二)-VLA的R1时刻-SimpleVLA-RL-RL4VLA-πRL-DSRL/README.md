# 第 41 期 · 具身智能中的强化学习（二）

**视频标题**：VLA 中的强化学习

「具身智能中的强化学习」第二期「VLA 的 R1 时刻」：一个几十亿参数、只会照着示范做的 VLA，怎么用强化学习接着练。开场从 DeepSeek-R1 的 aha moment 讲起（R1-Zero 的奖励只看答案对不对和格式，练着练着自己学会了回头检查），对到机器人上：SimpleVLA-RL 的示范里全是把罐子抓起来，只给成没成的 0/1 奖励，练完它自己学会了直接推过去（pushcut）；Physical Intelligence 的 RL Token 加了 RL 以后，插网线比人遥操作的示范还快一倍多。然后是给 VLA 做 RL 的两大难点（模型大；只会模仿：没有价值网络、flow 头算不出概率、一次出一整块动作）和各篇的解法，再按动作头把这期分成三种练法。先补 **GRPO**：和第一期的 PPO 对照（不要价值网络，同一个起点跑一组、组里比），分步算例、式子逐项拆开、DeepSeek 怎么训 R1-Zero、为什么能成、为什么适合 VLA、0/1 奖励的好处和代价。**范式一 · token 头直接练**：SimpleVLA-RL 照开源代码讲一步里进出的是什么、三个让它多试的技巧、每个任务只给 1 条示范从 48.9% 练到 96.9%、pushcut、起点 0% 练完还是 0%、前作和后续各改了哪一块；对照 RL4VLA（PPO + 接在 VLA 特征上的小价值头、泛化怎么测、PPO 比 GRPO 稳）。**范式二 · flow 头直接练**：πRL，flow 头卡在哪，Flow-Noise 和 Flow-SDE 两个办法怎么往去噪里加噪声、让概率能算，没有标准答案时一轮怎么练，结果和局限。**范式三 · 不改大模型**：DSRL，VLA 一个参数不动，只训一个挑起点噪声的小策略（SAC），真机上 π0 开烤面包机从 20 次成 5 次练到 18 次。最后把三种练法放在一起比，再说四件还在争的事：带来的是泛化还是熟练、PPO 还是 GRPO、改权重还是加小头、RL 只能放大已有的能力。

## 这期的材料从哪来

这期讲怎么给 VLA 做强化学习：主讲 SimpleVLA-RL、RL4VLA、πRL、DSRL 四篇，GRPO 按 DeepSeekMath 和 DeepSeek-R1 讲，动机和难点照斯坦福 CS224R 2026 年那一讲。下表是 deck 里用到的论文（arXiv 链接）、代码和官方页面，每页数字的出处在 `notes/技术事实核查.md`。deck 里的 4 段视频（第 5 页 RL Token 插网线、第 37 页 DSRL 开烤面包机）是从两个项目页的官方视频里切的，仓库里不放，视频页换成了片段的一帧。

| 类型 | 材料 | 链接 |
|---|---|---|
| 开场 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（R1-Zero、aha moment） | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| 开场 · 视频 | RL Token: Bootstrapping Online RL with Vision-Language-Action Models（Physical Intelligence，第 5 页插网线） | [arXiv:2604.23073](https://arxiv.org/abs/2604.23073) |
| 开场 · 视频 | RL Token 项目页（第 5 页左右两段视频都来自这里的官方对比视频） | <https://pi.website/research/rlt> |
| 开场 | 斯坦福 CS224R 2026「RL for Robot Foundation Models」课件（第 5–6 页的动机和难点） | <https://cs224r.stanford.edu/slides/17_cs224r_rl_vlas_2026.pdf> |
| GRPO | DeepSeekMath（GRPO 出自这篇） | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| GRPO | PPO（第一期讲过，这期拿来对照） | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| GRPO | GAE（RL4VLA、πRL 算优势用的，和第一期一样） | [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) |
| 范式一 · 主讲 | SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning（ICLR 2026） | [arXiv:2509.09674](https://arxiv.org/abs/2509.09674) |
| 范式一 · 代码 | SimpleVLA-RL 开源代码（第 18 页照它讲） | <https://github.com/PRIME-RL/SimpleVLA-RL> |
| 范式一 · 代码 | veRL（SimpleVLA-RL 基于的训练框架） | <https://github.com/volcengine/verl> |
| 范式一 · 技巧 | DAPO（第 19 页裁剪只放上限 [0.8, 1.28] 照它） | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| 范式一 · 底座 | OpenVLA | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |
| 范式一 · 底座 | OpenVLA-OFT（并行解码，SimpleVLA-RL 在它上面把输出换回 token） | [arXiv:2502.19645](https://arxiv.org/abs/2502.19645) |
| 范式一 · 对比 | RDT-1B（第 20 页 RoboTwin 2.0 上对比的模型之一） | [arXiv:2410.07864](https://arxiv.org/abs/2410.07864) |
| 范式一 · 前作和同期 | GRAPE: Generalizing Robot Policy via Preference Alignment | [arXiv:2411.19309](https://arxiv.org/abs/2411.19309) |
| 范式一 · 前作和同期 | iRe-VLA：Improving Vision-Language-Action Model with Online Reinforcement Learning | [arXiv:2501.16664](https://arxiv.org/abs/2501.16664) |
| 范式一 · 前作和同期 | RIPT-VLA：Interactive Post-Training for Vision-Language-Action Models | [arXiv:2505.17016](https://arxiv.org/abs/2505.17016) |
| 范式一 · 前作和同期 | VLA-RL | [arXiv:2505.18719](https://arxiv.org/abs/2505.18719) |
| 范式一 · 前作和同期 | TGRPO | [arXiv:2506.08440](https://arxiv.org/abs/2506.08440) |
| 范式一 · 对照 | RL4VLA：What Can RL Bring to VLA Generalization? An Empirical Study（NeurIPS 2025） | [arXiv:2505.19789](https://arxiv.org/abs/2505.19789) |
| 范式一 · 对照 | RL4VLA 项目页 | <https://rlvla.github.io/> |
| 范式一 · 对照 | RLinf-VLA（PPO 和 GRPO 的对照） | [arXiv:2510.06710](https://arxiv.org/abs/2510.06710) |
| 范式二 · 主讲 | πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models | [arXiv:2510.25889](https://arxiv.org/abs/2510.25889) |
| 范式二 · 代码 | RLinf（πRL 的开源代码） | <https://github.com/RLinf/RLinf> |
| 范式二 · 底座 | π0 | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
| 范式二 · 底座 | π0.5 | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) |
| 范式二 · 思路来源 | ReinFlow（Flow-Noise 的思路来自它） | [arXiv:2505.22094](https://arxiv.org/abs/2505.22094) |
| 范式二 · 思路来源 | Flow-GRPO（Flow-SDE 的思路来自图像生成里的它） | [arXiv:2505.05470](https://arxiv.org/abs/2505.05470) |
| 范式三 · 主讲 | DSRL: Steering Your Diffusion Policy with Latent Space Reinforcement Learning | [arXiv:2506.15799](https://arxiv.org/abs/2506.15799) |
| 范式三 · 视频 | DSRL 项目页（第 37 页开烤面包机的两段视频来自这里） | <https://diffusion-steering.github.io/> |
| 范式三 · 代码 | DSRL 开源代码（Robomimic / Gym） | <https://github.com/ajwagen/dsrl> |
| 范式三 · 代码 | DSRL 的 π0 版 | <https://github.com/nakamotoo/dsrl_pi0> |
| 范式三 · 相关 | SAC：Soft Actor-Critic | [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) |
| 范式三 · 相关 | DPPO（DSRL 在 Robomimic / Gym 上对比的方法之一） | [arXiv:2409.00588](https://arxiv.org/abs/2409.00588) |
| 范式三 · 相关 | GR-RL（字节，在线阶段用了 DSRL 的思路） | [arXiv:2512.01801](https://arxiv.org/abs/2512.01801) |
| benchmark | LIBERO | [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) |
| benchmark | RoboTwin 2.0 | [arXiv:2506.18088](https://arxiv.org/abs/2506.18088) |
| benchmark | ManiSkill3 | [arXiv:2410.00425](https://arxiv.org/abs/2410.00425) |
| 路线图 · 第三期 | WMPO | [arXiv:2511.09515](https://arxiv.org/abs/2511.09515) |
| 路线图 · 第三期 | RISE | [arXiv:2602.11075](https://arxiv.org/abs/2602.11075) |
| 路线图 · 第三期 | WoVR | [arXiv:2602.13977](https://arxiv.org/abs/2602.13977) |
| 路线图 · 第四期 | HIL-SERL | [arXiv:2410.21845](https://arxiv.org/abs/2410.21845) |
| 路线图 · 第四期 | π*0.6（第 23 期讲过；RL Token 的基座 π0.6 出自这里） | [arXiv:2511.14759](https://arxiv.org/abs/2511.14759) |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：每页数字的出处（论文版本、章节、表号、图号，SimpleVLA-RL 和 πRL 照开源代码核到行），每条标「已核 / 推算 / 判断 / 估读」；各段开头先列了版本和几处对不上的地方（SimpleVLA-RL 的 ICLR 正式版和 arXiv v1 数不一样、πRL 三个版本的 ManiSkill 数、DSRL「65 局约 1 万步」是真机任务不是 LIBERO 等）。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 4 段视频（第 5 页 RL Token 插网线左右两段、第 37 页 DSRL 开烤面包机两段），仓库里不放视频文件，视频页换成了片段的一帧，原片在 https://pi.website/research/rlt 和 https://diffusion-steering.github.io/ 。分步页（第 10、11、16、18、26、32、36 页）在这一页按 `G` 或 `J` 走一步、`K` 退一步。
