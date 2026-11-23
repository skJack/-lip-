# 第 50 期 · 真机部署方法总结（上）

**视频标题**：真机部署方法总结（上）

「真机部署方法总结」上集：模型不重训（最多微调一下），部署时就能改的 19 招。开场是 Dream Machines 的例子：同一份 π0.5 权重，只改两个部署参数，产线双臂插装成功率 53% → 93%。三段：**边做边想**（异步推理、RTC、训练时 RTC、VLASH、A2C2、FASTER，以及 RTC 上真机的两个旋钮怎么调）；**算得快**（一次看 → 想 → 动的时间账、推理放在有线局域网的 GPU 上、realtime-vla 用 CUDA Graph 和算子融合把 π0 一次推理从 106 毫秒压到 27 毫秒、bf16 + 编译、TensorRT 和低比特、少走几步去噪）；**送到电机**（位置目标 + 阻抗、插值升频、速度前馈、比示范跑得快、安全兜底），最后是上真机的排查顺序和开关清单。每一招讲出自哪篇、怎么做、开没开源。

## 这期的材料从哪来

这期讲部署时就能改的 19 招。下表是 deck 里用到的论文（arXiv 链接）、官方博客和开源代码，每页数字的出处在 `notes/技术事实核查.md`。deck 里的 6 段视频来自 Physical Intelligence 的 RTC 研究页和 VLASH 的官方 demo，仓库里不放，视频页换成了片段的一帧。

| 类型 | 材料 | 链接 |
|---|---|---|
| 钩子 · 第 2、20 页 | Dream Machines 博客：在一条真实产线上微调 π0.5（同一份权重只改两个部署参数，53% → 93%） | <https://dream-machines.eu/blog/pi05-fine-tuning> |
| 背景 · 第 4 页 | ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) |
| 背景 · 第 4 页 | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) |
| 背景 · 第 4 页 | What Makes a VLA Reactive? A Study of Execution Strategies for Dynamic Manipulation | [arXiv:2611.15491](https://arxiv.org/abs/2611.15491) |
| 背景 · 第 4 页 | π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities | [arXiv:2604.15483](https://arxiv.org/abs/2604.15483) |
| 异步推理 · 第 6–7 页 | SmolVLA（LeRobot 的异步推理） | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) |
| RTC · 第 8–10 页 | Real-Time Execution of Action Chunking Flow Policies（RTC） | [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) |
| RTC · 视频 | Physical Intelligence 的 RTC 研究页（第 2、6、8 页的视频来自这里） | <https://www.pi.website/research/real_time_chunking> |
| 训练时 RTC · 第 11–12 页 | Training-Time Action Conditioning for Efficient Real-Time Chunking | [arXiv:2512.05964](https://arxiv.org/abs/2512.05964) |
| VLASH · 第 14–16 页 | VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference | [arXiv:2512.01031](https://arxiv.org/abs/2512.01031) |
| VLASH · 代码 | mit-han-lab/vlash | <https://github.com/mit-han-lab/vlash> |
| VLASH · 视频 | VLASH 官方 demo 视频（第 14 页的乒乓球、打地鼠来自这里） | <https://www.youtube.com/watch?v=IgN7CNicJS8> |
| VLASH · 第 16 页 | Understanding Asynchronous Inference Methods for Vision-Language-Action Models | [arXiv:2605.08168](https://arxiv.org/abs/2605.08168) |
| A2C2 · 第 17 页 | Leave No Observation Behind: Real-time Correction for VLA Action Chunks（A2C2） | [arXiv:2509.23224](https://arxiv.org/abs/2509.23224) |
| FASTER · 第 18 页 | FASTER: Rethinking Real-Time Flow VLAs | [arXiv:2603.19199](https://arxiv.org/abs/2603.19199) |
| RTC 怎么调 · 第 20–21 页 | Alexander Soare：Smooth-As-Butter Robot Policies（RTC 上真机调参） | <https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html> |
| RTC 怎么调 · 第 21 页 | LeRobot《Unfolding Robotics》叠衣服实验 | <https://huggingface.co/spaces/lerobot/robot-folding> |
| 同一层的其他做法 · 第 22 页 | Bidirectional Decoding（BID） | [arXiv:2408.17355](https://arxiv.org/abs/2408.17355) |
| 同一层的其他做法 · 第 22 页 | Learning Native Continuation for Action Chunking Flow Policies | [arXiv:2602.12978](https://arxiv.org/abs/2602.12978) |
| 同一层的其他做法 · 第 22 页 | Barrier-Constrained Guidance For Real-time Asynchronous VLA Execution | [arXiv:2610.14997](https://arxiv.org/abs/2610.14997) |
| 同一层的其他做法 · 第 22 页 | DiscreteRTC | [arXiv:2604.25050](https://arxiv.org/abs/2604.25050) |
| 同一层的其他做法 · 第 22 页 | ACG: Action Coherence Guidance for Flow-based VLA Models | [arXiv:2510.22201](https://arxiv.org/abs/2510.22201) |
| 同一层的其他做法 · 第 22 页 | ABPolicy: Asynchronous B-Spline Flow Policy | [arXiv:2602.23901](https://arxiv.org/abs/2602.23901) |
| 同一层的其他做法 · 第 22 页 | REACT: Rolling Denoising and Dual Decoupling | [arXiv:2610.12007](https://arxiv.org/abs/2610.12007) |
| 同一层的其他做法 · 第 22 页 | FlashVLA: Streaming Action Decoding | [arXiv:2608.27384](https://arxiv.org/abs/2608.27384) |
| 同一层的其他做法 · 第 22 页 | RAVEL: Asynchronous Rolling Inference | [arXiv:2609.34170](https://arxiv.org/abs/2609.34170) |
| 同一层的其他做法 · 第 22 页 | PACT-VLA: Speculative Action-Chunk Verification | [arXiv:2610.28019](https://arxiv.org/abs/2610.28019) |
| 时间花在哪 · 第 25 页 | Fast-WAM: Do World Action Models Need Test-time Future Imagination? | [arXiv:2603.16666](https://arxiv.org/abs/2603.16666) |
| 时间花在哪 · 第 25 页 | World Action Models are Zero-shot Policies（DreamZero） | [arXiv:2602.15922](https://arxiv.org/abs/2602.15922) |
| 时间花在哪 · 第 25、41 页 | Realtime-VLA V2: Learning to Run VLAs Fast, Smooth, and Accurate | [arXiv:2603.26360](https://arxiv.org/abs/2603.26360) |
| 有线局域网 GPU · 第 26–27 页 | π0: A Vision-Language-Action Flow Model for General Robot Control | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
| 有线局域网 GPU · 第 26 页 | RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies | [arXiv:2506.18123](https://arxiv.org/abs/2506.18123) |
| 有线局域网 GPU · 第 27、31 页 | VLA-HWBench: Where the Time Goes When VLA Models Run on Edge Hardware | [arXiv:2610.34717](https://arxiv.org/abs/2610.34717) |
| realtime-vla · 第 28–29 页 | Running VLAs at Real-time Speed | [arXiv:2510.26742](https://arxiv.org/abs/2510.26742) |
| realtime-vla · 代码 | Dexmal/realtime-vla | <https://github.com/Dexmal/realtime-vla> |
| 机载 · 第 31 页 | OpenVLA: An Open-Source Vision-Language-Action Model | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |
| 少走几步去噪 · 第 32 页 | OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models | [arXiv:2502.19645](https://arxiv.org/abs/2502.19645) |
| 快慢双系统 · 第 33 页 | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) |
| 快慢双系统 · 第 33 页 | Gemini Robotics: Bringing AI into the Physical World | [arXiv:2503.20020](https://arxiv.org/abs/2503.20020) |
| 阻抗 · 第 35–36 页 | π0.5: a Vision-Language-Action Model with Open-World Generalization | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) |
| 阻抗 · 第 35–36 页 | Adaptive Compliance Policy | [arXiv:2410.09309](https://arxiv.org/abs/2410.09309) |
| 阻抗 · 第 35–36 页 | Enabling Dynamic Tracking in VLA Models via Time-Discrete and Time-Continuous Velocity | [arXiv:2603.16218](https://arxiv.org/abs/2603.16218) |
| 阻抗 · 代码 | CRISP controllers（ROS 2 的阻抗控制器） | <https://github.com/learnsyslab/crisp_controllers> |
| 插值升频 · 第 37–38 页 | LiPo: A Lightweight Post-optimization Framework for Smoothing Action Chunks | [arXiv:2506.05165](https://arxiv.org/abs/2506.05165) |
| 插值升频 · 第 38 页 | Ruckig: Jerk-limited Real-time Trajectory Generation with Arbitrary Target States | [arXiv:2105.04830](https://arxiv.org/abs/2105.04830) |
| 比示范跑得快 · 第 42 页 | DemoSpeedup: Accelerating Visuomotor Policies via Entropy-Guided Demonstration Acceleration | [arXiv:2506.05064](https://arxiv.org/abs/2506.05064) |
| 比示范跑得快 · 第 42 页 | SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies | [arXiv:2506.11948](https://arxiv.org/abs/2506.11948) |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：每页数字的出处（论文版本、章节、表号、图号、开源代码行号），每条标「已核 / 推算 / 判断 / 估读 / 示意」；标「示意」的数是为了一步步算给你看编的。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 6 段视频（第 2、6、8、14 页），仓库里不放视频文件，视频页换成了片段的一帧，原片在 Physical Intelligence 的 RTC 研究页和 VLASH 的官方 demo 视频，链接见上表。分步页（第 3、7、9、10、12、15、16、20、21、25、27、29、35、36、38、45 页）在这一页按 `G` 或 `J` 走一步、`K` 退一步。
