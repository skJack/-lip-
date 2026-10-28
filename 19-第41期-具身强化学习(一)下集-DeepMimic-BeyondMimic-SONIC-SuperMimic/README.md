# 第 41 期 · 具身智能中的强化学习（一 · 下集）

**视频标题**：SuperMimic：一张卡练 3 小时，超过英伟达 128 张卡练 7 天（下）

第一期「仿真里练」的下集，主角是 UC Berkeley 2026 年 10 月的 **SuperMimic**（CoRL 2026 Oral）：一张 H100 训 3 个小时，跟踪动作的分数就超过了英伟达 SONIC 用 128 张卡练 7 天的公开模型，GPU 小时省了一千倍以上。前面先铺垫人形机器人怎么照着人学：什么是 Mimic（给它一段人的动作，让它学着做出来）、放进上集的 PPO 只改两样；**DeepMimic** 当范式详讲（人的数据从哪来、长什么样、四元数、状态看 13 块、动作管 12 个关节、奖励 = 每一帧跟参考动作像不像、从中间开始和摔了就结束两个技巧、七步流程）；**BeyondMimic** 和之前最大的不同（一套配方、不用逐个动作调，但还是一段动作一个模型）、架构（参考动作也喂给策略）、核心贡献（不是新算法：先把仿真建准 → 随机化只留 3 样 → 奖励 7 项）、奖励怎么设计、以躯干为锚；**GMT、TWIST、SONIC** 一个策略跟所有动作，以及用的时候参考动作从哪来（存好的动作文件、真人实时带、生成模型现编、大模型给指令）。然后把 SuperMimic 拆开讲 20 页：PPO 的问题（每段动作都在重新试）、三条现成的路为什么不行、发明思路（动作千差万别，身体的物理只有一套）、训练全景（采数据 → 学一个可求导的世界模型 → 在想象里往前想 32 步、总分直接求导）、和 PPO「靠试出来估方向」的对照、价值网络怎么训、为什么省 1000 倍、12.9 万段动作只失败 25 段、零样本上宇树 G1 真机、局限；段末两页是我自己的理解。最后是现状：运动控制里 RL 已经是工业标配，离「解决」还有多远。

## 这期的材料从哪来

这期从 2018 年的 DeepMimic 讲到 2026 年 10 月的 SuperMimic，每一篇讲它改了 Mimic 范式里的哪一块。下表是 deck 里用到的论文（arXiv 链接）、代码和官方页面，每页数字的出处在 `notes/技术事实核查.md`。deck 里的 6 段视频都是从 SuperMimic 项目页的官方片子里切的，仓库里不放，视频页换成了片段的一帧。

| 类型 | 材料 | 链接 |
|---|---|---|
| 本集主角 | SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model（UC Berkeley，CoRL 2026 Oral） | [arXiv:2610.28480](https://arxiv.org/abs/2610.28480) |
| 本集主角 · 视频 | SuperMimic 项目页（第 34、41、43、48、49 页的视频都来自这里的官方片子） | <https://supermimic.github.io/> |
| 范式 · 详讲 | DeepMimic: Example-Guided Deep RL of Physics-Based Character Skills（SIGGRAPH 2018） | [arXiv:1804.02717](https://arxiv.org/abs/1804.02717) |
| 范式 · 代码 | DeepMimic 开源代码（第 12–14 页的数据格式和角色照它讲） | <https://github.com/xbpeng/DeepMimic> |
| 一个动作 | BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion | [arXiv:2508.08241](https://arxiv.org/abs/2508.08241) |
| 一个动作 | BeyondMimic 项目页 | <https://beyondmimic.github.io/> |
| 一个动作 · 代码 | BeyondMimic 开源代码 whole_body_tracking | <https://github.com/HybridRobotics/whole_body_tracking> |
| 一个动作 · 对比 | ASAP: Aligning Simulation and Real-World Physics | [arXiv:2502.01143](https://arxiv.org/abs/2502.01143) |
| 一个动作 · 对比 | KungfuBot | [arXiv:2506.12851](https://arxiv.org/abs/2506.12851) |
| 所有动作 | GMT: General Motion Tracking for Humanoid Whole-Body Control | [arXiv:2506.14770](https://arxiv.org/abs/2506.14770) |
| 所有动作 | TWIST: Teleoperated Whole-Body Imitation System | [arXiv:2505.02833](https://arxiv.org/abs/2505.02833) |
| 所有动作 | SONIC: Supersizing Motion Tracking（NVIDIA，128 张卡 7 天） | [arXiv:2511.07820](https://arxiv.org/abs/2511.07820) |
| 所有动作 | GR00T N1（第 32 页接在 SONIC 上面的 VLA 是它的 N1.5 版） | [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) |
| 铺垫 | Dactyl：Learning Dexterous In-Hand Manipulation（第 7 页域随机化） | [arXiv:1808.00177](https://arxiv.org/abs/1808.00177) |
| 铺垫 | Eureka：GPT-4 写奖励（第 8 页） | [arXiv:2310.12931](https://arxiv.org/abs/2310.12931) |
| 铺垫 | Real-World Humanoid Locomotion with Reinforcement Learning（Radosavovic 2024，第 19 页七步流程里的老师和学生） | [arXiv:2303.03381](https://arxiv.org/abs/2303.03381) |
| 同一条路线 | ControlVAE（第 52 页） | [arXiv:2210.06063](https://arxiv.org/abs/2210.06063) |
| 同一条路线 | MoConVQ（第 52 页） | [arXiv:2310.10198](https://arxiv.org/abs/2310.10198) |
| 现状 | Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes（Tang 2024） | [arXiv:2408.03539](https://arxiv.org/abs/2408.03539) |
| 现状 | Figure：Helix 02（第 54 页） | <https://www.figure.ai/news/helix-02> |
| 现状 | Boston Dynamics：Can football teach a robot to move（第 54 页） | <https://bostondynamics.com/blog/can-football-teach-a-robot-to-move/> |
| 现状 | Agility：Training a whole-body control foundation model（第 54 页） | <https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model> |
| 现状 | 宇树 unitree_rl_lab（动作模仿默认用 BeyondMimic） | <https://github.com/unitreerobotics/unitree_rl_lab> |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：每页数字的出处（论文章节、表号、图号），DeepMimic 的数据格式照开源代码核过；中间几段是常被问到的问题（用的时候参考动作从哪来、BeyondMimic 是单模型还是多模型、核心贡献到底在哪、世界模型看不到参考动作怎么打分）；沿用原来那版下集的铺垫页，出处附在文件末尾。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 6 段 SuperMimic 官方片子的片段（第 34、41、43、48、49 页，第 43 页两段），仓库里不放视频文件，视频页换成了片段的一帧，原片在 https://supermimic.github.io/ 。分步页（第 11、19、39、42 页）在这一页按 `G` 或 `J` 走一步、`K` 退一步。
