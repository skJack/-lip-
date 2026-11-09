# 第 41 期 · 具身智能中的强化学习（三）

**视频标题**：上真机：VLA 在真的机器人上，怎么用强化学习接着练

「具身智能中的强化学习」第三期「上真机」：VLA 怎么直接在真的机器人上用强化学习接着练。开场先放三段真机上练出来的结果（π*0.6 连续做 13 小时咖啡、GR-RL 穿鞋带 45.7% → 83.3%、RL Token 拧 M3 螺丝最难那段 20% → 65%），讲为什么非得上真机（仿真里刷到九十几分，搬到真机差一截；鞋带、网线、咖啡粉这些仿真搭不像）、从 QT-Opt 的 800 机器人小时到 RL Token 的 15 分钟、两个难点（真机贵、模型大）和各篇的答法，再按「大模型动多少」把这期分成三种。补课讲 off-policy：数据为什么能反复用、V 和 Q 差在哪（一个局面三个动作的算例）、Q 怎么学、TD 一轮一轮往回推、off-policy actor-critic 七步一轮、改策略的三种办法（PPO 改概率 / RECAP 挑好的学 / actor-critic 顺着 Q 挪），前传 HIL-SERL（人在旁边帮，1–2 小时练到 100%）。**RECAP（π*0.6）**：改整个模型但不在线改，价值函数从录像里数还剩几步，把「好 / 坏」写成一句话喂给 VLA 照监督学，整个训练流程和一条样本怎么练都分步讲。**GR-RL**：先用 critic 把人的示范里的失误删掉、左右镜像再重新模仿（45.7% → 72.7%），再冻住大模型，真机上只练一个挑起点噪声的小网络（→ 83.3%）。**RL Token**：π0.6 一个参数不改，把 VLM 最后一层压成一个 RL token，旁边挂一个小 actor-critic，只练最难的那几秒；经验池里每条记录存什么、人在一局里干的三件事（切换、接管、判成败），最难那段最多快 3 倍。最后三篇放一起比，带过车队一起练、蒸馏回通用模型、同类小头、奖励模型，再说四件还在争的事。

## 这期的材料从哪来

这期讲 VLA 怎么直接在真机上用强化学习练：主讲 RECAP（π*0.6）、GR-RL、RL Token 三篇，前传是 QT-Opt 和 HIL-SERL，收尾带过车队一起练、蒸馏回通用模型、同类小头和奖励模型。下表是 deck 里用到的论文（arXiv 链接）和官方页面，每页数字的出处在 `notes/技术事实核查.md`。deck 里的 14 段视频是从 Physical Intelligence、字节跳动 Seed 和 HIL-SERL 官方页面上的视频里切的，仓库里不放，视频页换成了片段的一帧。

| 类型 | 材料 | 链接 |
|---|---|---|
| 前传 | QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation（Google 2018，第 4、9 页） | [arXiv:1806.10293](https://arxiv.org/abs/1806.10293) |
| 前传 | HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning（第 14 页） | [arXiv:2410.21845](https://arxiv.org/abs/2410.21845) |
| 前传 · 视频 | HIL-SERL 项目页（第 14 页鞭子抽叠叠乐的视频来自这里） | <https://hil-serl.github.io/> |
| 前传 | RLPD: Efficient Online Reinforcement Learning with Offline Data（HIL-SERL 示范池、在线池各抽一半） | [arXiv:2302.02948](https://arxiv.org/abs/2302.02948) |
| 补课 | TD3: Addressing Function Approximation Error in Actor-Critic Methods（目标网络、两个 Q 取小；RL Token 的 critic 照它训） | [arXiv:1802.09477](https://arxiv.org/abs/1802.09477) |
| 重点 ① | π*0.6: a VLA That Learns From Experience（RECAP，Physical Intelligence） | [arXiv:2511.14759](https://arxiv.org/abs/2511.14759) |
| 重点 ① · 视频 | π*0.6 官方博客（第 2、18、23 页的视频来自这里） | <https://www.pi.website/blog/pistar06> |
| 重点 ② | GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation（字节跳动 Seed） | [arXiv:2512.01801](https://arxiv.org/abs/2512.01801) |
| 重点 ② · 视频 | GR-RL 项目页（第 2、25 页的视频来自这里，第 32 页两段在它的中文博客） | <https://seed.bytedance.com/en/gr_rl> |
| 重点 ② · 基座 | GR-3 Technical Report | [arXiv:2507.15493](https://arxiv.org/abs/2507.15493) |
| 重点 ② · 思路来源 | DSRL: Steering Your Diffusion Policy with Latent Space Reinforcement Learning（第二期范式三） | [arXiv:2506.15799](https://arxiv.org/abs/2506.15799) |
| 重点 ③ | RL Token: Bootstrapping Online RL with Vision-Language-Action Models（Physical Intelligence） | [arXiv:2604.23073](https://arxiv.org/abs/2604.23073) |
| 重点 ③ · 视频 | RL Token 研究页（第 2、36、40、41 页的视频来自这里） | <https://www.pi.website/research/rlt> |
| 收尾 · 车队 | SOP: A Scalable Online Post-Training System for Vision-Language-Action Models（智元） | [arXiv:2601.03044](https://arxiv.org/abs/2601.03044) |
| 收尾 · 车队 | Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies（第 4、44 页 16 台、8 个任务） | [arXiv:2605.00416](https://arxiv.org/abs/2605.00416) |
| 收尾 · 车队 | Learning While Deploying 项目页（第 44 页的图） | <https://learning-while-deploying.github.io/> |
| 收尾 · 蒸馏回去 | RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning | [arXiv:2412.09858](https://arxiv.org/abs/2412.09858) |
| 收尾 · 蒸馏回去 / 同类小头 | PLD: Self-Improving Vision-Language-Action Models with Data Generation via Residual RL | [arXiv:2511.00091](https://arxiv.org/abs/2511.00091) |
| 收尾 · 蒸馏回去 | π0.7（VLA 入门第 24 期讲过；第 44 页的图来自它的图 6） | [arXiv:2604.15483](https://arxiv.org/abs/2604.15483) |
| 收尾 · 同类小头 | EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models | [arXiv:2605.25477](https://arxiv.org/abs/2605.25477) |
| 收尾 · 奖励 | Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons | [arXiv:2603.02115](https://arxiv.org/abs/2603.02115) |
| 收尾 · 还在争 | 斯坦福 CS224R 2026「RL for Robot Foundation Models」课件（第 6、45 页） | <https://cs224r.stanford.edu/slides/17_cs224r_rl_vlas_2026.pdf> |
| 收尾 · 还在争 | FIND: Find Something You Can't Do: Agentic Real-World Reinforcement Learning for Self-Improving VLA Models | [arXiv:2609.32069](https://arxiv.org/abs/2609.32069) |
| 收尾 · 还在争 | ENPIRE: Agentic Robot Policy Self-Improvement in the Real World | [arXiv:2606.19980](https://arxiv.org/abs/2606.19980) |
| 路线图 · 第四期 | WMPO | [arXiv:2511.09515](https://arxiv.org/abs/2511.09515) |
| 路线图 · 第四期 | RISE | [arXiv:2602.11075](https://arxiv.org/abs/2602.11075) |
| 路线图 · 第四期 | WoVR | [arXiv:2602.13977](https://arxiv.org/abs/2602.13977) |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：每页数字的出处（论文版本、章节、表号、图号、LaTeX 源码行号），每条标「已核 / 推算 / 判断 / 估读」；三篇各自开头先列了版本和几处对不上的地方（比如 π*0.6 连续做咖啡，论文写 13 小时、博客说一整天；RL Token 的「4 个亚毫米级任务」其实按任务分毫米、亚毫米；GR-RL 在线约 150 局是项目页的说法）。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 14 段视频（第 2 页三段，第 14、18、23、25 页，第 32 页两段，第 36 页三段，第 40、41 页），仓库里不放视频文件，视频页换成了片段的一帧，原片在 π*0.6 博客、GR-RL 项目页和中文博客、RL Token 研究页、HIL-SERL 项目页，链接见上表。分步页（第 10、11、12、13、19、21、22、28、29、30、31、35、36、37、39、40 页）在这一页按 `G` 或 `J` 走一步、`K` 退一步。
