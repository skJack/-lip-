# 第 50 期 · 真机部署方法总结（下）

**视频标题**：真机部署方法总结（下）

「真机部署方法总结」下集：开训之前和数据这一侧要定下来的事，13 招分四段。**动作怎么写**：相对动作（UMI）、分位数归一化、夹爪单独处理、滑窗平均（蚂蚁灵波首届具身智能大模型挑战赛冠军队的做法）。**模型看什么**：少喂本体状态（State-free：同一批数据不喂关节状态，换桌高成功率 0% → 85%）、腕部广角相机、部署时修图（BYOVLA）。**整理数据**：延迟匹配（UMI，57.5% → 87.5%）、去静止帧（openpi 的 DROID 配方）、什么样的示范算好（八条标准，每条一个对照数字）、只留好的示范和 CUPID（影响函数，44% → 84%）。**偏了怎么补**：HG-DAgger（XR-2 叠衣服 58% → 93%）、RaC、CR-DAgger。每招都标了出处（论文 · 机构 · 会议年份）和现在谁在用（标配 / 主流 / 少数在用 / 还在论文里）。

## 这期的材料从哪来

这期讲开训之前和数据侧的 13 招，外加一页「什么样的示范算好」。下表是 deck 里用到的论文（arXiv 链接）、项目页和开源代码，每页数字的出处在 `notes/技术事实核查.md`。deck 里的 7 段视频来自 State-free Policy、BYOVLA、UMI、RaC、CR-DAgger 的项目页和 DAgger 作者的演示视频，仓库里不放，视频页换成了片段的一帧。第 8 页「滑窗平均」的做法出自蚂蚁灵波首届具身智能大模型挑战赛冠军 TARS 队在直播里的分享，没有论文。

| 类型 | 材料 | 链接 |
|---|---|---|
| 开场 · 第 1 页 / 少喂状态 · 第 10–11 页 | State-free Policy: Do You Need Proprioceptive States in Visuomotor Policies?（上海交大、千寻智能等） | [arXiv:2509.18644](https://arxiv.org/abs/2509.18644) |
| 少喂状态 · 视频 | State-free Policy 项目页（第 10 页换桌高的视频来自这里） | <https://statefreepolicy.github.io/> |
| 背景 · 第 2 页 | π0.5: a Vision-Language-Action Model with Open-World Generalization | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) |
| 相对动作 · 第 4–5 页 / 延迟匹配 · 第 16–17 页 | UMI: Universal Manipulation Interface（斯坦福宋舒然组，RSS 2024） | [arXiv:2402.10329](https://arxiv.org/abs/2402.10329) |
| UMI · 视频 | UMI 项目页（第 16 页扔东西的两段视频来自这里） | <https://umi-gripper.github.io/> |
| UMI · 代码 | real-stanford/universal_manipulation_interface（含三个延迟标定脚本） | <https://github.com/real-stanford/universal_manipulation_interface> |
| 多页 · 代码 | openpi（Physical Intelligence 开源的 π0 / π0.5 代码：相对动作、分位数归一化、DROID 去静止帧） | <https://github.com/Physical-Intelligence/openpi> |
| 分位数归一化 · 第 6 页 | OpenVLA: An Open-Source Vision-Language-Action Model（斯坦福等，CoRL 2024） | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |
| 夹爪单独处理 · 第 7 页 | Octo: An Open-Source Generalist Robot Policy | [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) |
| 夹爪单独处理 · 第 7 页 | HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL（伯克利） | [arXiv:2410.21845](https://arxiv.org/abs/2410.21845) |
| 滑窗平均 · 第 8 页 | ACG: Action Coherence Guidance for Flow-based VLA Models（KAIST） | [arXiv:2510.22201](https://arxiv.org/abs/2510.22201) |
| 少喂状态 · 第 10 页 | Fighting Copycat Agents in Behavioral Cloning from Observation Histories | [arXiv:2010.14876](https://arxiv.org/abs/2010.14876) |
| 腕部相机 · 第 12 页 | Vision-Based Manipulators Need to Also See from Their Hands（斯坦福，ICLR 2022） | [arXiv:2203.12677](https://arxiv.org/abs/2203.12677) |
| 部署时修图 · 第 13–14 页 | BYOVLA: Run-time Observation Interventions Make VLA Models More Visually Robust（普林斯顿，ICRA 2025） | [arXiv:2410.01971](https://arxiv.org/abs/2410.01971) |
| 部署时修图 · 视频与代码 | BYOVLA 项目页（第 13 页的视频来自这里） | <https://aasherh.github.io/byovla/> |
| 部署时修图 · 代码 | irom-lab/byovla | <https://github.com/irom-lab/byovla> |
| 延迟匹配 · 第 17 页 | Realtime-VLA V2（另一种测相机延迟的办法） | [arXiv:2603.26360](https://arxiv.org/abs/2603.26360) |
| 去静止帧 · 第 18–19 页 | DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset | [arXiv:2403.12945](https://arxiv.org/abs/2403.12945) |
| 去静止帧 · 第 19 页 | AgiBot World Colosseo（智元） | [arXiv:2503.06669](https://arxiv.org/abs/2503.06669) |
| 好示范 · 第 20 页 | What Matters in Learning from Offline Human Demonstrations for Robot Manipulation（robomimic） | [arXiv:2108.03298](https://arxiv.org/abs/2108.03298) |
| 好示范 · 第 20 页 | Hidden Variables Behind Real-Robot Data Quality（智元技术报告，2026-11） | [arXiv:2611.22388](https://arxiv.org/abs/2611.22388) |
| 好示范 · 第 20 页 | Learning from the Best: Smoothness-Driven Metrics for Data Quality in IL（RINSE） | [arXiv:2604.23000](https://arxiv.org/abs/2604.23000) |
| 好示范 · 第 20 页 | Is Diversity All You Need for Scalable Robotic Manipulation?（GO-1-Pro） | [arXiv:2507.06219](https://arxiv.org/abs/2507.06219) |
| 好示范 · 第 20 页 | Data Scaling Laws in Imitation Learning for Robotic Manipulation | [arXiv:2410.18647](https://arxiv.org/abs/2410.18647) |
| 好示范 · 第 20 页 | ALOHA Unleashed: A Simple Recipe for Robot Dexterity | [arXiv:2410.13126](https://arxiv.org/abs/2410.13126) |
| 只留好的示范 · 第 21 页 | CUPID: Curating Data your Robot Loves with Influence Functions（斯坦福、英伟达，CoRL 2025） | [arXiv:2506.19121](https://arxiv.org/abs/2506.19121) |
| 只留好的示范 · 项目页 | CUPID 项目页 | <https://cupid-curation.github.io> |
| 只留好的示范 · 代码 | agiachris/cupid | <https://github.com/agiachris/cupid> |
| HG-DAgger · 第 23 页 | DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning | [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) |
| HG-DAgger · 视频 | DAgger 作者的马里奥对比视频（第 23 页的视频来自这里） | <https://www.youtube.com/watch?v=anOI0xZ3kGM> |
| HG-DAgger · 第 23–24 页 | HG-DAgger: Interactive Imitation Learning with Human Experts | [arXiv:1810.02890](https://arxiv.org/abs/1810.02890) |
| HG-DAgger · 第 24 页 | XR-2: Scaling Bimanual Household Manipulation from 1,500 hours of Demonstrations to On-Policy Corrections | [arXiv:2609.03591](https://arxiv.org/abs/2609.03591) |
| HG-DAgger · 谁在用 | SOP: A Scalable Online Post-Training System for VLA Models（智元） | [arXiv:2601.03044](https://arxiv.org/abs/2601.03044) |
| HG-DAgger · 谁在用 | Meituan-Robotics-0（美团） | [arXiv:2611.08577](https://arxiv.org/abs/2611.08577) |
| HG-DAgger · 谁在用 | π*0.6: a VLA That Learns From Experience | [arXiv:2511.14759](https://arxiv.org/abs/2511.14759) |
| RaC · 第 25 页 | RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction（CMU） | [arXiv:2509.07953](https://arxiv.org/abs/2509.07953) |
| RaC · 视频 | RaC 项目页（第 25 页的视频来自这里） | <https://rac-scaling-robot.github.io/> |
| RaC · 谁在用 | Genie Centurion（智元，先倒带再纠正） | [arXiv:2505.18793](https://arxiv.org/abs/2505.18793) |
| CR-DAgger · 第 26 页 | Compliant Residual DAgger（斯坦福宋舒然组，NeurIPS 2025） | [arXiv:2506.16685](https://arxiv.org/abs/2506.16685) |
| CR-DAgger · 视频 | CR-DAgger 项目页（第 26 页翻书的视频来自这里） | <https://compliant-residual-dagger.github.io/> |
| CR-DAgger · 代码 | yifan-hou/cr-dagger | <https://github.com/yifan-hou/cr-dagger> |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：每页数字的出处（论文版本、章节、表号、图号、开源代码行号），每条标「已核 / 推算 / 判断 / 估读 / 示意」；每招「谁在用」的等级按哪些公开资料判断的，也写在里面。标「示意」的数是为了一步步算给你看编的。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 7 段视频（第 10、13、16、23、25、26 页），仓库里不放视频文件，视频页换成了片段的一帧，原片链接见上表。分步页（第 2、5、6、7、8、11、12、14、17、19、20、21、24、25、26 页）在这一页按 `G` 或 `J` 走一步、`K` 退一步。
