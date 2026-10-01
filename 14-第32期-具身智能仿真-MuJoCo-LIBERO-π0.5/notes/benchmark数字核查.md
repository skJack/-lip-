# 仿真 Benchmark 数字核查报告

- 核查日期：2026-09-28。下文所有网页、排行榜、GitHub 文件的访问日期都是这一天。
- 数字优先从 arXiv 的 LaTeX 源码（e-print）里取，再和 PDF 对一遍。各篇核的版本：
  - LIBERO：2306.03310v2
  - LIBERO-Plus：2510.13626v3（2025-12-26）
  - LIBERO-PRO：2510.03827，v1（2025-10-04）和 v2（2026-05-25，当前最新）都核了
  - RoboTwin 2.0：2506.18088v2（2025-08-27）
  - RoboCasa365：2603.04356v1
  - BEHAVIOR 冠军报告 2512.06951v2，亚军报告 2512.10071v3，BEHAVIOR-1K 2403.09227v1
  - RoboLab：2604.09860v4（2026-08-14）

## 结论速览

- **所有数字都对得上，没有 ✘。**
- **1 条未核实：** 8d。RoboLab 用“光线追踪渲染”这个说法，在它自己的论文、README、项目页和 NVIDIA 博客里都找不到。能确认的只有两点：它基于 Isaac Lab，README 要求 NVIDIA RTX GPU。“光追”是从 Isaac Sim 的 RTX 渲染器推出来的，不是 RoboLab 自己的表述。
- **需要补限定或改措辞的地方：**
  - 2d：LIBERO 论文原文的“四个套件”是 Spatial / Object / Goal / **LIBERO-100**。LIBERO-100 再拆成 LIBERO-90 + LIBERO-Long。“LIBERO-10”是代码里的叫法，即 LIBERO-Long。130 个任务这个数字没问题。
  - 2b：作者单位以 UT Austin 为主，另有 Sony AI 和清华大学。
  - 3f：LIBERO-Plus 没有评测 π0.5。论文、GitHub 排行榜、项目页都没有 π0.5 的数字。
  - 4：LIBERO-PRO v1 用的是表 2–5，v2 改成了图 7（v2 里没有表）。
    - v2 图 7 里 π0.5 的 Pos 值印成“38.0 / 20.0 / 0.08 / 0.17”，单位混用了，实际就是 0.38 / 0.20 / 0.08 / 0.17。
    - “Task 扰动下全是 0.00–0.01”只对 OpenVLA、π0、π0.5 这三个模型成立。v2 新加的 X-VLA 在 Task 扰动下达到 0.09 / 0.10。
  - 5：两组数字的设置不同，不能直接比。
    - 排行榜上的 π0.5 70.7/46.0 和 ME-Dex-1.0 是 **Co-train** 设置：一个策略在 50 个任务上联合训练。
    - 论文里的 π0 46.4/16.3 是 **Single** 设置：每个任务单独微调。
  - 6：RoboCasa365 的三处细节。
    - 论文 Table 1 的行名是 “Atomic”，排行榜才叫 “Atomic-Seen”。
    - Average 按 50 个目标任务加权（18 / 16 / 16），不是三列直接平均。
    - 排行榜上的 π0 和 GR00T N1.5 是用 v1.0.1 重测的，数字和论文 Table 1 不一样（见文末附注）。
  - 8k：RoboLab 排行榜的 42.9% 是 Default 指令档的成绩。换成 Vague 或 Specific 档，第一名会变。

## A. 逐条核查

| 声称 | 查到的 | 出处（表号/链接+访问日期） | 结论 |
|---|---|---|---|
| 1a openpi `pi05_libero`：LIBERO-Spatial 98.8 | 98.8 | openpi `examples/libero/README.md` 的 Results 表（行名 “π0.5 @ 30k (finetuned)”）。commit 215abfb（2026-08-24），与 github.com/Physical-Intelligence/openpi main 一致（2026-09-28） | ✔ |
| 1b Object 98.2 | 98.2 | 同上 | ✔ |
| 1c Goal 98.0 | 98.0 | 同上 | ✔ |
| 1d LIBERO-10 92.4 | 92.4 | 同上 | ✔ |
| 1e 平均 96.85 | 96.85 | 同上 | ✔ |
| 2a LIBERO：NeurIPS 2023 Datasets & Benchmarks | PDF 首页脚注原文：“37th Conference on Neural Information Processing Systems (NeurIPS 2023) Track on Datasets and Benchmarks.” | arXiv 2306.03310v2 第 1 页；LaTeX 用 `neurips_data_2023.sty [final]` | ✔ |
| 2b UT Austin | 作者单位：The University of Texas at Austin（主）、Sony AI、清华大学 | 同上，标题页 | ✔（可补充另两家单位） |
| 2c 共 130 个任务 | “four task suites (130 tasks in total)” | 摘要；§1；§4.2 | ✔ |
| 2d 四套件 Spatial/Object/Goal/LIBERO-10(Long) 各 10 个，外加 LIBERO-90 | 原文四套件：Spatial、Object、Goal 各 10 个，加上 LIBERO-100。LIBERO-100 拆成 LIBERO-90（90 个短程任务）和 LIBERO-Long（10 个长程任务）。代码里的套件名是 `libero_10` / `libero_90` | §4.2（“LIBERO has four task suites: … LIBERO-100”）；openpi `examples/libero/main.py` | ✔ 数字对；措辞上，论文的第 4 个套件是 LIBERO-100 |
| 2e 每任务 50 条人类演示 | “we provide 50 trajectories of high-quality demonstrations for every single task … collected by human experts through teleoperation with 3Dconnexion Spacemouse” | §4（BC 训练段） | ✔ |
| 2f 机器人是 Franka Panda | 论文正文没写机器人型号。代码 `env_wrapper.py` 默认 `robots=["Panda"]`；`robots/mounted_panda.py` 的注释写 “Panda is a sensitive single-arm robot designed by Franka” | github.com/Lifelong-Robot-Learning/LIBERO（master，2026-09-28） | ✔（出处是代码，不是论文） |
| 2g 基于 robosuite（MuJoCo） | “Our generation pipeline is built on top of Robosuite”；代码里 `import robosuite`、`import mujoco`、`renderer="mujoco"` | §4.1；`bddl_base_domain.py` | ✔ |
| 3a LIBERO-Plus：7 个扰动维度 | Objects Layout / Camera Viewpoints / Robot Initial States / Language / Light / Background Textures / Sensor Noise | 2510.13626v3 摘要、§2.1；README | ✔ |
| 3b 10,030 个任务 | “comprises 10,030 tasks spanning seven perturbation factors with twenty-one low-level components” | §6.1；Fig. 6 图注；附录 Table 7（Total 10030） | ✔ |
| 3c Table 1：OpenVLA-OFT 原始 97.1 / 相机 59.7 / 机器人初始状态 37.2 | 97.1 / 59.7 / 37.2 | Table 1（§2.3，单维扰动分析） | ✔ |
| 3d Table 1：π0 原始 94.2 / 相机 15.8 / 初始状态 6.6 | 94.2 / 15.8 / 6.6 | Table 1 | ✔ |
| 3e Table 2 Total：OpenVLA-OFT 69.6、π0-FAST 61.6、π0 53.6、OpenVLA 15.6 | 69.6 / 61.6 / 53.6 / 15.6 | Table 2（§6.2，在最终的 LIBERO-Plus 基准上评测）；GitHub README 排行榜数字相同 | ✔ |
| 3f π0.5 是否在 LIBERO-Plus 中评测过 | **没有。** 论文评了 10 个模型：OpenVLA、OpenVLA-OFT（含 _w、_m 两个变体）、π0、π0-fast、Nora、WorldVLA、UniVLA、RIPT-VLA。GitHub 排行榜只多了作者自己的 OpenVLA-OFT+，外加 3 个社区工作（AVA-VLA、MergeVLA、SRPO），也没有 π0.5 | 2510.13626v3 §2.2 与各表；github.com/sylvestf/LIBERO-plus README；sylvestf.github.io/LIBERO-plus（2026-09-28） | 无 π0.5 数字 |
| 4a LIBERO-PRO：Pos 扰动下 OpenVLA 四个套件都是 0.00 | 四个套件均为 0.00 | v1：Table 2（Goal）、3（Spatial）、4（LIBERO-10）、5（Object）的 Average 行；v2：Figure 7；github.com/Zxy-MLlab/LIBERO-PRO README 排行榜 | ✔ |
| 4b Pos 扰动下 π0 四个套件都是 0.00 | 四个套件均为 0.00 | 同上 | ✔ |
| 4c Pos 扰动下 π0.5：0.38 / 0.20 / 0.08 / 0.17（Goal / Spatial / LIBERO-10 / Object） | v1 表：0.38 / 0.20 / 0.08 / 0.17。v2 图 7 印成 “38.0 / 20.0 / 0.08 / 0.17”（单位混用）。README 排行榜：0.38 / 0.20 / 0.08 / 0.17 | 同上 | ✔ |
| 4d Task 扰动下三个模型都在 0.00–0.01 | OpenVLA 0.00×4；π0 0.00×4；π0.5 为 0.00 / 0.01 / 0.01 / 0.01 | 同上 | ✔（只限这三个模型。v2 新加的 X-VLA 为 0.09 / 0.00 / 0.10 / 0.08，MolmoAct 在 LIBERO-10 上为 0.06） |
| 4e 论文版本与表/图号 | v1（2025-10-04）用 4 张分套件表，即 Table 2–5，只评了 3 个模型。v2（2026-05-25，arXiv 注明 “0 tables”）把结果改成柱状图 Figure 7，新增 MolmoAct、NORA、X-VLA。 | arXiv abs 页的提交历史 | 引用时写“v1 表 2–5 / v2 图 7” |
| 5a RoboTwin 2.0：仿真器 SAPIEN | 代码 `envs/_base_task.py` 里有 `import sapien.core as sapien` 和 `self.engine = sapien.Engine()`。官方安装文档写 “Most constraints stem from what the SAPIEN package is capable of supporting” | github.com/RoboTwin-Platform/RoboTwin（main）；robotwin-platform.github.io/doc/usage/robotwin-install.html | ✔（论文正文没明写，出处是代码和文档） |
| 5b 50 个双臂任务 | “across 50 dual-arm tasks” | 2506.18088v2 摘要；Fig. 8 | ✔ |
| 5c 5 种本体 | “spanning five robot embodiments” | 摘要；Fig. 5 | ✔ |
| 5d 731 个物体（RoboTwin-OD，147 类） | “731 instances across 147 categories” | 摘要；Fig. 1、Fig. 7 | ✔ |
| 5e 10 万+ 轨迹 | “pre-collected over 100,000 dual-arm manipulation trajectories across 50 tasks” | §3.2；§1 贡献段 | ✔ |
| 5f 域随机化：杂物、背景、光照、桌高、语言 | “five axes: clutter, lighting, background, tabletop height and language instructions” | 摘要；§2.2 | ✔ |
| 5g 排行榜 π0.5 Easy 70.7 / Hard 46.0（RoboTwin 团队重测） | easy_mean 70.7、hard_mean 46.0，contributor 为 “RoboTwin Team”，上榜日期 26.08.10。设置是 Co-train，即一个策略在 50 个任务上联合训练 | robotwin-platform.github.io/leaderboard，数据文件 `data/robotwin_leaderboard.json`（updated 2026-09-24） | ✔ |
| 5h 第一名 ME-Dex-1.0 Easy 89.58 / Hard 68.12 | 89.58 / 68.12。按默认排序（(easy+hard)/2 = 78.85）第一，只按 Easy 或只按 Hard 排也都是第一。Co-train 设置，提交方 Li Auto Inc.，上榜日期 26.09.24 | 同上 | ✔ |
| 5i 论文单任务基线 π0 46.4 / 16.3（Easy/Hard） | 50 个任务平均：Easy 46.4 / Hard 16.3。设置：每任务用 50 条 clean 演示单独微调，Aloha-AgileX。排行榜 Single 档 π0 为 46.42 / 16.34 | Table 5（§4.5）、附录 Table 10；排行榜 | ✔ |
| 5j 会议 ICML 2026 | ICML 2026 poster，2026-07-08，Hall A #310。官方 README 也标了 ICML 2026 | icml.cc/virtual/2026/poster/62192 | ✔ |
| 6a RoboCasa365：ICLR 2026 | arXiv comments 写 “ICLR 2026”；LaTeX 用 `iclr2026_conference` + `\iclrfinalcopy` | arxiv.org/abs/2603.04356 | ✔ |
| 6b 365 个任务（65 atomic + 300 composite） | “365 everyday tasks: 65 atomic tasks and 300 composite tasks” | §3.3 | ✔ |
| 6c 2,500 个厨房 | 50 种布局 × 50 种风格 = 2,500 个预训练厨房，另有 10 个目标厨房 | §3.2；Fig. 1 | ✔ |
| 6d 612 小时人类演示 + 1,615 小时 MimicGen 数据 | 原文：“612 hours of human demonstration data and an additional 1615 hours of synthetic demonstration data using the MimicGen”。附录数据表：预训练人类数据 404 h + 目标任务人类数据 208 h = 612 h | §1；附录 dataset 表 | ✔ |
| 6e 基于 MuJoCo / robosuite | “built on top of RoboSuite, which uses the MuJoCo physics engine” | 附录 “Simulation Infrastructure” | ✔ |
| 6f Table 1：GR00T N1.5 平均 20.0 | 20.0（Atomic 43.0 / Composite-Seen 9.6 / Composite-Unseen 4.4） | Table 1 | ✔ |
| 6g π0.5 16.9 | 16.9（39.6 / 7.1 / 1.2） | Table 1 | ✔ |
| 6h π0 15.0 | 15.0（36.3 / 5.2 / 0.7） | Table 1 | ✔ |
| 6i Diffusion Policy 6.1 | 6.1（15.7 / 0.2 / 1.25） | Table 1 | ✔ |
| 6j 三组：Atomic-Seen / Composite-Seen / Composite-Unseen | 论文行名为 Atomic / Composite-Seen / Composite-Unseen，“Atomic-Seen”是排行榜 README 的叫法。Average 按任务数加权：Atomic 18、Composite-Seen 16、Composite-Unseen 16，数的是附录任务表 | Table 1；附录 task tables | ✔（名称有小差异） |
| 6k 排行榜 Composite-Unseen 最高 36.7（Paimon-0） | Paimon-0（Primotion，2026-09-11，闭源）：79.3 / 55.8 / 36.7；robocasa.ai 上 Overall 58.1，排第 1 | github.com/robocasa-benchmark/leaderboard 的 `submissions/Paimon-0_2026_09_11.json`；robocasa.ai/leaderboard.html（页面标 Updated 09/23/2026） | ✔ |
| 6l Xiaomi-Robotics-1 80.2 / 57.1 / 32.1 | 80.2 / 57.1 / 32.1（2026-07-08，开源），Overall 57.4，排第 2 | `submissions/Xiaomi-Robotics_2026_07_08.json`；同上 | ✔ |
| 7a BEHAVIOR Challenge 2025：50 个任务 | “50 full-length household tasks” | behavior.stanford.edu/challenge/archive/2025/index.html | ✔ |
| 7b OmniGibson，跑在 NVIDIA Isaac Sim 上 | 冠军报告：“The benchmark uses OmniGibson simulation built on NVIDIA Isaac Sim” | arXiv 2512.06951v2 §1.1；BEHAVIOR 文档导航里有 “Under the Hood - Isaac Sim” | ✔ |
| 7c 机器人 R1Pro | 2025 基线页的训练命令用 `robot=r1pro`；2026 评测页写 “the default R1Pro robot” | challenge/archive/2025/baselines.html；challenge/evaluation.html | ✔ |
| 7d 10,000 条遥操作演示 | “10,000 teleoperated demonstrations (1200+ hours)”；数据页统计 Total Trajectories 10,000 | archive/2025/index.html；archive/2025/dataset.html | ✔ |
| 7e 冠军 Robot Learning Collective，held-out Q-score 0.2599 | Held-out Test 0.2599（Public Validation 0.2605） | archive/2025/leaderboard.html（“Provisional 2025 Challenge Leaderboard”）；冠军报告 Table 1 | ✔ |
| 7f 冠军成功率 12.4% | Held-out 的 Full Task Success Rate 为 0.1240（Public 0.1120） | 同上 | ✔ |
| 7g 冠亚军都从 π0.5 微调 | 冠军报告：“Building on the Pi0.5 architecture”。亚军是 Comet（NVIDIA Research，held-out Q 0.2514），报告写 “Building on π0.5” | arXiv 2512.06951 摘要；arXiv 2512.10071 摘要（排行榜上亚军的 report 链接就是它） | ✔ |
| 7h BEHAVIOR Challenge 2026：100 个任务 | “100 full-length household tasks”；数据是 20,000 条演示、1,950 小时，7 个场景（其中 4 个新场景） | behavior.stanford.edu/challenge/index.html | ✔ |
| 7i 2026 基线：π0.5 与 GR00T N1.7 | “Baselines: π0.5 (pi0.5) and GR00T N1.7” | 同上；challenge/baselines.html | ✔ |
| 7j 2026 提交截止 2026-10-16 | “Submission Deadline: 10/16/2026”（获奖公布 11/04/2026） | 同上 | ✔ |
| 8a RoboLab：RSS 2026 | arXiv journal-ref：“Robotics: Science and Systems XXII, Sydney, Australia, 2026”；项目页 BibTeX 同为 RSS 2026 | arxiv.org/abs/2604.09860；research.nvidia.com/labs/srl/projects/robolab/ | ✔ |
| 8b NVIDIA | 作者单位：NVIDIA，另有 University of Toronto 和 The University of Sydney | 论文标题页 | ✔ |
| 8c 基于 Isaac Lab | 论文：“built on IsaacLab”。README：“built on NVIDIA Isaac Lab”，默认 IsaacSim 5.0 / IsaacLab 2.2.0 | §I 贡献段；github.com/NVLabs/RoboLab README | ✔ |
| 8d 光线追踪渲染 | RoboLab 的论文、README、项目页、NVIDIA 技术博客里都没有 ray tracing / ray-traced 的字样。README 只写 “NVIDIA RTX GPU required”。Isaac Sim 用的是 Omniverse RTX 渲染器，其文档写明不支持光追的 GPU 会被跳过 | 同上；docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html | 未核实（RoboLab 自己没这么说，只能间接推断） |
| 8e 120 个任务 | 基准名 RoboLab-120。附录能力表 Total 为 120（simple 64 + moderate 39 + complex 17）。正文 §IV-A 写 “65 simple, 38 moderate, 18 complex”，加起来是 121，这是论文内部的小矛盾 | §IV-A；附录 Table（categories details）；项目页 “120 tasks” | ✔ |
| 8f 不提供训练数据，评测用真机数据训练的策略（DROID 设置） | 项目页：“designed for policies trained on real-world data directly in simulation without co-training on simulation data”。论文：“All policies were fine-tuned on the DROID dataset”，仿真 “used only as a controlled evaluation environment”。评测用 DROID 配置：Franka Panda + Robotiq 2F-85 | 项目页；§IV-A；§II | ✔ |
| 8g TABLE I：π0.5 成功率 28.0% | 28.0%（score 0.43） | TABLE I | ✔ |
| 8h π0-FAST 15.5% | 15.5% | TABLE I | ✔ |
| 8i GR00T N1.6 7.2% | 7.2%。表中宏 `\groot` 的定义就是 GR00T N1.6 | TABLE I | ✔ |
| 8j π0 5.0% | 5.0%（另有 PaliGemma 3.4%） | TABLE I | ✔ |
| 8k 排行榜第一 42.9%（FLUX 3 Action） | FLUX 3 Action（WAM）：515/1200 = 42.9%，第二名 HiDream-O1-Embodied 39.9%。这是 Default 指令档。另外两档只统计有该档成绩的策略：Vague 档第一是 Atomic-WAM 31.6%，Specific 档第一是 Cosmos3-Nano-Policy 39.7% | research.nvidia.com/labs/srl/projects/robolab/leaderboard.html（2026-09-28） | ✔（需注明是 Default 指令档） |
| 9 Isaac Sim 5.x：不支持没有 RT Core 的 GPU（A100、H100） | 原文：“GPUs without RT Cores (A100, H100) are not supported.” | docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html（该页已被标为 “Unsupported release”）；最新 6.x 文档的同一页仍有这句 | ✔ |
| 10a MuJoCo 原始论文：Todorov, Erez, Tassa，IROS 2012 | “MuJoCo: A physics engine for model-based control”，E. Todorov, T. Erez, Y. Tassa，2012 IEEE/RSJ IROS，pp. 5026–5033，DOI 10.1109/IROS.2012.6386109 | Crossref（DOI 元数据） | ✔ |
| 10b DeepMind 2021 年 10 月收购并免费开放 | “acquired and made freely available by Google DeepMind in October 2021”；changelog 中 2.1.0 版日期为 2021-10-18 | mujoco.readthedocs.io/en/stable/overview.html；changelog | ✔ |
| 10c 2022 年 5 月开源（Apache 2.0） | “open sourced in May 2022”；changelog “Version 2.2.0 (May 23, 2022) … MuJoCo is now fully open-source software”；LICENSE 为 Apache License 2.0 | overview；changelog；github.com/google-deepmind/mujoco 的 LICENSE 与 README | ✔ |
| 10d 名称是 Multi-Joint dynamics with Contact | “MuJoCo stands for Multi-Joint dynamics with Contact” | github.com/google-deepmind/mujoco README | ✔ |
| 10e 默认软接触（允许少量穿透），solref 默认 timeconst 0.02、dampratio 1 | Computation 章 “Soft contact model”：放弃严格互补条件，接触时允许穿透。Modeling 章：solref: real(2), “0.02 1”，即 (timeconst, dampratio) | mujoco.readthedocs.io 的 computation 和 modeling 页 | ✔ |
| 10f 默认步长 0.002 s | XML reference：timestep: real, “0.002” | mujoco.readthedocs.io/en/stable/XMLreference.html | ✔ |
| 10g MuJoCo Menagerie 收录 Franka Panda | 有 “Franka Emika Panda”（`franka_emika_panda/`），另有 FR3 | github.com/google-deepmind/mujoco_menagerie README | ✔ |
| 11a LIBERO 示例 `replan_steps=5` | `replan_steps: int = 5`：每次推理出一个 action chunk，只执行前 5 步就重新推理 | `examples/libero/main.py` L29（与 GitHub main 一致） | ✔ |
| 11b `pi05_libero` 配置 `action_horizon=10` | `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)` | `src/openpi/training/config.py` 的 `pi05_libero`。GitHub main 上游 L744–745。模型默认 action_horizon 是 50（`pi0_config.py`），这里被覆盖成 10 | ✔ |

### 附注：顺带发现的小差异（不在待核声称内）

- **LIBERO-Plus：** GitHub 排行榜和论文 Table 2 有两处不一致。
  - UniVLA 的 Total，README 写 43.9，论文写 42.9。
  - 作者微调模型（OFT+）的 Total，README 写 79.6，论文表写 79.5，正文写 79.6。
  - 另外，Table 1 是单维扰动的前期分析，Table 2 是最终的 10,030 任务基准，两表的相机维数字不同（例如 OFT 59.7 vs 56.4），引用时别混在一起。
- **RoboCasa365：** 排行榜上的 π0 和 GR00T N1.5 是官方按 robocasa 1.0.1 重测的，和论文 Table 1 不同。

  | 模型 | 排行榜（Atomic / Comp-Seen / Comp-Unseen，Overall） | 论文 Table 1（同顺序，Average） |
  |---|---|---|
  | π0 | 34.6 / 6.1 / 1.1，Overall 14.8 | 36.3 / 5.2 / 0.7，15.0 |
  | GR00T N1.5 | 50.7 / 14.8 / 2.7，Overall 23.9 | 43.0 / 9.6 / 4.4，20.0 |

  π0.5 和 DP 两边一致。
- **RoboTwin 2.0：** 排行榜的 Easy 对应 demo_clean（clean2clean），Hard 对应 demo_randomized（clean2random）。训练数据是 50 条 clean 演示 × 50 个任务，每个任务评测 100 次。
- **BEHAVIOR 2025：** 冠军报告摘要里的 “26% q-score” 就是 0.2599 四舍五入。

## B. 配图

第 36 页的五张缩略图就是从下面这些原图裁出来的（LIBERO-PRO 那张最后没用上），原图没有放进仓库。

| 文件 | 来源 | 图号 | 原始文件名 | 尺寸 / 处理 |
|---|---|---|---|---|
| `fig_liberoplus.png` | LIBERO-Plus 官方 GitHub README 头图，和项目页主图是同一张（sylvestf.github.io/LIBERO-plus 的 `static/images/main_img.png`） | 论文里没有这张 teaser。论文 v3 的 Figure 1 是柱状图 “Robustness to object layout perturbations”，所以改用项目主图 | `static/images/libero-plus.jpg`（raw.githubusercontent.com/sylvestf/LIBERO-plus/main/…） | 3775×1534。JPG 转 PNG，裁白边 |
| `fig_liberopro.png` | LIBERO-PRO 论文 arXiv 2510.03827 源码。GitHub README 头图 `images/overall.png`（2392×1449）是同一张图 | v2 Figure 6 “Overview of LIBERO-PRO”（v1 为 Figure 3） | `figures/main_figure.pdf` | 4757×2885。300 dpi；这个 PDF 的 MediaBox 是 1701×1701 pt 的大画布，所以要加 `-cropbox` |
| `fig_robotwin2.png` | RoboTwin 2.0 论文 arXiv 2506.18088v2 源码 | Figure 1 “Overview of RoboTwin 2.0” | `figures/teaser.pdf` | 3171×1996。300 dpi，边距很小，没裁 |
| `fig_robocasa365.png` | RoboCasa365 论文 arXiv 2603.04356v1 源码 | Figure 1 “Overview of RoboCasa365” | `figures/pull/Figure1.pdf` | 8296×4421，文件约 16.6 MB。300 dpi，满版无白边；内嵌渲染图的有效分辨率约 250–520 ppi |
| `fig_behavior.png` | BEHAVIOR Challenge 2026 页面头图，是 teaser 视频（youtube.com/embed/ihihRCf5NI4）的封面帧：R1Pro 在酒吧柜台场景里的渲染画面，左上角叠了标题字 “2026 Announcing the 2nd BEHAVIOR Challenge” | 网页头图，没有图号 | `challenge_teaser_frame_240.png`（behavior.stanford.edu/assets/…） | 1920×1080，原样保存（满版）。备选是 BEHAVIOR-1K 论文 Figure 1（`figures/pull-new.pdf`），但它以图表和文字为主，只有两张小渲染图，所以没选 |
| `fig_robolab.png` | RoboLab 论文 arXiv 2604.09860v4 源码 | Fig. 1 “Overview of RoboLab” | `imgs/overview.png` | 2602×1252。RGBA 铺白底转 RGB，裁白边。上半部分是渲染场景，下半部分是分析示意图。项目页只有视频，没有静态头图 |
