# 第 33 期 · 具身智能基础（二）

**视频标题**：具身智能数据集，一次讲清楚：从 DROID、OXE 到 AgiBot、LIBERO

训练一个机器人，到底需要什么样的数据？第 2–10 页先把基础概念讲清楚：本体、观测、状态、动作、episode、遥操作示教和 play data 分别是什么，一次操作怎样记录成一条数据，数据集、评测基准、成功率这些词在实验里怎么读。然后按三类过一遍常用数据：**真实机器人数据**（BridgeData V2、DROID、OXE 和 OpenVLA 实际用的 27 项配方、ALOHA / UMI 两种采集方式、AgiBot World、RoboMIND、RH20T，再加 2026 年新出的 MolmoAct2 双臂 YAM、AgiBot World 2026、Unitree 全身 WBT、HABIT、ABC-130K，每个两页、一段官方视频）；**仿真示教与评测基准**（robomimic、LIBERO、CALVIN、MimicGen、RoboCasa365、RoboTwin 2.0、InternData-A1 等）；**人类操作与导航资源**（Ego4D、EgoDex、EgoScale、R2R、HM3D）。每项尽量说清本体、采集方式、数据类型、规模、场景任务、发表信息和开放下载情况，最后一页讲怎样按学习目标选数据。**这期讲到的 30 个数据集的论文、项目页、下载入口和许可，全部列在下面的表里。**

> deck 封面和每页左上角写的「第 31 期」是我本地的编号；仓库按主线期数算第 33 期（上一期「具身智能仿真」是第 32 期）。

## 这期讲到的数据集：链接一览

每一行：论文（arXiv 链接，括号里是正式发表去向）、项目页 / 数据下载 / 代码、官方标注的开放情况。页码是 `slides/deck.html` 的页码。
开放情况是 2026-10-01 核到的官方说明：**代码、数据、权重的许可分别看**，带申请的入口是否批准以提供方为准，许可名只是官方标注，不代替具体使用场景的判断。
每项的本体、数据形式、规模和场景见 `notes/数据集基本情况速查.md`；谁在用、榜上到什么程度见 `notes/使用现状与开放状态.md`。

### 真实机器人数据与采集系统（第 11–37 页）

| 资源 | 页 | 论文 | 项目 · 数据 · 代码 | 开放情况（官方标注） |
|---|---|---|---|---|
| **BridgeData V2** | 11–14 | [arXiv:2308.12952](https://arxiv.org/abs/2308.12952)（CoRL 2023） | [项目与数据](https://rail-berkeley.github.io/bridgedata/) · [训练代码](https://github.com/rail-berkeley/bridge_data_v2) | 数据、代码、权重公开；数据 CC BY 4.0 |
| **DROID** | 15–19 | [arXiv:2403.12945](https://arxiv.org/abs/2403.12945)（RSS 2024） | [项目](https://droid-dataset.github.io/) · [数据说明（RLDS / 原始包 / 100 条调试包）](https://droid-dataset.github.io/droid/the-droid-dataset) · [采集代码](https://github.com/droid-dataset/droid) · [openpi 里的 DROID 策略](https://github.com/Physical-Intelligence/openpi) | 数据和采集代码公开；数据许可本轮未独立确认 |
| **Open X-Embodiment（OXE）** | 20–25 | [arXiv:2310.08864](https://arxiv.org/abs/2310.08864)（ICRA 2024） | [项目](https://robotics-transformer-x.github.io/) · [代码与数据读取](https://github.com/google-deepmind/open_x_embodiment) · [官方数据目录（72 项）](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit) | 索引和读取代码公开；各子集的许可和下载条件分别看（见 `notes/OXE子集与OpenVLA配方.md`） |
| **OpenVLA 的 OXE 配方 / Octo** | 24 | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) | [OpenVLA 代码](https://github.com/openvla/openvla) · [数据配置 configs.py](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/configs.py) · [采样配方 mixtures.py](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/mixtures.py) · [Octo（约 800k 条混合）](https://github.com/octo-models/octo) | 两个用 OXE 子集训练的模型，不是数据集 |
| **ALOHA / ACT** | 26 | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)（RSS 2023） | [项目](https://tonyzhaozh.github.io/aloha/) · [ACT 代码](https://github.com/tonyzhaozh/act) | 硬件资料和代码公开；真机示教按任务和具体发布包查 |
| **UMI** | 26–28 | [arXiv:2402.10329](https://arxiv.org/abs/2402.10329)（RSS 2024） | [项目](https://umi-gripper.github.io/) · [代码与杯子实验数据](https://github.com/real-stanford/universal_manipulation_interface) | 处理 / 训练 / 部署代码公开（MIT），数据按发布说明 |
| **FastUMI** | 26–28 | [arXiv:2409.19499](https://arxiv.org/abs/2409.19499) | [数据仓库](https://github.com/zxzm-zak/FastUMI_Data) · [HF 数据](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI-Data) | 采集代码、数据和转换工具公开；各版本分开看 |
| **AgiBot World（Colosseo）** | 29–32 | [arXiv:2503.06669](https://arxiv.org/abs/2503.06669)（IROS 2025） | [代码 / 数据 / 模型入口](https://github.com/OpenDriveLab/AgiBot-World) · [HF Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) · [HF Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | Alpha / Beta 和 GO-1 公开；CC BY-NC-SA 4.0 |
| **RoboMIND** | 33–35 | [arXiv:2412.13877](https://arxiv.org/abs/2412.13877)（RSS 2025） | [项目](https://x-humanoid-robomind.github.io/) · [数据卡](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) · [训练工具链](https://github.com/x-humanoid-robomind/x-humanoid-training-toolchain/) | HF gated（登录接受条件）；数据卡标 Apache 2.0 |
| **RH20T** | 36 | [arXiv:2307.00595](https://arxiv.org/abs/2307.00595)（ICRA 2024） | [官方数据与许可](https://rh20t.github.io/) · [数据 API](https://github.com/rh20t/rh20t_api) | 数据和 API 公开；RH20T-C CC BY-SA 4.0，RH20T-NC CC BY-NC 4.0 |
| **Galaxea Open-World** | 37 | [arXiv:2509.00576](https://arxiv.org/abs/2509.00576) | [数据卡](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) · [G0 项目](https://opengalaxea.github.io/G0/) | HF gated；数据 CC BY-NC-SA 4.0 |
| **RoboCOIN** | 37 | [arXiv:2511.17441](https://arxiv.org/abs/2511.17441) | [官方数据组织](https://huggingface.co/RoboCOIN) · [申请与 DataManager](https://github.com/FlagOpen/RoboCOIN-DataManager) · [CoRobot](https://github.com/FlagOpen/CoRobot) | 数据需填表申请；工具和 LeRobot 接口公开 |

### 2026 年的新真机数据（第 38–47 页，每项两页、一段官方视频）

| 资源 | 页 | 论文 | 项目 · 数据 · 代码 | 开放情况（官方标注） |
|---|---|---|---|---|
| **MolmoAct2-BimanualYAM** | 38–39 | [arXiv:2605.02881](https://arxiv.org/abs/2605.02881) | [数据卡](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) · [代码](https://github.com/allenai/molmoact2) · [官方博客](https://allenai.org/blog/molmoact2) | Apache 2.0，非 gated |
| **AgiBot World 2026** | 40–41 | 官方数据发布，无独立论文 | [数据卡](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) · [官网](https://agibot-world.com/) · [发布公告](https://www.agibot.com.cn/article/315/detail/148.html) | CC BY-NC-SA 4.0，非 gated |
| **Unitree UnifoLM-WBT** | 42–43 | 官方数据发布，无独立论文 | [数据集合](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset) · [本期用的任务包（洗碗机）](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher) · [官方视频](https://www.youtube.com/watch?v=pN_bj5-QyW8) | 所选任务包 Apache 2.0，非 gated |
| **HABIT** | 44–45 | [arXiv:2606.31682](https://arxiv.org/abs/2606.31682) | [数据卡](https://huggingface.co/datasets/configinc/HABIT) · [项目](https://habit-dataset.github.io/) | CC BY 4.0，非 gated |
| **ABC-130K** | 46–47 | [arXiv:2606.27375](https://arxiv.org/abs/2606.27375) | [数据卡](https://huggingface.co/datasets/XDOF/ABC-130k) · [项目](https://abc.bot/) · [代码（含 prepare.py）](https://github.com/amazon-far/abc) · [真机示例包 bottles_in_bin](https://abc-data.timehorizons.org/dataset/dataset_preview/bottles_in_bin_real.tar) | Apache 2.0；全量 gated，需登录接受条件 |

### 仿真示教、生成系统与评测基准（第 48–69 页）

| 资源 | 页 | 论文 | 项目 · 数据 · 代码 | 开放情况（官方标注） |
|---|---|---|---|---|
| **robomimic** | 48–50 | [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)（CoRL 2021） | [代码](https://github.com/ARISE-Initiative/robomimic) · [历史数据说明（PH / MH / MG）](https://robomimic.github.io/docs/v0.4/datasets/robomimic_v0.1.html) | 代码、历史数据和部分预训练资源公开 |
| **LIBERO** | 51–54 | [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)（NeurIPS 2023 D&B） | [代码与示教](https://github.com/Lifelong-Robot-Learning/LIBERO) · [OpenVLA 的四套评测](https://github.com/openvla/openvla) · [openpi 的 LIBERO 权重](https://github.com/Physical-Intelligence/openpi) | 环境 MIT；示教 CC BY 4.0 |
| **CALVIN** | 55–57 | [arXiv:2112.03227](https://arxiv.org/abs/2112.03227)（RA-L 2022） | [项目](http://calvin.cs.uni-freiburg.de/) · [代码与数据](https://github.com/mees/calvin) | 数据、环境、评测公开 |
| **MimicGen** | 58–59 | [arXiv:2310.17596](https://arxiv.org/abs/2310.17596)（CoRL 2023） | [代码](https://github.com/NVlabs/mimicgen) · [CoRL 2023 数据包](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html) | 生成系统和数据包公开 |
| **RoboCasa365** | 60–63 | [arXiv:2603.04356](https://arxiv.org/abs/2603.04356)（ICLR 2026） | [项目](https://robocasa.ai/) · [代码](https://github.com/robocasa/robocasa) · [排行榜](https://robocasa.ai/leaderboard.html) | 代码 MIT；资产与数据 CC BY 4.0 |
| **RoboTwin 2.0** | 64–67 | [arXiv:2506.18088](https://arxiv.org/abs/2506.18088)（ICML 2026，依官方标注） | [项目](https://robotwin-platform.github.io/) · [代码](https://github.com/RoboTwin-Platform/RoboTwin) · [预采集数据](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) · [排行榜](https://robotwin-platform.github.io/leaderboard) | 环境、生成器、评测和预采集数据公开 |
| **InternData-A1** | 68 | [arXiv:2511.16651](https://arxiv.org/abs/2511.16651)（CVPR 2026） | [项目](https://internrobotics.github.io/interndata-a1.github.io/) · [数据卡](https://huggingface.co/datasets/InternRobotics/InternData-A1) · [InternVLA-A1 模型卡](https://huggingface.co/InternRobotics/InternVLA-A1-3B) | HF gated；统一许可和完整生成流水线本轮未核实 |
| **SynGrasp-1B / GraspVLA** | 69 | [arXiv:2505.03233](https://arxiv.org/abs/2505.03233)（CoRL 2025） | [代码](https://github.com/PKU-EPIC/GraspVLA) · [数据](https://huggingface.co/datasets/vegebirrd/SynGrasp-1B) | 代码、权重、数据入口公开；代码仓库有 CC BY-NC 4.0 提示，数据许可字段未明确 |
| **MolmoBot-Data** | 69 | [arXiv:2603.16861](https://arxiv.org/abs/2603.16861) | [数据卡](https://huggingface.co/datasets/allenai/MolmoBot-Data) · [模型与训练](https://github.com/allenai/MolmoBot) · [MolmoSpaces 生成工具](https://github.com/allenai/molmospaces) | 数据 ODC-By；代码、生成工具、模型公开 |

### 人类视频、动作与导航（第 70–75 页）

| 资源 | 页 | 论文 | 项目 · 数据 · 代码 | 开放情况（官方标注） |
|---|---|---|---|---|
| **Ego4D** | 70 | [arXiv:2110.07058](https://arxiv.org/abs/2110.07058)（CVPR 2022） | [下载与协议](https://ego4d-data.org/docs/start-here/) · [各基准](https://ego4d-data.org/docs/benchmarks/overview/) | 工具公开；视频和标注需接受协议、取得凭证 |
| **EgoDex** | 71–73 | [arXiv:2505.11709](https://arxiv.org/abs/2505.11709)（ICLR 2026） | [数据、许可与后续使用列表](https://github.com/apple-aiml-research/ml-egodex) | 数据 CC BY-NC-ND；脚本公开 |
| **EgoScale** | 74 | [arXiv:2602.16710](https://arxiv.org/abs/2602.16710) | [NVIDIA 项目页](https://research.nvidia.com/labs/gear/egoscale/) | 论文和项目页公开；全量语料未见下载入口 |
| **R2R（Room-to-Room）** | 75 | [arXiv:1711.07280](https://arxiv.org/abs/1711.07280)（CVPR 2018） | [模拟器与数据](https://github.com/peteanderson80/Matterport3DSimulator) | 任务和代码公开；Matterport3D 场景另需协议 |
| **HM3D** | 75 | [arXiv:2109.08238](https://arxiv.org/abs/2109.08238)（NeurIPS 2021 D&B） | [数据入口](https://aihabitat.org/datasets/hm3d/) · [仓库与说明](https://github.com/facebookresearch/habitat-matterport3d-dataset) | 学术非商业访问；模拟器许可不代替场景条款 |

### 数据格式（第 10 页）

| 资源 | 页 | 论文 | 项目 · 数据 · 代码 | 开放情况（官方标注） |
|---|---|---|---|---|
| **RLDS** | 10 | — | [官方说明](https://github.com/google-research/rlds) | OXE、DROID、Bridge 等用的 episode / step 组织方式 |
| **LeRobotDataset v3** | 10 | — | [官方文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) | MolmoAct2、AgiBot 2026、Unitree、HABIT、InternData-A1 等用的格式 |

## 笔记

这期没有单篇论文的调研笔记，`notes/` 里是六份整理：

- [`notes/数据集基本情况速查.md`](notes/数据集基本情况速查.md)：25 个资源一张表：本体 / 采集对象、记录的内容与容器格式、场景、开放情况
- [`notes/使用现状与开放状态.md`](notes/使用现状与开放状态.md)：每个资源现在谁在用、榜上到什么程度、代码 / 数据 / 权重分别开放到哪一步、许可和访问条件，每条带第一手来源链接；最后是视频里引用的那几个成绩必须一起说的条件（2026-10-01 核）
- [`notes/2026新真机数据核对.md`](notes/2026新真机数据核对.md)：MolmoAct2-BimanualYAM、AgiBot World 2026、Unitree WBT、HABIT、ABC-130K 五项的来源、规模口径、许可、下载实测到哪一层，和八个容易说错的地方
- [`notes/OXE子集与OpenVLA配方.md`](notes/OXE子集与OpenVLA配方.md)：OXE 官方目录 72 项的完整索引（本体、采集形式、场景、RLDS 注册名）和 OpenVLA 论文的 27 项采样权重
- [`notes/发表信息与选材依据.md`](notes/发表信息与选材依据.md)：每个资源的首次公开年份、正式发表去向、代表机构和证据链接；为什么详讲这些、容易误报的版本差异
- [`notes/技术事实核查.md`](notes/技术事实核查.md)：视频里每个数字的原文定位和限定条件（论文页码、图表号、实验设置），核查时用的论文版本，论文插图的出处表。表里的页码是 62 页版的，后来基础概念部分重做成 77 页，现行页码见 `slides/使用说明.md`

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 13 段视频（第 12、16、23、27、34、39、41、43、45、47、61、65 页，第 34 页两段），全是各数据集官方项目页 / 数据卡上的原视频，`deck.html` 在线加载，**看那几页需要联网**；每段的封面帧放在 `slides/media/video-posters/`，离线也能看到第一帧。每段视频的来源、类型和时长见 `slides/使用说明.md` 末尾的表。
