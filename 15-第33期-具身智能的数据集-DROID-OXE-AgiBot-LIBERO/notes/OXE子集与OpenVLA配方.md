# 第 33 期 · OXE：子数据集、本体与 OpenVLA 训练配方

核查日期：2026-10-01。**OXE是数据集合，RT-X是模型。** 这份材料补充 slides 第 21–24 页的代表子集表。

## 三个范围分别看

1. **论文版本**：OXE 论文 v9 正文与Figure 2记录60个数据集、22种本体、百万以上真实机器人轨迹。这是论文统计口径。
2. **当前官方目录**：本轮公开表有72个具名条目，其中69项给出Registered Dataset Name。它继续增补DROID、DobbE、FMB等，也包含仿真、人类动作和导航条目。具名条目不自动等于已验证可下载包。
3. **具体模型配方**：OpenVLA原论文选27项来源。它对本体、观测、动作作筛选，并用非均匀采样；目录全集不是某个模型的实际训练集。

官方目录：[Open X-Embodiment Dataset Overview](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit)。下表是 2026-10-01 的整理。论文：[Open X-Embodiment](https://arxiv.org/abs/2310.08864)；代码：[官方仓库](https://github.com/google-deepmind/open_x_embodiment)。

## 目录中的重要差异

- RT-1常以`fractal20220817_data`出现；QT-Opt常以`kuka`出现；TACO Play对应目录中的Freiburg Franka Play。显示名、论文名和代码注册名可能不同。
- `bridge`旧转换包与BridgeData V2全量/`bridge_orig`不是同一统计口径。目录的25,460不能拿来推翻论文60,096。
- 目录里的RT-1是73,499，DROID是92,233；都不直接替换原论文总采集量或成功轨迹量。任务过滤、成功/失败范围和版本需要另查。
- QT-Opt、Cable Routing等目录条目没有原生语言；其他条目有模板或自然语言。RLDS统一组织episode，并没有自动补齐所有模态和统一动作物理意义。
- 原始RT-X实验也只使用其中选定的操作数据，不能说所有60个来源都以同样方式进入RT-X。
- 当前表有元数据不一致处。例如Saytap的Robot列写Unitree A1，Description写Go1；下表保留Robot列原样，并标注待核对，不自行合并。部分场景、action分类同样只是目录的粗分类。

## 当前目录完整索引

以下顺序与本轮官方表一致；本体、动作和采集字段为官方目录原始标签。场景作轻量中文归纳，详细描述见官方目录。没有Registered Dataset Name的条目标记“未列”，不等于断言资源未公开。

| # | 数据集 | 本体（目录字段） | 采集形式 | 场景 | RLDS注册名 |
|---|---|---|---|---|---|
| 1 | RT-1 Robot Action | Google Robot | Human VR | 桌面, 厨房/玩具厨房 | fractal20220817_data |
| 2 | QT-Opt | Kuka iiwa | Expert Policy | 桌面 | kuka |
| 3 | Berkeley Bridge | WidowX | Human VR | 桌面, 厨房/玩具厨房, 其他家居 | bridge |
| 4 | Freiburg Franka Play | Franka | Human VR | 桌面 | taco_play |
| 5 | USC Jaco Play | Jaco 2 | Human VR | 桌面, 厨房/玩具厨房 | jaco_play |
| 6 | Berkeley Cable Routing | Franka | Human VR | 桌面 | berkeley_cable_routing |
| 7 | Roboturk | Sawyer | Human VR | 桌面 | roboturk |
| 8 | NYU VINN | Hello Stretch | Human Kinesthetic | 厨房/玩具厨房, 其他家居 | nyu_door_opening_surprising_effectiveness |
| 9 | Austin VIOLA | Franka | Human Spacemouse | 桌面 | viola |
| 10 | Berkeley Autolab UR5 | UR5 | Human Spacemouse | 桌面 | berkeley_autolab_ur5 |
| 11 | TOTO Benchmark | Franka | The dataset is collected in 3 ways: Human teleoperation -- VR Teleop, trained state-based BC policies, and trajectory replay with noise | 桌面 | toto |
| 12 | Language Table | xArm | Human VR | 桌面 | language_table |
| 13 | Columbia PushT Dataset | UR5 | Human VR | 桌面 | columbia_cairlab_pusht_real |
| 14 | Stanford Kuka Multimodal | Kuka iiwa | Expert Policy | 桌面 | stanford_kuka_multimodal_dataset_converted_externally_to_rlds |
| 15 | NYU ROT | xArm | Human Joystick | 桌面 | nyu_rot_dataset_converted_externally_to_rlds |
| 16 | Stanford HYDRA | Franka | Human VR | 桌面, 厨房/玩具厨房 | stanford_hydra_dataset_converted_externally_to_rlds |
| 17 | Austin BUDS | Franka | Human Spacemouse | 桌面 | austin_buds_dataset_converted_externally_to_rlds |
| 18 | NYU Franka Play | Franka | Human VR | 厨房/玩具厨房 | nyu_franka_play_dataset_converted_externally_to_rlds |
| 19 | Maniskill | Franka | Scripted | 桌面 | maniskill_dataset_converted_externally_to_rlds |
| 20 | Furniture Bench | Franka | Human VR | 桌面 | furniture_bench_dataset_converted_externally_to_rlds |
| 21 | CMU Franka Exploration | Franka | Expert Policy | 厨房/玩具厨房 | cmu_franka_exploration_dataset_converted_externally_to_rlds |
| 22 | UCSD Kitchen | xArm | Human VR | 厨房/玩具厨房 | ucsd_kitchen_dataset_converted_externally_to_rlds |
| 23 | UCSD Pick Place | xArm | Expert Policy | 桌面, 厨房/玩具厨房 | ucsd_pick_and_place_dataset_converted_externally_to_rlds |
| 24 | Austin Sailor | Franka | Human Spacemouse | 桌面, 厨房/玩具厨房 | austin_sailor_dataset_converted_externally_to_rlds |
| 25 | Austin Sirius | Franka | Human Spacemouse | 桌面 | austin_sirius_dataset_converted_externally_to_rlds |
| 26 | BC-Z | Google Robot | Human VR | 桌面 | bc_z |
| 27 | USC Cloth Sim | Franka | Scripted | 桌面, 厨房/玩具厨房 | usc_cloth_sim_converted_externally_to_rlds |
| 28 | Tokyo PR2 Fridge Opening | PR2 | Human VR | 厨房/玩具厨房 | utokyo_pr2_opening_fridge_converted_externally_to_rlds |
| 29 | Tokyo PR2 Tabletop Manipulation | PR2 | Human VR | 桌面 | utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds |
| 30 | Saytap（本体字段待核对） | Unitree A1 | Expert Policy | Indoor, on a flat floor | utokyo_saytap_converted_externally_to_rlds |
| 31 | UTokyo xArm PickPlace | xArm | Human Puppeteering | 桌面 | utokyo_xarm_pick_and_place_converted_externally_to_rlds |
| 32 | UTokyo xArm Bimanual | xArm Bimanual | Human Puppeteering | 桌面 | utokyo_xarm_bimanual_converted_externally_to_rlds |
| 33 | Robonet | Multi-Robot | Scripted | 桌面 | robo_net |
| 34 | Berkeley MVP Data | xArm | Human VR | 桌面, 厨房/玩具厨房 | berkeley_mvp_converted_externally_to_rlds |
| 35 | Berkeley RPT Data | Franka | Scripted | 桌面 | berkeley_rpt_converted_externally_to_rlds |
| 36 | KAIST Nonprehensile Objects | Franka | Expert Policy | 桌面 | kaist_nonprehensile_converted_externally_to_rlds |
| 37 | QUT Dynamic Grasping | Franka | Scripted | 桌面 | 未列 |
| 38 | Stanford MaskVIT Data | Sawyer | Scripted | 桌面 | stanford_mask_vit_converted_externally_to_rlds |
| 39 | LSMO Dataset | Cobotta | Expert Policy | 桌面 | tokyo_u_lsmo_converted_externally_to_rlds |
| 40 | DLR Sara Pour Dataset | DLR SARA | Expert Policy | 桌面, Household objects | dlr_sara_pour_converted_externally_to_rlds |
| 41 | DLR Sara Grid Clamp Dataset | DLR SARA | Expert Policy | 桌面, Workshop environment | dlr_sara_grid_clamp_converted_externally_to_rlds |
| 42 | DLR Wheelchair Shared Control | DLR EDAN | Human teleoperation using Shared Control Templates | 桌面, shelf  | dlr_edan_shared_control_converted_externally_to_rlds |
| 43 | ASU TableTop Manipulation | UR5 | Scripted | 桌面 | asu_table_top_converted_externally_to_rlds |
| 44 | Stanford Robocook | Franka | Scripted | 桌面, 厨房/玩具厨房 | stanford_robocook_converted_externally_to_rlds |
| 45 | ETH Agent Affordances | Franka | Expert Policy | 厨房/玩具厨房 | eth_agent_affordances |
| 46 | Imperial Wrist Cam | Sawyer | Human Kinesthetic | 桌面 | imperialcollege_sawyer_wrist_cam |
| 47 | CMU Franka Pick-Insert Data | Franka | Human VR | 桌面 | iamlab_cmu_pickup_insert_converted_externally_to_rlds |
| 48 | QUT Dexterous Manpulation | Franka | Human VR | 桌面 | qut_dexterous_manipulation |
| 49 | MPI Muscular Proprioception | PAMY2 | Scripted | The robot is alone in the environment, there are no other objects in the workspace. | 未列 |
| 50 | UIUC D3Field | Kinova Gen3 | Scripted | 桌面 | uiuc_d3field |
| 51 | Austin Mutex | Franka | Human Spacemouse | 桌面 | utaustin_mutex |
| 52 | Berkeley Fanuc Manipulation | Fanuc Mate | Human VR | 桌面 | berkeley_fanuc_manipulation |
| 53 | CMU Food Manipulation | Franka | Scripted | 桌面 | cmu_playing_with_food |
| 54 | CMU Play Fusion | Franka | Human VR | 桌面, 厨房/玩具厨房 | cmu_play_fusion |
| 55 | CMU Stretch | Hello Stretch | Expert Policy | 厨房/玩具厨房, 其他家居 | cmu_stretch |
| 56 | RECON | Jackal | Scripted | 室外 | berkeley_gnm_recon |
| 57 | CoryHall | RC Car | Expert Policy | 走廊 | berkeley_gnm_cory_hall |
| 58 | SACSoN | TurtleBot 2 | Expert Policy | 走廊 | berkeley_gnm_sac_son |
| 59 | RoboVQA | Google Robot | Human VR | 桌面, 厨房/玩具厨房, 其他家居, 走廊, anything within 3 entire office buildings | robot_vqa |
| 60 | ALOHA | ViperX Bimanual | Human Puppeteering | 桌面 | 未列 |
| 61 | DROID | Franka | Human VR | 桌面, 厨房/玩具厨房, 其他家居, 走廊 | droid |
| 62 | ConqHose | Spot | Scripted | 其他家居, 走廊 | conq_hose_manipulation |
| 63 | DobbE | Hello Stretch | Human collection using tools | 厨房/玩具厨房, 其他家居, 走廊 | dobbe |
| 64 | FMB | Franka | Human VR | 桌面 | fmb |
| 65 | IO-AI Office PicknPlace | Human | Directly collected on human body with mocap devices and aruco markers | 桌面 | io_ai_tech |
| 66 | MimicPlay | Franka | Human VR | 桌面 | mimic_play |
| 67 | MobileALOHA | MobileALOHA | Human Puppeteering | 桌面, 厨房/玩具厨房, 其他家居, 走廊 | aloha_mobile |
| 68 | RoboSet | Franka | Human VR | 桌面, 厨房/玩具厨房, 其他家居 | robo_set |
| 69 | TidyBot | TidyBot | Human writes preferred object placements in text form | 厨房/玩具厨房, 其他家居, living room, bedroom, kitchen, pantry room | tidybot |
| 70 | VIMA | UR5 | Scripted | 桌面 | vima_converted_externally_to_rlds |
| 71 | SPOC | Hello Stretch | Scripted | 厨房/玩具厨房, 其他家居, 走廊, multi room environments | spoc |
| 72 | Plex RoboSuite | Franka | Human Keyboard | 桌面, Tabletop with sections | plex_robosuite |

## OpenVLA原论文的27项配方

以下为论文附录Data Mixture Details的**训练采样权重**，不是轨迹占比、存储占比或使用率排名。数值按论文保留，存在四舍五入以及小于0.1%的条目。DROID在最后三分之一训练中移除，其权重重新分配给其余数据。

| 数据集 | 采样权重 |
|---|---:|
| Fractal | 12.7% |
| Kuka | 12.7% |
| Bridge | 13.3% |
| Taco Play | 3.0% |
| Jaco Play | 0.4% |
| Berkeley Cable Routing | 0.2% |
| Roboturk | 2.3% |
| Viola | 0.9% |
| Berkeley Autolab UR5 | 1.2% |
| Toto | 2.0% |
| Language Table | 4.4% |
| Stanford Hydra Dataset | 4.4% |
| Austin Buds Dataset | 0.2% |
| NYU Franka Play Dataset | 0.8% |
| Furniture Bench Dataset | 2.4% |
| UCSD Kitchen Dataset | <0.1% |
| Austin Sailor Dataset | 2.2% |
| Austin Sirius Dataset | 1.7% |
| DLR EDAN Shared Control | <0.1% |
| IAMLab CMU Pickup Insert | 0.9% |
| UTAustin Mutex | 2.2% |
| Berkeley Fanuc Manipulation | 0.7% |
| CMU Stretch | 0.2% |
| BC-Z | 7.5% |
| FMB Dataset | 7.1% |
| DobbE | 1.4% |
| DROID | 10.0% |

出处：[OpenVLA论文](https://arxiv.org/abs/2406.09246)，附录 Data Mixture Details（arXiv 源码 `06_appendix.tex` 的 `tab:data_mix`）；[官方mixture实现](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/mixtures.py)。论文配方与当前代码中的不同实验配置要分别理解。

## 视频和开放方式

OXE 的视频页（第 23 页）采用[官方项目页](https://robotics-transformer-x.github.io/)的teaser，属于项目概览，含RT-X模型执行，不代表每个镜头都是人工示教。数据目录与读取代码公开，各子数据的许可、访问条件和具体版本分别核对。
