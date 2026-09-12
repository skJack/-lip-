# Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan)

> 对应「可lip」VLA 入门第 21 期（一）视频。论文：[arXiv:2204.01691](https://arxiv.org/abs/2204.01691)。
> 这是我读论文时的整理笔记：数字都来自论文正文或图表，标了「估读」的是从没有数值的柱状图上估的；看图请对照 arXiv 上的原文。

- arXiv: 2204.01691（v2, 2022-08；CoRL 2022）· 机构: Robotics at Google / Everyday Robots · 代码: 只开源了一个桌面 Colab（say-can.github.io/#open-source：UR5 + CLIPort + ViLD + GPT-3），真机系统未开源 · 项目页: https://say-can.github.io
- 一句话: 用 LLM 给候选 skill 打"有用性"分、用 RL value function 打"可行性"分，二者相乘选下一步——第一个把 LLM 规划和机器人 affordance 接起来、在真机上跑长程指令的系统。

## 1. 要解决的问题
LLM 有丰富常识（"洒了饮料 → 拿海绵"），但没有身体、看不到场景，会给出机器人做不到的步骤（"用吸尘器"）。直接 prompt 让 LLM 分解任务也不够：输出可能不在 skill 库里，也不知道当前状态（手里已经有东西、面前没有苹果）。同期的 Huang et al. 2022（LLM as Zero-Shot Planners）只是生成计划再投影到最近的 skill，没有任何 grounding 机制——正好是本文的 Generative baseline。

## 2. 方法 / 系统设计
- (a) 模型：PaLM 540B，用 **scoring 模式**（对固定候选算 logprob）而不是生成；消融了 PaLM 8B/62B/540B 和 FLAN 137B。
- (b) 观测：LLM 本身**看不到图像**。场景信息只通过 affordance 标量间接进入；value function 的输入是 RGB 图像（Sec. 4）。
- (c) 动作接口：一组带自然语言描述 ℓ_π 的 skill（pick / go to / put down / open drawer…，从 551 个 skill × 17 个物体里挑成功率高的用）。LLM 对每个候选 ℓ_π 计算 p(ℓ_π | i, 历史)，不生成自由文本。
- (d) 回路：Algorithm 1——每步对所有 skill 算 p_combined = p(c_π | s, ℓ_π) · p(ℓ_π | i)，取 argmax 执行，把选中的 skill 追加进 prompt 再问，直到选中 "done"。**只在决策步之间闭环**，skill 执行过程中不监控。
- (e) memory：只有 prompt 里已执行的步骤序列，没有显式 memory 模块。
- (f) 失败检测/恢复：没有。Sec. 8 明确承认"skill 失败但 value 很高时系统无法反应"；后续 Inner Monologue [25] 补的就是这块。
- (g) 训练：LLM 纯 prompt（17 个 few-shot 例子，Listing 1）。skill policy 用 BC-Z 式 BC 训练；value function 用 MT-Opt 多任务 RL（仿真 + RetinaGAN sim-to-real），奖励是人工标注的稀疏 0/1。affordance 是归一化后的 value（Appendix D.2）：p_pick = clamp((v − 0.2)/(0.5 − 0.2), 0, 1)；go to 用距离；place 恒为 1.0；terminate 恒为 0.1。
- (h) 安全：靠 affordance 过滤不可行动作，没有额外安全层。

关键公式：p(c_i | i, s, ℓ_π) ∝ p(c_π | s, ℓ_π) · p(ℓ_π | i)。前者是 world-grounding（"我能做吗"），后者是 task-grounding（"这步有用吗"）。

## 3. 实验
- Setting：Everyday Robots 移动机械臂（7-DoF 臂 + 两指夹爪），真办公室厨房 + 复刻的 mock kitchen；15 个物体、5 个地点；101 条指令分 7 个家族（Table 1）。
- 指标：plan success 和 execution success，都由 3 个人评、2/3 同意算成功。
- 主结果（Table 2）：mock kitchen plan 84% / execute 74%；真厨房 81% / 60%。长程任务最差（plan 73% / exec 47%），主要是 LLM 过早输出 "done"。错误 65% 来自 LLM、35% 来自 affordance（Sec. 5.1）。
- 消融（Table 2）：去掉 value function（No VF）plan 掉到 67%；Generative（生成再投影，≈ Huang et al.）74%；不用 LLM、把指令直接喂 BC policy（BC NL）0%；BC USE 只在单原语上有 60%。
- LLM 规模（Table 3）：FLAN-SayCan plan 70% / exec 61%，PaLM-SayCan 84% / 74%——第一次展示"换更强的 LLM，机器人就变好"。
- prompt 例子数（Table 5，语言模拟器）：0 例只有 10%（要求输出 done）→ 4 例 82% → 17 例 88%。
- 加新 skill（Sec. 5.2）：加抽屉 skill 后 21 条指令 plan 100%，exec 只有 33%——瓶颈在低层 skill。

## 4. 局限
作者承认：继承 LLM 的偏差，否定句/歧义处理差（CoT 版可部分缓解）；瓶颈在 skill 库的范围和成功率；skill 执行失败无法反应。我读出来的：LLM 完全不看图，全部场景感知被压成一个标量 affordance，对"物体在哪、什么状态"一无所知；每步要对全部候选 skill 各跑一次 540B scoring，延迟和成本都高（论文未报延迟）；value function 要为每个 skill 用 RL 训并手工校准，扩展到开放词表很难。

## 5. 复现要点
- 真机系统（PaLM + Everyday Robots）不可复现。开源 Colab 只演示算法逻辑：UR5 桌面、CLIPort 做 pick-place、ViLD 做 affordance、GPT-3 做 LLM。
- 替代复现：任何能返回 logprob 的 LLM 都能做 scoring；用开源 VLM 输出"该 skill 是否可行"的 yes/no 概率替代 value function 是现在最常见的做法。
- 坑：scoring 对 skill 措辞极敏感（"a" vs "an"、拼写），Appendix D.3 专门说明。

## 6. 关键引用链
上游：Huang et al. 2022（LLM as Zero-Shot Planners）、MT-Opt、BC-Z、PaLM。下游：Inner Monologue（加环境反馈闭环）、Code as Policies（把 skill 序列换成代码）、PaLM-E / RT-2（把 LLM 和 policy 端到端合并）；本项目里的 Guava / Thea / FAEA 都是同一路线的现代版本。
