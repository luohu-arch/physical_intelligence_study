# Physical AI / VLA Reading Plan

目标：用 6 周建立 VLA 论文阅读框架，先理解机器人操作和模仿学习基础，再读通端到端 VLA、开源生态、连续动作生成和动作 tokenization。

## 使用方式

- 主线材料：`vla.md`
- 论文目录：`papers/`
- 论文矩阵：`tables/paper_matrix.csv`
- 精读笔记：`notes/`
- 建议节奏：每周 2-3 篇论文，每篇按“问题 -> 方法 -> 实验 -> 局限 -> 复现价值”阅读。

## 第 0 周：准备与基础框架

阅读：
- `vla.md` 第 1-2 章
- `notes/diffusion-policy.md` 的“预备知识”部分

必须掌握：
- 坐标系：世界坐标系、基坐标系、工具坐标系、工件坐标系
- 动作空间：关节空间、笛卡尔空间、轨迹空间
- 模仿学习：Behavior Cloning、Distribution Shift、Compounding Errors
- 机器人控制约束：实时性、延迟、安全、动力学限制

输出：
- 写一页 `notes/foundations.md`，解释“为什么 VLA 不是普通视觉语言模型”。

## 第 1 周：动作 token 化与 VLA 起点

阅读顺序：
1. RT-1
2. RT-2
3. FAST Tokenizer

核心问题：
- 为什么 RT-1 要把连续动作离散化？
- RT-2 如何把动作 token 融进 VLM 词表？
- FAST 为什么重新设计 action tokenizer？

输出：
- 对比 RT-1、RT-2、FAST 的动作表示方式。
- 回答：离散动作 token 的优点和上限分别是什么？

## 第 2 周：语义推理与空间接地

阅读顺序：
1. PaLM-E
2. VoxPoser
3. RT-Trajectory

核心问题：
- PaLM-E 如何把机器人状态接入语言模型？
- VoxPoser 为什么不直接输出动作，而是生成 3D value map？
- RT-Trajectory 如何用轨迹草图帮助空间泛化？

输出：
- 画一张“语言 -> 视觉/空间 -> 动作”的系统分层图。
- 总结 System 2 规划和 System 1 控制的区别。

## 第 3 周：数据规模化与跨机型泛化

阅读顺序：
1. RoboCat
2. Open X-Embodiment / RT-X
3. OpenVLA

核心问题：
- 机器人领域为什么数据比模型更难？
- 跨机型数据如何统一动作空间和观测空间？
- OpenVLA 为什么能成为开源 VLA 基线？

输出：
- 做一张“数据集 -> 模型 -> 机器人平台 -> 任务”的映射表。
- 回答：跨 embodiment 泛化为什么困难？

## 第 4 周：动作分块与连续动作生成

阅读顺序：
1. Mobile ALOHA / ACT
2. Diffusion Policy
3. Octo

核心问题：
- ACT 如何缓解长程任务的 compounding errors？
- Diffusion Policy 为什么比 MSE 回归更适合多峰动作分布？
- Octo 如何把多机器人、多相机、连续控制做成通用策略？

输出：
- 对比 ACT、Diffusion Policy、Octo 的动作生成头。
- 回答：什么时候该用 chunking，什么时候该用 diffusion？

## 第 5 周：物理精度与生成模型基础

阅读顺序：
1. Flow Matching
2. FAST Tokenizer 复读
3. 回看 Pi-Zero 相关章节

核心问题：
- Flow Matching 与 Diffusion 的训练/采样差异是什么？
- 连续动作生成为什么需要更快的推理？
- Pi-Zero 类模型为什么关注高频、平滑、低延迟？

输出：
- 写一页“Diffusion vs Flow Matching for robot action generation”。

## 第 6 周：综合复盘与项目化

复盘顺序：
1. RT-1 -> RT-2 -> OpenVLA：端到端 VLA 主线
2. ACT -> Diffusion Policy -> Octo：连续动作策略主线
3. Open X-Embodiment -> RoboCat：数据飞轮主线
4. VoxPoser -> RT-Trajectory：空间接地主线

输出：
- 更新 `tables/paper_matrix.csv` 的“my_status”和“my_notes”列。
- 为每篇论文补充 3 个问题：
  - 我真正理解了什么？
  - 我还不理解什么？
  - 这篇论文对工程实践有什么价值？

## 推荐阅读顺序总表

| 顺序 | 论文 | 原因 |
|---:|---|---|
| 1 | RT-1 | VLA 动作 token 化起点 |
| 2 | PaLM-E | 具身多模态语言模型基础 |
| 3 | RT-2 | 正式打通 VLM 与动作输出 |
| 4 | VoxPoser | 语言模型到 3D 空间约束 |
| 5 | RT-Trajectory | 轨迹提示与空间泛化 |
| 6 | Mobile ALOHA / ACT | 动作分块和低成本遥操作 |
| 7 | Diffusion Policy | 连续动作生成核心基线 |
| 8 | Open X-Embodiment | 跨机型大规模数据 |
| 9 | RoboCat | 自我改进与数据飞轮 |
| 10 | Octo | 开源通用机器人策略 |
| 11 | OpenVLA | 开源端到端 VLA 基线 |
| 12 | Flow Matching | 理解 Pi-Zero 类连续生成基础 |
| 13 | FAST Tokenizer | 新一代高效动作 tokenization |

