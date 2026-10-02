# GRITS: Spillage-Aware Guided Diffusion Policy for Robot Food Scooping

- 本地 PDF：`papers/rl/dexterous/GRITS_2510.00573.pdf`
- arXiv：https://arxiv.org/abs/2510.00573
- 年份：2025 (ICRA 2026 Best Paper Finalist on Robot Learning)
- 团队：NYCU + XYZ Robotics + NVIDIA
- 阶段：可微分引导扩散策略 — 溅洒预测器做 diffusion guidance

## 一句话总结

GRITS 提出溅洒感知的引导扩散策略：先训练 spillage predictor（4K 仿真轨迹 + 4 种 primitive shapes 生成、随机物理参数），再在 diffusion denoising 时用其可微分输出做 guidance（ρ=2.5、延迟 30 步后激活），把轨迹在去噪后期"推离"溅洒区域。仅 80 条真机 demo、6 类食物训练，10 类 unseen 食物测试。82% 成功率、4% 溅洒率——比无引导的 Diffusion Policy (70%/15%) 溅洒率降低 73% 以上，比带后处理的 DP (52%/8%) 成功率提高 30pp。

## 九问速览

1. **Problem**：机器人舀取食物时同时要"舀得起来"与"不洒出来"——成功但溅洒也不可接受。
2. **Bottleneck**：80 条成功 demo 里没有"洒"的知识；后处理修正空间小且引入新问题（52%/8%）。
3. **Insight**：失败经验可从仿真廉价获得——可微分的溅洒概率能在去噪过程中实时引导轨迹生成。
4. **Method**：Isaac Lab 4K 轨迹训 spillage predictor；denoising 后 30 步注入梯度 ρ=2.5 把轨迹推离溅洒区。
5. **Evidence**：10 类 unseen 食物 82% 成功/4% 溅洒（无引导 DP 70%/15%；后处理 DP 52%/8%；BC 45%/45%）。
6. **Ablation**：guidance 双指标同升（溅洒 15%→4%、相对降 73%+）；BC 证明安全无法从成功 demo 学到。
7. **Assumption**：失败模式可由 4 种 primitive shapes+随机物理参数的仿真覆盖；点云分割可靠。
8. **Failure**：粘稠物被误判零概率时退化为无引导 DP；遮挡/分割粘连污染 predictor 输入。
9. **Opportunity**：多 guidance 冲突仲裁、动态任务的 (delay, ρ) 调度、其他失败模式（碰撞/倾覆）接入。

| 维度 | 论文答案 |
|---|---|
| Perception | 分割点云（food 深度+SAM2 分割、spoon/bowl 用 CAD 模型）经 DP3 编码；无触觉 |
| Closed-loop | 闭环：扩散策略滚动重规划（控制器 10 Hz） |
| Correction | 无重试机制——guidance 在生成期预防失败（事前规避而非事后恢复） |
| Deployment | 仿真训 predictor+80 条真机 demo 训策略 → Franka 真机直接部署（predictor 在线推理） |

## 核心技术

![grits 架构图](figures/grits/fig2.png)

*论文 Figure 2（p3）：Fig. 2: The architecture of GRITS. GRITS is a guided diffusion policy designed for robotic food scoo*

1. **Spillage Predictor** — 在 Isaac Lab 中用 4K 轨迹训练（球/立方/锥/圆柱 4 种 primitive shapes、随机物理参数），从点云预测溅洒概率 $p_{spill}$；训练数据全部仿真生成，与策略的真机 demo 数据解耦
2. **Guided Diffusion** — predictor 输出可微分 guidance 信号，在 denoising 后期（30 步之后）引导轨迹远离溅洒区域；guidance 强度 ρ=2.5
3. **Segmented Point Cloud Input** — food（深度图 + SAM2 分割）+ spoon（CAD）+ bowl（CAD），DP3-style PointNet++ 编码，输入模态是点云而非像素

## 底层原理与数学推导

```mermaid
%%{init: {
  'theme':'base',
  'themeVariables':{
    'primaryColor':'#fafbfd','primaryBorderColor':'#4a5d7d','primaryTextColor':'#1f2937',
    'fontFamily':'"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif',
    'fontSize':'14px','clusterBkg':'#fbfcfe','clusterBorder':'#b9c6d8','edgeLabelBackground':'#ffffff'
  },
  'flowchart':{'curve':'basis','nodeSpacing':26,'rankSpacing':42,'padding':10}
}}%%
flowchart LR
    PC(["分割点云<br/>food: 深度 + SAM2 分割<br/>spoon / bowl: CAD 模型<br/>(DP3 式 PointNet++ 编码)"]) ==> DP["扩散策略<br/>80 条真机 demo 训练<br/>(只有成功经验)"]
    SIM[("4K 仿真轨迹 (Isaac Lab)<br/>4 种 primitive shapes + 随机物理参数<br/>2000 洒 / 2000 不洒")] ==> PRED["spillage predictor<br/>点云 -> 溅洒概率 p_spill<br/>(廉价获得失败经验)"]
    DP ==> DEN["DDIM 去噪过程"]
    PRED ==> GUIDE["可微分溅洒引导 (核心)<br/>去噪 30 步后延迟激活:<br/>x <- x - rho * grad log(1 - p_spill)<br/>rho = 2.5, 轨迹推离溅洒区"]
    GUIDE -.->|"梯度注入去噪后期<br/>(约束参与生成, 而非事后补救)"| DEN
    DEN ==> ACTN(["安全舀取动作 (10 Hz)"])
    ACTN ==> ROBOT["Franka 真机<br/>10 类 unseen 食物 82% 成功 / 4% 溅洒"]

    class PC data
    class SIM data
    class DP,PRED train
    class DEN,ACTN act
    class GUIDE key
    class ROBOT env
    classDef data fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef train fill:#fff3e0,stroke:#ef6c00,stroke-width:2.5px,color:#e65100
    classDef act fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.5px,color:#4a148c
    classDef key fill:#fff8e1,stroke:#ff8f00,stroke-width:3px,color:#e65100
    classDef env fill:#e0f2f1,stroke:#00695c,color:#004d40
```

扩散策略的 denoising 过程从噪声动作 $x_T$ 出发、迭代 $T$ 步还原出动作 $x_0$。标准 DDPM 更新为

$$x_{t-1} = \mu_\theta(x_t, c, t) + \sigma_t \cdot \epsilon$$

GRITS 的关键修改是在更新中加入 predictor 的梯度项：

$$x_{t-1} = \mu_\theta(x_t, c, t) + \sigma_t \cdot \epsilon - \rho \cdot \nabla_{x_t} \log(1 - p_{spill}(x_t, c))$$

其中 $p_{spill}(x_t, c)$ 是溅洒预测器对当前去噪中间动作 $x_t$ 的溅洒概率，$\rho = 2.5$ 是引导强度。梯度项把轨迹推向"预测溅洒概率低"的区域——这是 classifier guidance 的精确类比：预测器扮演 classifier，去噪过程扮演生成器。延迟激活（前 30 步不施加 guidance）的物理含义是：去噪早期轨迹还远未成型，过早引导会把轨迹压向 predictor 的 bias；后期轨迹接近最终动作，此时引导才精准有效。由于 predictor 是标准网络，梯度 $\nabla_{x_t} \log(1 - p_{spill})$ 可直接反向传播，整个引导过程完全可微分、零额外推理成本。

## 物理直觉解释

**"往杯子里倒啤酒"式的最后收手**。倒啤酒时，前半程可以大胆倒——液面离杯口还远，怎么倒都不会溢出；只有接近杯口时，酒保才开始小心翼翼地收流量，因为此时"流速"直接决定"是否溢出"。GRITS 的延迟 guidance 就是这个直觉的机械实现：去噪前 30 步（相当于液面还低的时候）不加引导，让策略自由发挥；30 步之后（相当于接近杯口），溅洒 predictor 的梯度开始把轨迹推向"不洒"的方向。ρ=2.5 就是"收手的灵敏度"——太大动作僵硬、太小收不住。这是"约束只在临界区生效"的物理直觉在轨迹空间的重现。

**溅洒预测器是"洒过无数次的人"**。策略只见过 80 条成功 demo——它知道怎么舀，但不知道"什么动作会洒"。而 predictor 在仿真里用 4000 条随机轨迹见过无数种"洒法"：球形的会滚、立方的会卡、锥形的会倾、圆柱的会滑，摩擦变了洒的角度也变。这就像**让一个从没打翻过汤的人掌勺 vs 让一个打翻过一千次汤的人掌勺**——后者不是更会舀，而是对"哪种姿势会洒"有条件反射。predictor 的"失败经验"（仿真数据）和策略的"成功经验"（真机 demo）互补：成功经验告诉它目标在哪，失败经验告诉它别往哪走。

**为什么"引导"比"后处理"强？** 后处理（DP + post-processing）是策略输出动作后再检查修正——相当于**先开过去再倒车**：动作已经成型，修正空间有限，52% 成功率/8% 溅洒率说明修正本身引入了新问题。而 diffusion guidance 是在动作**生成过程中**就避开溅洒区域——相当于**导航时提前绕开堵车路段**，而不是到了路口再掉头。去噪过程每走一步都知道"前方洒不洒"，轨迹自然落在安全区域。这个差别是 82% vs 52% 的根源：约束参与生成 vs 约束事后补救。

## 工程细节与实操指南

- 仿真：Isaac Lab，4K 轨迹、4 种 primitive shapes（球/立方/锥/圆柱）、随机物理参数训练 predictor
- Guidance: ρ=2.5, 去噪前 30 步不激活（delayed activation）
- 输入: 分割点云 — food (深度 + SAM2), spoon (CAD), bowl (CAD), DP3-style PointNet++
- 数据: 80 条真机 demo, 6 类食物训练（brown rice, soybeans, chocolate balls, dates 等）
- 测试: 10 类 unseen 食物（sago, red beans, marshmallows, gummies, macaroni, mixed nuts, milk tea 等）
- 指标: 成功率 + 溅洒率双指标（成功但溅洒被单独计为溅洒）

## 实验协议清单

| 项目 | 论文设置 | 来源与备注 |
|---|---|---|
| 观测 | 分割点云：food 由深度图+SAM2 重建、spoon/bowl 用已知 CAD（前向运动学/标定位姿对齐），各下采样至 3000 点 | Sec III-C |
| 动作空间 | 连续 EE 轨迹（DDIM 去噪生成） | Sec III-C |
| 控制频率 | 机器人控制器 10 Hz | Sec IV-A |
| 重规划频率 | 每次去噪输出一条轨迹；guidance 前 30 个去噪步不激活 | Sec III-C |
| 动作 horizon | DDIM 去噪步数（具体步数未报告） | Sec III |
| 数据 | 真机 80 条 demo（6 类食物×3 量级×5 碗位，kinesthetic 录制-回放）；仿真 4000 轨迹（2000 洒/2000 不洒，4 种 primitive shape 各 1000） | Sec III-C / IV-A |
| 奖励 | 无 RL 奖励；guidance 梯度 ∇log(1−p_spill)、ρ=2.5 替代奖励 | Sec III-C |
| Reset | 人工摆位（一碗固定，另一碗在 35×30 cm 工作区随机放置） | Sec IV-A |
| 成功定义 | 成功舀取且不溅洒（成功但溅洒单独计为溅洒） | Sec IV-C |
| 评估次数 | 10 类 unseen 食物×2 量级×5 trials=100 trials/方法 | Sec IV-A |
| 随机种子 | 未报告 | |
| 扰动测试 | 10 类未见食物（形状/质地泛化）；位置随机放置 | Sec IV-A |
| 真机 | 7-DoF Franka Panda+勺具+2×Orbbec Femto Bolt RGB-D；10 类食物 100 trials | Sec IV-A |
| 算力 | 未报告（训练/推理 GPU 未披露） | |
| 特权信息 | 无（predictor 输入为部署可得的分割点云；仿真训 predictor 真机照常运行） | |

**附录陷阱自查**：
- privileged 信息：无
- reward shaping：无奖励——guidance 梯度替代奖励，ρ=2.5 与延迟 30 步是两个关键手调超参（对任务敏感）
- reset 难度：人工摆位（食物随机放置补偿部分）
- eval budget：100 trials/方法，充足
- 底层控制栈：10 Hz 控制器，PD 细节未报告
- 数据优势：predictor 用 4K 仿真轨迹（baseline 均无此资产）；策略侧 80 条 demo 对各方法对齐

## 消融实验与分析

10 类 unseen 食物测试集上的成功率与溅洒率（论文 Table）：

| 方法 | 成功率 (%) | 溅洒率 (%) |
|------|-----------|-----------|
| **GRITS（引导扩散）** | **82.0%** | **4.0%** |
| Diffusion Policy（无引导） | 70.0% | 15.0% |
| DP + post-processing（后处理修正） | 52.0% | 8.0% |
| SCONE（对比方法） | 65.0% | 20.0% |
| BC（行为克隆） | 45.0% | 45.0% |

**核心结论**：(1) guidance 的直接收益是溅洒率 15%→4%（相对降低 73%+），成功率 70%→82%（+12pp）——同一扩散策略，仅加一个可微分引导信号就同时改善了双指标；(2) 后处理路线（52%/8%）证明"事后修正"不仅救不回成功率（-18pp vs 无引导 DP），溅洒率也只压到 8%——约束必须参与生成过程而非叠加在输出上；(3) 行为克隆 45%/45% 说明无强化/无引导的纯模仿在"安全约束"维度基本无效——安全不是从成功 demo 里自动学到的；(4) unseen 食物上的结果（10 类未见过的形状/质地）说明 predictor 的 primitive-shape 泛化足以覆盖未见食物类别——"洒"的物理（堆积角、滚动、滑动）比"食物类别"更通用。

## 技术权衡（Trade-off）

| 优势 | 劣势 |
|------|------|
| 不修改架构：diffusion policy 原生支持可微分 guidance | predictor 的仿真-真实 gap：仿真训练的"洒"模型在真实摩擦/刚度下可能偏差 |
| 失败预测器与策略数据解耦（仿真 vs 真机） | 延迟激活（30 步）与 ρ 需要调参，对任务敏感 |
| 仅 80 条真机 demo 即可训练 | 只验证了食物舀取任务，接触几何更复杂时 predictor 需要重训 |
| 任何"可预测的失败模式"都可套用（溢出/碰撞/倾覆） | 引导依赖 point cloud 输入的完整性，遮挡时 predictor 置信度下降 |

## 技术价值与演进定位

GRITS 的通用性在于它把"安全约束"变成了**扩散过程的即插即用信号**：任何"能预测失败"的模块（溅洒、碰撞、溢出、缠绕）都可以用同一套 guidance 公式接入去噪过程，无需重训策略、无需改架构。这与 classifier guidance 在图像生成中的角色完全同构——图像领域用它控制风格，机器人领域用它控制安全。它的意义还在于验证了"仿真训练失败预测器 + 真机训练策略"的混合数据范式：失败经验（负面数据）从仿真廉价获得，成功经验（正面数据）由真机 demo 提供，两种数据各得其所。与 HapticVLA（触觉蒸馏）相比，GRITS 蒸馏的是"失败知识"而非"传感器能力"；与 RL-100（三阶段 RL）相比，GRITS 用引导替代了部分在线 RL 的作用——这是"约束式安全"对"探索式安全"的一次低成本替代。

## 与其他论文的关系

- 与 Diffusion Policy 家族（DP、DP3）：GRITS 直接建立在 DP/DP3 之上，贡献是 guidance 层——DP3 的点云编码被保留，新增的只是 predictor 及其梯度注入。
- 与 classifier guidance（图像生成）：公式 $\nabla_{x_t}\log(1-p_{spill})$ 与 classifier guidance 同构，GRITS 是其在机器人动作轨迹空间的迁移，延迟激活是轨迹特有的工程适配。
- 与 HapticVLA（触觉蒸馏）：都使用"训练期信号 → 部署期轻量"的迁移结构——HapticVLA 迁移触觉能力，GRITS 迁移失败预测；HapticVLA 蒸馏到网络参数里，GRITS 保持为显式 guidance。
- 与 RL-100 / Z-1（RL 后训练）：RL 用探索修正策略，GRITS 用引导修正生成——GRITS 的路线不需要在线交互，适合"失败模式已知、交互成本高"的任务。

## 精读问题

1. **predictor 的泛化边界**：4 种 primitive shapes 训练出的"洒"模型，在什么形状/材质组合下失效？predictor 的分布偏移如何量化并影响 guidance 质量？
2. **延迟激活与 ρ 的联合敏感性**：30 步延迟与 ρ=2.5 是否是网格搜索的最优？当任务变快（勺子移动更急）时，最优 (delay, ρ) 如何移动？
3. **guidance 与去噪退火**：引导梯度与 DDPM 噪声项在晚期去噪的相互作用——ρ 过大是否造成轨迹"卡在 predictor 的盲区"？
4. **失败模式的可预测性假设**：如果溅洒 predictor 对某类食物（如吸附性强的粘稠物）误判为零概率，GRITS 是否退化为无引导 DP？误判概率多高时引导开始有害？
5. **点云完整性的依赖**：SAM2 分割错误（food 与 spoon 粘连）时 predictor 的输入被污染——输入污染对 guidance 信号的影响是"随机噪声"还是"系统性偏差"？
6. **扩展到其他安全约束**：同一 guidance 框架接入碰撞预测器/倾覆预测器时，多个 guidance 项的加权策略是什么？梯度方向冲突时如何仲裁？
