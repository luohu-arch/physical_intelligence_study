# 1X 世界模型线：Data Engine → Policy（大纲卡）

- Source：https://1x.tech（world model 博客 2024 起）、HPT-1X（arXiv ~2510，NeurIPS 系引用）、world-model-as-policy（2026-01，postcutoff 报道）、Business Insider 2026-01-16 专访
- Project：https://www.1x.tech
- Category：commercial physical ai / partially published（HPT-1X 有 arXiv；世界模型策略暂无论文）
- Priority：high
- 拉取日期：2026-10-03

## 一句话总结

1X（NEO 家用仿人，$20k/499 月订阅）的 AI 路线独树一帜：**世界模型三段论**——先做"data engine"（世界模型合成数据增广策略训练），再做"评测器"（World Model Challenge，HuggingFace 公开赛），2026-01 起做"策略本身"（模型先生成期望结果的视频预测，动作从预测导出，而非 pixels→actions 直映）——CEO 明确表示将减少对人类示范的依赖。

## 技术要点

- **HPT-1X**（有 arXiv，~2510）：异构预训练 Transformer 部署到 NEO 家用真机做 in-home manipulation——库内 univla-latent-actions 同属"从异构数据学统一表征"线。
- **World model as policy**（2026-01）：视频预测作为控制的中间表征——与库内 Cascaded WAM 线（UniPi 起点的"先预测后解码"）同构，是商业侧对 Cascaded 架构的押注。
- **World Model Lab**（2026-06 成立，Sam Sinha 领导）：大规模具身世界模型预训练。
- **世界模型评测公开化**：World Model Challenge 把世界模型质量变成公开竞赛——罕见的公司侧开放评测动作。

## 证据与开放性评估

- HPT-1X 有正式论文+开放引用，证据等级中上；世界模型策略与 30 亿参数级声明目前只有 blog + 媒体报道，无论文。
- 商业进展硬指标：NEO 量产开订（2025-10）、EQT 合作部署至多 10000 台（2025-12）、加州工厂——落地规模在仿人公司里最激进之一。
- demo 口碑：2026-07 手指级灵巧度 demo 被评价为"工业自动化做不到的速度"。

## 与本库的关系

- **unipi / leworldmodel**：Cascaded 世界模型（视频计划→动作解码）的学术对照——1X 是这条路线最大的商业押注。
- **wam-survey-brief**：综述里 Cascaded vs Joint 的分野，1X 押 Cascaded。
- **agibot-world-go1 / genie-envisioner**：智元同为公司侧路线，但 AgiBot 开放数据+论文，1X 开放评测+部分论文——两条开放策略不同。
- **pi0.7**：Physical Intelligence 走"流匹配动作专家+论文开放"，与 1X"世界模型优先"形成商业范式对照。
