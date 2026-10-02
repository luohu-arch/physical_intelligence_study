# 千寻智能 Spirit AI：RoboArena 连冠线（大纲卡）

- Source：https://www.163.com（Spirit v1.5 "金山" 报道 2026-02）、腾讯新闻（20 亿融资 2026-02）、RoboArena 榜单
- Project：千寻智能（Spirit AI，清华系）
- Category：commercial physical ai / no open paper（模型无论文，成绩公开可复现）
- Priority：medium-high
- 拉取日期：2026-10-03

## 一句话总结

千寻智能的具身大模型 Spirit v1.5/v1.6 在 **RoboArena**（全球可复现的真机长程操作竞技场）连续夺冠——被认为是中国具身模型首次在"全球公认、可复现规则"下与国际顶尖玩家同场竞技并取胜，2026-02 完成近 20 亿元两轮融资（云锋、红杉等）跻身百亿独角兽。

## 证据与开放性评估

- **硬证据**：RoboArena 是公开竞技场（非自选 benchmark），连冠成绩可复现——比多数公司"自训自测"可信一档。
- **弱证据**：模型、数据、论文均不公开；架构细节只有宣传口径（"堆金山"指海量真机数据积累）。
- **甄别结论**：作为"数据规模路线是否够用"的产业信号跟踪，不作为方法研究对象。

## 与本库的关系

- **RoboArena 语境**：与库内 RoboTwin/LIBERO 系 benchmark 记录的学术线对照——竞技场规则下的真机长程任务。
- **agibot-world-go1 / graspvla / wall-wm**：同为国内公司但选择"论文+开放"路线的对照样本——开放策略光谱：智元（数据+论文开放）> 自变量（论文开放）> Galaxea（技术报告）> 千寻（仅成绩）> Figure/1X（blog）。
