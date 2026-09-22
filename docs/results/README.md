# 2026-09-22 进度同步：图表与汇总

此目录提供可在 GitHub 上直接查看的实验统计。原始逐步日志、检查点、参考数据、
机器人资产和冻结源码快照保留在服务器的各实验目录，不包含在这里。
数字来自已完成实验，不代表已通过正式扩规模或论文匹配评估。

| 实验 | 范围与结论 | 图表 | 数据 |
|---|---|---|---|
| [v3 稳定性复验](../stability_validation_20260921.md) | 16 环境 × 20 更新，数值检查通过，行为未达标 | [PNG](stability_v3_20260921.png) | 详细统计见报告 |
| [v4 课程与 critic](../curriculum_critic_20260921.md) | 3 种子 × 64 环境 × 200 更新，critic 改善，控制行为未一致改善 | [PNG](curriculum_v4_20260921.png) | [训练内汇总 JSON](curriculum_v4_20260921.json) |
| [执行器对照](../actuator_physics_20260922.md) | 8 场无学习物理运行，新上限生效，首步冲击降低 | [PNG](actuator_comparison_20260922.png) | [JSON](actuator_comparison_20260922.json) / [CSV](actuator_comparison_20260922.csv) |
| [reset 对照](../reset_physics_20260922.md) | 6 场无学习物理运行，参考初始化降低冲击，存活与身体跟踪未一致改善 | [PNG](reset_comparison_20260922.png) | [JSON](reset_comparison_20260922.json) / [逐 episode CSV](reset_episodes_20260922.csv) |

物理对照每场 4 环境、3 秒，主要统计首次 episode；跟踪误差使用每对环境共同存活的时间段。
不能把这些短时播放结果作为经过训练的策略分数或 30 秒成功率。v4 最终策略评估继续后置。

图片直接复制既有产物；CSV 仅统一换行符，JSON 仅选取聚合字段，原始数值不变。
[发布来源清单](publication_manifest_20260922.json)记录服务器来源、选取字段及 SHA256；
[reset 验证记录](reset_validation_20260922.json)对应上一次实验完成时的验证状态，
包括 813 项 CPU 回归通过、2 项跳过，以及随后 6 项分析检查通过。

当前决策：保留逐关节上限与默认 reset，候选参考初始化作为显式实验开关。
下一步为固定简单片段的短 PPO 学习对照，安排见[下一阶段计划](../next_steps_20260922.md)。
